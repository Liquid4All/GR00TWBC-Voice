from __future__ import annotations

import logging
import re
import threading
import time
from typing import List, Optional

from .asr import build_asr
from .audio import Feedback, VoiceActivityDetector, WakeGate, build_capture, capture_fixed, capture_utterance
from .config import Config, ParserConfig, SafetyConfig
from .parsers import CommandParser, build_parser, split_segments
from .publisher import (
    PlannerCommandPublisher,
    PlannerStreamLoop,
    StubPublisher,
    build_publisher,
    tool_call_to_planner_fields,
)
from .parsers import ClarifyCommand, ParseResult, SetBoxingActionCommand, StopCommand

log = logging.getLogger(__name__)
_DISTANCE_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:meters?|metres?|m)\b", re.IGNORECASE)


class SafetyGuard:
    def __init__(self, cfg: SafetyConfig) -> None:
        self.cfg = cfg

    @staticmethod
    def passes_confidence(command, confidence: float, threshold: float) -> bool:
        if isinstance(command, (StopCommand, ClarifyCommand)):
            return True
        return confidence >= threshold

    def should_execute(self) -> bool:
        return bool(self.cfg.execute) and not bool(self.cfg.dry_run)


def extract_distance_m(text: str) -> Optional[float]:
    if not text:
        return None
    match = _DISTANCE_RE.search(text)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return value if value > 0 else None


class HeuristicDurationEstimator:
    def __init__(self, cfg: ParserConfig) -> None:
        self.min_s, self.max_s = float(cfg.duration_min_s), float(cfg.duration_max_s)

    def estimate(self, command, segment_text: str) -> float:
        tool = getattr(command, "tool", None)
        if tool in ("set_navigation", "set_crawl"):
            velocity = float(getattr(command, "velocity_mps", 0.0) or 0.0)
            heading = float(getattr(command, "heading_deg", 0.0) or 0.0)
            distance = extract_distance_m(segment_text)
            if velocity <= 1e-3:
                est = max(1.0, abs(heading) / 60.0)
            elif distance is not None:
                est = distance / velocity + 1.0
            else:
                est = 3.0
        elif tool == "rotate_in_place":
            angle = abs(float(getattr(command, "angle_deg", 0.0) or 0.0))
            rate = abs(float(getattr(command, "yaw_rate_dps", 90.0) or 90.0))
            est = max(0.5, angle / max(rate, 1.0))
        elif tool == "hold_pose":
            est = float(getattr(command, "duration_s", None) or 3.0)
        elif tool == "set_posture":
            est = 3.0  # this 3 is random, fix it later
        elif tool == "set_boxing_action":
            est = 1.5
        elif tool == "get_up":
            est = 4.5
        else:
            est = 3.0
        return max(self.min_s, min(self.max_s, est))


def command_summary(command) -> str:
    return "None" if command is None else str(command.model_dump(mode="json"))


def resolve_command_duration_s(command, *, default_s: float, min_s: float, max_s: float) -> float:
    """Return clamped hold time for a planner command (``duration_s`` or fallback)."""
    raw = getattr(command, "duration_s", None)
    duration = float(default_s if raw is None else raw)
    return max(min_s, min(max_s, duration))


class VoicePipeline:
    def __init__(
        self, config: Config, *, parser: Optional[CommandParser] = None,
        publisher: Optional[PlannerCommandPublisher] = None, feedback: Optional[Feedback] = None,
    ) -> None:
        self.config = config
        self.parser = parser or build_parser(config.parser)
        self.duration_estimator = HeuristicDurationEstimator(config.parser)
        self.safety = SafetyGuard(config.safety)
        self.feedback = feedback or Feedback()
        self.dry_run = not self.safety.should_execute()
        if publisher is None:
            publisher = StubPublisher() if self.dry_run else build_publisher(config.publisher)
        self.publisher = publisher
        self.stream = PlannerStreamLoop(
            publisher=publisher, command_timeout_s=config.safety.command_timeout_s,
            planner_dt=config.publisher.planner_dt,
        )
        self.boxing_active = False
        self._stream_thread: Optional[threading.Thread] = None
        self._stop_stream = threading.Event()
        if not self.dry_run:
            self.start_stream()

    def process_text(self, text: str) -> List[ParseResult]:
        if hasattr(self.parser, "parse_plan"):
            plan = self.parser.parse_plan(text, boxing_active=self.boxing_active)
            results: List[ParseResult] = []
            for i, parsed in enumerate(plan):
                if len(plan) > 1:
                    log.info("PLAN step %d/%d (lfm)", i + 1, len(plan))
                parsed.raw_text = text
                handled = self._handle_segment(parsed, original=text, skip_duration_estimate=True)
                results.append(handled)
                self._hold_command(handled)
            return results
        segments = split_segments(text) or [text]
        results: List[ParseResult] = []
        for i, segment in enumerate(segments):
            if len(segments) > 1:
                log.info("PLAN step %d/%d: %r", i + 1, len(segments), segment)
            handled = self._handle_segment(self._parse_segment(segment), original=text)
            results.append(handled)
            self._hold_command(handled)
        return results

    def _parse_segment(self, text: str) -> ParseResult:
        result = self.parser.parse(text, boxing_active=self.boxing_active)
        log.info("RAW=%r NORMALIZED=%r", result.raw_text, result.normalized_text)
        return result

    def _handle_segment(
        self, result: ParseResult, original: str, *, skip_duration_estimate: bool = False,
    ) -> ParseResult:
        command = result.command
        log.info("PARSED=%s confidence=%.2f", command_summary(command), result.confidence)
        if command is None or isinstance(command, ClarifyCommand):
            self.feedback.clarify(
                command.question if isinstance(command, ClarifyCommand) else "Please repeat that command."
            )
            return result
        if isinstance(command, StopCommand):
            self.stream.set_command(command)
            self.feedback.confirm("STOP")
            return ParseResult(
                ok=True, confidence=result.confidence, raw_text=result.raw_text,
                normalized_text=result.normalized_text, command=command, reason="stop",
            )
        if not SafetyGuard.passes_confidence(command, result.confidence, self.config.parser.confidence_threshold):
            clarify = ClarifyCommand(
                question="I'm not confident I understood. Please repeat.", original_text=original,
            )
            self.feedback.clarify(clarify.question)
            return ParseResult(
                ok=False, confidence=result.confidence, raw_text=result.raw_text,
                normalized_text=result.normalized_text, command=clarify, reason="low_confidence",
            )
        final = command
        if isinstance(final, SetBoxingActionCommand):
            self.boxing_active = True
        elif final.tool in ("set_navigation", "set_crawl", "rotate_in_place", "hold_pose"):
            self.boxing_active = False
        if (
            not skip_duration_estimate
            and hasattr(final, "duration_s")
            and getattr(final, "duration_s", None) is None
        ):
            est = self.duration_estimator.estimate(final, result.raw_text)
            final = final.model_copy(update={"duration_s": est})
            self.feedback.confirm(f"step duration ~{est:.1f}s")
        self.stream.set_command(final)
        self.feedback.confirm(command_summary(final))
        return ParseResult(
            ok=True, confidence=result.confidence, raw_text=result.raw_text,
            normalized_text=result.normalized_text, command=final, reason="accepted",
        )

    def _hold_command(self, result: ParseResult) -> None:
        """Hold each parsed command on the planner wire for its ``duration_s`` at planner_dt."""
        if self.dry_run:
            return
        command = result.command
        if command is None or isinstance(command, (ClarifyCommand, StopCommand)):
            return
        if not hasattr(command, "duration_s"):
            return
        duration = resolve_command_duration_s(
            command,
            default_s=self.config.publisher.segment_dwell_s,
            min_s=self.config.parser.duration_min_s,
            max_s=self.config.parser.duration_max_s,
        )
        bg = self._stream_thread is not None and self._stream_thread.is_alive()
        log.info(
            "Holding %s for %.2fs at %.0f Hz (background_stream=%s)",
            getattr(command, "tool", type(command).__name__), duration, 1.0 / self.config.publisher.planner_dt, bg,
        )
        self.stream.hold_for(duration, background_stream=bg)
        self.stream.return_to_standing()
        log.info(
            "Returned to standing after %s (%.2fs); holding IDLE until next command",
            getattr(command, "tool", type(command).__name__), duration,
        )

    def start_stream(self) -> None:
        if self._stream_thread is not None:
            return

        def _run() -> None:
            while not self._stop_stream.is_set():
                self.stream.tick()
                time.sleep(self.config.publisher.planner_dt)

        self._stop_stream.clear()
        self._stream_thread = threading.Thread(target=_run, daemon=True, name="planner-stream")
        self._stream_thread.start()

    def close(self) -> None:
        self._stop_stream.set()
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=1.0)
            self._stream_thread = None
        if not self.dry_run:
            self.stream.return_to_standing()
        self.stream.close()


def run_text_once(pipeline: VoicePipeline, text: str) -> List[ParseResult]:
    results = pipeline.process_text(text)
    print("\n=== Voice command result ===")
    print(f"raw_text       : {text!r}\ndry_run        : {pipeline.dry_run}")
    if not pipeline.dry_run:
        print(f"publisher      : {pipeline.config.publisher.backend} "
              f"-> {pipeline.config.publisher.local_planner_endpoint}")
    if len(results) > 1:
        print(f"plan steps     : {len(results)} (executed in order)")
    tracker = pipeline.stream.facing
    for i, result in enumerate(results):
        cmd = result.command
        prefix = f"step {i + 1} " if len(results) > 1 else ""
        print(f"{prefix}normalized : {result.normalized_text!r}\n{prefix}confidence : {result.confidence:.2f}")
        if cmd is not None:
            print(f"{prefix}tool_call  : {cmd.model_dump_json()}")
            if not isinstance(cmd, ClarifyCommand):
                print(f"{prefix}planner    : {tool_call_to_planner_fields(cmd, tracker).to_movement_state()}")
                print(f"{prefix}wire       : "
                      f"{'(dry-run; not sent to ZMQ)' if pipeline.dry_run else 'sent on ZMQ planner topic'}")
    print("============================\n")
    return results


def run_interactive_text(pipeline: VoicePipeline) -> None:
    print("Interactive text mode. Type a command, or 'quit' to exit.")
    if not pipeline.dry_run:
        pipeline.start_stream()
    try:
        while True:
            try:
                text = input("voice> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue
            run_text_once(pipeline, text)
    finally:
        pipeline.close()


def run_record(config: Config, out_path: str, seconds: float) -> int:
    import wave

    mic = build_capture(config.audio)
    mic.start()
    try:
        print(f"[voice] recording {seconds:.1f}s from {config.audio.source}... speak now")
        pcm = capture_fixed(mic, seconds)
    finally:
        mic.stop()
    if not pcm:
        print("[voice] no audio captured.")
        return 1
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(config.audio.sample_rate)
        wf.writeframes(pcm)
    peak = max((abs(int.from_bytes(pcm[i:i + 2], "little", signed=True)) for i in range(0, len(pcm), 2)), default=0)
    dur = (len(pcm) / 2) / config.audio.sample_rate
    print(f"[voice] wrote {out_path} ({dur:.1f}s, peak {peak}).")
    return 0


def run_microphone(pipeline: VoicePipeline, config: Config) -> None:
    asr = build_asr(config.audio, config.asr)
    mic = build_capture(config.audio)
    vad = VoiceActivityDetector(config.audio.vad_aggressiveness, config.audio.sample_rate) if config.audio.vad else None
    gate = WakeGate(config.wake, sample_rate=config.audio.sample_rate)
    mic.start()
    if not pipeline.dry_run:
        pipeline.start_stream()
    try:
        while True:
            if not gate.wait_for_trigger(mic):
                break
            print("[voice] listening... (speak now)")
            mic.flush()
            pcm = capture_utterance(mic, vad, config.audio.max_utterance_s, config.audio.phrase_timeout_s)
            if not pcm:
                print("[voice] no speech captured.")
                continue
            text = asr.transcribe(pcm)
            if not text:
                print("[voice] empty transcription.")
                continue
            run_text_once(pipeline, text)
    except KeyboardInterrupt:
        pass
    finally:
        mic.stop()
        pipeline.close()
