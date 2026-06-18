"""Voice control CLI and runtime orchestration.

Examples
--------
    python -m voice_control.cli --config configs/voice_control.yaml --dry-run
    python -m voice_control.cli --config configs/voice_control.yaml --execute
    python -m voice_control.cli --config configs/voice_control.yaml --backend vosk
    python -m voice_control.cli --config configs/voice_control.yaml --backend whisper_cpp
    python -m voice_control.cli --text "walk forward slowly" --dry-run
    python -m voice_control.cli --text "left jab" --dry-run
    python -m voice_control.cli --interactive-text --dry-run

``--text`` and ``--interactive-text`` bypass the microphone/ASR entirely and
exercise the parser + safety + publisher path directly. This is the primary way
to test the system without audio hardware.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from typing import List, Optional

from .config import Config, setup_logging
from .duration import LLMDurationEstimator
from .feedback import Feedback
from .llm_parser import LLMParser
from .parser import DeterministicParser
from .publisher import (
    PlannerCommandPublisher,
    PlannerStreamLoop,
    StubPublisher,
    build_publisher,
)
from .safety import SafetyGuard
from .schemas import (
    ClarifyCommand,
    ParseResult,
    SetBoxingActionCommand,
    StopCommand,
)

log = logging.getLogger(__name__)


class VoicePipeline:
    """End-to-end text/ASR -> parse -> safety -> publish pipeline."""

    def __init__(
        self,
        config: Config,
        publisher: Optional[PlannerCommandPublisher] = None,
        feedback: Optional[Feedback] = None,
    ) -> None:
        self.config = config
        self.parser = DeterministicParser(config.parser.confidence_threshold)
        self.llm = LLMParser(config.parser) if config.parser.use_llm_fallback else None
        self.duration_estimator = (
            LLMDurationEstimator(config.parser) if config.parser.use_llm_duration else None
        )
        self.safety = SafetyGuard(config.safety)
        self.feedback = feedback or Feedback()
        self.dry_run = not self.safety.should_execute()

        if publisher is None:
            # Dry-run NEVER constructs a real publisher.
            publisher = StubPublisher() if self.dry_run else build_publisher(config.publisher)
        self.publisher = publisher
        self.stream = PlannerStreamLoop(
            publisher=self.publisher,
            command_timeout_s=config.safety.command_timeout_s,
            planner_dt=config.publisher.planner_dt,
        )
        self.boxing_active = False
        self._stream_thread: Optional[threading.Thread] = None
        self._stop_stream = threading.Event()

    # ------------------------------------------------------------------ #
    def process_text(self, text: str) -> List[ParseResult]:
        """Process a (possibly compound) transcript end-to-end, in order.

        The utterance is split on natural connectors ("and then", "then", "and",
        ...) into a sequence of sub-commands, each parsed, gated, safety-checked
        and published in order. Returns one :class:`ParseResult` per executed
        step. A single command yields a one-element list.
        """

        from .parser import split_segments

        segments = split_segments(text) or [text]
        results: List[ParseResult] = []
        for i, segment in enumerate(segments):
            if len(segments) > 1:
                log.info("PLAN step %d/%d: %r", i + 1, len(segments), segment)
            parsed = self._parse_segment(segment)
            handled = self._handle_segment(parsed, original=text)
            results.append(handled)
            self._dwell_if_executing(handled, last=(i == len(segments) - 1))
        return results

    def _parse_segment(self, text: str) -> ParseResult:
        result = self.parser.parse(text, boxing_active=self.boxing_active)
        log.info("RAW=%r NORMALIZED=%r", result.raw_text, result.normalized_text)
        # Optional LLM fallback only when deterministic parsing was inconclusive.
        if result.is_clarify() and self.llm is not None:
            log.info("Deterministic parser unsure; consulting LLM fallback.")
            llm_result = self.llm.parse(text, result.normalized_text)
            if not llm_result.is_clarify():
                result = llm_result
        return result

    def _handle_segment(self, result: ParseResult, original: str) -> ParseResult:
        command = result.command
        log.info("PARSED=%s confidence=%.2f", _summary(command), result.confidence)

        if command is None or isinstance(command, ClarifyCommand):
            self.feedback.clarify(
                command.question if isinstance(command, ClarifyCommand)
                else "Please repeat that command."
            )
            log.info("REJECTED (clarify): reason=%s", result.reason)
            return result

        # Stop is always honoured immediately, regardless of confidence.
        if isinstance(command, StopCommand):
            self.stream.set_command(command)
            self.feedback.confirm("STOP")
            log.info("ACCEPTED stop")
            return ParseResult(
                ok=True, confidence=result.confidence, raw_text=result.raw_text,
                normalized_text=result.normalized_text, command=command, reason="stop",
            )

        # Confidence gate: never move on a low-confidence guess.
        if not SafetyGuard.passes_confidence(
            command, result.confidence, self.config.parser.confidence_threshold
        ):
            clarify = ClarifyCommand(
                question="I'm not confident I understood. Please repeat.",
                original_text=original,
            )
            self.feedback.clarify(clarify.question)
            log.info("REJECTED (low confidence %.2f < %.2f)",
                     result.confidence, self.config.parser.confidence_threshold)
            return ParseResult(
                ok=False, confidence=result.confidence, raw_text=result.raw_text,
                normalized_text=result.normalized_text, command=clarify,
                reason="low_confidence",
            )

        # No value clamping: the validated command is forwarded as-is.
        final = command

        # Track boxing context for ambiguous "side step".
        if isinstance(final, SetBoxingActionCommand):
            self.boxing_active = True
        elif final.tool in ("set_navigation", "set_crawl"):
            self.boxing_active = False

        # Dynamically decide how long this step should run before advancing.
        if (
            self.duration_estimator is not None
            and hasattr(final, "duration_s")
            and getattr(final, "duration_s", None) is None
        ):
            est = self.duration_estimator.estimate(final, result.raw_text)
            final = final.model_copy(update={"duration_s": est})
            self.feedback.confirm(f"step duration ~{est:.1f}s")
            log.info("ESTIMATED step duration=%.2fs", est)

        log.info("FINAL=%s dry_run=%s", _summary(final), self.dry_run)
        self.stream.set_command(final)
        self.feedback.confirm(_summary(final))
        return ParseResult(
            ok=True, confidence=result.confidence, raw_text=result.raw_text,
            normalized_text=result.normalized_text, command=final, reason="accepted",
        )

    def _dwell_if_executing(self, result: ParseResult, last: bool) -> None:
        """Hold a published step for its duration before moving to the next.

        In dry-run nothing is held (commands are just printed/recorded). In
        execute mode, each motion step is held for ``duration_s`` (if given) or
        ``publisher.segment_dwell_s``, re-publishing at planner_dt to maintain the
        velocity-conditioned hold. The final step is left active (not auto-held).
        """

        if self.dry_run or last:
            return
        command = result.command
        if command is None or isinstance(command, (ClarifyCommand, StopCommand)):
            return
        dwell = getattr(command, "duration_s", None) or self.config.publisher.segment_dwell_s
        deadline = time.monotonic() + float(dwell)
        while time.monotonic() < deadline:
            self.stream.tick()
            time.sleep(self.config.publisher.planner_dt)

    # ------------------------------------------------------------------ #
    def start_stream_thread(self) -> None:
        """Drive the velocity-hold + watchdog loop at planner_dt in background."""

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
        self.stream.close()


def _summary(command) -> str:
    if command is None:
        return "None"
    return command.model_dump(mode="json").__str__()


# --------------------------------------------------------------------------- #
# Runtime entry points
# --------------------------------------------------------------------------- #

def run_text_once(pipeline: VoicePipeline, text: str) -> List[ParseResult]:
    results = pipeline.process_text(text)
    print("\n=== Voice command result ===")
    print(f"raw_text       : {text!r}")
    print(f"dry_run        : {pipeline.dry_run}")
    if len(results) > 1:
        print(f"plan steps     : {len(results)} (executed in order)")
    for i, result in enumerate(results):
        cmd = result.command
        prefix = f"step {i + 1} " if len(results) > 1 else ""
        print(f"{prefix}normalized : {result.normalized_text!r}")
        print(f"{prefix}confidence : {result.confidence:.2f}")
        if cmd is not None:
            print(f"{prefix}tool_call  : {cmd.model_dump_json()}")
    print("============================\n")
    return results


def run_interactive_text(pipeline: VoicePipeline) -> None:
    print("Interactive text mode. Type a command, or 'quit' to exit.")
    if not pipeline.dry_run:
        pipeline.start_stream_thread()
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


def run_microphone(pipeline: VoicePipeline, config: Config) -> None:  # pragma: no cover - hw
    from .audio import MicrophoneCapture, VoiceActivityDetector, capture_utterance
    from .wake import WakeGate

    backend = config.audio.backend
    if backend == "vosk":
        from .asr_vosk import VoskASR

        asr = VoskASR(config.asr.vosk_model_path, config.audio.sample_rate)
    elif backend == "whisper_cpp":
        from .asr_whisper_cpp import WhisperCppASR

        asr = WhisperCppASR(
            config.asr.whisper_cpp_bin, config.asr.whisper_model_path, config.audio.sample_rate
        )
    else:
        raise ValueError(f"Unknown audio backend: {backend!r}")

    mic = MicrophoneCapture(config.audio.sample_rate, config.audio.device)
    vad = VoiceActivityDetector(config.audio.vad_aggressiveness, config.audio.sample_rate) \
        if config.audio.vad else None
    gate = WakeGate(config.wake.mode, config.wake.phrase)

    mic.start()
    if not pipeline.dry_run:
        pipeline.start_stream_thread()
    log.info("Microphone runtime started (backend=%s, wake=%s)", backend, config.wake.mode)
    try:
        while True:
            if not gate.wait_for_trigger():
                break
            pcm = capture_utterance(
                mic, vad, config.audio.max_utterance_s, config.audio.phrase_timeout_s
            )
            if not pcm:
                continue
            text = asr.transcribe(pcm)
            if not text:
                continue
            run_text_once(pipeline, text)
    except KeyboardInterrupt:
        pass
    finally:
        mic.stop()
        pipeline.close()


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="voice_control.cli",
        description="Offline voice command interface for the SONIC kinematic planner.",
    )
    p.add_argument("--config", type=str, default=None, help="Path to voice_control.yaml")
    p.add_argument("--text", type=str, default=None, help="Parse a single text command and exit")
    p.add_argument("--interactive-text", action="store_true", help="Type commands interactively")
    p.add_argument("--backend", type=str, default=None,
                   choices=["vosk", "whisper_cpp"], help="ASR backend override")
    p.add_argument("--dry-run", action="store_true", help="Never send to robot (default)")
    p.add_argument("--execute", action="store_true",
                   help="Actually send commands (requires config safety.execute: true)")
    p.add_argument("--publisher", type=str, default=None,
                   choices=["stub", "existing_repo", "zmq", "ros2"],
                   help="Publisher backend override")
    p.add_argument("--llm-duration", action="store_true",
                   help="Enable the LLM/heuristic per-step duration estimator "
                        "(overrides parser.use_llm_duration)")
    return p


def load_config_with_overrides(args: argparse.Namespace) -> Config:
    config = Config.from_yaml(args.config) if args.config else Config()

    if args.backend:
        config.audio.backend = args.backend
    if args.publisher:
        config.publisher.backend = args.publisher
    if args.llm_duration:
        config.parser.use_llm_duration = True

    # Dry-run / execute resolution. Dry-run wins if both are passed.
    if args.execute and not args.dry_run:
        config.safety.dry_run = False
        config.safety.execute = config.safety.execute  # keep config gate authoritative
        if not config.safety.execute:
            log.warning(
                "--execute passed but config safety.execute is false; staying in DRY-RUN. "
                "Set safety.execute: true in the config to allow real commands."
            )
            config.safety.dry_run = True
    if args.dry_run:
        config.safety.dry_run = True
    return config


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_config_with_overrides(args)
    setup_logging(config.logging)

    pipeline = VoicePipeline(config)

    if args.text is not None:
        run_text_once(pipeline, args.text)
        pipeline.close()
        return 0
    if args.interactive_text:
        run_interactive_text(pipeline)
        return 0

    # Default: microphone runtime.
    try:
        run_microphone(pipeline, config)
    except Exception as exc:  # pragma: no cover - hardware dependent
        log.error("Microphone runtime failed: %s", exc)
        print(f"[error] {exc}", file=sys.stderr)
        pipeline.close()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
