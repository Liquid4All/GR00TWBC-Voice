"""Voice control CLI and runtime orchestration.

Examples
--------
    python -m voice_control.cli --config configs/voice_control.yaml --dry-run
    python -m voice_control.cli --config configs/voice_control.yaml --execute
    python -m voice_control.cli --config configs/voice_control.yaml --backend whisper_cpp
    # Unitree G1 onboard mic (UDP multicast) + whisper.cpp:
    python -m voice_control.cli --source multicast --backend whisper_cpp --dry-run
    # Raw mic test (records a WAV, no ASR):
    python -m voice_control.cli --source multicast --record mic.wav --record-seconds 5
    python -m voice_control.cli --text "walk forward slowly" --dry-run
    python -m voice_control.cli --interactive-text --dry-run

``--text`` and ``--interactive-text`` bypass the microphone/ASR entirely and
exercise the parser + publisher path directly. This is the primary way to test
the system without audio hardware.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from typing import List, Optional

from .config import Config, setup_logging
from .duration import HeuristicDurationEstimator
from .feedback import Feedback
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
        self.duration_estimator = HeuristicDurationEstimator(config.parser)
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

        # Deterministically decide how long this step should run before advancing
        # (distance/speed, turn angle, posture settle time, ...).
        if hasattr(final, "duration_s") and getattr(final, "duration_s", None) is None:
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


def build_mic(config: Config):  # pragma: no cover - hw
    """Construct the configured capture source (sound card or G1 multicast)."""

    source = config.audio.source
    if source == "multicast":
        from .audio import MulticastMicrophone

        return MulticastMicrophone(
            sample_rate=config.audio.sample_rate,
            group=config.audio.mcast_group,
            port=config.audio.mcast_port,
            iface_ip=config.audio.mcast_iface_ip,
            frame_ms=config.audio.frame_ms,
        )
    if source == "device":
        from .audio import MicrophoneCapture

        return MicrophoneCapture(
            config.audio.sample_rate, config.audio.device, frame_ms=config.audio.frame_ms
        )
    raise ValueError(f"Unknown audio source: {source!r} (use 'device' or 'multicast')")


def build_asr(config: Config):  # pragma: no cover - hw
    backend = config.audio.backend
    if backend == "vosk":
        from .asr_vosk import VoskASR

        return VoskASR(config.asr.vosk_model_path, config.audio.sample_rate)
    if backend == "whisper_cpp":
        from .asr_whisper_cpp import WhisperCppASR

        return WhisperCppASR(
            config.asr.whisper_cpp_bin, config.asr.whisper_model_path, config.audio.sample_rate
        )
    raise ValueError(f"Unknown audio backend: {backend!r}")


def run_record(config: Config, out_path: str, seconds: float) -> int:  # pragma: no cover - hw
    """Record N seconds from the configured source to a 16 kHz mono WAV.

    Useful for verifying the Unitree G1 multicast mic before wiring up ASR:
        python -m voice_control.cli --source multicast --record mic.wav --record-seconds 5
    """

    import wave

    from .audio import capture_fixed

    mic = build_mic(config)
    mic.start()
    try:
        print(f"[voice] recording {seconds:.1f}s from {config.audio.source}... speak now")
        pcm = capture_fixed(mic, seconds)
    finally:
        mic.stop()

    if not pcm:
        print("[voice] no audio captured. For the G1 mic: is the voice assistant awake, "
              "and is audio.mcast_iface_ip your 192.168.123.x address?")
        return 1
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(config.audio.sample_rate)
        wf.writeframes(pcm)
    peak = max((abs(int.from_bytes(pcm[i:i + 2], "little", signed=True))
                for i in range(0, len(pcm), 2)), default=0)
    dur = (len(pcm) / 2) / config.audio.sample_rate
    print(f"[voice] wrote {out_path} ({dur:.1f}s, peak amplitude {peak}). "
          f"{'Looks like real audio.' if peak > 200 else 'Near-silence -- check the mic.'}")
    return 0


def run_microphone(pipeline: VoicePipeline, config: Config) -> None:  # pragma: no cover - hw
    from .audio import VoiceActivityDetector, capture_utterance
    from .wake import WakeGate

    asr = build_asr(config)
    mic = build_mic(config)
    vad = VoiceActivityDetector(config.audio.vad_aggressiveness, config.audio.sample_rate) \
        if config.audio.vad else None
    gate = WakeGate(config.wake.mode, config.wake.phrase)

    mic.start()
    if not pipeline.dry_run:
        pipeline.start_stream_thread()
    log.info("Microphone runtime started (backend=%s, source=%s, wake=%s)",
             config.audio.backend, config.audio.source, config.wake.mode)
    sr = config.audio.sample_rate
    try:
        while True:
            if not gate.wait_for_trigger():
                break
            print("[voice] listening... (speak now)")
            mic.flush()
            pcm = capture_utterance(
                mic, vad, config.audio.max_utterance_s, config.audio.phrase_timeout_s
            )
            dur_ms = (len(pcm) / 2) / sr * 1000.0  # 16-bit mono => 2 bytes/sample
            log.info("Captured %.0f ms of audio (%d bytes)", dur_ms, len(pcm))
            if not pcm:
                if config.audio.source == "multicast":
                    print("[voice] no audio from the G1 mic. Is the robot's voice assistant "
                          "awake? Is audio.mcast_iface_ip your 192.168.123.x address? "
                          "Verify with `--record mic.wav --record-seconds 5`.")
                else:
                    print("[voice] no speech captured. Is audio.device correct and the mic "
                          "unmuted/gain up? You can set audio.vad: false to force a fixed window.")
                continue
            text = asr.transcribe(pcm)
            log.info("ASR transcript: %r", text)
            if not text:
                print(f"[voice] captured {dur_ms:.0f} ms but transcription was empty. "
                      "Likely a mic-gain / sample-rate issue, or speech too quiet/far.")
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
    p.add_argument("--source", type=str, default=None,
                   choices=["device", "multicast"],
                   help="Capture source override (device sound card | G1 multicast mic)")
    p.add_argument("--dry-run", action="store_true", help="Never send to robot (default)")
    p.add_argument("--execute", action="store_true",
                   help="Actually send commands (requires config safety.execute: true)")
    p.add_argument("--publisher", type=str, default=None,
                   choices=["stub", "existing_repo", "zmq", "ros2"],
                   help="Publisher backend override")
    p.add_argument("--record", type=str, default=None, metavar="WAV",
                   help="Record from the configured source to a WAV and exit (mic test)")
    p.add_argument("--record-seconds", type=float, default=5.0,
                   help="Duration for --record (default 5s)")
    return p


def load_config_with_overrides(args: argparse.Namespace) -> Config:
    config = Config.from_yaml(args.config) if args.config else Config()

    if args.backend:
        config.audio.backend = args.backend
    if args.source:
        config.audio.source = args.source
    if args.publisher:
        config.publisher.backend = args.publisher

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

    if args.record is not None:
        return run_record(config, args.record, args.record_seconds)

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
