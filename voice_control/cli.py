"""Voice control CLI entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from .config import Config, setup_logging
from .pipeline import (
    VoicePipeline,
    run_interactive_text,
    run_microphone,
    run_record,
    run_text_once,
)

log = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="voice_control.cli",
        description="Offline voice command interface for the SONIC kinematic planner.",
    )
    p.add_argument("--config", type=str, default=None, help="Path to voice_control.yaml")
    p.add_argument("--text", type=str, default=None, help="Parse a single text command and exit")
    p.add_argument("--interactive-text", action="store_true", help="Type commands interactively")
    p.add_argument("--backend", type=str, default=None,
                   choices=["whisper_cpp"], help="ASR backend override")
    p.add_argument("--parser", type=str, default=None,
                   choices=["deterministic", "model"], help="Command parser backend override")
    p.add_argument("--source", type=str, default=None,
                   choices=["device", "multicast"],
                   help="Capture source override (device sound card | G1 multicast mic)")
    p.add_argument("--dry-run", action="store_true", help="Never send to robot (default)")
    p.add_argument("--execute", action="store_true",
                   help="Actually send commands (requires config safety.execute: true)")
    p.add_argument("--publisher", type=str, default=None,
                   choices=["stub", "local_planner", "zmq", "existing_repo"],
                   help="Publisher backend override")
    p.add_argument("--local-planner", action="store_true",
                   help="Publish directly to onboard planner via localhost ZMQ "
                        "(sets publisher.backend=local_planner)")
    p.add_argument("--record", type=str, default=None, metavar="WAV",
                   help="Record from the configured source to a WAV and exit (mic test)")
    p.add_argument("--record-seconds", type=float, default=5.0,
                   help="Duration for --record (default 5s)")
    return p


def load_config_with_overrides(args: argparse.Namespace) -> Config:
    config = Config.from_yaml(args.config) if args.config else Config()

    if args.backend:
        config.audio.backend = args.backend
    if args.parser:
        config.parser.backend = args.parser
    if args.source:
        config.audio.source = args.source
    if args.publisher:
        config.publisher.backend = args.publisher
    if args.local_planner:
        config.publisher.backend = "local_planner"

    if args.execute and not args.dry_run:
        config.safety.dry_run = False
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
