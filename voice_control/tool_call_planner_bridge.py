"""Bridge LFM G1 tool calls -> deploy kinematic planner inputs.

For a zero-dependency standalone script (stdlib only), use:

    python scripts/lfm_tool_call_to_planner.py --tool-calls '...'
    python scripts/lfm_tool_call_to_planner.py --movement-state '{"locomotion_mode":2,...}'

This module re-exports the same logic when voice_control is installed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from .config import ParserConfig
from .lfm_g1 import G1ToolMapper, extract_g1_tool_calls
from .parsers import PlannerToolCallType, tool_call_to_dict
from .pipeline import resolve_command_duration_s
from .publisher import (
    FacingTracker,
    LocalPlannerPublisher,
    PlannerFields,
)


def _wrap_tool_block(text: str) -> str:
    raw = text.strip()
    if "<|tool_call_start|>" in raw:
        return raw
    inner = raw if raw.startswith("[") else f"[{raw}]"
    end = "<|" + "redacted_tool_call_end_kimi" + "|>"
    return f"<|tool_call_start|>{inner}{end}"


def parse_tool_calls(text: str) -> List[tuple[str, dict[str, Any]]]:
    return extract_g1_tool_calls(_wrap_tool_block(text))


def map_tool_calls(
    calls: Sequence[tuple[str, dict[str, Any]]],
    *,
    cfg: Optional[ParserConfig] = None,
) -> List[PlannerToolCallType]:
    mapper = G1ToolMapper(cfg or ParserConfig())
    out: List[PlannerToolCallType] = []
    for name, args in calls:
        cmd = mapper.map(name, args)
        if cmd is not None:
            out.append(cmd)
    return out


@dataclass
class PlannerStep:
    tool_name: str
    tool_args: dict[str, Any]
    command: PlannerToolCallType
    fields: PlannerFields
    movement_state: dict[str, Any]
    onnx_inputs: dict[str, Any]
    wire_topic: str
    wire_size_bytes: int


def movement_state_to_onnx_inputs(state: dict[str, Any]) -> dict[str, Any]:
    """Fields written by LocalMotionPlanner::UpdateInputTensors (basic tier)."""
    return {
        "mode": int(state["locomotion_mode"]),
        "target_vel": float(state["movement_speed"]),
        "target_height": float(state["height"]),
        "movement_direction": list(state["movement_direction"]),
        "facing_direction": list(state["facing_direction"]),
    }


def tool_calls_to_planner_steps(
    text: str,
    *,
    cfg: Optional[ParserConfig] = None,
) -> tuple[List[tuple[str, dict[str, Any]]], List[PlannerStep]]:
    calls = parse_tool_calls(text)
    mapper = G1ToolMapper(cfg or ParserConfig())
    tracker = FacingTracker()
    steps: List[PlannerStep] = []
    for name, args in calls:
        cmd = mapper.map(name, args)
        if cmd is None:
            continue
        fields = tracker.to_planner_fields(cmd)
        state = fields.to_movement_state()
        wire = fields.to_planner_wire()
        steps.append(
            PlannerStep(
                tool_name=name,
                tool_args=args,
                command=cmd,
                fields=fields,
                movement_state=state,
                onnx_inputs=movement_state_to_onnx_inputs(state),
                wire_topic="planner",
                wire_size_bytes=len(wire),
            )
        )
    return calls, steps


def _print_steps(steps: List[PlannerStep], *, show_wire: bool) -> None:
    for i, step in enumerate(steps, start=1):
        print(f"\n=== Step {i}: {step.tool_name} ===")
        print(f"tool_args       : {json.dumps(step.tool_args, sort_keys=True)}")
        print(f"command         : {json.dumps(tool_call_to_dict(step.command), sort_keys=True)}")
        print(f"movement_state  : {json.dumps(step.movement_state, indent=2)}")
        print(f"onnx_inputs     : {json.dumps(step.onnx_inputs, indent=2)}")
        print(f"zmq             : topic={step.wire_topic!r} payload_bytes={step.wire_size_bytes}")
        if show_wire:
            wire = step.fields.to_planner_wire()
            print(f"wire_hex        : {wire[:32].hex()}... ({len(wire)} bytes total)")


def send_steps_over_zmq(
    steps: List[PlannerStep],
    *,
    endpoint: str,
    planner_dt: float,
    default_duration_s: float,
    min_s: float,
    max_s: float,
    interrupt_between: bool,
) -> None:
    pub = LocalPlannerPublisher(endpoint=endpoint)
    try:
        for i, step in enumerate(steps, start=1):
            duration = resolve_command_duration_s(
                step.command,
                default_s=default_duration_s,
                min_s=min_s,
                max_s=max_s,
            )
            print(f"\n[sending] step {i}/{len(steps)} hold {duration:.2f}s @ {1.0 / planner_dt:.0f} Hz")
            pub.publish_fields(step.fields)
            deadline = time.monotonic() + duration
            while time.monotonic() < deadline:
                pub.publish_fields(step.fields)
                time.sleep(planner_dt)
            if interrupt_between and i < len(steps):
                pub.return_to_standing()
                time.sleep(planner_dt)
        pub.return_to_standing()
        print("\n[done] all steps sent; holding IDLE standing (policy stays active)")
    finally:
        pub.close()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert LFM G1 tool calls to deploy kinematic planner inputs (and optional ZMQ send).",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--tool-calls",
        type=str,
        help="Tool call block or comma-separated calls, e.g. "
             "'planner_move(velocity_mps=1.0, heading_deg=0.0, yaw_rate_dps=0.0, duration_s=2.0)'",
    )
    src.add_argument(
        "--tool-calls-file",
        type=str,
        metavar="PATH",
        help="File containing a tool-call block or call list",
    )
    p.add_argument("--send-zmq", action="store_true", help="Publish steps to deploy ZMQManager")
    p.add_argument("--endpoint", type=str, default="tcp://127.0.0.1:5556", help="ZMQ PUB endpoint")
    p.add_argument("--planner-dt", type=float, default=0.1, help="Republish interval when sending (default 10 Hz)")
    p.add_argument("--segment-dwell-s", type=float, default=3.0, help="Default hold if duration_s missing")
    p.add_argument("--duration-min-s", type=float, default=0.5)
    p.add_argument("--duration-max-s", type=float, default=120.0)
    p.add_argument(
        "--no-interrupt-between",
        action="store_true",
        help="Do not send IDLE between sequential steps",
    )
    p.add_argument("--show-wire", action="store_true", help="Print first bytes of ZMQ wire payload")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    text = args.tool_calls
    if args.tool_calls_file:
        from pathlib import Path
        text = Path(args.tool_calls_file).read_text(encoding="utf-8")

    try:
        raw_calls, steps = tool_calls_to_planner_steps(text)
    except Exception as exc:
        print(f"[error] failed to parse/map tool calls: {exc}", file=sys.stderr)
        return 1

    if not steps:
        print("[warn] no executable tool calls (only select_motion_mode or empty input?)")
        print(f"parsed raw calls: {raw_calls}")
        return 0

    print("Parsed tool calls:")
    for name, call_args in raw_calls:
        print(f"  - {name}({', '.join(f'{k}={call_args[k]!r}' for k in call_args)})")

    _print_steps(steps, show_wire=args.show_wire)

    if args.send_zmq:
        send_steps_over_zmq(
            steps,
            endpoint=args.endpoint,
            planner_dt=args.planner_dt,
            default_duration_s=args.segment_dwell_s,
            min_s=args.duration_min_s,
            max_s=args.duration_max_s,
            interrupt_between=not args.no_interrupt_between,
        )
    else:
        print(
            "\n[dry-run] not sent to ZMQ. Re-run with --send-zmq while deploy is up "
            "(./deploy.sh --input-type zmq_manager real)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
