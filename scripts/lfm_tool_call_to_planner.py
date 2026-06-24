#!/usr/bin/env python3
"""
Standalone: LFM G1 tool calls -> deploy kinematic planner MovementState -> ONNX inputs.

No voice_control / pydantic / torch required. Stdlib only unless you use --send-zmq
(which needs: pip install pyzmq).

Examples
--------
# Tool call -> movement_state dict -> ONNX tensor mapping (dry-run):
python scripts/lfm_tool_call_to_planner.py \\
  --tool-calls 'planner_move(velocity_mps=0.5, heading_deg=0.0, yaw_rate_dps=0.0, duration_s=3.0)'

# You already have a movement_state dict (deploy MovementState format):
python scripts/lfm_tool_call_to_planner.py \\
  --movement-state '{"locomotion_mode":2,"movement_direction":[1,0,0],"facing_direction":[1,0,0],"movement_speed":0.5,"height":-1.0}'

# Send to robot (deploy must use --input-type zmq_manager):
python scripts/lfm_tool_call_to_planner.py \\
  --movement-state '{"locomotion_mode":2,"movement_direction":[1,0,0],"facing_direction":[1,0,0],"movement_speed":0.5,"height":-1.0}' \\
  --send-zmq --hold-seconds 3.0

Deploy ONNX path (C++)
----------------------
gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp
  MovementState -> planner_->UpdatePlanning(mode, movement_speed, height, movement_dir, facing_dir)

gear_sonic_deploy/.../localmotion_kplanner_tensorrt.hpp
  UpdateInputTensors(mode_value, target_vel, target_height, movement_direction, facing_direction, ...)
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import struct
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# Locomotion modes (gear_sonic_deploy localmotion_kplanner.hpp LocomotionMode)
# ---------------------------------------------------------------------------
IDLE = 0
WALK = 2
RUN = 3
SQUAT = 4
HAND_CRAWLING = 8
ELBOW_CRAWLING = 14

MODE_WALK = "walk"
MODE_RUN = "run"
STYLED_TO_MODE = {"happy": 17, "stealth": 18, "injured": 19}
NAV_MODE_TO_LOCO = {
    "slow_walk": 1, "walk": WALK, "run": RUN,
    "happy": 17, "stealth": 18, "injured": 19,
}

SPEED_DEFAULT = -1.0
HEIGHT_DEFAULT = -1.0
HEADER_SIZE = 1280
TOOL_CALL_START = "<|tool_call_start|>"
TOOL_CALL_END = "<|" + "redacted_tool_call_end_kimi" + "|>"


# ---------------------------------------------------------------------------
# Geometry (matches deploy gamepad: facing = (cos θ, sin θ))
# ---------------------------------------------------------------------------
def normalize_heading_deg(heading_deg: float) -> float:
    h = math.fmod(heading_deg, 360.0)
    if h > 180.0:
        h -= 360.0
    elif h < -180.0:
        h += 360.0
    return 180.0 if h == -180.0 else h


def heading_to_direction(heading_deg: float) -> Tuple[float, float, float]:
    rad = math.radians(heading_deg)
    x, y = math.cos(rad), math.sin(rad)
    return (
        0.0 if abs(x) < 1e-9 else x,
        0.0 if abs(y) < 1e-9 else y,
        0.0,
    )


# ---------------------------------------------------------------------------
# Tool-call parsing (stdlib ast — same as voice_control/lfm_g1.py)
# ---------------------------------------------------------------------------
def _ast_calls(source: str) -> List[Tuple[str, Dict[str, Any]]]:
    tree = ast.parse(source, mode="eval")
    if not isinstance(tree.body, ast.List):
        raise ValueError("expected a list of tool calls")
    out: List[Tuple[str, Dict[str, Any]]] = []
    for elt in tree.body.elts:
        if not isinstance(elt, ast.Call) or not isinstance(elt.func, ast.Name):
            raise ValueError(f"invalid tool call node: {ast.dump(elt)}")
        args = {kw.arg: ast.literal_eval(kw.value) for kw in elt.keywords if kw.arg}
        out.append((elt.func.id, args))
    return out


def wrap_tool_block(text: str) -> str:
    raw = text.strip()
    if TOOL_CALL_START in raw:
        return raw
    inner = raw if raw.startswith("[") else f"[{raw}]"
    return f"{TOOL_CALL_START}{inner}{TOOL_CALL_END}"


def extract_tool_calls(text: str) -> List[Tuple[str, Dict[str, Any]]]:
    wrapped = wrap_tool_block(text)
    for end in (TOOL_CALL_END, "<|tool_call_end|>"):
        pat = re.compile(
            re.escape(TOOL_CALL_START) + r"\s*(\[.*?\])\s*" + re.escape(end),
            re.DOTALL,
        )
        match = pat.search(wrapped)
        if match:
            return _ast_calls(match.group(1))
    start = wrapped.find(TOOL_CALL_START)
    if start >= 0:
        rest = wrapped[start + len(TOOL_CALL_START):]
        lb = rest.find("[")
        if lb >= 0:
            depth = 0
            for i, ch in enumerate(rest[lb:], start=lb):
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        return _ast_calls(rest[lb : i + 1])
    raise ValueError(f"no tool call block in: {text[:400]!r}")


# ---------------------------------------------------------------------------
# Internal command representation (no pydantic)
# ---------------------------------------------------------------------------
@dataclass
class NavMove:
    velocity_mps: float
    heading_deg: float
    duration_s: float = 3.0
    style: str = "walking"


@dataclass
class RotateInPlace:
    angle_deg: float
    yaw_rate_dps: float = 90.0
    duration_s: float = 1.0
    style: str = "walking"


@dataclass
class HoldPose:
    duration_s: float = 3.0
    style: str = "walking"


@dataclass
class Stop:
    reason: str = "user_request"


Command = Union[NavMove, RotateInPlace, HoldPose, Stop]


class ToolMapper:
  def __init__(self) -> None:
      self.mode = MODE_WALK

  def map(self, name: str, args: Dict[str, Any]) -> Optional[Command]:
      if name == "select_motion_mode":
          self.mode = str(args.get("mode", self.mode))
          return None
      if name == "stop":
          return Stop(reason=str(args.get("reason", "user_request")))
      if name == "hold_pose":
          return HoldPose(duration_s=float(args["duration_s"]), style=self._style())
      if name == "rotate_in_place":
          return RotateInPlace(
              angle_deg=float(args["angle_deg"]),
              yaw_rate_dps=float(args.get("yaw_rate_dps", 90.0)),
              duration_s=float(args.get("duration_s", 1.0)),
              style=self._style(),
          )
      if name == "planner_move":
          return NavMove(
              velocity_mps=float(args["velocity_mps"]),
              heading_deg=float(args["heading_deg"]),
              duration_s=float(args.get("duration_s", 3.0)),
              style=self._style(),
          )
      raise ValueError(f"unknown tool {name!r}")

  def _style(self) -> str:
      if self.mode == MODE_RUN:
          return "running"
      if self.mode in STYLED_TO_MODE:
          return self.mode
      return "walking"


# ---------------------------------------------------------------------------
# MovementState + ONNX mapping
# ---------------------------------------------------------------------------
@dataclass
class PlannerFields:
    mode: int
    movement: Tuple[float, float, float]
    facing: Tuple[float, float, float]
    speed: float = SPEED_DEFAULT
    height: float = HEIGHT_DEFAULT

    def to_movement_state(self) -> Dict[str, Any]:
        return {
            "locomotion_mode": int(self.mode),
            "movement_direction": list(self.movement),
            "facing_direction": list(self.facing),
            "movement_speed": float(self.speed),
            "height": float(self.height),
        }


class FacingTracker:
    def __init__(self) -> None:
        self.heading_deg = 0.0

    def facing_direction(self) -> Tuple[float, float, float]:
        return heading_to_direction(self.heading_deg)

    def _mode_for_style(self, style: str) -> int:
      if style == "running":
          return RUN
      if style in STYLED_TO_MODE:
          return STYLED_TO_MODE[style]
      return WALK

    def apply(self, cmd: Command) -> PlannerFields:
        if isinstance(cmd, Stop):
            return PlannerFields(
                IDLE, (0.0, 0.0, 0.0), self.facing_direction(),
                speed=SPEED_DEFAULT, height=HEIGHT_DEFAULT,
            )
        if isinstance(cmd, RotateInPlace):
            self.heading_deg = normalize_heading_deg(self.heading_deg + cmd.angle_deg)
            return PlannerFields(
                self._mode_for_style(cmd.style), (0.0, 0.0, 0.0),
                self.facing_direction(),
            )
        if isinstance(cmd, HoldPose):
            return PlannerFields(
                self._mode_for_style(cmd.style), (0.0, 0.0, 0.0),
                self.facing_direction(),
            )
        if isinstance(cmd, NavMove):
            self.heading_deg = normalize_heading_deg(cmd.heading_deg)
            direction = heading_to_direction(self.heading_deg)
            mode = self._mode_for_style(cmd.style)
            if cmd.velocity_mps <= 0.0:
                return PlannerFields(mode, (0.0, 0.0, 0.0), direction)
            return PlannerFields(mode, direction, direction, speed=cmd.velocity_mps)
        raise TypeError(cmd)


def movement_state_to_onnx_inputs(state: Dict[str, Any]) -> Dict[str, Any]:
    """Maps deploy MovementState -> LocalMotionPlanner::UpdateInputTensors arguments."""
    return {
        "mode": int(state["locomotion_mode"]),
        "target_vel": float(state["movement_speed"]),
        "target_height": float(state["height"]),
        "movement_direction": list(state["movement_direction"]),
        "facing_direction": list(state["facing_direction"]),
    }


def format_onnx_pass(state: Dict[str, Any]) -> str:
    """Human-readable trace of how fields land in the ONNX planner."""
    onnx = movement_state_to_onnx_inputs(state)
    lines = [
        "C++ call (g1_deploy_onnx_ref.cpp -> UpdatePlanning -> UpdateInputTensors):",
        "",
        "  planner_->UpdatePlanning(",
        f"    current_mode={onnx['mode']},",
        f"    movement_speed={onnx['target_vel']},",
        f"    target_height={onnx['target_height']},",
        f"    movement_direction=[{onnx['movement_direction'][0]:.4f}, "
        f"{onnx['movement_direction'][1]:.4f}, {onnx['movement_direction'][2]:.4f}],",
        f"    facing_direction=[{onnx['facing_direction'][0]:.4f}, "
        f"{onnx['facing_direction'][1]:.4f}, {onnx['facing_direction'][2]:.4f}],",
        "  );",
        "",
        "Tensor buffer writes (localmotion_kplanner_tensorrt.hpp):",
        f"  mode_values_[0]                  = {onnx['mode']}",
        f"  target_vel_values_[0]            = {onnx['target_vel']}",
        f"  target_height_values_[0]         = {onnx['target_height']}",
        f"  movement_direction_values_[0:3]  = {onnx['movement_direction']}",
        f"  facing_direction_values_[0:3]    = {onnx['facing_direction']}",
        "",
        "Field mapping (movement_state key -> ONNX tensor):",
        "  locomotion_mode      -> mode",
        "  movement_speed       -> target_vel",
        "  height               -> target_height",
        "  movement_direction   -> movement_direction",
        "  facing_direction     -> facing_direction",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ZMQ wire (optional pyzmq)
# ---------------------------------------------------------------------------
def build_planner_wire(fields: PlannerFields) -> bytes:
    header_json = json.dumps({
        "v": 1, "endian": "le", "count": 1,
        "fields": [
            {"name": "mode", "dtype": "i32", "shape": [1]},
            {"name": "movement", "dtype": "f32", "shape": [3]},
            {"name": "facing", "dtype": "f32", "shape": [3]},
            {"name": "speed", "dtype": "f32", "shape": [1]},
            {"name": "height", "dtype": "f32", "shape": [1]},
        ],
    }, separators=(",", ":")).encode("utf-8")
    header = header_json.ljust(HEADER_SIZE, b"\x00")
    payload = b"".join((
        struct.pack("<i", int(fields.mode)),
        struct.pack("<fff", *map(float, fields.movement)),
        struct.pack("<fff", *map(float, fields.facing)),
        struct.pack("<f", float(fields.speed)),
        struct.pack("<f", float(fields.height)),
    ))
    return b"planner" + header + payload


def build_command_wire(start: bool = True, stop: bool = False, planner: bool = True) -> bytes:
    header_json = json.dumps({
        "v": 1, "endian": "le", "count": 1,
        "fields": [
            {"name": "start", "dtype": "u8", "shape": [1]},
            {"name": "stop", "dtype": "u8", "shape": [1]},
            {"name": "planner", "dtype": "u8", "shape": [1]},
        ],
    }, separators=(",", ":")).encode("utf-8")
    header = header_json.ljust(HEADER_SIZE, b"\x00")
    payload = struct.pack("BBB", 1 if start else 0, 1 if stop else 0, 1 if planner else 0)
    return b"command" + header + payload


def send_movement_state_zmq(
    state: Dict[str, Any],
    endpoint: str,
    hold_seconds: float,
    planner_dt: float,
) -> None:
    try:
        import zmq  # type: ignore
    except ImportError:
        print("[error] --send-zmq requires pyzmq: pip install pyzmq", file=sys.stderr)
        sys.exit(1)

    fields = PlannerFields(
        mode=int(state["locomotion_mode"]),
        movement=tuple(state["movement_direction"]),
        facing=tuple(state["facing_direction"]),
        speed=float(state["movement_speed"]),
        height=float(state["height"]),
    )
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.bind(endpoint)
    time.sleep(0.3)
    sock.send(build_command_wire())
    wire = build_planner_wire(fields)
    print(f"[zmq] publishing to {endpoint} for {hold_seconds:.2f}s @ {1.0/planner_dt:.0f} Hz")
    deadline = time.monotonic() + hold_seconds
    while time.monotonic() < deadline:
        sock.send(wire)
        time.sleep(planner_dt)
    sock.close()


# ---------------------------------------------------------------------------
# Pipeline: tool calls -> steps
# ---------------------------------------------------------------------------
@dataclass
class Step:
    tool_name: str
    tool_args: Dict[str, Any]
    duration_s: float
    movement_state: Dict[str, Any]
    onnx_inputs: Dict[str, Any]


def tool_calls_to_steps(text: str) -> Tuple[List[Tuple[str, Dict[str, Any]]], List[Step]]:
    raw = extract_tool_calls(text)
    mapper = ToolMapper()
    tracker = FacingTracker()
    steps: List[Step] = []
    for name, args in raw:
        cmd = mapper.map(name, args)
        if cmd is None:
            continue
        fields = tracker.apply(cmd)
        state = fields.to_movement_state()
        dur = getattr(cmd, "duration_s", 3.0)
        steps.append(Step(
            tool_name=name,
            tool_args=args,
            duration_s=float(dur),
            movement_state=state,
            onnx_inputs=movement_state_to_onnx_inputs(state),
        ))
    return raw, steps


def print_movement_state_flow(state: Dict[str, Any], title: str = "") -> None:
    if title:
        print(title)
    print(f"movement_state  = {state!r}")
    print()
    print(format_onnx_pass(state))


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Standalone LFM tool-call -> MovementState -> ONNX kinematic planner inputs.",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--tool-calls", type=str, help="Tool call(s) or <|tool_call_start|> block")
    src.add_argument("--tool-calls-file", type=str, metavar="PATH")
    src.add_argument(
        "--movement-state",
        type=str,
        help='JSON movement_state dict, e.g. \'{"locomotion_mode":2,...}\'',
    )
    p.add_argument("--send-zmq", action="store_true")
    p.add_argument("--endpoint", default="tcp://127.0.0.1:5556")
    p.add_argument("--hold-seconds", type=float, default=3.0,
                   help="ZMQ republish duration (tool-call steps use each duration_s)")
    p.add_argument("--planner-dt", type=float, default=0.1)
    args = p.parse_args(argv)

    if args.movement_state:
        state = json.loads(args.movement_state)
        print_movement_state_flow(state, "=== Direct movement_state input ===")
        if args.send_zmq:
            send_movement_state_zmq(state, args.endpoint, args.hold_seconds, args.planner_dt)
        return 0

    text = args.tool_calls
    if args.tool_calls_file:
        from pathlib import Path
        text = Path(args.tool_calls_file).read_text(encoding="utf-8")

    try:
        raw_calls, steps = tool_calls_to_steps(text)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    if not steps:
        print("[warn] no executable steps:", raw_calls)
        return 0

    print("Parsed tool calls:")
    for name, a in raw_calls:
        print(f"  - {name}({', '.join(f'{k}={a[k]!r}' for k in a)})")

    for i, step in enumerate(steps, 1):
        print(f"\n=== Step {i}: {step.tool_name} (duration_s={step.duration_s}) ===")
        print(f"tool_args = {step.tool_args!r}")
        print_movement_state_flow(step.movement_state)

    if args.send_zmq:
        for i, step in enumerate(steps, 1):
            hold = step.duration_s if args.tool_calls or args.tool_calls_file else args.hold_seconds
            print(f"\n--- ZMQ step {i}/{len(steps)} ---")
            send_movement_state_zmq(step.movement_state, args.endpoint, hold, args.planner_dt)
    else:
        print("\n[dry-run] Add --send-zmq with deploy running (--input-type zmq_manager).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
