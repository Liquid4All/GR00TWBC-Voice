"""Tests for tool_call_planner_bridge."""

from __future__ import annotations

import pytest

from voice_control.tool_call_planner_bridge import (
    map_tool_calls,
    movement_state_to_onnx_inputs,
    parse_tool_calls,
    tool_calls_to_planner_steps,
)


def test_parse_single_planner_move():
    calls = parse_tool_calls(
        'planner_move(velocity_mps=1.0, heading_deg=0.0, yaw_rate_dps=0.0, duration_s=2.0)'
    )
    assert calls == [
        ("planner_move", {
            "velocity_mps": 1.0,
            "heading_deg": 0.0,
            "yaw_rate_dps": 0.0,
            "duration_s": 2.0,
        }),
    ]


def test_tool_calls_to_planner_steps_forward():
    _, steps = tool_calls_to_planner_steps(
        'planner_move(velocity_mps=1.0, heading_deg=0.0, yaw_rate_dps=0.0, duration_s=2.0)'
    )
    assert len(steps) == 1
    assert steps[0].onnx_inputs["target_vel"] == pytest.approx(1.0)
    assert steps[0].onnx_inputs["movement_direction"] == pytest.approx([1.0, 0.0, 0.0])
    assert steps[0].onnx_inputs["facing_direction"] == pytest.approx([1.0, 0.0, 0.0])


def test_tool_calls_to_planner_steps_multistep_facing():
    _, steps = tool_calls_to_planner_steps(
        'planner_move(velocity_mps=1.0, heading_deg=0.0, yaw_rate_dps=0.0, duration_s=1.0), '
        'rotate_in_place(angle_deg=-90.0, yaw_rate_dps=90.0, duration_s=1.0)'
    )
    assert len(steps) == 2
    assert steps[1].onnx_inputs["facing_direction"] == pytest.approx([0.0, -1.0, 0.0])
    assert steps[1].onnx_inputs["movement_direction"] == pytest.approx([0.0, 0.0, 0.0])


def test_movement_state_to_onnx_inputs_keys():
    state = {
        "locomotion_mode": 2,
        "movement_direction": [1.0, 0.0, 0.0],
        "facing_direction": [1.0, 0.0, 0.0],
        "movement_speed": 0.5,
        "height": -1.0,
    }
    onnx = movement_state_to_onnx_inputs(state)
    assert set(onnx) == {"mode", "target_vel", "target_height", "movement_direction", "facing_direction"}


def test_map_tool_calls_skips_select_motion_mode():
    cmds = map_tool_calls([("select_motion_mode", {"motion_set": "locomotion", "mode": "walk"})])
    assert cmds == []
