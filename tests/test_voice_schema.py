"""Tests for the closed Pydantic tool-call schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from voice_control.schemas import (
    SetNavigationCommand,
    StopCommand,
    validate_tool_call,
)


def test_valid_navigation_dict():
    cmd = validate_tool_call(
        {
            "tool": "set_navigation",
            "velocity_mps": 0.6,
            "heading_deg": 0.0,
            "style": "walking",
            "duration_s": None,
        }
    )
    assert isinstance(cmd, SetNavigationCommand)
    assert cmd.style == "walking"


def test_valid_stop_dict():
    cmd = validate_tool_call({"tool": "stop", "reason": "user_request"})
    assert isinstance(cmd, StopCommand)


def test_unknown_tool_rejected():
    with pytest.raises(ValidationError):
        validate_tool_call({"tool": "fly", "altitude": 10})


def test_extra_field_rejected():
    with pytest.raises(ValidationError):
        validate_tool_call(
            {"tool": "stop", "reason": "user_request", "danger": True}
        )


def test_high_nav_velocity_accepted_no_upper_clamp():
    # Safety clamps removed: high velocities are accepted (passed to planner).
    cmd = validate_tool_call(
        {"tool": "set_navigation", "velocity_mps": 10.0, "heading_deg": 0.0}
    )
    assert cmd.velocity_mps == 10.0


def test_nav_negative_velocity_rejected():
    with pytest.raises(ValidationError):
        validate_tool_call(
            {"tool": "set_navigation", "velocity_mps": -1.0, "heading_deg": 0.0}
        )


def test_invalid_enum_rejected():
    with pytest.raises(ValidationError):
        validate_tool_call(
            {"tool": "set_navigation", "velocity_mps": 0.5, "heading_deg": 0.0,
             "style": "moonwalk"}
        )


def test_missing_required_field_rejected():
    with pytest.raises(ValidationError):
        validate_tool_call({"tool": "set_navigation", "heading_deg": 0.0})


def test_malformed_tool_call_rejected():
    # Any malformed/out-of-schema payload must be rejected before publishing.
    bad_payloads = [
        {"tool": "set_navigation", "velocity_mps": "fast", "heading_deg": 0.0},
        {"tool": "set_posture", "posture": "backflip"},
        {"tool": "set_boxing_action", "action": "uppercut"},
        {"not_a_tool": 1},
    ]
    for payload in bad_payloads:
        with pytest.raises(ValidationError):
            validate_tool_call(payload)


def test_round_trip_json():
    cmd = validate_tool_call(
        {"tool": "set_posture", "posture": "squat", "pelvis_height_m": 0.5}
    )
    data = cmd.model_dump(mode="json")
    assert data["tool"] == "set_posture"
    assert data["posture"] == "squat"
