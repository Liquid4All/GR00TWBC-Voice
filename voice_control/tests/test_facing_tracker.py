"""Tests for facing-direction tracking into kinematic planner inputs."""

from __future__ import annotations

import pytest

from voice_control.config import ParserConfig
from voice_control.lfm_g1 import G1ToolMapper
from voice_control.parsers import (
    HoldPoseCommand,
    RotateInPlaceCommand,
    SetNavigationCommand,
)
from voice_control.publisher import FacingTracker, StubPublisher
from voice_control.skills import heading_to_direction


def test_heading_to_direction_matches_deploy_convention():
    assert heading_to_direction(0.0) == pytest.approx((1.0, 0.0, 0.0))
    assert heading_to_direction(90.0) == pytest.approx((0.0, 1.0, 0.0))
    assert heading_to_direction(-90.0) == pytest.approx((0.0, -1.0, 0.0))


def test_mapper_rotate_in_place_returns_relative_command():
    cmd = G1ToolMapper(ParserConfig()).map(
        "rotate_in_place",
        {"angle_deg": -90.0, "yaw_rate_dps": 90.0, "duration_s": 1.0},
    )
    assert isinstance(cmd, RotateInPlaceCommand)
    assert cmd.angle_deg == -90.0
    assert cmd.yaw_rate_dps == 90.0


def test_mapper_hold_pose_preserves_facing_semantics():
    cmd = G1ToolMapper(ParserConfig()).map("hold_pose", {"duration_s": 2.0})
    assert isinstance(cmd, HoldPoseCommand)


def test_facing_tracker_forward_then_turn_right():
    tracker = FacingTracker()
    pub = StubPublisher()
    pub.facing = tracker

    pub.publish(SetNavigationCommand(velocity_mps=1.0, heading_deg=0.0, duration_s=3.0))
    state1 = pub.published[-1]["movement_state"]
    assert state1["movement_direction"] == pytest.approx([1.0, 0.0, 0.0])
    assert state1["facing_direction"] == pytest.approx([1.0, 0.0, 0.0])
    assert state1["movement_speed"] == pytest.approx(1.0)

    pub.publish(RotateInPlaceCommand(angle_deg=-90.0, yaw_rate_dps=90.0, duration_s=1.0))
    state2 = pub.published[-1]["movement_state"]
    assert state2["movement_direction"] == pytest.approx([0.0, 0.0, 0.0])
    assert state2["facing_direction"] == pytest.approx([0.0, -1.0, 0.0])
    assert tracker.heading_deg == pytest.approx(-90.0)


def test_facing_tracker_interrupt_preserves_heading():
    tracker = FacingTracker()
    tracker.to_planner_fields(RotateInPlaceCommand(angle_deg=90.0, duration_s=1.0))
    assert tracker.heading_deg == pytest.approx(90.0)

    pub = StubPublisher()
    pub.facing = tracker
    pub.interrupt()
    idle = pub.published[-1]["movement_state"]
    assert idle["facing_direction"] == pytest.approx([0.0, 1.0, 0.0])
    assert idle["movement_direction"] == pytest.approx([0.0, 0.0, 0.0])
    assert idle["movement_speed"] == pytest.approx(-1.0)


def test_facing_tracker_absolute_turn_for_deterministic_parser():
    tracker = FacingTracker()
    fields = tracker.to_planner_fields(
        SetNavigationCommand(velocity_mps=0.0, heading_deg=180.0),
    )
    assert fields.facing == pytest.approx((-1.0, 0.0, 0.0))
    assert tracker.heading_deg == pytest.approx(180.0)
