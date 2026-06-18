"""Tests for the deterministic voice command parser."""

from __future__ import annotations

import pytest

from voice_control.parser import DeterministicParser, normalize_text
from voice_control.schemas import (
    ClarifyCommand,
    GetUpCommand,
    SetBoxingActionCommand,
    SetCrawlCommand,
    SetNavigationCommand,
    SetPostureCommand,
    StopCommand,
)


@pytest.fixture
def parser() -> DeterministicParser:
    return DeterministicParser(confidence_threshold=0.75)


def test_walk_forward(parser):
    cmd = parser.parse("walk forward").command
    assert isinstance(cmd, SetNavigationCommand)
    assert cmd.heading_deg == 0.0
    assert cmd.style == "walking"
    assert cmd.velocity_mps > 0.0


def test_walk_backward_heading_180(parser):
    cmd = parser.parse("walk backward").command
    assert isinstance(cmd, SetNavigationCommand)
    assert cmd.heading_deg == 180.0


def test_move_left_heading_left(parser):
    cmd = parser.parse("move left").command
    assert isinstance(cmd, SetNavigationCommand)
    assert cmd.heading_deg == -90.0


def test_move_right_heading_right(parser):
    cmd = parser.parse("move right").command
    assert isinstance(cmd, SetNavigationCommand)
    assert cmd.heading_deg == 90.0


def test_run_forward_running_style(parser):
    cmd = parser.parse("run forward").command
    assert isinstance(cmd, SetNavigationCommand)
    assert cmd.style == "running"
    assert cmd.velocity_mps >= 1.5


def test_move_forward_slowly(parser):
    cmd = parser.parse("move forward slowly").command
    assert isinstance(cmd, SetNavigationCommand)
    assert cmd.velocity_mps == pytest.approx(0.3)


def test_move_forward_fast(parser):
    cmd = parser.parse("walk forward fast").command
    assert isinstance(cmd, SetNavigationCommand)
    assert cmd.velocity_mps == pytest.approx(1.0)


def test_sprint_pre_clamp_velocity(parser):
    # The parser emits the nominal sprint velocity; safety clamps later.
    cmd = parser.parse("sprint forward").command
    assert isinstance(cmd, SetNavigationCommand)
    assert cmd.velocity_mps == pytest.approx(2.0)
    assert cmd.style == "running"


def test_turn_left_is_in_place(parser):
    cmd = parser.parse("turn left").command
    assert isinstance(cmd, SetNavigationCommand)
    assert cmd.heading_deg == -90.0
    assert cmd.velocity_mps == 0.0


def test_stop(parser):
    cmd = parser.parse("stop").command
    assert isinstance(cmd, StopCommand)


def test_emergency_stop(parser):
    cmd = parser.parse("emergency stop now").command
    assert isinstance(cmd, StopCommand)


def test_stop_words_variants(parser):
    for word in ["halt", "freeze", "cancel", "abort"]:
        assert isinstance(parser.parse(word).command, StopCommand)


def test_squat(parser):
    cmd = parser.parse("squat").command
    assert isinstance(cmd, SetPostureCommand)
    assert cmd.posture == "squat"
    assert cmd.pelvis_height_m == pytest.approx(0.5)


def test_squat_lower_height(parser):
    cmd = parser.parse("squat lower").command
    assert isinstance(cmd, SetPostureCommand)
    assert cmd.pelvis_height_m == pytest.approx(0.35)
    assert 0.3 <= cmd.pelvis_height_m <= 0.8


def test_squat_higher_height(parser):
    cmd = parser.parse("squat higher").command
    assert cmd.pelvis_height_m == pytest.approx(0.65)


def test_kneel_default_two_legs(parser):
    cmd = parser.parse("kneel").command
    assert isinstance(cmd, SetPostureCommand)
    assert cmd.posture == "kneel_two_legs"


def test_kneel_one_knee(parser):
    cmd = parser.parse("kneel on one knee").command
    assert isinstance(cmd, SetPostureCommand)
    assert cmd.posture == "kneel_one_leg"


def test_stand_up(parser):
    cmd = parser.parse("stand up").command
    assert isinstance(cmd, SetPostureCommand)
    assert cmd.posture == "stand"


def test_get_up(parser):
    cmd = parser.parse("get up").command
    assert isinstance(cmd, GetUpCommand)


def test_crawl_forward(parser):
    cmd = parser.parse("crawl forward").command
    assert isinstance(cmd, SetCrawlCommand)
    assert cmd.velocity_mps <= 0.5
    assert cmd.crawl_style == "elbow_knee"
    assert cmd.heading_deg == 0.0


def test_hand_crawl_forward(parser):
    cmd = parser.parse("hand crawl forward").command
    assert isinstance(cmd, SetCrawlCommand)
    assert cmd.crawl_style == "hand_crawl"


def test_left_jab(parser):
    cmd = parser.parse("left jab").command
    assert isinstance(cmd, SetBoxingActionCommand)
    assert cmd.action == "left_jab"


def test_right_hook(parser):
    cmd = parser.parse("right hook").command
    assert isinstance(cmd, SetBoxingActionCommand)
    assert cmd.action == "right_hook"


def test_block(parser):
    cmd = parser.parse("block").command
    assert isinstance(cmd, SetBoxingActionCommand)
    assert cmd.action == "block"


def test_boxing_stance(parser):
    cmd = parser.parse("boxing stance").command
    assert isinstance(cmd, SetBoxingActionCommand)
    assert cmd.action == "stance"


def test_bare_jab_is_clarify(parser):
    cmd = parser.parse("jab").command
    assert isinstance(cmd, ClarifyCommand)


def test_side_step_outside_boxing_is_clarify(parser):
    cmd = parser.parse("side step").command
    assert isinstance(cmd, ClarifyCommand)


def test_side_step_in_boxing(parser):
    cmd = parser.parse("side step", boxing_active=True).command
    assert isinstance(cmd, SetBoxingActionCommand)
    assert cmd.action == "side_step"


def test_dance_like_a_monkey_is_clarify(parser):
    result = parser.parse("dance like a monkey")
    assert isinstance(result.command, ClarifyCommand)
    assert result.confidence < 0.75


def test_unsupported_text_motion_is_clarify(parser):
    assert isinstance(parser.parse("sing a song").command, ClarifyCommand)


def test_explicit_velocity_spoken_number(parser):
    cmd = parser.parse("move forward at one point five meters per second").command
    assert isinstance(cmd, SetNavigationCommand)
    assert cmd.velocity_mps == pytest.approx(1.5)


def test_explicit_velocity_digits(parser):
    cmd = parser.parse("walk forward at 0.4 mps").command
    assert cmd.velocity_mps == pytest.approx(0.4)


def test_explicit_heading_degrees(parser):
    cmd = parser.parse("walk heading 45 degrees").command
    assert isinstance(cmd, SetNavigationCommand)
    assert cmd.heading_deg == pytest.approx(45.0)


def test_normalize_text_punctuation_and_case():
    assert normalize_text("Walk, Forward!!") == "walk forward"


def test_style_walking_variants(parser):
    cmd = parser.parse("stealth walk forward").command
    assert isinstance(cmd, SetNavigationCommand)
    assert cmd.style == "stealth"


def test_high_velocity_not_clamped_by_parser(parser):
    cmd = parser.parse("walk forward at velocity 5 meters per second").command
    assert isinstance(cmd, SetNavigationCommand)
    assert cmd.velocity_mps == pytest.approx(5.0)


def test_turn_around(parser):
    cmd = parser.parse("turn around").command
    assert isinstance(cmd, SetNavigationCommand)
    assert cmd.heading_deg == 180.0
    assert cmd.velocity_mps == 0.0


def test_split_segments_compound():
    from voice_control.parser import split_segments

    segs = split_segments(
        "walk forward at velocity 5 meters per second and then turn around "
        "and kneel on one leg"
    )
    assert segs == [
        "walk forward at velocity 5 meters per second",
        "turn around",
        "kneel on one leg",
    ]


def test_parse_plan_composition(parser):
    plan = parser.parse_plan(
        "walk forward at velocity 5 meters per second and then turn around "
        "and kneel on one leg"
    )
    assert len(plan) == 3

    nav = plan[0].command
    assert isinstance(nav, SetNavigationCommand)
    assert nav.heading_deg == 0.0
    assert nav.velocity_mps == pytest.approx(5.0)

    turn = plan[1].command
    assert isinstance(turn, SetNavigationCommand)
    assert turn.heading_deg == 180.0
    assert turn.velocity_mps == 0.0

    kneel = plan[2].command
    assert isinstance(kneel, SetPostureCommand)
    assert kneel.posture == "kneel_one_leg"


def test_parse_plan_single_command(parser):
    plan = parser.parse_plan("walk forward")
    assert len(plan) == 1
    assert isinstance(plan[0].command, SetNavigationCommand)


def test_parse_plan_threads_boxing_context(parser):
    plan = parser.parse_plan("boxing stance and then side step")
    assert len(plan) == 2
    assert isinstance(plan[1].command, SetBoxingActionCommand)
    assert plan[1].command.action == "side_step"
