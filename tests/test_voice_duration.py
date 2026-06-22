"""Tests for the deterministic per-step duration estimator."""

import pytest

from voice_control.config import ParserConfig
from voice_control.duration import HeuristicDurationEstimator, extract_distance_m
from voice_control.schemas import (
    SetCrawlCommand,
    SetNavigationCommand,
    SetPostureCommand,
)


@pytest.fixture
def estimator():
    return HeuristicDurationEstimator(ParserConfig())


def test_extract_distance_variants():
    assert extract_distance_m("walk forward 5 meters") == 5.0
    assert extract_distance_m("move 3.5 m ahead") == 3.5
    assert extract_distance_m("go 10 metres") == 10.0
    assert extract_distance_m("walk forward") is None


def test_distance_over_speed_plus_settle(estimator):
    cmd = SetNavigationCommand(velocity_mps=3.0, heading_deg=0.0)
    # 6 / 3 + 1 settle = 3.0
    assert estimator.estimate(cmd, "walk 6 meters at 3 mps") == pytest.approx(3.0)


def test_estimate_is_bounded(estimator):
    cmd = SetNavigationCommand(velocity_mps=0.01, heading_deg=0.0)
    # Slow speed + large distance would run away; must clamp to the ceiling.
    assert estimator.estimate(cmd, "walk 1000 meters") == ParserConfig().duration_max_s
    # The floor is respected for a configured higher minimum.
    floored = HeuristicDurationEstimator(
        ParserConfig(duration_min_s=2.0)
    ).estimate(SetPostureCommand(posture="stand"), "stand")
    assert floored >= 2.0


def test_distance_scales_with_distance(estimator):
    far = estimator.heuristic(
        SetNavigationCommand(velocity_mps=3.0, heading_deg=0.0), "walk 5 meters at 3 mps"
    )
    near = estimator.heuristic(
        SetNavigationCommand(velocity_mps=3.0, heading_deg=0.0), "walk 3 meters at 3 mps"
    )
    assert far > near
    assert far == pytest.approx(5 / 3 + 1.0)
    assert near == pytest.approx(3 / 3 + 1.0)


def test_slower_speed_takes_longer(estimator):
    slow = SetNavigationCommand(velocity_mps=1.0, heading_deg=0.0)
    fast = SetNavigationCommand(velocity_mps=3.0, heading_deg=0.0)
    assert estimator.heuristic(slow, "walk 6 meters") > estimator.heuristic(fast, "walk 6 meters")


def test_in_place_turn_scales_with_angle(estimator):
    turn180 = SetNavigationCommand(velocity_mps=0.0, heading_deg=180.0)
    turn90 = SetNavigationCommand(velocity_mps=0.0, heading_deg=90.0)
    assert estimator.heuristic(turn180, "turn around") > estimator.heuristic(turn90, "turn left")


def test_posture_and_crawl(estimator):
    posture = SetPostureCommand(posture="kneel_one_leg")
    crawl = SetCrawlCommand(velocity_mps=0.25, heading_deg=0.0)
    assert estimator.heuristic(posture, "kneel on one leg") == pytest.approx(3.0)
    # crawl with distance: 2 / 0.25 + 1 = 9.0
    assert estimator.heuristic(crawl, "crawl forward 2 meters") == pytest.approx(9.0)
