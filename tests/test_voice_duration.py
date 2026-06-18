"""Tests for the LLM-based per-step duration estimator."""

import pytest

import voice_control.duration as duration_mod
import voice_control.llm_parser as llm_mod
from voice_control.config import ParserConfig
from voice_control.duration import (
    LLMDurationEstimator,
    extract_distance_m,
    parse_duration_seconds,
)
from voice_control.schemas import (
    SetCrawlCommand,
    SetNavigationCommand,
    SetPostureCommand,
)


@pytest.fixture
def estimator():
    return LLMDurationEstimator(ParserConfig())


@pytest.fixture
def mock_llm(monkeypatch):
    def _set(value):
        monkeypatch.setattr(
            duration_mod, "call_llm_completion",
            lambda *a, **k: f'{{"duration_s": {value}}}',
        )
    return _set


def test_extract_distance_variants():
    assert extract_distance_m("walk forward 5 meters") == 5.0
    assert extract_distance_m("move 3.5 m ahead") == 3.5
    assert extract_distance_m("go 10 metres") == 10.0
    assert extract_distance_m("walk forward") is None


def test_llm_value_is_used(estimator, mock_llm):
    mock_llm(7.5)
    cmd = SetNavigationCommand(velocity_mps=3.0, heading_deg=0.0)
    assert estimator.estimate(cmd, "walk 5 meters at 3 mps") == 7.5


def test_llm_value_is_bounded(estimator, mock_llm):
    cmd = SetNavigationCommand(velocity_mps=3.0, heading_deg=0.0)
    mock_llm(9999)
    assert estimator.estimate(cmd, "x") == ParserConfig().llm_duration_max_s
    mock_llm(0.001)
    assert estimator.estimate(cmd, "x") == ParserConfig().llm_duration_min_s


def test_falls_back_to_heuristic_on_bad_json(estimator, monkeypatch):
    monkeypatch.setattr(
        duration_mod, "call_llm_completion", lambda *a, **k: "not json at all"
    )
    cmd = SetNavigationCommand(velocity_mps=3.0, heading_deg=0.0)
    # heuristic: distance / velocity + 1 settle = 6/3 + 1 = 3.0
    assert estimator.estimate(cmd, "walk 6 meters at 3 mps") == pytest.approx(3.0)


def test_falls_back_to_heuristic_when_llm_unavailable(estimator, monkeypatch):
    def boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(duration_mod, "call_llm_completion", boom)
    cmd = SetNavigationCommand(velocity_mps=3.0, heading_deg=0.0)
    assert estimator.estimate(cmd, "walk 6 meters at 3 mps") == pytest.approx(3.0)


def test_heuristic_distance_scales_with_distance(estimator):
    # Farther distance at the same speed must take longer.
    far = SetNavigationCommand(velocity_mps=3.0, heading_deg=0.0)
    near = SetNavigationCommand(velocity_mps=3.0, heading_deg=0.0)
    d_far = estimator.heuristic(far, "walk 5 meters at 3 mps")
    d_near = estimator.heuristic(near, "walk 3 meters at 3 mps")
    assert d_far > d_near
    assert d_far == pytest.approx(5 / 3 + 1.0)
    assert d_near == pytest.approx(3 / 3 + 1.0)


def test_heuristic_slower_speed_takes_longer(estimator):
    slow = SetNavigationCommand(velocity_mps=1.0, heading_deg=0.0)
    fast = SetNavigationCommand(velocity_mps=3.0, heading_deg=0.0)
    assert estimator.heuristic(slow, "walk 6 meters") > estimator.heuristic(fast, "walk 6 meters")


def test_heuristic_in_place_turn_scales_with_angle(estimator):
    turn180 = SetNavigationCommand(velocity_mps=0.0, heading_deg=180.0)
    turn90 = SetNavigationCommand(velocity_mps=0.0, heading_deg=90.0)
    assert estimator.heuristic(turn180, "turn around") > estimator.heuristic(turn90, "turn left")


def test_parse_duration_strict_json():
    assert parse_duration_seconds('{"duration_s": 2.67}') == 2.67


def test_parse_duration_free_text_key():
    assert parse_duration_seconds("answer duration_s: 4.5 seconds") == 4.5


def test_parse_duration_boxed_reasoning():
    # GRPO reasoning models often emit a \boxed{} final answer.
    text = "We move 5 m at 3 m/s ~ 1.67 s plus settle.\n\\boxed{2.67}"
    assert parse_duration_seconds(text) == 2.67


def test_parse_duration_trailing_number_with_unit():
    text = "distance 6 meters at 2 mps -> 6/2=3 plus 1 = 4 seconds"
    assert parse_duration_seconds(text) == 4.0


def test_parse_duration_none_on_garbage():
    assert parse_duration_seconds("no numbers here") is None
    assert parse_duration_seconds("") is None


def test_hf_backend_dispatch(monkeypatch):
    # Default backend is hf_transformers; call_llm_completion should route there.
    cfg = ParserConfig()
    assert cfg.llm_backend == "hf_transformers"
    import voice_control.hf_backend as hf
    monkeypatch.setattr(hf, "generate", lambda c, prompt, max_new_tokens=None: "\\boxed{7.5}")
    out = llm_mod.call_llm_completion(cfg, "prompt")
    assert "7.5" in out


def test_estimator_uses_hf_reasoning_output(monkeypatch):
    cfg = ParserConfig()
    import voice_control.hf_backend as hf
    monkeypatch.setattr(
        hf, "generate",
        lambda c, prompt, max_new_tokens=None: "think...\n\\boxed{7.5}",
    )
    est = LLMDurationEstimator(cfg)
    cmd = SetNavigationCommand(velocity_mps=3.0, heading_deg=0.0)
    assert est.estimate(cmd, "walk 5 meters at 3 mps") == 7.5


def test_heuristic_posture_and_crawl(estimator):
    posture = SetPostureCommand(posture="kneel_one_leg")
    crawl = SetCrawlCommand(velocity_mps=0.25, heading_deg=0.0)
    assert estimator.heuristic(posture, "kneel on one leg") == pytest.approx(3.0)
    # crawl with distance: 2 / 0.25 + 1 = 9.0
    assert estimator.heuristic(crawl, "crawl forward 2 meters") == pytest.approx(9.0)
