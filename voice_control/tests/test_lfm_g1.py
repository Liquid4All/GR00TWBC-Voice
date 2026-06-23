"""Unit tests for LFM G1 tool extraction and mapping (no model load)."""

from __future__ import annotations

import pytest

from voice_control.config import ParserConfig
from voice_control.lfm_g1 import G1ToolMapper, extract_g1_tool_calls, resolve_lfm_model_path
from voice_control.parsers import NavStyle, SetNavigationCommand, StopCommand, StopReason


def _sample_block(*calls: str) -> str:
    inner = ", ".join(calls)
    end = "<|" + "redacted_tool_call_end_kimi" + "|>"
    return f"<|tool_call_start|>[{inner}]{end}"


def test_extract_g1_tool_calls_single():
    text = _sample_block('planner_move(velocity_mps=0.5, heading_deg=0.0, yaw_rate_dps=45.0, duration_s=4.0)')
    calls = extract_g1_tool_calls(text)
    assert calls == [("planner_move", {"velocity_mps": 0.5, "heading_deg": 0.0, "yaw_rate_dps": 45.0, "duration_s": 4.0})]


def test_extract_g1_tool_calls_multi():
    text = _sample_block(
        'select_motion_mode(motion_set="locomotion", mode="walk")',
        'planner_move(velocity_mps=0.3, heading_deg=90.0, yaw_rate_dps=30.0, duration_s=2.0)',
        'stop(reason="user_request")',
    )
    calls = extract_g1_tool_calls(text)
    assert len(calls) == 3
    assert calls[0][0] == "select_motion_mode"
    assert calls[-1] == ("stop", {"reason": "user_request"})


def test_extract_g1_tool_calls_missing_block():
    with pytest.raises(ValueError, match="no tool call block"):
        extract_g1_tool_calls("walk forward")


def test_mapper_planner_move_with_mode():
    cfg = ParserConfig()
    m = G1ToolMapper(cfg)
    m.map("select_motion_mode", {"motion_set": "locomotion", "mode": "run"})
    cmd = m.map("planner_move", {"velocity_mps": 1.0, "heading_deg": 0.0, "duration_s": 5.0, "yaw_rate_dps": 45.0})
    assert isinstance(cmd, SetNavigationCommand)
    assert cmd.velocity_mps == 1.0
    assert cmd.style == NavStyle.RUNNING
    assert cmd.duration_s == 5.0


def test_mapper_stop():
    cmd = G1ToolMapper(ParserConfig()).map("stop", {"reason": "user_request"})
    assert isinstance(cmd, StopCommand)
    assert cmd.reason == StopReason.USER_REQUEST


def test_mapper_select_motion_mode_returns_none():
    assert G1ToolMapper(ParserConfig()).map("select_motion_mode", {"motion_set": "locomotion", "mode": "walk"}) is None


def test_resolve_lfm_model_path_local(tmp_path):
    ckpt = tmp_path / "lfm_g1"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}")
    path, local = resolve_lfm_model_path(str(ckpt))
    assert path == str(ckpt.resolve())
    assert local is True


def test_defaults_fill_duration():
    from voice_control.lfm_g1 import LFMG1Parser

    p = LFMG1Parser(ParserConfig(lfm_default_duration_s=3.5, lfm_default_velocity_mps=0.4))
    filled = p._defaults("planner_move", {"heading_deg": 90.0})
    assert filled["duration_s"] == 3.5
    assert filled["velocity_mps"] == 0.4
    assert filled["heading_deg"] == 90.0
