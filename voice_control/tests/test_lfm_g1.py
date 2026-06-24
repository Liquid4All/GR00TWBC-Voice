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


def test_trim_generation():
    from voice_control.lfm_g1 import trim_generation

    end = "<|" + "redacted_tool_call_end_kimi" + "|>"
    raw = f"<|tool_call_start|>[stop(reason=\"user_request\")]{end} extra prose"
    assert trim_generation(raw).endswith(end)
    assert "extra prose" not in trim_generation(raw)


def test_build_chat_messages_simple():
    from voice_control.lfm_g1 import SYSTEM_PROMPT, build_chat_messages

    msgs = build_chat_messages("walk forward")
    assert msgs == [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "walk forward"}]


def test_build_chat_messages_history():
    from voice_control.lfm_g1 import SYSTEM_PROMPT, build_chat_messages

    history = [{"role": "user", "content": "go left"}, {"role": "assistant", "content": "ok"}]
    msgs = build_chat_messages("now stop", history=history)
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == SYSTEM_PROMPT
    assert msgs[1:] == history
