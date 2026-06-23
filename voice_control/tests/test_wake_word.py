"""Wake word config and model resolution tests (no mic / openwakeword load)."""

from __future__ import annotations

from voice_control.audio import WakeGate, WakeMode, resolve_oww_model
from voice_control.config import WakeConfig


def test_resolve_oww_model_explicit():
    assert resolve_oww_model("alexa", "anything") == "alexa"


def test_resolve_oww_model_from_phrase():
    assert resolve_oww_model("", "hey jarvis") == "hey_jarvis"


def test_wake_gate_push_to_talk():
    gate = WakeGate(WakeConfig(mode="push_to_talk"))
    assert gate.mode == WakeMode.PUSH_TO_TALK


def test_wake_gate_wake_word_builds_detector():
    gate = WakeGate(WakeConfig(mode="wake_word", oww_model="hey_jarvis"))
    assert gate.mode == WakeMode.WAKE_WORD
    assert gate._oww is not None
    assert gate._oww.model_name == "hey_jarvis"
