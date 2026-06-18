"""Tests for safety gating (clamps were removed)."""

from __future__ import annotations

from voice_control.config import SafetyConfig
from voice_control.safety import SafetyGuard
from voice_control.schemas import (
    ClarifyCommand,
    SetNavigationCommand,
    StopCommand,
    StopReason,
)


def default_guard(**overrides) -> SafetyGuard:
    cfg = SafetyConfig(**overrides)
    return SafetyGuard(cfg)


def test_should_execute_requires_both_flags():
    assert not SafetyGuard(SafetyConfig(dry_run=True, execute=True)).should_execute()
    assert not SafetyGuard(SafetyConfig(dry_run=False, execute=False)).should_execute()
    assert SafetyGuard(SafetyConfig(dry_run=False, execute=True)).should_execute()


def test_confidence_gate_blocks_low_confidence_motion():
    cmd = SetNavigationCommand(velocity_mps=0.5, heading_deg=0.0)
    assert SafetyGuard.passes_confidence(cmd, confidence=0.9, threshold=0.75)
    assert not SafetyGuard.passes_confidence(cmd, confidence=0.5, threshold=0.75)


def test_confidence_gate_allows_stop_always():
    stop = StopCommand(reason=StopReason.USER_REQUEST)
    assert SafetyGuard.passes_confidence(stop, confidence=0.0, threshold=0.75)


def test_confidence_gate_allows_clarify_always():
    clarify = ClarifyCommand(question="q", original_text="x")
    assert SafetyGuard.passes_confidence(clarify, confidence=0.0, threshold=0.75)


def test_stop_for_safety():
    guard = default_guard()
    stop = guard.stop_for_safety()
    assert isinstance(stop, StopCommand)
    assert stop.reason == StopReason.SAFETY.value
