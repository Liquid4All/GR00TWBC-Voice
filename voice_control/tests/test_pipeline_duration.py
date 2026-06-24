"""Tests for duration-based planner command hold queue."""

from __future__ import annotations

from unittest.mock import MagicMock

from voice_control.config import Config, ParserConfig
from voice_control.parsers import RotateInPlaceCommand, SetNavigationCommand, StopCommand
from voice_control.pipeline import VoicePipeline, resolve_command_duration_s
from voice_control.publisher import PlannerStreamLoop, StubPublisher


def test_resolve_command_duration_s_uses_field():
    cmd = SetNavigationCommand(velocity_mps=1.0, heading_deg=0.0, duration_s=2.5)
    assert resolve_command_duration_s(cmd, default_s=3.0, min_s=0.5, max_s=120.0) == 2.5


def test_resolve_command_duration_s_clamps():
    cmd = SetNavigationCommand(velocity_mps=1.0, heading_deg=0.0, duration_s=200.0)
    assert resolve_command_duration_s(cmd, default_s=3.0, min_s=0.5, max_s=10.0) == 10.0


def test_hold_for_ticks_at_planner_dt(monkeypatch):
    sleeps: list[float] = []
    clock = {"t": 0.0}

    def advance() -> float:
        return clock["t"]

    def sleep(dt: float) -> None:
        sleeps.append(dt)
        clock["t"] += dt

    monkeypatch.setattr("voice_control.publisher.time.monotonic", advance)
    monkeypatch.setattr("voice_control.publisher.time.sleep", sleep)

    pub = StubPublisher()
    loop = PlannerStreamLoop(publisher=pub, planner_dt=0.1)
    loop.set_command(SetNavigationCommand(velocity_mps=0.5, heading_deg=0.0, duration_s=0.25))

    loop.hold_for(0.25, background_stream=False)

    assert len(sleeps) >= 2
    assert sum(sleeps) >= 0.25


def test_hold_command_runs_each_plan_step(monkeypatch):
    cfg = Config()
    cfg.safety.dry_run = False
    cfg.safety.execute = True
    cfg.parser.backend = "deterministic"

    pipeline = VoicePipeline(cfg, publisher=StubPublisher())
    holds: list[float] = []
    interrupts: list[str] = []
    monkeypatch.setattr(
        pipeline.stream, "hold_for",
        lambda duration, *, background_stream=False: holds.append(duration),
    )
    monkeypatch.setattr(
        pipeline.stream, "interrupt",
        lambda: interrupts.append("interrupt"),
    )

    step1 = SetNavigationCommand(velocity_mps=1.0, heading_deg=0.0, duration_s=1.0)
    step2 = RotateInPlaceCommand(angle_deg=90.0, yaw_rate_dps=90.0, duration_s=0.5)
    from voice_control.parsers import ParseResult, CONF_STRONG

    pipeline._hold_command(ParseResult(
        ok=True, confidence=CONF_STRONG, raw_text="x", normalized_text="x",
        command=step1, reason="test",
    ))
    pipeline._hold_command(ParseResult(
        ok=True, confidence=CONF_STRONG, raw_text="x", normalized_text="x",
        command=step2, reason="test",
    ))

    assert holds == [1.0, 0.5]
    assert interrupts == ["interrupt", "interrupt"]


def test_interrupt_clears_current_command():
    pub = StubPublisher()
    loop = PlannerStreamLoop(publisher=pub, planner_dt=0.1)
    loop.set_command(SetNavigationCommand(velocity_mps=1.0, heading_deg=0.0, duration_s=1.0))
    loop.interrupt()
    assert loop._current is None
    assert any(entry.get("interrupt") for entry in pub.published)


def test_hold_command_skips_stop():
    cfg = Config()
    cfg.safety.dry_run = False
    cfg.safety.execute = True
    pipeline = VoicePipeline(cfg, publisher=StubPublisher())
    loop = MagicMock()
    pipeline.stream = loop
    from voice_control.parsers import ParseResult, CONF_STRONG

    pipeline._hold_command(ParseResult(
        ok=True, confidence=CONF_STRONG, raw_text="x", normalized_text="x",
        command=StopCommand(), reason="stop",
    ))
    loop.hold_for.assert_not_called()
    loop.interrupt.assert_not_called()
