"""Tests for the publisher abstraction, mapping, stream loop and dry-run gating."""

from __future__ import annotations

from voice_control.cli import VoicePipeline
from voice_control.config import Config
from voice_control.publisher import (
    PlannerStreamLoop,
    StubPublisher,
    tool_call_to_planner_fields,
)
from voice_control.skills import LocomotionMode
from voice_control.schemas import (
    BoxingAction,
    NavStyle,
    Posture,
    SetBoxingActionCommand,
    SetCrawlCommand,
    SetNavigationCommand,
    SetPostureCommand,
    StopCommand,
)


# --------------------------------------------------------------------------- #
# Mapping
# --------------------------------------------------------------------------- #

def test_stop_maps_to_idle_and_is_stop():
    fields = tool_call_to_planner_fields(StopCommand())
    assert fields.is_stop is True
    assert fields.mode == int(LocomotionMode.IDLE)


def test_navigation_mapping_forward():
    cmd = SetNavigationCommand(velocity_mps=0.6, heading_deg=0.0, style=NavStyle.WALKING)
    fields = tool_call_to_planner_fields(cmd)
    assert fields.mode == int(LocomotionMode.WALK)
    assert fields.movement[0] > 0.9  # forward
    assert abs(fields.movement[1]) < 1e-6
    assert fields.speed == 0.6


def test_navigation_running_mode():
    cmd = SetNavigationCommand(velocity_mps=1.5, heading_deg=0.0, style=NavStyle.RUNNING)
    assert tool_call_to_planner_fields(cmd).mode == int(LocomotionMode.RUN)


def test_navigation_turn_has_no_translation():
    cmd = SetNavigationCommand(velocity_mps=0.0, heading_deg=-90.0)
    fields = tool_call_to_planner_fields(cmd)
    assert fields.movement == (0.0, 0.0, 0.0)
    assert fields.facing[1] > 0.9  # left = +Y


def test_posture_mapping_sets_height():
    cmd = SetPostureCommand(posture=Posture.SQUAT, pelvis_height_m=0.5)
    fields = tool_call_to_planner_fields(cmd)
    assert fields.mode == int(LocomotionMode.SQUAT)
    assert fields.height == 0.5


def test_crawl_mapping():
    cmd = SetCrawlCommand(velocity_mps=0.25, heading_deg=0.0)
    fields = tool_call_to_planner_fields(cmd)
    assert fields.mode == int(LocomotionMode.ELBOW_CRAWLING)
    assert fields.speed == 0.25


def test_boxing_mapping():
    cmd = SetBoxingActionCommand(action=BoxingAction.LEFT_JAB)
    assert tool_call_to_planner_fields(cmd).mode == int(LocomotionMode.LEFT_JAB)


# --------------------------------------------------------------------------- #
# StubPublisher
# --------------------------------------------------------------------------- #

def test_stub_publisher_records():
    pub = StubPublisher()
    pub.publish(SetNavigationCommand(velocity_mps=0.6, heading_deg=0.0))
    pub.stop()
    assert len(pub.published) == 2
    assert pub.published[-1]["tool_call"]["tool"] == "stop"


# --------------------------------------------------------------------------- #
# PlannerStreamLoop: hold + watchdog
# --------------------------------------------------------------------------- #

def test_stream_loop_publishes_on_set():
    pub = StubPublisher()
    loop = PlannerStreamLoop(pub, command_timeout_s=2.0, planner_dt=0.1)
    loop.set_command(SetNavigationCommand(velocity_mps=0.6, heading_deg=0.0), now=100.0)
    assert len(pub.published) == 1


def test_stream_loop_holds_at_planner_dt():
    pub = StubPublisher()
    loop = PlannerStreamLoop(pub, command_timeout_s=2.0, planner_dt=0.1)
    loop.set_command(SetNavigationCommand(velocity_mps=0.6, heading_deg=0.0), now=100.0)

    # Too soon -> no republish.
    loop.tick(now=100.05)
    assert len(pub.published) == 1

    # After planner_dt -> republish (velocity-conditioned hold, same command).
    loop.tick(now=100.15)
    assert len(pub.published) == 2
    assert pub.published[-1]["tool_call"]["velocity_mps"] == 0.6


def test_stream_loop_command_timeout_triggers_stop():
    pub = StubPublisher()
    loop = PlannerStreamLoop(pub, command_timeout_s=2.0, planner_dt=0.1)
    loop.set_command(SetNavigationCommand(velocity_mps=0.6, heading_deg=0.0), now=100.0)

    tick = loop.tick(now=103.0)  # 3s > 2s timeout
    assert tick.timed_out is True
    assert pub.published[-1]["tool_call"]["tool"] == "stop"

    # Latched: no further publishes after timeout.
    before = len(pub.published)
    loop.tick(now=104.0)
    assert len(pub.published) == before


# --------------------------------------------------------------------------- #
# Dry-run gating via the full pipeline
# --------------------------------------------------------------------------- #

def test_dry_run_uses_stub_publisher_not_real():
    cfg = Config()  # defaults: dry_run True, execute False, backend "zmq"? no -> stub
    cfg.publisher.backend = "existing_repo"  # even if a real backend is configured...
    pipeline = VoicePipeline(cfg)
    # ...dry-run must still use the StubPublisher, never the real one.
    assert isinstance(pipeline.publisher, StubPublisher)
    pipeline.process_text("walk forward")
    assert any(
        rec["tool_call"]["tool"] == "set_navigation" for rec in pipeline.publisher.published
    )
    pipeline.close()


def test_dry_run_stop_published():
    pipeline = VoicePipeline(Config())
    pipeline.process_text("stop")
    assert any(rec["tool_call"]["tool"] == "stop" for rec in pipeline.publisher.published)
    pipeline.close()


def test_low_confidence_command_not_published():
    pipeline = VoicePipeline(Config())
    pipeline.process_text("dance like a monkey")
    # Clarify never publishes a motion command.
    assert all(
        rec["tool_call"]["tool"] in ("stop",) or rec["tool_call"]["tool"] != "clarify"
        for rec in pipeline.publisher.published
    )
    assert not any(
        rec["tool_call"]["tool"] == "set_navigation" for rec in pipeline.publisher.published
    )
    pipeline.close()


def test_execute_without_config_execute_stays_dry():
    cfg = Config()
    cfg.safety.dry_run = False
    cfg.safety.execute = False  # config gate denies execution
    pipeline = VoicePipeline(cfg)
    assert pipeline.dry_run is True
    assert isinstance(pipeline.publisher, StubPublisher)
    pipeline.close()
