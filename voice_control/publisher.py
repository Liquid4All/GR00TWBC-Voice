from __future__ import annotations

import abc
import json
import logging
import struct
import time
from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Sequence, Tuple

from .config import PublisherConfig
from .parsers import (
    ClarifyCommand,
    GetUpCommand,
    HoldPoseCommand,
    ParseResult,
    PlannerToolCallType,
    RotateInPlaceCommand,
    SetBoxingActionCommand,
    SetCrawlCommand,
    SetNavigationCommand,
    SetPostureCommand,
    StopCommand,
    tool_call_to_dict,
)
from . import skills
from .skills import LocomotionMode, heading_to_direction, normalize_heading_deg

log = logging.getLogger(__name__)

_HEADER_SIZE = 1280
_FORWARD = (1.0, 0.0, 0.0)
_SPEED_DEFAULT = -1.0
_HEIGHT_DEFAULT = -1.0


def _zmq_header(fields: list, version: int = 1, count: int = 1) -> bytes:
    header_json = json.dumps({"v": version, "endian": "le", "count": count, "fields": fields},
                             separators=(",", ":")).encode("utf-8")
    if len(header_json) > _HEADER_SIZE:
        raise ValueError(f"Header too large: {len(header_json)} > {_HEADER_SIZE}")
    return header_json.ljust(_HEADER_SIZE, b"\x00")


def build_command_message(start: bool, stop: bool, planner: bool, delta_heading: float | None = None) -> bytes:
    fields = [
        {"name": "start", "dtype": "u8", "shape": [1]},
        {"name": "stop", "dtype": "u8", "shape": [1]},
        {"name": "planner", "dtype": "u8", "shape": [1]},
    ]
    payload = b"".join((struct.pack("B", 1 if start else 0), struct.pack("B", 1 if stop else 0),
                        struct.pack("B", 1 if planner else 0)))
    if delta_heading is not None:
        fields.append({"name": "delta_heading", "dtype": "f32", "shape": [1]})
        payload += struct.pack("<f", float(delta_heading))
    return b"command" + _zmq_header(fields) + payload


def build_planner_message(
    mode: int, movement: Sequence[float], facing: Sequence[float],
    speed: float = -1.0, height: float = -1.0,
) -> bytes:
    if len(movement) != 3 or len(facing) != 3:
        raise ValueError("movement and facing must have length 3")
    fields = [
        {"name": "mode", "dtype": "i32", "shape": [1]},
        {"name": "movement", "dtype": "f32", "shape": [3]},
        {"name": "facing", "dtype": "f32", "shape": [3]},
        {"name": "speed", "dtype": "f32", "shape": [1]},
        {"name": "height", "dtype": "f32", "shape": [1]},
    ]
    payload = b"".join((
        struct.pack("<i", int(mode)),
        struct.pack("<fff", *map(float, movement)),
        struct.pack("<fff", *map(float, facing)),
        struct.pack("<f", float(speed)),
        struct.pack("<f", float(height)),
    ))
    return b"planner" + _zmq_header(fields) + payload


def movement_state_from_planner_fields(fields: Mapping[str, object]) -> dict:
    return {
        "locomotion_mode": int(fields["mode"]),
        "movement_direction": list(fields["movement"]),
        "facing_direction": list(fields["facing"]),
        "movement_speed": float(fields["speed"]),
        "height": float(fields["height"]),
    }


@dataclass
class PlannerFields:
    mode: int
    movement: Tuple[float, float, float]
    facing: Tuple[float, float, float]
    speed: float = _SPEED_DEFAULT
    height: float = _HEIGHT_DEFAULT
    # When True, the publisher should send a "stop" on the command topic.
    is_stop: bool = False

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "movement": list(self.movement),
            "facing": list(self.facing),
            "speed": self.speed,
            "height": self.height,
            "is_stop": self.is_stop,
        }

    def to_movement_state(self) -> dict:
        return movement_state_from_planner_fields(self.as_dict())

    def to_planner_wire(self) -> bytes:
        return build_planner_message(
            mode=self.mode, movement=self.movement, facing=self.facing,
            speed=self.speed, height=self.height,
        )


@dataclass
class FacingTracker:
    """Tracks body-relative facing across sequential planner commands."""

    heading_deg: float = 0.0

    def reset(self) -> None:
        self.heading_deg = 0.0

    def facing_direction(self) -> Tuple[float, float, float]:
        return heading_to_direction(self.heading_deg)

    def to_planner_fields(self, command: PlannerToolCallType) -> PlannerFields:
        if isinstance(command, StopCommand):
            return PlannerFields(
                mode=int(LocomotionMode.IDLE), movement=(0.0, 0.0, 0.0),
                facing=self.facing_direction(), speed=_SPEED_DEFAULT,
                height=_HEIGHT_DEFAULT, is_stop=True,
            )

        if isinstance(command, RotateInPlaceCommand):
            self.heading_deg = normalize_heading_deg(self.heading_deg + command.angle_deg)
            mode = skills.STYLE_TO_MODE.get(command.style, LocomotionMode.WALK)
            return PlannerFields(
                mode=int(mode), movement=(0.0, 0.0, 0.0),
                facing=self.facing_direction(), speed=_SPEED_DEFAULT,
            )

        if isinstance(command, HoldPoseCommand):
            mode = skills.STYLE_TO_MODE.get(command.style, LocomotionMode.WALK)
            return PlannerFields(
                mode=int(mode), movement=(0.0, 0.0, 0.0),
                facing=self.facing_direction(), speed=_SPEED_DEFAULT,
            )

        if isinstance(command, SetNavigationCommand):
            mode = skills.STYLE_TO_MODE.get(command.style, LocomotionMode.WALK)
            if command.style == "drunken":
                log.warning("Style 'drunken' has no dedicated planner mode; using WALK.")
            self.heading_deg = normalize_heading_deg(command.heading_deg)
            direction = heading_to_direction(self.heading_deg)
            if command.velocity_mps <= 0.0:
                # Absolute target facing (deterministic "turn left" / "turn around").
                return PlannerFields(
                    mode=int(mode), movement=(0.0, 0.0, 0.0), facing=direction,
                    speed=_SPEED_DEFAULT,
                )
            return PlannerFields(
                mode=int(mode), movement=direction, facing=direction,
                speed=float(command.velocity_mps),
            )

        if isinstance(command, SetCrawlCommand):
            mode = skills.CRAWL_STYLE_TO_MODE.get(command.crawl_style, LocomotionMode.ELBOW_CRAWLING)
            if command.velocity_mps > 0.0:
                self.heading_deg = normalize_heading_deg(command.heading_deg)
            direction = heading_to_direction(self.heading_deg)
            speed = float(command.velocity_mps) if command.velocity_mps > 0.0 else _SPEED_DEFAULT
            movement = direction if command.velocity_mps > 0.0 else (0.0, 0.0, 0.0)
            return PlannerFields(mode=int(mode), movement=movement, facing=direction, speed=speed)

        if isinstance(command, SetPostureCommand):
            mode = skills.POSTURE_TO_MODE.get(command.posture, LocomotionMode.IDLE)
            height = command.pelvis_height_m if command.pelvis_height_m is not None else _HEIGHT_DEFAULT
            return PlannerFields(
                mode=int(mode), movement=(0.0, 0.0, 0.0), facing=self.facing_direction(),
                speed=_SPEED_DEFAULT, height=float(height),
            )

        if isinstance(command, SetBoxingActionCommand):
            mode = skills.BOXING_ACTION_TO_MODE.get(command.action, LocomotionMode.IDLE_BOXING)
            if command.action in ("block",):
                log.warning("Boxing action 'block' has no dedicated planner mode; using IDLE_BOXING.")
            return PlannerFields(
                mode=int(mode), movement=(0.0, 0.0, 0.0),
                facing=self.facing_direction(), speed=_SPEED_DEFAULT,
            )

        if isinstance(command, GetUpCommand):
            return PlannerFields(
                mode=int(LocomotionMode.IDLE), movement=(0.0, 0.0, 0.0),
                facing=self.facing_direction(), speed=_SPEED_DEFAULT, height=_HEIGHT_DEFAULT,
            )

        return PlannerFields(
            mode=int(LocomotionMode.IDLE), movement=(0.0, 0.0, 0.0),
            facing=self.facing_direction(), speed=_SPEED_DEFAULT, height=_HEIGHT_DEFAULT,
        )


def tool_call_to_planner_fields(
    command: PlannerToolCallType,
    tracker: Optional[FacingTracker] = None,
) -> PlannerFields:
    if tracker is None:
        tracker = FacingTracker()
    return tracker.to_planner_fields(command)


# Publisher abstraction + adapters

def build_planner_idle_wire(
    facing: Sequence[float] = _FORWARD,
) -> bytes:
    """ZMQ planner message: IDLE, zero movement — cuts an in-progress planner segment."""
    return build_planner_message(
        mode=int(LocomotionMode.IDLE),
        movement=(0.0, 0.0, 0.0),
        facing=facing,
        speed=0.0,
        height=_HEIGHT_DEFAULT,
    )


class PlannerCommandPublisher(abc.ABC):

    facing: FacingTracker

    def __init__(self) -> None:
        self.facing = FacingTracker()

    @abc.abstractmethod
    def publish(self, command: PlannerToolCallType) -> None:
        ...

    @abc.abstractmethod
    def publish_fields(self, fields: PlannerFields) -> None:
        ...

    @abc.abstractmethod
    def stop(self) -> None:
        ...

    @abc.abstractmethod
    def interrupt(self) -> None:
        """Send IDLE / zero velocity so deploy replans immediately (not command-topic e-stop)."""
        ...

    def close(self) -> None:  # default no-op
        return None


class StubPublisher(PlannerCommandPublisher):

    def __init__(self) -> None:
        super().__init__()
        self.published: List[dict] = []

    def publish(self, command: PlannerToolCallType) -> None:
        fields = self.facing.to_planner_fields(command)
        record = {
            "tool_call": tool_call_to_dict(command),
            "planner_fields": fields.as_dict(),
            "movement_state": fields.to_movement_state(),
        }
        self.published.append(record)
        log.info(
            "[StubPublisher] %s -> movement_state=%s",
            record["tool_call"], record["movement_state"],
        )

    def publish_fields(self, fields: PlannerFields) -> None:
        if fields.is_stop:
            self.stop()
            return
        record = {
            "planner_fields": fields.as_dict(),
            "movement_state": fields.to_movement_state(),
        }
        self.published.append(record)
        log.info("[StubPublisher] hold -> movement_state=%s", record["movement_state"])

    def stop(self) -> None:
        self.publish(StopCommand())

    def interrupt(self) -> None:
        state = movement_state_from_planner_fields({
            "mode": int(LocomotionMode.IDLE),
            "movement": [0.0, 0.0, 0.0],
            "facing": list(self.facing.facing_direction()),
            "speed": 0.0,
            "height": _HEIGHT_DEFAULT,
        })
        self.published.append({"interrupt": True, "movement_state": state})
        log.info("[StubPublisher] interrupt -> movement_state=%s", state)

    def close(self) -> None:
        log.debug("[StubPublisher] closed (%d commands published)", len(self.published))


class ZmqPublisher(PlannerCommandPublisher):

    def __init__(self, endpoint: str, bind: bool = True) -> None:
        super().__init__()
        self.endpoint = endpoint
        self._bind = bind
        self._socket = None
        self._context = None
        self._started = False
        self._connect()

    def _connect(self) -> None:
        try:
            import zmq  # type: ignore
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "pyzmq is required for the ZMQ publisher. Install with `pip install pyzmq`."
            ) from exc

        self._build_command_message = build_command_message
        self._build_planner_message = build_planner_message
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.PUB)
        if self._bind:
            self._socket.bind(self.endpoint)
        else:
            self._socket.connect(self.endpoint)
        # Allow subscribers to connect before first send.
        time.sleep(0.2)
        # Enable planner mode + start control.
        self._socket.send(self._build_command_message(start=True, stop=False, planner=True))
        self._started = True
        log.info("[ZmqPublisher] PUB on %s (planner mode enabled)", self.endpoint)

    def publish(self, command: PlannerToolCallType) -> None:
        self.publish_fields(self.facing.to_planner_fields(command))

    def publish_fields(self, fields: PlannerFields) -> None:
        if fields.is_stop:
            self.stop()
            return
        if self._socket is None:
            return
        msg = fields.to_planner_wire()
        self._socket.send(msg)
        log.info("[ZmqPublisher] planner <- %s", fields.to_movement_state())

    def stop(self) -> None:
        if self._socket is None:
            return
        self._socket.send(self._build_command_message(start=False, stop=True, planner=True))
        log.info("[ZmqPublisher] STOP sent on command topic")

    def interrupt(self) -> None:
        if self._socket is None:
            return
        self._socket.send(build_planner_idle_wire(self.facing.facing_direction()))
        log.info("[ZmqPublisher] interrupt -> IDLE (zero movement)")

    def close(self) -> None:
        if self._socket is not None:
            try:
                self.stop()
            finally:
                self._socket.close(linger=200)
                self._socket = None
        log.debug("[ZmqPublisher] closed")


class LocalPlannerPublisher(ZmqPublisher):

    def __init__(self, endpoint: str = "tcp://127.0.0.1:5556") -> None:
        super().__init__(endpoint=endpoint, bind=True)


class ExistingRepoPublisher(LocalPlannerPublisher):
    pass


def build_publisher(cfg: PublisherConfig) -> PlannerCommandPublisher:

    backend = cfg.backend.lower()
    if backend == "stub":
        return StubPublisher()
    if backend in ("local_planner", "local", "onboard"):
        return LocalPlannerPublisher(cfg.local_planner_endpoint)
    if backend == "zmq":
        return ZmqPublisher(cfg.zmq_endpoint, bind=cfg.zmq_bind)
    if backend in ("existing_repo", "existing"):
        return ExistingRepoPublisher(cfg.local_planner_endpoint)
    raise ValueError(f"Unknown publisher backend: {cfg.backend!r}")


# Stream loop: 10 Hz hold + watchdog

@dataclass
class StreamTick:
    published: Optional[PlannerToolCallType] = None
    timed_out: bool = False
    reason: str = ""


@dataclass
class PlannerStreamLoop:

    publisher: PlannerCommandPublisher
    command_timeout_s: float = 2.0
    planner_dt: float = 0.1
    _current: Optional[PlannerToolCallType] = field(default=None, init=False)
    _current_fields: Optional[PlannerFields] = field(default=None, init=False)
    _last_update: float = field(default=0.0, init=False)
    _last_publish: float = field(default=0.0, init=False)
    _timed_out: bool = field(default=False, init=False)

    @property
    def facing(self) -> FacingTracker:
        return self.publisher.facing

    def set_command(self, command: PlannerToolCallType, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        self._current = command
        self._current_fields = self.publisher.facing.to_planner_fields(command)
        self._last_update = now
        self._timed_out = False
        # Immediate replan on change.
        self.publisher.publish_fields(self._current_fields)
        self._last_publish = now

    def tick(self, now: Optional[float] = None) -> StreamTick:
        now = time.monotonic() if now is None else now
        if self._current is None:
            return StreamTick()

        # Stop commands latch; nothing to hold.
        if isinstance(self._current, StopCommand):
            return StreamTick()

        # Watchdog: no fresh command within the timeout -> safety stop once.
        if not self._timed_out and (now - self._last_update) > self.command_timeout_s:
            self._timed_out = True
            stop = StopCommand(reason="safety")
            self.publisher.stop()
            self._current = stop
            self._last_publish = now
            log.warning(
                "[PlannerStreamLoop] command timeout (%.2fs) -> safety stop",
                now - self._last_update,
            )
            return StreamTick(published=stop, timed_out=True, reason="command_timeout")

        if self._timed_out:
            return StreamTick(timed_out=True)

        # Velocity-conditioned hold: re-publish the same command at planner_dt.
        if (now - self._last_publish) >= self.planner_dt:
            if self._current_fields is not None:
                self.publisher.publish_fields(self._current_fields)
            self._last_publish = now
            self._last_update = now
            return StreamTick(published=self._current)

        return StreamTick()

    def hold_for(self, duration_s: float, *, background_stream: bool = False) -> None:
        """Republish the current command at ``planner_dt`` for ``duration_s`` seconds.

        Matches the deploy kinematic planner input rate (~10 Hz). When a background
        stream thread is already ticking, this only sleeps for the dwell period.
        """
        if duration_s <= 0 or self._current is None or isinstance(self._current, StopCommand):
            return
        deadline = time.monotonic() + float(duration_s)
        while time.monotonic() < deadline:
            if not background_stream:
                self.tick()
            time.sleep(self.planner_dt)

    def interrupt(self) -> None:
        """Clear the active hold and send IDLE so deploy cuts motion before the next tool call."""
        self._current = None
        self._current_fields = None
        self._timed_out = False
        self._last_update = time.monotonic()
        self.publisher.interrupt()

    def close(self) -> None:
        self.publisher.close()
