from __future__ import annotations

import abc
import json
import logging
import struct
import time
from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Sequence, Tuple

from .config import PublisherConfig
from .schemas import (
    GetUpCommand,
    PlannerToolCallType,
    SetBoxingActionCommand,
    SetCrawlCommand,
    SetNavigationCommand,
    SetPostureCommand,
    StopCommand,
    tool_call_to_dict,
)
from . import skills
from .skills import LocomotionMode, heading_to_direction

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


def tool_call_to_planner_fields(command: PlannerToolCallType) -> PlannerFields:

    if isinstance(command, StopCommand):
        return PlannerFields(
            mode=int(LocomotionMode.IDLE), movement=(0.0, 0.0, 0.0),
            facing=_FORWARD, speed=_SPEED_DEFAULT, height=_HEIGHT_DEFAULT, is_stop=True,
        )

    if isinstance(command, SetNavigationCommand):
        mode = skills.STYLE_TO_MODE.get(command.style, LocomotionMode.WALK)
        if command.style == "drunken":
            log.warning("Style 'drunken' has no dedicated planner mode; using WALK.")
        direction = heading_to_direction(command.heading_deg)
        if command.velocity_mps <= 0.0:
            # In-place turn: change facing only, no translation.
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
        direction = heading_to_direction(command.heading_deg)
        speed = float(command.velocity_mps) if command.velocity_mps > 0.0 else _SPEED_DEFAULT
        movement = direction if command.velocity_mps > 0.0 else (0.0, 0.0, 0.0)
        return PlannerFields(mode=int(mode), movement=movement, facing=direction, speed=speed)

    if isinstance(command, SetPostureCommand):
        mode = skills.POSTURE_TO_MODE.get(command.posture, LocomotionMode.IDLE)
        height = command.pelvis_height_m if command.pelvis_height_m is not None else _HEIGHT_DEFAULT
        return PlannerFields(
            mode=int(mode), movement=(0.0, 0.0, 0.0), facing=_FORWARD,
            speed=_SPEED_DEFAULT, height=float(height),
        )

    if isinstance(command, SetBoxingActionCommand):
        mode = skills.BOXING_ACTION_TO_MODE.get(command.action, LocomotionMode.IDLE_BOXING)
        if command.action in ("block",):
            log.warning("Boxing action 'block' has no dedicated planner mode; using IDLE_BOXING.")
        return PlannerFields(
            mode=int(mode), movement=(0.0, 0.0, 0.0), facing=_FORWARD, speed=_SPEED_DEFAULT,
        )

    if isinstance(command, GetUpCommand):
        # Recover to a standing idle pose; the planner fills in the transition.
        return PlannerFields(
            mode=int(LocomotionMode.IDLE), movement=(0.0, 0.0, 0.0), facing=_FORWARD,
            speed=_SPEED_DEFAULT, height=_HEIGHT_DEFAULT,
        )

    # clarify and anything else: no motion.
    return PlannerFields(
        mode=int(LocomotionMode.IDLE), movement=(0.0, 0.0, 0.0), facing=_FORWARD,
        speed=_SPEED_DEFAULT, height=_HEIGHT_DEFAULT,
    )


# Publisher abstraction + adapters

class PlannerCommandPublisher(abc.ABC):

    @abc.abstractmethod
    def publish(self, command: PlannerToolCallType) -> None:
        ...

    @abc.abstractmethod
    def stop(self) -> None:
        ...

    def close(self) -> None:  # default no-op
        return None


class StubPublisher(PlannerCommandPublisher):

    def __init__(self) -> None:
        self.published: List[dict] = []

    def publish(self, command: PlannerToolCallType) -> None:
        fields = tool_call_to_planner_fields(command)
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

    def stop(self) -> None:
        self.publish(StopCommand())

    def close(self) -> None:
        log.debug("[StubPublisher] closed (%d commands published)", len(self.published))


class ZmqPublisher(PlannerCommandPublisher):

    def __init__(self, endpoint: str, bind: bool = True) -> None:
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
        fields = tool_call_to_planner_fields(command)
        if fields.is_stop:
            self.stop()
            return
        msg = fields.to_planner_wire()
        self._socket.send(msg)
        log.info("[ZmqPublisher] planner <- %s", fields.to_movement_state())

    def stop(self) -> None:
        if self._socket is None:
            return
        self._socket.send(self._build_command_message(start=False, stop=True, planner=True))
        log.info("[ZmqPublisher] STOP sent on command topic")

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
    _last_update: float = field(default=0.0, init=False)
    _last_publish: float = field(default=0.0, init=False)
    _timed_out: bool = field(default=False, init=False)

    def set_command(self, command: PlannerToolCallType, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        self._current = command
        self._last_update = now
        self._timed_out = False
        # Immediate replan on change.
        self.publisher.publish(command)
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
            self.publisher.publish(self._current)
            self._last_publish = now
            return StreamTick(published=self._current)

        return StreamTick()

    def close(self) -> None:
        self.publisher.close()
