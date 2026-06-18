"""Planner command publishers.

A :class:`PlannerCommandPublisher` turns a *validated, safety-clamped* tool call
into a high-level planner command and delivers it to the robot. It never emits
joint targets or motion frames -- only the high-level fields the kinematic
planner consumes (mode, movement direction, facing direction, speed, height).

Adapters
--------
* :class:`StubPublisher`        -- logs only (default / dry-run / tests).
* :class:`ZmqPublisher`         -- the repo's real path: publishes ``command``
                                   and ``planner`` ZMQ topics using the existing
                                   wire builders in ``gear_sonic``.
* :class:`ExistingRepoPublisher`-- thin alias of the ZMQ path with the single
                                   integration point documented; raises a precise
                                   error if the repo builders are unavailable.
* :class:`Ros2Publisher`        -- optional ROS2 adapter (NOT the discovered
                                   path; the deploy stack uses ZMQ).

The :class:`PlannerStreamLoop` wraps any publisher to provide velocity-conditioned
holding (re-publish the same command at up to 10 Hz, never integrating speed),
immediate replan on change, and a command-timeout watchdog.
"""

from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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

_FORWARD = (1.0, 0.0, 0.0)
_SPEED_DEFAULT = -1.0
_HEIGHT_DEFAULT = -1.0


@dataclass
class PlannerFields:
    """Low-level planner-topic fields derived from a tool call."""

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


def tool_call_to_planner_fields(command: PlannerToolCallType) -> PlannerFields:
    """Map a validated tool call onto planner-topic fields.

    Only high-level fields are produced; the planner (not us) turns these into
    motion.
    """

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


# --------------------------------------------------------------------------- #
# Publisher abstraction + adapters
# --------------------------------------------------------------------------- #

class PlannerCommandPublisher(abc.ABC):
    """Abstract high-level planner command sink."""

    @abc.abstractmethod
    def publish(self, command: PlannerToolCallType) -> None:
        ...

    @abc.abstractmethod
    def stop(self) -> None:
        ...

    def close(self) -> None:  # default no-op
        return None


class StubPublisher(PlannerCommandPublisher):
    """Logs the final planner command. Used in dry-run and tests."""

    def __init__(self) -> None:
        self.published: List[dict] = []

    def publish(self, command: PlannerToolCallType) -> None:
        fields = tool_call_to_planner_fields(command)
        record = {"tool_call": tool_call_to_dict(command), "planner_fields": fields.as_dict()}
        self.published.append(record)
        log.info("[StubPublisher] %s -> %s", record["tool_call"], record["planner_fields"])

    def stop(self) -> None:
        self.publish(StopCommand())

    def close(self) -> None:
        log.debug("[StubPublisher] closed (%d commands published)", len(self.published))


class ZmqPublisher(PlannerCommandPublisher):
    """Real planner path: publishes the repo's ZMQ ``command``/``planner`` topics.

    This mirrors ``gear_sonic/scripts/pico_manager_thread_server.py``'s
    ``PlannerStreamer``: a ``zmq.PUB`` socket sends a ``command`` message
    (start/stop/planner) and ``planner`` messages (mode/movement/facing/
    speed/height) built by ``gear_sonic.utils.teleop.zmq.zmq_planner_sender``.
    """

    def __init__(self, endpoint: str, bind: bool = True) -> None:
        self.endpoint = endpoint
        self._bind = bind
        self._socket = None
        self._context = None
        self._build_command_message = None
        self._build_planner_message = None
        self._started = False
        self._connect()

    def _connect(self) -> None:
        try:
            import zmq  # type: ignore
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "pyzmq is required for the ZMQ publisher. Install with `pip install pyzmq`."
            ) from exc

        from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (  # type: ignore
            build_command_message,
            build_planner_message,
        )

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
        msg = self._build_planner_message(
            mode=fields.mode,
            movement=list(fields.movement),
            facing=list(fields.facing),
            speed=fields.speed,
            height=fields.height,
        )
        self._socket.send(msg)
        log.info("[ZmqPublisher] planner <- %s", fields.as_dict())

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


class ExistingRepoPublisher(ZmqPublisher):
    """Wires voice commands into the discovered SONIC planner command path.

    The discovered path is the ZMQ ``command``/``planner`` topic protocol used by
    the C++ ``ZMQManager`` deploy interface and the Python ``PlannerStreamer``.
    This subclass is the single, clearly-marked integration point. If the repo's
    wire builders cannot be imported, a precise error explains exactly what to
    connect.
    """

    def _connect(self) -> None:  # pragma: no cover - requires gear_sonic + pyzmq
        try:
            super()._connect()
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "ExistingRepoPublisher could not connect to the SONIC planner path.\n"
                "INTEGRATION POINT: this publisher expects to send the ZMQ 'command' and "
                "'planner' topics consumed by the C++ ZMQManager "
                "(gear_sonic_deploy/.../input_interface/zmq_manager.hpp).\n"
                "It uses build_command_message / build_planner_message from "
                "gear_sonic.utils.teleop.zmq.zmq_planner_sender.\n"
                f"Underlying import error: {exc}\n"
                "To fix: `pip install -e gear_sonic[teleop]` (provides pyzmq + the builders), "
                "set publisher.zmq_endpoint to the deploy ZMQManager host:port, then re-run "
                "with --execute and config safety.execute: true."
            ) from exc


class Ros2Publisher(PlannerCommandPublisher):
    """Optional ROS2 adapter.

    NOTE: the deployed SONIC stack consumes ZMQ, not ROS2, for planner commands.
    This adapter is provided for environments that bridge planner commands over a
    ROS2 topic. It publishes a ``geometry_msgs/Twist``-style velocity hint plus
    the mode as a separate field; adapt the message type to your bridge.
    """

    def __init__(self, topic: str) -> None:
        self.topic = topic
        self._node = None
        self._pub = None
        self._rclpy = None
        self._connect()

    def _connect(self) -> None:  # pragma: no cover - requires ROS2
        try:
            import rclpy  # type: ignore
            from geometry_msgs.msg import Twist  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Ros2Publisher requires a ROS2 (rclpy) environment. NOTE: the SONIC deploy "
                "stack uses ZMQ for planner commands; prefer ExistingRepoPublisher unless you "
                "have a ROS2 bridge. Underlying import error: " + str(exc)
            ) from exc
        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node("sonic_voice_control")
        self._pub = self._node.create_publisher(Twist, self.topic, 10)
        log.info("[Ros2Publisher] publishing on %s", self.topic)

    def publish(self, command: PlannerToolCallType) -> None:  # pragma: no cover - requires ROS2
        from geometry_msgs.msg import Twist  # type: ignore

        fields = tool_call_to_planner_fields(command)
        msg = Twist()
        speed = fields.speed if fields.speed > 0 else 0.0
        msg.linear.x = float(fields.movement[0]) * speed
        msg.linear.y = float(fields.movement[1]) * speed
        msg.angular.z = 0.0
        self._pub.publish(msg)
        log.info("[Ros2Publisher] %s (mode=%d)", fields.as_dict(), fields.mode)

    def stop(self) -> None:  # pragma: no cover - requires ROS2
        from geometry_msgs.msg import Twist  # type: ignore

        self._pub.publish(Twist())

    def close(self) -> None:  # pragma: no cover - requires ROS2
        if self._node is not None:
            self._node.destroy_node()


def build_publisher(cfg: PublisherConfig) -> PlannerCommandPublisher:
    """Factory selecting a publisher adapter from config."""

    backend = cfg.backend.lower()
    if backend == "stub":
        return StubPublisher()
    if backend == "zmq":
        return ZmqPublisher(cfg.zmq_endpoint)
    if backend in ("existing_repo", "existing"):
        return ExistingRepoPublisher(cfg.zmq_endpoint)
    if backend == "ros2":
        return Ros2Publisher(cfg.ros2_topic)
    raise ValueError(f"Unknown publisher backend: {cfg.backend!r}")


# --------------------------------------------------------------------------- #
# Stream loop: 10 Hz hold + watchdog
# --------------------------------------------------------------------------- #

@dataclass
class StreamTick:
    published: Optional[PlannerToolCallType] = None
    timed_out: bool = False
    reason: str = ""


@dataclass
class PlannerStreamLoop:
    """Velocity-conditioned holding + replan-on-change + command-timeout watchdog.

    Call :meth:`set_command` when a new command is accepted, and :meth:`tick`
    every loop iteration (the runtime drives this at <= 10 Hz). Holding a command
    re-publishes the *same* command (no speed integration). If no new command
    arrives within ``command_timeout_s``, the loop publishes a single safety stop.
    """

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
