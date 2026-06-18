"""Skill inventory, synonym dictionaries and planner-mode mappings.

This module is the single source of truth for:

* which spoken phrases map to which concept (synonyms),
* how a validated tool call maps onto the planner's ``LocomotionMode`` integer
  and movement / facing / speed / height fields,
* the heading-degree -> world-frame direction-vector convention.

The ``LocomotionMode`` indices follow the deployed planner ONNX V2 model as
documented in ``docs/source/references/planner_onnx.md`` (27 modes). They are
intentionally kept here (rather than imported from the deploy code) so this
package has no hard dependency on the heavy runtime; the publisher uses these
ints directly on the ZMQ ``planner`` topic, which the C++ ``ZMQManager`` casts
back to ``LocomotionMode``.
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import Dict, List, Tuple

from .schemas import (
    BoxingAction,
    CrawlStyle,
    NavStyle,
    Posture,
)


class LocomotionMode(IntEnum):
    """Planner ONNX V2 mode indices (see planner_onnx.md)."""

    IDLE = 0
    SLOW_WALK = 1
    WALK = 2
    RUN = 3
    SQUAT = 4
    KNEEL_TWO_LEG = 5
    KNEEL_ONE_LEG = 6
    LYING_FACEDOWN = 7
    HAND_CRAWLING = 8
    IDLE_BOXING = 9
    WALK_BOXING = 10
    LEFT_JAB = 11
    RIGHT_JAB = 12
    RANDOM_PUNCHES = 13
    ELBOW_CRAWLING = 14
    LEFT_HOOK = 15
    RIGHT_HOOK = 16
    HAPPY = 17
    STEALTH = 18
    INJURED = 19
    CAREFUL = 20
    OBJECT_CARRYING = 21
    CROUCH = 22
    HAPPY_DANCE = 23
    ZOMBIE = 24
    POINT = 25
    SCARED = 26


# --------------------------------------------------------------------------- #
# Tool-call -> planner mode mappings
# --------------------------------------------------------------------------- #

#: Navigation style -> planner mode. ``drunken`` has no dedicated planner mode
#: in the V2 model, so it falls back to plain WALK (logged by the publisher).
STYLE_TO_MODE: Dict[str, LocomotionMode] = {
    NavStyle.WALKING.value: LocomotionMode.WALK,
    NavStyle.RUNNING.value: LocomotionMode.RUN,
    NavStyle.HAPPY.value: LocomotionMode.HAPPY,
    NavStyle.STEALTH.value: LocomotionMode.STEALTH,
    NavStyle.INJURED.value: LocomotionMode.INJURED,
    NavStyle.DRUNKEN.value: LocomotionMode.WALK,
}

CRAWL_STYLE_TO_MODE: Dict[str, LocomotionMode] = {
    CrawlStyle.ELBOW_KNEE.value: LocomotionMode.ELBOW_CRAWLING,
    CrawlStyle.HAND_CRAWL.value: LocomotionMode.HAND_CRAWLING,
}

POSTURE_TO_MODE: Dict[str, LocomotionMode] = {
    Posture.SQUAT.value: LocomotionMode.SQUAT,
    Posture.KNEEL_ONE_LEG.value: LocomotionMode.KNEEL_ONE_LEG,
    Posture.KNEEL_TWO_LEGS.value: LocomotionMode.KNEEL_TWO_LEG,
    Posture.STAND.value: LocomotionMode.IDLE,
}

#: Boxing action -> planner mode. ``block`` has no dedicated mode (mapped to the
#: boxing idle/guard stance); ``side_step`` maps to the walking-boxing mode.
BOXING_ACTION_TO_MODE: Dict[str, LocomotionMode] = {
    BoxingAction.IDLE.value: LocomotionMode.IDLE_BOXING,
    BoxingAction.STANCE.value: LocomotionMode.IDLE_BOXING,
    BoxingAction.BLOCK.value: LocomotionMode.IDLE_BOXING,
    BoxingAction.LEFT_JAB.value: LocomotionMode.LEFT_JAB,
    BoxingAction.RIGHT_JAB.value: LocomotionMode.RIGHT_JAB,
    BoxingAction.LEFT_HOOK.value: LocomotionMode.LEFT_HOOK,
    BoxingAction.RIGHT_HOOK.value: LocomotionMode.RIGHT_HOOK,
    BoxingAction.SIDE_STEP.value: LocomotionMode.WALK_BOXING,
}

#: Modes that are "static" (the planner ignores movement speed for these); used
#: for documentation / sanity, mirrors ``is_static_motion_mode`` in the C++.
STATIC_MODES = {
    LocomotionMode.IDLE,
    LocomotionMode.SQUAT,
    LocomotionMode.KNEEL_ONE_LEG,
    LocomotionMode.KNEEL_TWO_LEG,
    LocomotionMode.LYING_FACEDOWN,
    LocomotionMode.IDLE_BOXING,
}


# --------------------------------------------------------------------------- #
# Heading convention
# --------------------------------------------------------------------------- #

def heading_to_direction(heading_deg: float) -> Tuple[float, float, float]:
    """Convert a heading in degrees to a world-frame unit direction vector.

    Convention (chosen to satisfy *both* the voice command table and the repo's
    Z-up, X-forward, Y-left frame):

    * ``0``   -> forward  -> ``( 1,  0, 0)``
    * ``-90`` -> left     -> ``( 0,  1, 0)``  (robot +Y is left)
    * ``+90`` -> right    -> ``( 0, -1, 0)``
    * ``180`` -> backward -> ``(-1,  0, 0)``

    i.e. heading is measured clockwise-positive (compass-like), and we map it to
    the repo's counter-clockwise-positive frame via ``angle = -heading``.
    """

    rad = math.radians(heading_deg)
    x = math.cos(rad)
    y = -math.sin(rad)
    # Guard against tiny floating point dust so vectors look clean.
    if abs(x) < 1e-9:
        x = 0.0
    if abs(y) < 1e-9:
        y = 0.0
    return (x, y, 0.0)


def normalize_heading_deg(heading_deg: float) -> float:
    """Normalise a heading to the ``[-180, 180]`` convention used in the repo."""

    h = math.fmod(heading_deg, 360.0)
    if h > 180.0:
        h -= 360.0
    elif h < -180.0:
        h += 360.0
    # -180 and 180 are equivalent; keep 180 for "backward" readability.
    if h == -180.0:
        h = 180.0
    return h


# --------------------------------------------------------------------------- #
# Synonym dictionaries (used by the deterministic parser)
# --------------------------------------------------------------------------- #

#: Words that immediately trigger a stop, even without a wake word.
STOP_WORDS: List[str] = [
    "stop",
    "halt",
    "freeze",
    "cancel",
    "abort",
    "emergency stop",
    "emergency",
    "e stop",
    "estop",
    "whoa",
]

#: Direction phrase -> heading degrees (clockwise-positive; see convention).
DIRECTION_TO_HEADING: Dict[str, float] = {
    "forward": 0.0,
    "forwards": 0.0,
    "ahead": 0.0,
    "straight": 0.0,
    "backward": 180.0,
    "backwards": 180.0,
    "back": 180.0,
    "reverse": 180.0,
    "left": -90.0,
    "leftward": -90.0,
    "leftwards": -90.0,
    "right": 90.0,
    "rightward": 90.0,
    "rightwards": 90.0,
}

#: Style trigger words -> NavStyle.
STYLE_WORDS: Dict[str, NavStyle] = {
    "happy": NavStyle.HAPPY,
    "happily": NavStyle.HAPPY,
    "stealth": NavStyle.STEALTH,
    "stealthy": NavStyle.STEALTH,
    "stealthily": NavStyle.STEALTH,
    "sneak": NavStyle.STEALTH,
    "sneaky": NavStyle.STEALTH,
    "injured": NavStyle.INJURED,
    "limp": NavStyle.INJURED,
    "limping": NavStyle.INJURED,
    "hurt": NavStyle.INJURED,
    "drunken": NavStyle.DRUNKEN,
    "drunk": NavStyle.DRUNKEN,
}

#: Locomotion verbs that indicate a navigation command.
WALK_VERBS = ["walk", "move", "go", "step", "head", "proceed", "advance"]
RUN_WORDS = ["run", "running", "jog", "jogging"]
SPRINT_WORDS = ["sprint", "sprinting", "dash"]

#: Speed modifiers -> velocity (m/s) for navigation.
SLOW_WORDS = ["slow", "slowly", "carefully", "gently"]
FAST_WORDS = ["fast", "quickly", "quick", "faster", "rapidly"]

# Default nominal velocities (subject to safety clamping downstream).
DEFAULT_NAV_VELOCITY = 0.6
SLOW_NAV_VELOCITY = 0.3
FAST_NAV_VELOCITY = 1.0
RUN_VELOCITY = 1.5
SPRINT_VELOCITY = 2.0
SIDE_NAV_VELOCITY = 0.4
BACKWARD_NAV_VELOCITY = 0.4

DEFAULT_CRAWL_VELOCITY = 0.25
HAND_CRAWL_VELOCITY = 0.20

# Default pelvis heights for postures (metres, before safety clamp).
SQUAT_HEIGHT = 0.5
SQUAT_LOWER_HEIGHT = 0.35
SQUAT_HIGHER_HEIGHT = 0.65
KNEEL_HEIGHT = 0.45


#: Human-readable inventory of everything the voice interface supports.
SUPPORTED_SKILLS: Dict[str, List[str]] = {
    "navigation": [
        "forward", "backward", "left", "right",
        "turn left", "turn right", "arbitrary heading (degrees)",
        "walking", "running / sprinting",
        "happy walking", "stealth walking", "injured walking", "drunken walking",
    ],
    "posture": [
        "squat", "squat lower", "squat higher",
        "kneel", "kneel on one knee", "kneel on both knees",
        "stand up", "get up",
    ],
    "crawling": [
        "crawl forward", "crawl backward", "crawl left", "crawl right",
        "hand crawl", "elbow/knee crawl",
    ],
    "boxing": [
        "stance", "idle", "block",
        "left jab", "right jab", "left hook", "right hook", "side step",
    ],
}
