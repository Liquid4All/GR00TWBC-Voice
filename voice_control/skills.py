from __future__ import annotations

import math
from enum import Enum, IntEnum
from typing import Dict, List, Tuple


class _StrEnum(str, Enum):
    pass


class NavStyle(_StrEnum):
    WALKING = "walking"
    RUNNING = "running"
    HAPPY = "happy"
    STEALTH = "stealth"
    INJURED = "injured"
    DRUNKEN = "drunken"


class CrawlStyle(_StrEnum):
    ELBOW_KNEE = "elbow_knee"
    HAND_CRAWL = "hand_crawl"


class Posture(_StrEnum):
    SQUAT = "squat"
    KNEEL_ONE_LEG = "kneel_one_leg"
    KNEEL_TWO_LEGS = "kneel_two_legs"
    STAND = "stand"


class BoxingAction(_StrEnum):
    IDLE = "idle"
    STANCE = "stance"
    BLOCK = "block"
    LEFT_JAB = "left_jab"
    RIGHT_JAB = "right_jab"
    LEFT_HOOK = "left_hook"
    RIGHT_HOOK = "right_hook"
    SIDE_STEP = "side_step"


class StopReason(_StrEnum):
    USER_REQUEST = "user_request"
    SAFETY = "safety"
    UNKNOWN = "unknown"


class LocomotionMode(IntEnum):
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


def heading_to_direction(heading_deg: float) -> Tuple[float, float, float]:
    """Body-relative unit vector matching deploy gamepad/keyboard (cos θ, sin θ, 0).

    Heading convention (LFM / training): 0° forward, +90° left, −90° / 270° right.
    """
    rad = math.radians(heading_deg)
    x, y = math.cos(rad), math.sin(rad)
    return (0.0 if abs(x) < 1e-9 else x, 0.0 if abs(y) < 1e-9 else y, 0.0)


def normalize_heading_deg(heading_deg: float) -> float:
    h = math.fmod(heading_deg, 360.0)
    if h > 180.0:
        h -= 360.0
    elif h < -180.0:
        h += 360.0
    return 180.0 if h == -180.0 else h


STOP_WORDS: List[str] = [
    "stop", "halt", "freeze", "cancel", "abort", "emergency stop", "emergency",
    "e stop", "estop", "whoa",
]

DIRECTION_TO_HEADING: Dict[str, float] = {
    "forward": 0.0, "forwards": 0.0, "ahead": 0.0, "straight": 0.0,
    "backward": 180.0, "backwards": 180.0, "back": 180.0, "reverse": 180.0,
    "left": -90.0, "leftward": -90.0, "leftwards": -90.0,
    "right": 90.0, "rightward": 90.0, "rightwards": 90.0,
}

STYLE_WORDS: Dict[str, NavStyle] = {
    "happy": NavStyle.HAPPY, "happily": NavStyle.HAPPY,
    "stealth": NavStyle.STEALTH, "stealthy": NavStyle.STEALTH, "stealthily": NavStyle.STEALTH,
    "sneak": NavStyle.STEALTH, "sneaky": NavStyle.STEALTH,
    "injured": NavStyle.INJURED, "limp": NavStyle.INJURED, "limping": NavStyle.INJURED, "hurt": NavStyle.INJURED,
    "drunken": NavStyle.DRUNKEN, "drunk": NavStyle.DRUNKEN,
}

WALK_VERBS = ["walk", "move", "go", "step", "head", "proceed", "advance"]
RUN_WORDS = ["run", "running", "jog", "jogging"]
SPRINT_WORDS = ["sprint", "sprinting", "dash"]
SLOW_WORDS = ["slow", "slowly", "carefully", "gently"]
FAST_WORDS = ["fast", "quickly", "quick", "faster", "rapidly"]

DEFAULT_NAV_VELOCITY = 0.6
SLOW_NAV_VELOCITY = 0.3
FAST_NAV_VELOCITY = 1.0
RUN_VELOCITY = 1.5
SPRINT_VELOCITY = 2.0
SIDE_NAV_VELOCITY = 0.4
BACKWARD_NAV_VELOCITY = 0.4
DEFAULT_CRAWL_VELOCITY = 0.25
HAND_CRAWL_VELOCITY = 0.20
SQUAT_HEIGHT = 0.5
SQUAT_LOWER_HEIGHT = 0.35
SQUAT_HIGHER_HEIGHT = 0.65
KNEEL_HEIGHT = 0.45
