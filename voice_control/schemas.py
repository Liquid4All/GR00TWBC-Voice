"""Closed tool-call schema for the SONIC voice control interface.

Every spoken command must be reduced to exactly one of the validated Pydantic
models defined here. Nothing else is ever published to the kinematic planner.

The schema is intentionally small and *closed*: ``extra="forbid"`` means any
stray field (e.g. from a hallucinating LLM) is rejected at validation time.

Coordinate / unit conventions (matching the planner ONNX interface and the
repo's ZMQ ``planner`` topic):

* ``velocity_mps``  -- target locomotion speed in metres / second.
* ``heading_deg``   -- desired heading, degrees. 0 = forward, +90 = right,
                       -90 = left, 180 = backward (see ``skills.heading_to_direction``).
* ``pelvis_height_m`` -- target pelvis/root height in metres.
* ``duration_s``    -- optional hold duration; ``None`` means "hold until changed".
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class _StrEnum(str, Enum):
    """str-backed enum so values serialise to plain strings in JSON."""

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


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


class _ToolBase(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class StopCommand(_ToolBase):
    tool: Literal["stop"] = "stop"
    reason: StopReason = StopReason.USER_REQUEST


class SetNavigationCommand(_ToolBase):
    tool: Literal["set_navigation"] = "set_navigation"
    # NOTE: safety clamps have been removed -- velocity is passed through to the
    # planner unmodified. Only non-negativity is enforced (direction is encoded
    # by heading, not the sign of the speed).
    velocity_mps: float = Field(ge=0.0)
    heading_deg: float
    style: NavStyle = NavStyle.WALKING
    duration_s: Optional[float] = Field(default=None, ge=0.0)


class SetCrawlCommand(_ToolBase):
    tool: Literal["set_crawl"] = "set_crawl"
    velocity_mps: float = Field(ge=0.0)
    heading_deg: float
    crawl_style: CrawlStyle = CrawlStyle.ELBOW_KNEE
    duration_s: Optional[float] = Field(default=None, ge=0.0)


class SetPostureCommand(_ToolBase):
    tool: Literal["set_posture"] = "set_posture"
    posture: Posture
    pelvis_height_m: Optional[float] = Field(default=None, ge=0.0)
    duration_s: Optional[float] = Field(default=None, ge=0.0)


class SetBoxingActionCommand(_ToolBase):
    tool: Literal["set_boxing_action"] = "set_boxing_action"
    action: BoxingAction
    duration_s: Optional[float] = Field(default=None, ge=0.0)


class GetUpCommand(_ToolBase):
    tool: Literal["get_up"] = "get_up"


class ClarifyCommand(_ToolBase):
    tool: Literal["clarify"] = "clarify"
    question: str
    original_text: str


# Discriminated union over the closed tool set. The ``tool`` literal is the
# discriminator, so validation is fast and unambiguous.
PlannerToolCall = Annotated[
    Union[
        StopCommand,
        SetNavigationCommand,
        SetCrawlCommand,
        SetPostureCommand,
        SetBoxingActionCommand,
        GetUpCommand,
        ClarifyCommand,
    ],
    Field(discriminator="tool"),
]

# Reusable validator for dict/JSON payloads (e.g. LLM output).
PLANNER_TOOL_CALL_ADAPTER: TypeAdapter = TypeAdapter(PlannerToolCall)

#: Tools that command physical motion (used by safety / dry-run gating).
MOTION_TOOLS = {
    "set_navigation",
    "set_crawl",
    "set_posture",
    "set_boxing_action",
    "get_up",
}


def validate_tool_call(data: object) -> "PlannerToolCallType":
    """Validate an arbitrary dict / JSON-like object into a tool call.

    Raises ``pydantic.ValidationError`` if the payload does not match the
    closed schema (this is how malformed LLM output is rejected).
    """

    return PLANNER_TOOL_CALL_ADAPTER.validate_python(data)


def tool_call_to_dict(tool_call: BaseModel) -> dict:
    """Serialise a validated tool call to a plain JSON-able dict."""

    return tool_call.model_dump(mode="json")


# Convenience type alias used in annotations across the package.
PlannerToolCallType = Union[
    StopCommand,
    SetNavigationCommand,
    SetCrawlCommand,
    SetPostureCommand,
    SetBoxingActionCommand,
    GetUpCommand,
    ClarifyCommand,
]


class ParseResult(BaseModel):
    """Result returned by every parser (deterministic or LLM)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ok: bool
    confidence: float = Field(ge=0.0, le=1.0)
    raw_text: str
    normalized_text: str
    command: Optional[PlannerToolCallType] = None
    reason: Optional[str] = None

    def is_clarify(self) -> bool:
        return self.command is None or getattr(self.command, "tool", None) == "clarify"
