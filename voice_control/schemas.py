"""Closed Pydantic tool-call schema for the SONIC voice interface."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class _StrEnum(str, Enum):
    def __str__(self) -> str:
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


PlannerToolCall = Annotated[
    Union[
        StopCommand, SetNavigationCommand, SetCrawlCommand, SetPostureCommand,
        SetBoxingActionCommand, GetUpCommand, ClarifyCommand,
    ],
    Field(discriminator="tool"),
]

PLANNER_TOOL_CALL_ADAPTER: TypeAdapter = TypeAdapter(PlannerToolCall)

MOTION_TOOLS = {"set_navigation", "set_crawl", "set_posture", "set_boxing_action", "get_up"}


def validate_tool_call(data: object) -> "PlannerToolCallType":
    return PLANNER_TOOL_CALL_ADAPTER.validate_python(data)


def tool_call_to_dict(tool_call: BaseModel) -> dict:
    return tool_call.model_dump(mode="json")


PlannerToolCallType = Union[
    StopCommand, SetNavigationCommand, SetCrawlCommand, SetPostureCommand,
    SetBoxingActionCommand, GetUpCommand, ClarifyCommand,
]


class ParseResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    ok: bool
    confidence: float = Field(ge=0.0, le=1.0)
    raw_text: str
    normalized_text: str
    command: Optional[PlannerToolCallType] = None
    reason: Optional[str] = None

    def is_clarify(self) -> bool:
        return self.command is None or getattr(self.command, "tool", None) == "clarify"
