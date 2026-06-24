"""Deterministic and model-based command parsers."""

from __future__ import annotations

import logging
import re
from typing import Annotated, Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from . import skills
from .config import ParserConfig
from .skills import BoxingAction, CrawlStyle, NavStyle, Posture, StopReason


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


class RotateInPlaceCommand(_ToolBase):
    """Relative in-place turn (+angle = left, −angle = right)."""

    tool: Literal["rotate_in_place"] = "rotate_in_place"
    angle_deg: float
    yaw_rate_dps: float = 90.0
    style: NavStyle = NavStyle.WALKING
    duration_s: Optional[float] = Field(default=None, ge=0.0)


class HoldPoseCommand(_ToolBase):
    tool: Literal["hold_pose"] = "hold_pose"
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


PlannerToolCallType = Union[
    StopCommand, SetNavigationCommand, RotateInPlaceCommand, HoldPoseCommand,
    SetCrawlCommand, SetPostureCommand, SetBoxingActionCommand, GetUpCommand,
    ClarifyCommand,
]
PlannerToolCall = Annotated[PlannerToolCallType, Field(discriminator="tool")]
_TOOL_ADAPTER: TypeAdapter = TypeAdapter(PlannerToolCall)


def validate_tool_call(data: object) -> PlannerToolCallType:
    return _TOOL_ADAPTER.validate_python(data)


def tool_call_to_dict(tool_call: BaseModel) -> dict:
    return tool_call.model_dump(mode="json")


class ParseResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    ok: bool
    confidence: float = Field(ge=0.0, le=1.0)
    raw_text: str
    normalized_text: str
    command: Optional[PlannerToolCallType] = None
    reason: Optional[str] = None


log = logging.getLogger(__name__)

# Confidence levels.
CONF_EXACT = 0.99
CONF_STRONG = 0.92
CONF_OK = 0.85
CONF_WEAK = 0.6
CONF_NONE = 0.0

# Connectors that separate composed/sequential commands. Longest phrases first
# so "and then" wins over "and"/"then".
_CONNECTOR_RE = re.compile(
    r"\s*(?:,|\band then\b|\bafter that\b|\bfollowed by\b|\bthen\b|\bnext\b|\band\b)\s*"
)

_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}


# Normalisation

def split_segments(text: str) -> List[str]:
    """Split a (possibly compound) utterance into ordered sub-commands.

    Splits on natural connectors ("and then", "then", "and", ",", "after that",
    "followed by", "next"). A single command yields a one-element list. Empty
    fragments are dropped.

    Example::

        "walk forward at velocity 5 meters per second and then turn around and
         kneel on one leg"
        -> ["walk forward at velocity 5 meters per second",
            "turn around",
            "kneel on one leg"]
    """

    if not text or not text.strip():
        return []
    parts = _CONNECTOR_RE.split(text.strip())
    return [p.strip() for p in parts if p and p.strip()]


def normalize_text(text: str) -> str:

    t = text.lower().strip()
    # Keep digits, letters, decimal points, minus signs and percent.
    t = re.sub(r"[^a-z0-9.\-\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = _words_to_numbers(t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _words_to_numbers(text: str) -> str:

    tokens = text.split()
    out: List[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in _NUMBER_WORDS:
            value, consumed = _consume_number(tokens, i)
            out.append(_format_number(value))
            i += consumed
            continue
        out.append(tok)
        i += 1
    return " ".join(out)


def _consume_number(tokens: List[str], start: int) -> Tuple[float, int]:

    i = start
    integer_part = 0
    have_int = False
    while i < len(tokens) and tokens[i] in _NUMBER_WORDS:
        word = tokens[i]
        val = _NUMBER_WORDS[word]
        if word == "hundred":
            integer_part = (integer_part or 1) * 100
        else:
            integer_part += val
        have_int = True
        i += 1
    value = float(integer_part) if have_int else 0.0
    # Decimal via "point <digit> <digit> ..."
    if i < len(tokens) and tokens[i] == "point":
        i += 1
        decimals = ""
        while i < len(tokens) and tokens[i] in _NUMBER_WORDS and _NUMBER_WORDS[tokens[i]] < 10:
            decimals += str(_NUMBER_WORDS[tokens[i]])
            i += 1
        if decimals:
            value = float(f"{int(value)}.{decimals}")
    return value, (i - start)


def _format_number(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return str(value)


# Numeric extraction

_VELOCITY_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(?:m/s|mps|meters? per second|metres? per second)"
)
_HEADING_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:deg|degree|degrees)")
_HEADING_KW_RE = re.compile(r"(?:heading|bearing|to)\s+(-?\d+(?:\.\d+)?)")


def _extract_velocity(text: str) -> Optional[float]:
    m = _VELOCITY_RE.search(text)
    if m:
        return float(m.group(1))
    return None


def _extract_heading(text: str) -> Optional[float]:
    m = _HEADING_RE.search(text)
    if m:
        return float(m.group(1))
    m = _HEADING_KW_RE.search(text)
    if m:
        return float(m.group(1))
    return None


def _contains_any(text: str, words) -> bool:
    return any(re.search(rf"\b{re.escape(w)}\b", text) for w in words)


def _first_direction(text: str) -> Optional[Tuple[str, float]]:
    for phrase, heading in skills.DIRECTION_TO_HEADING.items():
        if re.search(rf"\b{re.escape(phrase)}\b", text):
            return phrase, heading
    return None


def _detect_style(text: str) -> Optional[NavStyle]:
    for word, style in skills.STYLE_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", text):
            return style
    return None


# Main parser

class DeterministicParser:

    def __init__(self, confidence_threshold: float = 0.75) -> None:
        self.confidence_threshold = confidence_threshold

    def parse(self, raw_text: str, boxing_active: bool = False) -> ParseResult:
        norm = normalize_text(raw_text)

        if not norm:
            return self._clarify(raw_text, norm, "Empty transcript.")

        # 1) Stop -- highest priority, accepted even mid-sentence.
        if self._is_stop(norm):
            return ParseResult(
                ok=True, confidence=CONF_EXACT, raw_text=raw_text,
                normalized_text=norm,
                command=StopCommand(reason=StopReason.USER_REQUEST),
            )

        # 2) Get up (distinct from "stand up").
        if re.search(r"\bget up\b", norm) or re.search(r"\bgetup\b", norm) or re.search(
            r"\bstand up from\b", norm
        ):
            return self._ok(raw_text, norm, GetUpCommand(), CONF_STRONG)

        # 3) Boxing actions.
        boxing = self._parse_boxing(raw_text, norm, boxing_active)
        if boxing is not None:
            return boxing

        # 4) Crawling.
        crawl = self._parse_crawl(raw_text, norm)
        if crawl is not None:
            return crawl

        # 5) Posture (squat / kneel / stand).
        posture = self._parse_posture(raw_text, norm)
        if posture is not None:
            return posture

        # 6) Navigation.
        nav = self._parse_navigation(raw_text, norm)
        if nav is not None:
            return nav

        # 7) Fallback: unsupported / ambiguous.
        return self._clarify(
            raw_text, norm,
            "Unrecognised or unsupported command for the kinematic planner.",
        )

    def parse_plan(self, raw_text: str, boxing_active: bool = False) -> List[ParseResult]:
        """Parse a (possibly compound) utterance into an ordered list of results.

        Each connector-separated segment is parsed independently and in order, so
        "walk forward ... and then turn around and kneel on one leg" becomes a
        three-step plan. ``boxing_active`` is threaded through the sequence so an
        earlier boxing command disambiguates a later "side step".
        """

        segments = split_segments(raw_text)
        if len(segments) <= 1:
            return [self.parse(raw_text, boxing_active=boxing_active)]

        results: List[ParseResult] = []
        active = boxing_active
        for segment in segments:
            result = self.parse(segment, boxing_active=active)
            cmd = result.command
            if cmd is not None and cmd.tool == "set_boxing_action":
                active = True
            elif cmd is not None and cmd.tool in ("set_navigation", "set_crawl"):
                active = False
            results.append(result)
        return results

    # ------------------------------------------------------------------ #
    def _is_stop(self, norm: str) -> bool:
        for w in skills.STOP_WORDS:
            if re.search(rf"\b{re.escape(w)}\b", norm):
                return True
        return False

    def _parse_boxing(
        self, raw_text: str, norm: str, boxing_active: bool
    ) -> Optional[ParseResult]:
        if re.search(r"\bleft\b.*\bjab\b", norm) or re.search(r"\bjab\b.*\bleft\b", norm):
            return self._ok(raw_text, norm, SetBoxingActionCommand(action=BoxingAction.LEFT_JAB), CONF_STRONG)
        if re.search(r"\bright\b.*\bjab\b", norm) or re.search(r"\bjab\b.*\bright\b", norm):
            return self._ok(raw_text, norm, SetBoxingActionCommand(action=BoxingAction.RIGHT_JAB), CONF_STRONG)
        if re.search(r"\bleft\b.*\bhook\b", norm) or re.search(r"\bhook\b.*\bleft\b", norm):
            return self._ok(raw_text, norm, SetBoxingActionCommand(action=BoxingAction.LEFT_HOOK), CONF_STRONG)
        if re.search(r"\bright\b.*\bhook\b", norm) or re.search(r"\bhook\b.*\bright\b", norm):
            return self._ok(raw_text, norm, SetBoxingActionCommand(action=BoxingAction.RIGHT_HOOK), CONF_STRONG)
        # Bare jab/hook default to the left side is unsafe to guess -> clarify.
        if re.search(r"\bjab\b", norm) or re.search(r"\bhook\b", norm):
            return self._clarify(raw_text, norm, "Which side -- left or right?")
        if re.search(r"\bblock\b", norm):
            return self._ok(raw_text, norm, SetBoxingActionCommand(action=BoxingAction.BLOCK), CONF_STRONG)
        if re.search(r"\b(boxing stance|stance)\b", norm):
            return self._ok(raw_text, norm, SetBoxingActionCommand(action=BoxingAction.STANCE), CONF_STRONG)
        if re.search(r"\b(idle boxing|boxing idle)\b", norm):
            return self._ok(raw_text, norm, SetBoxingActionCommand(action=BoxingAction.IDLE), CONF_STRONG)
        if re.search(r"\b(side step|sidestep)\b", norm):
            if boxing_active:
                return self._ok(
                    raw_text, norm, SetBoxingActionCommand(action=BoxingAction.SIDE_STEP), CONF_OK
                )
            return self._clarify(
                raw_text, norm,
                "Side step is ambiguous outside boxing mode. Say 'boxing stance' first, "
                "or 'move left'/'move right' to strafe.",
            )
        return None

    def _parse_crawl(self, raw_text: str, norm: str) -> Optional[ParseResult]:
        if not re.search(r"\bcrawl(ing)?\b", norm):
            return None
        crawl_style = CrawlStyle.ELBOW_KNEE
        velocity = skills.DEFAULT_CRAWL_VELOCITY
        if re.search(r"\bhand\b", norm):
            crawl_style = CrawlStyle.HAND_CRAWL
            velocity = skills.HAND_CRAWL_VELOCITY
        elif re.search(r"\b(elbow|knee)\b", norm):
            crawl_style = CrawlStyle.ELBOW_KNEE

        direction = _first_direction(norm)
        heading = direction[1] if direction else 0.0
        explicit_heading = _extract_heading(norm)
        if explicit_heading is not None:
            heading = explicit_heading
        explicit_v = _extract_velocity(norm)
        if explicit_v is not None:
            velocity = explicit_v
        return self._ok(
            raw_text, norm,
            SetCrawlCommand(
                velocity_mps=velocity,
                heading_deg=skills.normalize_heading_deg(heading),
                crawl_style=crawl_style,
            ),
            CONF_STRONG,
        )

    def _parse_posture(self, raw_text: str, norm: str) -> Optional[ParseResult]:
        # Stand up (but not "get up", handled earlier).
        if re.search(r"\bstand( up)?\b", norm) and not re.search(r"\bstand up from\b", norm):
            return self._ok(raw_text, norm, SetPostureCommand(posture=Posture.STAND), CONF_STRONG)

        if re.search(r"\bsquat\b", norm) or re.search(r"\bcrouch\b", norm):
            height = skills.SQUAT_HEIGHT
            if re.search(r"\blower\b|\bdeeper\b|\bdown\b", norm):
                height = skills.SQUAT_LOWER_HEIGHT
            elif re.search(r"\bhigher\b|\bup\b|\bshallower\b", norm):
                height = skills.SQUAT_HIGHER_HEIGHT
            explicit_h = self._extract_height(norm)
            if explicit_h is not None:
                height = explicit_h
            return self._ok(
                raw_text, norm,
                SetPostureCommand(posture=Posture.SQUAT, pelvis_height_m=height),
                CONF_STRONG,
            )

        if re.search(r"\bkneel\b", norm):
            posture = Posture.KNEEL_TWO_LEGS
            # NOTE: number words are normalised to digits upstream (one -> 1).
            if re.search(r"\b(one|1|single)\b.*\bknee\b|\b(one|1|single) leg\b", norm):
                posture = Posture.KNEEL_ONE_LEG
            elif re.search(r"\b(both|two|2)\b", norm):
                posture = Posture.KNEEL_TWO_LEGS
            height = skills.KNEEL_HEIGHT
            explicit_h = self._extract_height(norm)
            if explicit_h is not None:
                height = explicit_h
            return self._ok(
                raw_text, norm,
                SetPostureCommand(posture=posture, pelvis_height_m=height),
                CONF_STRONG,
            )
        return None

    def _parse_navigation(self, raw_text: str, norm: str) -> Optional[ParseResult]:
        has_verb = _contains_any(norm, skills.WALK_VERBS)
        has_run = _contains_any(norm, skills.RUN_WORDS)
        has_sprint = _contains_any(norm, skills.SPRINT_WORDS)
        direction = _first_direction(norm)
        style_word = _detect_style(norm)
        is_turn = bool(re.search(r"\bturn\b|\brotate\b|\bspin\b|\bface\b", norm))
        is_turn_around = bool(re.search(r"\b(turn around|turn about|about face|180)\b", norm))
        explicit_v = _extract_velocity(norm)
        explicit_heading = _extract_heading(norm)

        # Nothing navigation-like at all.
        if not (
            has_verb or has_run or has_sprint or direction or style_word
            or is_turn or is_turn_around or explicit_heading is not None
        ):
            return None

        # Determine style + nominal velocity.
        style = NavStyle.WALKING
        velocity = skills.DEFAULT_NAV_VELOCITY
        if has_sprint:
            style = NavStyle.RUNNING
            velocity = skills.SPRINT_VELOCITY
        elif has_run:
            style = NavStyle.RUNNING
            velocity = skills.RUN_VELOCITY
        if style_word is not None:
            style = style_word

        # Speed modifiers (only override the base nominal for walking-class).
        if _contains_any(norm, skills.SLOW_WORDS) and not (has_run or has_sprint):
            velocity = skills.SLOW_NAV_VELOCITY
        elif _contains_any(norm, skills.FAST_WORDS) and not (has_run or has_sprint):
            velocity = skills.FAST_NAV_VELOCITY

        # Heading.
        heading = 0.0
        confidence = CONF_OK
        if direction is not None:
            heading = direction[1]
            confidence = CONF_STRONG
            if direction[0] in ("left", "leftward", "leftwards", "right",
                                 "rightward", "rightwards") and not (has_run or has_sprint):
                velocity = (
                    skills.SIDE_NAV_VELOCITY if explicit_v is None and velocity ==
                    skills.DEFAULT_NAV_VELOCITY else velocity
                )
            if direction[0] in ("backward", "backwards", "back", "reverse") and not (
                has_run or has_sprint
            ):
                velocity = (
                    skills.BACKWARD_NAV_VELOCITY if explicit_v is None and velocity ==
                    skills.DEFAULT_NAV_VELOCITY else velocity
                )
        if explicit_heading is not None:
            heading = explicit_heading
            confidence = CONF_STRONG

        if explicit_v is not None:
            velocity = explicit_v

        # "turn around" -> face backward in place.
        if is_turn_around and explicit_heading is None:
            heading = 180.0
            if explicit_v is None:
                velocity = 0.0
            confidence = CONF_STRONG

        # In-place turn: keep facing change but no forward motion.
        if is_turn and direction is not None and explicit_v is None:
            velocity = 0.0
            confidence = CONF_STRONG

        return self._ok(
            raw_text, norm,
            SetNavigationCommand(
                velocity_mps=max(0.0, velocity),
                heading_deg=skills.normalize_heading_deg(heading),
                style=style,
            ),
            confidence,
        )

    # ------------------------------------------------------------------ #
    def _extract_height(self, norm: str) -> Optional[float]:
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:m|meter|meters|metre|metres)\b", norm)
        if m:
            return float(m.group(1))
        return None

    def _ok(self, raw_text: str, norm: str, command, confidence: float) -> ParseResult:
        return ParseResult(
            ok=True, confidence=confidence, raw_text=raw_text,
            normalized_text=norm, command=command,
        )

    def _clarify(self, raw_text: str, norm: str, reason: str) -> ParseResult:
        question = (
            "I didn't understand that as a planner command. "
            "Try e.g. 'walk forward', 'squat', 'left jab', or 'stop'."
        )
        return ParseResult(
            ok=False, confidence=CONF_NONE, raw_text=raw_text, normalized_text=norm,
            command=ClarifyCommand(question=question, original_text=raw_text),
            reason=reason,
        )


def parse_text(text: str, confidence_threshold: float = 0.75,
               boxing_active: bool = False) -> ParseResult:

    return DeterministicParser(confidence_threshold).parse(text, boxing_active=boxing_active)


def parse_plan(text: str, confidence_threshold: float = 0.75,
               boxing_active: bool = False) -> List[ParseResult]:
    return DeterministicParser(confidence_threshold).parse_plan(
        text, boxing_active=boxing_active
    )


_PARSER_BACKENDS: Dict[str, Callable[[ParserConfig], "CommandParser"]] = {}


def register(name: str):
    def decorator(factory: Callable[[ParserConfig], "CommandParser"]):
        _PARSER_BACKENDS[name.lower()] = factory
        return factory
    return decorator


def build_parser(cfg: ParserConfig) -> "CommandParser":
    key = cfg.backend.lower()
    if key not in _PARSER_BACKENDS:
        known = ", ".join(sorted(_PARSER_BACKENDS)) or "(none)"
        raise ValueError(f"Unknown parser backend {cfg.backend!r}. Known: {known}")
    return _PARSER_BACKENDS[key](cfg)


class CommandParser:
    def parse(self, text: str, *, boxing_active: bool = False) -> ParseResult:
        ...


class ModelParser:
    def __init__(self, cfg: ParserConfig) -> None:
        self.cfg = cfg

    def parse(self, text: str, *, boxing_active: bool = False) -> ParseResult:
        normalized = normalize_text(text)
        try:
            payload = self.predict_tool_call(
                text, normalized=normalized, boxing_active=boxing_active
            )
            command = validate_tool_call(payload)
        except NotImplementedError:
            raise
        except ValidationError as exc:
            log.warning("Model parser output failed schema validation: %s", exc)
            return self._model_clarify(text, normalized, "model output failed schema validation")
        except Exception as exc:
            log.warning("Model parser failed: %s", exc)
            return self._model_clarify(text, normalized, f"model parser error: {exc}")
        confidence = CONF_NONE if getattr(command, "tool", None) == "clarify" else CONF_STRONG
        return ParseResult(
            ok=confidence > 0.0, confidence=confidence, raw_text=text,
            normalized_text=normalized, command=command, reason="model",
        )

    def predict_tool_call(
        self, text: str, *, normalized: str, boxing_active: bool,
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            "Implement predict_tool_call() in a ModelParser subclass, or register "
            "a custom CommandParser via voice_control.parsers.register()."
        )

    def _model_clarify(self, raw: str, normalized: str, reason: str) -> ParseResult:
        return ParseResult(
            ok=False, confidence=CONF_NONE, raw_text=raw, normalized_text=normalized,
            command=ClarifyCommand(
                question="Could you rephrase that as a movement command?",
                original_text=raw,
            ),
            reason=reason,
        )


@register("deterministic")
def _build_deterministic(cfg: ParserConfig) -> CommandParser:
    return DeterministicParser(cfg.confidence_threshold)


@register("model")
def _build_model(cfg: ParserConfig) -> CommandParser:
    return ModelParser(cfg)


from . import lfm_g1  # noqa: F401 — registers "lfm_g1" backend
from . import lfm_g1_gguf  # noqa: F401 — registers "lfm_g1_gguf" backend
