"""LLM-based step-duration estimation for the compositional pipeline.

When a compound command is split into sequential steps (see
:func:`voice_control.parser.split_segments`), the runtime must decide how long to
hold each step before advancing to the next. Instead of a fixed dwell, this
module asks a *local* LLM to estimate, from the action and its arguments, how
long the robot needs to *finish* that step -- e.g. moving 5 m at 3 m/s should
take longer than moving 3 m at 3 m/s.

The LLM is given the parsed tool call (as JSON) plus the original phrase (so it
can see distances like "5 meters" that are not planner-command fields) and must
return a single number. If the LLM is unavailable or returns garbage, a
deterministic ``heuristic`` (distance / speed, turn angle, posture settle time)
is used so the pipeline still works fully offline.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from .config import ParserConfig
from .llm_parser import _extract_json, call_llm_completion

log = logging.getLogger(__name__)

# Constrain llama.cpp to exactly {"duration_s": <number>}.
DURATION_GRAMMAR = r"""
root   ::= ws "{" ws "\"duration_s\"" ws ":" ws number ws "}" ws
number ::= "-"? [0-9]+ ("." [0-9]+)?
ws     ::= [ \t\n]*
"""

SYSTEM_PROMPT = (
    "You estimate how many SECONDS a humanoid robot needs to COMPLETE a single "
    "motion command before it should move on to the next step in a sequence.\n"
    "You are given the parsed command as JSON and the original spoken phrase.\n"
    "Think briefly if needed, then give the FINAL answer on its own last line as "
    "JSON: {\"duration_s\": <positive number>}.\n\n"
    "Guidance:\n"
    "- Navigation/crawl with a stated distance: time = distance_meters / speed_mps, "
    "plus ~1 s to settle. Lower speed or larger distance => more time.\n"
    "- Navigation/crawl with no distance: assume a short move of ~3 s.\n"
    "- In-place turn (velocity 0): ~1 s per 60 degrees of |heading| (180 deg ~ 3 s).\n"
    "- Posture changes (squat/kneel/stand): ~2-4 s to settle.\n"
    "- Boxing actions: ~1-2 s.\n"
    "- get_up: ~4-5 s.\n"
    "- Never return 0 or negative.\n"
)

_DURATION_KEY_RE = re.compile(
    r"duration_s[\"']?\s*[:=]\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE
)
_BOXED_RE = re.compile(r"\\boxed\{\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_TRAILING_NUMBER_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)?\s*[.}\]\"']*\s*$",
    re.IGNORECASE,
)


def parse_duration_seconds(text: str) -> Optional[float]:
    """Extract a duration in seconds from a (possibly chatty) model output.

    Handles strict JSON, a ``duration_s: N`` fragment, a ``\\boxed{N}`` answer
    (common for GRPO reasoning models), or a bare trailing number -- because a
    reasoning model may emit chain-of-thought before its final answer.
    """

    if not text:
        return None

    # 1) Explicit duration_s key (in JSON or free text).
    m = _DURATION_KEY_RE.search(text)
    if m:
        return _to_positive_float(m.group(1))

    # 2) Structured JSON object somewhere in the text.
    payload = _extract_json(text)
    if payload and "duration_s" in payload:
        return _to_positive_float(payload["duration_s"])

    # 3) \boxed{N} final answer.
    m = _BOXED_RE.search(text)
    if m:
        return _to_positive_float(m.group(1))

    # 4) Bare trailing number (last line of a reasoning trace).
    last_line = text.strip().splitlines()[-1] if text.strip() else ""
    m = _TRAILING_NUMBER_RE.search(last_line)
    if m:
        return _to_positive_float(m.group(1))

    return None


def _to_positive_float(value) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f > 0 else None

_DISTANCE_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(?:meters?|metres?|m)\b", re.IGNORECASE
)


def extract_distance_m(text: str) -> Optional[float]:
    """Pull a distance in metres out of the spoken phrase, if present."""

    if not text:
        return None
    match = _DISTANCE_RE.search(text)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return value if value > 0 else None


class LLMDurationEstimator:
    """Estimate per-step durations, LLM-first with a deterministic fallback."""

    def __init__(self, cfg: ParserConfig) -> None:
        self.cfg = cfg
        self.min_s = float(cfg.llm_duration_min_s)
        self.max_s = float(cfg.llm_duration_max_s)

    # ------------------------------------------------------------------ #
    def estimate(self, command, segment_text: str) -> float:
        """Return seconds to run ``command``. Uses the LLM, else the heuristic."""

        llm_value = self._estimate_llm(command, segment_text)
        if llm_value is not None:
            return self._bound(llm_value)
        return self._bound(self.heuristic(command, segment_text))

    # ------------------------------------------------------------------ #
    def _estimate_llm(self, command, segment_text: str) -> Optional[float]:
        try:
            command_json = command.model_dump_json()
        except Exception:  # pragma: no cover - defensive
            command_json = json.dumps(getattr(command, "__dict__", {}), default=str)

        prompt = (
            f"{SYSTEM_PROMPT}\n"
            f"Command JSON: {command_json}\n"
            f"Spoken phrase: {segment_text!r}\n"
            f"JSON:"
        )
        try:
            completion = call_llm_completion(
                self.cfg, prompt, grammar=DURATION_GRAMMAR, n_predict=48, stop=["\n\n"]
            )
        except Exception as exc:
            # Log the concrete error (type + repr) and full traceback; a bare
            # "%s" can be empty for exceptions with no message.
            log.warning(
                "Duration LLM call failed (%s: %r); using heuristic.",
                type(exc).__name__, exc, exc_info=True,
            )
            return None

        value = parse_duration_seconds(completion)
        if value is None:
            log.warning("Duration LLM returned no usable number; using heuristic.")
            return None
        log.info("LLM duration estimate: %.2f s for %s", value, command_json)
        return value

    # ------------------------------------------------------------------ #
    def heuristic(self, command, segment_text: str) -> float:
        """Deterministic fallback duration in seconds."""

        tool = getattr(command, "tool", None)

        if tool in ("set_navigation", "set_crawl"):
            velocity = float(getattr(command, "velocity_mps", 0.0) or 0.0)
            heading = float(getattr(command, "heading_deg", 0.0) or 0.0)
            distance = extract_distance_m(segment_text)
            if velocity <= 1e-3:
                # In-place turn: ~1 s per 60 deg.
                return max(1.0, abs(heading) / 60.0)
            if distance is not None:
                return distance / velocity + 1.0
            return 3.0

        if tool == "set_posture":
            return 3.0
        if tool == "set_boxing_action":
            return 1.5
        if tool == "get_up":
            return 4.5
        return 3.0

    # ------------------------------------------------------------------ #
    def _bound(self, value: float) -> float:
        return max(self.min_s, min(self.max_s, float(value)))
