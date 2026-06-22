"""Deterministic per-step duration estimation for the compositional pipeline.

When a compound command is split into sequential steps (see
:func:`voice_control.parser.split_segments`), the runtime must decide how long to
hold each step before advancing to the next. Rather than a single fixed dwell,
this module derives a sensible duration from the action and its arguments -- e.g.
moving 5 m at 3 m/s takes longer than moving 3 m at 3 m/s, and a 180 deg turn
takes longer than a 90 deg turn.

This is fully deterministic and offline: no LLM, no network. The estimate is
bounded by ``duration_min_s`` / ``duration_max_s`` from the parser config.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from .config import ParserConfig

log = logging.getLogger(__name__)

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


class HeuristicDurationEstimator:
    """Estimate how long each step should run, deterministically."""

    def __init__(self, cfg: ParserConfig) -> None:
        self.cfg = cfg
        self.min_s = float(cfg.duration_min_s)
        self.max_s = float(cfg.duration_max_s)

    def estimate(self, command, segment_text: str) -> float:
        """Return seconds to run ``command`` before advancing to the next step."""

        return self._bound(self.heuristic(command, segment_text))

    def heuristic(self, command, segment_text: str) -> float:
        """Deterministic duration in seconds derived from the command + phrase."""

        tool = getattr(command, "tool", None)

        if tool in ("set_navigation", "set_crawl"):
            velocity = float(getattr(command, "velocity_mps", 0.0) or 0.0)
            heading = float(getattr(command, "heading_deg", 0.0) or 0.0)
            distance = extract_distance_m(segment_text)
            if velocity <= 1e-3:
                # In-place turn: ~1 s per 60 deg.
                return max(1.0, abs(heading) / 60.0)
            if distance is not None:
                # time = distance / speed, plus ~1 s to settle.
                return distance / velocity + 1.0
            return 3.0

        if tool == "set_posture":
            return 3.0
        if tool == "set_boxing_action":
            return 1.5
        if tool == "get_up":
            return 4.5
        return 3.0

    def _bound(self, value: float) -> float:
        return max(self.min_s, min(self.max_s, float(value)))
