"""Command gating for validated tool calls.

Value clamps have been intentionally **removed**: navigation/crawl speeds,
pelvis heights and headings are passed through to the planner unchanged. The
guard only provides the non-clamping gates the runtime relies on:

* dry-run / execute gating (never send to the robot unless explicitly allowed),
* confidence gating (do not move on a low-confidence guess).
"""

from __future__ import annotations

import logging

from .config import SafetyConfig
from .schemas import (
    ClarifyCommand,
    PlannerToolCallType,
    StopCommand,
    StopReason,
)

log = logging.getLogger(__name__)


class SafetyGuard:
    def __init__(self, cfg: SafetyConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------ #
    @staticmethod
    def passes_confidence(
        command: PlannerToolCallType, confidence: float, threshold: float
    ) -> bool:
        """Whether a command is confident enough to act on.

        Stop and clarify are always allowed through (stop is safety-critical,
        clarify does not move the robot). Any motion command below threshold is
        rejected so the runtime never moves on a low-confidence guess.
        """

        if isinstance(command, (StopCommand, ClarifyCommand)):
            return True
        return confidence >= threshold

    def should_execute(self) -> bool:
        """True only if it is safe to actually send commands to the robot."""

        return bool(self.cfg.execute) and not bool(self.cfg.dry_run)

    def stop_for_safety(self) -> StopCommand:
        return StopCommand(reason=StopReason.SAFETY)
