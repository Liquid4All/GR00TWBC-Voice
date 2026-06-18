"""Offline voice command interface for the SONIC kinematic motion planner.

Pipeline: microphone -> wake/VAD -> ASR -> deterministic parser (optional LLM
fallback) -> validated PlannerCommand -> safety clamp -> existing SONIC planner
command path (ZMQ). Everything runs on-device; no cloud services.

See ``voice_control/README.md`` for architecture and usage.
"""

from __future__ import annotations

from .config import Config
from .duration import LLMDurationEstimator, extract_distance_m
from .parser import DeterministicParser, normalize_text, parse_plan, parse_text, split_segments
from .safety import SafetyGuard
from .schemas import (
    ClarifyCommand,
    GetUpCommand,
    ParseResult,
    PlannerToolCall,
    SetBoxingActionCommand,
    SetCrawlCommand,
    SetNavigationCommand,
    SetPostureCommand,
    StopCommand,
    validate_tool_call,
)

__all__ = [
    "Config",
    "DeterministicParser",
    "parse_text",
    "parse_plan",
    "split_segments",
    "normalize_text",
    "LLMDurationEstimator",
    "extract_distance_m",
    "SafetyGuard",
    "ParseResult",
    "PlannerToolCall",
    "StopCommand",
    "SetNavigationCommand",
    "SetCrawlCommand",
    "SetPostureCommand",
    "SetBoxingActionCommand",
    "GetUpCommand",
    "ClarifyCommand",
    "validate_tool_call",
]

__version__ = "0.1.0"
