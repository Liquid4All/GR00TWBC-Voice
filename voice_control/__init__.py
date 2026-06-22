"""Offline voice command interface for the SONIC kinematic motion planner.

Pipeline: microphone (sound card or Unitree G1 multicast) -> wake/VAD -> ASR
(whisper.cpp / Vosk) -> deterministic parser -> validated PlannerCommand ->
existing SONIC planner command path (ZMQ). Everything runs on-device; no cloud
services and no LLM.

See ``voice_control/README.md`` for architecture and usage.
"""

from __future__ import annotations

from .config import Config
from .duration import HeuristicDurationEstimator, extract_distance_m
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
    "HeuristicDurationEstimator",
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
