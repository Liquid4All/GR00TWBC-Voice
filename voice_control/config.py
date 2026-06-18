"""Configuration loading for the voice control package.

The config is a small set of nested dataclasses with defaults that exactly
mirror ``configs/voice_control.yaml``. YAML is loaded with PyYAML when present;
otherwise a tiny built-in parser handles the (simple, list-free) config so the
package stays usable offline without extra dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import logging
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #

@dataclass
class AudioConfig:
    backend: str = "vosk"  # "vosk" | "whisper_cpp"
    sample_rate: int = 16000
    device: Optional[int] = None
    vad: bool = True
    vad_aggressiveness: int = 2
    phrase_timeout_s: float = 1.0
    max_utterance_s: float = 8.0


@dataclass
class WakeConfig:
    mode: str = "push_to_talk"  # no_wake_debug | push_to_talk | wake_word | vad_only
    phrase: str = "hey sonic"
    require_wake_word_for_motion: bool = True


@dataclass
class AsrConfig:
    vosk_model_path: str = "models/vosk-model-small-en-us"
    whisper_cpp_bin: Optional[str] = None
    whisper_model_path: Optional[str] = None


@dataclass
class ParserConfig:
    use_llm_fallback: bool = False
    confidence_threshold: float = 0.75
    # Backend for any local LLM call: "llama_cpp" (HTTP /completion server) or
    # "hf_transformers" (load a HuggingFace checkpoint in-process).
    llm_backend: str = "hf_transformers"
    llm_endpoint: str = "http://127.0.0.1:8080/completion"
    # HuggingFace transformers backend (used when llm_backend == "hf_transformers").
    # Default is the LiquidAI LFM2 on-device reasoning checkpoint.
    hf_model_id: str = (
        "LiquidAI/tim_grpo230M_from978997_multidomain_lr3e-6_ent0.00_"
        "kl0.001_b256_n16_res4k_step100_987260_HF"
    )
    hf_device: str = "auto"          # auto | cpu | cuda | mps
    hf_dtype: str = "auto"           # auto | float16 | bfloat16 | float32
    hf_max_new_tokens: int = 256     # room for the reasoning model to think + answer
    hf_temperature: float = 0.0      # 0 => greedy/deterministic
    # Use a local LLM to dynamically decide how long each step of a composed
    # command should run before advancing to the next one. When the LLM is
    # unavailable, a deterministic heuristic (distance / speed, etc.) is used.
    use_llm_duration: bool = False
    # Sanity bounds for an estimated step duration (seconds). Not a safety clamp;
    # just guards against a degenerate 0 s or a runaway estimate.
    llm_duration_min_s: float = 0.5
    llm_duration_max_s: float = 120.0


@dataclass
class SafetyConfig:
    # Velocity/height clamps were removed; only the dry-run/execute gate and the
    # command-timeout watchdog remain.
    dry_run: bool = True
    execute: bool = False
    command_timeout_s: float = 2.0


@dataclass
class PublisherConfig:
    backend: str = "stub"  # stub | existing_repo | zmq | ros2
    # Planner *input* path for the C++ ZMQManager (--zmq-port, default 5556).
    # NOTE: 5557 is the deploy debug *output* port, not a command input.
    zmq_endpoint: str = "tcp://127.0.0.1:5556"
    ros2_topic: str = "/sonic/planner_command"
    # Planner replan period (seconds). 0.1 == 10 Hz, per the paper / deploy stack.
    planner_dt: float = 0.1
    # Default time to hold each step of a composed/sequential command when no
    # explicit duration is given (seconds).
    segment_dwell_s: float = 3.0


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_file: Optional[str] = "logs/voice_control.log"


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    wake: WakeConfig = field(default_factory=WakeConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    parser: ParserConfig = field(default_factory=ParserConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    publisher: PublisherConfig = field(default_factory=PublisherConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Config":
        cfg = cls()
        if not data:
            return cfg
        for f in fields(cls):
            section = data.get(f.name)
            if section is None:
                continue
            if not isinstance(section, dict):
                log.warning("Config section %r is not a mapping; ignoring", f.name)
                continue
            current = getattr(cfg, f.name)
            setattr(cfg, f.name, _apply_dataclass(current, section))
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        data = _load_yaml(Path(path))
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        return _dataclass_to_dict(self)


def _apply_dataclass(instance: Any, overrides: Dict[str, Any]) -> Any:
    for key, value in overrides.items():
        if not hasattr(instance, key):
            log.warning("Unknown config key %r; ignoring", key)
            continue
        setattr(instance, key, value)
    return instance


def _dataclass_to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _dataclass_to_dict(getattr(obj, f.name)) for f in fields(obj)}
    return obj


# --------------------------------------------------------------------------- #
# YAML loading (PyYAML if available, else a minimal fallback)
# --------------------------------------------------------------------------- #

def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except ImportError:
        log.warning("PyYAML not installed; using minimal built-in YAML parser.")
        return _minimal_yaml_parse(text)


def _coerce_scalar(token: str) -> Any:
    token = token.strip()
    if token == "" or token in ("null", "~", "None"):
        return None
    low = token.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if (token.startswith('"') and token.endswith('"')) or (
        token.startswith("'") and token.endswith("'")
    ):
        return token[1:-1]
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    return token


def _minimal_yaml_parse(text: str) -> Dict[str, Any]:
    """Parse the limited YAML subset used by voice_control.yaml.

    Supports two-level nested mappings of scalars. Lists / multi-line / anchors
    are *not* supported (the shipped config does not use them).
    """

    root: Dict[str, Any] = {}
    current_section: Optional[Dict[str, Any]] = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip()
        if indent == 0:
            if value == "":
                current_section = {}
                root[key] = current_section
            else:
                root[key] = _coerce_scalar(value)
                current_section = None
        else:
            if current_section is None:
                current_section = {}
                root[key] = current_section
            current_section[key] = _coerce_scalar(value)
    return root


def setup_logging(cfg: LoggingConfig) -> None:
    """Configure root logging from the logging config section."""

    level = getattr(logging, str(cfg.level).upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if cfg.log_file:
        try:
            log_path = Path(cfg.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_path))
        except OSError as exc:  # pragma: no cover - filesystem dependent
            log.warning("Could not open log file %s: %s", cfg.log_file, exc)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=handlers,
        force=True,
    )
