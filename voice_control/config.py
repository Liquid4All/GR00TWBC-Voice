from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import logging
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    backend: str = "whisper_cpp"
    source: str = "device"
    sample_rate: int = 16000
    device: Optional[int] = None
    frame_ms: int = 30
    vad: bool = True
    vad_aggressiveness: int = 2
    phrase_timeout_s: float = 1.0
    max_utterance_s: float = 8.0
    mcast_group: str = "239.168.123.161"
    mcast_port: int = 5555
    mcast_iface_ip: Optional[str] = None


@dataclass
class WakeConfig:
    mode: str = "push_to_talk"
    phrase: str = "hey sonic"


@dataclass
class AsrConfig:
    whisper_cpp_bin: Optional[str] = None
    whisper_model_path: Optional[str] = None
    whisper_language: str = "en"
    whisper_extra_args: Optional[list[str]] = None


@dataclass
class ParserConfig:
    backend: str = "deterministic"
    confidence_threshold: float = 0.75
    duration_min_s: float = 0.5
    duration_max_s: float = 120.0


@dataclass
class SafetyConfig:
    dry_run: bool = True
    execute: bool = False
    command_timeout_s: float = 2.0


@dataclass
class PublisherConfig:
    backend: str = "stub"
    local_planner_endpoint: str = "tcp://127.0.0.1:5556"
    zmq_endpoint: str = "tcp://127.0.0.1:5556"
    zmq_bind: bool = True
    planner_dt: float = 0.1
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
        return cls.from_dict(_load_yaml(Path(path)))

    def to_dict(self) -> Dict[str, Any]:
        return _dataclass_to_dict(self)


def _apply_dataclass(instance: Any, overrides: Dict[str, Any]) -> Any:
    for key, value in overrides.items():
        if hasattr(instance, key):
            setattr(instance, key, value)
        else:
            log.warning("Unknown config key %r; ignoring", key)
    return instance


def _dataclass_to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _dataclass_to_dict(getattr(obj, f.name)) for f in fields(obj)}
    return obj


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
    if (token.startswith('"') and token.endswith('"')) or (token.startswith("'") and token.endswith("'")):
        return token[1:-1]
    for caster in (int, float):
        try:
            return caster(token)
        except ValueError:
            pass
    return token


def _minimal_yaml_parse(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    current_section: Optional[Dict[str, Any]] = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip() or ":" not in line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, value = line.strip().partition(":")
        key, value = key.strip(), value.strip()
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
    level = getattr(logging, str(cfg.level).upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if cfg.log_file:
        try:
            log_path = Path(cfg.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_path))
        except OSError as exc:
            log.warning("Could not open log file %s: %s", cfg.log_file, exc)
    logging.basicConfig(
        level=level, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=handlers, force=True,
    )
