"""whisper.cpp ASR backend with registry for custom backends."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
import wave
from typing import Callable, Dict, Optional

from .config import AsrConfig, AudioConfig

log = logging.getLogger(__name__)

_ASR_BACKENDS: Dict[str, Callable[[AudioConfig, AsrConfig], "SpeechRecognizer"]] = {}


def register(name: str):
    def decorator(factory: Callable[[AudioConfig, AsrConfig], "SpeechRecognizer"]):
        _ASR_BACKENDS[name.lower()] = factory
        return factory
    return decorator


def build_asr(audio: AudioConfig, asr: AsrConfig) -> "SpeechRecognizer":
    key = audio.backend.lower()
    if key not in _ASR_BACKENDS:
        known = ", ".join(sorted(_ASR_BACKENDS)) or "(none)"
        raise ValueError(f"Unknown ASR backend {audio.backend!r}. Known: {known}")
    return _ASR_BACKENDS[key](audio, asr)


class SpeechRecognizer:
    def transcribe(self, pcm: bytes) -> str:
        ...


class WhisperCppASR:
    def __init__(
        self, binary_path: str, model_path: str, sample_rate: int = 16000,
        language: str = "en", extra_args: Optional[list[str]] = None,
    ) -> None:
        if not binary_path or not os.path.exists(binary_path):
            raise RuntimeError(f"whisper.cpp binary not found at {binary_path!r}")
        if not model_path or not os.path.exists(model_path):
            raise RuntimeError(f"whisper.cpp model not found at {model_path!r}")
        self.binary_path, self.model_path = binary_path, model_path
        self.sample_rate, self.language = sample_rate, language
        self.extra_args = extra_args or []

    def transcribe(self, pcm: bytes) -> str:
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="sonic_voice_")
        os.close(fd)
        try:
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(pcm)
            cmd = [
                self.binary_path, "-m", self.model_path, "-f", path,
                "-l", self.language, "-nt", *self.extra_args,
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
            except (subprocess.TimeoutExpired, OSError) as exc:
                log.error("whisper.cpp invocation failed: %s", exc)
                return ""
            if proc.returncode != 0:
                log.error("whisper.cpp returned %d: %s", proc.returncode, proc.stderr.strip())
                return ""
            lines = []
            for line in proc.stdout.splitlines():
                line = re.sub(r"^\[[^\]]*\]\s*", "", line.strip())
                if line:
                    lines.append(line)
            return " ".join(lines).strip()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass


@register("whisper_cpp")
def _build_whisper_cpp(audio: AudioConfig, asr: AsrConfig) -> WhisperCppASR:
    if not asr.whisper_cpp_bin or not asr.whisper_model_path:
        raise RuntimeError("whisper_cpp requires asr.whisper_cpp_bin and asr.whisper_model_path")
    return WhisperCppASR(
        asr.whisper_cpp_bin, asr.whisper_model_path, sample_rate=audio.sample_rate,
        language=asr.whisper_language, extra_args=asr.whisper_extra_args or None,
    )
