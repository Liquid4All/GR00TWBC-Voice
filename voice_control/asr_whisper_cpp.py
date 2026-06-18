"""whisper.cpp ASR backend (optional).

Wraps a local whisper.cpp binary (e.g. ``main``/``whisper-cli``) and a local
GGUF/GGML model file (``tiny.en`` or ``base.en``) via subprocess. Paths are
provided by config; nothing is downloaded automatically.

The recorded utterance PCM is written to a temporary 16 kHz mono WAV, passed to
the binary, and the printed transcript is parsed back out.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
import wave
from typing import Optional

log = logging.getLogger(__name__)


class WhisperCppASR:
    def __init__(
        self,
        binary_path: str,
        model_path: str,
        sample_rate: int = 16000,
        language: str = "en",
        extra_args: Optional[list[str]] = None,
    ) -> None:
        if not binary_path or not os.path.exists(binary_path):
            raise RuntimeError(
                f"whisper.cpp binary not found at {binary_path!r}. Build whisper.cpp and set "
                "asr.whisper_cpp_bin to the compiled binary path."
            )
        if not model_path or not os.path.exists(model_path):
            raise RuntimeError(
                f"whisper.cpp model not found at {model_path!r}. Download e.g. "
                "ggml-tiny.en.bin and set asr.whisper_model_path."
            )
        self.binary_path = binary_path
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.language = language
        self.extra_args = extra_args or []

    def transcribe(self, pcm: bytes) -> str:
        wav_path = self._write_wav(pcm)
        try:
            return self._run_whisper(wav_path)
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

    def _write_wav(self, pcm: bytes) -> str:
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="sonic_voice_")
        os.close(fd)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm)
        return path

    def _run_whisper(self, wav_path: str) -> str:
        cmd = [
            self.binary_path,
            "-m", self.model_path,
            "-f", wav_path,
            "-l", self.language,
            "-nt",  # no timestamps
            *self.extra_args,
        ]
        log.debug("Running whisper.cpp: %s", " ".join(cmd))
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60, check=False
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            log.error("whisper.cpp invocation failed: %s", exc)
            return ""
        if proc.returncode != 0:
            log.error("whisper.cpp returned %d: %s", proc.returncode, proc.stderr.strip())
            return ""
        return self._parse_output(proc.stdout)

    @staticmethod
    def _parse_output(stdout: str) -> str:
        lines = []
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            # Strip any leftover "[..] " timestamp prefix defensively.
            line = re.sub(r"^\[[^\]]*\]\s*", "", line)
            lines.append(line)
        text = " ".join(lines).strip()
        log.debug("whisper.cpp transcript: %r", text)
        return text
