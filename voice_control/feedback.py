"""Operator feedback (console + optional offline TTS).

Used to confirm accepted commands, report clamps, and ask clarifying questions.
TTS is optional (``pyttsx3``) and degrades gracefully to console output.
"""

from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)


class Feedback:
    def __init__(self, enable_tts: bool = False) -> None:
        self._engine = None
        if enable_tts:
            self._engine = _try_init_tts()

    def notify(self, message: str) -> None:
        log.info("[feedback] %s", message)
        print(f"[voice] {message}")
        self._speak(message)

    def confirm(self, command_summary: str) -> None:
        self.notify(f"OK: {command_summary}")

    def clarify(self, question: str) -> None:
        self.notify(f"Clarify: {question}")

    def warn(self, message: str) -> None:
        log.warning("[feedback] %s", message)
        print(f"[voice][warn] {message}")

    def _speak(self, message: str) -> None:
        if self._engine is None:
            return
        try:  # pragma: no cover - hardware dependent
            self._engine.say(message)
            self._engine.runAndWait()
        except Exception as exc:  # pragma: no cover
            log.debug("TTS failed: %s", exc)


def _try_init_tts() -> Optional[object]:
    try:  # pragma: no cover - optional dependency
        import pyttsx3  # type: ignore

        return pyttsx3.init()
    except Exception as exc:  # pragma: no cover
        log.warning("pyttsx3 TTS unavailable (%s); using console feedback only.", exc)
        return None
