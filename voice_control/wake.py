"""Wake-word / push-to-talk / VAD gating.

Modes:
  * ``no_wake_debug`` -- always listening (desktop testing only).
  * ``push_to_talk``  -- press Enter (or a key) to start capture.
  * ``wake_word``     -- openWakeWord or Porcupine, if installed.
  * ``vad_only``      -- start capture when speech begins (debug only).

Default safe modes are ``push_to_talk`` and ``wake_word`` (never always-on for
motion). Stop commands may still be accepted in continuous modes by the runtime.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

log = logging.getLogger(__name__)


class WakeMode(str, Enum):
    NO_WAKE_DEBUG = "no_wake_debug"
    PUSH_TO_TALK = "push_to_talk"
    WAKE_WORD = "wake_word"
    VAD_ONLY = "vad_only"


CONTINUOUS_MODES = {WakeMode.NO_WAKE_DEBUG, WakeMode.VAD_ONLY}


class WakeGate:
    """Decides when the runtime is allowed to begin capturing an utterance."""

    def __init__(self, mode: str, phrase: str = "hey sonic") -> None:
        try:
            self.mode = WakeMode(mode)
        except ValueError:
            log.warning("Unknown wake mode %r; falling back to push_to_talk.", mode)
            self.mode = WakeMode.PUSH_TO_TALK
        self.phrase = phrase
        self._detector = None
        if self.mode == WakeMode.WAKE_WORD:
            self._detector = _try_init_wakeword(phrase)

    @property
    def is_continuous(self) -> bool:
        return self.mode in CONTINUOUS_MODES

    def wait_for_trigger(self) -> bool:
        """Block until the gate opens. Returns False to request shutdown."""

        if self.mode == WakeMode.NO_WAKE_DEBUG:
            return True
        if self.mode == WakeMode.PUSH_TO_TALK:
            try:  # pragma: no cover - interactive
                input("[voice] Press Enter to talk (Ctrl-C to quit)... ")
                return True
            except (EOFError, KeyboardInterrupt):
                return False
        if self.mode == WakeMode.VAD_ONLY:
            return True  # capture loop relies on VAD to detect speech
        if self.mode == WakeMode.WAKE_WORD:
            return self._wait_for_wakeword()
        return True

    def _wait_for_wakeword(self) -> bool:  # pragma: no cover - hardware dependent
        if self._detector is None:
            log.warning("No wake-word engine available; treating as push_to_talk.")
            try:
                input("[voice] (no wake engine) Press Enter to talk... ")
                return True
            except (EOFError, KeyboardInterrupt):
                return False
        return self._detector.wait()


def _try_init_wakeword(phrase: str):  # pragma: no cover - optional deps
    try:
        from openwakeword.model import Model  # type: ignore

        log.info("Using openWakeWord for wake phrase %r", phrase)
        return _OpenWakeWordDetector(Model())
    except ImportError:
        pass
    try:
        import pvporcupine  # type: ignore

        log.info("Using Porcupine for wake phrase %r", phrase)
        return _PorcupineDetector(pvporcupine)
    except ImportError:
        log.warning(
            "Neither openwakeword nor pvporcupine installed; wake_word mode will "
            "fall back to push_to_talk."
        )
        return None


class _OpenWakeWordDetector:  # pragma: no cover - optional deps
    def __init__(self, model) -> None:
        self.model = model

    def wait(self) -> bool:
        raise NotImplementedError(
            "openWakeWord streaming detection must be wired to the MicrophoneCapture "
            "stream in your deployment; see voice_control/README.md."
        )


class _PorcupineDetector:  # pragma: no cover - optional deps
    def __init__(self, pvporcupine) -> None:
        self.pvporcupine = pvporcupine

    def wait(self) -> bool:
        raise NotImplementedError(
            "Porcupine streaming detection must be wired to the MicrophoneCapture "
            "stream in your deployment; see voice_control/README.md."
        )
