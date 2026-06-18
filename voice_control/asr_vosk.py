"""Vosk ASR backend (default).

Vosk is the default backend because the command vocabulary is small and
deterministic, and Vosk supports a *grammar* (restricted vocabulary) mode that
greatly improves accuracy and latency for a closed command set. The model runs
fully offline.

``vosk`` is imported lazily so the package imports without it. Only final
transcripts are returned for parsing (partials are logged for debugging).
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from . import skills

log = logging.getLogger(__name__)


def build_command_grammar() -> List[str]:
    """Build a Vosk grammar vocabulary from the command set + synonyms."""

    words: set[str] = set()
    for phrase in skills.STOP_WORDS:
        words.update(phrase.split())
    for phrase in skills.DIRECTION_TO_HEADING:
        words.update(phrase.split())
    for phrase in skills.STYLE_WORDS:
        words.update(phrase.split())
    words.update(skills.WALK_VERBS)
    words.update(skills.RUN_WORDS)
    words.update(skills.SPRINT_WORDS)
    words.update(skills.SLOW_WORDS)
    words.update(skills.FAST_WORDS)
    # Command nouns / fillers commonly spoken.
    words.update(
        [
            "stop", "halt", "freeze", "cancel", "emergency",
            "squat", "lower", "higher", "kneel", "knee", "knees", "one", "two",
            "both", "stand", "up", "get", "crawl", "hand", "elbow",
            "box", "boxing", "stance", "idle", "block", "jab", "hook", "side", "step",
            "turn", "rotate", "left", "right", "forward", "backward", "back",
            "meters", "meter", "per", "second", "degrees", "degree", "heading",
            "point", "zero", "three", "four", "five", "six", "seven", "eight", "nine",
            "the", "to", "a", "and", "please", "robot", "now",
        ]
    )
    return sorted(w for w in words if w)


class VoskASR:
    """Offline ASR using Vosk, optionally constrained to the command grammar."""

    def __init__(
        self,
        model_path: str,
        sample_rate: int = 16000,
        use_grammar: bool = True,
    ) -> None:
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.use_grammar = use_grammar
        self._model = None
        self._recognizer = None
        self._init_model()

    def _init_model(self) -> None:
        try:
            from vosk import KaldiRecognizer, Model  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "vosk is required for the Vosk ASR backend. Install with `pip install vosk` "
                "and download a model (e.g. vosk-model-small-en-us-0.15) to "
                f"{self.model_path!r}."
            ) from exc

        import os

        if not os.path.exists(self.model_path):
            raise RuntimeError(
                f"Vosk model not found at {self.model_path!r}. Download a small English model "
                "from https://alphacephei.com/vosk/models and unpack it there."
            )

        self._model = Model(self.model_path)
        if self.use_grammar:
            grammar = json.dumps(build_command_grammar() + ["[unk]"])
            self._recognizer = KaldiRecognizer(self._model, self.sample_rate, grammar)
        else:
            self._recognizer = KaldiRecognizer(self._model, self.sample_rate)
        self._recognizer.SetWords(True)
        log.info("Vosk model loaded (grammar=%s)", self.use_grammar)

    def transcribe(self, pcm: bytes) -> str:
        """Transcribe a complete utterance (16-bit mono PCM). Final text only."""

        if self._recognizer is None:  # pragma: no cover - defensive
            raise RuntimeError("Vosk recognizer not initialised.")
        self._recognizer.AcceptWaveform(pcm)
        result = json.loads(self._recognizer.FinalResult())
        text = result.get("text", "").strip()
        log.debug("Vosk final transcript: %r", text)
        return text

    def partial(self, pcm: bytes) -> Optional[str]:  # pragma: no cover - streaming
        if self._recognizer is None:
            return None
        self._recognizer.AcceptWaveform(pcm)
        partial = json.loads(self._recognizer.PartialResult()).get("partial", "")
        if partial:
            log.debug("Vosk partial: %r", partial)
        return partial or None
