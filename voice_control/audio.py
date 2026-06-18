"""Microphone capture and voice-activity detection.

All heavy audio dependencies (``sounddevice``, ``webrtcvad``) are imported
lazily so the package (and its tests) import cleanly on machines without audio
hardware or those libraries. ``--text`` mode never touches this module.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Iterator, Optional

log = logging.getLogger(__name__)


@dataclass
class AudioFrame:
    pcm: bytes  # 16-bit little-endian mono PCM
    sample_rate: int


class MicrophoneError(RuntimeError):
    pass


class MicrophoneCapture:
    """Threaded microphone capture into a frame queue.

    Produces 16-bit mono PCM frames of ``frame_ms`` length. Designed so the
    audio thread never blocks the control / planner loops.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        device: Optional[int] = None,
        frame_ms: int = 30,
        max_queue: int = 100,
    ) -> None:
        self.sample_rate = sample_rate
        self.device = device
        self.frame_ms = frame_ms
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self._queue: "queue.Queue[AudioFrame]" = queue.Queue(maxsize=max_queue)
        self._stream = None
        self._stop = threading.Event()

    def _require_sounddevice(self):
        try:
            import sounddevice as sd  # type: ignore
        except ImportError as exc:
            raise MicrophoneError(
                "sounddevice is required for microphone capture. "
                "Install with `pip install sounddevice` (Jetson: also `sudo apt install "
                "libportaudio2`)."
            ) from exc
        return sd

    def start(self) -> None:
        sd = self._require_sounddevice()

        def _callback(indata, frames, time_info, status):  # pragma: no cover - hw
            if status:
                log.debug("audio status: %s", status)
            try:
                self._queue.put_nowait(AudioFrame(bytes(indata), self.sample_rate))
            except queue.Full:
                pass  # drop frames rather than block

        self._stop.clear()
        self._stream = sd.RawInputStream(  # pragma: no cover - hw
            samplerate=self.sample_rate,
            blocksize=self.frame_samples,
            device=self.device,
            dtype="int16",
            channels=1,
            callback=_callback,
        )
        self._stream.start()
        log.info("Microphone started (sr=%d, device=%s)", self.sample_rate, self.device)

    def frames(self) -> Iterator[AudioFrame]:
        while not self._stop.is_set():
            try:
                yield self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:  # pragma: no cover - hw
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:
                log.debug("error closing stream: %s", exc)
            self._stream = None


class VoiceActivityDetector:
    """Thin wrapper over webrtcvad (optional)."""

    def __init__(self, aggressiveness: int = 2, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self._vad = None
        try:
            import webrtcvad  # type: ignore

            self._vad = webrtcvad.Vad(int(aggressiveness))
        except ImportError:
            log.warning("webrtcvad not installed; VAD disabled (treating all frames as speech).")

    def is_speech(self, frame: AudioFrame) -> bool:
        if self._vad is None:
            return True
        try:
            return self._vad.is_speech(frame.pcm, frame.sample_rate)
        except Exception:  # pragma: no cover
            return True


def capture_utterance(
    mic: MicrophoneCapture,
    vad: Optional[VoiceActivityDetector],
    max_utterance_s: float,
    phrase_timeout_s: float,
) -> bytes:
    """Collect PCM for one utterance, ending after a trailing silence.

    Returns concatenated 16-bit PCM bytes. Blocks until the utterance completes
    or ``max_utterance_s`` elapses.
    """

    collected = bytearray()
    speech_started = False
    silence_ms = 0.0
    elapsed_ms = 0.0
    for frame in mic.frames():  # pragma: no cover - hardware dependent
        elapsed_ms += mic.frame_ms
        is_speech = vad.is_speech(frame) if vad is not None else True
        if is_speech:
            speech_started = True
            silence_ms = 0.0
            collected.extend(frame.pcm)
        elif speech_started:
            silence_ms += mic.frame_ms
            collected.extend(frame.pcm)
            if silence_ms >= phrase_timeout_s * 1000:
                break
        if elapsed_ms >= max_utterance_s * 1000:
            break
    return bytes(collected)
