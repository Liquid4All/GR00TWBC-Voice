"""Microphone capture, VAD, wake gating, and operator feedback."""

from __future__ import annotations

import logging
import queue
import socket
import struct
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Optional

log = logging.getLogger(__name__)


@dataclass
class AudioFrame:
    pcm: bytes  # 16-bit little-endian mono PCM
    sample_rate: int


class MicrophoneError(RuntimeError):
    pass


class MicrophoneCapture:

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

    def flush(self) -> None:
        """Drop any buffered frames (call right before a fresh utterance)."""

        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:  # pragma: no cover - hw
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:
                log.debug("error closing stream: %s", exc)
            self._stream = None


def find_robot_subnet_ip(prefix: str = "192.168.123.") -> Optional[str]:

    candidates: list[str] = []
    try:  # pragma: no cover - platform dependent
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            candidates.append(info[4][0])
    except OSError:
        pass
    # Fallback: probe the address used to reach the robot subnet.
    try:  # pragma: no cover - platform dependent
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect((prefix + "1", 9))
            candidates.append(probe.getsockname()[0])
        finally:
            probe.close()
    except OSError:
        pass
    for ip in candidates:
        if ip.startswith(prefix):
            return ip
    return None


class MulticastMicrophone:

    def __init__(
        self,
        sample_rate: int = 16000,
        group: str = "239.168.123.161",
        port: int = 5555,
        iface_ip: Optional[str] = None,
        frame_ms: int = 30,
        max_queue: int = 100,
    ) -> None:
        self.sample_rate = sample_rate
        self.group = group
        self.port = port
        self.iface_ip = iface_ip
        self.frame_ms = frame_ms
        self.frame_samples = int(sample_rate * frame_ms / 1000)
        self.frame_bytes = self.frame_samples * 2  # 16-bit mono
        self._queue: "queue.Queue[AudioFrame]" = queue.Queue(maxsize=max_queue)
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._buffer = bytearray()

    def start(self) -> None:
        iface_ip = self.iface_ip or find_robot_subnet_ip() or "0.0.0.0"
        if iface_ip == "0.0.0.0":
            log.warning(
                "No 192.168.123.x interface found; joining multicast on INADDR_ANY. "
                "If you receive no/zeroed audio, set audio.mcast_iface_ip to this "
                "host's robot-subnet IP."
            )
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", self.port))
        mreq = struct.pack(
            "4s4s", socket.inet_aton(self.group), socket.inet_aton(iface_ip)
        )
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        sock.settimeout(0.5)
        self._sock = sock
        self._stop.clear()
        self._thread = threading.Thread(target=self._recv_loop, daemon=True, name="g1-mic")
        self._thread.start()
        log.info(
            "G1 multicast mic started (group=%s:%d, iface=%s, sr=%d)",
            self.group, self.port, iface_ip, self.sample_rate,
        )

    def _recv_loop(self) -> None:  # pragma: no cover - network/hardware
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data:
                continue
            self._buffer.extend(data)
            while len(self._buffer) >= self.frame_bytes:
                chunk = bytes(self._buffer[: self.frame_bytes])
                del self._buffer[: self.frame_bytes]
                try:
                    self._queue.put_nowait(AudioFrame(chunk, self.sample_rate))
                except queue.Full:
                    pass  # drop frames rather than block

    def frames(self) -> Iterator[AudioFrame]:
        while not self._stop.is_set():
            try:
                yield self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

    def flush(self) -> None:
        """Drop any buffered frames (call right before a fresh utterance)."""

        self._buffer.clear()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:  # pragma: no cover - hw
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._sock is not None:  # pragma: no cover - hw
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None


class VoiceActivityDetector:

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


def capture_fixed(mic, seconds: float) -> bytes:

    collected = bytearray()
    elapsed_ms = 0.0
    for frame in mic.frames():  # pragma: no cover - hardware dependent
        collected.extend(frame.pcm)
        elapsed_ms += mic.frame_ms
        if elapsed_ms >= seconds * 1000:
            break
    return bytes(collected)


def build_capture(cfg) -> MicrophoneCapture | MulticastMicrophone:

    if cfg.source == "multicast":
        return MulticastMicrophone(
            sample_rate=cfg.sample_rate,
            group=cfg.mcast_group,
            port=cfg.mcast_port,
            iface_ip=cfg.mcast_iface_ip,
            frame_ms=cfg.frame_ms,
        )
    if cfg.source == "device":
        return MicrophoneCapture(cfg.sample_rate, cfg.device, frame_ms=cfg.frame_ms)
    raise ValueError(f"Unknown audio source: {cfg.source!r} (use 'device' or 'multicast')")


class Feedback:
    def __init__(self, enable_tts: bool = False) -> None:
        self._engine = None
        if enable_tts:
            try:
                import pyttsx3  # type: ignore
                self._engine = pyttsx3.init()
            except Exception as exc:
                log.warning("pyttsx3 unavailable (%s); console feedback only.", exc)

    def notify(self, message: str) -> None:
        log.info("[feedback] %s", message)
        print(f"[voice] {message}")
        if self._engine:
            try:
                self._engine.say(message)
                self._engine.runAndWait()
            except Exception as exc:
                log.debug("TTS failed: %s", exc)

    def confirm(self, command_summary: str) -> None:
        self.notify(f"OK: {command_summary}")

    def clarify(self, question: str) -> None:
        self.notify(f"Clarify: {question}")

    def warn(self, message: str) -> None:
        log.warning("[feedback] %s", message)
        print(f"[voice][warn] {message}")


class WakeMode(str, Enum):
    NO_WAKE_DEBUG = "no_wake_debug"
    PUSH_TO_TALK = "push_to_talk"
    WAKE_WORD = "wake_word"
    VAD_ONLY = "vad_only"


class WakeGate:
    def __init__(self, mode: str, phrase: str = "hey sonic") -> None:
        try:
            self.mode = WakeMode(mode)
        except ValueError:
            log.warning("Unknown wake mode %r; falling back to push_to_talk.", mode)
            self.mode = WakeMode.PUSH_TO_TALK
        self.phrase = phrase

    def wait_for_trigger(self) -> bool:
        if self.mode in (WakeMode.NO_WAKE_DEBUG, WakeMode.VAD_ONLY):
            return True
        if self.mode == WakeMode.PUSH_TO_TALK:
            try:
                input("[voice] Press Enter to talk (Ctrl-C to quit)... ")
                return True
            except (EOFError, KeyboardInterrupt):
                return False
        if self.mode == WakeMode.WAKE_WORD:
            log.warning("wake_word not wired; falling back to push_to_talk.")
            try:
                input("[voice] Press Enter to talk... ")
                return True
            except (EOFError, KeyboardInterrupt):
                return False
        return True
