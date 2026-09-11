"""Deterrent playback.

Built-in cues are synthesised rather than shipped as files, so there are no binary
assets and every cue is reproducible. The device is opened once and buffers are
pre-rendered, because the latency budget allows 60 ms between the decision and the
first audible sample (§7 of the design doc).

Playback failure is reported, never swallowed: "Sound could not play" is a real
state the user has to see (E7).
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SAMPLE_RATE = 44_100
BUILT_INS = ("chirp", "clack", "hiss", "warble")


def _envelope(n: int, attack: float = 0.01, release: float = 0.35) -> np.ndarray:
    a = max(1, int(attack * n))
    r = max(1, int(release * n))
    env = np.ones(n)
    env[:a] = np.linspace(0.0, 1.0, a)
    env[-r:] = np.linspace(1.0, 0.0, r) ** 1.6
    return env


def synthesise(name: str, seconds: float = 0.42, seed: int = 0) -> np.ndarray:
    """Render one built-in cue as mono float32 in [-1, 1]."""
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    rng = np.random.default_rng(seed)

    if name == "chirp":
        f = np.linspace(1800.0, 3400.0, n)
        wave_ = np.sin(2 * np.pi * np.cumsum(f) / SAMPLE_RATE)
    elif name == "clack":
        # Two short noise transients: reads as a physical knock, not an alarm.
        wave_ = np.zeros(n)
        for offset in (0.0, 0.085):
            i = int(offset * SAMPLE_RATE)
            burst = min(int(0.03 * SAMPLE_RATE), n - i)
            if burst > 0:
                wave_[i:i + burst] = rng.normal(0, 1, burst) * np.linspace(1, 0, burst) ** 2
    elif name == "hiss":
        noise = rng.normal(0, 1, n)
        # Crude one-pole high-pass, so it sits where a cat actually reacts.
        wave_ = np.diff(noise, prepend=noise[0])
    elif name == "warble":
        f = 2600.0 + 700.0 * np.sin(2 * np.pi * 11.0 * t)
        wave_ = np.sin(2 * np.pi * np.cumsum(f) / SAMPLE_RATE)
    else:
        raise KeyError(f"unknown built-in sound {name!r}; choose from {BUILT_INS}")

    out = wave_ * _envelope(n)
    peak = float(np.max(np.abs(out))) or 1.0
    return (out / peak * 0.9).astype(np.float32)


@dataclass
class PlaybackResult:
    ok: bool
    latency_ms: float = 0.0
    backend: str = ""
    error: str = ""


@dataclass
class Player:
    """Pre-renders cues and plays them with the lowest-latency backend available."""

    volume: float = 0.6
    custom: dict[str, Path] = field(default_factory=dict)
    _cache: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _stream: object | None = field(default=None, repr=False)
    _sd: object | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    last_error: str = ""

    def __post_init__(self) -> None:
        for name in BUILT_INS:
            self._cache[name] = synthesise(name)
        try:
            import sounddevice as sd  # noqa: PLC0415 — optional, checked at runtime

            sd.check_output_settings(samplerate=SAMPLE_RATE, channels=1)
            self._sd = sd
        except Exception as exc:  # no device, no library, or a locked device
            self._sd = None
            self.last_error = str(exc)

    # ---------------------------------------------------------------- backends

    @property
    def backend(self) -> str:
        if self._sd is not None:
            return "sounddevice"
        if shutil.which("afplay"):
            return "afplay"
        return "none"

    def available(self) -> bool:
        return self.backend != "none"

    def sounds(self) -> list[str]:
        return list(BUILT_INS) + sorted(self.custom)

    # ------------------------------------------------------------------- play

    def play(self, name: str = "chirp", volume: float | None = None) -> PlaybackResult:
        started = time.perf_counter()
        gain = self.volume if volume is None else volume
        try:
            samples = self._samples(name)
        except Exception as exc:
            self.last_error = str(exc)
            return PlaybackResult(False, error=str(exc))

        buf = np.clip(samples * float(np.clip(gain, 0.0, 1.0)), -1.0, 1.0)
        with self._lock:
            if self._sd is not None:
                try:
                    self._sd.play(buf, SAMPLE_RATE, blocking=False)
                    return PlaybackResult(
                        True, (time.perf_counter() - started) * 1e3, "sounddevice"
                    )
                except Exception as exc:
                    self.last_error = str(exc)
                    self._sd = None  # fall through to afplay for the rest of the run
            if shutil.which("afplay"):
                path = self._to_wav(name, buf)
                try:
                    subprocess.Popen(
                        ["afplay", str(path)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    return PlaybackResult(True, (time.perf_counter() - started) * 1e3, "afplay")
                except Exception as exc:
                    self.last_error = str(exc)
        msg = self.last_error or "no audio output device is available"
        return PlaybackResult(False, error=msg, backend="none")

    def self_test(self) -> PlaybackResult:
        """Play a silent buffer to prove the device really accepts audio (heartbeat)."""
        if self._sd is None:
            return PlaybackResult(
                self.backend != "none",
                backend=self.backend,
                error="" if self.backend != "none" else "no audio output device",
            )
        started = time.perf_counter()
        try:
            self._sd.play(np.zeros(256, np.float32), SAMPLE_RATE, blocking=True)
            return PlaybackResult(True, (time.perf_counter() - started) * 1e3, "sounddevice")
        except Exception as exc:
            self.last_error = str(exc)
            return PlaybackResult(False, error=str(exc), backend="sounddevice")

    # -------------------------------------------------------------- internals

    def _samples(self, name: str) -> np.ndarray:
        if name in self._cache:
            return self._cache[name]
        path = self.custom.get(name)
        if path is None:
            raise KeyError(f"unknown sound {name!r}")
        self._cache[name] = _read_wav(path)
        return self._cache[name]

    def _to_wav(self, name: str, buf: np.ndarray) -> Path:
        path = Path.home() / "Library" / "Caches" / "SurfaceGuard" / f"{name}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        pcm = (buf * 32767).astype("<i2")
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm.tobytes())
        return path


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        frames = w.readframes(w.getnframes())
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            data = data.reshape(-1, w.getnchannels()).mean(axis=1)
    return data
