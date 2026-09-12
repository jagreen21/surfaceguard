"""The replaceable camera interface (decision D5).

Every camera the app can use — the unofficial Eufy bridge, a generic RTSP feed, a
recorded clip, or the synthetic room used for tests and demos — implements this.
The Eufy integration is unofficial and will break; the point of this seam is that
when it does, only one file is wrong.
"""

from __future__ import annotations

import queue
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


class SourceError(RuntimeError):
    """Raised for a failure the user needs to be told about in plain words."""


@dataclass(frozen=True)
class Frame:
    """One image, with everything needed to reason about latency and pose."""

    image: np.ndarray
    seq: int
    ts_received: float                  # time.monotonic() when we got the pixels
    ts_capture: float | None = None     # camera clock, if the source reports one
    pan: float | None = None
    tilt: float | None = None

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return (w, h)

    def age_s(self, now: float | None = None) -> float:
        return (now if now is not None else time.monotonic()) - self.ts_received


@dataclass(frozen=True)
class PetEvent:
    """A camera-side 'pet detected' notification: a watchdog signal, not a trigger."""

    ts_received: float
    device: str = ""
    raw: dict = field(default_factory=dict)


@dataclass
class Capabilities:
    """What a source can actually do — answers Phase 0's open questions directly."""

    name: str
    model: str = ""
    has_ptz: bool = False
    reports_angles: bool = False
    has_speaker: bool = False
    emits_pet_events: bool = False
    auto_tracks_motion: bool = False
    pan_range: tuple[float, float] | None = None
    tilt_range: tuple[float, float] | None = None
    notes: list[str] = field(default_factory=list)

    def as_rows(self) -> list[tuple[str, str]]:
        def yn(v: bool) -> str:
            return "yes" if v else "no"

        rows = [
            ("source", self.name),
            ("model", self.model or "unknown"),
            ("pan/tilt control", yn(self.has_ptz)),
            ("reports angles", yn(self.reports_angles)),
            ("camera speaker reachable", yn(self.has_speaker)),
            ("emits pet events", yn(self.emits_pet_events)),
            ("camera follows motion", yn(self.auto_tracks_motion)),
        ]
        if self.pan_range:
            rows.append(("pan range", f"{self.pan_range[0]:.0f} to {self.pan_range[1]:.0f} deg"))
        if self.tilt_range:
            rows.append(("tilt range", f"{self.tilt_range[0]:.0f} to {self.tilt_range[1]:.0f} deg"))
        return rows


class CameraSource(ABC):
    """A warm, continuously-running video source.

    Implementations must keep the stream open between reads (decision D6): the
    deterrent latency budget has no room for per-event stream startup.
    """

    def __init__(self) -> None:
        self._events: "queue.Queue[PetEvent]" = queue.Queue(maxsize=64)
        self._seq = 0

    # ------------------------------------------------------------- lifecycle

    @abstractmethod
    def start(self) -> None:
        ...

    @abstractmethod
    def stop(self) -> None:
        ...

    @abstractmethod
    def read(self, timeout: float = 2.0) -> Frame | None:
        """Return the newest frame, or None if the stream produced nothing in time."""

    @property
    @abstractmethod
    def capabilities(self) -> Capabilities:
        ...

    def __enter__(self) -> "CameraSource":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -------------------------------------------------------------- optional

    def move_to(self, pan: float, tilt: float = 0.0, settle_s: float = 1.5) -> bool:
        """Point the camera. Returns False when the source has no PTZ."""
        return False

    def current_angles(self) -> tuple[float | None, float | None]:
        return (None, None)

    def suspend_auto_tracking(self) -> object:
        """Pause camera-owned steering for calibration; return a restore token."""
        return None

    def restore_auto_tracking(self, token: object) -> None:
        """Restore the state returned by :meth:`suspend_auto_tracking`."""

    def play_sound_on_camera(self, name: str = "default") -> bool:
        """Play a deterrent through the camera's own speaker, if it has one.

        This is the single most valuable capability to discover in Phase 0: it
        removes the laptop from the critical path entirely (risk R1).
        """
        return False

    def next_pet_event(self, timeout: float = 0.0) -> PetEvent | None:
        try:
            return self._events.get(timeout=timeout) if timeout else self._events.get_nowait()
        except queue.Empty:
            return None

    def _publish_pet_event(self, event: PetEvent) -> None:
        try:
            self._events.put_nowait(event)
        except queue.Full:
            pass

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq
