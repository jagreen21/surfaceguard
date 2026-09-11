"""Eufy camera video, over the supervised bridge.

The protocol lives in :mod:`surfaceguard.bridge.client`; this file is only the
:class:`CameraSource` on top of it. H.264 is decoded in-process with PyAV rather
than by piping to an ffmpeg binary — one less thing to ship, and one less process
between the camera and the latency budget.

Two limits this layer reports rather than hides:

* Eufy's pan/tilt command is **directional**, not absolute. ``move_to`` dead-reckons
  from timed nudges and ``Capabilities.reports_angles`` stays False, which is
  exactly why registration works from image features (D1).
* The bridge relays no capture timestamp, so per-frame latency cannot be measured
  in-band. ``tools/phase0.py`` measures it out of band and says so.
"""

from __future__ import annotations

import base64
import queue
import threading
import time

import numpy as np

from ...bridge.client import BridgeClient, BridgeError, DriverPhase, is_pan_tilt
from .base import Capabilities, CameraSource, Frame, PetEvent, SourceError

DIR_ROTATE_RIGHT, DIR_ROTATE_LEFT, DIR_ROTATE_UP, DIR_ROTATE_DOWN = 0, 1, 2, 3

# Seeded, then measured per model by tools/phase0.py.
DEGREES_PER_NUDGE = 5.0
NUDGE_INTERVAL_S = 0.25

PET_PROPERTIES = ("petDetected", "petDetection")


class EufyBridgeCamera(CameraSource):
    def __init__(
        self,
        client: BridgeClient,
        serial: str,
        model: str = "",
        name: str = "",
        owns_client: bool = False,
    ) -> None:
        super().__init__()
        self.client = client
        self.serial = serial
        self.model = model
        self.device_name = name
        self.owns_client = owns_client

        self._decoder = None
        self._frames: "queue.Queue[Frame]" = queue.Queue(maxsize=3)
        self._stop = threading.Event()
        self._properties: dict = {}
        self._pan = 0.0
        self._tilt = 0.0
        self.degrees_per_nudge = DEGREES_PER_NUDGE
        self._first_video_at: float | None = None
        self._requested_at: float | None = None
        self._decode_errors = 0
        self.last_error = ""

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> None:
        if not self.client.connected:
            self.client.connect()
        if self.client.driver.phase is not DriverPhase.CONNECTED:
            raise SourceError(
                "Not signed in to Eufy yet. Open Settings and sign in to the camera account."
            )
        self._stop.clear()
        self._open_decoder()
        try:
            self._properties = self.client.device_properties(self.serial)
        except BridgeError as exc:
            self._properties = {}
            self.last_error = str(exc)

        self.client.add_handler(self._on_event)
        self._requested_at = time.monotonic()
        try:
            self.client.send_wait("device.start_livestream", serialNumber=self.serial, timeout=25.0)
        except BridgeError as exc:
            raise SourceError(
                f"The camera would not start streaming. {exc} "
                "It may be busy in the Eufy app — close that and try again."
            ) from exc

    def stop(self) -> None:
        self._stop.set()
        try:
            if self.client.connected:
                self.client.send("device.stop_livestream", serialNumber=self.serial)
        except Exception:
            pass
        self._decoder = None
        # Stop listening, or a camera that has been swapped out keeps receiving
        # frames on the shared connection.
        self.client.remove_handler(self._on_event)
        if self.owns_client:
            self.client.close()

    def _open_decoder(self) -> None:
        try:
            import av  # noqa: PLC0415 — optional at import time, required to stream

            av.logging.set_level(av.logging.PANIC)
            self._decoder = av.CodecContext.create("h264", "r")
        except Exception as exc:
            raise SourceError(
                "The video decoder is missing from this install. Reinstall Surface Guard. "
                f"({exc})"
            ) from exc

    # ------------------------------------------------------------- capabilities

    @property
    def capabilities(self) -> Capabilities:
        keys = set(self._properties)
        ptz = is_pan_tilt(self.model) or "panAndTilt" in keys or "rotationSpeed" in keys
        return Capabilities(
            name=self.device_name or f"Eufy {self.model or 'camera'}".strip(),
            model=self.model,
            has_ptz=ptz,
            # Eufy exposes no absolute pan/tilt read-out over this bridge.
            reports_angles=False,
            has_speaker=bool(keys & {"speaker", "speakerVolume"}),
            emits_pet_events=bool(keys & set(PET_PROPERTIES)),
            pan_range=(-170.0, 170.0) if ptz else None,
            tilt_range=(-35.0, 35.0) if ptz else None,
            notes=[
                "Angles are dead-reckoned from directional nudges, not read from the "
                "camera; nothing downstream may treat them as measurements.",
                "The bridge relays no capture timestamp, so frame latency is measured "
                "out of band by tools/phase0.py.",
            ],
        )

    @property
    def properties(self) -> dict:
        return dict(self._properties)

    @property
    def stream_startup_s(self) -> float | None:
        if self._first_video_at is None or self._requested_at is None:
            return None
        return self._first_video_at - self._requested_at

    # ------------------------------------------------------------------- video

    def _on_event(self, event: dict) -> None:
        if event.get("serialNumber") not in (None, self.serial):
            return
        name = event.get("event")
        if name == "livestream video data":
            self._feed(event.get("buffer"))
        elif name == "property changed" and event.get("name") in PET_PROPERTIES:
            if event.get("value"):
                self._publish_pet_event(
                    PetEvent(time.monotonic(), device=self.serial, raw=event)
                )
        elif name == "livestream stopped":
            self.last_error = "The camera stopped the video stream."

    def _feed(self, buffer: object) -> None:
        chunk = _decode_buffer(buffer)
        if not chunk or self._decoder is None or self._stop.is_set():
            return
        try:
            for packet in self._decoder.parse(chunk):
                for frame in self._decoder.decode(packet):
                    self._emit(frame)
        except Exception:
            # A corrupt packet mid-stream is normal over P2P; drop it and carry on.
            # A sustained run of them is what the heartbeat's video check catches.
            self._decode_errors += 1

    def _emit(self, av_frame) -> None:
        image = av_frame.to_ndarray(format="bgr24")
        now = time.monotonic()
        if self._first_video_at is None:
            self._first_video_at = now
        _put_newest(self._frames, Frame(
            image=image,
            seq=self._next_seq(),
            ts_received=now,
            ts_capture=None,
            pan=None,
            tilt=None,
        ))

    def read(self, timeout: float = 2.0) -> Frame | None:
        deadline = time.monotonic() + timeout
        newest: Frame | None = None
        try:
            newest = self._frames.get(timeout=max(0.0, deadline - time.monotonic()))
        except queue.Empty:
            return None
        # Always judge the freshest frame: a backlog is worse than a dropped frame.
        while True:
            try:
                newest = self._frames.get_nowait()
            except queue.Empty:
                return newest

    # --------------------------------------------------------------------- ptz

    def move_to(self, pan: float, tilt: float = 0.0, settle_s: float = 1.5) -> bool:
        caps = self.capabilities
        if not caps.has_ptz:
            return False
        lo, hi = caps.pan_range or (-170.0, 170.0)
        tlo, thi = caps.tilt_range or (-35.0, 35.0)
        for current, target, pos_dir, neg_dir, setter in (
            (self._pan, float(np.clip(pan, lo, hi)), DIR_ROTATE_RIGHT, DIR_ROTATE_LEFT, "_pan"),
            (self._tilt, float(np.clip(tilt, tlo, thi)), DIR_ROTATE_UP, DIR_ROTATE_DOWN, "_tilt"),
        ):
            delta = target - current
            nudges = int(round(abs(delta) / self.degrees_per_nudge))
            direction = pos_dir if delta > 0 else neg_dir
            for _ in range(min(nudges, 80)):
                if self._stop.is_set():
                    return False
                try:
                    self.client.send(
                        "device.pan_and_tilt", serialNumber=self.serial, direction=direction
                    )
                except BridgeError:
                    return False
                time.sleep(NUDGE_INTERVAL_S)
            setattr(self, setter, target)
        if settle_s:
            time.sleep(settle_s)
        return True

    def current_angles(self) -> tuple[float | None, float | None]:
        # Deliberately None: an open-loop estimate is not a measurement, and
        # registration must never be handed one as if it were.
        return (None, None)

    def dead_reckoned_angles(self) -> tuple[float, float]:
        return (self._pan, self._tilt)

    def play_sound_on_camera(self, name: str = "default") -> bool:
        if not self.capabilities.has_speaker:
            return False
        for command, payload in (
            ("device.quick_response", {"voiceId": 0}),
            ("device.trigger_alarm", {"seconds": 2}),
        ):
            try:
                self.client.send_wait(command, serialNumber=self.serial, timeout=6.0, **payload)
                return True
            except BridgeError as exc:
                self.last_error = str(exc)
        return False


def _decode_buffer(buffer: object) -> bytes:
    """The bridge sends Node Buffers as {'type':'Buffer','data':[...]} or base64."""
    if isinstance(buffer, dict):
        data = buffer.get("data")
        if isinstance(data, list):
            return bytes(data)
        if isinstance(data, str):
            return base64.b64decode(data)
    if isinstance(buffer, str):
        return base64.b64decode(buffer)
    if isinstance(buffer, (bytes, bytearray)):
        return bytes(buffer)
    return b""


def _put_newest(q: "queue.Queue[Frame]", frame: Frame) -> None:
    try:
        q.put_nowait(frame)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(frame)
        except queue.Full:
            pass
