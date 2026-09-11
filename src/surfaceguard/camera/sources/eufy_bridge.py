"""Eufy camera over the unofficial eufy-security-ws bridge.

This is the replaceable implementation behind :class:`CameraSource` (D5). It is
deliberately the only file that knows the bridge's protocol, its Buffer encoding,
or that H.264 has to go through ffmpeg to become pixels.

Two things this layer cannot paper over, and which the Phase 0 harness therefore
reports rather than assumes:

* Eufy's pan/tilt command is **directional**, not absolute — there is no "go to
  32 degrees". ``move_to`` dead-reckons from timed nudges and
  ``Capabilities.reports_angles`` is False, which is exactly why registration
  works from image features rather than trusting angles (D1).
* The bridge exposes a camera-side audio path only on some models and firmware, so
  :meth:`play_sound_on_camera` probes and reports instead of promising.
"""

from __future__ import annotations

import base64
import json
import queue
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field

import numpy as np

from .base import Capabilities, CameraSource, Frame, PetEvent, SourceError

SCHEMA_VERSION = 21
DEFAULT_URL = "ws://127.0.0.1:3000"

# Directional pan/tilt: the bridge's own enum.
DIR_ROTATE_RIGHT, DIR_ROTATE_LEFT, DIR_ROTATE_UP, DIR_ROTATE_DOWN = 0, 1, 2, 3

# Measured empirically per model by the Phase 0 harness; this is only the seed.
DEGREES_PER_NUDGE = 5.0

PET_PROPERTIES = ("petDetected", "petDetection")


@dataclass
class _Pending:
    event: threading.Event = field(default_factory=threading.Event)
    result: dict | None = None


class EufyBridgeCamera(CameraSource):
    """Live video + events from one Eufy device via eufy-security-ws."""

    def __init__(
        self,
        url: str = DEFAULT_URL,
        serial: str | None = None,
        frame_size: tuple[int, int] = (960, 540),
        connect_timeout: float = 10.0,
    ) -> None:
        super().__init__()
        self.url = url
        self.serial = serial
        self.frame_w, self.frame_h = frame_size
        self.connect_timeout = connect_timeout

        self._ws = None
        self._ffmpeg: subprocess.Popen | None = None
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._frames: "queue.Queue[Frame]" = queue.Queue(maxsize=4)
        self._pending: dict[str, _Pending] = {}
        self._pending_lock = threading.Lock()

        self._device: dict = {}
        self._properties: dict = {}
        self._caps = Capabilities(name="Eufy bridge")
        self._pan = 0.0          # dead-reckoned, not reported by the camera
        self._tilt = 0.0
        self.degrees_per_nudge = DEGREES_PER_NUDGE
        self.last_error = ""
        self._first_video_at: float | None = None
        self._livestream_requested_at: float | None = None

    # ---------------------------------------------------------------- connect

    def start(self) -> None:
        from websockets.sync.client import connect  # lazy: keeps import cost off the UI

        self._stop.clear()
        try:
            self._ws = connect(self.url, open_timeout=self.connect_timeout, max_size=None)
        except Exception as exc:
            raise SourceError(
                f"Could not reach the camera bridge at {self.url}. Is eufy-security-ws "
                f"running? ({exc})"
            ) from exc

        self._spawn(self._reader_loop, "sg-eufy-ws")
        self._send_wait("set_api_schema", schemaVersion=SCHEMA_VERSION)
        state = self._send_wait("start_listening")
        self._adopt_state(state)
        self._start_ffmpeg()
        self._livestream_requested_at = time.monotonic()
        self._send_wait("device.start_livestream", serialNumber=self.serial)

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._ws is not None and self.serial:
                self._send("device.stop_livestream", serialNumber=self.serial)
        except Exception:
            pass
        for closer in (self._close_ffmpeg, self._close_ws):
            try:
                closer()
            except Exception:
                pass
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()

    def _close_ws(self) -> None:
        if self._ws is not None:
            self._ws.close()
            self._ws = None

    def _close_ffmpeg(self) -> None:
        if self._ffmpeg is not None:
            if self._ffmpeg.stdin:
                self._ffmpeg.stdin.close()
            self._ffmpeg.terminate()
            try:
                self._ffmpeg.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._ffmpeg.kill()
            self._ffmpeg = None

    def _spawn(self, target, name: str) -> None:
        t = threading.Thread(target=target, name=name, daemon=True)
        t.start()
        self._threads.append(t)

    # ------------------------------------------------------------- capabilities

    def _adopt_state(self, state: dict) -> None:
        devices = (state or {}).get("state", {}).get("devices", [])
        if not devices:
            raise SourceError(
                "The bridge connected but reported no cameras. Check the Eufy account "
                "credentials in the bridge's configuration."
            )
        if self.serial:
            match = next((d for d in devices if d.get("serialNumber") == self.serial), None)
            if match is None:
                found = ", ".join(str(d.get("serialNumber")) for d in devices)
                raise SourceError(f"Camera {self.serial} is not on this bridge. Found: {found}")
            self._device = match
        else:
            self._device = devices[0]
            self.serial = self._device.get("serialNumber")

        try:
            props = self._send_wait("device.get_properties", serialNumber=self.serial)
            self._properties = (props or {}).get("properties", {}) or {}
        except Exception as exc:
            self._properties = {}
            self.last_error = f"could not read device properties: {exc}"

        keys = set(self._properties)
        model = str(self._device.get("model") or self._properties.get("model") or "")
        has_ptz = any(k in keys for k in ("panAndTilt", "rotationSpeed")) or _looks_ptz(model)
        self._caps = Capabilities(
            name=f"Eufy {model or 'camera'}".strip(),
            model=model,
            has_ptz=has_ptz,
            # Eufy exposes no absolute pan/tilt read-out over this bridge.
            reports_angles=False,
            has_speaker=any(k in keys for k in ("speaker", "speakerVolume")),
            emits_pet_events=any(k in keys for k in PET_PROPERTIES),
            pan_range=(-170.0, 170.0) if has_ptz else None,
            tilt_range=(-35.0, 35.0) if has_ptz else None,
            notes=[
                "Angles are dead-reckoned from directional nudges, not read from "
                "the camera; registration must not depend on them.",
            ],
        )

    @property
    def capabilities(self) -> Capabilities:
        return self._caps

    @property
    def properties(self) -> dict:
        return dict(self._properties)

    @property
    def stream_startup_s(self) -> float | None:
        """Seconds from asking for the livestream to the first decoded frame."""
        if self._first_video_at is None or self._livestream_requested_at is None:
            return None
        return self._first_video_at - self._livestream_requested_at

    # ---------------------------------------------------------------- commands

    def _send(self, command: str, **payload) -> str:
        if self._ws is None:
            raise SourceError("Not connected to the camera bridge")
        message_id = uuid.uuid4().hex[:12]
        self._ws.send(json.dumps({"messageId": message_id, "command": command, **payload}))
        return message_id

    def _send_wait(self, command: str, timeout: float = 12.0, **payload) -> dict:
        pending = _Pending()
        with self._pending_lock:
            message_id = self._send(command, **payload)
            self._pending[message_id] = pending
        if not pending.event.wait(timeout):
            with self._pending_lock:
                self._pending.pop(message_id, None)
            raise SourceError(f"The bridge did not answer '{command}' within {timeout:.0f}s")
        result = pending.result or {}
        if not result.get("success", True):
            raise SourceError(f"The bridge refused '{command}': {result.get('errorCode', result)}")
        return result

    def move_to(self, pan: float, tilt: float = 0.0, settle_s: float = 1.5) -> bool:
        """Dead-reckon towards an angle with directional nudges.

        The camera reports no absolute position, so this is open-loop and drifts.
        Registration is what makes that acceptable: the app never trusts the angle
        it thinks it is at, only the features it can see.
        """
        if not self._caps.has_ptz:
            return False
        lo, hi = self._caps.pan_range or (-170.0, 170.0)
        target_pan = float(np.clip(pan, lo, hi))
        tlo, thi = self._caps.tilt_range or (-35.0, 35.0)
        target_tilt = float(np.clip(tilt, tlo, thi))

        for axis, current, target, pos_dir, neg_dir in (
            ("pan", self._pan, target_pan, DIR_ROTATE_RIGHT, DIR_ROTATE_LEFT),
            ("tilt", self._tilt, target_tilt, DIR_ROTATE_UP, DIR_ROTATE_DOWN),
        ):
            delta = target - current
            nudges = int(round(abs(delta) / self.degrees_per_nudge))
            direction = pos_dir if delta > 0 else neg_dir
            for _ in range(min(nudges, 80)):
                try:
                    self._send("device.pan_and_tilt", serialNumber=self.serial, direction=direction)
                except SourceError:
                    return False
                time.sleep(0.25)
            if axis == "pan":
                self._pan = target_pan
            else:
                self._tilt = target_tilt
        if settle_s:
            time.sleep(settle_s)
        return True

    def current_angles(self) -> tuple[float | None, float | None]:
        # Deliberately None, not the dead-reckoned guess: callers must not treat
        # an open-loop estimate as a measurement.
        return (None, None)

    def dead_reckoned_angles(self) -> tuple[float, float]:
        return (self._pan, self._tilt)

    def play_sound_on_camera(self, name: str = "default") -> bool:
        """Try the camera's own speaker. Returns False if this model has none."""
        if not self._caps.has_speaker:
            return False
        for command, payload in (
            ("device.quick_response", {"voiceId": 0}),
            ("device.trigger_alarm", {"seconds": 2}),
        ):
            try:
                self._send_wait(command, serialNumber=self.serial, timeout=6.0, **payload)
                return True
            except SourceError as exc:
                self.last_error = str(exc)
        return False

    # ------------------------------------------------------------------- video

    def _start_ffmpeg(self) -> None:
        self._ffmpeg = subprocess.Popen(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-fflags", "nobuffer", "-flags", "low_delay",
                "-probesize", "32", "-analyzeduration", "0",
                "-f", "h264", "-i", "pipe:0",
                "-vf", f"scale={self.frame_w}:{self.frame_h}",
                "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1",
            ],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        self._spawn(self._video_loop, "sg-eufy-video")

    def _video_loop(self) -> None:
        assert self._ffmpeg is not None and self._ffmpeg.stdout is not None
        nbytes = self.frame_w * self.frame_h * 3
        out = self._ffmpeg.stdout
        while not self._stop.is_set():
            buf = out.read(nbytes)
            if not buf or len(buf) < nbytes:
                break
            now = time.monotonic()
            if self._first_video_at is None:
                self._first_video_at = now
            frame = Frame(
                image=np.frombuffer(buf, np.uint8).reshape(self.frame_h, self.frame_w, 3).copy(),
                seq=self._next_seq(),
                ts_received=now,
                # The bridge relays no capture timestamp, so end-to-end latency
                # has to be measured out of band (see tools/phase0.py).
                ts_capture=None,
                pan=None,
                tilt=None,
            )
            _put_newest(self._frames, frame)

    def read(self, timeout: float = 2.0) -> Frame | None:
        deadline = time.monotonic() + timeout
        newest: Frame | None = None
        while True:
            try:
                newest = self._frames.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                return newest
            # Drain: always judge the freshest frame, never a backlog.
            while True:
                try:
                    newest = self._frames.get_nowait()
                except queue.Empty:
                    return newest

    # ------------------------------------------------------------------ events

    def _reader_loop(self) -> None:
        while not self._stop.is_set() and self._ws is not None:
            try:
                raw = self._ws.recv(timeout=1.0)
            except TimeoutError:
                continue
            except Exception:
                break
            try:
                message = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            self._dispatch(message)

    def _dispatch(self, message: dict) -> None:
        kind = message.get("type")
        if kind in ("result", "version"):
            with self._pending_lock:
                pending = self._pending.pop(message.get("messageId", ""), None)
            if pending is not None:
                pending.result = message
                pending.event.set()
            return
        if kind != "event":
            return

        event = message.get("event", {})
        name = event.get("event")
        if name in ("livestream video data", "livestream audio data"):
            if name == "livestream video data":
                self._feed_video(event.get("buffer"))
            return
        if name == "property changed" and event.get("name") in PET_PROPERTIES:
            if event.get("value"):
                self._publish_pet_event(PetEvent(
                    time.monotonic(), device=str(event.get("serialNumber") or ""), raw=event
                ))

    def _feed_video(self, buffer: object) -> None:
        chunk = _decode_buffer(buffer)
        if not chunk or self._ffmpeg is None or self._ffmpeg.stdin is None:
            return
        try:
            self._ffmpeg.stdin.write(chunk)
            self._ffmpeg.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass  # the decoder died; the heartbeat's video check will notice


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
    """Keep the queue shallow: a stale frame is worse than a dropped one."""
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


def _looks_ptz(model: str) -> bool:
    # T8417 is the Indoor Cam E30; the other prefixes are its pan/tilt siblings.
    return any(model.upper().startswith(p) for p in ("T8410", "T8414", "T8416", "T8417", "T8441"))
