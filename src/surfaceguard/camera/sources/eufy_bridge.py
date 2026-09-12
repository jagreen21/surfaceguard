"""Eufy camera video, over the supervised bridge.

The protocol lives in :mod:`surfaceguard.bridge.client`; this file is only the
:class:`CameraSource` on top of it. H.264/H.265 is decoded in-process with PyAV rather
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
from ...logging_setup import get as get_logger
from .base import Capabilities, CameraSource, Frame, PetEvent, SourceError

logger = get_logger("camera.eufy")

DIR_ROTATE_RIGHT, DIR_ROTATE_LEFT, DIR_ROTATE_UP, DIR_ROTATE_DOWN = 0, 1, 2, 3

# Seeded, then measured per model by tools/phase0.py.
# A camera needs a moment to release a stream before it will start another.
RELEASE_SETTLE_S = 2.5

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
        self._decoder_codec = ""
        self._reported_codec = ""
        self._sniffed_codec = ""
        self._codec_override = ""
        self._codec_fallback_attempted = False
        self._first_chunk_logged = False
        self._video_chunks = 0
        self._video_bytes = 0
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
        self._first_video_at = None
        self._decode_errors = 0
        self._decoder_codec = ""
        self._reported_codec = ""
        self._sniffed_codec = ""
        self._codec_override = ""
        self._codec_fallback_attempted = False
        self._first_chunk_logged = False
        self._video_chunks = 0
        self._video_bytes = 0
        self.last_error = ""
        while not self._frames.empty():
            try:
                self._frames.get_nowait()
            except queue.Empty:
                break
        # Prove the bundled PyAV install can create a decoder now. The bridge
        # reports the actual codec with its first chunk; H.265 then replaces it.
        self._open_decoder("h264")
        try:
            self._properties = self.client.device_properties(self.serial)
        except BridgeError as exc:
            self._properties = {}
            self.last_error = str(exc)

        self.client.add_handler(self._on_event)
        self._requested_at = time.monotonic()
        try:
            self._start_livestream()
        except BridgeError as exc:
            # One retry after telling the camera to stop: the usual reason a start
            # is refused is a stream this app itself left open a moment ago.
            try:
                self.client.send_wait(
                    "device.stop_livestream", serialNumber=self.serial, timeout=6.0
                )
            except BridgeError:
                pass
            time.sleep(RELEASE_SETTLE_S)
            try:
                self._start_livestream()
            except BridgeError:
                raise SourceError(
                    "This camera would not start streaming. It is usually because it "
                    "is open in the Eufy app on a phone — close it there and try "
                    f"again. ({exc})"
                ) from exc

    def _start_livestream(self) -> None:
        logger.info("requesting livestream for %s (%s)", self.serial, self.model or "unknown model")
        self.client.send_wait(
            "device.start_livestream", serialNumber=self.serial, timeout=30.0
        )
        logger.info("bridge accepted livestream request for %s", self.serial)

    def stop(self) -> None:
        self._stop.set()
        try:
            if self.client.connected:
                # Wait for the bridge to confirm, and give the camera a moment to
                # actually let go. Fire-and-forget left the stream open on the
                # camera's side, so choosing a different camera and coming back was
                # met with "it may be busy in the Eufy app" — when the thing holding
                # it was this app.
                try:
                    self.client.send_wait(
                        "device.stop_livestream", serialNumber=self.serial, timeout=6.0
                    )
                except BridgeError:
                    self.client.send("device.stop_livestream", serialNumber=self.serial)
                time.sleep(RELEASE_SETTLE_S)
        except Exception:
            pass
        self._decoder = None
        # Stop listening, or a camera that has been swapped out keeps receiving
        # frames on the shared connection.
        self.client.remove_handler(self._on_event)
        if self.owns_client:
            self.client.close()

    def _open_decoder(self, codec: str) -> None:
        try:
            import av  # noqa: PLC0415 — optional at import time, required to stream

            av.logging.set_level(av.logging.PANIC)
            self._decoder = av.CodecContext.create(codec, "r")
            self._decoder_codec = codec
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
            self._feed(event.get("buffer"), event.get("metadata"))
        elif name == "livestream started":
            logger.info("livestream started for %s", self.serial)
        elif name == "property changed" and event.get("name") in PET_PROPERTIES:
            if event.get("value"):
                self._publish_pet_event(
                    PetEvent(time.monotonic(), device=self.serial, raw=event)
                )
        elif name == "livestream stopped":
            self.last_error = "The camera stopped the video stream."
            logger.warning("livestream stopped for %s", self.serial)

    def _feed(self, buffer: object, metadata: object = None) -> None:
        chunk = _decode_buffer(buffer)
        if not chunk or self._stop.is_set():
            return
        self._video_chunks += 1
        self._video_bytes += len(chunk)
        details = metadata if isinstance(metadata, dict) else {}
        reported = _decoder_name(details.get("videoCodec"))
        sniffed = _sniff_decoder_name(chunk)
        self._reported_codec = reported
        if sniffed:
            self._sniffed_codec = sniffed
        # Some T8417 firmware reports streamType=1/H264 even when its Annex-B
        # payload contains HEVC parameter sets. Conclusive bytes beat metadata.
        codec = self._codec_override or sniffed or self._sniffed_codec or reported
        if codec != self._decoder_codec:
            logger.info(
                "video codec selected for %s: %s reported=%s sniffed=%s",
                self.serial, codec, reported, sniffed or "unknown",
            )
            try:
                self._open_decoder(codec)
            except SourceError as exc:
                self.last_error = str(exc)
                logger.error("could not open %s decoder for %s: %s", codec, self.serial, exc)
                return
        if self._decoder is None:
            return
        if not self._first_chunk_logged:
            self._first_chunk_logged = True
            logger.info(
                "first video chunk for %s: %d bytes codec=%s size=%sx%s fps=%s",
                self.serial, len(chunk), codec, details.get("videoWidth", "?"),
                details.get("videoHeight", "?"), details.get("videoFPS", "?"),
            )
            logger.info("first video bytes for %s: %s", self.serial, chunk[:24].hex())
        try:
            self._decode_chunk(chunk)
        except Exception as exc:
            # A corrupt packet mid-stream is normal over P2P; drop it and carry on.
            # A sustained run of them is what the heartbeat's video check catches.
            self._decode_errors += 1
            if self._decode_errors == 1 or self._decode_errors % 100 == 0:
                logger.warning(
                    "video decode error %d for %s using %s: %s",
                    self._decode_errors, self.serial, self._decoder_codec, exc,
                )
            if (
                self._first_video_at is None
                and not self._codec_fallback_attempted
                and not self._sniffed_codec
            ):
                self._codec_fallback_attempted = True
                alternate = "hevc" if self._decoder_codec == "h264" else "h264"
                logger.warning(
                    "trying alternate %s decoder for %s after %s rejected the payload",
                    alternate, self.serial, self._decoder_codec,
                )
                try:
                    self._open_decoder(alternate)
                    self._decode_chunk(chunk)
                    # Retain the alternate even when later metadata repeats the
                    # incorrect codec label.
                    self._codec_override = alternate
                except Exception as alternate_exc:
                    logger.warning(
                        "alternate %s decoder also rejected video for %s: %s",
                        alternate, self.serial, alternate_exc,
                    )

    def _decode_chunk(self, chunk: bytes) -> None:
        if self._decoder is None:
            return
        for packet in self._decoder.parse(chunk):
            for frame in self._decoder.decode(packet):
                self._emit(frame)

    def video_diagnostics(self) -> dict[str, object]:
        """Image-free counters suitable for a redacted support report."""
        return {
            "reported_codec": self._reported_codec or "unknown",
            "sniffed_codec": self._sniffed_codec or "unknown",
            "decoder": self._decoder_codec or "none",
            "codec_override": self._codec_override or "none",
            "chunks": self._video_chunks,
            "bytes": self._video_bytes,
            "decode_errors": self._decode_errors,
            "frames": self._seq,
            "stream_error": self.last_error or "none",
        }

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


def _decoder_name(value: object) -> str:
    """Translate the bridge's VideoCodec enum name to PyAV's decoder name."""
    if value == 1:
        return "hevc"
    name = str(value or "H264").upper().replace(".", "")
    return "hevc" if name in {"H265", "HEVC"} else "h264"


def _sniff_decoder_name(chunk: bytes) -> str:
    """Identify Annex-B AVC/HEVC parameter sets without trusting metadata."""
    limit = min(len(chunk), 512)
    for index in range(max(0, limit - 3)):
        if chunk[index:index + 4] == b"\x00\x00\x00\x01":
            header_at = index + 4
        elif chunk[index:index + 3] == b"\x00\x00\x01":
            header_at = index + 3
        else:
            continue
        if header_at >= len(chunk):
            continue
        header = chunk[header_at]
        h264_type = header & 0x1F
        h265_type = (header >> 1) & 0x3F
        if h265_type in {32, 33, 34}:  # VPS/SPS/PPS
            return "hevc"
        if h264_type in {7, 8}:  # SPS/PPS
            return "h264"
    return ""


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
