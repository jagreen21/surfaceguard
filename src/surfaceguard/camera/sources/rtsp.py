"""Generic RTSP / HTTP / local-file video via OpenCV.

The fallback when the unofficial bridge breaks, and the way a user with a
different camera gets the same app. No PTZ, so the scan scheduler will report a
single fixed view and coverage accordingly.
"""

from __future__ import annotations

import queue
import threading
import time

import cv2

from .base import Capabilities, CameraSource, Frame, SourceError


class RtspCamera(CameraSource):
    def __init__(self, url: str, frame_size: tuple[int, int] | None = None, name: str = "") -> None:
        super().__init__()
        self.url = url
        self.frame_size = frame_size
        self._name = name or "RTSP camera"
        self._cap: cv2.VideoCapture | None = None
        self._frames: "queue.Queue[Frame]" = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._cap = cv2.VideoCapture(self.url)
        if not self._cap.isOpened():
            raise SourceError(f"Could not open the video stream at {self.url}")
        self._thread = threading.Thread(target=self._loop, name="sg-rtsp", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _loop(self) -> None:
        assert self._cap is not None
        while not self._stop.is_set():
            ok, image = self._cap.read()
            if not ok:
                time.sleep(0.2)
                continue
            if self.frame_size:
                image = cv2.resize(image, self.frame_size)
            frame = Frame(image, self._next_seq(), time.monotonic())
            try:
                self._frames.put_nowait(frame)
            except queue.Full:
                try:
                    self._frames.get_nowait()
                    self._frames.put_nowait(frame)
                except (queue.Empty, queue.Full):
                    pass

    def read(self, timeout: float = 2.0) -> Frame | None:
        try:
            return self._frames.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(name=self._name, model="rtsp", notes=["No pan/tilt: one fixed view."])
