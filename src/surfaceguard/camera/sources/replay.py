"""Replay recorded clips through the real pipeline.

This is not a test fixture bolted on at the end — it is how thresholds get tuned
(§9). Every clip the user marks "not a cat" becomes a file here, and a threshold
change can then be justified with a number instead of a feeling.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from .base import Capabilities, CameraSource, Frame, SourceError

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


class ReplayCamera(CameraSource):
    """Plays a video file or a directory of stills.

    ``realtime`` paces frames at the clip's own rate, for latency-shaped tests;
    turn it off to push a whole clip through as fast as the pipeline will go.
    """

    def __init__(
        self,
        path: str | Path,
        realtime: bool = True,
        loop: bool = False,
        frame_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.path = Path(path)
        self.realtime = realtime
        self.loop = loop
        self.frame_size = frame_size
        self._cap: cv2.VideoCapture | None = None
        self._stills: list[Path] = []
        self._index = 0
        self._fps = 15.0
        self._last = 0.0
        self.exhausted = False

    def start(self) -> None:
        if not self.path.exists():
            raise SourceError(f"No clip at {self.path}")
        if self.path.is_dir():
            self._stills = sorted(
                p for p in self.path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
            )
            if not self._stills:
                raise SourceError(f"{self.path} has no images in it")
        elif self.path.suffix.lower() in VIDEO_SUFFIXES:
            self._cap = cv2.VideoCapture(str(self.path))
            if not self._cap.isOpened():
                raise SourceError(f"Could not read the clip at {self.path}")
            self._fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 15.0
        else:
            raise SourceError(f"{self.path.suffix} is not a clip this app can replay")
        self.exhausted = False
        self._index = 0

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read(self, timeout: float = 2.0) -> Frame | None:
        if self.realtime:
            wait = (1.0 / self._fps) - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(min(wait, timeout))
        self._last = time.monotonic()

        image = self._next_image()
        if image is None:
            self.exhausted = True
            return None
        if self.frame_size:
            image = cv2.resize(image, self.frame_size)
        return Frame(image, self._next_seq(), time.monotonic())

    def _next_image(self):
        if self._stills:
            if self._index >= len(self._stills):
                if not self.loop:
                    return None
                self._index = 0
            path = self._stills[self._index]
            self._index += 1
            return cv2.imread(str(path))
        assert self._cap is not None
        ok, image = self._cap.read()
        if not ok:
            if not self.loop:
                return None
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, image = self._cap.read()
            if not ok:
                return None
        return image

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            name=f"Replay: {self.path.name}",
            model="replay",
            notes=[f"{'Directory of stills' if self._stills else 'Video file'}, {self._fps:.0f} fps"],
        )
