"""Where in the frame the detector actually needs to look.

The model sees a letterboxed square. From a 1080p frame at 640 that is a 3x
downscale, so a cat 70 px tall in the picture reaches the network as about 23 px —
near the floor of what a small-object detector resolves, and the reason detections
get missed on a machine that cannot afford a bigger model or a bigger input.

But the app already knows where it cares about: ``surface.project(pose)`` gives the
polygon in frame pixels, and ``surface.expected_height`` predicts how tall a cat
standing there should look. Cropping to that union and running the model on the
crop means the same pixels reach the network at a much larger scale, at the same
cost — which is what lets a smaller model stay viable on her Intel laptop.

What this deliberately does not do is narrow the *person* check. A person walking
up to a counter starts outside it, and the ``no_person`` gate exists to stop the
sound going off while she is cooking. Engine keeps a slower full-frame pass for
that; this module only decides where the cats are worth looking for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry.projection import Box, Pose, transform_points
from ..geometry.surface import Surface

# A cat can stand at the edge of a surface with most of its body outside it, and
# it arrives from somewhere. Pad by this many predicted cat heights.
PAD_CAT_HEIGHTS = 1.5
# Never crop below this, or the letterbox upscales noise into the network.
MIN_CROP_PX = 256
# A crop bigger than this fraction of the frame is not worth the bookkeeping.
MAX_CROP_FRACTION = 0.92


@dataclass(frozen=True)
class Roi:
    """A crop region in frame pixels, and what it cost to choose it."""

    x1: int
    y1: int
    x2: int
    y2: int
    magnification: float = 1.0
    full_frame: bool = False

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def crop(self, image: np.ndarray) -> np.ndarray:
        if self.full_frame:
            return image
        return image[self.y1:self.y2, self.x1:self.x2]

    def to_frame(self, box: Box) -> Box:
        """Map a detection found in the crop back to frame coordinates."""
        if self.full_frame or (self.x1 == 0 and self.y1 == 0):
            return box
        return Box(
            x1=box.x1 + self.x1, y1=box.y1 + self.y1,
            x2=box.x2 + self.x1, y2=box.y2 + self.y1,
            score=box.score, label=box.label,
            clipped_bottom=box.clipped_bottom,
        )


def full_frame(frame_size: tuple[int, int]) -> Roi:
    w, h = frame_size
    return Roi(0, 0, int(w), int(h), magnification=1.0, full_frame=True)


def for_surfaces(
    surfaces: list[Surface],
    pose: Pose,
    pad_heights: float = PAD_CAT_HEIGHTS,
) -> Roi:
    """The crop covering every visible surface, padded for an approaching cat.

    Falls back to the whole frame when nothing is visible, when the surfaces span
    most of the picture anyway, or when no plane model exists yet to say how big a
    cat should be — in which case cropping would be guesswork.
    """
    width, height = (int(v) for v in pose.frame_size)
    frame = full_frame((width, height))
    if not surfaces:
        return frame

    lows: list[np.ndarray] = []
    highs: list[np.ndarray] = []
    pads: list[float] = []
    for surface in surfaces:
        if not surface.enabled:
            continue
        points = transform_points(pose.map_to_frame, surface.polygon)
        if not np.all(np.isfinite(points)):
            continue
        lows.append(points.min(axis=0))
        highs.append(points.max(axis=0))
        centre = points.mean(axis=0)
        expected = surface.expected_height(pose, (float(centre[0]), float(centre[1])))
        if expected and expected > 1.0:
            pads.append(float(expected) * pad_heights)
    if not lows:
        return frame

    lo = np.min(np.stack(lows), axis=0)
    hi = np.max(np.stack(highs), axis=0)

    # No plane model yet means no idea how big a cat should look here. Pad by a
    # share of the surface instead of inventing a number.
    pad = max(pads) if pads else 0.35 * float(np.linalg.norm(hi - lo))

    x1 = int(max(0, np.floor(lo[0] - pad)))
    y1 = int(max(0, np.floor(lo[1] - pad)))
    x2 = int(min(width, np.ceil(hi[0] + pad)))
    y2 = int(min(height, np.ceil(hi[1] + pad)))
    if x2 - x1 < MIN_CROP_PX:
        x1, x2 = _widen(x1, x2, MIN_CROP_PX, width)
    if y2 - y1 < MIN_CROP_PX:
        y1, y2 = _widen(y1, y2, MIN_CROP_PX, height)
    if x2 <= x1 or y2 <= y1:
        return frame

    area = (x2 - x1) * (y2 - y1)
    if area >= MAX_CROP_FRACTION * width * height:
        return frame

    # How much larger the same object now reaches the network, versus the whole
    # frame letterboxed into the same square.
    longest_full = max(width, height)
    longest_crop = max(x2 - x1, y2 - y1)
    magnification = longest_full / longest_crop if longest_crop else 1.0
    return Roi(x1, y1, x2, y2, magnification=magnification, full_frame=False)


def _widen(low: int, high: int, minimum: int, limit: int) -> tuple[int, int]:
    """Grow a span to the minimum, staying inside the frame."""
    if limit <= minimum:
        return 0, limit
    need = minimum - (high - low)
    low -= need // 2
    high += need - need // 2
    if low < 0:
        high -= low
        low = 0
    if high > limit:
        low -= high - limit
        high = limit
    return max(0, low), min(limit, high)
