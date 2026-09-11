"""A protected surface: a polygon in map space, plus the plane it lies on.

The scale model here is the whole of decision D3's "apparent size fits plane"
gate. For a planar surface, apparent depth is proportional to ``1 / w`` where
``w`` is the plane homography's perspective denominator, so the expected on-screen
height of a fixed real-world object is ``k * w`` for a single scalar ``k``. That
makes the gate self-calibrating from confirmed detections instead of needing the
user to measure anything.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field

import numpy as np

from .projection import (
    Array,
    Box,
    Pose,
    perspective_weight,
    point_in_polygon,
    quad_to_plane_homography,
    transform_points,
    visible_fraction,
)

# Below this many samples the scale gate reports itself advisory rather than
# required, so it can never be the reason a real cat is missed while it is still
# learning. There is deliberately no seeded default: real-world scale cannot be
# inferred from a drawn polygon, and a guess here would veto real cats.
MIN_CALIBRATION_SAMPLES = 8

# Samples kept per surface for the running median.
MAX_HEIGHT_SAMPLES = 64


@dataclass
class ScheduleWindow:
    """Local-time window during which a surface is armed. ``days``: 0 = Monday."""

    start_minute: int = 0
    end_minute: int = 24 * 60
    days: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)

    def contains(self, weekday: int, minute_of_day: int) -> bool:
        if weekday not in self.days:
            return False
        if self.start_minute <= self.end_minute:
            return self.start_minute <= minute_of_day < self.end_minute
        # Window crosses midnight (e.g. 22:00 -> 06:00).
        return minute_of_day >= self.start_minute or minute_of_day < self.end_minute


@dataclass
class Deterrent:
    """Per-surface deterrent settings, in the user's vocabulary."""

    sound: str = "chirp"
    volume: float = 0.6
    delay_s: float = 0.0
    cooldown_s: float = 20.0
    vary_sound: bool = True


@dataclass
class Surface:
    """One protected surface, stored in map coordinates.

    ``polygon`` is the shape the user drew (any number of points). ``plane_quad``
    is the four corners used for the plane model; for a 4-point polygon it is the
    polygon itself, which is the common case for counters and tables.
    """

    name: str
    polygon: Array
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    enabled: bool = True
    plane_quad: Array | None = None
    height_samples: list[float] = field(default_factory=list)
    deterrent: Deterrent = field(default_factory=Deterrent)
    schedule: ScheduleWindow = field(default_factory=ScheduleWindow)

    def __post_init__(self) -> None:
        self.polygon = np.asarray(self.polygon, dtype=np.float64).reshape(-1, 2)
        if self.plane_quad is not None:
            self.plane_quad = np.asarray(self.plane_quad, dtype=np.float64).reshape(4, 2)

    # ------------------------------------------------------------------ shape

    @property
    def quad(self) -> Array:
        """The four corners used for the plane model."""
        if self.plane_quad is not None:
            return self.plane_quad
        if len(self.polygon) == 4:
            return self.polygon
        return _extreme_quad(self.polygon)

    def project(self, pose: Pose) -> Array:
        """The polygon in this frame's pixels."""
        return transform_points(pose.map_to_frame, self.polygon)

    def visible_fraction(self, pose: Pose) -> float:
        return visible_fraction(self.project(pose), pose.frame_size)

    def contains_paw(self, pose: Pose, paw: tuple[float, float]) -> bool:
        return point_in_polygon(self.project(pose), paw)

    # ------------------------------------------------------------ scale model

    def plane_homography(self, pose: Pose) -> Array:
        """Frame pixels -> rectified plane, for this frame's pose."""
        return quad_to_plane_homography(transform_points(pose.map_to_frame, self.quad))

    @property
    def size_scale(self) -> float | None:
        """The calibrated constant ``k``, or None while still learning.

        A median rather than a running mean: one mis-sized box from a cat caught
        mid-leap should not drag the model with it.
        """
        if not self.height_samples:
            return None
        return float(np.median(self.height_samples))

    @property
    def calibration_samples(self) -> int:
        return len(self.height_samples)

    def expected_height(self, pose: Pose, paw: tuple[float, float]) -> float | None:
        """Predicted on-screen height of a cat standing at ``paw`` on this surface.

        None until the surface has seen at least one confirmed detection: there is
        no honest prior for real-world size, so the gate stays unavailable rather
        than inventing one.
        """
        scale = self.size_scale
        if scale is None:
            return None
        w = self.plane_weight(pose, paw)
        return None if w is None else scale * w

    def plane_weight(self, pose: Pose, paw: tuple[float, float]) -> float | None:
        w = float(perspective_weight(self.plane_homography(pose), [paw])[0])
        return w if math.isfinite(w) and w > 0.0 else None

    def observe_height(self, pose: Pose, box: Box) -> None:
        """Fold one confirmed on-surface detection into the scale model."""
        w = self.plane_weight(pose, box.paw_point)
        if w is None or box.height <= 1.0:
            return
        self.height_samples.append(box.height / w)
        if len(self.height_samples) > MAX_HEIGHT_SAMPLES:
            del self.height_samples[0]

    def forget_last_observation(self) -> None:
        """Undo the most recent sample, for when the user marks it 'not a cat'."""
        if self.height_samples:
            self.height_samples.pop()

    @property
    def scale_gate_is_required(self) -> bool:
        return self.calibration_samples >= MIN_CALIBRATION_SAMPLES


def _extreme_quad(poly: Array) -> Array:
    """Best four-corner stand-in for a polygon with more than four points."""
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    s, d = p.sum(axis=1), p[:, 0] - p[:, 1]
    idx = [int(np.argmin(s)), int(np.argmax(d)), int(np.argmax(s)), int(np.argmin(d))]
    return p[idx]
