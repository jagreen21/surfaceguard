"""Locate a live frame in map space (decision D1).

Registration is on the critical path for every frame, so it runs on a downscaled
greyscale image with cached reference descriptors. The inlier count it returns is
the app's confidence signal: too few inliers means the frame cannot be located,
and the honest response is to refuse to arm rather than to project polygons onto
an unknown view (E2, E7).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..geometry.projection import Pose

# Default gate on RANSAC inliers. Below this the pose is reported as unusable.
MIN_INLIERS = 30

# Registration runs on frames downscaled to this width; homographies are scaled
# back up afterwards. 480 px keeps ORB under ~8 ms on Apple silicon.
WORK_WIDTH = 480


@dataclass
class Keyframe:
    """One reference view, with its place in the map already known."""

    id: str
    keypoints: tuple
    descriptors: np.ndarray
    work_to_map: np.ndarray          # work-resolution pixels -> map coords
    pan: float | None = None
    tilt: float | None = None
    shape: tuple[int, int] = (0, 0)  # (h, w) at work resolution


@dataclass
class RegistrationResult:
    pose: Pose | None
    inliers: int
    matches: int
    elapsed_ms: float
    keyframe_id: str = ""
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.pose is not None


def _to_work(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Greyscale + downscale. Returns the image and the scale that produced it."""
    if image.ndim == 3:
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grey = image
    h, w = grey.shape[:2]
    if w <= WORK_WIDTH:
        return grey, 1.0
    scale = WORK_WIDTH / float(w)
    return cv2.resize(grey, (WORK_WIDTH, max(1, int(round(h * scale))))), scale


def _scale_matrix(s: float) -> np.ndarray:
    return np.array([[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, 1.0]])


class Registrar:
    """Holds the map's keyframes and locates incoming frames against them."""

    def __init__(
        self,
        n_features: int = 800,
        min_inliers: int = MIN_INLIERS,
        ratio: float = 0.75,
        angle_window: float = 35.0,
    ) -> None:
        self._orb = cv2.ORB_create(nfeatures=n_features, fastThreshold=12)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.min_inliers = min_inliers
        self.ratio = ratio
        # Only keyframes within this many degrees of the reported pan/tilt are
        # tried first. Angles are a prior, never the answer.
        self.angle_window = angle_window
        self.keyframes: list[Keyframe] = []
        self._last_keyframe_id: str | None = None
        self._preferred_keyframe_ids: list[str] = []

    # ------------------------------------------------------------- keyframes

    def describe(self, image: np.ndarray) -> tuple[tuple, np.ndarray | None, float, tuple[int, int]]:
        work, scale = _to_work(image)
        kp, des = self._orb.detectAndCompute(work, None)
        return kp, des, scale, work.shape[:2]

    def add_keyframe(
        self,
        key_id: str,
        image: np.ndarray,
        full_to_map: np.ndarray,
        pan: float | None = None,
        tilt: float | None = None,
    ) -> Keyframe | None:
        """Register a reference view whose place in the map is already known.

        ``full_to_map`` maps this view's full-resolution pixels into map
        coordinates; the downscale used for matching is folded in here so callers
        never have to think about work resolution.
        """
        kp, des, scale, shape = self.describe(image)
        if des is None or len(kp) < 12:
            return None
        work_to_map = np.asarray(full_to_map, float) @ _scale_matrix(1.0 / scale)
        kf = Keyframe(key_id, kp, des, work_to_map, pan, tilt, shape)
        self.keyframes.append(kf)
        return kf

    def clear(self) -> None:
        self.keyframes.clear()
        self._last_keyframe_id = None
        self._preferred_keyframe_ids = []

    def prefer_keyframes(self, keyframe_ids: list[str]) -> None:
        """Try atlas tiles containing protected surfaces before other views."""
        known = {kf.id for kf in self.keyframes}
        self._preferred_keyframe_ids = [
            key_id for key_id in dict.fromkeys(keyframe_ids) if key_id in known
        ]

    # ------------------------------------------------------------- the fit

    def register(
        self,
        image: np.ndarray,
        pan: float | None = None,
        tilt: float | None = None,
    ) -> RegistrationResult:
        started = time.perf_counter()
        if not self.keyframes:
            return RegistrationResult(None, 0, 0, 0.0, reason="no keyframes in map")

        kp, des, scale, _shape = self.describe(image)
        if des is None or len(kp) < 12:
            return RegistrationResult(
                None, 0, 0, (time.perf_counter() - started) * 1e3,
                reason=f"only {len(kp)} features in frame (textureless view?)",
            )

        best: RegistrationResult | None = None
        for kf in self._candidates(pan, tilt):
            result = self._fit(kf, kp, des, scale, image, started)
            if best is None or result.inliers > best.inliers:
                best = result
            if result.ok:
                # A camera-reported angle makes the nearest successful keyframe
                # a trustworthy prior. When Eufy's tracker supplies no angle, a
                # tile containing a protected surface is also preferred so an
                # adjacent duplicate view cannot hide the user's drawn zone.
                strong = result.inliers >= max(60, self.min_inliers * 2)
                protected_view = (
                    pan is None and kf.id in self._preferred_keyframe_ids
                )
                if pan is not None or strong or protected_view:
                    self._last_keyframe_id = kf.id
                    return result

        assert best is not None
        if best.ok:
            self._last_keyframe_id = best.keyframe_id
        return best

    def _candidates(self, pan: float | None, tilt: float | None) -> list[Keyframe]:
        """Keyframes to try, most-likely first.

        With an angle hint, nearest-angle wins: the first keyframe that fits is
        the one used, so trying a distant view first would succeed with fewer
        inliers and more geometric error. Without a hint, the view that worked
        last frame is the best guess, since views change slowly.
        """
        ordered: list[Keyframe] = []
        seen: set[str] = set()

        def push(kf: Keyframe) -> None:
            if kf.id not in seen:
                seen.add(kf.id)
                ordered.append(kf)

        if pan is not None:
            near = [
                kf for kf in self.keyframes
                if kf.pan is not None and abs(kf.pan - pan) <= self.angle_window
                and (tilt is None or kf.tilt is None or abs(kf.tilt - tilt) <= self.angle_window)
            ]
            for kf in sorted(near, key=lambda k: abs((k.pan or 0.0) - pan)):
                push(kf)
        else:
            for key_id in self._preferred_keyframe_ids:
                for kf in self.keyframes:
                    if kf.id == key_id:
                        push(kf)
                        break
        for kf in self.keyframes:
            if kf.id == self._last_keyframe_id:
                push(kf)
        for kf in self.keyframes:
            push(kf)
        return ordered

    def _fit(
        self,
        kf: Keyframe,
        kp: tuple,
        des: np.ndarray,
        scale: float,
        image: np.ndarray,
        started: float,
    ) -> RegistrationResult:
        pairs = self._matcher.knnMatch(des, kf.descriptors, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < self.ratio * n.distance]
        elapsed = lambda: (time.perf_counter() - started) * 1e3  # noqa: E731
        if len(good) < 12:
            return RegistrationResult(
                None, 0, len(good), elapsed(), kf.id,
                reason=f"{len(good)} good matches against {kf.id}",
            )

        src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kf.keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        h_work, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0, maxIters=2000, confidence=0.995)
        inliers = int(mask.sum()) if mask is not None else 0
        if h_work is None or inliers < self.min_inliers:
            return RegistrationResult(
                None, inliers, len(good), elapsed(), kf.id,
                reason=f"{inliers} inliers against {kf.id} (need {self.min_inliers})",
            )

        # work(frame) -> work(keyframe) -> map, then undo the downscale so the
        # pose speaks in full-resolution frame pixels.
        frame_to_map = kf.work_to_map @ h_work @ _scale_matrix(scale)
        if not np.isfinite(frame_to_map).all() or abs(np.linalg.det(frame_to_map)) < 1e-12:
            return RegistrationResult(
                None, inliers, len(good), elapsed(), kf.id, reason="degenerate homography",
            )

        h, w = image.shape[:2]
        pose = Pose(
            map_to_frame=np.linalg.inv(frame_to_map),
            frame_size=(w, h),
            inliers=inliers,
            reference_id=kf.id,
            pan=kf.pan,
            tilt=kf.tilt,
        )
        return RegistrationResult(pose, inliers, len(good), elapsed(), kf.id)
