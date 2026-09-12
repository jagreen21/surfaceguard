"""Build the room map by sweeping the camera (decision D2).

The map is a visual atlas of everything the camera can reach. Connected views are
stitched into panorama sections; a blank wall or a view with no overlap starts a
new section instead of throwing away the whole scan. The user draws surfaces on
the atlas once, and each live frame is located against its keyframes at runtime.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .registration import Registrar, _scale_matrix, _to_work
from .sources.base import CameraSource, Frame

MAX_CANVAS_PX = 40_000_000  # refuse to allocate a map bigger than this
DEFAULT_SWEEP_STEP_DEG = 15.0
DEFAULT_SWEEP_MARGIN_DEG = 5.0
MIN_STITCH_INLIERS = 18
MIN_SHOT_FEATURES = 12
MAP_SECTION_GAP_PX = 48


@dataclass
class MapKeyframe:
    """A sweep position and where its pixels landed in the map."""

    id: str
    pan: float | None
    tilt: float | None
    full_to_map: np.ndarray
    image: np.ndarray = field(repr=False)
    features: int = 0


@dataclass
class RoomMap:
    """The stitched panorama plus the keyframes that reference into it."""

    canvas: np.ndarray = field(repr=False)
    keyframes: list[MapKeyframe] = field(default_factory=list)
    built_at: float = field(default_factory=time.time)
    source_name: str = ""

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.canvas.shape[:2]
        return (w, h)

    def pan_to_x(self, pan: float) -> float | None:
        """Approximate map x for a pan angle, by interpolating keyframe centres."""
        pts = []
        for kf in self.keyframes:
            if kf.pan is None:
                continue
            height, width = kf.image.shape[:2]
            centre = np.float32([[[width / 2.0, height / 2.0]]])
            mapped = cv2.perspectiveTransform(centre, kf.full_to_map)[0, 0]
            pts.append((kf.pan, float(mapped[0])))
        pts.sort()
        if len(pts) < 2:
            return None
        pans = np.array([p for p, _ in pts])
        xs = np.array([x for _, x in pts])
        return float(np.interp(pan, pans, xs))

    def install_into(self, registrar: Registrar) -> None:
        registrar.clear()
        for kf in self.keyframes:
            registrar.add_keyframe(kf.id, kf.image, kf.full_to_map, kf.pan, kf.tilt)


class StitchError(RuntimeError):
    """A sweep that could not be stitched, with a reason worth showing the user."""


def _pair_homography(
    registrar: Registrar, img_a: np.ndarray, img_b: np.ndarray, min_inliers: int
) -> tuple[np.ndarray, int]:
    """Full-resolution homography mapping ``img_a`` pixels onto ``img_b`` pixels."""
    kp_a, des_a, scale_a, _ = registrar.describe(img_a)
    kp_b, des_b, scale_b, _ = registrar.describe(img_b)
    if des_a is None or des_b is None:
        raise StitchError("one sweep position had no usable features")

    pairs = registrar._matcher.knnMatch(des_a, des_b, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 12:
        raise StitchError(
            f"only {len(good)} matches between adjacent sweep positions — "
            "overlap is too small or the view is too plain"
        )

    src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    h_work, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0, maxIters=3000, confidence=0.995)
    inliers = int(mask.sum()) if mask is not None else 0
    if h_work is None or inliers < min_inliers:
        raise StitchError(f"{inliers} inliers between adjacent sweep positions (need {min_inliers})")
    # work(a) -> work(b), lifted to full resolution on both sides.
    return _scale_matrix(1.0 / scale_b) @ h_work @ _scale_matrix(scale_a), inliers


def build_room_map(
    source: CameraSource,
    pan_positions: list[float] | None = None,
    tilt: float = 0.0,
    settle_s: float = 1.2,
    min_inliers: int = MIN_STITCH_INLIERS,
    registrar: Registrar | None = None,
    progress=None,
) -> RoomMap:
    """Sweep ``source`` across ``pan_positions`` and stitch the result.

    Adjacent positions that do not overlap begin a new atlas section. A
    :class:`StitchError` is reserved for failures that leave no usable map.
    """
    caps = source.capabilities
    angles = getattr(source, "current_angles", None)
    start_pan, start_tilt = angles() if callable(angles) else (None, None)
    if start_pan is None:
        estimate = getattr(source, "dead_reckoned_angles", None)
        if callable(estimate):
            start_pan, estimated_tilt = estimate()
            if start_tilt is None:
                start_tilt = estimated_tilt
    centre_pan = float(start_pan or 0.0)
    centre_tilt = float(start_tilt if start_tilt is not None else tilt)
    if pan_positions is None:
        lo, hi = caps.pan_range or (-40.0, 40.0)
        # Auto-tracking can point the E30 anywhere, so calibration must cover its
        # full reachable view. Fifteen-degree steps give this real camera more
        # overlap than the original 20-degree sweep; a plain wall frame is retried
        # and omitted below rather than making the useful room fail.
        left = lo + DEFAULT_SWEEP_MARGIN_DEG
        right = hi - DEFAULT_SWEEP_MARGIN_DEG
        pan_positions = (
            list(np.arange(left, right + 0.1, DEFAULT_SWEEP_STEP_DEG))
            if caps.has_ptz else [centre_pan]
        )

    reg = registrar or Registrar()
    shots: list[tuple[float | None, float | None, np.ndarray]] = []
    suspend_tracking = getattr(source, "suspend_auto_tracking", None)
    restore_tracking = getattr(source, "restore_auto_tracking", None)
    tracking_token = suspend_tracking() if callable(suspend_tracking) else None
    moved = False
    try:
        # Verify video before issuing even one PTZ command. Camera control and
        # video are separate paths on Eufy hardware.
        if source.read(timeout=5.0) is None:
            raise StitchError(
                "no camera picture arrived, so the room scan did not move the camera"
            )
        for i, pan in enumerate(pan_positions):
            if caps.has_ptz and not source.move_to(pan, centre_tilt, settle_s=settle_s):
                raise StitchError("the camera did not respond while looking around the room")
            moved = moved or caps.has_ptz
            frame, features = _best_feature_frame(source, reg)
            if frame is None:
                raise StitchError(
                    f"the camera stopped sending video after {i} of {len(pan_positions)} positions"
                )
            # A blank wall at one edge should shorten the panorama, not destroy a
            # scan whose useful views are already good. Retry transient blur first,
            # then omit only the genuinely textureless position.
            if features >= MIN_SHOT_FEATURES:
                shots.append((
                    frame.pan if frame.pan is not None else pan,
                    frame.tilt,
                    frame.image,
                ))
            if progress:
                progress(i + 1, len(pan_positions))
    finally:
        if moved:
            # Leave the camera where setup began even when capture or stitching
            # fails. A setup error must never strand it facing a wall.
            try:
                source.move_to(centre_pan, centre_tilt, settle_s=0.0)
            except Exception:
                pass
        if callable(restore_tracking):
            try:
                restore_tracking(tracking_token)
            except Exception:
                pass

    if not shots:
        raise StitchError(
            "the camera only saw a plain wall or a very dark view — point it at "
            "the surfaces you want to protect and try again"
        )

    # Chain matching neighbours. A transition with six matches is not evidence
    # that either picture is bad; it only means they cannot share coordinates.
    # Keep both as separate atlas sections so every trackable view remains usable.
    sections: list[list[tuple[np.ndarray, tuple[float | None, float | None, np.ndarray]]]] = []
    for shot in shots:
        if not sections:
            sections.append([(np.eye(3), shot)])
            continue
        previous_h, previous = sections[-1][-1]
        try:
            pair_h, _inliers = _pair_homography(
                reg, shot[2], previous[2], min_inliers
            )
        except StitchError:
            sections.append([(np.eye(3), shot)])
        else:
            sections[-1].append((previous_h @ pair_h, shot))

    # Lay independent panorama sections left-to-right in one drawable atlas.
    placed: list[tuple[np.ndarray, tuple[float | None, float | None, np.ndarray]]] = []
    cursor_x = 0
    height = 0
    for section in sections:
        corners = []
        for transform, (_pan, _tilt, image) in section:
            hh, ww = image.shape[:2]
            quad = np.float32([[0, 0], [ww, 0], [ww, hh], [0, hh]]).reshape(-1, 1, 2)
            corners.append(cv2.perspectiveTransform(quad, transform).reshape(-1, 2))
        allc = np.vstack(corners)
        x0, y0 = np.floor(allc.min(axis=0)).astype(int)
        x1, y1 = np.ceil(allc.max(axis=0)).astype(int)
        section_width, section_height = int(x1 - x0), int(y1 - y0)
        section_shift = np.array([
            [1.0, 0.0, float(cursor_x - x0)],
            [0.0, 1.0, float(-y0)],
            [0.0, 0.0, 1.0],
        ])
        placed.extend((section_shift @ transform, shot) for transform, shot in section)
        cursor_x += section_width + MAP_SECTION_GAP_PX
        height = max(height, section_height)

    width = cursor_x - MAP_SECTION_GAP_PX
    if width <= 0 or height <= 0 or width * height > MAX_CANVAS_PX:
        raise StitchError(f"stitched map came out an implausible {width}x{height} px")

    canvas = np.zeros((height, width, 3), np.uint8)
    filled = np.zeros((height, width), np.uint8)
    keyframes: list[MapKeyframe] = []

    for i, (full_to_map, (pan, tlt, img)) in enumerate(placed):
        warped = cv2.warpPerspective(img, full_to_map, (width, height))
        mask = cv2.warpPerspective(
            np.full(img.shape[:2], 255, np.uint8), full_to_map, (width, height)
        )
        # Later shots only fill what is still empty, so seams land in overlap
        # regions instead of ghosting across the whole map.
        fresh = (mask > 0) & (filled == 0)
        canvas[fresh] = warped[fresh]
        filled[mask > 0] = 255
        work, _ = _to_work(img)
        kp = reg._orb.detect(work, None)
        keyframes.append(
            MapKeyframe(f"kf{i:02d}", pan, tlt, full_to_map, img, features=len(kp) if kp else 0)
        )

    return RoomMap(canvas=canvas, keyframes=keyframes, source_name=caps.name)


def _best_feature_frame(
    source: CameraSource,
    registrar: Registrar,
    attempts: int = 3,
) -> tuple[Frame | None, int]:
    """Retry a blurred/blank arrival and retain the most stitchable frame."""
    best = None
    best_features = 0
    for attempt in range(attempts):
        frame = source.read(timeout=5.0 if attempt == 0 else 1.0)
        if frame is None:
            continue
        keypoints, descriptors, _scale, _shape = registrar.describe(frame.image)
        count = len(keypoints) if descriptors is not None else 0
        if best is None or count > best_features:
            best, best_features = frame, count
        if count >= MIN_SHOT_FEATURES:
            break
    return best, best_features
