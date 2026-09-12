"""Build the room map by sweeping the camera (decision D2).

The map is a visual atlas of everything the camera can reach. Connected views are
stitched into panorama sections; a blank wall or a view with no overlap starts a
new section instead of throwing away the whole scan. The user draws surfaces on
the atlas once, and each live frame is located against its keyframes at runtime.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .registration import Registrar, _scale_matrix, _to_work
from .sources.base import CameraSource, Frame

MAX_CANVAS_PX = 40_000_000  # refuse to allocate a map bigger than this
MAX_CANVAS_DIM_PX = 32_000  # OpenCV remap uses 16-bit coordinates internally
MIN_ATLAS_SCALE = 0.25
DEFAULT_SWEEP_STEP_DEG = 15.0
DEFAULT_SWEEP_MARGIN_DEG = 5.0
MIN_STITCH_INLIERS = 18
MIN_SHOT_FEATURES = 12
MAP_SECTION_GAP_PX = 48
MIN_ADJACENT_AREA_RATIO = 0.15
MAX_ADJACENT_AREA_RATIO = 6.0
MIN_ADJACENT_OVERLAP = 0.01
MAX_QUAD_EDGE_RATIO = 20.0

log = logging.getLogger(__name__)


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


def _projected_quad(image: np.ndarray, transform: np.ndarray) -> np.ndarray | None:
    """Return the image boundary after ``transform``, or None if it is invalid."""
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return None
    height, width = image.shape[:2]
    source = np.float32(
        [[0, 0], [width, 0], [width, height], [0, height]]
    ).reshape(-1, 1, 2)
    try:
        quad = cv2.perspectiveTransform(source, matrix).reshape(-1, 2)
    except cv2.error:
        return None
    return quad if np.isfinite(quad).all() else None


def _plausible_adjacent_transform(
    current_transform: np.ndarray,
    current_image: np.ndarray,
    previous_transform: np.ndarray,
    previous_image: np.ndarray,
) -> bool:
    """Reject homographies that cannot describe two neighbouring camera views.

    RANSAC can occasionally find enough coincidental matches on repeated kitchen
    edges to return a numerical homography that magnifies a frame by hundreds of
    times. Inlier count alone cannot catch that failure. Adjacent sweep views must
    retain orientation, remain convex, have comparable scale, and actually overlap.
    """
    current = _projected_quad(current_image, current_transform)
    previous = _projected_quad(previous_image, previous_transform)
    if current is None or previous is None:
        return False

    current32 = current.astype(np.float32)
    previous32 = previous.astype(np.float32)
    if not cv2.isContourConvex(current32) or not cv2.isContourConvex(previous32):
        return False

    current_signed = float(cv2.contourArea(current32, oriented=True))
    previous_signed = float(cv2.contourArea(previous32, oriented=True))
    if current_signed * previous_signed <= 0:
        return False
    current_area, previous_area = abs(current_signed), abs(previous_signed)
    if current_area < 1.0 or previous_area < 1.0:
        return False
    area_ratio = current_area / previous_area
    if not MIN_ADJACENT_AREA_RATIO <= area_ratio <= MAX_ADJACENT_AREA_RATIO:
        return False

    edges = np.linalg.norm(current - np.roll(current, -1, axis=0), axis=1)
    if float(edges.min()) < 1.0 or float(edges.max() / edges.min()) > MAX_QUAD_EDGE_RATIO:
        return False

    try:
        overlap_area, _intersection = cv2.intersectConvexConvex(current32, previous32)
    except cv2.error:
        return False
    overlap_ratio = float(overlap_area) / min(current_area, previous_area)
    return math.isfinite(overlap_ratio) and overlap_ratio >= MIN_ADJACENT_OVERLAP


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
            else:
                log.warning(
                    "Skipping featureless sweep view at pan %.1f (%d features)",
                    float(pan),
                    features,
                )
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
            candidate_h = previous_h @ pair_h
            if not _plausible_adjacent_transform(
                candidate_h, shot[2], previous_h, previous[2]
            ):
                raise StitchError("adjacent transform had implausible geometry")
        except StitchError as exc:
            log.warning(
                "Starting a new room-map section at pan %s: %s",
                shot[0],
                exc,
            )
            sections.append([(np.eye(3), shot)])
        else:
            sections[-1].append((candidate_h, shot))

    # Lay independent panorama sections left-to-right in one drawable atlas.
    placed: list[tuple[np.ndarray, tuple[float | None, float | None, np.ndarray]]] = []
    cursor_x = 0
    height = 0
    for section in sections:
        corners = []
        for transform, (_pan, _tilt, image) in section:
            quad = _projected_quad(image, transform)
            if quad is None:
                raise StitchError("a room-map section contained invalid geometry")
            corners.append(quad)
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
    if width <= 0 or height <= 0:
        raise StitchError(f"stitched map came out an implausible {width}x{height} px")

    # A room with several blank transitions can legitimately contain many
    # independent full-resolution panels. Fit a moderately oversized valid atlas
    # into OpenCV's safe limits while keeping every original keyframe for runtime
    # registration. Extreme scaling still indicates corrupt geometry and fails.
    atlas_scale = min(
        1.0,
        math.sqrt(MAX_CANVAS_PX / float(width * height)),
        MAX_CANVAS_DIM_PX / float(width),
        MAX_CANVAS_DIM_PX / float(height),
    )
    if atlas_scale < MIN_ATLAS_SCALE:
        raise StitchError(f"stitched map came out an implausible {width}x{height} px")
    if atlas_scale < 1.0:
        scale = _scale_matrix(atlas_scale)
        placed = [(scale @ transform, shot) for transform, shot in placed]
        width = max(1, int(math.ceil(width * atlas_scale)))
        height = max(1, int(math.ceil(height * atlas_scale)))
        log.info("Scaled room-map atlas to %dx%d", width, height)

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
