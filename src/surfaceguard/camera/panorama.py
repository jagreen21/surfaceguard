"""Build an undistorted room-view atlas by sweeping the camera (decision D2).

A PTZ camera rotates through a three-dimensional room. Trying to flatten that full
turn into one projective panorama creates poles at 90 degrees and turns ordinary
frames into huge triangles. The drawable map is therefore a contact sheet of clean,
unwarped reference views. Each tile remains a registration keyframe, so live video
can still be located after Eufy's tracker moves the camera.
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

MAX_CANVAS_PX = 16_000_000  # bound peak memory while warping on an older laptop
MAX_CANVAS_DIM_PX = 16_000
DEFAULT_SWEEP_STEP_DEG = 15.0
DEFAULT_SWEEP_MARGIN_DEG = 5.0
MIN_STITCH_INLIERS = 18
MIN_SHOT_FEATURES = 12
MAP_TILE_GAP_PX = 24
ATLAS_TARGET_ASPECT = 16.0 / 9.0

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
    """The drawable view atlas plus the keyframes that reference into it."""

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

    def keyframe_at(self, point: tuple[float, float]) -> str | None:
        """Return the atlas tile containing ``point``, if any."""
        x, y = float(point[0]), float(point[1])
        for kf in self.keyframes:
            height, width = kf.image.shape[:2]
            corners = np.float32(
                [[0, 0], [width, 0], [width, height], [0, height]]
            ).reshape(-1, 1, 2)
            quad = cv2.perspectiveTransform(corners, kf.full_to_map).reshape(-1, 2)
            if cv2.pointPolygonTest(quad.astype(np.float32), (x, y), False) >= 0:
                return kf.id
        return None

    def install_into(self, registrar: Registrar) -> None:
        registrar.clear()
        for kf in self.keyframes:
            registrar.add_keyframe(kf.id, kf.image, kf.full_to_map, kf.pan, kf.tilt)


class StitchError(RuntimeError):
    """A sweep that could not produce a usable atlas, with a user-facing reason."""


def build_room_map(
    source: CameraSource,
    pan_positions: list[float] | None = None,
    tilt: float = 0.0,
    settle_s: float = 1.2,
    min_inliers: int = MIN_STITCH_INLIERS,
    registrar: Registrar | None = None,
    progress=None,
) -> RoomMap:
    """Sweep ``source`` and arrange every usable view in a clean drawable atlas."""
    del min_inliers  # retained in the public API for compatibility with older callers
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
            # Leave the camera where setup began even when capture or atlas building
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

    # Keep the camera frames rectangular and arrange them in reading order. This
    # is intentionally not a stitched panorama: a flat homography cannot express
    # a full camera turn without severe projective distortion.
    cell_width = max(image.shape[1] for _pan, _tilt, image in shots)
    cell_height = max(image.shape[0] for _pan, _tilt, image in shots)
    count = len(shots)
    columns = min(
        range(1, count + 1),
        key=lambda candidate: abs(math.log(
            ((candidate * cell_width + (candidate - 1) * MAP_TILE_GAP_PX)
             / (math.ceil(count / candidate) * cell_height
                + (math.ceil(count / candidate) - 1) * MAP_TILE_GAP_PX))
            / ATLAS_TARGET_ASPECT
        )),
    )
    rows = math.ceil(count / columns)
    raw_width = columns * cell_width + (columns - 1) * MAP_TILE_GAP_PX
    raw_height = rows * cell_height + (rows - 1) * MAP_TILE_GAP_PX
    atlas_scale = min(
        1.0,
        math.sqrt(MAX_CANVAS_PX / float(raw_width * raw_height)),
        MAX_CANVAS_DIM_PX / float(raw_width),
        MAX_CANVAS_DIM_PX / float(raw_height),
    )
    width = max(1, int(math.ceil(raw_width * atlas_scale)))
    height = max(1, int(math.ceil(raw_height * atlas_scale)))
    scale = _scale_matrix(atlas_scale)
    log.info(
        "Building an undistorted %dx%d room-view atlas (%d views, %dx%d grid)",
        width, height, count, columns, rows,
    )

    canvas = np.full((height, width, 3), (11, 19, 24), np.uint8)
    keyframes: list[MapKeyframe] = []

    for i, (pan, tlt, img) in enumerate(shots):
        row, column = divmod(i, columns)
        image_height, image_width = img.shape[:2]
        x = column * (cell_width + MAP_TILE_GAP_PX) + (cell_width - image_width) / 2.0
        y = row * (cell_height + MAP_TILE_GAP_PX) + (cell_height - image_height) / 2.0
        tile = np.array([
            [1.0, 0.0, x],
            [0.0, 1.0, y],
            [0.0, 0.0, 1.0],
        ])
        full_to_map = scale @ tile
        warped = cv2.warpPerspective(img, full_to_map, (width, height))
        mask = cv2.warpPerspective(
            np.full(img.shape[:2], 255, np.uint8), full_to_map, (width, height)
        )
        canvas[mask > 0] = warped[mask > 0]
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
    """Retry a blurred/blank arrival and retain the most recognizable frame."""
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
