"""Map space <-> frame space.

A *pose* is the homography that carries points from map coordinates (the stitched
room panorama, where surfaces are stored) into the pixels of one live frame.
Everything in this module is pure numpy so the geometry can be tested without a
camera, OpenCV, or a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Array = np.ndarray


# --------------------------------------------------------------------------- pose


@dataclass(frozen=True)
class Pose:
    """Where one frame sits in map space.

    ``map_to_frame`` is the 3x3 homography; ``inliers`` is the RANSAC inlier
    count that produced it, which is the app's confidence signal (see E2).
    """

    map_to_frame: Array
    frame_size: tuple[int, int]  # (width, height)
    inliers: int = 0
    reference_id: str = ""
    pan: float | None = None
    tilt: float | None = None

    @property
    def frame_to_map(self) -> Array:
        return np.linalg.inv(self.map_to_frame)

    @staticmethod
    def identity(frame_size: tuple[int, int], inliers: int = 10_000) -> "Pose":
        return Pose(np.eye(3), frame_size, inliers=inliers, reference_id="identity")


# ------------------------------------------------------------------- transforms


def to_homogeneous(points: Array) -> Array:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return np.hstack([points, np.ones((len(points), 1))])


def transform_points(h: Array, points: Array) -> Array:
    """Apply a 3x3 homography to an (N, 2) array of points."""
    pts = to_homogeneous(points)
    out = pts @ np.asarray(h, dtype=np.float64).T
    w = out[:, 2:3]
    # A point on the horizon has w == 0 and no finite image; push it far away
    # rather than emitting inf/nan, so downstream clipping stays well-defined.
    w = np.where(np.abs(w) < 1e-12, np.sign(w) * 1e-12 + 1e-12, w)
    return out[:, :2] / w


def perspective_weight(h: Array, points: Array) -> Array:
    """The homography's third-row denominator at each point.

    For a planar surface this is proportional to ``1 / depth``, and therefore to
    the apparent on-screen size of a fixed real-world object standing on it. That
    is what makes the scale gate in :mod:`surfaceguard.geometry.surface` a
    one-constant model rather than a calibration rig.

    Homographies from :func:`quad_to_plane_homography` are sign-normalised so
    this is positive across the surface; a non-positive result means the point is
    behind the plane's horizon and has no meaningful expected size.
    """
    pts = to_homogeneous(points)
    return pts @ np.asarray(h, dtype=np.float64).T[:, 2]


# ---------------------------------------------------------------- polygon tools


def polygon_area(poly: Array) -> float:
    """Unsigned shoelace area."""
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def point_in_polygon(poly: Array, point: tuple[float, float]) -> bool:
    """Crossing-number test. Points exactly on an edge count as inside."""
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if len(p) < 3:
        return False
    x, y = float(point[0]), float(point[1])
    a = p
    b = np.roll(p, -1, axis=0)

    # On-edge check first: collinear and within the segment's bounding box.
    cross = (b[:, 0] - a[:, 0]) * (y - a[:, 1]) - (b[:, 1] - a[:, 1]) * (x - a[:, 0])
    within = (
        (np.minimum(a[:, 0], b[:, 0]) - 1e-9 <= x)
        & (x <= np.maximum(a[:, 0], b[:, 0]) + 1e-9)
        & (np.minimum(a[:, 1], b[:, 1]) - 1e-9 <= y)
        & (y <= np.maximum(a[:, 1], b[:, 1]) + 1e-9)
    )
    if np.any((np.abs(cross) < 1e-9) & within):
        return True

    straddles = (a[:, 1] > y) != (b[:, 1] > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_at_y = a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1])
    crossings = int(np.count_nonzero(straddles & (x < x_at_y)))
    return crossings % 2 == 1


def clip_polygon_to_rect(poly: Array, width: float, height: float) -> Array:
    """Sutherland-Hodgman clip against the frame rectangle."""
    subject = np.asarray(poly, dtype=np.float64).reshape(-1, 2).tolist()
    #      (inside test,                    intersect axis, bound)
    edges = [
        (lambda p: p[0] >= 0.0, 0, 0.0),
        (lambda p: p[0] <= width, 0, width),
        (lambda p: p[1] >= 0.0, 1, 0.0),
        (lambda p: p[1] <= height, 1, height),
    ]
    for inside, axis, bound in edges:
        if not subject:
            return np.empty((0, 2))
        output: list[list[float]] = []
        prev = subject[-1]
        prev_in = inside(prev)
        for cur in subject:
            cur_in = inside(cur)
            if cur_in != prev_in:
                other = 1 - axis
                span = cur[axis] - prev[axis]
                t = 0.0 if abs(span) < 1e-12 else (bound - prev[axis]) / span
                crossing = [0.0, 0.0]
                crossing[axis] = bound
                crossing[other] = prev[other] + t * (cur[other] - prev[other])
                output.append(crossing)
            if cur_in:
                output.append(list(cur))
            prev, prev_in = cur, cur_in
        subject = output
    return np.asarray(subject, dtype=np.float64).reshape(-1, 2)


def visible_fraction(poly_frame: Array, frame_size: tuple[int, int]) -> float:
    """How much of a projected polygon actually lands inside the frame.

    Gate 6 refuses to judge a cat against a surface that is only partly in
    view, rather than guessing about the part it cannot see.
    """
    total = polygon_area(poly_frame)
    if total <= 0.0:
        return 0.0
    w, h = frame_size
    return min(1.0, polygon_area(clip_polygon_to_rect(poly_frame, w, h)) / total)


# ------------------------------------------------------------------- quad plane


def quad_to_plane_homography(quad: Array, aspect: float = 1.0) -> Array:
    """Homography from image pixels to a rectified view of a *rectangular* surface.

    Counters, tables, desks and shelves are rectangles in the world, so the four
    corners the user clicks are all the plane calibration this app ever needs.
    Returns the 3x3 matrix mapping the quad's pixels onto ``[0,1] x [0,aspect]``.
    """
    src = _order_quad(np.asarray(quad, dtype=np.float64).reshape(4, 2))
    dst = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, aspect], [0.0, aspect]])
    h = _dlt_homography(src, dst)
    # The SVD solution is defined only up to an overall sign, and negating the
    # matrix leaves the mapping identical while flipping the perspective
    # denominator. Fix the sign so w > 0 across the surface, which lets callers
    # read w directly as "proportional to apparent size" (see Surface).
    if float(perspective_weight(h, [src.mean(axis=0)])[0]) < 0.0:
        h = -h
    return h


def _order_quad(quad: Array) -> Array:
    """Sort four corners into tl, tr, br, bl so the plane basis is consistent."""
    centre = quad.mean(axis=0)
    angles = np.arctan2(quad[:, 1] - centre[1], quad[:, 0] - centre[0])
    ordered = quad[np.argsort(angles)]
    # np.argsort on atan2 starts at -pi (left, slightly above centre); rotate so
    # the top-left corner (smallest x + y) comes first.
    start = int(np.argmin(ordered.sum(axis=1)))
    return np.roll(ordered, -start, axis=0)


def _dlt_homography(src: Array, dst: Array) -> Array:
    """Exact 4-point homography by direct linear transform (no OpenCV needed)."""
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    h = vt[-1].reshape(3, 3)
    if abs(h[2, 2]) > 1e-12:
        h = h / h[2, 2]
    return h


# ------------------------------------------------------------------ bbox helper


@dataclass(frozen=True)
class Box:
    """A detection box in frame pixels, plus the derived paw point."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 0.0
    label: str = "cat"
    clipped_bottom: bool = field(default=False)

    @property
    def paw_point(self) -> tuple[float, float]:
        """Bottom-centre: the best cheap proxy for where the animal is standing."""
        return ((self.x1 + self.x2) / 2.0, self.y2)

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def width(self) -> float:
        return self.x2 - self.x1
