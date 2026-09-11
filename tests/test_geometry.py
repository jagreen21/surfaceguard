"""Geometry: the part that must be right before anything else can be."""

import numpy as np
import pytest

from surfaceguard.geometry.projection import (
    Box,
    Pose,
    clip_polygon_to_rect,
    perspective_weight,
    point_in_polygon,
    polygon_area,
    quad_to_plane_homography,
    transform_points,
    visible_fraction,
)
from surfaceguard.geometry.surface import MIN_CALIBRATION_SAMPLES, Surface

SQUARE = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
# A counter in perspective: the far edge is shorter than the near edge.
COUNTER = np.array([[220, 300], [420, 300], [520, 420], [120, 420]], float)


def test_area_and_containment():
    assert polygon_area(SQUARE) == pytest.approx(100.0)
    assert point_in_polygon(SQUARE, (5, 5))
    assert not point_in_polygon(SQUARE, (11, 5))
    assert point_in_polygon(SQUARE, (0, 0)), "a point on the edge counts as inside"


def test_concave_polygon_containment():
    # An L-shape: the notch must not read as inside.
    ell = np.array([[0, 0], [10, 0], [10, 4], [4, 4], [4, 10], [0, 10]], float)
    assert point_in_polygon(ell, (2, 8))
    assert not point_in_polygon(ell, (8, 8))


def test_clipping_and_visible_fraction():
    assert visible_fraction(SQUARE, (100, 100)) == pytest.approx(1.0)
    assert visible_fraction(SQUARE, (5, 100)) == pytest.approx(0.5)
    off = np.array([[-30, -30], [-20, -30], [-20, -20], [-30, -20]], float)
    assert visible_fraction(off, (100, 100)) == 0.0
    assert len(clip_polygon_to_rect(off, 100, 100)) == 0


def test_quad_homography_maps_corners_to_unit_square():
    h = quad_to_plane_homography(COUNTER)
    mapped = transform_points(h, COUNTER)
    expected = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    assert np.allclose(mapped, expected, atol=1e-6)


def test_perspective_weight_is_positive_and_grows_towards_the_camera():
    """Sign normalisation matters: w must read as 'apparent size', not its negative."""
    h = quad_to_plane_homography(COUNTER)
    far = float(perspective_weight(h, [(320, 305)])[0])
    near = float(perspective_weight(h, [(320, 415)])[0])
    assert far > 0 and near > 0
    assert near > far


def test_scale_model_predicts_a_larger_cat_nearer_the_camera():
    s = Surface("Counter", COUNTER)
    pose = Pose.identity((640, 480))
    for _ in range(10):
        s.observe_height(pose, Box(280, 300, 360, 360))
    far = s.expected_height(pose, (320, 305))
    near = s.expected_height(pose, (320, 415))
    assert far is not None and near is not None
    assert near / far == pytest.approx(1.88, abs=0.05)


def test_scale_gate_has_no_prior_before_calibration():
    """There is no honest default for real-world size; None, never a guess."""
    s = Surface("Counter", COUNTER)
    assert s.expected_height(Pose.identity((640, 480)), (320, 360)) is None
    assert not s.scale_gate_is_required


def test_scale_gate_becomes_required_only_after_enough_samples():
    s = Surface("Counter", COUNTER)
    pose = Pose.identity((640, 480))
    for _ in range(MIN_CALIBRATION_SAMPLES - 1):
        s.observe_height(pose, Box(280, 300, 360, 360))
    assert not s.scale_gate_is_required
    s.observe_height(pose, Box(280, 300, 360, 360))
    assert s.scale_gate_is_required


def test_median_calibration_resists_one_bad_sample():
    s = Surface("Counter", COUNTER)
    pose = Pose.identity((640, 480))
    for _ in range(9):
        s.observe_height(pose, Box(280, 300, 360, 360))       # 60 px tall
    good = s.size_scale
    s.observe_height(pose, Box(280, 100, 360, 360))           # 260 px: a mid-leap cat
    assert s.size_scale == pytest.approx(good, rel=0.02)


def test_forget_last_observation_undoes_a_false_positive():
    s = Surface("Counter", COUNTER)
    pose = Pose.identity((640, 480))
    s.observe_height(pose, Box(280, 300, 360, 360))
    assert s.calibration_samples == 1
    s.forget_last_observation()
    assert s.calibration_samples == 0
    assert s.size_scale is None


def test_paw_point_is_bottom_centre():
    assert Box(10, 20, 50, 80).paw_point == (30.0, 80.0)
