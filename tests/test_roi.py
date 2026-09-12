"""Region-of-interest inference.

From 1080p, letterboxing the whole frame to 640 is a 3x downscale: a 70 px cat
reaches the network at ~23 px, near the floor of what a small-object detector
resolves. Cropping to the surfaces puts the same pixels in front of the model
several times larger, which is what keeps a small model viable on an Intel laptop.
"""

import numpy as np
import pytest

from surfaceguard.detection import roi
from surfaceguard.geometry.projection import Box, Pose
from surfaceguard.geometry.surface import Surface

POSE = Pose.identity((1920, 1080))


def counter(calibrated: bool = True) -> Surface:
    s = Surface("Counter", np.array([[820, 560], [1180, 560], [1240, 700], [760, 700]], float))
    if calibrated:
        for _ in range(10):
            s.observe_height(POSE, Box(900, 490, 980, 560))     # a 70 px cat
    return s


def test_cropping_magnifies_the_surface():
    region = roi.for_surfaces([counter()], POSE)
    assert not region.full_frame
    assert region.magnification > 2.0, f"only {region.magnification:.2f}x"
    reaching = 70 * 640 / max(region.width, region.height)
    assert reaching > 50, f"a 70 px cat still only reaches the network at {reaching:.0f} px"


def test_the_crop_contains_the_surface_with_room_around_it():
    surface = counter()
    region = roi.for_surfaces([surface], POSE)
    points = surface.project(POSE)
    assert region.x1 <= points[:, 0].min() and region.x2 >= points[:, 0].max()
    assert region.y1 <= points[:, 1].min() and region.y2 >= points[:, 1].max()
    # Padded, because a cat arrives from outside the surface.
    assert region.x1 < points[:, 0].min() - 10


def test_detections_map_back_to_frame_coordinates():
    region = roi.for_surfaces([counter()], POSE)
    mapped = region.to_frame(Box(10, 20, 90, 100, score=0.9))
    assert mapped.x1 == 10 + region.x1
    assert mapped.y2 == 100 + region.y1
    assert mapped.score == 0.9


def test_the_crop_never_leaves_the_frame():
    edge = Surface("Edge", np.array([[0, 0], [200, 0], [200, 120], [0, 120]], float))
    for _ in range(10):
        edge.observe_height(POSE, Box(20, 50, 80, 120))
    region = roi.for_surfaces([edge], POSE)
    assert region.x1 >= 0 and region.y1 >= 0
    assert region.x2 <= 1920 and region.y2 <= 1080


def test_it_falls_back_to_the_whole_frame_when_cropping_buys_nothing():
    assert roi.for_surfaces([], POSE).full_frame, "no surfaces"
    wall = Surface("Wall", np.array([[10, 10], [1900, 10], [1900, 1070], [10, 1070]], float))
    assert roi.for_surfaces([wall], POSE).full_frame, "surface fills the frame"
    disabled = counter()
    disabled.enabled = False
    assert roi.for_surfaces([disabled], POSE).full_frame, "surface disabled"


def test_an_uncalibrated_surface_still_gets_a_sane_crop():
    """With no plane model there is no predicted cat height, so the padding falls
    back to a share of the surface rather than an invented number."""
    region = roi.for_surfaces([counter(calibrated=False)], POSE)
    assert not region.full_frame
    assert region.width >= roi.MIN_CROP_PX and region.height >= roi.MIN_CROP_PX


def test_several_surfaces_are_covered_by_one_crop():
    left = counter()
    right = Surface("Table", np.array([[1400, 600], [1700, 600], [1720, 720], [1380, 720]], float))
    for _ in range(10):
        right.observe_height(POSE, Box(1450, 530, 1520, 600))
    region = roi.for_surfaces([left, right], POSE)
    for surface in (left, right):
        points = surface.project(POSE)
        assert region.x1 <= points[:, 0].min() and region.x2 >= points[:, 0].max()


def test_cropping_an_image_gives_the_right_pixels():
    image = np.zeros((1080, 1920, 3), np.uint8)
    image[600:650, 900:950] = 255
    region = roi.for_surfaces([counter()], POSE)
    cropped = region.crop(image)
    assert cropped.shape[0] == region.height and cropped.shape[1] == region.width
    assert cropped.max() == 255, "the marked pixels were cropped away"


def test_the_detector_contract_is_crop_relative():
    """A detector returns boxes in the coordinates of the image it was given.
    SyntheticDetector holds full-frame ground truth, so it needs the origin to
    honour that — without it the engine offsets already-absolute boxes twice."""
    from surfaceguard.camera.sources.synthetic import SyntheticCamera
    from surfaceguard.detection.cat_detector import SyntheticDetector

    camera = SyntheticCamera()
    camera.start()
    try:
        camera.place_cat(600.0, 300.0, 70.0) if hasattr(camera, "place_cat") else None
        detector = SyntheticDetector(camera, jitter_px=0.0, miss_rate=0.0)
        frame = camera.read(timeout=2.0)
        assert frame is not None
        whole = detector.detect(frame.image)
        shifted = detector.detect(frame.image, (100, 50))
        if whole and shifted:
            assert shifted[0].x1 == pytest.approx(whole[0].x1 - 100, abs=1e-6)
            assert shifted[0].y1 == pytest.approx(whole[0].y1 - 50, abs=1e-6)
    finally:
        camera.stop()
