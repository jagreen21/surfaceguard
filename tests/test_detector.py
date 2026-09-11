"""The ONNX detection path.

Two levels. The shape tests fabricate a YOLOv8 output tensor and always run, so a
regression in the transpose/NMS/letterbox maths is caught anywhere. The
ground-truth test runs the real model against a reference image and compares with
Ultralytics' own inference, and skips cleanly where the model is not installed.
"""

import numpy as np
import pytest

from surfaceguard.detection.cat_detector import (
    COCO_CAT,
    COCO_PERSON,
    Detector,
    OnnxDetector,
    bundled_model_path,
)
from surfaceguard.geometry.projection import Box

REFERENCE_IMAGE = "tests/assets/bus.jpg"

# What Ultralytics itself reports for bus.jpg at conf=0.35, imgsz=640, per model.
# Regenerate with packaging/export_model.py if a model is ever re-exported.
ULTRALYTICS_BUS = {
    "yolov8n.onnx": [
        ("person", 0.891, (670, 381, 810, 880)),
        ("person", 0.883, (222, 407, 344, 856)),
        ("person", 0.878, (51, 397, 244, 905)),
        ("person", 0.436, (0, 550, 58, 868)),
    ],
    "yolov8s.onnx": [
        ("person", 0.915, (50, 398, 247, 904)),
        ("person", 0.885, (666, 393, 809, 879)),
        ("person", 0.846, (223, 407, 348, 859)),
        ("person", 0.608, (0, 551, 75, 874)),
    ],
    "yolov8m.onnx": [
        ("person", 0.937, (50, 400, 248, 903)),
        ("person", 0.908, (223, 410, 345, 860)),
        ("person", 0.888, (668, 395, 809, 881)),
        ("person", 0.654, (1, 548, 78, 872)),
    ],
}


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union else 0.0


model_required = pytest.mark.skipif(
    bundled_model_path() is None,
    reason="no detection model installed; run packaging/export_model.py",
)


# ------------------------------------------------------------------ shape maths


class _FakeSession:
    """Stands in for onnxruntime so the decode maths can be tested in isolation."""

    def __init__(self, raw):
        self._raw = raw

    def get_inputs(self):
        return [type("I", (), {"name": "images"})()]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, _outputs, _feed):
        return [self._raw]


def _detector_with(raw) -> OnnxDetector:
    det = OnnxDetector.__new__(OnnxDetector)
    det._session = _FakeSession(raw)
    det._input = "images"
    det.input_size = 640
    det.conf = 0.35
    det.iou = 0.45
    det._path = __import__("pathlib").Path("fake.onnx")
    return det


def _yolo_output(entries, n_boxes=8400, n_classes=80):
    """Build a (1, 4+nc, n) tensor the way YOLOv8 emits one."""
    out = np.zeros((1, 4 + n_classes, n_boxes), np.float32)
    for i, (cx, cy, w, h, cls, score) in enumerate(entries):
        out[0, 0, i], out[0, 1, i], out[0, 2, i], out[0, 3, i] = cx, cy, w, h
        out[0, 4 + cls, i] = score
    return out


def test_decoder_handles_the_transposed_yolov8_layout():
    """(1, 84, 8400) must be transposed before the columns mean anything."""
    raw = _yolo_output([(320.0, 320.0, 100.0, 200.0, COCO_CAT, 0.9)])
    det = _detector_with(raw)
    boxes = det.detect(np.zeros((640, 640, 3), np.uint8))
    assert len(boxes) == 1
    box = boxes[0]
    assert box.label == "cat" and box.score == pytest.approx(0.9, abs=1e-3)
    assert box.x1 == pytest.approx(270, abs=1) and box.x2 == pytest.approx(370, abs=1)
    assert box.y1 == pytest.approx(220, abs=1) and box.y2 == pytest.approx(420, abs=1)


def test_decoder_maps_letterboxed_coordinates_back_to_the_original_frame():
    """A non-square frame is padded before inference; boxes must come back unpadded."""
    raw = _yolo_output([(320.0, 320.0, 64.0, 64.0, COCO_PERSON, 0.8)])
    det = _detector_with(raw)
    wide = np.zeros((360, 1280, 3), np.uint8)          # 16:9, heavily letterboxed
    box = det.detect(wide)[0]
    centre_x, centre_y = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
    assert centre_x == pytest.approx(640, abs=2), "x should map to the frame centre"
    assert centre_y == pytest.approx(180, abs=2), "y should map to the frame centre"
    assert 0 <= box.y1 and box.y2 <= 360


def test_decoder_only_returns_cats_and_people():
    """80 COCO classes exist; this app must ignore 78 of them."""
    raw = _yolo_output([
        (100.0, 100.0, 50.0, 50.0, COCO_CAT, 0.9),
        (200.0, 200.0, 50.0, 50.0, COCO_PERSON, 0.9),
        (300.0, 300.0, 50.0, 50.0, 16, 0.99),      # dog
        (400.0, 400.0, 50.0, 50.0, 5, 0.99),       # bus
    ])
    boxes = _detector_with(raw).detect(np.zeros((640, 640, 3), np.uint8))
    assert sorted(b.label for b in boxes) == ["cat", "person"]


def test_decoder_drops_low_confidence():
    raw = _yolo_output([(320.0, 320.0, 50.0, 50.0, COCO_CAT, 0.20)])
    assert _detector_with(raw).detect(np.zeros((640, 640, 3), np.uint8)) == []


def test_decoder_suppresses_duplicate_boxes():
    """Without NMS the same cat triggers several times in one frame."""
    raw = _yolo_output([
        (320.0, 320.0, 100.0, 100.0, COCO_CAT, 0.90),
        (322.0, 321.0, 102.0, 99.0, COCO_CAT, 0.85),
        (318.0, 319.0, 98.0, 101.0, COCO_CAT, 0.80),
    ])
    boxes = _detector_with(raw).detect(np.zeros((640, 640, 3), np.uint8))
    assert len(boxes) == 1


def test_clipped_boxes_are_flagged_for_the_occlusion_gate():
    """A box running off the bottom has an unreliable paw point (§6)."""
    raw = _yolo_output([(320.0, 620.0, 100.0, 200.0, COCO_CAT, 0.9)])
    box = _detector_with(raw).detect(np.zeros((640, 640, 3), np.uint8))[0]
    assert box.clipped_bottom


# ------------------------------------------------------------- real model


@model_required
def test_real_model_loads_and_reports_itself():
    det = OnnxDetector(bundled_model_path())
    assert det.info.available and det.info.backend == "onnxruntime"
    assert det.info.providers, "no execution provider was selected"


@model_required
@pytest.mark.skipif(not __import__("pathlib").Path(REFERENCE_IMAGE).exists(),
                    reason="reference image not present")
def test_real_model_matches_ultralytics_on_the_reference_image():
    """The decoder is only correct if it agrees with the exporter's own inference.

    Boxes are matched by overlap, not by score order: two detections can tie on
    confidence and come back in either order.
    """
    import cv2

    model = bundled_model_path()
    reference = ULTRALYTICS_BUS.get(model.name)
    if reference is None:
        pytest.skip(f"no reference output recorded for {model.name}")

    det = OnnxDetector(model, conf=0.35)
    boxes = det.detect(cv2.imread(REFERENCE_IMAGE))
    _cats, people = Detector.split(boxes)
    assert len(people) == len(reference), f"expected {len(reference)} people, got {len(people)}"

    for _label, ref_score, ref_box in reference:
        best = max(people, key=lambda b: iou((b.x1, b.y1, b.x2, b.y2), ref_box))
        overlap = iou((best.x1, best.y1, best.x2, best.y2), ref_box)
        assert overlap > 0.98, f"box {ref_box} only matched at IoU {overlap:.3f}"
        assert abs(best.score - ref_score) < 0.02


@model_required
def test_real_model_is_inside_the_inference_budget():
    """§7 budgets 40 ms for inference."""
    import time

    import cv2

    det = OnnxDetector(bundled_model_path())
    frame = (cv2.imread(REFERENCE_IMAGE) if __import__("pathlib").Path(REFERENCE_IMAGE).exists()
             else np.random.randint(0, 255, (540, 960, 3), dtype=np.uint8))
    for _ in range(3):
        det.detect(frame)                              # warm up CoreML compilation
    times = []
    for _ in range(8):
        t0 = time.perf_counter()
        det.detect(frame)
        times.append((time.perf_counter() - t0) * 1e3)
    # Best of N, not the median. The question is whether the model *can* meet the
    # budget; contention from the rest of the suite only ever adds time, so a
    # median here measures the test runner's load as much as the detector and
    # fails at random. The floor is the honest estimate.
    best = min(times)
    assert best <= 40.0, (
        f"fastest of {len(times)} runs was {best:.0f} ms (budget 40 ms); "
        f"median {sorted(times)[len(times) // 2]:.0f} ms"
    )


# ------------------------------------------------------------------ selection


class _Syntheticish:
    def ground_truth_boxes(self):
        return []


@model_required
def test_a_synthetic_camera_keeps_the_synthetic_detector():
    """The real model sees nothing in a drawn room, which would make the demo
    silently useless for showing what a trigger looks like."""
    from surfaceguard.detection.cat_detector import SyntheticDetector, load_detector

    assert isinstance(load_detector(None, _Syntheticish()), SyntheticDetector)


@model_required
def test_a_real_camera_gets_the_bundled_model_with_no_configuration():
    from surfaceguard.detection.cat_detector import OnnxDetector, load_detector

    class RealCamera:
        pass

    det = load_detector(None, RealCamera())
    assert isinstance(det, OnnxDetector) and det.info.available


def test_no_model_and_no_synthetic_camera_is_an_error_not_a_silent_fallback(monkeypatch):
    import surfaceguard.detection.cat_detector as mod

    monkeypatch.setattr(mod, "bundled_model_path", lambda: None)

    class RealCamera:
        pass

    with pytest.raises(RuntimeError, match="nothing can be detected"):
        mod.load_detector(None, RealCamera())
