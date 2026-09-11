"""The fine-tuning pipeline: label mapping, leakage, and the shipping gate.

The gate tests matter most — it is the only thing standing between an
auto-updating app and a silently worse detector.
"""

import numpy as np
import pytest

from surfaceguard.storage.activity_log import Event
from surfaceguard.training import dataset, evaluate
from surfaceguard.training.dataset import CAT, PERSON


def ev(i, verdict, box=(10.0, 20.0, 110.0, 220.0), thumb="t.jpg", strip=("a.jpg", "b.jpg")):
    return Event(id=i, ts=0.0, surface_id="s1", surface_name="Counter", fired=True,
                 reason="r", gates=[], score=0.9, box=box, latency_ms=None,
                 thumbnail=thumb, feedback=verdict, map_point=(1.0, 2.0), strip=list(strip))


@pytest.fixture
def picture_dir(tmp_path):
    import cv2

    for name in ("t.jpg", "a.jpg", "b.jpg"):
        cv2.imwrite(str(tmp_path / name), np.full((240, 320, 3), 128, np.uint8))
    return tmp_path


# ------------------------------------------------------------- label mapping


def test_correct_becomes_a_positive_cat_box(picture_dir):
    out, stats = dataset.collect([ev(1, "correct")], picture_dir)
    assert stats.cat_boxes == 1
    assert out[0].boxes[0][0] == CAT


def test_not_on_surface_is_a_positive_for_the_detector(picture_dir):
    """A negative for the app — the cat was not on the counter — but the detector
    was right that there was a cat, so it is a positive label."""
    out, stats = dataset.collect([ev(2, "not_on_surface")], picture_dir)
    assert stats.cat_boxes == 1 and out[0].boxes[0][0] == CAT


def test_person_becomes_a_person_box(picture_dir):
    out, stats = dataset.collect([ev(3, "person")], picture_dir)
    assert stats.person_boxes == 1 and out[0].boxes[0][0] == PERSON


def test_not_a_cat_becomes_a_hard_negative_with_no_boxes(picture_dir):
    """The most valuable label: an image the detector fired on that had no cat."""
    out, stats = dataset.collect([ev(4, "not_a_cat")], picture_dir)
    assert stats.negatives >= 1
    assert all(e.is_negative for e in out)


def test_missed_without_a_proposal_is_counted_not_guessed(picture_dir):
    """A cat with no box cannot be invented; it waits for a confirmed proposal."""
    out, stats = dataset.collect([ev(5, "missed")], picture_dir)
    assert out == [] and stats.needs_proposal == 1


def test_missed_with_a_confirmed_proposal_is_usable(picture_dir):
    proposals = {6: [(CAT, (12.0, 22.0, 100.0, 200.0))]}
    out, stats = dataset.collect([ev(6, "missed")], picture_dir, proposals)
    assert stats.cat_boxes == 1 and out[0].proposed is True


def test_unsure_and_unreviewed_are_excluded(picture_dir):
    out, stats = dataset.collect([ev(7, "unsure"), ev(8, None)], picture_dir)
    assert out == []
    assert stats.skipped["unsure"] == 1 and stats.skipped["unreviewed"] == 1


def test_an_event_with_no_picture_is_skipped(tmp_path):
    out, stats = dataset.collect([ev(9, "correct")], tmp_path)
    assert out == [] and stats.skipped["no picture"] == 1


# ------------------------------------------------------------------ leakage


def test_frames_from_one_event_never_straddle_the_split(picture_dir):
    """Correlated frames in both train and val would make the gate meaningless."""
    examples = [dataset.Example(picture_dir / "t.jpg", [(CAT, 1.0, 2.0, 3.0, 4.0)], event_id=i)
                for i in range(10) for _ in range(3)]
    train, val = dataset.split(examples, val_fraction=0.3)
    assert {e.event_id for e in train}.isdisjoint({e.event_id for e in val})
    assert val, "the split produced no validation set"


def test_yolo_conversion_clamps_and_normalises():
    cx, cy, w, h = dataset.to_yolo((-50, -10, 700, 500), 640, 480)
    assert (cx, cy, w, h) == (0.5, 0.5, 1.0, 1.0)


def test_readiness_is_honest_about_small_datasets():
    assert not dataset.DatasetStats(images=50, cat_boxes=20).ready
    assert dataset.DatasetStats(images=300, cat_boxes=200).ready


# ------------------------------------------------------------------ the gate


class FakeDetector:
    """Returns canned boxes so the gate's arithmetic can be tested exactly."""

    def __init__(self, boxes_by_image):
        self._by = boxes_by_image

    def detect(self, _frame):
        return self._by


def _example(path, boxes):
    return dataset.Example(image=path, boxes=boxes)


def test_gate_refuses_without_a_control_set():
    v = evaluate.gate("a.onnx", "b.onnx", [object()], None)
    assert not v.passed and "control set" in v.reasons[0]


def test_gate_refuses_without_held_out_data():
    v = evaluate.gate("a.onnx", "b.onnx", [], [object()])
    assert not v.passed


def test_gate_blocks_a_candidate_that_barely_improves(monkeypatch):
    """Shipping a model for +0.005 F1 is churn, not progress."""
    scores = iter([
        evaluate.Score(tp=10, fp=2, fn=2),    # incumbent, her data
        evaluate.Score(tp=10, fp=2, fn=2),    # candidate,  her data (identical)
        evaluate.Score(tp=10, fp=1, fn=1),    # incumbent, control
        evaluate.Score(tp=10, fp=1, fn=1),    # candidate, control
    ])
    monkeypatch.setattr(evaluate, "OnnxDetector", lambda *a, **k: object())
    monkeypatch.setattr(evaluate, "score_model", lambda d, e: next(scores))
    v = evaluate.gate("a", "b", [object()], [object()])
    assert not v.passed and "worth the risk" in v.reasons[0]


def test_gate_blocks_a_candidate_that_forgot_other_cats(monkeypatch):
    """The catastrophic-forgetting case: great on hers, worse on everyone else's."""
    scores = iter([
        evaluate.Score(tp=6, fp=4, fn=4),     # incumbent, her data  (F1 0.60)
        evaluate.Score(tp=10, fp=0, fn=0),    # candidate,  her data  (F1 1.00)
        evaluate.Score(tp=10, fp=0, fn=0),    # incumbent, control    (F1 1.00)
        evaluate.Score(tp=4, fp=6, fn=6),     # candidate, control    (F1 0.40)
    ])
    monkeypatch.setattr(evaluate, "OnnxDetector", lambda *a, **k: object())
    monkeypatch.setattr(evaluate, "score_model", lambda d, e: next(scores))
    v = evaluate.gate("a", "b", [object()], [object()])
    assert not v.passed
    assert any("forgetting" in r for r in v.reasons)


def test_gate_passes_a_genuine_improvement(monkeypatch):
    scores = iter([
        evaluate.Score(tp=6, fp=4, fn=4),     # incumbent, her data
        evaluate.Score(tp=9, fp=1, fn=1),     # candidate,  her data
        evaluate.Score(tp=10, fp=1, fn=1),    # incumbent, control
        evaluate.Score(tp=10, fp=1, fn=1),    # candidate, control (unchanged)
    ])
    monkeypatch.setattr(evaluate, "OnnxDetector", lambda *a, **k: object())
    monkeypatch.setattr(evaluate, "score_model", lambda d, e: next(scores))
    v = evaluate.gate("a", "b", [object()], [object()])
    assert v.passed, v.reasons
    assert "F1 on her data" in v.reasons[0]


def test_f1_arithmetic():
    s = evaluate.Score(tp=8, fp=2, fn=2)
    assert s.precision == pytest.approx(0.8)
    assert s.recall == pytest.approx(0.8)
    assert s.f1 == pytest.approx(0.8)
    assert evaluate.Score().f1 == 0.0


# ------------------------------------------------- the four gaps, closed


def test_general_images_go_into_train_only(tmp_path):
    """Mixing them into validation would hide the regression val exists to catch."""
    import json

    import cv2

    frame = tmp_path / "f.jpg"
    cv2.imwrite(str(frame), np.full((240, 320, 3), 120, np.uint8))
    hers = [dataset.Example(frame, [(CAT, 10.0, 10.0, 60.0, 80.0)], event_id=i)
            for i in range(20)]
    general = [dataset.Example(frame, [(CAT, 20.0, 20.0, 90.0, 110.0)], event_id=-i)
               for i in range(1, 60)]
    dataset.write(hers, tmp_path / "ds", mix=general, mix_ratio=0.3)
    manifest = json.loads((tmp_path / "ds" / "manifest.json").read_text())
    assert manifest["general_images_mixed_in"] > 0
    # Val is drawn from her events only; the mix is appended to train afterwards.
    assert manifest["val"] == len({e.event_id for e in hers}) // 5


def test_general_mix_is_capped(tmp_path):
    """A mix that swamps her data trains a general model, not hers."""
    import json

    import cv2

    frame = tmp_path / "f.jpg"
    cv2.imwrite(str(frame), np.full((240, 320, 3), 120, np.uint8))
    hers = [dataset.Example(frame, [(CAT, 1.0, 1.0, 9.0, 9.0)], event_id=i) for i in range(10)]
    general = [dataset.Example(frame, [(CAT, 1.0, 1.0, 9.0, 9.0)], event_id=-i)
               for i in range(1, 500)]
    dataset.write(hers, tmp_path / "ds", mix=general, mix_ratio=5.0)   # absurd request
    manifest = json.loads((tmp_path / "ds" / "manifest.json").read_text())
    her_train = manifest["train"] - manifest["general_images_mixed_in"]
    assert manifest["general_images_mixed_in"] <= her_train * dataset.MAX_MIX_RATIO + 1


def test_the_training_script_disables_telemetry_before_importing_yolo():
    """Ultralytics posts analytics by default; 'it stays on your Mac' has to mean it."""
    from surfaceguard.training import finetune

    source = finetune.finetune.__doc__ or ""
    import inspect

    body = inspect.getsource(finetune.finetune)
    assert '"sync": False' in body
    assert body.index('SETTINGS.update') < body.index("from ultralytics import YOLO")
    assert "TELEMETRY_STILL_ON" in body, "no check that the setting actually took"


def test_training_refuses_when_telemetry_could_not_be_turned_off(monkeypatch, tmp_path):
    import subprocess

    from surfaceguard.training import finetune

    monkeypatch.setattr(finetune, "build_kit", lambda work, progress=None: tmp_path / "py")
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 1, "TELEMETRY_STILL_ON", ""),
    )
    result = finetune.finetune(tmp_path / "data.yaml", out_dir=tmp_path)
    assert not result.ok
    assert "analytics" in result.message


def test_control_set_parsing_drops_crowds_and_other_species():
    """A crowd box around six cats teaches the wrong shape and unfairly fails a candidate."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("fcs", "tools/fetch_control_set.py")
    fcs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fcs)

    annotations = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 640, "height": 480},
                   {"id": 2, "file_name": "b.jpg", "width": 100, "height": 100}],
        "annotations": [
            {"image_id": 1, "category_id": 17, "bbox": [100, 100, 200, 150], "iscrowd": 0},
            {"image_id": 1, "category_id": 18, "bbox": [0, 0, 50, 50], "iscrowd": 0},
            {"image_id": 2, "category_id": 17, "bbox": [0, 0, 100, 100], "iscrowd": 1},
        ],
    }
    kept = fcs.cat_images(annotations)
    assert [img["file_name"] for img, _ in kept] == ["a.jpg"]
    assert len(kept[0][1]) == 1, "the dog was kept as a cat"
