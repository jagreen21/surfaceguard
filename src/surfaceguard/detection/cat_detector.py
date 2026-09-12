"""Cat (and person) detection.

Two backends. :class:`OnnxDetector` runs a YOLO-family ONNX model through
onnxruntime, which on Apple silicon uses CoreML without dragging PyTorch into the
bundle. :class:`SyntheticDetector` reads the shapes drawn by the synthetic camera,
so the full pipeline — gates, policy, UI, heartbeat — can be run and regression
tested with no model and no hardware.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..geometry.projection import Box

# COCO indices for the only two classes this app cares about.
COCO_CAT = 15
COCO_PERSON = 0

# yolov8m, not yolov8n. Measured on this machine at 960x540 with CoreML:
#   n  13 MB   9.2 ms   ~37 COCO mAP
#   s  45 MB  12.0 ms   ~45
#   m 104 MB  17.7 ms   ~50
# The §7 budget is 40 ms, so the largest of the three fits with room to spare, and
# accuracy matters more here than bundle size: she downloads the app once, and
# model updates ship on their own channel afterwards.
MODEL_FILENAMES = ("yolov8m.onnx", "yolov8s.onnx", "yolov8n.onnx")
MODEL_FILENAME = MODEL_FILENAMES[0]


def worker_threads() -> int:
    """How many cores inference may use.

    Left alone, onnxruntime takes every core and OpenCV does the same, on a
    machine that is also running the Qt interface, ORB registration and a Node
    bridge. On the four-core i5 this ships to, that means the UI stutters and the
    camera service is starved while a frame is being judged. Leaving a core free
    costs a little inference time and buys a responsive app.
    """
    import os

    cores = os.cpu_count() or 4
    return max(1, min(cores - 1, 4))


def _session_options():
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = worker_threads()
    # One graph at a time; parallel branches would fight the same cores.
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return options


def bundled_model_path() -> Path | None:
    """Find the detection model without anyone having to configure a path.

    Inside the .app it sits in Resources; in a checkout it sits in models/. She
    should never have to know either of those things.
    """
    # A model delivered by an update outranks the bundled one: it is newer by
    # definition, and it must not require reinstalling the app to take effect.
    roots = [Path.home() / "Library" / "Application Support" / "SurfaceGuard" / "models"]
    frozen = getattr(sys, "_MEIPASS", None)
    if frozen:
        roots += [Path(frozen) / "models",
                  Path(sys.executable).resolve().parent.parent / "Resources" / "models"]
    roots.append(Path(__file__).resolve().parents[3] / "models")
    # Preference order, so a hand-dropped smaller model is still picked up if the
    # bundled one is removed to save space.
    for name in MODEL_FILENAMES:
        for root in roots:
            if (root / name).exists():
                return root / name
    return None


@dataclass
class DetectorInfo:
    name: str
    backend: str
    input_size: int
    providers: tuple[str, ...] = ()
    available: bool = True
    note: str = ""


class Detector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Box]:
        """Return cat and person boxes in full-resolution frame pixels."""

    @property
    @abstractmethod
    def info(self) -> DetectorInfo:
        ...

    @staticmethod
    def split(boxes: list[Box]) -> tuple[list[Box], list[Box]]:
        cats = [b for b in boxes if b.label == "cat"]
        people = [b for b in boxes if b.label == "person"]
        return cats, people


# --------------------------------------------------------------------- ONNX


class OnnxDetector(Detector):
    """YOLOv8-style ONNX model. Output is expected as (1, 4 + n_classes, n_boxes)."""

    def __init__(
        self,
        model_path: str | Path,
        input_size: int = 640,
        conf: float = 0.35,
        iou: float = 0.45,
    ) -> None:
        import onnxruntime as ort  # imported lazily: the UI must start without it

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"No detection model at {path}. Drop a YOLOv8 ONNX export there "
                "(e.g. yolov8n.onnx) or run with the synthetic detector."
            )
        preferred = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        available = ort.get_available_providers()
        self._session = ort.InferenceSession(
            str(path), sess_options=_session_options(),
            providers=[p for p in preferred if p in available] or None,
        )
        self._input = self._session.get_inputs()[0].name
        self.input_size = input_size
        self.conf = conf
        self.iou = iou
        self._path = path

    @property
    def info(self) -> DetectorInfo:
        return DetectorInfo(
            self._path.name, "onnxruntime", self.input_size,
            tuple(self._session.get_providers()),
        )

    def detect(self, image: np.ndarray) -> list[Box]:
        blob, scale, pad = self._letterbox(image)
        raw = self._session.run(None, {self._input: blob})[0]
        return self._decode(raw, scale, pad, image.shape[:2])

    def _letterbox(self, image: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        h, w = image.shape[:2]
        s = min(self.input_size / w, self.input_size / h)
        nw, nh = int(round(w * s)), int(round(h * s))
        canvas = np.full((self.input_size, self.input_size, 3), 114, np.uint8)
        px, py = (self.input_size - nw) // 2, (self.input_size - nh) // 2
        canvas[py:py + nh, px:px + nw] = cv2.resize(image, (nw, nh))
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return blob.transpose(2, 0, 1)[None], s, (px, py)

    def _decode(
        self, raw: np.ndarray, scale: float, pad: tuple[int, int], shape: tuple[int, int]
    ) -> list[Box]:
        pred = np.squeeze(raw)
        if pred.shape[0] < pred.shape[1]:   # (4 + nc, n) -> (n, 4 + nc)
            pred = pred.T
        boxes_xywh, scores = pred[:, :4], pred[:, 4:]
        wanted = {COCO_CAT: "cat", COCO_PERSON: "person"}
        out: list[Box] = []
        h, w = shape
        px, py = pad
        for cls, label in wanted.items():
            if cls >= scores.shape[1]:
                continue
            col = scores[:, cls]
            keep = np.nonzero(col >= self.conf)[0]
            if not len(keep):
                continue
            xywh = boxes_xywh[keep]
            rects = np.stack([
                (xywh[:, 0] - xywh[:, 2] / 2 - px) / scale,
                (xywh[:, 1] - xywh[:, 3] / 2 - py) / scale,
                xywh[:, 2] / scale,
                xywh[:, 3] / scale,
            ], axis=1)
            idx = cv2.dnn.NMSBoxes(rects.tolist(), col[keep].tolist(), self.conf, self.iou)
            for i in np.asarray(idx).reshape(-1):
                x, y, bw, bh = rects[int(i)]
                x1, y1 = max(0.0, float(x)), max(0.0, float(y))
                x2, y2 = min(float(w), float(x + bw)), min(float(h), float(y + bh))
                out.append(Box(x1, y1, x2, y2, float(col[keep][int(i)]), label,
                               clipped_bottom=y2 >= h - 1.5))
        return out


# ---------------------------------------------------------------- synthetic


class SyntheticDetector(Detector):
    """Reads what :class:`~surfaceguard.camera.sources.synthetic.SyntheticCamera` drew.

    Deliberately imperfect rather than oracular: boxes are jittered, a fraction of
    frames miss entirely, and the paw edge can be nudged. That keeps dwell,
    hysteresis and the scale tolerance under realistic pressure without needing a
    model. Colour thresholding was tried first and picked up the room's own
    furniture, which is exactly the false-positive class the gates exist for — but
    not a useful one to bake into a test fixture.
    """

    def __init__(
        self,
        camera,
        jitter_px: float = 3.0,
        miss_rate: float = 0.04,
        seed: int = 3,
    ) -> None:
        self._camera = camera
        self.jitter_px = jitter_px
        self.miss_rate = miss_rate
        self._rng = np.random.default_rng(seed)

    @property
    def info(self) -> DetectorInfo:
        return DetectorInfo(
            "synthetic", "ground-truth + noise", 0,
            note=f"test backend, {self.jitter_px:.0f} px jitter, {self.miss_rate:.0%} miss rate",
        )

    def detect(self, image: np.ndarray) -> list[Box]:
        h, w = image.shape[:2]
        out: list[Box] = []
        for label, x1, y1, x2, y2 in self._camera.ground_truth_boxes():
            if self._rng.random() < self.miss_rate:
                continue
            j = self._rng.normal(0.0, self.jitter_px, 4)
            bx1, by1 = float(x1 + j[0]), float(y1 + j[1])
            bx2, by2 = float(x2 + j[2]), float(y2 + j[3])
            if bx2 <= 0 or by2 <= 0 or bx1 >= w or by1 >= h or bx2 - bx1 < 6:
                continue  # off-frame, like a real detector
            cx1, cy1 = max(0.0, bx1), max(0.0, by1)
            cx2, cy2 = min(float(w), bx2), min(float(h), by2)
            out.append(Box(
                cx1, cy1, cx2, cy2,
                score=float(0.80 + 0.18 * self._rng.random()),
                label=label,
                clipped_bottom=by2 > h - 1.5,
            ))
        return out


def load_detector(model_path: str | Path | None = None, camera=None) -> Detector:
    """Pick a backend.

    Order matters. An explicitly configured model always wins. Otherwise a
    synthetic camera gets the synthetic detector — the real model would look at a
    drawn cartoon cat and correctly see nothing, which makes the demo useless for
    showing what a trigger looks like. Only then does the bundled model apply.

    Raises rather than silently degrading: a detector that cannot detect must be
    an error the heartbeat can report, not a quiet fallback.
    """
    if model_path is not None:
        return OnnxDetector(model_path)
    if camera is not None and hasattr(camera, "ground_truth_boxes"):
        return SyntheticDetector(camera)
    bundled = bundled_model_path()
    if bundled is not None:
        return OnnxDetector(bundled)
    raise RuntimeError(
        "No detection model is installed, so nothing can be detected. "
        f"Expected {MODEL_FILENAME} alongside the app."
    )
