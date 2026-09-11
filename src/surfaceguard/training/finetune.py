"""The fine-tuning recipe — problems 2 and 3.

**Privacy (problem 2).** Training runs where the pictures already are. Nothing
leaves the machine: the dataset is written to a local directory, the training kit
is fetched on demand into a temporary environment, and it is deleted afterwards.
Her kitchen never lands on someone else's disk.

**Forgetting (problem 3).** Training a detector on one kitchen is the textbook way
to make it forget what a cat looks like anywhere else. Four defences, all cheap:

  * freeze the backbone and tune only the head — the features stay general
  * a low learning rate and few epochs, so weights move a little
  * mix in a slice of general cat images, if one is available
  * and the evaluate.py gate, which refuses to ship a model that regressed

The first three reduce the chance of forgetting. Only the last one *catches* it,
which is why it is not optional.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Freeze this many leading layers. For yolov8 that is the backbone; the detection
# head still learns her cats, the features stay the general ones.
FREEZE_LAYERS = 10
EPOCHS = 24
LEARNING_RATE = 0.001          # ~1/10 of a from-scratch run
PATIENCE = 6                   # stop early rather than grind into overfitting
BATCH = 8


@dataclass
class TrainingResult:
    ok: bool
    weights: Path | None = None
    onnx: Path | None = None
    message: str = ""
    log_tail: str = ""
    metrics: dict = field(default_factory=dict)


def kit_available(python: Path) -> bool:
    return subprocess.run([str(python), "-c", "import ultralytics"],
                          capture_output=True).returncode == 0


def build_kit(work: Path, progress=None) -> Path:
    """Create the throwaway training environment. ~1.1 GB, deleted afterwards."""
    venv = work / "kit"
    if progress:
        progress("Preparing the training kit (about a gigabyte, downloaded once)…")
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    pip = venv / "bin" / "pip"
    subprocess.run([str(pip), "install", "-q", "--upgrade", "pip"], check=True)
    subprocess.run([str(pip), "install", "-q", "ultralytics", "onnx"], check=True)
    return venv / "bin" / "python"


def finetune(
    data_yaml: Path,
    base_weights: str = "yolov8m.pt",
    out_dir: Path | None = None,
    epochs: int = EPOCHS,
    freeze: int = FREEZE_LAYERS,
    imgsz: int = 640,
    keep_kit: bool = False,
    progress=None,
) -> TrainingResult:
    """Fine-tune, then export to ONNX. Returns where the new model landed."""
    work = Path(tempfile.mkdtemp(prefix="sg-train-"))
    out_dir = Path(out_dir or work / "out")
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        python = build_kit(work, progress)
        script = f'''
import json, torch
from ultralytics import YOLO
# MPS where available: an M-series Mac trains this in minutes, and it means the
# pictures never have to leave the machine.
device = "mps" if torch.backends.mps.is_available() else "cpu"
m = YOLO({base_weights!r})
m.train(
    data={str(data_yaml)!r},
    epochs={epochs}, imgsz={imgsz}, batch={BATCH},
    freeze={freeze}, lr0={LEARNING_RATE}, patience={PATIENCE},
    device=device, project={str(out_dir)!r}, name="finetune",
    exist_ok=True, verbose=False, plots=False, val=True,
)
metrics = {{k: float(v) for k, v in (m.trainer.metrics or {{}}).items()
           if isinstance(v, (int, float))}}
onnx = m.export(format="onnx", imgsz={imgsz}, opset=12, dynamic=False)
print("RESULT " + json.dumps({{"onnx": str(onnx), "metrics": metrics,
                              "best": str(m.trainer.best), "device": device}}))
'''
        if progress:
            progress("Training on this Mac. Nothing is uploaded.")
        run = subprocess.run([str(python), "-c", script], cwd=work,
                             capture_output=True, text=True)
        tail = (run.stdout + run.stderr)[-2000:]
        if run.returncode != 0:
            return TrainingResult(False, message="Training failed.", log_tail=tail)

        payload = next((line[len("RESULT "):] for line in run.stdout.splitlines()[::-1]
                        if line.startswith("RESULT ")), None)
        if payload is None:
            return TrainingResult(False, message="Training produced no model.", log_tail=tail)
        info = json.loads(payload)
        onnx = Path(info["onnx"])
        if not onnx.exists():
            return TrainingResult(False, message="The exported model is missing.", log_tail=tail)
        final = out_dir / "candidate.onnx"
        shutil.copy2(onnx, final)
        return TrainingResult(
            True, weights=Path(info.get("best", "")), onnx=final,
            message=f"Trained on {info.get('device', '?')}.",
            log_tail=tail, metrics=info.get("metrics", {}),
        )
    finally:
        if not keep_kit:
            shutil.rmtree(work / "kit", ignore_errors=True)
