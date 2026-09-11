#!/usr/bin/env python3
"""Export the detection model, and fetch the image its tests compare against.

Run on the build machine only. Ultralytics pulls in PyTorch, which must never end
up in the app's own virtualenv — PyInstaller would bundle it and add a gigabyte to
a download that is already 130 MB. So this installs into a throwaway environment,
exports, copies the result into models/, and leaves the app venv untouched.

    python packaging/export_model.py

Licence note: YOLOv8 is AGPL-3.0. That is fine for personal use. If this is ever
distributed more widely, swap in an Apache-licensed model (YOLOX, RT-DETR) — the
decoder in cat_detector.py reads the standard YOLO output layout either way.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
ASSETS = ROOT / "tests" / "assets"
WEIGHTS = "yolov8m.pt"
OUTPUT = "yolov8m.onnx"
# Ultralytics' standard sample; the detector test compares against the exporter's
# own inference on it, which is the only way to know the decoder is right.
REFERENCE = "bus.jpg"
REFERENCE_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg"
)


def run(cmd: list[str], **kw) -> None:
    subprocess.run(cmd, check=True, **kw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep-venv", action="store_true", help="do not delete the export venv")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--weights", default=WEIGHTS,
                    help="yolov8n/s/m .pt; m fits the 40 ms budget on Apple silicon")
    args = ap.parse_args()

    MODELS.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="sg-export-"))
    venv = work / "venv"

    print(f"Building an isolated export environment in {venv}")
    run([sys.executable, "-m", "venv", str(venv)])
    pip, python = venv / "bin" / "pip", venv / "bin" / "python"
    run([str(pip), "install", "-q", "--upgrade", "pip"])
    print("Installing ultralytics (this pulls PyTorch and takes a minute)…")
    run([str(pip), "install", "-q", "ultralytics", "onnx"])

    script = (
        "from ultralytics import YOLO\n"
        f"m = YOLO('{args.weights}')\n"
        "names = m.names\n"
        "assert 'cat' in names.values() and 'person' in names.values()\n"
        f"p = m.export(format='onnx', imgsz={args.imgsz}, opset=12, dynamic=False)\n"
        "print(p)\n"
    )
    print("Exporting to ONNX…")
    out = subprocess.run([str(python), "-c", script], cwd=work,
                         capture_output=True, text=True, check=True)
    exported = next((Path(line.strip()) for line in out.stdout.splitlines()[::-1]
                     if line.strip().endswith(".onnx")), None)
    if exported is None or not exported.exists():
        exported = work / OUTPUT
    if not exported.exists():
        raise SystemExit(f"export produced no file\n{out.stdout[-800:]}")
    shutil.copy2(exported, MODELS / exported.name)
    print(f"  model -> {MODELS / exported.name} ({(MODELS / exported.name).stat().st_size / 1e6:.1f} MB)")

    if not (ASSETS / REFERENCE).exists():
        print("Fetching the reference image for the decoder test…")
        sys.path.insert(0, str(ROOT / "src"))
        from surfaceguard.net import get as http_get

        (ASSETS / REFERENCE).write_bytes(http_get(REFERENCE_URL))
        print(f"  image -> {ASSETS / REFERENCE}")

    if not args.keep_venv:
        shutil.rmtree(work, ignore_errors=True)
        print("Removed the export environment; the app venv was never touched.")

    print("\nNow rebuild with the model included:")
    print("  python packaging/build_app.py --version <v> --key <hex> --with-onnx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
