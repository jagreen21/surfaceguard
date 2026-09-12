#!/usr/bin/env python3
"""Per-stage timings from the architecture that matters.

    arch -x86_64 .venv-x86/bin/python tools/bench.py        # her Intel laptop
    .venv/bin/python tools/bench.py                          # this Mac

Every performance claim in docs/PERF.md comes from here. The numbers in the
cat_detector comment were measured on Apple silicon with CoreML and do not
describe the machine this ships to.

Runs against the synthetic room, so there is no camera in the loop and two runs
are comparable.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packaging"))
from _env import require_venv  # noqa: E402

require_venv("numpy", "cv2")

import numpy as np  # noqa: E402

from surfaceguard.camera.registration import Registrar  # noqa: E402
from surfaceguard.camera.sources.synthetic import SyntheticCamera  # noqa: E402
from surfaceguard.detection import roi  # noqa: E402
from surfaceguard.detection.cat_detector import (  # noqa: E402
    OnnxDetector,
    bundled_model_path,
    worker_threads,
)
from surfaceguard.detection.gates import evaluate, surfaces_for_pose  # noqa: E402
from surfaceguard.detection.motion import MotionGate  # noqa: E402
from surfaceguard.geometry.projection import Box  # noqa: E402
from surfaceguard.geometry.surface import Surface  # noqa: E402


def rss_mb() -> float:
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 1e6 if sys.platform == "darwin" else peak / 1e3
    except Exception:
        return float("nan")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def run(frames: int, model: Path | None, use_roi: bool, use_motion: bool) -> dict:
    camera = SyntheticCamera()
    camera.start()
    registrar = Registrar()
    detector = OnnxDetector(model) if model else None
    gate = MotionGate()

    # One surface, calibrated, so the ROI crop and the scale gate are exercised.
    first = camera.read(timeout=5.0)
    assert first is not None
    h, w = first.image.shape[:2]
    poly = np.array([[w * .40, h * .55], [w * .62, h * .55],
                     [w * .66, h * .72], [w * .36, h * .72]], float)
    surface = Surface("Counter", poly)

    stages: dict[str, list[float]] = {k: [] for k in
                                      ("read", "register", "detect", "gates")}
    skipped = 0
    magnifications: list[float] = []
    started = time.perf_counter()

    for i in range(frames):
        t = time.perf_counter(); frame = camera.read(timeout=2.0)
        stages["read"].append((time.perf_counter() - t) * 1e3)
        if frame is None:
            continue

        t = time.perf_counter(); reg = registrar.register(frame.image)
        stages["register"].append((time.perf_counter() - t) * 1e3)

        pose = reg.pose
        if pose is None:
            registrar.add_keyframe(f"kf{i}", frame.image, np.eye(3))
            continue
        if i == 1:
            for _ in range(10):
                surface.observe_height(pose, Box(w * .45, h * .45, w * .50, h * .55))

        visible = surfaces_for_pose([surface], pose)
        region = roi.for_surfaces(visible, pose) if use_roi else roi.full_frame((w, h))
        magnifications.append(region.magnification)

        run_detector = True
        if use_motion and reg.work is not None:
            run_detector = not gate.consider(reg.work, region, now=time.monotonic()).skipped
        if not run_detector:
            skipped += 1
            continue

        if detector is not None:
            t = time.perf_counter()
            detector.detect(region.crop(frame.image), (region.x1, region.y1))
            stages["detect"].append((time.perf_counter() - t) * 1e3)

        t = time.perf_counter()
        for s in visible:
            evaluate(s, pose, Box(w * .45, h * .45, w * .50, h * .55), [], 600)
        stages["gates"].append((time.perf_counter() - t) * 1e3)

    elapsed = time.perf_counter() - started
    camera.stop()

    return {
        "frames": frames,
        "elapsed_s": round(elapsed, 2),
        "fps": round(frames / elapsed, 1) if elapsed else 0.0,
        "skipped": skipped,
        "skip_rate": round(skipped / frames, 3) if frames else 0.0,
        "magnification": round(statistics.median(magnifications), 2) if magnifications else 1.0,
        "rss_mb": round(rss_mb(), 0),
        "stages": {
            name: {"p50": round(percentile(v, .50), 1), "p95": round(percentile(v, .95), 1),
                   "n": len(v)}
            for name, v in stages.items()
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--model", default="",
                    help="which model to measure, e.g. yolov8n.onnx — the Intel "
                         "build ships n, but bundled_model_path prefers m")
    args = ap.parse_args()

    model = (Path(__file__).resolve().parent.parent / "models" / args.model
             if args.model else bundled_model_path())
    header = {
        "machine": platform.machine(),
        "python": platform.python_version(),
        "macos": platform.mac_ver()[0],
        "cpus": __import__("os").cpu_count(),
        "worker_threads": worker_threads(),
        "model": model.name if model else "none",
    }
    variants = {
        "full frame, every frame": dict(use_roi=False, use_motion=False),
        "ROI crop, every frame": dict(use_roi=True, use_motion=False),
        "ROI crop + motion gate": dict(use_roi=True, use_motion=True),
    }
    results = {name: run(args.frames, model, **kw) for name, kw in variants.items()}

    if args.json:
        print(json.dumps({"header": header, "results": results}, indent=2))
        return 0

    print(f"\n{header['machine']}  ·  {header['cpus']} cores  ·  "
          f"{header['worker_threads']} inference threads  ·  {header['model']}")
    print(f"macOS {header['macos']}, Python {header['python']}, {args.frames} frames\n")
    cols = ("read", "register", "detect", "gates")
    print(f"{'variant':<26}" + "".join(f"{c:>18}" for c in cols)
          + f"{'fps':>7}{'skipped':>9}{'magnif':>8}{'RSS MB':>9}")
    print("-" * (26 + 18 * len(cols) + 33))
    for name, r in results.items():
        cells = "".join(f"{r['stages'][c]['p50']:>8.1f}/{r['stages'][c]['p95']:<9.1f}"
                        for c in cols)
        print(f"{name:<26}{cells}{r['fps']:>7.1f}{r['skip_rate']:>9.0%}"
              f"{r['magnification']:>8.2f}{r['rss_mb']:>9.0f}")
    print("\np50/p95 in ms per stage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
