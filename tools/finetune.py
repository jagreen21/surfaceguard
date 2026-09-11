#!/usr/bin/env python3
"""Fine-tune the detector on this machine's own reviewed history.

    python tools/finetune.py --status                 # is there enough data yet?
    python tools/finetune.py --review                 # confirm boxes for missed cats
    python tools/finetune.py --train --control DIR    # build, train, and gate

Nothing leaves this machine. The pictures stay where they are, training runs
locally, and the resulting model is only kept if it beats the one in use on held
out data *and* does not regress on a general control set.

The control set is a directory of images of other people's cats with YOLO label
files beside them — COCO val2017 filtered to the cat class is the easy source. It
is required: without it, a model that has forgotten every cat but hers would pass.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packaging"))
from _env import require_venv  # noqa: E402

require_venv('numpy', 'cv2')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from surfaceguard.detection.cat_detector import bundled_model_path  # noqa: E402
from surfaceguard.storage.activity_log import ActivityLog  # noqa: E402
from surfaceguard.training import dataset, evaluate, finetune, propose  # noqa: E402


def load_control(directory: Path) -> list:
    """Images with YOLO .txt labels beside them."""
    out = []
    for image in sorted(directory.glob("*.jpg")) + sorted(directory.glob("*.png")):
        label = image.with_suffix(".txt")
        boxes = []
        if label.exists():
            import cv2

            frame = cv2.imread(str(image))
            if frame is None:
                continue
            h, w = frame.shape[:2]
            for line in label.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
                boxes.append((cls, (cx - bw / 2) * w, (cy - bh / 2) * h,
                              (cx + bw / 2) * w, (cy + bh / 2) * h))
        out.append(dataset.Example(image=image, boxes=boxes))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--propose", action="store_true", help="list proposals without a UI")
    ap.add_argument("--review", action="store_true",
                    help="confirm proposed boxes in a window, the only way 'missed' "
                         "events become training data")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--control", type=Path, help="directory of general cat images")
    ap.add_argument("--out", type=Path, default=Path("training-out"))
    ap.add_argument("--epochs", type=int, default=finetune.EPOCHS)
    ap.add_argument("--yes", action="store_true",
                    help="accept every proposal unseen — testing only; a confirmed "
                         "box nobody looked at is not supervision")
    args = ap.parse_args()

    log = ActivityLog()
    events = log.recent(limit=100_000)
    proposals_map = {}

    if args.propose or args.review or args.train:
        model = bundled_model_path()
        if model is None:
            raise SystemExit("No detection model installed.")
        found = propose.propose(events, log.thumb_dir, model)
        print(f"  {found.summary()}")
        if args.yes:
            for p in found.proposals:
                p.confirmed = True
        elif args.review or args.train:
            if found.proposals:
                from surfaceguard.ui.proposals import review_proposals

                confirmed = review_proposals(found.proposals)
                print(f"  {confirmed} of {len(found.proposals)} proposals confirmed")
            else:
                print("  nothing to confirm")
        proposals_map = found.confirmed_map()
        if (args.propose or args.review) and not args.train:
            for p in found.proposals[:20]:
                print(f"    event {p.event_id}: {p.cls} {tuple(round(v) for v in p.box)} "
                      f"score {p.score:.2f} -> {p.image.name}")
            return 0

    examples, stats = dataset.collect(events, log.thumb_dir, proposals_map)
    print(f"  {stats.summary()}")
    for reason, n in stats.skipped.most_common():
        print(f"    skipped {n}: {reason}")

    if args.status or not args.train:
        print(f"\n  ready to train: {stats.ready} "
              f"(need {dataset.MIN_CAT_BOXES} cat boxes and {dataset.MIN_IMAGES} images)")
        return 0

    if not stats.ready:
        print("\n  Not enough reviewed history yet. Keep using the app.")
        return 1
    if not args.control:
        print("\n  --control is required: see the docstring.")
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    train_set, holdout = dataset.split(examples)
    data_yaml = dataset.write(examples, args.out / "dataset")
    print(f"  dataset -> {data_yaml}  ({len(train_set)} train, {len(holdout)} held out)")

    result = finetune.finetune(data_yaml, out_dir=args.out, epochs=args.epochs,
                               progress=lambda m: print(f"  {m}"))
    if not result.ok:
        print(f"  {result.message}\n{result.log_tail[-600:]}")
        return 3
    print(f"  {result.message}  candidate -> {result.onnx}")

    verdict = evaluate.gate(bundled_model_path(), result.onnx, holdout,
                            load_control(args.control))
    print()
    print(verdict.report())
    if not verdict.passed:
        return 4
    keep = args.out / "approved.onnx"
    keep.write_bytes(result.onnx.read_bytes())
    print(f"\n  approved -> {keep}")
    print("  ship it with: python packaging/make_release.py --model", keep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
