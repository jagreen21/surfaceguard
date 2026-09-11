#!/usr/bin/env python3
"""Build the general control set the shipping gate requires.

    python tools/fetch_control_set.py --out control-set --limit 200

The gate in ``training/evaluate.py`` will not pass a fine-tuned model without a set
of cats that are *not* hers. That is the only thing that catches a model which has
learned her kitchen and forgotten cats everywhere else, so the requirement is not
negotiable — but until now nothing could produce the set, which made the whole
pipeline impossible to finish. This does.

Source is COCO val2017, filtered to the cat class. Only the annotation file and
the matching images are fetched — a few hundred images, not the full 1 GB set. The
annotations are cached, so a second run costs nothing.

Runs on the build machine. Nothing here touches her Mac or her pictures.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packaging"))
from _env import require_venv  # noqa: E402

require_venv('certifi')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from surfaceguard.net import get as http_get  # noqa: E402

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANNOTATION_MEMBER = "annotations/instances_val2017.json"
COCO_CAT_ID = 17          # COCO's own category id for "cat" (not the 0-79 index)
# Class ids in *this* project's two-class dataset.
OUR_CAT = 0


def load_annotations(cache: Path, progress=None) -> dict:
    """Fetch and cache instances_val2017.json, pulling only that member."""
    cached = cache / "instances_val2017.json"
    if cached.exists():
        if progress:
            progress(f"using cached annotations at {cached}")
        return json.loads(cached.read_text())
    cache.mkdir(parents=True, exist_ok=True)
    if progress:
        progress("downloading COCO val2017 annotations (~241 MB, once)")
    blob = http_get(ANNOTATIONS_URL, timeout=900.0)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        cached.write_bytes(zf.read(ANNOTATION_MEMBER))
    return json.loads(cached.read_text())


def cat_images(annotations: dict, limit: int | None = None) -> list[tuple[dict, list]]:
    """Images containing at least one cat, with those cat boxes.

    Crowd annotations are dropped: a box around six overlapping cats teaches a
    detector the wrong shape and would unfairly penalise a candidate.
    """
    by_id = {img["id"]: img for img in annotations.get("images", [])}
    grouped: dict[int, list] = {}
    for ann in annotations.get("annotations", []):
        if ann.get("category_id") != COCO_CAT_ID or ann.get("iscrowd"):
            continue
        x, y, w, h = ann["bbox"]
        if w <= 1 or h <= 1:
            continue
        grouped.setdefault(ann["image_id"], []).append((x, y, x + w, y + h))

    out = []
    for image_id in sorted(grouped):
        image = by_id.get(image_id)
        if image is None:
            continue
        out.append((image, grouped[image_id]))
        if limit and len(out) >= limit:
            break
    return out


def write_control_set(entries: list[tuple[dict, list]], out_dir: Path, progress=None) -> int:
    """Download each image and write a YOLO label file beside it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for i, (image, boxes) in enumerate(entries, 1):
        name = image["file_name"]
        target = out_dir / name
        if not target.exists():
            url = image.get("coco_url") or (
                f"http://images.cocodataset.org/val2017/{name}"
            )
            try:
                target.write_bytes(http_get(url, timeout=60.0))
            except Exception:
                continue
        width, height = float(image["width"]), float(image["height"])
        lines = []
        for x1, y1, x2, y2 in boxes:
            cx, cy = ((x1 + x2) / 2) / width, ((y1 + y2) / 2) / height
            bw, bh = (x2 - x1) / width, (y2 - y1) / height
            lines.append(f"{OUR_CAT} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        target.with_suffix(".txt").write_text("\n".join(lines))
        written += 1
        if progress and i % 25 == 0:
            progress(f"  {i}/{len(entries)}")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("control-set"))
    ap.add_argument("--limit", type=int, default=200,
                    help="how many cat images to keep (200 is plenty to catch forgetting)")
    ap.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "surfaceguard")
    args = ap.parse_args()

    print("Building the general control set")
    annotations = load_annotations(args.cache, progress=lambda m: print(f"  {m}"))
    entries = cat_images(annotations, args.limit)
    print(f"  {len(entries)} val2017 images contain a cat")
    written = write_control_set(entries, args.out, progress=print)
    boxes = sum(len(b) for _img, b in entries)
    print(f"\n  {written} images, {boxes} cat boxes -> {args.out}")
    print(f"  use it with: python tools/finetune.py --train --control {args.out}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
