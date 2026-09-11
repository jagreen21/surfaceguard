"""Turn reviewed history into a training set — problem 1, the label gap.

The review's verdicts are judgements about the *app's decision*, not bounding
boxes, and the two do not map one-to-one:

    correct         the app fired and was right   -> a cat box, positive
    not_on_surface  a real cat, just not on it    -> a cat box, positive
                    (a negative for the app, a POSITIVE for the detector)
    person          it was a person               -> a person box, positive
    not_a_cat       the app fired at nothing      -> hard negative: this image,
                    with no cat in it. The most valuable label there is, because
                    it is exactly what the detector got wrong.
    missed          a cat was there, undetected   -> no box exists. Needs one
                    proposed (see propose.py) before it can be used.
    unsure          excluded entirely

Frames come from the 5-frame strips the engine already keeps, so one reviewed
event yields several near-identical examples. They are correlated, so they are
kept together in the same split — putting frame 2 in train and frame 3 in val
would leak and make the evaluation meaningless.
"""

from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# YOLO class ids for the dataset this app trains. Deliberately two classes, not
# COCO's 80: everything else is noise for this problem.
CLASSES = ("cat", "person")
CAT, PERSON = 0, 1

# Verdicts that yield a usable example without any new annotation.
POSITIVE_CAT = {"correct", "not_on_surface"}
POSITIVE_PERSON = {"person"}
HARD_NEGATIVE = {"not_a_cat"}
NEEDS_PROPOSAL = {"missed"}
EXCLUDED = {"unsure", None, ""}


@dataclass
class Example:
    """One image and its boxes, in the frame's own pixel coordinates."""

    image: Path
    boxes: list[tuple[int, float, float, float, float]] = field(default_factory=list)
    event_id: int = 0
    source_verdict: str = ""
    proposed: bool = False          # a box suggested by a model, confirmed by a person

    @property
    def is_negative(self) -> bool:
        return not self.boxes


@dataclass
class DatasetStats:
    events: int = 0
    images: int = 0
    cat_boxes: int = 0
    person_boxes: int = 0
    negatives: int = 0
    skipped: Counter = field(default_factory=Counter)
    needs_proposal: int = 0

    def summary(self) -> str:
        return (
            f"{self.images} images from {self.events} reviewed events — "
            f"{self.cat_boxes} cat boxes, {self.person_boxes} person boxes, "
            f"{self.negatives} negatives"
            + (f", {self.needs_proposal} awaiting a box" if self.needs_proposal else "")
        )

    @property
    def ready(self) -> bool:
        """Enough to be worth training on at all."""
        return self.cat_boxes >= MIN_CAT_BOXES and self.images >= MIN_IMAGES


# Below this there is not enough signal to beat the stock model, and training
# would mostly memorise noise. Chosen to be honest rather than encouraging.
MIN_CAT_BOXES = 150
MIN_IMAGES = 200


def to_yolo(box: tuple[float, float, float, float], width: int, height: int) -> tuple:
    """Pixel xyxy -> YOLO normalised cx, cy, w, h, clamped to the frame."""
    x1, y1, x2, y2 = box
    x1, x2 = max(0.0, min(x1, x2)), min(float(width), max(x1, x2))
    y1, y2 = max(0.0, min(y1, y2)), min(float(height), max(y1, y2))
    return (
        ((x1 + x2) / 2) / width,
        ((y1 + y2) / 2) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def collect(
    events: list,
    thumb_dir: Path,
    proposals: dict[int, list[tuple[int, tuple[float, float, float, float]]]] | None = None,
) -> tuple[list[Example], DatasetStats]:
    """Build examples from reviewed events. ``proposals`` fills the 'missed' gap."""
    proposals = proposals or {}
    out: list[Example] = []
    stats = DatasetStats()

    for event in events:
        verdict = event.feedback
        if verdict in EXCLUDED:
            stats.skipped[verdict or "unreviewed"] += 1
            continue

        frames = [thumb_dir / name for name in (event.strip or [])]
        if event.thumbnail:
            frames.append(thumb_dir / event.thumbnail)
        frames = [f for f in frames if f.exists()]
        if not frames:
            stats.skipped["no picture"] += 1
            continue

        boxes: list[tuple[int, float, float, float, float]] = []
        proposed = False
        if verdict in POSITIVE_CAT and event.box:
            boxes = [(CAT, *event.box)]
        elif verdict in POSITIVE_PERSON and event.box:
            boxes = [(PERSON, *event.box)]
        elif verdict in HARD_NEGATIVE:
            boxes = []                      # the whole point: an image with no cat
        elif verdict in NEEDS_PROPOSAL:
            if event.id not in proposals:
                stats.needs_proposal += 1
                stats.skipped["missed, no box yet"] += 1
                continue
            boxes = [(cls, *b) for cls, b in proposals[event.id]]
            proposed = True
        else:
            stats.skipped[f"{verdict}, no box"] += 1
            continue

        stats.events += 1
        for frame in frames:
            # The strip's other frames share the event's box. That is an
            # approximation — the cat moves between frames — so only the keyframe
            # carries boxes; the rest are used only as negatives when the event is
            # one, where movement does not matter.
            if boxes and frame.name != (event.thumbnail or ""):
                continue
            out.append(Example(frame, list(boxes), event.id, verdict or "", proposed))
            stats.images += 1
            stats.cat_boxes += sum(1 for b in boxes if b[0] == CAT)
            stats.person_boxes += sum(1 for b in boxes if b[0] == PERSON)
            stats.negatives += 1 if not boxes else 0
    return out, stats


def split(examples: list[Example], val_fraction: float = 0.2, seed: int = 11):
    """Split by *event*, never by frame, so correlated frames cannot leak."""
    by_event: dict[int, list[Example]] = {}
    for ex in examples:
        by_event.setdefault(ex.event_id, []).append(ex)
    ids = sorted(by_event)
    random.Random(seed).shuffle(ids)
    cut = max(1, int(len(ids) * val_fraction)) if len(ids) > 4 else 0
    val_ids = set(ids[:cut])
    train = [e for i in ids if i not in val_ids for e in by_event[i]]
    val = [e for i in val_ids for e in by_event[i]]
    return train, val


# General images are capped at this fraction of her own training images. Enough to
# anchor the model to cats in general, not so much that it drowns out the point of
# training at all.
MAX_MIX_RATIO = 0.4


def write(
    examples: list[Example],
    out_dir: Path,
    val_fraction: float = 0.2,
    mix: list[Example] | None = None,
    mix_ratio: float = 0.3,
    seed: int = 11,
) -> Path:
    """Write an Ultralytics-format dataset. Returns the data.yaml path.

    ``mix`` is general cat imagery folded into the *training* split only. It is
    the third defence against forgetting: the model keeps seeing cats that are not
    hers while it learns the ones that are. It is deliberately kept out of the
    validation split, because validation exists to measure how well the model does
    on *her* cats, and diluting it would hide exactly the regression that matters.
    """
    import cv2

    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    train, val = split(examples, val_fraction, seed)

    mixed_in = 0
    if mix:
        cap = int(len(train) * min(mix_ratio, MAX_MIX_RATIO))
        chosen = list(mix)
        random.Random(seed).shuffle(chosen)
        chosen = chosen[:cap]
        train = train + chosen
        mixed_in = len(chosen)

    for name, subset in (("train", train), ("val", val)):
        (out_dir / "images" / name).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / name).mkdir(parents=True, exist_ok=True)
        for i, ex in enumerate(subset):
            img = cv2.imread(str(ex.image))
            if img is None:
                continue
            h, w = img.shape[:2]
            stem = f"{ex.event_id}_{i}"
            cv2.imwrite(str(out_dir / "images" / name / f"{stem}.jpg"), img)
            lines = []
            for cls, x1, y1, x2, y2 in ex.boxes:
                cx, cy, bw, bh = to_yolo((x1, y1, x2, y2), w, h)
                if bw <= 0 or bh <= 0:
                    continue
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            # A negative is an empty label file, which is exactly how YOLO wants
            # "this image contains nothing" expressed.
            (out_dir / "labels" / name / f"{stem}.txt").write_text("\n".join(lines))

    yaml = out_dir / "data.yaml"
    yaml.write_text(
        f"path: {out_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {list(CLASSES)}\n"
    )
    (out_dir / "manifest.json").write_text(json.dumps({
        "train": len(train), "val": len(val),
        "general_images_mixed_in": mixed_in,
        "classes": list(CLASSES),
        "proposed_boxes": sum(1 for e in examples if e.proposed),
    }, indent=2))
    return yaml
