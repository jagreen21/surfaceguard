"""The gate that stops a worse model shipping — problem 4.

An app that updates itself and can install a worse detector is a regression
machine. A candidate must clear two bars:

  1. beat the incumbent on *her* held-out data, and
  2. not regress on a general control set of cats it was not trained on.

The second matters because fine-tuning on one kitchen is exactly how a model
forgets what a cat looks like anywhere else. The control set is required, not
optional: a gate that silently skips half its check is not a gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..detection.cat_detector import OnnxDetector

# The candidate must win by more than noise on her data...
MIN_IMPROVEMENT = 0.02
# ...and may lose no more than this on the general control set.
MAX_REGRESSION = 0.03
IOU_MATCH = 0.5


@dataclass
class Score:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def line(self, label: str) -> str:
        return (f"{label:12s} precision {self.precision:.3f}  recall {self.recall:.3f}  "
                f"F1 {self.f1:.3f}   (tp {self.tp} fp {self.fp} fn {self.fn})")


@dataclass
class Verdict:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    incumbent_home: Score = field(default_factory=Score)
    candidate_home: Score = field(default_factory=Score)
    incumbent_control: Score = field(default_factory=Score)
    candidate_control: Score = field(default_factory=Score)

    def report(self) -> str:
        lines = [
            "her data (held out):",
            "  " + self.incumbent_home.line("incumbent"),
            "  " + self.candidate_home.line("candidate"),
            "general control:",
            "  " + self.incumbent_control.line("incumbent"),
            "  " + self.candidate_control.line("candidate"),
            "",
            ("PASS — safe to ship" if self.passed else "BLOCKED — will not ship"),
        ]
        lines += [f"  · {r}" for r in self.reasons]
        return "\n".join(lines)


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def score_model(detector: OnnxDetector, examples: list) -> Score:
    """Greedy IoU matching of predicted cat boxes against labelled ones."""
    import cv2

    out = Score()
    for ex in examples:
        frame = cv2.imread(str(ex.image))
        if frame is None:
            continue
        predicted = [(b.x1, b.y1, b.x2, b.y2) for b in detector.detect(frame)
                     if b.label == "cat"]
        truth = [(x1, y1, x2, y2) for cls, x1, y1, x2, y2 in ex.boxes if cls == 0]
        unmatched = list(predicted)
        for t in truth:
            best = max(unmatched, key=lambda p: iou(p, t), default=None)
            if best is not None and iou(best, t) >= IOU_MATCH:
                out.tp += 1
                unmatched.remove(best)
            else:
                out.fn += 1
        out.fp += len(unmatched)
    return out


def gate(
    incumbent_path: Path,
    candidate_path: Path,
    home_holdout: list,
    control_set: list | None,
) -> Verdict:
    """Decide whether ``candidate`` may replace ``incumbent``."""
    reasons: list[str] = []
    if not home_holdout:
        return Verdict(False, ["no held-out data of her own to judge against"])
    if not control_set:
        return Verdict(False, [
            "no general control set supplied, so a model that has forgotten what "
            "cats look like elsewhere would pass unnoticed. See README."
        ])

    incumbent = OnnxDetector(incumbent_path)
    candidate = OnnxDetector(candidate_path)
    ih, ch = score_model(incumbent, home_holdout), score_model(candidate, home_holdout)
    ic, cc = score_model(incumbent, control_set), score_model(candidate, control_set)

    gained = ch.f1 - ih.f1
    lost = ic.f1 - cc.f1
    if gained < MIN_IMPROVEMENT:
        reasons.append(
            f"only {gained:+.3f} F1 on her data; needs at least +{MIN_IMPROVEMENT:.2f} "
            "to be worth the risk of shipping"
        )
    if lost > MAX_REGRESSION:
        reasons.append(
            f"lost {lost:.3f} F1 on the control set; it has started forgetting cats "
            f"that are not hers (limit {MAX_REGRESSION:.2f})"
        )
    passed = not reasons
    if passed:
        reasons.append(f"{gained:+.3f} F1 on her data, {lost:+.3f} on the control set")
    return Verdict(passed, reasons, ih, ch, ic, cc)
