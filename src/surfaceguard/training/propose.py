"""Propose boxes for the cats the detector missed — the other half of problem 1.

A ``missed`` verdict says a cat was there and nothing was detected, so there is no
box to learn from. Rather than ask her to drag rectangles, a heavier model runs
over those frames at a low threshold and *suggests* a box.

A suggestion is never used as a label on its own. It is shown, confirmed, and only
then written. An unconfirmed proposal training the model to find what it already
thinks it sees is a feedback loop, not supervision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..detection.cat_detector import COCO_CAT, COCO_PERSON, OnnxDetector
from .dataset import CAT, PERSON

# Low enough to surface what the shipping model rejected, high enough not to
# propose noise. The shipping threshold is 0.35.
PROPOSAL_CONF = 0.10

# A proposal weaker than this is not worth showing: she would be confirming a
# guess, and a wrong confirmed box is worse than a missing one.
SHOW_ABOVE = 0.15


@dataclass
class Proposal:
    event_id: int
    image: Path
    cls: int
    box: tuple[float, float, float, float]
    score: float
    confirmed: bool | None = None       # None until a person has said

    @property
    def usable(self) -> bool:
        return self.confirmed is True


@dataclass
class ProposalSet:
    proposals: list[Proposal] = field(default_factory=list)
    no_candidate: list[int] = field(default_factory=list)

    def confirmed_map(self) -> dict[int, list[tuple[int, tuple]]]:
        """The shape :func:`dataset.collect` wants for its ``proposals`` argument."""
        out: dict[int, list[tuple[int, tuple]]] = {}
        for p in self.proposals:
            if p.usable:
                out.setdefault(p.event_id, []).append((p.cls, p.box))
        return out

    def summary(self) -> str:
        confirmed = sum(1 for p in self.proposals if p.usable)
        pending = sum(1 for p in self.proposals if p.confirmed is None)
        return (f"{len(self.proposals)} proposals ({confirmed} confirmed, {pending} "
                f"to check), {len(self.no_candidate)} frames with nothing to suggest")


def propose(
    events: list,
    thumb_dir: Path,
    model_path: Path,
    conf: float = PROPOSAL_CONF,
) -> ProposalSet:
    """Suggest a box for every ``missed`` event that has a picture."""
    detector = OnnxDetector(model_path, conf=conf)
    result = ProposalSet()

    for event in events:
        if event.feedback != "missed":
            continue
        name = event.thumbnail or (event.strip[0] if event.strip else None)
        if not name:
            continue
        image = thumb_dir / name
        if not image.exists():
            continue

        import cv2

        frame = cv2.imread(str(image))
        if frame is None:
            continue
        boxes = [b for b in detector.detect(frame) if b.score >= SHOW_ABOVE]
        if not boxes:
            result.no_candidate.append(event.id)
            continue
        # The best cat candidate; a person proposal here would be a different
        # mistake and is left for her to reject.
        best = max(boxes, key=lambda b: (b.label == "cat", b.score))
        result.proposals.append(Proposal(
            event_id=event.id,
            image=image,
            cls=CAT if best.label == "cat" else PERSON,
            box=(best.x1, best.y1, best.x2, best.y2),
            score=best.score,
        ))
    return result
