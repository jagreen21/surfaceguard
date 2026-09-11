"""The per-frame gates of decision D3.

Each gate is independent and individually reported, so a false positive is
attributable to one named condition instead of "the model". A gate can also
report itself *unavailable* — it could not run — which is different from passing,
and the heartbeat treats it accordingly (E7).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np

from ..geometry.projection import Box, Pose, point_in_polygon, transform_points
from ..geometry.surface import Surface

# A surface less visible than this is not judged at all: we cannot reason about
# the part of it that is off-frame.
MIN_VISIBLE_FRACTION = 0.85

# Reject a detection whose on-screen height is off the plane's prediction by more
# than this factor in either direction.
SCALE_TOLERANCE = 1.5

# A person box within this many surface-widths of the surface suppresses it.
PERSON_PROXIMITY = 0.6


class Status(enum.Enum):
    PASS = "pass"
    FAIL = "fail"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class GateResult:
    name: str
    status: Status
    detail: str = ""
    required: bool = True

    @property
    def blocks(self) -> bool:
        """Does this result stop a trigger?

        Only *required* gates veto. An advisory gate that fails or cannot run is
        reported as a warning instead — which is what lets the scale gate be
        collected-into rather than depended on before it has been calibrated.
        """
        return self.required and self.status is not Status.PASS

    @property
    def warns(self) -> bool:
        return not self.required and self.status is not Status.PASS


@dataclass
class Verdict:
    """The outcome of evaluating one detection against one surface."""

    surface_id: str
    surface_name: str
    gates: list[GateResult] = field(default_factory=list)
    box: Box | None = None

    @property
    def on_surface(self) -> bool:
        return bool(self.gates) and not any(g.blocks for g in self.gates)

    @property
    def blocked_by(self) -> list[str]:
        return [g.name for g in self.gates if g.blocks]

    @property
    def degraded(self) -> list[str]:
        """Advisory gates that did not pass — reduced confidence, not a veto."""
        return [g.name for g in self.gates if g.warns]

    def summary(self) -> str:
        if self.on_surface:
            note = f" (degraded: {', '.join(self.degraded)})" if self.degraded else ""
            return f"on {self.surface_name}{note}"
        return f"not on {self.surface_name} — blocked by {', '.join(self.blocked_by)}"


def evaluate(
    surface: Surface,
    pose: Pose,
    cat: Box,
    people: list[Box] | None = None,
    minute_of_day: int | None = None,
) -> Verdict:
    """Run every stateless gate for one cat box against one surface.

    ``minute_of_day`` is only needed by the adjustments the weekly review makes —
    a blind spot can be limited to after dark. Leave it None and those gates
    behave as if the spot applied all day, which is the safe direction for a
    caller that does not know the time.
    """
    verdict = Verdict(surface.id, surface.name, box=cat)
    add = verdict.gates.append
    poly = surface.project(pose)

    # --- visibility -------------------------------------------------------
    visible = surface.visible_fraction(pose)
    add(GateResult(
        "visible",
        Status.PASS if visible >= MIN_VISIBLE_FRACTION else Status.FAIL,
        f"{visible * 100:.0f}% of the surface in frame",
    ))

    # --- paw point inside the projected polygon ---------------------------
    paw = cat.paw_point
    inside = point_in_polygon(poly, paw)
    add(GateResult(
        "inside",
        Status.PASS if inside else Status.FAIL,
        f"paw at ({paw[0]:.0f}, {paw[1]:.0f})",
    ))

    # --- apparent size consistent with the surface plane ------------------
    expected = surface.expected_height(pose, paw) if inside else None
    if expected is None or expected <= 1.0:
        add(GateResult(
            "scale",
            Status.UNAVAILABLE,
            "no plane model for this point yet",
            required=False,
        ))
    else:
        ratio = cat.height / expected
        tol = surface.scale_tolerance(SCALE_TOLERANCE)
        ok = (1.0 / tol) <= ratio <= tol
        add(GateResult(
            "scale",
            Status.PASS if ok else Status.FAIL,
            f"{cat.height:.0f} px observed vs {expected:.0f} px expected ({ratio:.2f}x)",
            required=surface.scale_gate_is_required,
        ))

    # --- no person near the surface ---------------------------------------
    near = _person_near(poly, people or [])
    add(GateResult(
        "no_person",
        Status.FAIL if near else Status.PASS,
        "someone is at the surface" if near else "no person nearby",
    ))

    # --- adjustments the weekly review made -------------------------------
    # These gates exist only once the user has tuned this surface, so the gate
    # list stays a record of what the app decided, not a list of inactive knobs.
    tuning = surface.tuning
    if tuning.min_score is not None:
        add(GateResult(
            "confidence",
            Status.PASS if cat.score >= tuning.min_score else Status.FAIL,
            f"{cat.score:.0%} sure, and you asked for {tuning.min_score:.0%} here",
        ))

    if tuning.blind_spots and inside:
        spot = tuning.blind_spot_at(surface.paw_in_map(pose, paw), minute_of_day)
        add(GateResult(
            "known_false_alarm",
            Status.FAIL if spot is not None else Status.PASS,
            f"this is {spot.describe()}" if spot is not None
            else "not one of the spots you marked",
        ))

    # --- occlusion warning ------------------------------------------------
    # A box whose bottom is flush with the frame edge is clipped, so its paw point
    # is not a paw point. Advisory: it lowers confidence rather than vetoing.
    if cat.clipped_bottom or cat.y2 >= pose.frame_size[1] - 2:
        add(GateResult(
            "unclipped",
            Status.UNAVAILABLE,
            "box is clipped at the frame edge; paw point unreliable",
            required=False,
        ))

    return verdict


def _person_near(poly_frame: np.ndarray, people: list[Box]) -> bool:
    if not people:
        return False
    poly = np.asarray(poly_frame, dtype=np.float64).reshape(-1, 2)
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    pad = PERSON_PROXIMITY * float(np.linalg.norm(hi - lo))
    for person in people:
        # Inflated bounding-box overlap: cheap, and generous in the right
        # direction — we would rather stay silent than blast someone cooking.
        if (person.x2 >= lo[0] - pad and person.x1 <= hi[0] + pad
                and person.y2 >= lo[1] - pad and person.y1 <= hi[1] + pad):
            return True
    return False


def surfaces_for_pose(surfaces: list[Surface], pose: Pose) -> list[Surface]:
    """Surfaces whose projected polygon overlaps this frame at all."""
    w, h = pose.frame_size
    out = []
    for s in surfaces:
        if not s.enabled:
            continue
        p = transform_points(pose.map_to_frame, s.polygon)
        if p[:, 0].max() >= 0 and p[:, 0].min() <= w and p[:, 1].max() >= 0 and p[:, 1].min() <= h:
            out.append(s)
    return out
