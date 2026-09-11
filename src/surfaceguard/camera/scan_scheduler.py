"""Weighted pan scheduling, and the preference for not panning at all (§5).

Coverage with one camera is a duty cycle. This module turns a set of surfaces and
their priorities into a sequence of (pan, dwell) stops — and, crucially, detects
the case where every surface fits in one frame so the camera can stay still.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..geometry.surface import Surface


@dataclass(frozen=True)
class Stop:
    pan: float
    dwell_s: float
    surfaces: tuple[str, ...]


@dataclass
class Plan:
    stops: list[Stop] = field(default_factory=list)
    unreachable: list[str] = field(default_factory=list)
    single_view: bool = False

    @property
    def needs_scanning(self) -> bool:
        return len(self.stops) > 1

    def coverage(self) -> dict[str, float]:
        """Fraction of a full cycle each surface is in view for."""
        total = sum(s.dwell_s for s in self.stops) or 1.0
        out: dict[str, float] = {sid: 0.0 for sid in self.unreachable}
        for stop in self.stops:
            for sid in stop.surfaces:
                out[sid] = out.get(sid, 0.0) + stop.dwell_s / total
        return out


def plan_scan(
    surfaces: list[Surface],
    surface_x: dict[str, float],
    frame_width: float,
    pan_to_x,
    pan_range: tuple[float, float],
    weights: dict[str, float] | None = None,
    cycle_s: float = 60.0,
    min_dwell_s: float = 6.0,
) -> Plan:
    """Group surfaces that share a view, then split the cycle by weight.

    ``surface_x`` gives each surface's centre in map pixels; ``pan_to_x`` maps a
    pan angle to map pixels so the grouping can be expressed back in degrees.
    """
    active = [s for s in surfaces if s.enabled and s.id in surface_x]
    if not active:
        return Plan(unreachable=[s.id for s in surfaces if s.enabled])

    lo, hi = pan_range
    # Map x of the reachable pan extremes, so an out-of-range surface is detectable.
    x_lo, x_hi = pan_to_x(lo), pan_to_x(hi)
    if x_lo is None or x_hi is None:
        x_lo, x_hi = -np.inf, np.inf
    reach_lo, reach_hi = min(x_lo, x_hi), max(x_lo, x_hi)

    reachable, unreachable = [], []
    half = frame_width / 2.0
    for s in active:
        x = surface_x[s.id]
        # A surface is reachable if some frame centred within the pan range can
        # contain it with a little margin.
        if reach_lo - half * 0.9 <= x <= reach_hi + half * 0.9:
            reachable.append(s)
        else:
            unreachable.append(s.id)

    if not reachable:
        return Plan(unreachable=unreachable + [s.id for s in surfaces if s.id not in surface_x])

    xs = np.array([surface_x[s.id] for s in reachable])
    if float(xs.max() - xs.min()) <= frame_width * 0.8:
        centre_x = float(xs.mean())
        return Plan(
            stops=[Stop(_x_to_pan(centre_x, pan_to_x, pan_range), cycle_s,
                        tuple(s.id for s in reachable))],
            unreachable=unreachable,
            single_view=True,
        )

    # Greedy left-to-right grouping: each stop takes everything that fits.
    order = np.argsort(xs)
    groups: list[list[Surface]] = []
    for i in order:
        s = reachable[int(i)]
        if groups:
            g_xs = [surface_x[m.id] for m in groups[-1]] + [surface_x[s.id]]
            if max(g_xs) - min(g_xs) <= frame_width * 0.8:
                groups[-1].append(s)
                continue
        groups.append([s])

    w = weights or {}
    group_weight = [sum(float(w.get(s.id, 1.0)) for s in g) for g in groups]
    total_weight = sum(group_weight) or 1.0
    stops: list[Stop] = []
    for g, gw in zip(groups, group_weight):
        g_xs = [surface_x[s.id] for s in g]
        dwell = max(min_dwell_s, cycle_s * gw / total_weight)
        stops.append(Stop(
            _x_to_pan(float(np.mean(g_xs)), pan_to_x, pan_range),
            dwell,
            tuple(s.id for s in g),
        ))
    return Plan(stops=stops, unreachable=unreachable)


def _x_to_pan(x: float, pan_to_x, pan_range: tuple[float, float]) -> float:
    """Invert ``pan_to_x`` by sampling: it is monotonic and cheap to evaluate."""
    lo, hi = pan_range
    angles = np.linspace(lo, hi, 181)
    coords = np.array([pan_to_x(float(a)) for a in angles], dtype=float)
    if np.isnan(coords).any():
        return 0.0
    return float(angles[int(np.argmin(np.abs(coords - x)))])


@dataclass
class ScanRunner:
    """Walks a plan, telling the engine when to move and where."""

    plan: Plan = field(default_factory=Plan)
    index: int = 0
    entered_at: float = field(default_factory=time.monotonic)

    def set_plan(self, plan: Plan) -> None:
        self.plan = plan
        self.index = 0
        self.entered_at = time.monotonic()

    @property
    def current(self) -> Stop | None:
        if not self.plan.stops:
            return None
        return self.plan.stops[self.index % len(self.plan.stops)]

    def due(self, now: float | None = None) -> Stop | None:
        """Return the next stop to move to, or None to stay put."""
        if not self.plan.needs_scanning:
            return None
        now = time.monotonic() if now is None else now
        stop = self.current
        if stop is None or now - self.entered_at < stop.dwell_s:
            return None
        self.index = (self.index + 1) % len(self.plan.stops)
        self.entered_at = now
        return self.current
