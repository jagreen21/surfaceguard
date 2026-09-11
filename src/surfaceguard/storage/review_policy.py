"""What the weekly review asks about, and what it is allowed to change.

Three ideas hold this together.

**Sample where a label changes a decision.** A random week of events is mostly
confident, correct and worthless to ask about. The deck is built from the events
that sat near a boundary: the ones a gate only just blocked, the ones that fired
with an advisory gate warning, the ones that happened somewhere the app had never
seen anything before. Twelve of those are worth more than two hundred taken at
random, and twelve is a minute of someone's evening.

**One question per thing, not per event.** Fourteen near-identical 3 a.m. events
on the same shelf are one question. They are clustered by where they happened in
*map* coordinates, so "the same spot again" survives the camera panning away and
coming back — the same reason surfaces live in map space (D1).

**Every change is a record that can be taken back.** The review never mutates a
surface in place; it produces :class:`Adjustment` values that know both their new
value and the one they replaced. A tuning system that cannot say exactly what it
did, and undo it, is the silent failure this app exists to avoid — wearing a
different hat.

Nothing here imports Qt. The deck, the clustering and the adjustments are all
decidable from an activity log and a list of surfaces, which is what makes them
testable against a replayed session.
"""

from __future__ import annotations

import enum
import math
import time
from dataclasses import dataclass, field

from ..geometry.surface import MIN_CALIBRATION_SAMPLES, BlindSpot, Surface, is_night
from .activity_log import Event

# --- how often, and how big ---------------------------------------------------

REVIEW_INTERVAL_S = 7 * 86_400.0
# Below this many worthwhile cards the review is not worth anyone's minute, and
# offering it anyway spends the credibility needed for the week that matters.
MIN_DECK = 6
# The hard cap is the total: it is what the "~1 min" on the button promises, and
# the split below is only the preferred shape of those cards. A week that is all
# false alarms must still be reviewable — that user needs this more than anyone —
# so alerts take any capacity the grid does not use.
MAX_CARDS = 12
MAX_ALERTS = 5
MAX_GRID = 9
# Two ignored invitations mean no. Asking a third time teaches people to dismiss
# the card without reading it, which costs more than the labels are worth.
MAX_DECLINES = 2

# Cluster radius in map units. Roughly "the same corner of the counter".
CLUSTER_CELL = 24.0

# A blind spot is only proposed once a place has misfired this many times.
# One false alarm is an anecdote; the app should not go blind on an anecdote.
MIN_CLUSTER_FOR_BLIND_SPOT = 2
BLIND_SPOT_MARGIN = 12.0
# Confirmations needed on a surface before the size check is tightened.
MIN_CONFIRMATIONS_TO_TIGHTEN = 4
TIGHTEN_FACTOR = 0.9
RELAX_FACTOR = 1.25
MIN_TOLERANCE = 1.15
MAX_TOLERANCE = 2.5


class Kind(enum.Enum):
    """Which act of the review a card belongs to."""

    ALERT = "alert"          # it played a sound; asked one at a time, with context
    CANDIDATE = "candidate"  # it stayed silent; asked as a grid


# Why a silent event was worth asking about, and how much a label on it is worth.
# Ordered by how much the answer can actually change: a size call the app can
# retune beats a visibility problem it can only report.
_SILENT_REASONS: dict[str, tuple[str, float]] = {
    "scale": ("I thought it was the wrong size to be a cat there", 0.9),
    "known_false_alarm": ("I ignored this because you marked this spot", 0.85),
    "confidence": ("I was not sure enough to act", 0.8),
    "no_person": ("someone was standing near the surface", 0.4),
    "visible": ("I could not see enough of the surface", 0.2),
}
_DEGRADED_BONUS = 0.2
_NOVELTY_BONUS = 0.3


@dataclass
class Candidate:
    """One card. May stand for several events that happened in the same place."""

    event: Event
    kind: Kind
    reason: str
    weight: float
    duplicates: list[Event] = field(default_factory=list)

    @property
    def count(self) -> int:
        return 1 + len(self.duplicates)

    @property
    def ids(self) -> list[int]:
        return [self.event.id, *(e.id for e in self.duplicates)]

    @property
    def repeat_note(self) -> str:
        return "" if self.count == 1 else f"This happened {self.count} times."

    @property
    def has_picture(self) -> bool:
        return bool(self.event.thumbnail or self.event.strip)


@dataclass
class Deck:
    """A week's worth of questions, already ordered and capped."""

    alerts: list[Candidate] = field(default_factory=list)
    grid: list[Candidate] = field(default_factory=list)
    window_start: float = 0.0
    window_end: float = 0.0
    total_events: int = 0
    total_fired: int = 0
    skipped_no_picture: int = 0

    @property
    def cards(self) -> list[Candidate]:
        return [*self.alerts, *self.grid]

    @property
    def size(self) -> int:
        return len(self.alerts) + len(self.grid)

    @property
    def is_empty(self) -> bool:
        return self.size == 0

    @property
    def events_represented(self) -> int:
        return sum(c.count for c in self.cards)

    def surface_names(self) -> list[str]:
        seen: list[str] = []
        for c in self.cards:
            if c.event.surface_name not in seen:
                seen.append(c.event.surface_name)
        return seen


# ------------------------------------------------------------------- sampling


def build_deck(
    events: list[Event],
    window_start: float = 0.0,
    window_end: float | None = None,
) -> Deck:
    """Choose what to ask about from one window of history."""
    window_end = time.time() if window_end is None else window_end
    deck = Deck(window_start=window_start, window_end=window_end)
    deck.total_events = len(events)
    deck.total_fired = sum(1 for e in events if e.fired)

    seen_buckets: set[tuple[str, bool]] = set()
    alerts: list[Candidate] = []
    silent: list[Candidate] = []

    for event in events:
        bucket = (event.surface_id, is_night(event.minute_of_day))
        novel = bucket not in seen_buckets
        seen_buckets.add(bucket)
        if event.feedback:
            continue  # already answered, in the Activity tab or a previous review
        picked = _score(event, novel)
        if picked is None:
            continue
        reason, weight = picked
        candidate = Candidate(event, Kind.ALERT if event.fired else Kind.CANDIDATE,
                              reason, weight)
        if not candidate.has_picture:
            # Nothing to show means nothing to ask. Counted so the invitation can
            # say why a quiet week produced no deck.
            deck.skipped_no_picture += 1
            continue
        (alerts if event.fired else silent).append(candidate)

    ranked_alerts = _rank(cluster(alerts))
    ranked_grid = _rank(cluster(silent))
    deck.alerts = ranked_alerts[:MAX_ALERTS]
    deck.grid = ranked_grid[:min(MAX_GRID, MAX_CARDS - len(deck.alerts))]
    spare = MAX_CARDS - deck.size
    if spare > 0:
        # Sounds that actually played get first claim on the spare room: they are
        # the only events the user experienced, and the only ones with a cost.
        deck.alerts.extend(ranked_alerts[len(deck.alerts):len(deck.alerts) + spare])
        spare = MAX_CARDS - deck.size
    if spare > 0:
        deck.grid.extend(ranked_grid[len(deck.grid):len(deck.grid) + spare])
    return deck


def _score(event: Event, novel: bool) -> tuple[str, float] | None:
    """Why this event is worth a question, and how much the answer is worth."""
    if event.fired:
        # Every sound played is worth asking about: it is the only event type the
        # user actually experienced, and the only one with a cost when wrong.
        weight = 1.0 + (_DEGRADED_BONUS if event.degraded else 0.0)
        reason = ("I played the sound, but some of my checks were unsure"
                  if event.degraded else "I played the sound")
        return reason, weight + (_NOVELTY_BONUS if novel else 0.0)

    blocked = event.blocked_by
    if blocked:
        # The gate listed first in _SILENT_REASONS is the most informative one to
        # ask about, not necessarily the first that happened to fail.
        for name, (reason, weight) in _SILENT_REASONS.items():
            if name in blocked:
                break
        else:
            return None
    elif event.degraded:
        reason, weight = "I saw something but my checks were unsure", 0.5
    else:
        # Nothing blocked it: the policy held it back for dwell, cooldown or the
        # surface's schedule. Worth a look, but it is rarely the interesting case.
        reason, weight = f"I stayed quiet — {event.reason}", 0.3

    if event.degraded and blocked:
        weight += _DEGRADED_BONUS
    if novel:
        weight += _NOVELTY_BONUS
    return reason, weight


def cluster(candidates: list[Candidate]) -> list[Candidate]:
    """Fold candidates that happened in the same place into one card each."""
    heads: dict[tuple, Candidate] = {}
    for cand in candidates:
        key = _cluster_key(cand.event)
        head = heads.get(key)
        if head is None:
            heads[key] = cand
        elif cand.weight > head.weight:
            # Keep the most informative one as the face of the cluster, and the
            # picture the user will actually judge from.
            cand.duplicates = [head.event, *head.duplicates]
            heads[key] = cand
        else:
            head.duplicates.append(cand.event)
    return list(heads.values())


def _cluster_key(event: Event) -> tuple:
    night = is_night(event.minute_of_day)
    if event.map_point is None:
        # No pose for this one: fall back to "same surface, same reason, same
        # half of the day", which is coarse but never merges unlike things.
        return (event.surface_id, night, tuple(event.blocked_by), event.fired)
    cx = math.floor(event.map_point[0] / CLUSTER_CELL)
    cy = math.floor(event.map_point[1] / CLUSTER_CELL)
    return (event.surface_id, night, cx, cy, event.fired)


def _rank(candidates: list[Candidate]) -> list[Candidate]:
    """Most informative first, with repeats counting for something."""
    return sorted(
        candidates,
        key=lambda c: (c.weight + 0.1 * min(c.count - 1, 5), c.event.ts),
        reverse=True,
    )


# ------------------------------------------------------------------ scheduling


@dataclass
class Invitation:
    """Whether to offer the review, and what the home card should say."""

    offer: bool
    headline: str = ""
    body: str = ""
    reason: str = ""  # why not, when offer is False — for Diagnostics, not the user


def consider(
    deck: Deck,
    last_review_at: float,
    enabled: bool = True,
    declines: int = 0,
    now: float | None = None,
) -> Invitation:
    """The gate in front of the invitation card.

    Deck quality first, calendar second. A weekly prompt that fires on the
    calendar alone will eventually arrive with three boring cards in it, and that
    is the one that teaches the user to stop opening it.
    """
    now = time.time() if now is None else now
    if not enabled:
        return Invitation(False, reason="turned off in Settings")
    if declines >= MAX_DECLINES:
        return Invitation(False, reason="declined twice; waiting to be asked for")
    if now - last_review_at < REVIEW_INTERVAL_S:
        days = (REVIEW_INTERVAL_S - (now - last_review_at)) / 86_400.0
        return Invitation(False, reason=f"reviewed recently; {days:.1f} days to go")
    if deck.size < MIN_DECK:
        return Invitation(False, reason=f"only {deck.size} things worth asking about")

    fired = deck.total_fired
    unsure = deck.size
    headline = (f"I played {fired} sound{'s' if fired != 1 else ''} this week."
                if fired else "I stayed quiet all week.")
    body = f"There are {unsure} I'm not sure about. Grade me?"
    return Invitation(True, headline, body)


# ----------------------------------------------------------------- adjustments


@dataclass
class Adjustment:
    """One change the review wants to make, and everything needed to undo it.

    ``previous`` is captured at plan time rather than read back at undo time, so
    undo restores what was actually replaced even if something else has since
    touched the surface.
    """

    surface_id: str
    kind: str            # blind_spot | min_score | scale_tolerance | forget_blind_spot | none
    title: str
    detail: str
    payload: dict = field(default_factory=dict)
    previous: dict = field(default_factory=dict)
    event_ids: list[int] = field(default_factory=list)
    applied: bool = False

    @property
    def reversible(self) -> bool:
        return self.kind != "none"

    def apply(self, surface: Surface) -> None:
        if self.applied or self.kind == "none":
            self.applied = True
            return
        t = surface.tuning
        if self.kind == "blind_spot":
            t.blind_spots.append(BlindSpot(**self.payload))
        elif self.kind == "forget_blind_spot":
            t.blind_spots = [b for b in t.blind_spots
                             if not _same_spot(b, self.payload)]
        elif self.kind == "min_score":
            t.min_score = self.payload["min_score"]
        elif self.kind == "scale_tolerance":
            t.scale_tolerance = self.payload["scale_tolerance"]
        self.applied = True

    def undo(self, surface: Surface) -> None:
        if not self.applied or self.kind == "none":
            self.applied = False
            return
        t = surface.tuning
        if self.kind == "blind_spot":
            t.blind_spots = [b for b in t.blind_spots if not _same_spot(b, self.payload)]
        elif self.kind == "forget_blind_spot":
            t.blind_spots.append(BlindSpot(**self.payload))
        elif self.kind == "min_score":
            t.min_score = self.previous.get("min_score")
        elif self.kind == "scale_tolerance":
            t.scale_tolerance = self.previous.get("scale_tolerance")
        self.applied = False


def _same_spot(spot: BlindSpot, payload: dict) -> bool:
    return (abs(spot.x - payload["x"]) < 1e-6 and abs(spot.y - payload["y"]) < 1e-6
            and abs(spot.radius - payload["radius"]) < 1e-6)


@dataclass
class Outcome:
    """What a finished review produced — the whole payoff screen, as data."""

    adjustments: list[Adjustment] = field(default_factory=list)
    regression_cases: int = 0
    right: int = 0
    wrong: int = 0
    unsure: int = 0
    labelled_events: int = 0

    @property
    def graded(self) -> int:
        return self.right + self.wrong

    @property
    def accuracy(self) -> float | None:
        return None if self.graded == 0 else self.right / self.graded

    @property
    def changed_nothing(self) -> bool:
        return not any(a.kind != "none" for a in self.adjustments)

    def headline(self) -> str:
        real = [a for a in self.adjustments if a.kind != "none"]
        if not real and self.wrong == 0 and self.graded:
            return "Nothing needed changing — you agreed with all of them."
        if not real:
            return "Thanks — nothing here I can retune on my own."
        return f"Thanks — that changed {_count_word(len(real))}."


def plan_adjustments(
    labels: dict[int, str],
    deck: Deck,
    surfaces: list[Surface],
) -> Outcome:
    """Turn a set of verdicts into the changes they justify — and no more.

    ``labels`` maps an event id to one of ``activity_log.VERDICTS``. Only the
    cards in ``deck`` are consulted, because a card knows how many events it
    stands for and a bare event does not.
    """
    by_id = {s.id: s for s in surfaces}
    outcome = Outcome()
    # surface id -> lists of (candidate, verdict)
    per_surface: dict[str, list[tuple[Candidate, str]]] = {}

    for card in deck.cards:
        verdict = labels.get(card.event.id)
        if not verdict:
            continue
        outcome.labelled_events += card.count
        per_surface.setdefault(card.event.surface_id, []).append((card, verdict))
        if verdict == "unsure":
            outcome.unsure += card.count
        elif _app_was_right(card, verdict):
            outcome.right += card.count
        else:
            outcome.wrong += card.count

    for surface_id, graded in per_surface.items():
        surface = by_id.get(surface_id)
        if surface is None:
            continue
        outcome.adjustments.extend(_adjust_surface(surface, graded, outcome))
    return outcome


def app_was_right(fired: bool, verdict: str) -> bool | None:
    """Was the app right, given what the user said? None when they could not say.

    Note the asymmetry: a sound that should not have played and a cat that should
    have been deterred are both wrong, but only one of them was ever visible to
    the user before this screen existed. Counting both is the only way the score
    means anything.
    """
    if verdict in ("", "unsure"):
        return None
    if fired:
        return verdict == "correct"
    return verdict != "missed"


def _app_was_right(card: Candidate, verdict: str) -> bool:
    return bool(app_was_right(card.kind is Kind.ALERT, verdict))


def weekly_accuracy(
    events: list[Event], weeks: int = 5, now: float | None = None
) -> list[tuple[int, int]]:
    """(right, graded) per week, oldest first — the trend on the payoff screen."""
    now = time.time() if now is None else now
    buckets = [[0, 0] for _ in range(weeks)]
    for event in events:
        if not event.feedback:
            continue
        age = now - event.ts
        index = weeks - 1 - int(age // REVIEW_INTERVAL_S)
        if not 0 <= index < weeks:
            continue
        right = app_was_right(event.fired, event.feedback)
        if right is None:
            continue
        buckets[index][1] += 1
        buckets[index][0] += int(right)
    return [(r, g) for r, g in buckets]


def _adjust_surface(
    surface: Surface,
    graded: list[tuple[Candidate, str]],
    outcome: Outcome,
) -> list[Adjustment]:
    out: list[Adjustment] = []
    false_alarms = [c for c, v in graded if c.kind is Kind.ALERT and v in
                    ("not_a_cat", "not_on_surface")]
    confirmed = [c for c, v in graded if c.kind is Kind.ALERT and v == "correct"]
    missed = [c for c, v in graded if v == "missed"]
    people = [c for c, v in graded if v == "person"]

    # --- repeated false alarms in one place become a blind spot ---------------
    for group in _group_by_place(false_alarms):
        count = sum(c.count for c in group)
        located = [c for c in group if c.event.map_point is not None]
        if count < MIN_CLUSTER_FOR_BLIND_SPOT or not located:
            # Not enough to go blind on. It still drops the size sample it fed,
            # and it is still a case the app can be replayed against.
            outcome.regression_cases += count
            continue
        xs = [c.event.map_point[0] for c in located]
        ys = [c.event.map_point[1] for c in located]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        spread = max((math.dist((x, y), (cx, cy)) for x, y in zip(xs, ys)), default=0.0)
        night_only = all(is_night(c.event.minute_of_day) for c in group)
        when = " after dark" if night_only else ""
        out.append(Adjustment(
            surface_id=surface.id,
            kind="blind_spot",
            title=f"One spot on the {surface.name.lower()}{when}",
            detail=(f"All {count} of these came from one spot. "
                    f"I'll stop calling that a cat{when}."),
            payload={
                "x": cx, "y": cy, "radius": spread + BLIND_SPOT_MARGIN,
                "night_only": night_only,
                "note": f"a spot on the {surface.name.lower()} you marked",
                "created": time.time(),
            },
            event_ids=[i for c in group for i in c.ids],
        ))

    # --- missed cats, folded into one change per knob -------------------------
    # Two cards can both point at the same dial. Emitting an adjustment for each
    # would show the user the same sentence twice and, worse, make the second
    # one's "undo" restore a value the first had already moved.
    reopen: dict[tuple, Candidate] = {}
    widen: list[Candidate] = []
    lower: list[Candidate] = []
    unfixable: list[Candidate] = []

    for card in missed:
        blocked = card.event.blocked_by
        spot = None
        if "known_false_alarm" in blocked and card.event.map_point is not None:
            spot = surface.tuning.blind_spot_at(card.event.map_point,
                                                card.event.minute_of_day)
        if spot is not None:
            reopen.setdefault((spot.x, spot.y, spot.radius), card)
        elif "scale" in blocked:
            widen.append(card)
        elif "confidence" in blocked and surface.tuning.min_score is not None:
            lower.append(card)
        else:
            unfixable.append(card)

    for (x, y, radius), card in reopen.items():
        spot = next(b for b in surface.tuning.blind_spots
                    if (b.x, b.y, b.radius) == (x, y, radius))
        out.append(Adjustment(
            surface_id=surface.id, kind="forget_blind_spot",
            title=f"I'll stop ignoring that spot on the {surface.name.lower()}",
            detail="You marked it as a false alarm before, and a real cat has now "
                   "been there.",
            payload={"x": spot.x, "y": spot.y, "radius": spot.radius,
                     "night_only": spot.night_only, "note": spot.note,
                     "created": spot.created},
            event_ids=card.ids,
        ))

    if widen:
        current = surface.scale_tolerance(1.5)
        widened = min(MAX_TOLERANCE, round(current * RELAX_FACTOR, 2))
        if widened > current:
            n = sum(c.count for c in widen)
            out.append(Adjustment(
                surface_id=surface.id, kind="scale_tolerance",
                title=f"I'll accept odder-looking cats on the {surface.name.lower()}",
                detail=f"{_were(n)} the wrong size by my reckoning. Widening the "
                       f"size check from {current:.2f}x to {widened:.2f}x.",
                payload={"scale_tolerance": widened},
                previous={"scale_tolerance": surface.tuning.scale_tolerance},
                event_ids=[i for c in widen for i in c.ids],
            ))
        else:
            unfixable.extend(widen)

    if lower and surface.tuning.min_score is not None:
        lowered = round(max(0.35, surface.tuning.min_score - 0.1), 2)
        if lowered < surface.tuning.min_score:
            out.append(Adjustment(
                surface_id=surface.id, kind="min_score",
                title=f"I'll act on less certainty on the {surface.name.lower()}",
                detail=f"Lowering the bar from {surface.tuning.min_score:.0%} to "
                       f"{lowered:.0%}.",
                payload={"min_score": lowered},
                previous={"min_score": surface.tuning.min_score},
                event_ids=[i for c in lower for i in c.ids],
            ))
        else:
            unfixable.extend(lower)

    if unfixable:
        # Nothing honest to change: no dial the app owns would have caught these.
        # Say so plainly rather than inventing a knob that did not move.
        n = sum(c.count for c in unfixable)
        outcome.regression_cases += n
        out.append(Adjustment(
            surface_id=surface.id, kind="none",
            title=f"{_count_word(n).capitalize()} I missed on the {surface.name.lower()}",
            detail="I can't fix "
                   + ("these" if n > 1 else "this one")
                   + " by myself — I never saw a cat there at all. "
                   + ("They're" if n > 1 else "It's") + " saved as test cases.",
            event_ids=[i for c in unfixable for i in c.ids],
        ))

    # --- consistent agreement earns a tighter size check ----------------------
    confirmed_count = sum(c.count for c in confirmed)
    if (confirmed_count >= MIN_CONFIRMATIONS_TO_TIGHTEN and not false_alarms
            and not missed and surface.calibration_samples >= MIN_CALIBRATION_SAMPLES):
        current = surface.scale_tolerance(1.5)
        tightened = max(MIN_TOLERANCE, round(current * TIGHTEN_FACTOR, 2))
        if tightened < current:
            out.append(Adjustment(
                surface_id=surface.id, kind="scale_tolerance",
                title=f"{surface.name} is now stricter",
                detail=f"You confirmed {confirmed_count} of {confirmed_count}. "
                       f"Tightening the size check from {current:.2f}x to "
                       f"{tightened:.2f}x.",
                payload={"scale_tolerance": tightened},
                event_ids=[i for c in confirmed for i in c.ids],
            ))

    outcome.regression_cases += sum(c.count for c in people)
    return out


def _group_by_place(cards: list[Candidate]) -> list[list[Candidate]]:
    groups: dict[tuple, list[Candidate]] = {}
    for card in cards:
        groups.setdefault(_cluster_key(card.event), []).append(card)
    return list(groups.values())


def _were(n: int) -> str:
    return "This one was" if n == 1 else f"{_count_word(n).split()[0].capitalize()} of these were"


def _count_word(n: int) -> str:
    words = {1: "one thing", 2: "two things", 3: "three things", 4: "four things"}
    return words.get(n, f"{n} things")
