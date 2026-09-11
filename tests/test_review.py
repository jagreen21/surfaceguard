"""The weekly review's decisions: what it asks about, and what it changes.

All of this is deliberately reachable without Qt. The interesting failures here
are sampling the wrong events and applying a change nobody agreed to, and neither
needs a window to go wrong.
"""

import time

import numpy as np
import pytest

from surfaceguard.geometry.surface import MIN_CALIBRATION_SAMPLES, BlindSpot, Surface
from surfaceguard.storage.activity_log import Event
from surfaceguard.storage.review_policy import (
    MAX_CARDS,
    Kind,
    app_was_right,
    build_deck,
    consider,
    plan_adjustments,
    weekly_accuracy,
)

DAY = 86_400.0
COUNTER = np.array([[220, 300], [420, 300], [520, 420], [120, 420]], float)


def counter(**kw) -> Surface:
    return Surface("Kitchen counter", COUNTER.copy(), id="s1", **kw)


def at(hour: int, days_ago: float = 1.0) -> float:
    """A timestamp at a given local hour, roughly ``days_ago`` back."""
    base = time.time() - days_ago * DAY
    lt = time.localtime(base)
    return base + (hour - lt.tm_hour) * 3600.0


def event(
    event_id: int,
    *,
    fired: bool = False,
    blocked: tuple[str, ...] = (),
    ts: float | None = None,
    map_point: tuple[float, float] | None = (300.0, 350.0),
    picture: bool = True,
    feedback: str | None = None,
    degraded: tuple[str, ...] = (),
    reason: str = "cooling down",
) -> Event:
    gates = [{"name": "inside", "status": "pass", "detail": "", "required": True}]
    gates += [{"name": n, "status": "fail", "detail": "", "required": True} for n in blocked]
    gates += [{"name": n, "status": "unavailable", "detail": "", "required": False}
              for n in degraded]
    return Event(
        id=event_id, ts=at(3) if ts is None else ts, surface_id="s1",
        surface_name="Kitchen counter", fired=fired,
        reason="deterrent played" if fired else reason,
        gates=gates, score=0.6, box=(10, 10, 50, 60), latency_ms=300.0,
        thumbnail="t.jpg" if picture else None, feedback=feedback,
        map_point=map_point, strip=["a.jpg", "b.jpg"] if picture else [],
    )


# ------------------------------------------------------------------- sampling


def test_every_alert_is_asked_about():
    deck = build_deck([
        event(1, fired=True, map_point=(0.0, 0.0)),
        event(2, fired=True, map_point=(300.0, 0.0)),
    ])
    assert len(deck.alerts) == 2
    assert all(c.kind is Kind.ALERT for c in deck.alerts)


def test_events_already_answered_are_not_asked_again():
    deck = build_deck([event(1, fired=True, feedback="correct")])
    assert deck.is_empty


def test_an_event_with_no_picture_is_never_a_card():
    """Nothing to show means nothing to ask — counted, so the reason survives."""
    deck = build_deck([event(1, fired=True, picture=False)])
    assert deck.is_empty
    assert deck.skipped_no_picture == 1


def test_a_cat_somewhere_else_entirely_is_not_worth_a_card():
    """Only gates the review can act on earn a question."""
    deck = build_deck([event(1, blocked=("visible",), ts=at(3))])
    # 'visible' is sampled, but at the bottom; it must not outrank a size call.
    better = build_deck([event(1, blocked=("visible",)), event(2, blocked=("scale",))])
    assert better.grid[0].event.id == 2


def test_repeats_in_one_place_become_a_single_card():
    events = [event(i, fired=True, ts=at(3, days_ago=1 + i * 0.01)) for i in range(1, 15)]
    deck = build_deck(events)
    assert len(deck.alerts) == 1
    assert deck.alerts[0].count == 14
    assert "14 times" in deck.alerts[0].repeat_note
    assert len(deck.alerts[0].ids) == 14


def test_the_same_place_at_a_different_time_of_day_is_a_different_question():
    deck = build_deck([event(1, fired=True, ts=at(3)), event(2, fired=True, ts=at(14))])
    assert len(deck.alerts) == 2


def test_far_apart_on_the_same_surface_are_different_questions():
    deck = build_deck([
        event(1, fired=True, map_point=(0.0, 0.0)),
        event(2, fired=True, map_point=(400.0, 400.0)),
    ])
    assert len(deck.alerts) == 2


def test_a_quiet_week_of_near_misses_fills_the_grid():
    events = [
        event(i, blocked=("scale",), map_point=(float(i) * 50, 0.0), ts=at(3, 1 + i * 0.01))
        for i in range(60)
    ]
    deck = build_deck(events)
    assert deck.alerts == []
    assert len(deck.grid) == MAX_CARDS


# ----------------------------------------------------------------- scheduling


def test_a_thin_week_is_not_worth_asking_about():
    deck = build_deck([event(1, fired=True)])
    assert not consider(deck, last_review_at=0.0).offer


def test_a_good_deck_is_offered_once_the_week_is_up():
    deck = build_deck([event(i, fired=True, map_point=(float(i) * 50, 0.0)) for i in range(8)])
    assert consider(deck, last_review_at=0.0).offer


def test_it_does_not_ask_twice_in_one_week():
    deck = build_deck([event(i, fired=True, map_point=(float(i) * 50, 0.0)) for i in range(8)])
    assert not consider(deck, last_review_at=time.time() - DAY).offer


def test_a_week_of_nothing_but_false_alarms_is_still_reviewable():
    """The user drowning in false alarms is the one who most needs to be asked."""
    deck = build_deck([
        event(i, fired=True, map_point=(float(i) * 50, 0.0), ts=at(3, 1 + i * 0.01))
        for i in range(9)
    ])
    assert deck.grid == []
    assert len(deck.alerts) == 9
    assert consider(deck, last_review_at=0.0).offer


def test_the_whole_deck_stays_inside_the_promised_minute():
    events = [
        event(i, fired=i % 2 == 0, blocked=() if i % 2 == 0 else ("scale",),
              map_point=(float(i) * 50, 0.0), ts=at(3, 1 + i * 0.01))
        for i in range(40)
    ]
    deck = build_deck(events)
    assert deck.size == 12


def test_two_refusals_stop_the_asking():
    deck = build_deck([event(i, fired=True, map_point=(float(i) * 50, 0.0)) for i in range(8)])
    assert not consider(deck, last_review_at=0.0, declines=2).offer
    assert not consider(deck, last_review_at=0.0, enabled=False).offer


# ---------------------------------------------------------------- adjustments


def test_repeated_false_alarms_in_one_spot_earn_a_blind_spot():
    events = [event(i, fired=True, ts=at(3, 1 + i * 0.01)) for i in range(1, 5)]
    deck = build_deck(events)
    card = deck.alerts[0]
    surface = counter()
    outcome = plan_adjustments({card.event.id: "not_a_cat"}, deck, [surface])

    assert [a.kind for a in outcome.adjustments] == ["blind_spot"]
    adjustment = outcome.adjustments[0]
    assert adjustment.payload["night_only"] is True
    adjustment.apply(surface)
    assert surface.tuning.blind_spot_at((300.0, 350.0), 3 * 60) is not None
    # ...and only at night, which is what the card said it would do.
    assert surface.tuning.blind_spot_at((300.0, 350.0), 14 * 60) is None


def test_one_false_alarm_does_not_blind_the_app_to_a_spot():
    """An anecdote is not a pattern. It still becomes a case to replay."""
    deck = build_deck([event(1, fired=True)])
    surface = counter()
    outcome = plan_adjustments({1: "not_a_cat"}, deck, [surface])
    assert outcome.adjustments == []
    assert outcome.regression_cases == 1
    assert surface.tuning.is_empty


def test_every_adjustment_can_be_taken_back_exactly():
    events = [event(i, fired=True, ts=at(3, 1 + i * 0.01)) for i in range(1, 5)]
    deck = build_deck(events)
    surface = counter()
    surface.tuning.min_score = 0.5
    outcome = plan_adjustments({deck.alerts[0].event.id: "not_a_cat"}, deck, [surface])
    before = len(surface.tuning.blind_spots)

    adjustment = outcome.adjustments[0]
    adjustment.apply(surface)
    assert len(surface.tuning.blind_spots) == before + 1
    adjustment.undo(surface)
    assert len(surface.tuning.blind_spots) == before
    assert surface.tuning.min_score == 0.5   # untouched by an unrelated undo


def test_a_missed_cat_reopens_a_spot_the_app_had_been_told_to_ignore():
    """The review has to be able to undo its own past advice."""
    surface = counter()
    surface.tuning.blind_spots.append(
        BlindSpot(x=300.0, y=350.0, radius=40.0, note="the lamp shadow")
    )
    deck = build_deck([event(1, blocked=("known_false_alarm",))])
    outcome = plan_adjustments({1: "missed"}, deck, [surface])

    assert [a.kind for a in outcome.adjustments] == ["forget_blind_spot"]
    outcome.adjustments[0].apply(surface)
    assert surface.tuning.blind_spots == []
    outcome.adjustments[0].undo(surface)
    assert len(surface.tuning.blind_spots) == 1


def test_a_missed_cat_blocked_on_size_widens_the_size_check():
    deck = build_deck([event(1, blocked=("scale",))])
    surface = counter()
    outcome = plan_adjustments({1: "missed"}, deck, [surface])
    outcome.adjustments[0].apply(surface)
    assert surface.tuning.scale_tolerance > 1.5


def test_a_cat_the_detector_never_saw_is_admitted_not_papered_over():
    """The honest answer to "you missed one" is sometimes "I can't fix that"."""
    deck = build_deck([event(1, blocked=("no_person",))])
    surface = counter()
    outcome = plan_adjustments({1: "missed"}, deck, [surface])
    assert [a.kind for a in outcome.adjustments] == ["none"]
    assert outcome.regression_cases == 1
    assert surface.tuning.is_empty
    assert not outcome.adjustments[0].reversible


def test_consistent_agreement_tightens_a_calibrated_surface():
    events = [
        event(i, fired=True, map_point=(float(i) * 100, 0.0), ts=at(3, 1 + i * 0.01))
        for i in range(1, 6)
    ]
    deck = build_deck(events)
    surface = counter(height_samples=[1.0] * MIN_CALIBRATION_SAMPLES)
    outcome = plan_adjustments({c.event.id: "correct" for c in deck.alerts}, deck, [surface])
    assert [a.kind for a in outcome.adjustments] == ["scale_tolerance"]
    outcome.adjustments[0].apply(surface)
    assert surface.tuning.scale_tolerance < 1.5


def test_an_uncalibrated_surface_is_not_tightened():
    """Tightening a size check that has no size model would veto real cats."""
    events = [
        event(i, fired=True, map_point=(float(i) * 100, 0.0), ts=at(3, 1 + i * 0.01))
        for i in range(1, 6)
    ]
    deck = build_deck(events)
    outcome = plan_adjustments({c.event.id: "correct" for c in deck.alerts}, deck, [counter()])
    assert outcome.adjustments == []


def test_agreeing_with_everything_changes_nothing_and_says_so():
    events = [
        event(i, fired=True, map_point=(float(i) * 100, 0.0), ts=at(3, 1 + i * 0.01))
        for i in range(1, 4)
    ]
    deck = build_deck(events)
    outcome = plan_adjustments({c.event.id: "correct" for c in deck.alerts}, deck, [counter()])
    assert outcome.changed_nothing
    assert "agreed" in outcome.headline()


def test_a_cluster_scores_for_every_event_it_stands_for():
    events = [event(i, fired=True, ts=at(3, 1 + i * 0.01)) for i in range(1, 15)]
    deck = build_deck(events)
    outcome = plan_adjustments({deck.alerts[0].event.id: "correct"}, deck, [counter()])
    assert outcome.right == 14


# --------------------------------------------------------------------- scoring


@pytest.mark.parametrize("fired,verdict,expected", [
    (True, "correct", True),
    (True, "not_a_cat", False),
    (True, "person", False),
    (False, "missed", False),          # silence can be wrong too
    (False, "not_on_surface", True),
    (True, "unsure", None),
])
def test_who_was_right(fired, verdict, expected):
    assert app_was_right(fired, verdict) is expected


def test_the_trend_buckets_by_week():
    events = [
        event(1, fired=True, ts=time.time() - 0.5 * DAY, feedback="correct"),
        event(2, fired=True, ts=time.time() - 0.6 * DAY, feedback="not_a_cat"),
        event(3, fired=True, ts=time.time() - 8 * DAY, feedback="correct"),
        event(4, fired=True, ts=time.time() - 100 * DAY, feedback="correct"),
    ]
    trend = weekly_accuracy(events, weeks=5)
    assert trend[-1] == (1, 2)
    assert trend[-2] == (1, 1)
    assert sum(g for _, g in trend) == 3      # the 100-day-old one falls outside
