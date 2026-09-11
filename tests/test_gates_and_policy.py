"""The decision core: which gate blocked, and when does the policy fire."""

import numpy as np
import pytest

from surfaceguard.detection.gates import Status, evaluate, surfaces_for_pose
from surfaceguard.detection.trigger_policy import Presence, TriggerPolicy
from surfaceguard.geometry.projection import Box, Pose
from surfaceguard.geometry.surface import Surface

POSE = Pose.identity((640, 480))
COUNTER = np.array([[220, 300], [420, 300], [520, 420], [120, 420]], float)


def counter() -> Surface:
    return Surface("Kitchen counter", COUNTER.copy())


def cat_on_counter() -> Box:
    return Box(280, 300, 360, 360)          # paw at (320, 360): inside


def cat_on_floor() -> Box:
    return Box(280, 420, 360, 470)          # paw at (320, 470): below the counter


def gate(verdict, name):
    return next(g for g in verdict.gates if g.name == name)


def test_cat_on_surface_passes_every_gate():
    v = evaluate(counter(), POSE, cat_on_counter(), [])
    assert v.on_surface
    assert gate(v, "inside").status is Status.PASS
    assert gate(v, "visible").status is Status.PASS


def test_cat_on_the_floor_is_blocked_by_inside_only():
    v = evaluate(counter(), POSE, cat_on_floor(), [])
    assert not v.on_surface
    assert v.blocked_by == ["inside"]


def test_person_near_the_surface_suppresses_the_trigger():
    """The most likely reason the app gets switched off, if it gets this wrong."""
    person = Box(300, 200, 380, 460, label="person")
    v = evaluate(counter(), POSE, cat_on_counter(), [person])
    assert not v.on_surface
    assert "no_person" in v.blocked_by


def test_partly_visible_surface_is_not_judged():
    s = counter()
    narrow = Pose(np.eye(3), (300, 480))    # the counter runs off the right edge
    v = evaluate(s, narrow, cat_on_counter(), [])
    assert "visible" in v.blocked_by


def test_scale_gate_is_advisory_before_calibration_and_does_not_veto():
    v = evaluate(counter(), POSE, cat_on_counter(), [])
    assert gate(v, "scale").status is Status.UNAVAILABLE
    assert v.on_surface, "an uncalibrated size check must never block a real cat"
    assert "scale" in v.degraded


def test_scale_gate_rejects_a_cat_that_is_too_large_once_calibrated():
    s = counter()
    for _ in range(10):
        s.observe_height(POSE, cat_on_counter())
    mid_leap = Box(240, 180, 400, 360)      # same paw pixel, ~3x the height
    v = evaluate(s, POSE, mid_leap, [])
    assert gate(v, "scale").status is Status.FAIL
    assert "scale" in v.blocked_by
    assert evaluate(s, POSE, cat_on_counter(), []).on_surface


def test_clipped_box_is_flagged_but_does_not_veto():
    clipped = Box(280, 320, 360, 480, clipped_bottom=True)
    v = evaluate(counter(), POSE, clipped, [])
    assert "unclipped" in v.degraded
    assert "unclipped" not in v.blocked_by


def test_surfaces_for_pose_skips_disabled_and_off_frame():
    near = counter()
    far = Surface("Hallway shelf", COUNTER + np.array([5000.0, 0.0]))
    off = counter()
    off.enabled = False
    assert [s.name for s in surfaces_for_pose([near, far, off], POSE)] == ["Kitchen counter"]


# ------------------------------------------------------------------- policy


def test_dwell_is_required_before_firing():
    s, policy = counter(), TriggerPolicy(enter_s=0.5, exit_s=2.0)
    v = evaluate(s, POSE, cat_on_counter(), [])
    assert not policy.update(s, v, now=100.0).fire
    assert policy.state_of(s.id) is Presence.ARRIVING
    assert not policy.update(s, v, now=100.3).fire
    assert policy.update(s, v, now=100.6).fire


def test_a_single_frame_cannot_flip_presence_either_way():
    """Asymmetric hysteresis: quick to commit, slow to let go."""
    s, policy = counter(), TriggerPolicy(enter_s=0.5, exit_s=2.0)
    on = evaluate(s, POSE, cat_on_counter(), [])
    off = evaluate(s, POSE, cat_on_floor(), [])
    policy.update(s, on, now=100.0)
    policy.update(s, on, now=100.6)
    assert policy.state_of(s.id) is Presence.PRESENT
    policy.update(s, off, now=100.7)                 # one dropped frame
    assert policy.state_of(s.id) is Presence.LEAVING
    policy.update(s, on, now=100.8)
    assert policy.state_of(s.id) is Presence.PRESENT


def test_one_visit_fires_once_even_with_a_dropped_frame():
    s, policy = counter(), TriggerPolicy(enter_s=0.5, exit_s=2.0)
    on = evaluate(s, POSE, cat_on_counter(), [])
    off = evaluate(s, POSE, cat_on_floor(), [])
    policy.update(s, on, now=100.0)
    assert policy.update(s, on, now=100.6).fire
    policy.update(s, off, now=101.0)                 # flicker
    assert not policy.update(s, on, now=101.2).fire, "same visit must not re-fire"


def test_a_new_visit_fires_again_after_cooldown():
    s, policy = counter(), TriggerPolicy(enter_s=0.5, exit_s=2.0)
    s.deterrent.cooldown_s = 5.0
    on = evaluate(s, POSE, cat_on_counter(), [])
    off = evaluate(s, POSE, cat_on_floor(), [])
    policy.update(s, on, now=100.0)
    assert policy.update(s, on, now=100.6).fire
    for t in (101.0, 103.0, 104.0):                  # gone long enough to end the visit
        policy.update(s, off, now=t)
    assert policy.state_of(s.id) is Presence.CLEAR
    policy.update(s, on, now=110.0)
    assert policy.update(s, on, now=110.6).fire


def test_cooldown_blocks_a_second_visit_that_is_too_soon():
    s, policy = counter(), TriggerPolicy(enter_s=0.5, exit_s=1.0)
    s.deterrent.cooldown_s = 60.0
    on = evaluate(s, POSE, cat_on_counter(), [])
    off = evaluate(s, POSE, cat_on_floor(), [])
    policy.update(s, on, now=100.0)
    assert policy.update(s, on, now=100.6).fire
    for t in (101.0, 102.5):
        policy.update(s, off, now=t)
    policy.update(s, on, now=103.0)
    decision = policy.update(s, on, now=103.6)
    assert not decision.fire and "cooling down" in decision.reason


def test_schedule_blocks_outside_its_window():
    s, policy = counter(), TriggerPolicy(enter_s=0.0, exit_s=1.0)
    s.schedule.start_minute, s.schedule.end_minute = 9 * 60, 17 * 60
    on = evaluate(s, POSE, cat_on_counter(), [])
    assert policy.update(s, on, now=100.0, weekday=0, minute_of_day=12 * 60).fire
    policy.reset()
    d = policy.update(s, on, now=200.0, weekday=0, minute_of_day=3 * 60)
    assert not d.fire and "schedule" in d.reason


def test_zero_dwell_fires_on_the_first_qualifying_frame():
    s, policy = counter(), TriggerPolicy(enter_s=0.0, exit_s=1.0)
    on = evaluate(s, POSE, cat_on_counter(), [])
    assert policy.update(s, on, now=100.0).fire


def test_schedule_window_across_midnight():
    s = counter()
    s.schedule.start_minute, s.schedule.end_minute = 22 * 60, 6 * 60
    assert s.schedule.contains(0, 23 * 60)
    assert s.schedule.contains(0, 2 * 60)
    assert not s.schedule.contains(0, 12 * 60)
