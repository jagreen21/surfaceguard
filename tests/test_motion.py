"""Motion gating.

A room is empty almost all of the time, and running a network on every frame is
the dominant cost on a four-core laptop with no Neural Engine. The gate exists to
skip that work — and must never make "nothing moved" look like "we stopped
looking", which is the silent failure the whole design is built against.
"""

import numpy as np

from surfaceguard.detection.motion import (
    MAX_CONSECUTIVE_SKIPS,
    SILENCE_IS_SUSPICIOUS_S,
    MotionGate,
    MotionState,
    interval_for,
)

STILL = np.full((270, 480), 100, np.uint8)


def test_an_unchanged_frame_is_skipped():
    gate = MotionGate()
    gate.consider(STILL, now=1.0)                      # first frame always runs
    assert gate.consider(STILL, now=2.0).skipped


def test_sensor_noise_does_not_wake_the_detector():
    """Noise on a dim indoor camera sits near 1.0 mean absolute difference."""
    gate = MotionGate()
    rng = np.random.default_rng(0)
    gate.consider(STILL, now=1.0)
    for i in range(5):
        noisy = np.clip(STILL.astype(np.int16) + rng.integers(-2, 3, STILL.shape), 0, 255)
        assert gate.consider(noisy.astype(np.uint8), now=2.0 + i).skipped


def test_a_cat_wakes_it():
    gate = MotionGate()
    gate.consider(STILL, now=1.0)
    gate.consider(STILL, now=2.0)
    moved = STILL.copy()
    moved[100:180, 200:300] = 200
    state = gate.consider(moved, now=3.0)
    assert not state.skipped and state.moving
    assert state.difference > gate.threshold


def test_a_cat_that_settles_is_still_looked_at():
    """A cat sitting perfectly still stops producing motion, and is still on the
    counter. The gate must force a real look rather than skip forever."""
    gate = MotionGate()
    gate.consider(STILL, now=0.0)
    skips = 0
    for i in range(MAX_CONSECUTIVE_SKIPS + 3):
        if gate.consider(STILL, now=float(i + 1)).skipped:
            skips += 1
        else:
            break
    assert skips <= MAX_CONSECUTIVE_SKIPS, "the detector could sleep indefinitely"


def test_the_first_frame_after_the_camera_moves_is_examined():
    """A different crop shape means there is nothing to compare against."""
    gate = MotionGate()
    gate.consider(STILL, now=1.0)
    gate.consider(STILL, now=2.0)
    assert not gate.consider(np.full((200, 300), 100, np.uint8), now=3.0).skipped


def test_the_cadence_inverts_with_motion():
    still = interval_for(MotionState(moving=False), target_fps=8.0)
    moving = interval_for(MotionState(moving=True), target_fps=8.0)
    assert moving < 1.0 / 8.0 < still, "a moving room must be sampled faster than nominal"


def test_a_quiet_room_is_not_a_stopped_detector():
    """The invariant: skipping is expected; going silent for too long is a fault."""
    state = MotionState(last_inference_at=100.0)
    assert not state.stale(100.0 + SILENCE_IS_SUSPICIOUS_S - 1)
    assert state.stale(100.0 + SILENCE_IS_SUSPICIOUS_S + 1)
    assert not MotionState().stale(1e9), "never having run yet is not staleness"


def test_the_heartbeat_distinguishes_the_two():
    import time as _t

    from surfaceguard.health.heartbeat import Metrics, run_checks
    from surfaceguard.state import Phase, StateStore

    def report(**kw):
        m = Metrics(last_frame_at=_t.monotonic(), frames=500, inference_ms=20,
                    inliers=180, registered=True, fps=8.0, **kw)
        store = StateStore(has_map=True, has_surfaces=True)
        store.arm()
        store.report = run_checks(m, audio_ok=True)
        return store

    quiet = report(motion_skip_rate=0.9)
    assert quiet.report.ok, "skipping a still room must not be a fault"
    assert quiet.state().phase is Phase.GUARDING
    # The headline stays "Protecting / Looking for cats" — she does not need to
    # know about skipped frames. Diagnostics carries the detail.
    detector = next(c for c in quiet.report.checks if c.name == "detector")
    assert "still" in detector.detail.lower()

    stopped = report(detector_idle=True)
    assert not stopped.report.ok, "a detector that stopped looking must be a fault"
    assert stopped.state().phase is Phase.PROBLEM
