"""Whether anything moved, so the detector can stay asleep when nothing did.

A room is empty almost all of the time. Running a neural network on every frame
regardless is the dominant cost on the machine this ships to — a four-core i5 with
no Neural Engine — and it runs while ``caffeinate`` holds the laptop awake, so it
is heat, fan noise and battery for nothing.

Registration already produces a small greyscale of every frame
(``registration._to_work``). Differencing consecutive ones inside the projected
surface bounds costs microseconds and answers the only question that matters
before inference: is this frame worth looking at?

The budget is then inverted rather than simply reduced. Still room: sample slowly.
Something moving: run faster than the nominal rate, because time-to-sound is what
the deterrent depends on.

One thing this must never do is make "nothing moved" look like "we stopped
looking". The heartbeat derives what the app claims about itself, and an idle
detector that reports health is exactly the silent failure the design forbids —
so a skip is recorded as a *decision*, with a timestamp, and going quiet for
longer than SILENCE_IS_SUSPICIOUS_S is reported as a fault.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Mean absolute difference, on 0-255 greyscale, below which a frame is unchanged.
# Sensor noise on a dim indoor camera sits near 1.0; a cat crossing the crop is
# tens. Deliberately closer to the noise floor than to the signal: a missed cat
# costs far more than a wasted inference.
STILL_THRESHOLD = 2.2

# Never skip more than this many frames in a row, whatever the difference says.
# A cat that settles perfectly still on a counter stops producing motion, and it
# is still standing on the counter.
MAX_CONSECUTIVE_SKIPS = 12

# If the detector has not run for this long, something is wrong regardless of how
# quiet the room is. The heartbeat reports it rather than trusting the gate.
SILENCE_IS_SUSPICIOUS_S = 30.0


@dataclass
class MotionState:
    """What the gate decided, in terms the heartbeat and diagnostics can use."""

    moving: bool = True
    difference: float = 0.0
    skipped: bool = False
    consecutive_skips: int = 0
    last_inference_at: float = 0.0
    skipped_total: int = 0
    examined_total: int = 0

    @property
    def skip_rate(self) -> float:
        return self.skipped_total / self.examined_total if self.examined_total else 0.0

    def stale(self, now: float) -> bool:
        """Has the detector been quiet longer than a quiet room can explain?"""
        if not self.last_inference_at:
            return False
        return (now - self.last_inference_at) > SILENCE_IS_SUSPICIOUS_S


class MotionGate:
    """Decides whether a frame is worth running the detector on."""

    def __init__(self, threshold: float = STILL_THRESHOLD) -> None:
        self.threshold = threshold
        self.state = MotionState()
        self._previous: np.ndarray | None = None

    def reset(self) -> None:
        self._previous = None
        self.state = MotionState(last_inference_at=self.state.last_inference_at)

    def consider(self, work: np.ndarray, region=None, now: float = 0.0) -> MotionState:
        """Compare this frame with the last, inside the region that matters."""
        patch = self._patch(work, region)
        state = self.state
        state.examined_total += 1

        previous, self._previous = self._previous, patch
        if previous is None or previous.shape != patch.shape:
            # First frame, or the crop changed because the camera moved. Both mean
            # there is nothing to compare against, so look properly.
            return self._run(state, difference=float("inf"), now=now)

        difference = float(np.mean(np.abs(patch.astype(np.int16) - previous.astype(np.int16))))
        if difference >= self.threshold:
            return self._run(state, difference, now)
        if state.consecutive_skips >= MAX_CONSECUTIVE_SKIPS:
            # A cat that has settled stops moving, and is still on the counter.
            return self._run(state, difference, now)

        state.moving = False
        state.skipped = True
        state.difference = difference
        state.consecutive_skips += 1
        state.skipped_total += 1
        return state

    def note_inference(self, now: float) -> None:
        self.state.last_inference_at = now

    def _run(self, state: MotionState, difference: float, now: float) -> MotionState:
        state.moving = difference >= self.threshold
        state.skipped = False
        state.difference = difference
        state.consecutive_skips = 0
        state.last_inference_at = now or state.last_inference_at
        return state

    @staticmethod
    def _patch(work: np.ndarray, region) -> np.ndarray:
        """The part of the small greyscale frame the surfaces occupy."""
        if region is None or getattr(region, "full_frame", True):
            return work
        h, w = work.shape[:2]
        # `region` is in full-frame pixels; `work` is the downscaled greyscale.
        scale_x = w / max(1.0, float(region.frame_width or w))
        scale_y = h / max(1.0, float(region.frame_height or h))
        x1 = max(0, min(w - 1, int(region.x1 * scale_x)))
        x2 = max(x1 + 1, min(w, int(region.x2 * scale_x)))
        y1 = max(0, min(h - 1, int(region.y1 * scale_y)))
        y2 = max(y1 + 1, min(h, int(region.y2 * scale_y)))
        return work[y1:y2, x1:x2]


def interval_for(state: MotionState, target_fps: float,
                 still_fps: float = 2.0, active_fps: float = 12.0) -> float:
    """How long to wait before the next frame.

    Inverted rather than merely reduced: a still room is sampled slowly, and a
    moving one faster than the nominal rate, because time-to-sound is the thing
    the deterrent actually depends on.
    """
    fps = active_fps if state.moving else still_fps
    fps = max(1.0, min(fps, max(target_fps, active_fps)))
    return 1.0 / fps
