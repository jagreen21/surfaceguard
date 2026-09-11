"""The self-test behind decision D4.

The app must never display "Protection on" unless something has just verified that
it could actually act. Every check here is cheap, runs while armed, and reports a
sentence the user can read — not a status code.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# Budgets, all deliberately looser than the §7 latency table: these detect
# breakage, not slow frames.
MAX_FRAME_AGE_S = 6.0
MAX_INFERENCE_MS = 250.0
MAX_REGISTRATION_MS = 120.0


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True
    # Text shown to the user when this check fails; the reason *and* the fix.
    remedy: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)
    at: float = field(default_factory=time.time)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks if c.required)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.ok and c.required]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if not c.ok and not c.required]

    def headline(self) -> str:
        if self.ok:
            return "All checks passing"
        first = self.failures[0]
        return first.detail


@dataclass
class Metrics:
    """Rolling numbers the engine keeps and the heartbeat judges."""

    last_frame_at: float = 0.0
    frames: int = 0
    inference_ms: float = 0.0
    registration_ms: float = 0.0
    inliers: int = 0
    registered: bool = False
    fps: float = 0.0
    end_to_end_ms: list[float] = field(default_factory=list)

    def note_latency(self, ms: float, keep: int = 200) -> None:
        self.end_to_end_ms.append(ms)
        if len(self.end_to_end_ms) > keep:
            del self.end_to_end_ms[: len(self.end_to_end_ms) - keep]

    def p95_latency_ms(self) -> float | None:
        if not self.end_to_end_ms:
            return None
        s = sorted(self.end_to_end_ms)
        return s[min(len(s) - 1, int(0.95 * len(s)))]


def run_checks(
    metrics: Metrics,
    *,
    audio_ok: bool,
    audio_detail: str = "",
    min_inliers: int = 30,
    surfaces_total: int = 0,
    surfaces_covered: int = 0,
    detector_available: bool = True,
    detector_note: str = "",
    bridge=None,
    update_token=None,
    now: float | None = None,
) -> Report:
    """Assemble one heartbeat report from the engine's current metrics."""
    now = time.monotonic() if now is None else now
    checks: list[Check] = []

    age = now - metrics.last_frame_at if metrics.last_frame_at else float("inf")
    checks.append(Check(
        "video",
        age <= MAX_FRAME_AGE_S,
        "Receiving video" if age <= MAX_FRAME_AGE_S else (
            "No video from the camera" if age == float("inf")
            else f"No video for {age:.0f} seconds"
        ),
        remedy="Check the camera is powered on and on the same network.",
    ))

    if not detector_available:
        # A detector that cannot run is worse than a slow one: it never returns
        # anything, so every other check passes and the app looks healthy while
        # being completely blind. This must fail loudly (D4).
        checks.append(Check(
            "detector",
            False,
            detector_note or "No cat detector is installed, so nothing can be detected",
            remedy="Install a detection model — see Settings.",
        ))
    else:
        fast_enough = metrics.frames > 0 and metrics.inference_ms <= MAX_INFERENCE_MS
        checks.append(Check(
            "detector",
            fast_enough,
            "Looking for cats" if fast_enough
            else f"Detection is too slow ({metrics.inference_ms:.0f} ms per frame)",
            remedy="Close other heavy apps, or lower the frame rate in Settings.",
        ))

    checks.append(Check(
        "audio",
        audio_ok,
        "Sound ready" if audio_ok else "Sound could not play",
        remedy=audio_detail or "Check the output device in System Settings > Sound.",
    ))

    checks.append(Check(
        "view",
        metrics.registered and metrics.inliers >= min_inliers,
        "Camera view recognised" if metrics.registered and metrics.inliers >= min_inliers
        else "Camera view has moved",
        remedy="Point the camera back at the room, or re-scan the room in Settings.",
    ))

    if bridge is not None and not bridge.healthy:
        checks.append(Check(
            "camera_service",
            False,
            bridge.fatal or "The camera service is not running",
            remedy=(
                "Sign in to the camera account again in Settings."
                if bridge.fatal else "It is restarting on its own; this usually clears."
            ),
        ))

    if update_token is not None and (not update_token.ok or update_token.expiring_soon):
        # Advisory: updates failing does not stop the app guarding. But it must be
        # visible, because a token that lapses in silence means she simply stops
        # receiving fixes and nothing ever says so.
        checks.append(Check(
            "updates",
            False,
            update_token.summary(),
            required=False,
            remedy=update_token.remedy or
            "Create a new access token on GitHub and paste it into Settings.",
        ))

    if surfaces_total:
        covered = surfaces_covered >= surfaces_total
        checks.append(Check(
            "coverage",
            covered,
            "All surfaces watched" if covered
            else f"{surfaces_total - surfaces_covered} of {surfaces_total} surfaces are not being watched",
            required=False,
            remedy="Move the camera so it can see them, or turn those surfaces off.",
        ))

    return Report(checks)
