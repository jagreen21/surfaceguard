"""The single state machine the whole UI renders from (decision D4).

Two inputs only: what the user asked for, and what the heartbeat found. The
displayed state is *derived* from both, so there is no code path that can show
"Protection on" while something underneath is broken — the combination is not
representable.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field

from .health.heartbeat import Report


class Intent(enum.Enum):
    """What the user asked for."""

    OFF = "off"
    ARMED = "armed"
    PAUSED = "paused"


class Severity(enum.Enum):
    GOOD = "good"
    WAITING = "waiting"
    PROBLEM = "problem"


class Phase(enum.Enum):
    NEEDS_SETUP = "needs_setup"
    OFF = "off"
    PAUSED = "paused"
    STARTING = "starting"
    GUARDING = "guarding"
    ALERTING = "alerting"
    PROBLEM = "problem"


@dataclass(frozen=True)
class AppState:
    """Everything the home screen needs, already in the user's words."""

    phase: Phase
    headline: str
    detail: str = ""
    severity: Severity = Severity.GOOD
    remedy: str = ""
    paused_until: float | None = None

    @property
    def is_guarding(self) -> bool:
        return self.phase in (Phase.GUARDING, Phase.ALERTING)


@dataclass
class StateStore:
    """Holds intent and the latest heartbeat, and derives the displayed state."""

    intent: Intent = Intent.OFF
    paused_until: float | None = None
    report: Report | None = None
    has_map: bool = False
    has_surfaces: bool = False
    alert_until: float = 0.0
    started_at: float = field(default_factory=time.monotonic)
    # Grace period after arming, before a missing heartbeat counts as a problem.
    startup_grace_s: float = 8.0

    # ------------------------------------------------------------------ intent

    def arm(self) -> None:
        self.intent = Intent.ARMED
        self.paused_until = None
        self.started_at = time.monotonic()

    def turn_off(self) -> None:
        self.intent = Intent.OFF
        self.paused_until = None

    def pause_for(self, seconds: float) -> None:
        self.intent = Intent.PAUSED
        self.paused_until = time.time() + seconds

    def note_alert(self, hold_s: float = 4.0) -> None:
        self.alert_until = time.monotonic() + hold_s

    def tick(self) -> None:
        """Expire a pause whose time has run out."""
        if self.intent is Intent.PAUSED and self.paused_until is not None:
            if time.time() >= self.paused_until:
                self.arm()

    # ------------------------------------------------------------------ derive

    def state(self, now: float | None = None) -> AppState:
        now = time.monotonic() if now is None else now
        self.tick()

        if not self.has_map or not self.has_surfaces:
            missing = "Connect your camera to get started" if not self.has_map \
                else "Draw the first surface you want protected"
            return AppState(Phase.NEEDS_SETUP, "Setup not finished", missing, Severity.WAITING)

        if self.intent is Intent.OFF:
            return AppState(Phase.OFF, "Protection is off", "Nothing is being watched", Severity.WAITING)

        if self.intent is Intent.PAUSED:
            return AppState(
                Phase.PAUSED, "Protection is paused", _until_text(self.paused_until),
                Severity.WAITING, paused_until=self.paused_until,
            )

        report = self.report
        if report is None or report.at == 0:
            if now - self.started_at < self.startup_grace_s:
                return AppState(Phase.STARTING, "Starting up", "Connecting to the camera", Severity.WAITING)
            return AppState(
                Phase.PROBLEM, "Not protecting", "No status from the camera yet",
                Severity.PROBLEM, "Try turning protection off and on again.",
            )

        if not report.ok:
            first = report.failures[0]
            return AppState(Phase.PROBLEM, "Not protecting", first.detail, Severity.PROBLEM, first.remedy)

        warning = report.warnings[0].detail if report.warnings else ""
        if now < self.alert_until:
            return AppState(Phase.ALERTING, "Cat spotted", "Playing the deterrent now", Severity.GOOD)
        return AppState(Phase.GUARDING, "Protecting", warning or "Looking for cats", Severity.GOOD)


def _until_text(until: float | None) -> str:
    if until is None:
        return "Paused"
    remaining = max(0.0, until - time.time())
    if remaining >= 3600:
        return f"Resumes in {remaining / 3600:.0f} hours"
    if remaining >= 90:
        return f"Resumes in {remaining / 60:.0f} minutes"
    return f"Resumes in {remaining:.0f} seconds"
