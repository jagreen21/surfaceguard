"""Dwell, hysteresis, schedule and cooldown — the stateful half of D3.

The prototype used a majority vote over a fixed window, which flips symmetrically.
Deterrence wants asymmetry: commit quickly when a cat arrives so the sound lands
inside the association window, and let go slowly so a dropped frame or a turned
head does not end the event and immediately re-trigger it.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field

from ..geometry.surface import Surface
from .gates import Verdict

# Asymmetric hysteresis: 0.5 s to enter, 2.0 s to leave.
ENTER_S = 0.5
EXIT_S = 2.0


class Presence(enum.Enum):
    CLEAR = "clear"
    ARRIVING = "arriving"
    PRESENT = "present"
    LEAVING = "leaving"


@dataclass
class Decision:
    """What the policy wants to happen right now."""

    surface_id: str
    surface_name: str
    fire: bool = False
    presence: Presence = Presence.CLEAR
    reason: str = ""
    verdict: Verdict | None = None


@dataclass
class _SurfaceState:
    presence: Presence = Presence.CLEAR
    since: float = 0.0
    last_fired: float = -1e9
    last_seen_on_surface: float = -1e9
    fired_this_visit: bool = False


@dataclass
class TriggerPolicy:
    """Per-surface presence tracking and firing decisions."""

    enter_s: float = ENTER_S
    exit_s: float = EXIT_S
    states: dict[str, _SurfaceState] = field(default_factory=dict)

    def state_of(self, surface_id: str) -> Presence:
        return self.states.get(surface_id, _SurfaceState()).presence

    def reset(self, surface_id: str | None = None) -> None:
        if surface_id is None:
            self.states.clear()
        else:
            self.states.pop(surface_id, None)

    def update(
        self,
        surface: Surface,
        verdict: Verdict | None,
        now: float | None = None,
        weekday: int | None = None,
        minute_of_day: int | None = None,
    ) -> Decision:
        """Fold one frame's verdict into this surface's presence state."""
        now = time.monotonic() if now is None else now
        st = self.states.setdefault(surface.id, _SurfaceState(since=now))
        on_surface = verdict is not None and verdict.on_surface
        decision = Decision(surface.id, surface.name, presence=st.presence, verdict=verdict)

        if on_surface:
            st.last_seen_on_surface = now
            if st.presence in (Presence.CLEAR, Presence.LEAVING):
                # Coming back during the exit window resumes the same visit, so a
                # dropped frame cannot manufacture a second trigger.
                resumed = st.presence is Presence.LEAVING
                st.presence = Presence.PRESENT if resumed else Presence.ARRIVING
                if not resumed:
                    st.since = now
                    st.fired_this_visit = False
            if st.presence is Presence.ARRIVING and now - st.since >= self.enter_s:
                st.presence = Presence.PRESENT
        else:
            if st.presence is Presence.PRESENT:
                st.presence = Presence.LEAVING
                st.since = now
            elif st.presence is Presence.ARRIVING:
                st.presence = Presence.CLEAR  # never reached dwell
                st.since = now
            elif st.presence is Presence.LEAVING and now - st.last_seen_on_surface >= self.exit_s:
                st.presence = Presence.CLEAR
                st.since = now
                st.fired_this_visit = False

        decision.presence = st.presence
        if st.presence is not Presence.PRESENT:
            decision.reason = f"presence: {st.presence.value}"
            return decision

        if st.fired_this_visit:
            decision.reason = "already deterred during this visit"
            return decision

        if weekday is not None and minute_of_day is not None:
            if not surface.schedule.contains(weekday, minute_of_day):
                decision.reason = "outside this surface's schedule"
                return decision

        remaining = surface.deterrent.cooldown_s - (now - st.last_fired)
        if remaining > 0:
            decision.reason = f"cooling down for another {remaining:.0f} s"
            return decision

        st.last_fired = now
        st.fired_this_visit = True
        decision.fire = True
        decision.reason = "cat on surface, all gates passed"
        return decision
