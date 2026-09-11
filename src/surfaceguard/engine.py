"""The detection engine: one warm loop, everything else hangs off it.

Ordering matters here and follows Fig 1 of the design doc: locate the frame, then
detect, then gate, then decide. Nothing is tested against a surface until the frame
has a pose, and nothing fires until every required gate has passed.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .audio.player import Player
from .camera.panorama import RoomMap
from .camera.registration import MIN_INLIERS, Registrar
from .camera.scan_scheduler import Plan, ScanRunner, plan_scan
from .camera.sources.base import CameraSource, Frame
from .detection.cat_detector import Detector
from .detection.gates import Verdict, evaluate, surfaces_for_pose
from .detection.trigger_policy import Decision, Presence, TriggerPolicy
from .geometry.projection import Box, Pose
from .health.heartbeat import Metrics, Report, run_checks
from .state import Intent, StateStore
from .storage.activity_log import ActivityLog
from .storage.preferences import Preferences

HEARTBEAT_EVERY_S = 20.0
SCAN_REPLAN_EVERY_S = 30.0


@dataclass
class FrameResult:
    """Everything one pass produced — the UI's live view renders straight from this."""

    frame: Frame
    pose: Pose | None
    cats: list[Box] = field(default_factory=list)
    people: list[Box] = field(default_factory=list)
    verdicts: list[Verdict] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    registration_ms: float = 0.0
    inference_ms: float = 0.0
    registration_reason: str = ""


class Engine:
    """Owns the loop. Thread-safe to start, stop and query; callbacks fire on its thread."""

    def __init__(
        self,
        source: CameraSource,
        detector: Detector,
        prefs: Preferences,
        room_map: RoomMap | None = None,
        player: Player | None = None,
        log: ActivityLog | None = None,
        state: StateStore | None = None,
        min_inliers: int = MIN_INLIERS,
    ) -> None:
        self.source = source
        self.detector = detector
        self.prefs = prefs
        self.player = player or Player(volume=prefs.master_volume)
        self.log = log
        self.state = state or StateStore()
        self.registrar = Registrar(min_inliers=min_inliers)
        self.policy = TriggerPolicy()
        self.metrics = Metrics()
        self.scan = ScanRunner()
        self.room_map: RoomMap | None = None
        self.last_result: FrameResult | None = None
        self.last_report: Report | None = None

        self.on_frame: Callable[[FrameResult], None] | None = None
        self.on_trigger: Callable[[Decision, FrameResult], None] | None = None
        self.on_heartbeat: Callable[[Report], None] | None = None

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._last_heartbeat = 0.0
        self._last_replan = 0.0
        self._missed_pet_events = 0

        if room_map is not None:
            self.set_room_map(room_map)
        self.state.has_surfaces = bool(prefs.surfaces)

    # ----------------------------------------------------------------- wiring

    def set_room_map(self, room_map: RoomMap) -> None:
        with self._lock:
            self.room_map = room_map
            room_map.install_into(self.registrar)
            self.state.has_map = True
            self._last_replan = 0.0

    def set_surfaces(self, surfaces: list) -> None:
        with self._lock:
            self.prefs.surfaces = surfaces
            self.state.has_surfaces = bool(surfaces)
            self.policy.reset()
            self._last_replan = 0.0

    # -------------------------------------------------------------- lifecycle

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.source.start()
        self._thread = threading.Thread(target=self._run, name="sg-engine", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        self.source.stop()

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ------------------------------------------------------------- the loop

    def _run(self) -> None:
        interval = 1.0 / max(1.0, self.prefs.target_fps)
        while not self._stop.is_set():
            cycle_started = time.monotonic()
            try:
                self._tick()
            except Exception as exc:  # a bad frame must not kill the guard
                self.metrics.registered = False
                self.last_report = None
                print(f"[engine] frame failed: {exc}")
            slack = interval - (time.monotonic() - cycle_started)
            if slack > 0:
                self._stop.wait(slack)

    def _tick(self) -> None:
        frame = self.source.read(timeout=2.0)
        now = time.monotonic()
        self._drain_pet_events(frame, now)

        if frame is None:
            self._maybe_heartbeat(now)
            return

        self.metrics.last_frame_at = frame.ts_received
        self.metrics.frames += 1
        if self.metrics.fps == 0.0:
            self.metrics.fps = self.prefs.target_fps
        result = FrameResult(frame=frame, pose=None)

        # --- locate the frame -------------------------------------------
        reg = self.registrar.register(frame.image, pan=frame.pan, tilt=frame.tilt)
        result.pose = reg.pose
        result.registration_ms = reg.elapsed_ms
        result.registration_reason = reg.reason
        self.metrics.registration_ms = reg.elapsed_ms
        self.metrics.inliers = reg.inliers
        self.metrics.registered = reg.ok

        # --- detect ------------------------------------------------------
        t0 = time.perf_counter()
        boxes = self.detector.detect(frame.image)
        result.inference_ms = (time.perf_counter() - t0) * 1e3
        self.metrics.inference_ms = result.inference_ms
        result.cats, result.people = Detector.split(boxes)

        if reg.ok and self.state.intent is Intent.ARMED:
            self._judge(result, now)
        elif reg.ok:
            # Not armed: keep presence state from going stale so arming mid-visit
            # does not immediately fire on a cat that was already there.
            self.policy.reset()

        self._maybe_scan(now)
        self._maybe_heartbeat(now)
        with self._lock:
            self.last_result = result
        if self.on_frame:
            self.on_frame(result)

    # ---------------------------------------------------------------- judging

    def _judge(self, result: FrameResult, now: float) -> None:
        pose = result.pose
        assert pose is not None
        lt = time.localtime()
        weekday, minute = lt.tm_wday, lt.tm_hour * 60 + lt.tm_min

        for surface in surfaces_for_pose(self.prefs.surfaces, pose):
            best: Verdict | None = None
            for cat in result.cats:
                verdict = evaluate(surface, pose, cat, result.people)
                if best is None or (verdict.on_surface and not best.on_surface):
                    best = verdict
                if verdict.on_surface:
                    break
            if best is not None:
                result.verdicts.append(best)

            decision = self.policy.update(surface, best, now, weekday, minute)
            result.decisions.append(decision)

            if decision.fire:
                self._fire(surface, decision, result)
            elif best is not None and not best.on_surface and best.blocked_by != ["inside"]:
                # Near-misses are worth logging, but "the cat was somewhere else"
                # is not a near-miss and would bury the log.
                self._record(surface, decision, result, fired=False)

    def _fire(self, surface, decision: Decision, result: FrameResult) -> None:
        delay = surface.deterrent.delay_s
        if delay > 0:
            time.sleep(min(delay, 5.0))

        sound = surface.deterrent.sound
        if surface.deterrent.vary_sound:
            sound = self._vary(sound)

        played, via = False, "none"
        if self.prefs.prefer_camera_speaker and self.source.capabilities.has_speaker:
            played = self.source.play_sound_on_camera(sound)
            via = "camera speaker" if played else via
        if not played:
            outcome = self.player.play(sound, surface.deterrent.volume)
            played, via = outcome.ok, outcome.backend

        # Latency measured from the camera's own capture clock where it has one,
        # which is the number §7 budgets and E1 accepts on.
        origin = result.frame.ts_capture or result.frame.ts_received
        latency_ms = (time.monotonic() - origin) * 1e3
        self.metrics.note_latency(latency_ms)

        if played and decision.verdict is not None and decision.verdict.box is not None:
            # Only confirmed on-surface detections feed the plane's scale model.
            surface.observe_height(result.pose, decision.verdict.box)

        self.state.note_alert()
        self._record(surface, decision, result, fired=played, latency_ms=latency_ms, via=via)
        if self.on_trigger:
            self.on_trigger(decision, result)

    def _vary(self, preferred: str) -> str:
        """Rotate the cue so the cat does not learn one specific sound (D7)."""
        options = [s for s in self.player.sounds() if s != preferred]
        if not options:
            return preferred
        return preferred if np.random.random() < 0.5 else str(np.random.choice(options))

    def _record(
        self,
        surface,
        decision: Decision,
        result: FrameResult,
        fired: bool,
        latency_ms: float | None = None,
        via: str = "",
    ) -> None:
        if self.log is None:
            return
        verdict = decision.verdict
        gates = [
            {"name": g.name, "status": g.status.value, "detail": g.detail, "required": g.required}
            for g in (verdict.gates if verdict else [])
        ]
        box = verdict.box if verdict else None
        reason = decision.reason if not fired else f"deterrent played via {via}"
        self.log.record(
            surface.id, surface.name, fired, reason, gates,
            score=box.score if box else None,
            box=(box.x1, box.y1, box.x2, box.y2) if box else None,
            latency_ms=latency_ms,
            frame=result.frame.image if fired else None,
            save_thumbnail=self.prefs.save_thumbnails,
        )

    # ------------------------------------------------------------------ scan

    def _maybe_scan(self, now: float) -> None:
        caps = self.source.capabilities
        if not caps.has_ptz or self.room_map is None:
            return
        if now - self._last_replan > SCAN_REPLAN_EVERY_S:
            self._last_replan = now
            self.scan.set_plan(self.build_plan())
        stop = self.scan.due(now)
        if stop is not None:
            self.source.move_to(stop.pan, settle_s=0.0)

    def build_plan(self) -> Plan:
        """Current scan plan, also used by the UI to show honest coverage."""
        if self.room_map is None:
            return Plan()
        caps = self.source.capabilities
        frame_w = self.last_result.frame.size[0] if self.last_result else 960
        centres = {
            s.id: float(np.mean(np.asarray(s.polygon)[:, 0])) for s in self.prefs.surfaces
        }
        return plan_scan(
            self.prefs.surfaces, centres, float(frame_w), self.room_map.pan_to_x,
            caps.pan_range or (-40.0, 40.0), self.prefs.scan_weights,
        )

    # ------------------------------------------------------------- heartbeat

    def _maybe_heartbeat(self, now: float) -> None:
        if now - self._last_heartbeat < HEARTBEAT_EVERY_S:
            return
        self._last_heartbeat = now
        audio = self.player.self_test()
        plan = self.build_plan()
        coverage = plan.coverage()
        enabled = [s for s in self.prefs.surfaces if s.enabled]
        covered = sum(1 for s in enabled if coverage.get(s.id, 0.0) > 0.0)
        report = run_checks(
            self.metrics,
            audio_ok=audio.ok,
            audio_detail=audio.error,
            min_inliers=self.registrar.min_inliers,
            surfaces_total=len(enabled),
            surfaces_covered=covered,
            now=now,
        )
        self.last_report = report
        self.state.report = report
        if self.on_heartbeat:
            self.on_heartbeat(report)

    def _drain_pet_events(self, frame: Frame | None, now: float) -> None:
        """The camera's own pet event is a watchdog, never a trigger (D6, §8)."""
        while True:
            event = self.source.next_pet_event()
            if event is None:
                return
            stale = frame is None or (now - self.metrics.last_frame_at) > 6.0
            if stale:
                self._missed_pet_events += 1
                if self.log is not None:
                    self.log.record(
                        "-", "Camera", False,
                        "Your camera saw a pet but Surface Guard was not receiving video",
                    )

    @property
    def missed_pet_events(self) -> int:
        return self._missed_pet_events
