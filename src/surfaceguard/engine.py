"""The detection engine: one warm loop, everything else hangs off it.

Ordering matters here and follows Fig 1 of the design doc: locate the frame, then
detect, then gate, then decide. Nothing is tested against a surface until the frame
has a pose, and nothing fires until every required gate has passed.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .audio.player import Player
from .logging_setup import get as get_logger
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

logger = get_logger("engine")

HEARTBEAT_EVERY_S = 20.0
SCAN_REPLAN_EVERY_S = 30.0

# --- evidence for the weekly review -------------------------------------------
# Near-misses are logged every frame a cat is near a surface, so keeping a picture
# for each one would write hundreds of near-identical stills during a single
# visit. One picture per surface per window is plenty: the review clusters
# repeats into a single question anyway, and it only needs one good look.
CANDIDATE_PICTURE_EVERY_S = 30.0

# Which blocked gates are worth photographing. "visible" is deliberately absent:
# a surface half out of frame is a real problem, but not one a label can fix, and
# the review would rather spend a card on something the user's answer can change.
REVIEWABLE_GATES = frozenset({"scale", "known_false_alarm", "confidence", "no_person"})

# Frames kept in the ring buffer, and how many of them a strip uses. At the
# default 8 fps this reaches about two seconds back, which is enough motion to
# settle most of the calls the review picks precisely because they are hard.
STRIP_BUFFER = 16
STRIP_FRAMES = 5


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
        # Set by the app when a supervised bridge and an updater exist, so their
        # health reaches the same derived state everything else does (D4).
        self.bridge = None
        self.updater = None

        self.on_frame: Callable[[FrameResult], None] | None = None
        self.on_trigger: Callable[[Decision, FrameResult], None] | None = None
        self.on_heartbeat: Callable[[Report], None] | None = None

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._last_heartbeat = 0.0
        self._last_replan = 0.0
        self._missed_pet_events = 0
        self._recent_frames: deque = deque(maxlen=STRIP_BUFFER)
        self._last_candidate_picture: dict[str, float] = {}

        if room_map is not None:
            self.set_room_map(room_map)
        self.state.has_surfaces = bool(prefs.surfaces)

    # ----------------------------------------------------------------- wiring

    def set_room_map(self, room_map: RoomMap) -> None:
        with self._lock:
            self.room_map = room_map
            room_map.install_into(self.registrar)
            self._prefer_surface_keyframes()
            self.state.has_map = True
            self._last_replan = 0.0

    def set_surfaces(self, surfaces: list) -> None:
        with self._lock:
            self.prefs.surfaces = surfaces
            self.state.has_surfaces = bool(surfaces)
            self.policy.reset()
            self._prefer_surface_keyframes()
            self._last_replan = 0.0

    def _prefer_surface_keyframes(self) -> None:
        """Anchor angleless Eufy registration to tiles with protected surfaces."""
        if self.room_map is None:
            self.registrar.prefer_keyframes([])
            return
        keyframe_ids = []
        for surface in self.prefs.surfaces:
            if not surface.enabled or len(surface.polygon) == 0:
                continue
            centre = np.asarray(surface.polygon, float).mean(axis=0)
            key_id = self.room_map.keyframe_at((float(centre[0]), float(centre[1])))
            if key_id is not None:
                keyframe_ids.append(key_id)
        self.registrar.prefer_keyframes(keyframe_ids)

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
                logger.exception("frame failed: %s", exc)
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
        self._remember_frame(frame)
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
            # Every cat is judged, not just the first one that qualifies. A
            # household has more than one, and stopping at the first match logs a
            # two-cat incident as one event, shows the review a single card for it,
            # and feeds only one of them to the size model.
            best: Verdict | None = None
            on_surface: list[Verdict] = []
            for cat in result.cats:
                verdict = evaluate(surface, pose, cat, result.people, minute)
                if verdict.on_surface:
                    on_surface.append(verdict)
                if best is None or (verdict.on_surface and not best.on_surface):
                    best = verdict
            if best is not None:
                result.verdicts.append(best)
            occupancy = len(on_surface)
            if occupancy > 1:
                logger.info("%d cats on %s at once", occupancy, surface.name)

            decision = self.policy.update(surface, best, now, weekday, minute)
            result.decisions.append(decision)

            if decision.fire:
                self._fire(surface, decision, result, occupancy=occupancy,
                           extra=on_surface[1:])
            elif best is not None and not best.on_surface and best.blocked_by != ["inside"]:
                # Near-misses are worth logging, but "the cat was somewhere else"
                # is not a near-miss and would bury the log.
                self._record(surface, decision, result, fired=False, now=now)

    def _fire(
        self,
        surface,
        decision: Decision,
        result: FrameResult,
        occupancy: int = 1,
        extra: list | None = None,
    ) -> None:
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

        if played:
            # Every cat that was on the surface feeds the size model, so a
            # household of differing sizes widens the band instead of one of them
            # permanently looking wrong.
            seen = [decision.verdict] if decision.verdict is not None else []
            seen += list(extra or [])
            for verdict in seen:
                if verdict is not None and verdict.box is not None and verdict.on_surface:
                    surface.observe_height(result.pose, verdict.box)

        self.state.note_alert()
        self._record(surface, decision, result, fired=played, latency_ms=latency_ms,
                     via=via, now=time.monotonic())
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
        now: float | None = None,
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

        # Where this happened in the room, not in this frame. It is what lets the
        # review ask "this same spot again?" after the camera has panned away and
        # come back, and what a blind spot is anchored to (D1).
        map_point = None
        if box is not None and result.pose is not None:
            try:
                map_point = surface.paw_in_map(result.pose, box.paw_point)
            except Exception:  # a degenerate pose must not lose the event itself
                map_point = None

        keep = self._wants_picture(surface, verdict, fired, now)
        allowed = self.prefs.save_thumbnails or self.prefs.review_pictures_only
        self.log.record(
            surface.id, surface.name, fired, reason, gates,
            score=box.score if box else None,
            box=(box.x1, box.y1, box.x2, box.y2) if box else None,
            latency_ms=latency_ms,
            frame=result.frame.image if keep else None,
            save_thumbnail=allowed,
            map_point=map_point,
            strip_frames=self._strip_frames() if keep else None,
        )

    def _wants_picture(self, surface, verdict, fired: bool, now: float | None) -> bool:
        """Is this event one the weekly review might want to show?

        Everything that played a sound, plus a rate-limited sample of the silent
        calls that a label could actually change.
        """
        if fired:
            return True
        if verdict is None:
            return False
        interesting = REVIEWABLE_GATES.intersection(verdict.blocked_by) or verdict.degraded
        if not interesting:
            return False
        now = time.monotonic() if now is None else now
        last = self._last_candidate_picture.get(surface.id, -1e9)
        if now - last < CANDIDATE_PICTURE_EVERY_S:
            return False
        self._last_candidate_picture[surface.id] = now
        return True

    def _remember_frame(self, frame: Frame) -> None:
        image = frame.image
        if image is None or image.size == 0:
            return
        self._recent_frames.append(image.copy())

    def _strip_frames(self) -> list:
        """A handful of frames spread across the buffer, ending at the newest."""
        buffered = list(self._recent_frames)
        if not buffered:
            return []
        if len(buffered) <= STRIP_FRAMES:
            return buffered
        step = (len(buffered) - 1) / (STRIP_FRAMES - 1)
        return [buffered[round(i * step)] for i in range(STRIP_FRAMES)]

    # ------------------------------------------------------------------ scan

    def _maybe_scan(self, now: float) -> None:
        caps = self.source.capabilities
        if not caps.has_ptz or self.room_map is None:
            return
        # Eufy's tracker already centres the live view on motion. Issuing our own
        # dead-reckoned patrol moves at the same time makes both controllers wrong;
        # registration can locate the tracked picture against every map keyframe.
        if caps.auto_tracks_motion:
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
        info = self.detector.info
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
            detector_available=info.available,
            detector_note=info.note,
            bridge=self.bridge.status if self.bridge is not None else None,
            update_token=self.updater.status.token if self.updater is not None else None,
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
