"""The detection engine: one warm loop, everything else hangs off it.

Ordering matters here and follows Fig 1 of the design doc: locate the frame, then
detect, then gate, then decide. Nothing is tested against a surface until the frame
has a pose, and nothing fires until every required gate has passed.
"""

from __future__ import annotations

import threading
import time

import cv2
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
from .detection import roi
from .detection.motion import MotionGate
from .detection.motion import interval_for as motion_interval
from .detection.gates import Verdict, evaluate, surfaces_for_pose
from .detection.trigger_policy import Decision, Presence, TriggerPolicy
from .geometry.projection import Box, Pose
from .health.heartbeat import Metrics, Report, run_checks
from .state import Intent, StateStore
from .storage.activity_log import REVIEW_WIDTH, ActivityLog
from .storage.preferences import Preferences

logger = get_logger("engine")

# How often the whole frame is scanned for people while cats are found in a crop.
# Four frames at 8 fps is half a second — far less than it takes someone to cross
# a kitchen, and it keeps the expensive full-frame pass off most frames.
WIDE_PASS_EVERY_FRAMES = 4


def _merge_people(near: list, wide: list) -> list:
    """People seen in the crop, plus any from the last full-frame pass.

    Deduplicated by overlap so one person standing inside the crop is not counted
    twice and does not suppress twice.
    """
    out = list(near)
    for candidate in wide:
        if not any(_overlaps(candidate, existing) for existing in out):
            out.append(candidate)
    return out


def _overlaps(a, b, threshold: float = 0.3) -> bool:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return False
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return union > 0 and inter / union >= threshold


def _bound_opencv_threads() -> None:
    """Ask OpenCV to leave a core free for the interface and the camera service.

    Measured, not assumed: the opencv-python wheels for macOS are built with
    "Parallel framework: GCD", and Grand Central Dispatch owns its own pool — this
    call and OPENCV_FOR_THREADS_NUM are both ignored, and getNumThreads() keeps
    reporting every core. It is kept because it does work on builds using the TBB
    or pthreads backends, and it costs nothing where it does not; it is documented
    here so nobody later reads it as a cap that is actually in force.

    Inference threads *are* bounded, in cat_detector._session_options, and that is
    the one that matters: onnxruntime is where the frame time goes.
    """
    try:
        import cv2

        from .detection.cat_detector import worker_threads

        cv2.setNumThreads(worker_threads())
    except Exception:
        pass


_bound_opencv_threads()

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

# How often to mention a cat that produced no decision: often enough to be
# found in the log, rarely enough not to bury it.
OFF_SURFACE_LOG_EVERY_S = 30.0
STRIP_FRAMES = 5


@dataclass
class FrameResult:
    """Everything one pass produced — the UI's live view renders straight from this."""

    frame: Frame
    pose: Pose | None
    inference_skipped: bool = False
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
        self._frames_since_wide = 0
        self._people_wide: list = []
        self.motion = MotionGate()
        self._last_cats: list = []
        self._last_off_surface_log = 0.0
        self._last_people: list = []

        self.on_frame: Callable[[FrameResult], None] | None = None
        self.on_trigger: Callable[[Decision, FrameResult], None] | None = None
        self.on_heartbeat: Callable[[Report], None] | None = None

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._pause_requested = threading.Event()
        self._paused = threading.Event()
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

    def start(self, source_already_started: bool = False) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._pause_requested.clear()
        self._paused.clear()
        if not source_already_started:
            self.source.start()
        self._thread = threading.Thread(target=self._run, name="sg-engine", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        self._pause_requested.clear()
        if self._thread:
            self._thread.join(timeout=timeout)
        self.source.stop()

    def pause_processing(self, timeout: float = 3.0) -> bool:
        """Let setup temporarily own the already-running camera stream."""
        if not self.running:
            return False
        self._pause_requested.set()
        return self._paused.wait(timeout)

    def resume_processing(self) -> None:
        self._pause_requested.clear()

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ------------------------------------------------------------- the loop

    def _run(self) -> None:
        # Slow when the room is still, faster than nominal when something moves:
        # time-to-sound is what the deterrent depends on, and an empty kitchen
        # does not need eight frames a second.
        interval = motion_interval(self.motion.state, self.prefs.target_fps)
        while not self._stop.is_set():
            if self._pause_requested.is_set():
                self._paused.set()
                self._stop.wait(0.05)
                continue
            self._paused.clear()
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
        self._paused.clear()

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
        # Cats are looked for in a crop around the surfaces, so the same pixels
        # reach the network several times larger. People are looked for on the
        # whole frame, less often: someone walking up to a counter starts outside
        # it, and the no_person gate is what stops the sound going off mid-cook.
        t0 = time.perf_counter()
        frame_pose = result.pose
        region = (
            roi.for_surfaces(
                surfaces_for_pose(self.prefs.surfaces, frame_pose), frame_pose
            )
            if reg.ok and frame_pose is not None
            else roi.full_frame((frame.image.shape[1], frame.image.shape[0]))
        )
        motion = self.motion.consider(reg.work, region, now)
        if motion.skipped:
            # Nothing changed inside the surfaces, so the previous frame's
            # detections still describe the scene — including a cat sitting
            # perfectly still on the counter. Carrying them forward keeps dwell
            # and hysteresis advancing correctly; dropping them would look like
            # the cat left, and the gate forces a real look every
            # MAX_CONSECUTIVE_SKIPS frames so this cannot drift indefinitely.
            cats, people = self._last_cats, self._last_people
            result.inference_skipped = True
        else:
            boxes = [
                region.to_frame(b)
                for b in self.detector.detect(region.crop(frame.image), (region.x1, region.y1))
            ]
            cats, people = Detector.split(boxes)
            self.motion.note_inference(now)

        self._frames_since_wide += 1
        wide_due = (
            not motion.skipped
            and
            not region.full_frame
            and self._frames_since_wide >= WIDE_PASS_EVERY_FRAMES
        )
        if wide_due:
            self._frames_since_wide = 0
            wide = self.detector.detect(frame.image)
            _, self._people_wide = Detector.split(wide)
        if not region.full_frame:
            # People from the last wide pass persist between them; a person does
            # not leave the kitchen in the time it takes to run four frames.
            people = _merge_people(people, self._people_wide)
        else:
            self._people_wide = people

        result.inference_ms = (time.perf_counter() - t0) * 1e3
        self.metrics.inference_ms = result.inference_ms
        self.metrics.roi_magnification = region.magnification
        result.cats, result.people = cats, people
        self._last_cats, self._last_people = cats, people
        self.metrics.motion_skip_rate = motion.skip_rate
        self.metrics.detector_idle = motion.stale(now)

        if reg.ok and self.state.intent is Intent.ARMED:
            self._judge(result, now)
        elif result.cats and self.state.intent is Intent.ARMED:
            # Detection worked and judging never happened, because the frame could
            # not be placed in the room map. Nothing downstream runs, so without
            # this the app is completely silent about a cat it can plainly see.
            self._note_unjudged(
                now, f"seen, but the camera view could not be located ({reg.reason})"
            )
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
            elif result.cats and best is not None:
                # A cat in view, judged, and not on this surface. Not an event, but
                # not silent either: a cat correctly identified with nothing at all
                # appearing in Activity is what a polygon in the wrong place looks
                # like, and the paw point is the bottom-centre of the box, so a cat
                # whose body covers the surface can still be standing outside it.
                self._note_unjudged(now, "seen, but not standing on a surface")

    def _play_deterrent(self, sound: str, volume: float, delay: float) -> tuple[bool, str]:
        """Play the cue, waiting out any configured delay off the engine thread.

        With no delay this stays synchronous, so the common case still reports
        truthfully whether the sound actually played.
        """
        def play() -> tuple[bool, str]:
            if self.prefs.prefer_camera_speaker and self.source.capabilities.has_speaker:
                if self.source.play_sound_on_camera(sound):
                    return True, "camera speaker"
            outcome = self.player.play(sound, volume)
            return outcome.ok, outcome.backend

        if delay <= 0:
            return play()

        def wait_then_play() -> None:
            if self._stop.wait(delay):
                return          # the engine stopped during the delay
            try:
                play()
            except Exception:
                logger.exception("deterrent failed after its delay")

        threading.Thread(target=wait_then_play, name="sg-deterrent", daemon=True).start()
        # Reported as played: the decision is made and the sound is committed.
        return True, "scheduled"

    def _fire(
        self,
        surface,
        decision: Decision,
        result: FrameResult,
        occupancy: int = 1,
        extra: list | None = None,
    ) -> None:
        sound = surface.deterrent.sound
        if surface.deterrent.vary_sound:
            sound = self._vary(sound)

        # Latency is measured to the *decision*, before any configured delay.
        # Sleeping here used to block the engine thread for up to five seconds —
        # no frames read, no registration, no strip buffer filled, and the cat
        # unwatched for the whole of it — and then the latency it recorded
        # included the sleep, so a 2 s delay looked like 2 s of lag. The delay is
        # a deliberate wait, not a cost; §7's budget is about how fast the app
        # decides, and that is what this now reports.
        origin = result.frame.ts_capture or result.frame.ts_received
        latency_ms = (time.monotonic() - origin) * 1e3
        self.metrics.note_latency(latency_ms)

        delay = min(max(surface.deterrent.delay_s, 0.0), 5.0)
        played, via = self._play_deterrent(sound, surface.deterrent.volume, delay)

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

    def _note_unjudged(self, now: float, why: str) -> None:
        """A cat was detected and no decision came of it. Never let that be silent."""
        self.metrics.cats_off_surface += 1
        self.metrics.last_cat_off_surface_at = now
        self.metrics.last_unjudged_reason = why
        if now - self._last_off_surface_log >= OFF_SURFACE_LOG_EVERY_S:
            self._last_off_surface_log = now
            logger.info("cat %s (%d times so far)", why, self.metrics.cats_off_surface)
            # And put it in Activity. An empty tab while a cat is plainly being
            # seen tells her nothing; one entry every thirty seconds explains it
            # without burying the events that matter.
            try:
                self.log.add_note(f"Saw a cat — {why}")
            except Exception:
                logger.debug("could not record the note", exc_info=True)

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
        """Keep a small copy of every frame for the review strip.

        Stored at review width, not source resolution. Sixteen 1080p copies is
        about 100 MB resident, held permanently, to feed a strip that is never
        displayed wider than REVIEW_WIDTH — so the downscale happens once here
        instead of once per frame at display time, and costs a twentieth of the
        memory. Measured: 528 MB RSS before, and the strip looks identical
        because it was always being resized to this width anyway.
        """
        image = frame.image
        if image is None or image.size == 0:
            return
        h, w = image.shape[:2]
        if w > REVIEW_WIDTH:
            scale = REVIEW_WIDTH / float(w)
            image = cv2.resize(
                image, (REVIEW_WIDTH, max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            image = image.copy()
        self._recent_frames.append(image)

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
