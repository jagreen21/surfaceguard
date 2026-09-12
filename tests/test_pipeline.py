"""End-to-end: registration, scan planning, state derivation and the whole engine.

These run against the synthetic room, so they are the regression net that lets
thresholds change on evidence instead of on feeling.
"""

import time

import numpy as np
import pytest

from surfaceguard.audio.player import Player
from surfaceguard.camera.panorama import build_room_map
from surfaceguard.camera.registration import Registrar
from surfaceguard.camera.scan_scheduler import plan_scan
from surfaceguard.camera.sources.synthetic import SyntheticCamera
from surfaceguard.detection.cat_detector import SyntheticDetector
from surfaceguard.engine import Engine
from surfaceguard.geometry.projection import transform_points
from surfaceguard.geometry.surface import Surface
from surfaceguard.health.heartbeat import Metrics, run_checks
from surfaceguard.state import Phase, StateStore
from surfaceguard.storage.preferences import Preferences, surface_from_dict, surface_to_dict

PANS = list(range(-80, 81, 20))
# Counter and table rectangles in the synthetic room's own coordinates.
COUNTER_ROOM = [[170, 432], [680, 432], [690, 500], [160, 500]]
COUNTER_PAN = -65


def test_room_scan_does_not_move_a_camera_without_video():
    from surfaceguard.camera.panorama import StitchError
    from surfaceguard.camera.sources.base import Capabilities

    class ControlOnlyCamera:
        capabilities = Capabilities(name="E30", has_ptz=True, pan_range=(-170, 170))

        def __init__(self):
            self.moves = []

        def read(self, timeout=0):
            return None

        def move_to(self, pan, tilt=0, settle_s=0):
            self.moves.append((pan, tilt))
            return True

    camera = ControlOnlyCamera()
    with pytest.raises(StitchError, match="did not move"):
        build_room_map(camera, settle_s=0)
    assert camera.moves == []


@pytest.fixture(scope="module")
def room_fixture():
    cam = SyntheticCamera(fps=60)
    cam.start()
    room = build_room_map(cam, pan_positions=PANS, settle_s=0)
    reg = Registrar()
    room.install_into(reg)
    yield cam, room, reg
    cam.stop()


def room_to_map(cam, reg, pts, pan):
    cam.move_to(pan)
    frame = cam.read()
    result = reg.register(frame.image, pan=frame.pan)
    assert result.ok, result.reason
    x0, y0 = cam._window_origin()
    return transform_points(result.pose.frame_to_map, np.asarray(pts, float) - [x0, y0])


# ------------------------------------------------------------- registration


def test_registration_succeeds_across_the_whole_pan_range(room_fixture):
    cam, _room, reg = room_fixture
    for pan in (-85, -50, -15, 0, 20, 60, 88):
        cam.move_to(pan)
        result = reg.register(cam.read().image, pan=pan)
        assert result.ok, f"pan {pan}: {result.reason}"
        assert result.inliers >= 30, f"pan {pan}: only {result.inliers} inliers (E2)"


def test_registration_is_inside_the_latency_budget(room_fixture):
    cam, _room, reg = room_fixture
    cam.move_to(0)
    times = [reg.register(cam.read().image, pan=0).elapsed_ms for _ in range(5)]
    assert max(times) <= 15.0, f"registration took {max(times):.1f} ms (budget 15 ms)"


def test_registration_absorbs_backlash(room_fixture):
    """The point of D1: the same requested angle lands on different pixels."""
    _cam, room, reg = room_fixture
    jumpy = SyntheticCamera(fps=60, backlash_px=40.0)
    jumpy.start()
    room.install_into(reg)
    centres = []
    for _ in range(4):
        jumpy.move_to(60)
        jumpy.move_to(-20)
        result = reg.register(jumpy.read().image, pan=-20)
        assert result.ok, result.reason
        w, h = jumpy.read().size
        centres.append(transform_points(result.pose.frame_to_map, [(w / 2, h / 2)])[0])
    jumpy.stop()
    spread = float(np.linalg.norm(np.asarray(centres).std(axis=0)))
    assert spread < 60.0, f"registration did not absorb the backlash ({spread:.1f} px)"


def test_registration_reports_why_it_failed_on_an_unknown_view(room_fixture):
    _cam, _room, reg = room_fixture
    blank = np.full((540, 960, 3), 128, np.uint8)
    result = reg.register(blank)
    assert not result.ok
    assert result.reason, "a failure must carry a reason the user can be shown"


# ----------------------------------------------------------------- planning


def test_scan_plan_prefers_not_to_move_when_everything_fits():
    pan_to_x = lambda p: 1200 + p * 12.0  # noqa: E731
    near = [Surface(f"S{i}", np.array([[x, 400], [x + 100, 400], [x + 100, 470], [x, 470]], float))
            for i, x in enumerate((1100, 1250))]
    centres = {s.id: float(np.mean(s.polygon[:, 0])) for s in near}
    plan = plan_scan(near, centres, 960, pan_to_x, (-95, 95))
    assert plan.single_view and not plan.needs_scanning


def test_scan_plan_splits_the_cycle_by_weight():
    pan_to_x = lambda p: 1200 + p * 12.0  # noqa: E731
    a = Surface("Counter", np.array([[400, 400], [560, 400], [560, 470], [400, 470]], float))
    b = Surface("Shelf", np.array([[1900, 400], [2060, 400], [2060, 470], [1900, 470]], float))
    centres = {s.id: float(np.mean(s.polygon[:, 0])) for s in (a, b)}
    plan = plan_scan([a, b], centres, 700, pan_to_x, (-95, 95), weights={a.id: 7.0, b.id: 3.0})
    cov = plan.coverage()
    assert cov[a.id] > cov[b.id]
    assert cov[a.id] + cov[b.id] == pytest.approx(1.0, abs=0.01)


def test_unreachable_surface_is_reported_as_zero_coverage():
    pan_to_x = lambda p: 1200 + p * 12.0  # noqa: E731
    reachable = Surface("Counter", np.array([[1150, 400], [1250, 400], [1250, 470], [1150, 470]], float))
    far = Surface("Hall", np.array([[9000, 400], [9100, 400], [9100, 470], [9000, 470]], float))
    centres = {s.id: float(np.mean(s.polygon[:, 0])) for s in (reachable, far)}
    plan = plan_scan([reachable, far], centres, 960, pan_to_x, (-95, 95))
    assert far.id in plan.unreachable
    assert plan.coverage().get(far.id, 0.0) == 0.0


# -------------------------------------------------------------------- state


def test_state_never_claims_protection_without_a_passing_heartbeat():
    store = StateStore(has_map=True, has_surfaces=True)
    store.arm()
    store.started_at = time.monotonic() - 60          # past the startup grace
    store.report = None
    assert store.state().phase is Phase.PROBLEM

    stale = Metrics(last_frame_at=time.monotonic() - 30, frames=5, inliers=99, registered=True)
    store.report = run_checks(stale, audio_ok=True)
    state = store.state()
    assert state.phase is Phase.PROBLEM
    assert "No video" in state.detail
    assert state.remedy, "a problem must come with something the user can do"

    healthy = Metrics(last_frame_at=time.monotonic(), frames=5, inliers=99, registered=True)
    store.report = run_checks(healthy, audio_ok=True)
    assert store.state().phase is Phase.GUARDING


def test_coverage_shortfall_warns_without_claiming_failure():
    store = StateStore(has_map=True, has_surfaces=True)
    store.arm()
    m = Metrics(last_frame_at=time.monotonic(), frames=5, inliers=99, registered=True)
    store.report = run_checks(m, audio_ok=True, surfaces_total=3, surfaces_covered=1)
    state = store.state()
    assert state.phase is Phase.GUARDING
    assert "not being watched" in state.detail


def test_pause_expires_back_into_guarding():
    store = StateStore(has_map=True, has_surfaces=True)
    store.arm()
    store.pause_for(-1.0)                              # already elapsed
    assert store.state().phase is not Phase.PAUSED


# ---------------------------------------------------------------- persistence


def test_surface_round_trips_through_json(tmp_path):
    s = Surface("Kitchen counter", np.array([[10, 10], [90, 12], [95, 60], [8, 55]], float))
    s.height_samples = [61.0, 62.5, 60.2]
    s.deterrent.cooldown_s = 45.0
    prefs = Preferences(surfaces=[s])
    prefs.save(tmp_path / "p.json")
    back = Preferences.load(tmp_path / "p.json").surfaces[0]
    assert back.id == s.id and back.name == s.name
    assert back.calibration_samples == 3
    assert back.size_scale == pytest.approx(s.size_scale)
    assert back.deterrent.cooldown_s == 45.0


def test_corrupt_preferences_do_not_stop_the_app_starting(tmp_path):
    bad = tmp_path / "p.json"
    bad.write_text("{ not json")
    assert Preferences.load(bad).surfaces == []


# ------------------------------------------------------------------- engine


def test_engine_fires_on_the_surface_and_not_on_the_floor(room_fixture, tmp_path):
    cam, room, reg = room_fixture
    surface = Surface("Kitchen counter", room_to_map(cam, reg, COUNTER_ROOM, COUNTER_PAN))
    prefs = Preferences(surfaces=[surface], target_fps=30.0, save_thumbnails=False,
                        prefer_camera_speaker=False)
    store = StateStore()
    store.arm()
    engine = Engine(cam, SyntheticDetector(cam, miss_rate=0.0), prefs,
                    room_map=room, player=Player(volume=0.0), state=store)
    engine.policy.enter_s, engine.policy.exit_s = 0.2, 0.6
    fired = []
    engine.on_trigger = lambda d, r: fired.append(d)

    cam.move_to(COUNTER_PAN)
    engine.start()
    try:
        time.sleep(0.5)
        cam.place_cat(430, 600, height=80)            # on the floor in front
        time.sleep(0.8)
        assert not fired, "a cat on the floor must not trigger"

        cam.place_cat(430, 434, height=78)            # up on the counter
        time.sleep(1.2)
        assert fired, "a cat on the counter must trigger"

        before = len(fired)
        time.sleep(1.0)
        assert len(fired) == before, "one visit must not re-fire"

        cam.place_person(380, 600, height=300)
        engine.policy.reset()
        time.sleep(0.8)
        verdict = engine.last_result.verdicts[0]
        assert "no_person" in verdict.blocked_by
    finally:
        engine.stop()

    assert engine.metrics.p95_latency_ms() is not None
    assert engine.metrics.p95_latency_ms() < 1200.0, "E1: p95 must stay under 1200 ms"


def test_engine_flags_a_pet_event_arriving_while_video_is_down(room_fixture, tmp_path):
    """The watchdog use of the camera's own event (D6)."""
    cam, room, _reg = room_fixture
    prefs = Preferences(surfaces=[], save_thumbnails=False)
    engine = Engine(cam, SyntheticDetector(cam), prefs, room_map=room,
                    player=Player(volume=0.0))
    engine.metrics.last_frame_at = time.monotonic() - 60      # video has been dead a while
    cam.inject_pet_event()
    engine._drain_pet_events(None, time.monotonic())
    assert engine.missed_pet_events == 1
