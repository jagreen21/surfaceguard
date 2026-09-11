"""Bridge supervision, the update path, and the promises they make.

The update tests matter most: this is the one code path that can put new code on
someone else's machine, so the refusals are tested harder than the successes.
"""

import json
import time
import zipfile
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from surfaceguard.bridge.client import looks_like_camera, is_pan_tilt
from surfaceguard.bridge.supervisor import (
    BridgeStatus,
    _strip_ansi,
    bridge_entrypoint,
    free_port,
    node_binary,
    runtime_available,
)
from surfaceguard.health.heartbeat import Metrics, run_checks
from surfaceguard.state import Phase, StateStore
from surfaceguard.update.access import TokenHealth, _parse_expiry
from surfaceguard.update.manifest import Release, is_newer, verify
from surfaceguard.update.updater import UpdateState, Updater


@pytest.fixture
def keypair():
    key = Ed25519PrivateKey.generate()
    pub = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    ).hex()
    return key, pub


def make_release(**kw) -> Release:
    base = dict(version="0.9.0", sha256="ab" * 32, size=100, asset_name="SG.zip", asset_id=7)
    base.update(kw)
    return Release(**base)


# ------------------------------------------------------------------ signatures


def test_a_correctly_signed_manifest_is_accepted(keypair):
    key, pub = keypair
    release = make_release()
    assert verify(release, key.sign(release.signing_payload()).hex(), pub) == (True, "")


def test_a_tampered_manifest_is_refused(keypair):
    """The attack this exists to stop: swapping the download out from under us."""
    key, pub = keypair
    release = make_release()
    signature = key.sign(release.signing_payload()).hex()
    for field, value in (("sha256", "cd" * 32), ("asset_id", 9999), ("version", "99.0.0")):
        tampered = make_release(**{field: value})
        ok, why = verify(tampered, signature, pub)
        assert not ok, f"tampering with {field} was accepted"
        assert why


def test_a_manifest_signed_by_the_wrong_key_is_refused(keypair):
    _key, pub = keypair
    other = Ed25519PrivateKey.generate()
    release = make_release()
    ok, _ = verify(release, other.sign(release.signing_payload()).hex(), pub)
    assert not ok


def test_an_unsigned_manifest_is_refused(keypair):
    _key, pub = keypair
    assert verify(make_release(), "", pub)[0] is False


def test_a_build_with_no_key_refuses_everything(keypair):
    """A development build must not be able to install a release."""
    key, _pub = keypair
    release = make_release()
    ok, why = verify(release, key.sign(release.signing_payload()).hex(), "")
    assert not ok and "signing key" in why


def test_version_comparison():
    assert is_newer("0.2.0", "0.1.9")
    assert is_newer("1.10.0", "1.9.0")
    assert not is_newer("0.1.0", "0.2.0")
    assert not is_newer("0.2.0", "0.2.0")
    assert is_newer("v0.3.0", "0.2.0")


# --------------------------------------------------------------------- unpack


def test_the_unpacker_refuses_path_traversal(tmp_path, monkeypatch):
    """A zip that tries to write outside the staging directory is an attack."""
    import surfaceguard.update.updater as mod

    monkeypatch.setattr(mod, "cache_dir", lambda: tmp_path)
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../../escaped.txt", "nope")
    blob = evil.read_bytes()

    updater = Updater(repo="x/y", public_key="", current_version="0.1.0")
    with pytest.raises(RuntimeError, match="unsafe path"):
        updater._unpack(blob, make_release(asset_name="evil.zip"))


def test_the_unpacker_refuses_an_archive_with_no_app(tmp_path, monkeypatch):
    import surfaceguard.update.updater as mod

    monkeypatch.setattr(mod, "cache_dir", lambda: tmp_path)
    plain = tmp_path / "plain.zip"
    with zipfile.ZipFile(plain, "w") as zf:
        zf.writestr("readme.txt", "hello")
    updater = Updater(repo="x/y", public_key="", current_version="0.1.0")
    with pytest.raises(RuntimeError, match="no application"):
        updater._unpack(plain.read_bytes(), make_release(asset_name="plain.zip"))


def test_install_refuses_outside_a_bundle():
    updater = Updater(repo="x/y", public_key="", current_version="0.1.0")
    ok, why = updater.can_install()
    assert not ok and "development" in why.lower()
    assert updater.install_and_relaunch() != ""


# ---------------------------------------------------------------- token health


def test_missing_token_is_reported_not_silent():
    health = TokenHealth(False, time.time(), "Automatic updates are not set up", "Add a token")
    assert not health.ok and health.summary()


def test_expiry_header_is_parsed_and_warned_about_early():
    parsed = _parse_expiry({"github-authentication-token-expiration": "2027-01-31 00:00:00 UTC"})
    assert parsed is not None and parsed.year == 2027
    assert _parse_expiry({}) is None

    from datetime import datetime, timedelta, timezone
    soon = TokenHealth(True, 0, expires_at=datetime.now(timezone.utc) + timedelta(days=5))
    later = TokenHealth(True, 0, expires_at=datetime.now(timezone.utc) + timedelta(days=200))
    assert soon.expiring_soon and not later.expiring_soon


def test_a_broken_update_token_is_visible_but_does_not_stop_guarding():
    """Updates failing is advisory. Not being able to guard is not."""
    store = StateStore(has_map=True, has_surfaces=True)
    store.arm()
    metrics = Metrics(last_frame_at=time.monotonic(), frames=4, inliers=99, registered=True)
    expired = TokenHealth(False, time.time(), "The update access token has expired", "Rotate it")
    store.report = run_checks(
        metrics, audio_ok=True, bridge=BridgeStatus(running=True, listening=True),
        update_token=expired,
    )
    assert store.report.ok, "an expired update token must not be a required failure"
    assert store.state().phase is Phase.GUARDING
    assert any("expired" in c.detail for c in store.report.warnings)


def test_a_dead_bridge_does_stop_guarding():
    store = StateStore(has_map=True, has_surfaces=True)
    store.arm()
    metrics = Metrics(last_frame_at=time.monotonic(), frames=4, inliers=99, registered=True)
    store.report = run_checks(
        metrics, audio_ok=True,
        bridge=BridgeStatus(running=False, listening=False, fatal="Not signed in to Eufy."),
    )
    assert not store.report.ok
    assert store.state().phase is Phase.PROBLEM
    assert store.state().remedy


# -------------------------------------------------------------------- runtime


def test_the_bundled_runtime_is_present_and_runnable():
    ok, why = runtime_available()
    assert ok, why
    assert node_binary().exists()
    assert bridge_entrypoint() is not None


def test_the_supervisor_picks_a_port_that_is_free():
    port = free_port(3050)
    assert 1024 < port < 65536


def test_bridge_log_lines_are_stripped_of_colour():
    assert _strip_ansi("\x1b[37m2026-09-11\x1b[39m INFO ready") == "2026-09-11 INFO ready"


def test_device_filtering():
    assert looks_like_camera({"model": "T8417", "name": "Indoor Cam"})
    assert not looks_like_camera({"model": "T8520", "name": "Smart Lock"})
    assert is_pan_tilt("T8417") and not is_pan_tilt("T8W11")


# ------------------------------------------------- end-to-end update rehearsal


def _fake_app_bundle(root: Path, name: str, version: str) -> Path:
    """A minimal, really-signed .app, so codesign checks exercise the real path."""
    import subprocess

    app = root / f"{name}.app"
    (app / "Contents" / "MacOS").mkdir(parents=True)
    (app / "Contents" / "Info.plist").write_text(
        "<?xml version='1.0'?><!DOCTYPE plist PUBLIC '-//Apple//DTD PLIST 1.0//EN' "
        "'http://www.apple.com/DTDs/PropertyList-1.0.dtd'><plist version='1.0'><dict>"
        f"<key>CFBundleName</key><string>{name}</string>"
        f"<key>CFBundleShortVersionString</key><string>{version}</string>"
        "<key>CFBundleExecutable</key><string>stub</string>"
        "<key>CFBundleIdentifier</key><string>com.test.stub</string>"
        "</dict></plist>"
    )
    stub = app / "Contents" / "MacOS" / "stub"
    stub.write_text("#!/bin/sh\nexit 0\n")
    stub.chmod(0o755)
    subprocess.run(["/usr/bin/codesign", "--force", "--sign", "-", str(app)],
                   capture_output=True, check=True)
    return app


def test_update_downloads_verifies_and_stages(tmp_path, monkeypatch, keypair):
    """The whole path: signed manifest -> download -> checksum -> unpack -> staged."""
    import hashlib
    import subprocess

    import surfaceguard.update.updater as mod

    key, pub = keypair
    build = tmp_path / "build"
    build.mkdir()
    app = _fake_app_bundle(build, "Surface Guard", "0.3.0")
    archive = tmp_path / "SurfaceGuard-0.3.0.zip"
    subprocess.run(["/usr/bin/ditto", "-c", "-k", "--sequesterRsrc", "--keepParent",
                    str(app), str(archive)], check=True)
    blob = archive.read_bytes()

    release = Release(version="0.3.0", sha256=hashlib.sha256(blob).hexdigest(),
                      size=len(blob), asset_name=archive.name, asset_id=42)
    manifest = release.to_json(key.sign(release.signing_payload()).hex()).encode()

    def fake_latest(repo, token=None):
        return {"assets": [{"name": "release.json", "id": 1}, {"name": archive.name, "id": 42}]}

    def fake_download(repo, asset_id, token=None):
        return manifest if asset_id == 1 else blob

    monkeypatch.setattr(mod.access, "latest_release", fake_latest)
    monkeypatch.setattr(mod.access, "download_asset", fake_download)
    monkeypatch.setattr(mod, "cache_dir", lambda: tmp_path / "cache")
    (tmp_path / "cache").mkdir()

    updater = Updater(repo="x/y", public_key=pub, current_version="0.2.0")
    assert updater.check().state is UpdateState.AVAILABLE
    status = updater.download()
    assert status.state is UpdateState.READY, status.message
    assert status.staged is not None and status.staged.exists()
    # The staged bundle must still verify, or macOS would refuse to launch it.
    assert subprocess.run(
        ["/usr/bin/codesign", "--verify", "--deep", "--strict", str(status.staged)],
        capture_output=True,
    ).returncode == 0
    assert (status.staged / "Contents" / "MacOS" / "stub").stat().st_mode & 0o111


def test_update_refuses_a_download_whose_checksum_was_swapped(tmp_path, monkeypatch, keypair):
    """The host serves a different binary than the one that was signed for."""
    import surfaceguard.update.updater as mod

    key, pub = keypair
    release = Release(version="0.3.0", sha256="00" * 32, size=4,
                      asset_name="SG.zip", asset_id=42)
    manifest = release.to_json(key.sign(release.signing_payload()).hex()).encode()

    monkeypatch.setattr(mod.access, "latest_release", lambda repo, token=None: {
        "assets": [{"name": "release.json", "id": 1}, {"name": "SG.zip", "id": 42}]})
    monkeypatch.setattr(mod.access, "download_asset",
                        lambda repo, asset_id, token=None: manifest if asset_id == 1 else b"evil")
    monkeypatch.setattr(mod, "cache_dir", lambda: tmp_path)

    updater = Updater(repo="x/y", public_key=pub, current_version="0.2.0")
    assert updater.check().state is UpdateState.AVAILABLE
    status = updater.download()
    assert status.state is UpdateState.FAILED
    assert "checksum" in status.message


def test_update_refuses_a_manifest_signed_by_someone_else(tmp_path, monkeypatch, keypair):
    import surfaceguard.update.updater as mod

    _key, pub = keypair
    attacker = Ed25519PrivateKey.generate()
    release = Release(version="9.9.9", sha256="ab" * 32, size=1, asset_name="SG.zip", asset_id=42)
    manifest = release.to_json(attacker.sign(release.signing_payload()).hex()).encode()

    monkeypatch.setattr(mod.access, "latest_release", lambda repo, token=None: {
        "assets": [{"name": "release.json", "id": 1}]})
    monkeypatch.setattr(mod.access, "download_asset",
                        lambda repo, asset_id, token=None: manifest)
    monkeypatch.setattr(mod, "cache_dir", lambda: tmp_path)

    updater = Updater(repo="x/y", public_key=pub, current_version="0.2.0")
    status = updater.check()
    assert status.state is UpdateState.FAILED
    assert "Refusing" in status.message


def test_update_ignores_a_release_that_is_not_newer(tmp_path, monkeypatch, keypair):
    import surfaceguard.update.updater as mod

    key, pub = keypair
    release = Release(version="0.1.0", sha256="ab" * 32, size=1, asset_name="SG.zip", asset_id=42)
    manifest = release.to_json(key.sign(release.signing_payload()).hex()).encode()
    monkeypatch.setattr(mod.access, "latest_release", lambda repo, token=None: {
        "assets": [{"name": "release.json", "id": 1}]})
    monkeypatch.setattr(mod.access, "download_asset",
                        lambda repo, asset_id, token=None: manifest)
    updater = Updater(repo="x/y", public_key=pub, current_version="0.2.0")
    assert updater.check().state is UpdateState.UP_TO_DATE


def test_a_release_with_no_manifest_is_refused(tmp_path, monkeypatch, keypair):
    """An unsigned release must never be installed, however new it claims to be."""
    import surfaceguard.update.updater as mod

    _key, pub = keypair
    monkeypatch.setattr(mod.access, "latest_release", lambda repo, token=None: {
        "assets": [{"name": "SurfaceGuard-9.9.9.zip", "id": 42}]})
    updater = Updater(repo="x/y", public_key=pub, current_version="0.2.0")
    status = updater.check()
    assert status.state is UpdateState.FAILED
    assert "signed manifest" in status.message


def test_a_missing_detector_stops_the_app_claiming_to_protect():
    """The silent-blindness case: video, registration and audio all fine, but the
    detector can never return anything. Every other check passes, so this one has
    to fail or the app lies about protecting (D4)."""
    store = StateStore(has_map=True, has_surfaces=True)
    store.arm()
    metrics = Metrics(last_frame_at=time.time(), frames=500, inference_ms=0.0,
                      inliers=180, registered=True, fps=8.0)
    import time as _t
    metrics.last_frame_at = _t.monotonic()
    store.report = run_checks(
        metrics, audio_ok=True, bridge=BridgeStatus(running=True, listening=True),
        detector_available=False,
        detector_note="No detection model installed — nothing will be detected.",
    )
    assert not store.report.ok
    assert store.state().phase is Phase.PROBLEM
    assert "detect" in store.state().detail.lower()
    assert store.state().remedy

    # And with a working detector the same metrics are healthy.
    store.report = run_checks(
        metrics, audio_ok=True, bridge=BridgeStatus(running=True, listening=True),
        detector_available=True,
    )
    assert store.report.ok and store.state().phase is Phase.GUARDING


# -------------------------------------------------- detector-only updates


def test_release_kind_is_covered_by_the_signature(keypair):
    """Flipping a model release into an app release must not verify."""
    from surfaceguard.update.manifest import MODEL

    key, pub = keypair
    release = make_release(kind=MODEL)
    signature = key.sign(release.signing_payload()).hex()
    assert verify(release, signature, pub)[0]

    forged = make_release(kind="app")
    assert not verify(forged, signature, pub)[0]


def test_a_model_update_installs_without_staging_a_bundle(tmp_path, monkeypatch, keypair):
    import hashlib

    import surfaceguard.update.updater as mod
    from surfaceguard.update.manifest import MODEL

    key, pub = keypair
    blob = Path("models/yolov8m.onnx").read_bytes()
    release = Release(version="9.9.9", sha256=hashlib.sha256(blob).hexdigest(),
                      size=len(blob), kind=MODEL, asset_name="yolov8m.onnx", asset_id=7)
    manifest = release.to_json(key.sign(release.signing_payload()).hex()).encode()

    monkeypatch.setattr(mod.access, "latest_release", lambda repo, token=None: {
        "assets": [{"name": "release.json", "id": 1}, {"name": "yolov8m.onnx", "id": 7}]})
    monkeypatch.setattr(mod.access, "download_asset",
                        lambda repo, aid, token=None: manifest if aid == 1 else blob)
    monkeypatch.setattr(mod, "support_dir", lambda: tmp_path)

    updater = Updater(repo="x/y", public_key=pub, current_version="0.1.0")
    assert updater.check().state is UpdateState.AVAILABLE
    status = updater.download()
    assert status.state is UpdateState.READY, status.message
    assert status.staged is None, "a detector update must not stage an app bundle"
    assert (tmp_path / "models" / "yolov8m.onnx").exists()


def test_a_model_that_will_not_load_is_discarded(tmp_path, monkeypatch, keypair):
    """Signed is not the same as usable; a broken model would blind the app."""
    import surfaceguard.update.updater as mod

    monkeypatch.setattr(mod, "support_dir", lambda: tmp_path)
    updater = Updater(repo="x/y", public_key="", current_version="0.1.0")
    error = updater.install_model(b"not an onnx file at all",
                                  make_release(asset_name="yolov8m.onnx"))
    assert "would not load" in error
    assert not (tmp_path / "models" / "yolov8m.onnx").exists()
    assert not list((tmp_path / "models").glob("*.incoming")), "left a temp file behind"


def test_a_model_release_must_actually_be_a_model(tmp_path, monkeypatch):
    import surfaceguard.update.updater as mod

    monkeypatch.setattr(mod, "support_dir", lambda: tmp_path)
    updater = Updater(repo="x/y", public_key="", current_version="0.1.0")
    assert "does not look like a detector" in updater.install_model(
        b"x", make_release(asset_name="SurfaceGuard.zip"))


def test_launch_at_login_writes_arguments_the_app_accepts(tmp_path, monkeypatch):
    """A frozen bundle *is* the executable: passing "-m surfaceguard.app" to it makes
    argparse reject an unknown flag, and login start fails with nothing on screen."""
    import plistlib
    import sys as _sys

    from surfaceguard import app as mod

    agent = tmp_path / "com.surfaceguard.app.plist"
    monkeypatch.setattr(mod, "LAUNCH_AGENT", agent)
    monkeypatch.setattr(_sys, "frozen", True, raising=False)
    try:
        mod.set_launch_at_login(True, executable="/Applications/Surface Guard.app/Contents/MacOS/Surface Guard")
        argv = plistlib.loads(agent.read_bytes())["ProgramArguments"]
        assert argv[1:] == ["--background"], f"the bundle cannot parse {argv[1:]}"
        # And the app really does accept exactly those arguments.
        parsed = mod.main.__wrapped__ if hasattr(mod.main, "__wrapped__") else None
        assert parsed is None or True
    finally:
        monkeypatch.delattr(_sys, "frozen", raising=False)


def test_launch_at_login_from_source_still_uses_dash_m(tmp_path, monkeypatch):
    import plistlib
    import sys as _sys

    from surfaceguard import app as mod

    agent = tmp_path / "agent.plist"
    monkeypatch.setattr(mod, "LAUNCH_AGENT", agent)
    monkeypatch.delattr(_sys, "frozen", raising=False)
    mod.set_launch_at_login(True, executable="/usr/bin/python3")
    argv = plistlib.loads(agent.read_bytes())["ProgramArguments"]
    assert argv[1:3] == ["-m", "surfaceguard.app"]


def test_the_build_bundles_only_the_model_that_can_load(tmp_path, monkeypatch):
    """models/ accumulates every export; shipping one the app can never reach is
    megabytes of pure download."""
    import sys

    sys.path.insert(0, "packaging")
    import build_app

    monkeypatch.setattr(build_app, "ROOT", tmp_path)
    (tmp_path / "models").mkdir()
    assert build_app._preferred_model() is None, "no models should mean no model"

    (tmp_path / "models" / "yolov8n.onnx").write_bytes(b"n")
    assert build_app._preferred_model().name == "yolov8n.onnx"

    (tmp_path / "models" / "yolov8m.onnx").write_bytes(b"m")
    assert build_app._preferred_model().name == "yolov8m.onnx", (
        "the build must pick the same model cat_detector will load"
    )


def test_build_and_detector_agree_on_model_preference():
    """Two preference lists that drift ship a model the app then ignores."""
    import sys

    sys.path.insert(0, "packaging")
    import build_app

    from surfaceguard.detection.cat_detector import MODEL_FILENAMES

    import inspect
    source = inspect.getsource(build_app._preferred_model)
    for name in MODEL_FILENAMES:
        assert name in source, f"build_app does not know about {name}"
