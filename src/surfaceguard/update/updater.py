"""Self-update: check, verify, swap, relaunch.

Nothing is installed that was not signed by the build machine's Ed25519 key
(:mod:`surfaceguard.update.manifest`). The bundle is swapped whole rather than
patched in place, so the replacement keeps the signature it was built with and
macOS will still launch it.

Updates install themselves. The alternative — a button in Settings — meant that in
practice she would stay on whatever version was handed to her, because nobody opens
Settings to look for work. The cost is a restart of a few seconds at an arbitrary
moment, during which nothing is watched; that is a real cost, and it is smaller
than running a version with a known fault in it for months.

Three things keep that from becoming its own fault: a candidate that fails
verification is never installed, a version that fails to start twice is rolled
back, and every install is logged and announced afterwards so a restart is never
unexplained.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from ..logging_setup import get as get_logger
from ..storage.preferences import support_dir
from . import access, build_info
from .manifest import MODEL, Release, is_newer, verify

logger = get_logger("update")

MANIFEST_ASSET = "release.json"
CHECK_EVERY_S = 6 * 3600

# A build that cannot start is worse than an old one. If the version we installed
# is still not running after this many attempts, put the previous bundle back.
MAX_START_ATTEMPTS = 2


class UpdateState(Enum):
    IDLE = "idle"
    CHECKING = "checking"
    UP_TO_DATE = "up_to_date"
    AVAILABLE = "available"
    DOWNLOADING = "downloading"
    READY = "ready"
    FAILED = "failed"


@dataclass
class UpdateStatus:
    state: UpdateState = UpdateState.IDLE
    message: str = "Not checked yet"
    release: Release | None = None
    staged: Path | None = None
    last_checked: float = 0.0
    token: access.TokenHealth = field(default_factory=access.TokenHealth)

    @property
    def actionable(self) -> bool:
        return self.state in (UpdateState.AVAILABLE, UpdateState.READY)


def cache_dir() -> Path:
    d = Path.home() / "Library" / "Caches" / "SurfaceGuard" / "updates"
    d.mkdir(parents=True, exist_ok=True)
    return d


def pending_note_path() -> Path:
    return support_dir() / "updated-to.txt"


def note_update(version: str) -> None:
    """Leave a note the next start can turn into 'updated to X'."""
    try:
        pending_note_path().write_text(version)
    except OSError:
        pass


def take_update_note() -> str:
    """Read and clear the note, so a restart is never unexplained."""
    path = pending_note_path()
    try:
        version = path.read_text().strip()
        path.unlink(missing_ok=True)
        return version
    except OSError:
        return ""


def cleanup_previous() -> None:
    """Remove the bundle left behind by a previous update. Safe to call at startup.

    Reaching this point means the new version started, so the old one is no longer
    needed as a way back.
    """
    current = build_info.bundle_path()
    if current is None:
        return
    old = current.parent / (current.name + ".old")
    if old.exists():
        logger.info("previous version cleaned up after a successful start")
        shutil.rmtree(old, ignore_errors=True)


class Updater:
    """Owns the update lifecycle and the daily token-health check."""

    def __init__(
        self,
        repo: str | None = None,
        public_key: str | None = None,
        current_version: str | None = None,
    ) -> None:
        self.repo = repo or build_info.env_override("UPDATE_REPO", build_info.UPDATE_REPO)
        self.public_key = public_key if public_key is not None else build_info.UPDATE_PUBLIC_KEY
        self.current_version = current_version or build_info.VERSION
        self.status = UpdateStatus()
        # Installing without being asked is the whole point; a caller can turn it
        # off, but nothing in the app does.
        self.auto_install = True
        # Called with the new version just before the process is replaced, so the
        # app can stop cleanly and leave a note to show after the restart.
        self.on_before_install = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_token_check = 0.0

    # ------------------------------------------------------------- background

    def start(self) -> None:
        """Poll for updates, and check the access token once a day."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="sg-updater", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None

    def _loop(self) -> None:
        # A short delay so startup is not competing with the camera connecting.
        self._stop.wait(45.0)
        while not self._stop.is_set():
            try:
                self.refresh_token_health()
                self.check()
                if self.auto_install and self.status.state is UpdateState.AVAILABLE:
                    self._auto()
            except Exception as exc:
                logger.exception("update cycle failed: %s", exc)
                self._fail(f"Could not check for updates: {exc}")
            self._stop.wait(CHECK_EVERY_S)

    def _auto(self) -> None:
        """Download, verify and install without anyone being asked."""
        release = self.status.release
        version = release.version if release else "?"
        ok, why = self.can_install()
        if not ok:
            logger.warning("update %s is ready but cannot be installed: %s", version, why)
            return
        logger.info("installing update %s automatically", version)
        if self.download().state is not UpdateState.READY:
            logger.error("update %s did not download: %s", version, self.status.message)
            return
        if self.on_before_install is not None:
            try:
                self.on_before_install(version)
            except Exception:
                logger.exception("pre-install hook failed; installing anyway")
        error = self.install_and_relaunch()
        if error:
            logger.error("update %s could not be installed: %s", version, error)
            self._fail(error)

    def refresh_token_health(self, force: bool = False) -> access.TokenHealth:
        """Daily: is the update token still valid, and how long until it expires?

        Without this, a token that quietly expires stops updates with no symptom at
        all — the app would simply never hear about a new version again.
        """
        now = time.time()
        if not force and now - self._last_token_check < access.CHECK_EVERY_S:
            return self.status.token
        self._last_token_check = now
        health = access.check_token(self.repo)
        self.status.token = health
        return health

    # ------------------------------------------------------------------ check

    def check(self) -> UpdateStatus:
        with self._lock:
            self.status.state = UpdateState.CHECKING
            self.status.message = "Checking for updates…"
        try:
            release_json = access.latest_release(self.repo)
        except Exception as exc:
            return self._fail(str(exc))
        self.status.last_checked = time.time()

        if not release_json:
            return self._set(UpdateState.UP_TO_DATE, "No updates published yet")

        manifest_asset = access.find_asset(release_json, MANIFEST_ASSET)
        if manifest_asset is None:
            return self._fail("The latest release has no signed manifest; refusing it")

        try:
            raw = access.download_asset(self.repo, int(manifest_asset["id"]))
            release, signature = Release.from_json(raw.decode())
        except Exception as exc:
            return self._fail(f"Could not read the update manifest: {exc}")

        ok, why = verify(release, signature, self.public_key)
        if not ok:
            # A manifest that does not verify is a security event, not a hiccup.
            return self._fail(f"Refusing this update: {why}")

        if not is_newer(release.version, self.current_version):
            return self._set(UpdateState.UP_TO_DATE,
                             f"Surface Guard {self.current_version} is up to date")

        self.status.release = release
        return self._set(UpdateState.AVAILABLE, f"Version {release.version} is available")

    # --------------------------------------------------------------- download

    def download(self, progress=None) -> UpdateStatus:
        release = self.status.release
        if release is None:
            return self._fail("No update to download")
        self._set(UpdateState.DOWNLOADING, f"Downloading version {release.version}…")
        try:
            blob = access.download_asset(self.repo, release.asset_id)
        except Exception as exc:
            return self._fail(f"Downloading the update failed: {exc}")

        digest = hashlib.sha256(blob).hexdigest()
        if digest != release.sha256:
            return self._fail("The downloaded update did not match its signed checksum")
        if release.size and len(blob) != release.size:
            return self._fail("The downloaded update was the wrong size")
        if progress:
            progress(len(blob), len(blob))

        if release.kind == MODEL:
            error = self.install_model(blob, release)
            if error:
                return self._fail(error)
            self.status.staged = None
            return self._set(
                UpdateState.READY,
                f"A better cat detector ({release.version}) is installed.",
            )

        try:
            staged = self._unpack(blob, release)
        except Exception as exc:
            return self._fail(f"Could not unpack the update: {exc}")

        self.status.staged = staged
        return self._set(UpdateState.READY, f"Version {release.version} is ready to install")

    def install_model(self, blob: bytes, release: Release) -> str:
        """Install a detector update. Returns '' on success.

        No bundle swap and no restart: the file lands beside the app's settings,
        where cat_detector looks before the bundled copy, and the engine picks it
        up on the next detector reload.
        """
        target_dir = support_dir() / "models"
        target_dir.mkdir(parents=True, exist_ok=True)
        name = release.asset_name or "yolov8m.onnx"
        if not name.endswith(".onnx"):
            return "That update does not look like a detector."
        staged = target_dir / (name + ".incoming")
        try:
            staged.write_bytes(blob)
            # Prove it loads before it becomes the model in use. A signed file that
            # onnxruntime cannot open would otherwise blind the app on next start.
            from ..detection.cat_detector import OnnxDetector

            OnnxDetector(staged)
            staged.replace(target_dir / name)
        except Exception as exc:
            staged.unlink(missing_ok=True)
            return f"The new detector would not load, so it was discarded: {exc}"
        return ""

    def _unpack(self, blob: bytes, release: Release) -> Path:
        work = Path(tempfile.mkdtemp(prefix="sg-update-", dir=str(cache_dir())))
        archive = work / (release.asset_name or "update.zip")
        archive.write_bytes(blob)
        extract = work / "extracted"
        extract.mkdir()

        # Check for path traversal before anything is written...
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                target = (extract / member).resolve()
                if not str(target).startswith(str(extract.resolve())):
                    raise RuntimeError(f"the update archive contains an unsafe path: {member}")

        # ...then unpack with ditto, not zipfile. An .app bundle is full of
        # symlinks, and Python's zipfile writes each one as a regular file holding
        # the target path. That silently breaks the code signature, and macOS then
        # refuses the update with "code object is not signed at all". ditto is what
        # made the archive and is what has to open it.
        result = subprocess.run(
            ["/usr/bin/ditto", "-x", "-k", str(archive), str(extract)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"could not unpack the update ({result.stderr.strip()[:160]})")
        archive.unlink(missing_ok=True)

        apps = [p for p in extract.rglob("*.app") if p.is_dir()]
        if not apps:
            raise RuntimeError("the update archive contains no application")
        app = min(apps, key=lambda p: len(p.parts))
        _restore_exec_bits(app)

        # The bundle must still be validly signed after the round trip, or macOS
        # will refuse to launch it and she is left with nothing.
        result = subprocess.run(
            ["/usr/bin/codesign", "--verify", "--deep", "--strict", str(app)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "the downloaded application is not correctly signed "
                f"({result.stderr.strip()[:160]})"
            )
        return app

    # ---------------------------------------------------------------- install

    def can_install(self) -> tuple[bool, str]:
        current = build_info.bundle_path()
        if current is None:
            return False, "Updates only apply to the installed app, not a development run."
        if not os.access(current.parent, os.W_OK):
            return False, (
                f"Surface Guard cannot update itself in {current.parent}. "
                "Move it to the Applications folder in your home folder."
            )
        return True, ""

    def install_and_relaunch(self) -> str:
        """Swap the bundle and start the new one. Returns '' on success."""
        staged = self.status.staged
        if staged is None or not staged.exists():
            return "No downloaded update is ready."
        ok, why = self.can_install()
        if not ok:
            return why

        current = build_info.bundle_path()
        assert current is not None
        backup = current.parent / (current.name + ".old")
        shutil.rmtree(backup, ignore_errors=True)

        try:
            os.rename(current, backup)
        except OSError as exc:
            return f"Could not move the current version aside: {exc}"
        try:
            shutil.move(str(staged), str(current))
        except Exception as exc:
            os.rename(backup, current)          # put it back; a failed update must not brick it
            return f"Could not put the new version in place: {exc}"

        # The note survives the restart; cleanup_previous() on the next start is
        # what confirms the new version actually ran.
        note_update(self.status.release.version if self.status.release else "")

        # Launch the new copy before this process exits. The old bundle is left as
        # .old and cleaned up by the next start, because deleting a bundle that is
        # still executing can crash it mid-teardown.
        subprocess.Popen(["/usr/bin/open", "-n", str(current)])
        logger.info("relaunching into %s", current)
        return ""

    # ---------------------------------------------------------------- helpers

    def _set(self, state: UpdateState, message: str) -> UpdateStatus:
        with self._lock:
            self.status.state = state
            self.status.message = message
        return self.status

    def _fail(self, message: str) -> UpdateStatus:
        return self._set(UpdateState.FAILED, message)


def _restore_exec_bits(app: Path) -> None:
    """Python's zipfile drops the executable bit; without it the app cannot launch."""
    for path in (app / "Contents" / "MacOS").glob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode | 0o111)
    for path in app.rglob("*"):
        if path.is_file() and path.suffix in ("", ".dylib", ".so") and path.parent.name in (
            "MacOS", "bin", "Helpers"
        ):
            path.chmod(path.stat().st_mode | 0o111)
