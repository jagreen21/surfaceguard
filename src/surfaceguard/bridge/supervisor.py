"""Owns the bundled Eufy bridge process.

The bridge is a Node service. Shipping it inside the app is what keeps the promise
in the README: she double-clicks one thing, and there is no Docker, no terminal and
no Node install. This module starts it, watches it, restarts it with backoff, and
reports in plain words when it cannot.

The Eufy account password has to reach the bridge through a config file — it reads
no environment variables. So the file is written 0600, and unlinked as soon as the
bridge has read it, which is a couple of seconds rather than forever.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from ..logging_setup import get as get_logger
from ..storage.preferences import support_dir
from .credentials import EufyAccount

# The bridge prints this once its WebSocket server is up.
logger = get_logger("bridge")

LISTENING_MARKER = "server listening"

# The bridge prints this per failed P2P attempt. A burst of them means the camera
# is not reachable on the network, which is a different problem from a bad password
# and deserves different words.
P2P_FAILURE_MARKER = "send cam check - error"
P2P_FAILURES_BEFORE_REPORTING = 5

# Lines worth keeping even though the bridge logs them at INFO.
NOTABLE_MARKERS = (
    "captcha", "verify code", "2fa", "login", "locked", "session", "logged in",
)

START_TIMEOUT_S = 45.0
BACKOFF_S = (2.0, 5.0, 15.0, 30.0, 60.0)
LOG_LINES = 400


def runtime_root() -> Path:
    """Where the bundled Node and bridge live, in development or inside the .app."""
    frozen = getattr(sys, "_MEIPASS", None)
    if frozen:
        # PyInstaller puts data files next to the executable's Resources.
        candidate = Path(frozen) / "runtime"
        if candidate.exists():
            return candidate
        bundled = Path(sys.executable).resolve().parent.parent / "Resources" / "runtime"
        if bundled.exists():
            return bundled
    return Path(__file__).resolve().parents[3] / "runtime"


def node_binary() -> Path:
    return runtime_root() / "node" / "bin" / "node"


def bridge_entrypoint() -> Path | None:
    root = runtime_root() / "bridge" / "node_modules" / "eufy-security-ws"
    for candidate in ("dist/bin/server.js", "dist/index.js", "bin/server.js"):
        path = root / candidate
        if path.exists():
            return path
    return None


def runtime_available() -> tuple[bool, str]:
    if not node_binary().exists():
        return False, f"The bundled camera service is missing from this install ({node_binary()})."
    if bridge_entrypoint() is None:
        return False, "The bundled camera service is incomplete; reinstall Surface Guard."
    return True, ""


def free_port(preferred: int = 3050) -> int:
    """A port of our own, so a bridge the user already runs is left alone."""
    for port in range(preferred, preferred + 40):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@dataclass
class BridgeStatus:
    running: bool = False
    listening: bool = False
    lan_unreachable: bool = False
    port: int = 0
    pid: int | None = None
    restarts: int = 0
    message: str = "Not started"
    fatal: str = ""          # set when restarting cannot help

    @property
    def healthy(self) -> bool:
        return self.running and self.listening and not self.fatal


class BridgeSupervisor:
    """Starts the bridge, keeps it alive, and says why when it will not start."""

    def __init__(self, account: EufyAccount, port: int | None = None) -> None:
        self.account = account
        self.port = port or free_port()
        self.status = BridgeStatus(port=self.port)
        self._proc: subprocess.Popen | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._logs: deque[str] = deque(maxlen=LOG_LINES)
        self._listening = threading.Event()
        self._p2p_failures = 0
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ paths

    @property
    def work_dir(self) -> Path:
        d = support_dir() / "bridge"
        d.mkdir(parents=True, exist_ok=True)
        os.chmod(d, 0o700)
        return d

    @property
    def url(self) -> str:
        # Explicitly 127.0.0.1: the bridge's default host is "localhost", which on
        # macOS resolves to ::1 first and then refuses IPv4 connections.
        return f"ws://127.0.0.1:{self.port}"

    # -------------------------------------------------------------- lifecycle

    def start(self, wait: bool = True) -> BridgeStatus:
        ok, why = runtime_available()
        if not ok:
            self.status.fatal = why
            self.status.message = why
            return self.status
        if not self.account.username:
            self.status.fatal = "No Eufy account has been set up yet."
            self.status.message = self.status.fatal
            return self.status
        if self.account.password is None:
            self.status.fatal = (
                "The Eufy password is not in the Keychain. Sign in again in Settings."
            )
            self.status.message = self.status.fatal
            return self.status

        self._stop.clear()
        self.status.fatal = ""
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._supervise, name="sg-bridge", daemon=True)
            self._thread.start()
        if wait:
            self._listening.wait(timeout=START_TIMEOUT_S)
        return self.status

    def stop(self) -> None:
        self._stop.set()
        self._kill()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        self.status.running = False
        self.status.listening = False
        self.status.message = "Stopped"

    def _kill(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()

    # ---------------------------------------------------------------- the loop

    def _supervise(self) -> None:
        attempt = 0
        while not self._stop.is_set():
            started = time.monotonic()
            code = self._run_once()
            if self._stop.is_set():
                return
            uptime = time.monotonic() - started
            if uptime > 120:
                attempt = 0            # it was healthy for a while; not a crash loop
            delay = BACKOFF_S[min(attempt, len(BACKOFF_S) - 1)]
            attempt += 1
            self.status.restarts += 1
            self.status.running = False
            self.status.listening = False
            self._listening.clear()
            self.status.message = (
                f"The camera service stopped (exit {code}). Restarting in {delay:.0f}s."
            )
            logger.warning("bridge exited (%s) after %.0fs; restart #%d in %.0fs",
                        code, uptime, self.status.restarts, delay)
            self._stop.wait(delay)

    def _run_once(self) -> int | None:
        config = self._write_config()
        try:
            proc = subprocess.Popen(
                [
                    str(node_binary()), str(bridge_entrypoint()),
                    "-c", str(config),
                    "-p", str(self.port),
                    "-H", "127.0.0.1",
                ],
                cwd=str(self.work_dir),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                # Deliberately NOT start_new_session=True. setsid() makes the bridge
                # its own responsible process for macOS privacy, so it stops
                # inheriting the app's Local Network permission — and the Eufy
                # cameras are on the LAN. The symptom is not a permission dialog but
                # a stream of "[p2p] Send cam check - Error": the user allowed local
                # network access for Surface Guard, and node was never covered by it.
            )
        except OSError as exc:
            config.unlink(missing_ok=True)
            self.status.fatal = f"Could not start the camera service: {exc}"
            self.status.message = self.status.fatal
            return None

        with self._lock:
            self._proc = proc
        self.status.running = True
        self.status.pid = proc.pid
        self.status.message = "Starting the camera service…"

        threading.Thread(
            target=self._drain, args=(proc, config), name="sg-bridge-log", daemon=True
        ).start()
        return proc.wait()

    def _drain(self, proc: subprocess.Popen, config: Path) -> None:
        """Read the bridge's output, and clean the config file up once it is loaded."""
        assert proc.stdout is not None
        cleaned = False
        for line in proc.stdout:
            text = _strip_ansi(line.rstrip())
            if text:
                self._logs.append(text)
                if P2P_FAILURE_MARKER in text.lower():
                    self._p2p_failures += 1
                    if self._p2p_failures == P2P_FAILURES_BEFORE_REPORTING:
                        self.status.lan_unreachable = True
                        logger.error(
                            "the camera is not answering on the local network "
                            "(%d p2p failures)", self._p2p_failures,
                        )
                elif "ERROR" in text or "FATAL" in text:
                    logger.error("bridge: %s", text[:300])
                elif any(word in text.lower() for word in NOTABLE_MARKERS):
                    # Login progress is INFO, not ERROR, so the old filter dropped
                    # it — including "captcha required", which was the entire reason
                    # sign-in was failing and never appeared in a single log.
                    logger.info("bridge: %s", text[:300])
            if not cleaned and LISTENING_MARKER in text.lower():
                self.status.listening = True
                self.status.message = f"Camera service running on port {self.port}"
                self._listening.set()
                logger.info("bridge listening on port %d", self.port)
                # The bridge has parsed the config; the password no longer needs
                # to exist on disk.
                config.unlink(missing_ok=True)
                cleaned = True
        if not cleaned:
            config.unlink(missing_ok=True)

    def _write_config(self) -> Path:
        password = self.account.password or ""
        path = self.work_dir / "config.json"
        # Create 0600 *before* writing, so the secret is never briefly world-readable.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            json.dump({
                "username": self.account.username,
                "password": password,
                "country": self.account.country,
                "language": self.account.language,
                "trustedDeviceName": "Surface Guard",
                "persistentDir": str(self.work_dir),
                "eventDurationSeconds": 10,
                "acceptInvitations": False,
            }, fh)
        return path

    # ------------------------------------------------------------------ output

    def logs(self, limit: int = 60) -> list[str]:
        return list(self._logs)[-limit:]

    def last_error(self) -> str:
        for line in reversed(self._logs):
            if "ERROR" in line or "FATAL" in line:
                return line[:300]
        return ""


def _strip_ansi(text: str) -> str:
    out, i = [], 0
    while i < len(text):
        if text[i] == "\x1b" and i + 1 < len(text) and text[i + 1] == "[":
            j = i + 2
            while j < len(text) and not text[j].isalpha():
                j += 1
            i = j + 1
            continue
        out.append(text[i])
        i += 1
    return "".join(out)
