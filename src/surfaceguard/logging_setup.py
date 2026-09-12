"""File logging, because a packaged app has nowhere to print.

PyInstaller's windowed bootloader detaches stdout, so on her Mac every ``print``
goes nowhere. When something breaks there and nobody technical is in the room, the
log file is the only evidence that exists — so it is written from the first line of
startup, rotated, and reachable from a button in the app.

Secrets never reach it: the Keychain is read through a separate path and passwords
are never passed to a logger.
"""

from __future__ import annotations

import logging
import logging.handlers
import platform
import re
import sys
from pathlib import Path

LOG_DIR = Path.home() / "Library" / "Logs" / "SurfaceGuard"
LOG_FILE = LOG_DIR / "surfaceguard.log"
MAX_BYTES = 2_000_000
BACKUPS = 3

_configured = False
_EMAIL = re.compile(r"[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}")
_EUFY_SERIAL = re.compile(r"\bT[A-Z0-9]{8,}\b", re.IGNORECASE)


def log_dir() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR


def setup(verbose: bool = False) -> logging.Logger:
    """Install file logging. Safe to call more than once."""
    global _configured
    root = logging.getLogger("surfaceguard")
    if _configured:
        return root

    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-7s %(name)-28s %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    try:
        handler = logging.handlers.RotatingFileHandler(
            log_dir() / LOG_FILE.name, maxBytes=MAX_BYTES, backupCount=BACKUPS
        )
        handler.setFormatter(fmt)
        root.addHandler(handler)
    except OSError:
        pass  # a read-only home is not a reason to refuse to start

    # Keep console output when run from a terminal; harmless when detached.
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    root.addHandler(console)

    _configured = True
    root.info("=" * 70)
    root.info("Surface Guard starting on %s, Python %s",
              platform.platform(), platform.python_version())
    _install_excepthook(root)
    return root


def get(name: str) -> logging.Logger:
    return logging.getLogger(f"surfaceguard.{name}")


def _install_excepthook(logger: logging.Logger) -> None:
    """An unhandled crash must land in the log, not vanish with the process."""
    previous = sys.excepthook

    def hook(exc_type, exc, tb) -> None:
        logger.critical("Unhandled exception", exc_info=(exc_type, exc, tb))
        previous(exc_type, exc, tb)

    sys.excepthook = hook

    try:
        import threading

        def thread_hook(args) -> None:
            logger.critical(
                "Unhandled exception in thread %s", args.thread_name if args else "?",
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )

        threading.excepthook = thread_hook
    except Exception:
        pass


def tail(lines: int = 200) -> str:
    """The end of the log, for the diagnostics screen and the copy button."""
    path = LOG_DIR / LOG_FILE.name
    if not path.exists():
        return "(no log yet)"
    try:
        text = path.read_text(errors="replace").splitlines()
    except OSError as exc:
        return f"(could not read the log: {exc})"
    return "\n".join(text[-lines:])


def redact_support_text(text: str) -> str:
    """Remove account and device IDs while retaining useful model numbers."""
    return _EUFY_SERIAL.sub("T-REDACTED", _EMAIL.sub("EMAIL-REDACTED", text))
