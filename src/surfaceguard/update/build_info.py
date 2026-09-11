"""Identity of this build. Overwritten by packaging/build_app.py at build time.

The values here are the development defaults. A packaged build gets a generated
copy of this file with the real version, repository and update signing key, so the
shipped app knows what it is and whose updates it will accept.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

VERSION = "0.1.0"

# owner/repo holding the releases. Private is fine; see update.access.
UPDATE_REPO = "jagreen21/surfaceguard"

# Ed25519 public key (hex). Its private half never leaves the build machine.
# Empty in development, which makes the updater refuse to install anything.
UPDATE_PUBLIC_KEY = ""

UPDATE_CHANNEL = "dev"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def bundle_path() -> Path | None:
    """The .app directory this process is running from, or None in development."""
    if not is_frozen():
        return None
    exe = Path(sys.executable).resolve()
    for parent in exe.parents:
        if parent.suffix == ".app":
            return parent
    return None


def describe() -> str:
    where = bundle_path() or Path(__file__).resolve().parents[3]
    return f"Surface Guard {VERSION} ({UPDATE_CHANNEL}) at {where}"


def env_override(name: str, default: str) -> str:
    """Let a developer point a build at a test release channel."""
    return os.environ.get(f"SURFACEGUARD_{name}", default)
