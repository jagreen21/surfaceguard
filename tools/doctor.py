#!/usr/bin/env python3
"""Check this machine can build and run Surface Guard, and say what is missing.

    make doctor
    .venv/bin/python tools/doctor.py

Written after "command not found: python" — every check here is something that
produces a confusing error somewhere else when it is wrong.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv" / "bin" / "python"

OK, WARN, BAD = "  ok  ", " warn ", " MISS "


def line(status: str, label: str, detail: str = "", fix: str = "") -> bool:
    print(f"[{status}] {label}" + (f"  —  {detail}" if detail else ""))
    if fix and status != OK:
        print(f"         fix: {fix}")
    return status == OK


def main() -> int:
    print(f"Surface Guard doctor\n  {sys.executable}\n  {ROOT}\n")
    results = []

    # --- interpreter --------------------------------------------------------
    running_in_venv = Path(sys.executable).resolve() == VENV.resolve() if VENV.exists() else False
    if VENV.exists():
        results.append(line(OK if running_in_venv else WARN, "project virtualenv",
                            "in use" if running_in_venv else "exists, but this is not it",
                            f"run commands as .venv/bin/python, or use make"))
    else:
        results.append(line(BAD, "project virtualenv", "not created", "make setup"))

    results.append(line(OK if shutil.which("python3") else BAD, "python3 on PATH",
                        sys.version.split()[0],
                        "install Python 3 from python.org or Homebrew"))
    if not shutil.which("python"):
        line(OK, "bare `python`", "absent, as expected on macOS — use .venv/bin/python")

    # --- libraries ----------------------------------------------------------
    for module, why in (("numpy", "geometry"), ("cv2", "frames and stitching"),
                        ("PySide6", "the interface"), ("av", "H.264 decoding"),
                        ("onnxruntime", "the detector"), ("cryptography", "update signatures"),
                        ("certifi", "TLS trust"), ("websockets", "the Eufy bridge"),
                        ("sounddevice", "the deterrent")):
        try:
            __import__(module)
            results.append(line(OK, module, why))
        except ImportError:
            results.append(line(BAD, module, why, "make setup"))

    # --- bundled assets -----------------------------------------------------
    node = ROOT / "runtime" / "node" / "bin" / "node"
    results.append(line(OK if node.exists() else BAD, "bundled Node + Eufy bridge",
                        "runtime/" if node.exists() else "not fetched",
                        "make runtime"))
    models = sorted((ROOT / "models").glob("*.onnx"))
    results.append(line(OK if models else BAD, "detection model",
                        ", ".join(m.name for m in models) or "none",
                        "make model"))

    # --- release toolchain --------------------------------------------------
    identity = subprocess.run(["/usr/bin/security", "find-identity", "-v", "-p", "codesigning"],
                              capture_output=True, text=True).stdout
    has_devid = "Developer ID Application" in identity
    has_dev = "Apple Development" in identity
    line(OK if (has_devid or has_dev) else WARN, "code signing identity",
         "Developer ID (notarisable)" if has_devid
         else ("Apple Development — stable, not notarisable" if has_dev else "ad-hoc only"),
         "builds still work; first launch needs a right-click → Open")

    if shutil.which("gh"):
        authed = subprocess.run(["gh", "auth", "status"], capture_output=True).returncode == 0
        line(OK if authed else WARN, "gh (publishing releases)",
             "authenticated" if authed else "installed, not signed in", "gh auth login")
    else:
        line(WARN, "gh (publishing releases)", "not installed",
             "brew install gh — only needed to publish updates")

    failed = [r for r in results if not r]
    print()
    if failed:
        print(f"{len(failed)} thing(s) need attention above.")
        return 1
    print("Everything needed to build and run is present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
