#!/usr/bin/env python3
"""Build Surface Guard.app.

    python packaging/build_app.py --version 0.2.0 --repo owner/name --key <hex>

Signing, in order of preference:

  1. Developer ID Application  — clean install, notarisable
  2. Apple Development         — no Gatekeeper benefit, but a *stable* identity,
                                 which keeps Keychain access across updates
  3. ad-hoc                    — works, but the signature changes every build

Two is worth preferring over three even without a paid account: Keychain grants
are per accessing binary, so a signature that changes on every OTA update can make
macOS re-prompt. (Surface Guard reads secrets through /usr/bin/security to dodge
that, but a stable identity is still the better default.)
"""

from __future__ import annotations

import argparse
import plistlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DIST = ROOT / "dist"
BUILD = ROOT / "build"
RUNTIME = ROOT / "runtime"
APP_NAME = "Surface Guard"
BUNDLE_ID = "com.surfaceguard.app"

# Qt ships far more than this app uses; each of these is tens of MB.
EXCLUDE_QT = [
    "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtQuick3D", "PySide6.QtQuickWidgets",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.QtMultimedia", "PySide6.QtMultimediaWidgets", "PySide6.Qt3DCore",
    "PySide6.Qt3DRender", "PySide6.Qt3DAnimation", "PySide6.Qt3DExtras",
    "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtBluetooth",
    "PySide6.QtPositioning", "PySide6.QtSerialPort", "PySide6.QtTest",
    "PySide6.QtSql", "PySide6.QtPdf", "PySide6.QtPdfWidgets", "PySide6.QtDesigner",
    "PySide6.QtHelp", "PySide6.QtUiTools", "PySide6.QtSpatialAudio",
]
EXCLUDE_OTHER = ["tkinter", "matplotlib", "IPython", "pytest", "setuptools", "pip"]


def log(msg: str) -> None:
    print(f"  {msg}", flush=True)


def signing_identity(preferred: str | None = None) -> tuple[str, str]:
    """Return (identity, human description)."""
    if preferred:
        return preferred, f"requested identity {preferred!r}"
    out = subprocess.run(
        ["/usr/bin/security", "find-identity", "-v", "-p", "codesigning"],
        capture_output=True, text=True,
    ).stdout
    for pattern, label in (
        (r'"(Developer ID Application[^"]+)"', "Developer ID (notarisable)"),
        (r'"(Apple Development[^"]+)"', "Apple Development (stable identity, not notarisable)"),
    ):
        match = re.search(pattern, out)
        if match:
            return match.group(1), f"{label}: {match.group(1)}"
    return "-", "ad-hoc (signature changes every build)"


def write_build_info(version: str, repo: str, public_key: str, channel: str) -> Path:
    """Stamp identity into the source tree so the frozen app knows what it is."""
    path = SRC / "surfaceguard" / "update" / "build_info.py"
    original = path.read_text()
    stamped = original
    for name, value in (
        ("VERSION", version), ("UPDATE_REPO", repo),
        ("UPDATE_PUBLIC_KEY", public_key), ("UPDATE_CHANNEL", channel),
    ):
        stamped = re.sub(rf'^{name} = ".*"$', f'{name} = "{value}"', stamped, flags=re.M)
    backup = path.with_suffix(".py.orig")
    backup.write_text(original)
    path.write_text(stamped)
    return backup


def restore_build_info(backup: Path) -> None:
    target = backup.with_suffix("")
    if backup.exists():
        target.write_text(backup.read_text())
        backup.unlink()


def info_plist(version: str) -> dict:
    return {
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleShortVersionString": version,
        "CFBundleVersion": version,
        "CFBundlePackageType": "APPL",
        "LSMinimumSystemVersion": "12.0",
        "NSHighResolutionCapable": True,
        # macOS 15+ requires this before an app may reach a camera on the LAN.
        # Without it the bridge connects to nothing and never says why.
        "NSLocalNetworkUsageDescription":
            "Surface Guard connects to your camera on your home network to watch the "
            "surfaces you have chosen.",
        "NSBonjourServices": ["_eufy._tcp", "_http._tcp"],
        # The app keeps running from the menu bar with no window open.
        "LSUIElement": False,
        "NSSupportsAutomaticTermination": False,
        "NSSupportsSuddenTermination": False,
    }


def build(args) -> Path:
    if not (RUNTIME / "node" / "bin" / "node").exists():
        raise SystemExit(
            "runtime/ is missing. Run: python packaging/fetch_runtime.py"
        )
    model = ROOT / "models" / "yolov8n.onnx"
    if args.with_onnx and not model.exists():
        raise SystemExit(
            f"--with-onnx needs a detection model at {model}. "
            "See README: export one with packaging/export_model.py"
        )
    for path in (DIST, BUILD):
        shutil.rmtree(path, ignore_errors=True)

    backup = write_build_info(args.version, args.repo, args.key, args.channel)
    try:
        cmd = [
            str(ROOT / ".venv" / "bin" / "pyinstaller"),
            "--noconfirm", "--clean", "--windowed",
            "--name", APP_NAME,
            "--osx-bundle-identifier", BUNDLE_ID,
            "--paths", str(SRC),
            # The whole Node + bridge runtime rides along inside Resources.
            "--add-data", f"{RUNTIME}:runtime",
            "--add-data", f"{ROOT / 'models'}:models",
            "--collect-submodules", "surfaceguard",
            "--hidden-import", "surfaceguard.app",
            "--collect-binaries", "av",
            "--collect-data", "certifi",
            "--osx-entitlements-file", str(entitlements_file()),
            str(SRC / "surfaceguard" / "__main__.py"),
        ]
        for module in EXCLUDE_QT + EXCLUDE_OTHER:
            cmd += ["--exclude-module", module]
        if not args.with_onnx:
            cmd += ["--exclude-module", "onnxruntime"]

        log("running pyinstaller (this takes a few minutes)")
        subprocess.run(cmd, cwd=ROOT, check=True)
    finally:
        restore_build_info(backup)

    app = DIST / f"{APP_NAME}.app"
    if not app.exists():
        raise SystemExit("pyinstaller finished but produced no .app")

    plist_path = app / "Contents" / "Info.plist"
    merged = plistlib.loads(plist_path.read_bytes())
    merged.update(info_plist(args.version))
    plist_path.write_bytes(plistlib.dumps(merged))

    # The bundled node must stay executable through the copy.
    node = app / "Contents" / "Resources" / "runtime" / "node" / "bin" / "node"
    if node.exists():
        node.chmod(0o755)
    else:
        raise SystemExit(f"the bundled node is missing from the app at {node}")

    identity, description = signing_identity(args.identity)
    log(f"signing with {description}")
    # Sign inner binaries first, then the bundle, or the outer signature is invalid.
    subprocess.run(
        ["/usr/bin/codesign", "--force", "--sign", identity, "--timestamp=none",
         "--options", "runtime", "--entitlements", str(entitlements_file()), str(node)],
        check=False, capture_output=True,
    )
    result = subprocess.run(
        ["/usr/bin/codesign", "--force", "--deep", "--sign", identity,
         "--timestamp=none", "--options", "runtime",
         "--entitlements", str(entitlements_file()), str(app)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        log(f"WARNING: signing failed: {result.stderr.strip()[:300]}")
    else:
        verify = subprocess.run(
            ["/usr/bin/codesign", "--verify", "--deep", "--strict", str(app)],
            capture_output=True, text=True,
        )
        log("signature verifies" if verify.returncode == 0
            else f"WARNING: verification failed: {verify.stderr.strip()[:200]}")

    size = sum(f.stat().st_size for f in app.rglob("*") if f.is_file())
    log(f"built {app} ({size / 1e6:.0f} MB)")
    return app


def entitlements_file() -> Path:
    path = BUILD / "surfaceguard.entitlements"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(plistlib.dumps({
        # PyInstaller's bootloader and Python itself need these under hardened runtime.
        "com.apple.security.cs.allow-unsigned-executable-memory": True,
        "com.apple.security.cs.disable-library-validation": True,
        "com.apple.security.cs.allow-dyld-environment-variables": True,
        "com.apple.security.network.client": True,
        "com.apple.security.network.server": True,
    }))
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", default="0.2.0")
    ap.add_argument("--repo", default="jagreen21/surfaceguard")
    ap.add_argument("--key", default="", help="Ed25519 update public key (hex)")
    ap.add_argument("--channel", default="release")
    ap.add_argument("--identity", default=None, help="override the signing identity")
    ap.add_argument("--with-onnx", action="store_true",
                    help="bundle onnxruntime (needed for real detection)")
    args = ap.parse_args()

    if not args.key:
        log("WARNING: no update public key. The build will refuse every update.")
    print(f"Building {APP_NAME} {args.version}")
    app = build(args)
    print(f"\nDone: {app}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
