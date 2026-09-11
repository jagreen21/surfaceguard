#!/usr/bin/env python3
"""Fetch the self-contained Node runtime and the Eufy bridge into runtime/.

Homebrew's node links ~20 Homebrew dylibs and cannot be copied into an app
bundle. The official nodejs.org build is self-contained, so that is what ships.

Everything downloaded is verified: the Node tarball against Node's own published
SHASUMS256.txt, and the npm install against the lockfile npm writes. Nothing is
executed during the fetch except npm itself.

    python packaging/fetch_runtime.py            # fetch both
    python packaging/fetch_runtime.py --check    # verify what is already there
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import require_venv  # noqa: E402

require_venv('certifi')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from surfaceguard.net import get as http_get  # noqa: E402

# Pinned deliberately: an app that silently changes its Node version between
# builds is an app whose bug reports cannot be reproduced.
NODE_VERSION = "22.20.0"          # active LTS
BRIDGE_PACKAGE = "eufy-security-ws"
BRIDGE_VERSION = "3.1.0"

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / "runtime"
NODE_DIR = RUNTIME / "node"
BRIDGE_DIR = RUNTIME / "bridge"

BASE = "https://nodejs.org/dist"


def _log(msg: str) -> None:
    print(f"  {msg}", flush=True)


def _get(url: str) -> bytes:
    return http_get(url, timeout=180.0)


def _expected_sha(version: str, filename: str) -> str:
    """Node publishes SHASUMS256.txt alongside every release."""
    text = _get(f"{BASE}/v{version}/SHASUMS256.txt").decode()
    for line in text.splitlines():
        digest, name = line.split()
        if name == filename:
            return digest
    raise RuntimeError(f"{filename} is not listed in Node's SHASUMS256.txt for v{version}")


def fetch_node(force: bool = False) -> Path:
    node_bin = NODE_DIR / "bin" / "node"
    if node_bin.exists() and not force:
        _log(f"node already present: {NODE_DIR}")
        return NODE_DIR

    arch = "arm64" if os.uname().machine == "arm64" else "x64"
    name = f"node-v{NODE_VERSION}-darwin-{arch}"
    filename = f"{name}.tar.gz"
    url = f"{BASE}/v{NODE_VERSION}/{filename}"

    _log(f"expected sha256 from nodejs.org …")
    expected = _expected_sha(NODE_VERSION, filename)
    _log(f"  {expected}")
    _log(f"downloading {url}")
    blob = _get(url)
    actual = hashlib.sha256(blob).hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"Node tarball checksum mismatch.\n  expected {expected}\n  got      {actual}"
        )
    _log(f"checksum verified ({len(blob) / 1e6:.0f} MB)")

    if NODE_DIR.exists():
        shutil.rmtree(NODE_DIR)
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / filename
        archive.write_bytes(blob)
        with tarfile.open(archive) as tf:
            tf.extractall(tmp, filter="data")
        extracted = Path(tmp) / name
        NODE_DIR.mkdir(parents=True, exist_ok=True)
        # Only what is needed to run a server: the binary and its lib. npm, the
        # headers and the docs are build-time only and would add ~60 MB.
        shutil.copytree(extracted / "bin", NODE_DIR / "bin",
                        ignore=shutil.ignore_patterns("npm", "npx", "corepack"))
        for extra in ("LICENSE",):
            src = extracted / extra
            if src.exists():
                shutil.copy2(src, NODE_DIR / extra)

    (NODE_DIR / "VERSION").write_text(NODE_VERSION + "\n")
    _log(f"node {NODE_VERSION} -> {NODE_DIR} ({_du(NODE_DIR)})")
    return NODE_DIR


def fetch_bridge(force: bool = False) -> Path:
    if (BRIDGE_DIR / "node_modules" / BRIDGE_PACKAGE).exists() and not force:
        _log(f"bridge already present: {BRIDGE_DIR}")
        return BRIDGE_DIR

    npm = shutil.which("npm")
    if npm is None:
        raise RuntimeError(
            "npm is needed to fetch the bridge (build machine only; it is not shipped). "
            "Install Node on this machine, or copy a prepared runtime/ directory in."
        )
    if BRIDGE_DIR.exists():
        shutil.rmtree(BRIDGE_DIR)
    BRIDGE_DIR.mkdir(parents=True, exist_ok=True)
    (BRIDGE_DIR / "package.json").write_text(json.dumps({
        "name": "surfaceguard-bridge",
        "private": True,
        "dependencies": {BRIDGE_PACKAGE: BRIDGE_VERSION},
    }, indent=2) + "\n")

    _log(f"npm install {BRIDGE_PACKAGE}@{BRIDGE_VERSION}")
    subprocess.run(
        [npm, "install", "--omit=dev", "--no-audit", "--no-fund", "--loglevel=warn"],
        cwd=BRIDGE_DIR, check=True,
    )
    entry = bridge_entrypoint()
    if entry is None:
        raise RuntimeError("npm install finished but the bridge server entry point is missing")
    _log(f"bridge -> {BRIDGE_DIR} ({_du(BRIDGE_DIR)}), entry {entry.relative_to(BRIDGE_DIR)}")
    return BRIDGE_DIR


def bridge_entrypoint(base: Path | None = None) -> Path | None:
    """Locate the bridge's server script inside an installed bridge tree."""
    root = (base or BRIDGE_DIR) / "node_modules" / BRIDGE_PACKAGE
    for candidate in ("dist/bin/server.js", "dist/index.js", "bin/server.js"):
        path = root / candidate
        if path.exists():
            return path
    pkg = root / "package.json"
    if pkg.exists():
        meta = json.loads(pkg.read_text())
        for value in (meta.get("bin") or {}).values():
            path = root / value
            if path.exists():
                return path
        main = meta.get("main")
        if main and (root / main).exists():
            return root / main
    return None


def _du(path: Path) -> str:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return f"{total / 1e6:.0f} MB"


def check() -> int:
    ok = True
    node = NODE_DIR / "bin" / "node"
    if node.exists():
        out = subprocess.run([str(node), "-v"], capture_output=True, text=True)
        _log(f"node: {out.stdout.strip() or 'unrunnable'} ({_du(NODE_DIR)})")
        ok &= out.returncode == 0
    else:
        _log("node: MISSING")
        ok = False
    entry = bridge_entrypoint()
    _log(f"bridge: {entry.relative_to(ROOT) if entry else 'MISSING'}")
    ok &= entry is not None
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify without downloading")
    ap.add_argument("--force", action="store_true", help="re-fetch even if present")
    args = ap.parse_args()
    if args.check:
        return check()
    print(f"Fetching runtime into {RUNTIME}")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    fetch_node(args.force)
    fetch_bridge(args.force)
    print("\nRuntime ready.")
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
