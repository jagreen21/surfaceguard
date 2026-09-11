#!/usr/bin/env python3
"""Package, sign and publish a release.

    python packaging/make_release.py --version 0.2.0            # build + publish
    python packaging/make_release.py --show-key                 # the public key
    python packaging/make_release.py --version 0.2.0 --no-upload

The Ed25519 signing key lives in *your* Keychain and never enters the repository
or a build. Its public half is compiled into the app, so a release the app will
install must have been signed on this machine — whoever else controls the GitHub
account cannot put code on her Mac.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import require_venv  # noqa: E402

require_venv('cryptography')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from surfaceguard.bridge.credentials import get_secret, set_secret  # noqa: E402

SIGNING_KEY_ACCOUNT = "update-signing-key"
DIST = ROOT / "dist"
APP_NAME = "Surface Guard"
MANIFEST_ASSET = "release.json"


def log(msg: str) -> None:
    print(f"  {msg}", flush=True)


# ------------------------------------------------------------------ signing key


def load_or_create_key():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    stored = get_secret(SIGNING_KEY_ACCOUNT)
    if stored:
        return Ed25519PrivateKey.from_private_bytes(bytes.fromhex(stored))
    key = Ed25519PrivateKey.generate()
    raw = key.private_bytes(
        serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    set_secret(SIGNING_KEY_ACCOUNT, raw.hex(), label="Surface Guard — update signing key")
    log("created a new update signing key and stored it in your Keychain")
    log("back it up: losing it means shipped apps will reject every future update")
    return key


def public_key_hex(key) -> str:
    from cryptography.hazmat.primitives import serialization

    return key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    ).hex()


# --------------------------------------------------------------------- packaging


def zip_app(app: Path, version: str) -> Path:
    """Zip the bundle, preserving the symlinks and permission bits macOS needs."""
    out = DIST / f"SurfaceGuard-{version}.zip"
    out.unlink(missing_ok=True)
    # ditto keeps resource forks, symlinks and the code signature intact; a plain
    # zipfile walk does not, and the result fails codesign --verify on the far side.
    result = subprocess.run(
        ["/usr/bin/ditto", "-c", "-k", "--sequesterRsrc", "--keepParent",
         str(app), str(out)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"ditto failed: {result.stderr.strip()}")
    return out


def verify_roundtrip(archive: Path) -> None:
    """Unzip into a temp dir and confirm the signature survives, before publishing."""
    scratch = DIST / "_verify"
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    subprocess.run(["/usr/bin/ditto", "-x", "-k", str(archive), str(scratch)], check=True)
    apps = [p for p in scratch.rglob("*.app") if p.is_dir()]
    if not apps:
        raise SystemExit("the archive contains no .app")
    result = subprocess.run(
        ["/usr/bin/codesign", "--verify", "--deep", "--strict", str(apps[0])],
        capture_output=True, text=True,
    )
    shutil.rmtree(scratch, ignore_errors=True)
    if result.returncode != 0:
        raise SystemExit(
            "the zipped app fails signature verification, so the updater would "
            f"refuse it: {result.stderr.strip()[:200]}"
        )
    log("round-trip verified: the zipped app is still correctly signed")


# --------------------------------------------------------------------- publishing


def publish(repo: str, version: str, archive: Path, manifest: Path, notes: str) -> None:
    if shutil.which("gh") is None:
        raise SystemExit("gh is not installed; publish the assets manually")
    tag = f"v{version}"
    exists = subprocess.run(
        ["gh", "release", "view", tag, "--repo", repo], capture_output=True
    ).returncode == 0
    if exists:
        log(f"release {tag} exists; replacing its assets")
        subprocess.run(
            ["gh", "release", "upload", tag, str(archive), str(manifest),
             "--repo", repo, "--clobber"], check=True,
        )
    else:
        subprocess.run(
            ["gh", "release", "create", tag, str(archive), str(manifest),
             "--repo", repo, "--title", f"Surface Guard {version}",
             "--notes", notes or f"Surface Guard {version}"], check=True,
        )
    log(f"published {tag} to {repo}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version")
    ap.add_argument("--repo", default="jagreen21/surfaceguard")
    ap.add_argument("--notes", default="")
    ap.add_argument("--show-key", action="store_true")
    ap.add_argument("--no-build", action="store_true", help="use the existing dist/*.app")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--with-onnx", action="store_true")
    ap.add_argument("--model", type=Path,
                    help="publish a detector update instead of a whole app: ~99 MB "
                         "rather than 245 MB, and it installs without a restart")
    args = ap.parse_args()

    key = load_or_create_key()
    pub = public_key_hex(key)
    if args.show_key:
        print(pub)
        return 0
    if not args.version:
        ap.error("--version is required")

    if args.model:
        return _publish_model(args, key, pub)

    app = DIST / f"{APP_NAME}.app"
    if not args.no_build:
        cmd = [sys.executable, str(ROOT / "packaging" / "build_app.py"),
               "--version", args.version, "--repo", args.repo, "--key", pub]
        if args.with_onnx:
            cmd.append("--with-onnx")
        subprocess.run(cmd, check=True)
    if not app.exists():
        raise SystemExit(f"no app at {app}; drop --no-build")

    log("zipping")
    archive = zip_app(app, args.version)
    verify_roundtrip(archive)

    blob = archive.read_bytes()
    from surfaceguard.update.manifest import Release

    release = Release(
        version=args.version,
        sha256=hashlib.sha256(blob).hexdigest(),
        size=len(blob),
        notes=args.notes,
        asset_name=archive.name,
        min_macos="12.0",
    )
    manifest = DIST / MANIFEST_ASSET

    if not args.no_upload:
        publish(args.repo, args.version, archive, _write_manifest(manifest, release, key), args.notes)
        # asset_id is only known once GitHub has the file, so stamp it and re-upload.
        asset_id = _lookup_asset_id(args.repo, args.version, archive.name)
        release.asset_id = asset_id
        _write_manifest(manifest, release, key)
        subprocess.run(
            ["gh", "release", "upload", f"v{args.version}", str(manifest),
             "--repo", args.repo, "--clobber"], check=True,
        )
        log(f"manifest points at asset {asset_id}")
    else:
        _write_manifest(manifest, release, key)
        log(f"built but not uploaded: {archive} ({len(blob) / 1e6:.0f} MB)")

    print(f"\nRelease {args.version} ready.")
    print(f"  public key: {pub}")
    print(f"  archive   : {archive}")
    print(f"  manifest  : {manifest}")
    return 0


def _publish_model(args, key, pub) -> int:
    """Publish a detector on its own, verified before anyone downloads it."""
    from surfaceguard.update.manifest import MODEL, Release

    model = args.model
    if not model.exists():
        raise SystemExit(f"no model at {model}")
    if model.suffix != ".onnx":
        raise SystemExit("a detector update must be a .onnx file")

    # Load it here rather than discovering on her Mac that it is unusable.
    sys.path.insert(0, str(ROOT / "src"))
    from surfaceguard.detection.cat_detector import OnnxDetector

    log(f"checking {model.name} loads")
    OnnxDetector(model)

    blob = model.read_bytes()
    DIST.mkdir(parents=True, exist_ok=True)
    staged = DIST / model.name
    if staged.resolve() != model.resolve():
        staged.write_bytes(blob)

    release = Release(
        version=args.version, sha256=hashlib.sha256(blob).hexdigest(), size=len(blob),
        kind=MODEL, notes=args.notes, asset_name=model.name,
    )
    manifest = DIST / MANIFEST_ASSET
    if args.no_upload:
        _write_manifest(manifest, release, key)
        log(f"built but not uploaded: {staged} ({len(blob) / 1e6:.0f} MB)")
    else:
        publish(args.repo, args.version, staged, _write_manifest(manifest, release, key),
                args.notes)
        release.asset_id = _lookup_asset_id(args.repo, args.version, model.name)
        _write_manifest(manifest, release, key)
        subprocess.run(["gh", "release", "upload", f"v{args.version}", str(manifest),
                        "--repo", args.repo, "--clobber"], check=True)
    print(f"\nDetector release {args.version} ready.")
    print(f"  public key: {pub}")
    print(f"  model     : {staged} ({len(blob) / 1e6:.0f} MB)")
    return 0


def _write_manifest(path: Path, release, key) -> Path:
    signature = key.sign(release.signing_payload()).hex()
    path.write_text(release.to_json(signature))
    return path


def _lookup_asset_id(repo: str, version: str, name: str) -> int:
    out = subprocess.run(
        ["gh", "api", f"repos/{repo}/releases/tags/v{version}"],
        capture_output=True, text=True, check=True,
    ).stdout
    for asset in json.loads(out).get("assets", []):
        if asset.get("name") == name:
            return int(asset["id"])
    raise SystemExit(f"{name} is not attached to release v{version}")


if __name__ == "__main__":
    raise SystemExit(main())
