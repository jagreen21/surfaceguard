#!/usr/bin/env python3
"""Build the thing you AirDrop: the app plus a one-click installer.

    .venv/bin/python packaging/make_installer.py
    .venv/bin/python packaging/make_installer.py --token github_pat_...

Everything in the runbook that a script *can* do, this does: copies the app to
~/Applications, clears the quarantine flag AirDrop sets, turns on start-at-login,
and opens the app. No update token is needed — the release repository is public,
and it is the signature on each build, not the repository's privacy, that stops
anything else installing.

Two steps stay manual because macOS will not let them be otherwise, and pretending
they are automated is how someone ends up staring at an app that silently never
finds the camera:

  * **Local Network permission** — an OS prompt. She has to click Allow.
  * **Her Eufy password** — it goes into the Keychain via the app's own sign-in.
    An installer that asked for it would be teaching her to type her password into
    whatever asks.

The installer is a signed .app bundle, not a .command file. A loose shell script
cannot be code-signed, so macOS rejects one delivered by AirDrop outright — spctl
says "no usable signature" and there is no working way past it. Wrapped in a signed
bundle the same logic gets the ordinary unidentified-developer dialog that
right-click -> Open clears. Its executable is still a plain shell script anyone can
read, and its dialogs are native rather than a Terminal window.
"""

from __future__ import annotations

import argparse
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _env import require_venv  # noqa: E402

require_venv()

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
APP_NAME = "Surface Guard"

SCRIPT = (ROOT / "packaging" / "installer_script.sh").read_text()

INSTALLER_PLIST = {
    "CFBundleName": "Install Surface Guard",
    "CFBundleDisplayName": "Install Surface Guard",
    "CFBundleIdentifier": "com.surfaceguard.installer",
    "CFBundleExecutable": "install",
    "CFBundlePackageType": "APPL",
    "LSMinimumSystemVersion": "12.0",
    "NSHighResolutionCapable": True,
}


TOKEN_BLOCK = r"""printf '  Storing the update key\n'
/usr/bin/security add-generic-password -a update-token -s SurfaceGuard \
  -l 'Surface Guard - update access token' -U -w '__TOKEN__' >/dev/null 2>&1
"""

READ_ME = """Surface Guard

1. RIGHT-CLICK "Install Surface Guard" and choose Open.
   Then click Open again in the box that appears.

   Do not double-click it the first time. macOS will say it cannot check the
   app for malicious software and offer only Cancel - that is just what it
   says about anything not bought from the App Store. Right-click and Open
   gets the button you need. You only do this once.

2. Click Install, and wait a few seconds.

3. Say Allow when macOS asks about finding devices on the local network.
   The camera is on the Wi-Fi; without this the app cannot find it.

4. Sign in with the Eufy account - the same email and password as the Eufy app.

That's it.
"""


def signing_identity() -> tuple[str, str]:
    """The same identity the app itself was signed with, if there is one."""
    out = subprocess.run(["/usr/bin/security", "find-identity", "-v", "-p", "codesigning"],
                         capture_output=True, text=True).stdout
    for pattern, label in ((r'"(Developer ID Application[^"]+)"', "Developer ID"),
                           (r'"(Apple Development[^"]+)"', "Apple Development")):
        found = re.search(pattern, out)
        if found:
            return found.group(1), label
    return "-", "ad-hoc"


def build_installer_app(staging: Path, token: str, version: str) -> Path:
    """A signed .app around the shell script, because a loose script cannot be signed."""
    bundle = staging / "Install Surface Guard.app"
    macos = bundle / "Contents" / "MacOS"
    macos.mkdir(parents=True)

    script = SCRIPT.replace(
        "__TOKEN_BLOCK__",
        TOKEN_BLOCK.replace("__TOKEN__", token) if token else
        "# the release repository is public; no credential is stored",
    )
    executable = macos / "install"
    executable.write_text(script)
    executable.chmod(0o755)

    plist = dict(INSTALLER_PLIST)
    plist["CFBundleShortVersionString"] = version
    plist["CFBundleVersion"] = version
    (bundle / "Contents" / "Info.plist").write_bytes(plistlib.dumps(plist))

    identity, label = signing_identity()
    print(f"  signing the installer with {label}")
    subprocess.run(["/usr/bin/codesign", "--force", "--deep", "--sign", identity,
                    "--timestamp=none", str(bundle)], check=False, capture_output=True)
    verify = subprocess.run(["/usr/bin/codesign", "--verify", "--deep", "--strict", str(bundle)],
                            capture_output=True, text=True)
    if verify.returncode != 0:
        print(f"  WARNING: the installer is not correctly signed: {verify.stderr.strip()[:160]}")
    return bundle


def build(app: Path, token: str, out: Path, version: str) -> Path:
    staging = Path(tempfile.mkdtemp(prefix="sg-installer-")) / APP_NAME
    staging.mkdir(parents=True)

    print("  copying the app")
    subprocess.run(["/usr/bin/ditto", str(app), str(staging / app.name)], check=True)

    build_installer_app(staging, token, version)
    (staging / "Read me first.txt").write_text(READ_ME)

    out.unlink(missing_ok=True)
    print("  building the disk image")
    subprocess.run(
        ["/usr/bin/hdiutil", "create", "-volname", APP_NAME, "-srcfolder", str(staging),
         "-ov", "-format", "UDZO", "-quiet", str(out)],
        check=True,
    )
    shutil.rmtree(staging.parent, ignore_errors=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--app", type=Path, default=DIST / f"{APP_NAME}.app")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--token", default="",
                    help="optional: only needed if the release repo is private. The "
                         "installer then contains a credential — delete it after use.")
    args = ap.parse_args()

    if not args.app.exists():
        raise SystemExit(f"No app at {args.app}. Build one first: make app VERSION=x.y.z")

    version = plistlib.loads((args.app / "Contents" / "Info.plist").read_bytes()).get(
        "CFBundleShortVersionString", "0.0.0")
    out = args.out or DIST / f"SurfaceGuard-{version}-Installer.dmg"
    print(f"Building the installer for {APP_NAME} {version}")
    image = build(args.app, args.token, out, version)

    size = image.stat().st_size / 1e6
    print(f"\n  {image}  ({size:.0f} MB)")
    if args.token:
        print("  NOTE: this image contains your update token. Delete it after use.")
    print("\n  AirDrop that one file. She right-clicks 'Install Surface Guard',")
    print("  chooses Open, clicks Install, and that is the whole install.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
