#!/usr/bin/env python3
"""Build the thing you AirDrop: the app plus a one-click installer.

    .venv/bin/python packaging/make_installer.py
    .venv/bin/python packaging/make_installer.py --token github_pat_...

Everything in section 2 of docs/INSTALL.md that a script *can* do, this does:
copies the app to ~/Applications, clears the quarantine flag AirDrop sets, turns on
start-at-login, optionally stores the update token, and opens the app.

Two steps stay manual because macOS will not let them be otherwise, and pretending
they are automated is how someone ends up staring at an app that silently never
finds the camera:

  * **Local Network permission** — an OS prompt. She has to click Allow.
  * **Her Eufy password** — it goes into the Keychain via the app's own sign-in.
    An installer that asked for it would be teaching her to type her password into
    whatever asks.

The output is a .dmg. The installer inside is a readable shell script, deliberately:
the one thing she has to bypass Gatekeeper for should be something she can look at.
"""

from __future__ import annotations

import argparse
import plistlib
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

INSTALLER = r"""#!/bin/bash
# Surface Guard installer. Everything it does is visible below.
set -u
cd "$(dirname "$0")"

APP="Surface Guard.app"
DEST="$HOME/Applications"
AGENT="$HOME/Library/LaunchAgents/com.surfaceguard.app.plist"

printf '\n  Installing Surface Guard\n\n'

if [ ! -d "$APP" ]; then
  printf '  Could not find %s next to this installer.\n' "$APP"
  printf '  Keep both files together and try again.\n\n'
  read -r -p '  Press return to close. ' _; exit 1
fi

# An older copy may be running; it cannot be replaced while it is.
if pgrep -f "$DEST/$APP/Contents/MacOS" >/dev/null 2>&1; then
  printf '  Closing the running copy...\n'
  pkill -f "$DEST/$APP/Contents/MacOS" >/dev/null 2>&1
  sleep 2
fi

printf '  Copying to %s\n' "$DEST"
mkdir -p "$DEST"
rm -rf "$DEST/$APP.old"
[ -d "$DEST/$APP" ] && mv "$DEST/$APP" "$DEST/$APP.old"
if ! cp -R "$APP" "$DEST/"; then
  printf '  Copy failed.\n'
  [ -d "$DEST/$APP.old" ] && mv "$DEST/$APP.old" "$DEST/$APP"
  read -r -p '  Press return to close. ' _; exit 1
fi
rm -rf "$DEST/$APP.old"

# AirDrop marks everything it receives as untrusted. Clearing it here is why the
# app itself does not need a right-click to open.
printf '  Clearing the download flag\n'
xattr -dr com.apple.quarantine "$DEST/$APP" 2>/dev/null

printf '  Turning on start-at-login\n'
mkdir -p "$(dirname "$AGENT")"
cat > "$AGENT" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.surfaceguard.app</string>
  <key>ProgramArguments</key>
  <array>
    <string>$DEST/$APP/Contents/MacOS/Surface Guard</string>
    <string>--background</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><false/>
  <key>WorkingDirectory</key><string>$HOME</string>
</dict></plist>
PLIST
launchctl unload "$AGENT" >/dev/null 2>&1
launchctl load "$AGENT" >/dev/null 2>&1

__TOKEN_BLOCK__

printf '  Starting Surface Guard\n\n'
open "$DEST/$APP"

printf '  Done. Two things only you can do:\n\n'
printf '    1. When macOS asks about finding devices on the local network,\n'
printf '       choose Allow. The camera is on the Wi-Fi and cannot be found\n'
printf '       without it.\n\n'
printf '    2. Sign in with your Eufy account when the app asks. It is the\n'
printf '       same email and password you use in the Eufy app.\n\n'
read -r -p '  Press return to close this window. ' _
"""

TOKEN_BLOCK = r"""printf '  Storing the update key\n'
/usr/bin/security add-generic-password -a update-token -s SurfaceGuard \
  -l 'Surface Guard - update access token' -U -w '__TOKEN__' >/dev/null 2>&1
"""

READ_ME = """Surface Guard

1. Double-click "Install Surface Guard".

   macOS will refuse the first time and say it cannot check it for malicious
   software. That is expected - it means the app was not bought from the App
   Store. Right-click "Install Surface Guard" instead, choose Open, then Open
   again. You only do this once.

2. Follow what it prints. It takes a few seconds.

3. Say Allow when macOS asks about finding devices on the local network.
   The camera is on the Wi-Fi; without this the app cannot find it.

4. Sign in with the Eufy account - the same one as the Eufy app.

That's it.
"""


def build(app: Path, token: str, out: Path) -> Path:
    staging = Path(tempfile.mkdtemp(prefix="sg-installer-")) / APP_NAME
    staging.mkdir(parents=True)

    print("  copying the app")
    subprocess.run(["/usr/bin/ditto", str(app), str(staging / app.name)], check=True)

    script = INSTALLER.replace(
        "__TOKEN_BLOCK__",
        TOKEN_BLOCK.replace("__TOKEN__", token) if token else
        "# no update token was baked in; it can be pasted in Settings later",
    )
    installer = staging / "Install Surface Guard.command"
    installer.write_text(script)
    installer.chmod(0o755)
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
                    help="GitHub update token to store on her Mac. The installer then "
                         "contains a credential — delete it after use.")
    args = ap.parse_args()

    if not args.app.exists():
        raise SystemExit(f"No app at {args.app}. Build one first: make app VERSION=x.y.z")

    version = plistlib.loads((args.app / "Contents" / "Info.plist").read_bytes()).get(
        "CFBundleShortVersionString", "0.0.0")
    out = args.out or DIST / f"SurfaceGuard-{version}-Installer.dmg"
    print(f"Building the installer for {APP_NAME} {version}")
    image = build(args.app, args.token, out)

    size = image.stat().st_size / 1e6
    print(f"\n  {image}  ({size:.0f} MB)")
    if args.token:
        print("  NOTE: this image contains your update token. Delete it after use.")
    print("\n  AirDrop that one file. She opens it, right-clicks")
    print("  'Install Surface Guard', chooses Open, and that is the whole install.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
