#!/bin/bash
# Surface Guard installer.
#
# This runs as the executable of a real .app bundle rather than as a loose
# .command file, because a shell script cannot be code-signed and macOS refuses to
# run unsigned scripts that arrived by AirDrop — "no usable signature", with no
# working way past it. A signed app bundle gets the ordinary unidentified-developer
# dialog that right-click -> Open actually clears.
#
# Dialogs come from osascript so there is no Terminal window, and the logic below
# is the same sequence that was already tested end to end.
set -u

TITLE="Surface Guard"
APP="Surface Guard.app"
DEST="$HOME/Applications"
AGENT="$HOME/Library/LaunchAgents/com.surfaceguard.app.plist"
LOG="/tmp/surfaceguard-install.log"

# $0 is .../Install Surface Guard.app/Contents/MacOS/install — the delivered
# folder is three levels up.
HERE="$(cd "$(dirname "$0")/../../.." && pwd)"
SRC="$HERE/$APP"

exec 2>>"$LOG"
echo "--- $(date) installing from $HERE" >>"$LOG"

say() {
  osascript -e "display dialog \"$1\" buttons {\"OK\"} default button 1 with title \"$TITLE\"" >/dev/null 2>&1
}
fail() {
  osascript -e "display dialog \"$1\" buttons {\"OK\"} default button 1 with icon stop with title \"$TITLE\"" >/dev/null 2>&1
  exit 1
}

if [ ! -d "$SRC" ]; then
  fail "Could not find $APP next to this installer.\n\nKeep both together and try again."
fi

ANSWER=$(osascript -e "display dialog \"This will install Surface Guard into your Applications folder and open it.\n\nIt takes a few seconds.\" buttons {\"Cancel\",\"Install\"} default button \"Install\" with title \"$TITLE\"" 2>/dev/null) || exit 0
case "$ANSWER" in *Install*) ;; *) exit 0 ;; esac

# A running copy cannot be replaced while it is running.
pkill -f "$DEST/$APP/Contents/MacOS" >/dev/null 2>&1
sleep 1

mkdir -p "$DEST" || fail "Could not create $DEST."
rm -rf "$DEST/$APP.old"
[ -d "$DEST/$APP" ] && mv "$DEST/$APP" "$DEST/$APP.old"
if ! /usr/bin/ditto "$SRC" "$DEST/$APP"; then
  [ -d "$DEST/$APP.old" ] && mv "$DEST/$APP.old" "$DEST/$APP"
  fail "Copying Surface Guard failed.\n\nSee $LOG."
fi
rm -rf "$DEST/$APP.old"

# AirDrop marks everything it delivers as untrusted. Clearing it here is why the
# app itself never needs a right-click.
xattr -dr com.apple.quarantine "$DEST/$APP" >/dev/null 2>&1

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
</dict></plist>
PLIST
launchctl unload "$AGENT" >/dev/null 2>&1
launchctl load "$AGENT" >/dev/null 2>&1

__TOKEN_BLOCK__

open "$DEST/$APP"

say "Surface Guard is installed and opening.\n\nTwo things only you can do:\n\n1.  When macOS asks about finding devices on the local network, choose Allow. The camera is on the Wi-Fi and cannot be found without it.\n\n2.  Sign in with your Eufy account when the app asks — the same email and password as the Eufy app."
exit 0
