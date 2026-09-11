#!/bin/bash
# Surface Guard installer.
#
# Runs as the executable of a signed .app bundle, not as a loose .command file: a
# shell script cannot be code-signed, and macOS refuses to run an unsigned script
# delivered by AirDrop at all.
#
# Two things this file has already got wrong once each, both preserved as comments
# where they bit:
#   * Surface Guard lives INSIDE this bundle, never beside it (App Translocation).
#   * Nothing waits on a dialog to decide whether to continue (silent exits).
set -u

TITLE="Surface Guard"
APP="Surface Guard.app"
DEST="$HOME/Applications"
AGENT="$HOME/Library/LaunchAgents/com.surfaceguard.app.plist"
LOG="$HOME/Library/Logs/surfaceguard-install.log"

# ---------------------------------------------------------------- functions
# Defined before anything calls them. An earlier version called log() three lines
# above its own definition, so the first thing the installer did was fail quietly.

mkdir -p "$(dirname "$LOG")" 2>/dev/null
exec 2>>"$LOG"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*" >>"$LOG"; }

# "activate" first, or the dialog opens behind whatever she is looking at.
dialog() {
  osascript -e 'activate' \
            -e "display dialog \"$1\" buttons {\"OK\"} default button 1 with title \"$TITLE\"" \
            >/dev/null 2>&1
}

say() { log "dialog: $1"; dialog "$1" & }

fail() {
  log "FAILED: $1"
  dialog "$1"
  exit 1
}

# ------------------------------------------------------------------- locate
# Surface Guard is inside this bundle. Looking for a sibling does not work: macOS
# App Translocation runs a quarantined app from a randomised read-only copy, so at
# runtime the installer's folder holds only itself, and the file plainly next to it
# in Finder is not next to it on disk.

RESOURCES="$(cd "$(dirname "$0")/../Resources" 2>/dev/null && pwd)" || RESOURCES=""
SRC="$RESOURCES/$APP"
if [ ! -d "$SRC" ]; then
  BESIDE="$(cd "$(dirname "$0")/../../.." 2>/dev/null && pwd)" || BESIDE=""
  [ -n "$BESIDE" ] && [ -d "$BESIDE/$APP" ] && SRC="$BESIDE/$APP"
fi

log "=== installer starting"
log "argv0  : $0"
log "source : $SRC"
log "dest   : $DEST/$APP"

[ -d "$SRC" ] || fail "This installer is damaged — Surface Guard is missing from inside it."

# ------------------------------------------------------------------ install
# No "are you sure" dialog. Opening an installer is already that answer, and the
# dialog could only ever stop the install: osascript's dialog belongs to a
# background process, so anything but a literal click on Install — a timeout, a
# dismissal, never seeing it — fell through to a silent exit with no trace.

# ---------------------------------------------------------------- clean up
# Everything a previous install could have left behind. Five rounds of debugging
# left orphaned bridges holding ports 3050 to 3054 on this machine, and copies of
# the app in more than one folder; an installer that only writes the new version
# leaves all of that running.

log "closing any running copy"
# The app itself, wherever it was launched from...
pkill -f "Surface Guard.app/Contents/MacOS/Surface Guard" >/dev/null 2>&1
# ...and any bridge it left behind. Matched on the bundled path so nothing else
# node-related on her Mac is touched.
pkill -f "Surface Guard.app/Contents/.*runtime/node/bin/node" >/dev/null 2>&1
sleep 2

log "removing older copies"
for OLD in "$HOME/Applications/$APP" "/Applications/$APP" "$HOME/Desktop/$APP" \
           "$HOME/Downloads/$APP" "$HOME/Applications/$APP.old" "/Applications/$APP.old"; do
  if [ -e "$OLD" ]; then
    if rm -rf "$OLD" 2>/dev/null; then
      log "  removed $OLD"
    else
      # /Applications may need an admin password; not worth demanding one, but say so.
      log "  could not remove $OLD (permission denied)"
    fi
  fi
done

# A launch agent from an older install points at a path that may no longer exist.
launchctl unload "$AGENT" >/dev/null 2>&1
rm -f "$AGENT"

log "copying"
mkdir -p "$DEST" || fail "Could not create $DEST."
if ! /usr/bin/ditto "$SRC" "$DEST/$APP"; then
  fail "Copying Surface Guard failed. See $LOG."
fi
log "copied"

# AirDrop marks everything it delivers as untrusted. Clearing it here is why the
# app itself never needs a right-click.
log "clearing the quarantine flag"
xattr -dr com.apple.quarantine "$DEST/$APP" >/dev/null 2>&1

log "writing the launch agent"
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
# Deliberately not "launchctl load" here. RunAtLoad means loading it *starts* the
# app, and the next thing this script does is open it — which produced two running
# copies, two bridges, and two concurrent logins to the same Eufy account. Writing
# the file is enough: it takes effect at her next login, and we open one copy now.
launchctl unload "$AGENT" >/dev/null 2>&1

__TOKEN_BLOCK__

log "opening the app"
open "$DEST/$APP" || fail "Surface Guard was installed but would not open."

log "=== install complete"
say "Surface Guard is installed and opening.\n\nTwo things only you can do:\n\n1.  When macOS asks about finding devices on the local network, choose Allow. The camera is on the Wi-Fi and cannot be found without it.\n\n2.  Sign in with your Eufy account when the app asks — the same email and password as the Eufy app."
exit 0
