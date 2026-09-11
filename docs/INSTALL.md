# Putting Surface Guard on her MacBook

Everything the app needs ships inside it: the camera service, Node, the detector,
the video decoder. **She installs nothing** — no Docker, no Homebrew, no terminal.

Read this in order. The first section is you, at your desk. The second is you, at
her Mac, once.

---

## 1. On your Mac

### 1.1 Make the update token

Updates come from this private repo, so her Mac needs a read-only token.

1. github.com → Settings → Developer settings → **Fine-grained tokens** → Generate new
2. Repository access: **Only select repositories** → `jagreen21/surfaceguard`
3. Permissions → Repository permissions → **Contents: Read-only**
4. Expiry: as long as GitHub allows. The app checks daily and warns 14 days out,
   but a rotation you have to remember is still a rotation.
5. Copy it. You will paste it once, on her Mac.

### 1.2 Set up the project environment

Once per machine. macOS has no bare `python`, and its `python3` has none of the
dependencies, so everything here runs the project's own interpreter — either
through `make`, or as `.venv/bin/python` explicitly.

```bash
make setup      # creates .venv and installs everything
make doctor     # checks this machine has what it needs, and says what is missing
```

If anything below ever fails confusingly, run `make doctor` first.

### 1.3 Fetch what gets bundled

Once per machine:

```bash
make runtime    # Node + the Eufy bridge, checksum-verified
make model      # the yolov8m detector (~99 MB)
```

### 1.4 Build and publish

```bash
make release VERSION=1.0.0
```

That builds the `.app`, signs it, zips it, checks the signature survives the round
trip, signs a manifest with an Ed25519 key from your Keychain, and publishes the
release.

> **Back up the signing key.** It is in your Keychain as *"Surface Guard — update
> signing key"*. Lose it and every installed copy will reject all future updates,
> and the only fix is reinstalling by hand.
> `python packaging/make_release.py --show-key` prints the public half.

### 1.5 Sanity-check the build

```bash
"dist/Surface Guard.app/Contents/MacOS/Surface Guard" --demo --selftest /tmp/check.png
```

Exits non-zero if the packaged app starts but renders nothing — which looks
identical to a healthy app from the outside.

---

## 2. Make the thing you AirDrop

```bash
make installer                      # or: make installer TOKEN=github_pat_...
```

That produces `dist/SurfaceGuard-<version>-Installer.dmg` — the app, a one-click
installer, and a short read-me. **AirDrop that one file.**

Passing `TOKEN=` stores the update token on her Mac during install, so updates work
without anyone touching Settings. The image then contains a credential: delete it
once she has it, and do not leave it in Downloads.

## 3. On her Mac, once

She opens the image and double-clicks **Install Surface Guard**.

macOS will refuse the first time — it cannot check an app that did not come from
the App Store. She right-clicks **Install Surface Guard**, chooses **Open**, then
**Open** again. Once, on a short shell script she can read, rather than on a 500 MB
application.

The installer then does the rest by itself:

- closes any running copy
- copies the app to `~/Applications` (the right place — it replaces its own bundle
  when updating, and `/Applications` would demand an admin password every time)
- clears the quarantine flag AirDrop sets, so the app itself never needs a bypass
- turns on start-at-login
- stores the update token, if you baked one in
- opens the app

### The two steps it cannot do

macOS does not allow either to be automated, and pretending otherwise is how
someone ends up staring at an app that silently never finds the camera.

**Allow Local Network.** macOS asks whether Surface Guard may find devices on the
local network. It must be allowed — the camera is on the Wi-Fi. If the prompt was
dismissed: System Settings → Privacy & Security → Local Network → enable Surface
Guard.

**Sign in to Eufy.** The password goes into the Keychain through the app's own
sign-in. An installer that asked for it would be teaching her to type her password
into whatever asks.

### Have these ready before she starts

- Her **Eufy account email and password** — the same as the Eufy app
- Her **region** (the account is region-locked; the wrong one fails confusingly)
- Access to that **email inbox**, in case Eufy sends a verification code

### Setup itself

Seven steps, or eight if Eufy asks for a code. The app counts them honestly.

1. **Connect your camera** — choose *A Eufy camera*
2. **Sign in to Eufy** — email, password, region
3. *(only if asked)* **One more check** — the emailed code, or a captcha
4. **Choose your camera** — the one pointing at the counter
5. **Is this the right camera?** — aim it now and leave it there
6. **Looking around the room** — it pans slowly and stitches one wide picture,
   about 15 seconds. Surfaces are drawn on this once and work from any angle.
7. **Check you can hear it** — turn the volume up now
8. **Draw your first surface** — click round the edge of the counter, press Enter,
   name it

Then **Turn protection on**.

## 4. What to tell her

- It lives in the **menu bar**. Closing the window does not stop it.
- The dot is the status: green guarding, grey off or paused, red **not** protecting.
- **Pause 30 minutes** exists for when people are over.
- It will ask her to **grade it** about once a week. Answering makes it better;
  ignoring it twice makes it stop asking.
- If it ever misbehaves: **Diagnostics → Copy diagnostics**, and paste that to you.
  It contains no passwords or tokens.

## 5. When something breaks and you are not there

Logs are at `~/Library/Logs/SurfaceGuard/`. **Diagnostics → Show log files** opens
the folder, **Copy diagnostics** puts a summary on the clipboard.

The app is built not to fail quietly: if video stops, the detector cannot run, the
sound device disappears, the camera view stops being recognised, or the camera
service dies, the headline changes to **"Not protecting"** with the reason and
something to do about it — and it pushes a notification rather than waiting to be
noticed.

## 6. Checking the camera properly (optional)

There is a harness that measures what casual use does not: how far the camera
lands from where it was asked to return, whether registration holds across the
*whole* pan range, whether angle hints matter, and recovery from a deliberately
interrupted stream.

It needs the app set up first — it uses that camera — so it is not a first step:

```bash
.venv/bin/python tools/phase0.py --from-settings
```

It writes `phase0-out/phase0-report.md` with a **GO / NO-GO** verdict.

You do not need this to install or run anything. Reach for it when something is
off and you want numbers instead of impressions, or if you want the two answers
that would change the design rather than the settings:

- **Is the camera's own speaker reachable?** If so the deterrent can play from the
  camera, and her laptop stops being on the critical path at all.
- **Is capture latency under a second?** Above that, a startle cue lands too late
  to mean anything, and that is a design problem rather than a tuning one.

Both also show up in the app's own **Diagnostics** screen once it is running, which
is usually enough.

## 7. Shipping her an update later

```bash
.venv/bin/python packaging/make_release.py --version 1.0.1 --with-onnx        # the whole app
.venv/bin/python packaging/make_release.py --version 1.1.0 --model models/yolov8m.onnx
```

The second ships **only the detector** — 99 MB instead of 245 MB, and it installs
without a restart, because nothing else changed. Her Mac checks every six hours and
verifies both the checksum and the signature before installing anything.
