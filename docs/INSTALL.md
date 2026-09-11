# Putting Surface Guard on her MacBook

Everything the app needs ships inside it: the camera service, Node, the detector,
the video decoder. **She installs nothing** — no Docker, no Homebrew, no terminal.

Read this in order. The first section is you, at your desk. The second is you, at
her Mac, once.

---

## 0. Before anything else: does her camera work?

This has never been tested against a real E30. Run it before you rely on any of
the rest — it takes about two minutes and can still invalidate the design.

You will need the bridge running against her Eufy account. The easiest way is to
finish setup once on **your** Mac (section 2), then:

```bash
.venv/bin/python tools/phase0.py --source eufy --url ws://127.0.0.1:3050
```

It writes `phase0-out/phase0-report.md` with a **GO / NO-GO** verdict covering
stream startup, registration across the pan range, return-to-position
repeatability, whether the camera's own speaker is reachable, end-to-end latency
and reconnection.

Two answers change the plan:

- **Camera speaker reachable?** If yes, the deterrent plays from the camera and
  her laptop stops mattering. If no, the sound comes from the MacBook, and a
  closed lid or a muted machine is a silent failure.
- **Capture latency.** If it is above about a second on its own, deterrence will
  not work and the design needs revisiting, not the UI.

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

## 2. On her Mac, once

Bring the `.app` on a USB stick, over AirDrop, or from the release page.

### 2.1 Put it in the right place

Drag it to **`~/Applications`** — her home folder, *not* `/Applications`.

This matters: the app replaces its own bundle when it updates, which needs write
access to the containing folder. In `/Applications` every update would ask for an
admin password, which is exactly the kind of thing nobody does.

### 2.2 Let it open

The app is signed but not notarised, so Gatekeeper will block the first launch.
Either:

- **Right-click the app → Open → Open** (the button only appears on right-click), or
- from your terminal: `xattr -dr com.apple.quarantine ~/Applications/"Surface Guard.app"`

Once only. Updates afterwards install silently, because files the app downloads
itself are not quarantined the way browser downloads are.

### 2.3 Say yes to Local Network

macOS will ask whether Surface Guard may find devices on the local network. **It
must be allowed** — the camera is on the Wi-Fi, and without this the app simply
never finds it and cannot say why.

If it was dismissed: System Settings → Privacy & Security → Local Network → enable
Surface Guard.

### 2.4 Have these ready before you start setup

- Her **Eufy account email and password** — the same ones she uses in the Eufy app
- Her **region** (the account is region-locked; the wrong one fails confusingly)
- Access to that **email inbox**, in case Eufy sends a verification code

### 2.5 Run setup

Seven steps, or eight if Eufy asks for a code. The app counts them honestly.

1. **Connect your camera** — choose *A Eufy camera*
2. **Sign in to Eufy** — email, password, region
3. *(only if asked)* **One more check** — the code Eufy emails, or a captcha
4. **Choose your camera** — the one pointing at the counter
5. **Is this the right camera?** — a still from it. Aim it now and leave it there.
6. **Looking around the room** — it pans slowly and stitches one wide picture,
   about 15 seconds. Surfaces get drawn on this once and work from any angle.
7. **Check you can hear it** — play the deterrent. Turn the volume up now.
8. **Draw your first surface** — click around the edge of the counter, press Enter,
   name it.

Then **Turn protection on**.

### 2.6 Finish the settings

In **Settings**:

- Paste the update token into *Automatic updates* and press **Check for updates now**.
  It should say *"Update access is working (N days until the token expires)"*.
- Tick **Open Surface Guard automatically when this Mac starts**.

---

## 3. What to tell her

- It lives in the **menu bar**. Closing the window does not stop it.
- The dot is the status: green guarding, grey off or paused, red **not** protecting.
- **Pause 30 minutes** exists for when people are over.
- It will ask her to **grade it** about once a week. Answering makes it better;
  ignoring it twice makes it stop asking.
- If it ever misbehaves: **Diagnostics → Copy diagnostics**, and paste that to you.
  It contains no passwords or tokens.

## 4. When something breaks and you are not there

Logs are at `~/Library/Logs/SurfaceGuard/`. **Diagnostics → Show log files** opens
the folder, **Copy diagnostics** puts a summary on the clipboard.

The app is built not to fail quietly: if video stops, the detector cannot run, the
sound device disappears, the camera view stops being recognised, or the camera
service dies, the headline changes to **"Not protecting"** with the reason and
something to do about it — and it pushes a notification rather than waiting to be
noticed.

## 5. Shipping her an update later

```bash
.venv/bin/python packaging/make_release.py --version 1.0.1 --with-onnx        # the whole app
.venv/bin/python packaging/make_release.py --version 1.1.0 --model models/yolov8m.onnx
```

The second ships **only the detector** — 99 MB instead of 245 MB, and it installs
without a restart, because nothing else changed. Her Mac checks every six hours and
verifies both the checksum and the signature before installing anything.
