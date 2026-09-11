# Surface Guard

Choose a surface in the camera view, turn protection on, and forget about it.

A desktop app that watches a camera, notices when a cat is standing on a surface
you have drawn — a counter, a table, a shelf — and plays a short deterrent. It
runs from the menu bar, needs no terminal after installation, and says plainly
when it is *not* protecting.

The desktop experience is organized around Home, Rooms, Detection, Audio,
Camera, Devices, and Settings. See the [UI audit](docs/UI_AUDIT.md) for the
implemented visualizer mapping and the capabilities intentionally kept out of the
interface until a real backend exists for them.

The design rationale lives in the design doc. The short version of the two
decisions that shape everything else:

- **Surfaces are stored in room coordinates, not frame pixels.** Every frame is
  located in a stitched panorama of the room before anything is tested against a
  surface. A polygon therefore survives the camera turning, being bumped, or
  landing 30 px off when it returns to a preset.
- **The displayed state is derived from a heartbeat.** There is no code path that
  shows "Protecting" without a self-check having just passed. Silent failure is
  the only bug that fully breaks the promise in the first line.

## Putting it on someone else's Mac

See **[docs/INSTALL.md](docs/INSTALL.md)** for the full runbook: what to
build, how to get it past Gatekeeper, the Local Network permission macOS
requires before the camera can be found, and what she needs to hand.

## Install (for development)

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[detect,dev]"
```

## Try it without a camera

The synthetic room is a real, feature-rich panorama with a virtual pan/tilt
camera, backlash and sensor noise. The whole pipeline runs against it.

```bash
.venv/bin/python -m surfaceguard.app --demo
```

## Run it for real

```bash
.venv/bin/python -m surfaceguard.app --model models/yolov8n.onnx
```

Setup is guided: connect the camera, confirm the view, let it look around the
room, check you can hear the sound, draw your first surface.

Video comes from [eufy-security-ws](https://github.com/bropat/eufy-security-ws)
for Eufy cameras, or any RTSP URL. Detection needs a YOLOv8 ONNX export in
`models/`; without one the app still installs, opens and runs setup, and says in
Diagnostics that nothing will be detected.

## Phase 0: is this feasible on your camera?

Run this **before** relying on the app. It measures, on your actual camera and
firmware, the things that can invalidate the design — then writes a report with a
GO / NO-GO verdict.

```bash
.venv/bin/python tools/phase0.py --source eufy --url ws://127.0.0.1:3000
```

It answers: what the camera can actually do, how long the stream takes to start
and whether it stays up, how far off a return-to-position lands, whether
registration works across the whole pan range and how fast, whether angle hints
matter, whether the camera's own speaker is reachable, end-to-end latency, and
recovery after an interruption.

## Shipping it to someone else's Mac

Everything the app needs is inside the bundle. There is no Docker, no Node
install, no terminal, and no `ffmpeg` on the target machine.

### One-time setup on the build machine

```bash
python packaging/fetch_runtime.py
```

Downloads the self-contained Node runtime from nodejs.org (verified against
Node's published SHA-256 sums) and `npm install`s the Eufy bridge into
`runtime/`. Homebrew's node links ~20 Homebrew dylibs and cannot be bundled.

### Build and publish

```bash
python packaging/make_release.py --version 0.2.0 --with-onnx
```

Builds the `.app`, signs it, zips it with `ditto`, verifies the signature
survives the round trip, signs a manifest with an Ed25519 key from your Keychain,
and publishes the release to GitHub. `--show-key` prints the public half;
`--no-upload` stops before publishing.

The signing key is generated on first use and stored in your Keychain. **Back it
up.** Losing it means every shipped app will reject all future updates, and the
only fix is reinstalling by hand.

### Signing

The build picks the best identity available, in this order:

1. **Developer ID Application** — notarisable, no Gatekeeper warning
2. **Apple Development** — what you have now
3. **ad-hoc**

Apple Development is preferred over ad-hoc even though it buys nothing from
Gatekeeper, because it is a *stable* identity across builds. An ad-hoc signature
changes every build, which makes macOS treat each update as a different app.

Without a Developer ID, her first launch needs a one-time bypass:
right-click the app → **Open** → **Open**, or `xattr -dr com.apple.quarantine`
when you install it. After that, updates install silently — files fetched by
Python are not quarantined the way browser downloads are.

Install it to **`~/Applications`**, not `/Applications`. The app replaces its own
bundle when updating, which needs write access to the containing folder.

### Updates

The app checks every 6 hours and verifies twice: the download's SHA-256 must
match the manifest, and the manifest must carry a valid Ed25519 signature from
your key. Whoever controls the release host still cannot put code on her Mac.

Her Mac holds a read-only fine-grained GitHub token in the Keychain, pasted into
**Settings → Automatic updates**. Because a token that expires would stop updates
with no visible symptom, it is re-checked **daily** and GitHub's own expiry header
is used to warn 14 days ahead. A dead token shows in Settings and in the app's
self-check — as a warning, not a failure, since updates being stuck does not stop
the app guarding.

### When it breaks and you are not there

A packaged app has nowhere to print, so everything goes to
`~/Library/Logs/SurfaceGuard/`. In **Diagnostics** there are two buttons:
**Copy diagnostics** puts version, camera state, self-check results, metrics and
the last log lines on the clipboard for her to paste to you — with no passwords or
tokens in it — and **Show log files** opens the folder.

### Checking a build really works

```bash
"dist/Surface Guard.app/Contents/MacOS/Surface Guard" --demo --selftest /tmp/check.png
```

Starts the packaged app, renders the window, saves a screenshot and exits non-zero
if it drew nothing. A frozen app that starts but renders nothing looks identical
to a healthy one from the outside, so this is checked rather than assumed.

## Connecting her camera

**Settings → Sign in to the camera account** asks for the Eufy email, password and
region, handles two-factor codes and captchas, then lists the cameras on the
account so she can pick one.

The password goes straight to the macOS Keychain and never into the app's own
files. The bridge needs it in a config file — it reads no environment variables —
so that file is written `0600` and deleted the moment the bridge has read it,
which is about two seconds per start.

Secrets are read through `/usr/bin/security` rather than directly, because the
Keychain grants access per accessing binary; reading it from the app itself would
re-prompt after every update.

## The weekly review

Once a week — and only when there is something worth asking about — the home
screen offers a one-minute session where the app asks to be graded. It borrows a
CAPTCHA's interaction (quick image judgements, batched, almost no reading) and
inverts the power dynamic: it blocks nothing, never suspends protection, and
quitting halfway keeps every answer already given. The user is the examiner.

The deck is sampled where an answer changes a decision, not at random: alerts that
played a sound, near-misses a gate only just blocked, and detections somewhere the
app had never seen anything before. Repeats in the same place are clustered into
one card, so fourteen 3 a.m. false alarms on the same shelf are one question.

It is honest about what a label can do. The detection model is a frozen ONNX
export and nothing here retrains it. What answers actually move:

| What you say | What changes |
| --- | --- |
| Repeated false alarms in one place | a blind spot on that surface, in map coordinates, optionally after dark only |
| "You missed one", blocked on size | that surface's size tolerance widens |
| "You missed one" somewhere you'd marked | the blind spot is taken back off |
| Consistent agreement on a calibrated surface | its size check tightens |
| Anything the detector simply never saw | nothing — it says so, and keeps the case to replay |

Every change is a record with the value it replaced, shown on a summary screen
with its own undo. An adjustment that could not be named and taken back would be
the silent failure this app exists to avoid, wearing a different hat.

Sampling, clustering and the adjustments live in `storage/review_policy.py` and
import no Qt, so they are decidable — and tested — from an activity log alone.

## Tests

```bash
.venv/bin/python -m pytest -q
```

Everything runs against the synthetic room, so there is no hardware in the loop.

## Layout

```
src/surfaceguard/
├── engine.py            one warm loop: locate, detect, gate, decide
├── state.py             the FSM the UI renders from
├── camera/
│   ├── registration.py  frame -> pose in map space (features + angle prior)
│   ├── panorama.py      sweep and stitch the room map
│   ├── scan_scheduler.py weighted pan duty cycle, and when not to pan
│   └── sources/         eufy_bridge · rtsp · replay · synthetic
├── geometry/
│   ├── surface.py       map polygon, plane homography, size model
│   └── projection.py    map <-> frame, visible fraction
├── detection/
│   ├── cat_detector.py  ONNX and synthetic backends
│   ├── gates.py         the independent, individually-logged checks
│   └── trigger_policy.py dwell, hysteresis, cooldown, schedule
├── audio/player.py      synthesised cues, pre-opened device
├── health/heartbeat.py  the self-test behind "Protecting"
├── bridge/
│   ├── supervisor.py    owns the bundled Node bridge process
│   ├── client.py        the eufy-security-ws protocol
│   └── credentials.py   Keychain, via /usr/bin/security
├── update/
│   ├── updater.py       check · verify · stage · swap · relaunch
│   ├── manifest.py      Ed25519-signed release descriptions
│   └── access.py        GitHub, and whether the token still works
├── logging_setup.py     file logging; a packaged app cannot print
├── storage/             preferences · room map · activity log ·
│                        review_policy (what the weekly review asks and changes)
└── ui/                  onboarding · home · surface editor · activity ·
                         diagnostics · settings · connect · review
tools/phase0.py          feasibility harness
packaging/               fetch_runtime · build_app · make_release
runtime/                 bundled Node + Eufy bridge (not in git)
legacy/                  the original Raspberry Pi prototype
```

`camera/sources/replay.py` is not a test fixture. Clips the user marks "not a
cat" get replayed through the real pipeline, which is how thresholds get tuned on
evidence rather than on feeling.

## Not built, deliberately

The Pi prototype in `legacy/` drove a vibration-collar relay and a 25 kHz
ultrasonic piezo. Both are gone and staying gone: an automatic actuator on an
animal, fired by a detector with an unmeasured false-positive rate, is the one
component here that can do real harm — and the first misfire destroys trust in
everything else the app says. Laptop speakers cannot produce meaningful
ultrasonic output either. The deterrent is an ordinary audible cue, varied so the
cat does not learn one specific sound, and suppressed entirely while a person is
near the surface.
