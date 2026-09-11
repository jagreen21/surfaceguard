# Surface Guard

Choose a surface in the camera view, turn protection on, and forget about it.

A desktop app that watches a camera, notices when a cat is standing on a surface
you have drawn — a counter, a table, a shelf — and plays a short deterrent. It
runs from the menu bar, needs no terminal after installation, and says plainly
when it is *not* protecting.

The design rationale lives in the design doc. The short version of the two
decisions that shape everything else:

- **Surfaces are stored in room coordinates, not frame pixels.** Every frame is
  located in a stitched panorama of the room before anything is tested against a
  surface. A polygon therefore survives the camera turning, being bumped, or
  landing 30 px off when it returns to a preset.
- **The displayed state is derived from a heartbeat.** There is no code path that
  shows "Protecting" without a self-check having just passed. Silent failure is
  the only bug that fully breaks the promise in the first line.

## Install

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
├── storage/             preferences · room map · activity log
└── ui/                  onboarding · home · surface editor · activity · diagnostics
tools/phase0.py          feasibility harness
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
