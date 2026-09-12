# Surface Guard — agent brief

You are working in `~/Code/Test/catdetector`, a macOS menu-bar app (PySide6 + onnxruntime)
that watches a camera, decides whether a cat is standing on a user-drawn surface, and plays
a short deterrent. Read `README.md` and `docs/UI_AUDIT.md` before your first edit.

There are **four workstreams** below. Do them in the order given. After each one, stop, run
the suite, and report what changed with numbers — do not chain all four and present a single
diff.

---

## 0. The two machines (this shapes everything)

- **Build machine:** Apple Silicon Mac. Has MPS, CoreML/ANE, and the training toolchain.
- **Target machine:** an **Intel i5 MacBook Pro** — no Neural Engine, no MPS, no ANE.
  `packaging/build_app.py:171-174` already cross-builds x86_64 under Rosetta for it, and
  `runtime-x64/` holds its own Node runtime.

Almost every performance assumption in the code was measured on the build machine and is
wrong for the target. The comment in `detection/cat_detector.py` quoting
`n 9.2ms / s 12.0ms / m 17.7ms` is Apple-silicon-with-CoreML. On an Intel i5 with no ANE,
expect yolov8m at 640 to land somewhere in the **hundreds of milliseconds per frame**. The
app asks for 8 fps. It will not get it.

**Do not take my estimate on faith, and do not take the code comment on faith either.**
Your first task is to measure.

---

## Invariants you must not break

These are load-bearing design decisions, documented in the README and the module docstrings.
If a change would violate one, stop and say so instead of doing it.

1. **The displayed state is derived from a heartbeat.** There is no code path that may show
   "Protecting" without a self-check having passed (`state.py`). Performance work must not
   create a path where the detector is idle but the UI still claims protection.
2. **Surfaces live in map coordinates, not frame pixels.** A surface survives the camera
   turning. Any editor change keeps that.
3. **Her pictures do not leave her Mac** without an explicit, visible, per-export consent.
4. **No actuator on the animal.** Audible cue only — no ultrasonic claims, no collars, no
   escalation. See "Not built, deliberately" in the README.
5. **A detector that cannot detect is an error the heartbeat reports**, never a silent
   fallback (`load_detector` raises on purpose — keep it that way).

---

## Workstream A — make it viable on an Intel i5

### A0. Measure first (do this before changing anything)

Add `tools/bench.py` that runs on the target architecture and reports, as a table:

- per-stage wall time over ≥300 frames: `source.read`, `Registrar.register`,
  `detector.detect`, gate evaluation, UI paint;
- p50 / p95 for each, plus achieved end-to-end fps;
- the same for each of `yolov8n/s/m` × input size `{320, 416, 512, 640}` × `{fp32, fp16, int8}`;
- RSS at steady state.

It must be runnable as `arch -x86_64 .venv-x86/bin/python tools/bench.py` so the numbers come
from the architecture that matters. Commit the baseline table into `docs/PERF.md`. Every
change below is justified against that table or it does not land.

### A1. Stop shipping yolov8m to the Intel build

`MODEL_FILENAMES = ("yolov8m.onnx", "yolov8s.onnx", "yolov8n.onnx")` with
`MODEL_FILENAME = MODEL_FILENAMES[0]` means the 104 MB fp32 model is the preferred bundle
everywhere. Make model selection a **build-target decision**, not a global constant:
`packaging/build_app.py --arch x86_64` bundles the model the bench says clears the budget on
Intel, and `export_model.py` gains `--imgsz` / `--weights` wiring so both variants can be
produced from one command. Record which model shipped in `build_info` and show it in
Diagnostics.

### A2. Region-of-interest inference

`engine.py:217` runs the detector on the **full frame**, letterboxed to 640. From 1080p that
is a 3× downscale — a 70 px cat reaches the network as ~23 px.

The app already knows where to look: `surface.project(pose)` gives the polygon in frame
pixels, and `surface.expected_height(pose, paw)` predicts how tall a cat standing there
should appear. Crop the union of projected surface bounds, pad by one predicted cat height
plus the `PERSON_PROXIMITY` margin, and run the model on that crop.

This is the single biggest change in this brief: on Intel it lets you drop to a smaller model
and a smaller input size *without* losing small-object recall, because the target now fills
far more of the input. Keep a lower-cadence full-frame pass for the `no_person` gate — a
person approaching from off-surface still has to be seen.

### A3. Motion-gate the detector

The loop runs the model on every frame regardless of whether anything moved, while
`_hold_awake` keeps `caffeinate -i` running. On a fanned Intel laptop that is heat, fan noise
and battery, permanently.

`registration._to_work()` already produces a 480 px greyscale of every frame. Difference
consecutive ones **inside the projected surface bounds** and skip inference when nothing
changed. Then invert the budget: when there *is* motion, raise the frame rate. Lower
time-to-sound when it matters, near-idle when it doesn't.

Guard rail per invariant 1: "nothing moved" and "we stopped looking" must stay distinguishable
to the heartbeat.

### A4. Stop copying every frame three times

- `engine._remember_frame` does `image.copy()` into a 16-deep deque on **every frame**. At
  1080p that is ~6 MB per copy and ~100 MB resident. Store JPEG-encoded or half-scale frames —
  the strip is shown at 640 px wide in the review anyway (`activity_log.REVIEW_WIDTH`).
- `app.py::_on_frame` calls `update_result` on **both** `home.live` and `camera_screen.live`,
  so `qtutil.bgr_to_pixmap` runs twice per frame regardless of which page is visible. Update
  only the visible one.
- `bgr_to_pixmap` itself copies twice: `np.ascontiguousarray(image[:, :, ::-1])` and then
  `QImage(...).copy()`. Convert once, into a reused buffer, at display resolution rather than
  source resolution.

### A5. Stop re-polishing stylesheets 8× a second

`_fast` fires every 120 ms. `HomeScreen.render_state` calls `self.dot.setStyleSheet(...)`
unconditionally, and `DetectionScreen.render_state` calls `StatusPill.set_status`, which does
`style().unpolish()` + `polish()`. Qt style recomputation is expensive and this runs whether
or not anything changed. Make every one of these setters a no-op when the value is unchanged,
and drop `_fast` to ~250 ms — nothing on screen needs 8 Hz except the video, which is driven
by `on_frame`, not by the timer.

### A6. Bound the thread pools

Nothing in the codebase sets `ort.SessionOptions`, `cv2.setNumThreads`, or any OMP variable.
onnxruntime defaults to all cores; OpenCV does too; ORB registration, the Qt UI thread and
the Node bridge are all competing on a 4-core laptop. Set explicit intra-op/inter-op thread
counts and an OpenCV thread cap, tuned from the A0 bench, and expose them in Diagnostics.

### A7. `_fire` must not sleep on the engine thread

`engine._fire` calls `time.sleep(min(delay, 5.0))`. For the duration of a user-configured
response delay the loop reads no frames, runs no registration, and fills no strip buffer —
and the latency it records immediately afterwards *includes the sleep*. Schedule the sound
instead; measure latency to the decision, and report the configured delay separately.

**Acceptance for Workstream A:** on the x86_64 build, sustained ≥8 fps effective judging
during motion, <15% CPU when the room is still, steady-state RSS under 400 MB, and `docs/PERF.md`
showing before/after from the same bench.

---

## Workstream B — surface selection UX

Read `ui/surface_editor.py` first. Current behaviour: the stitched room panorama is fit whole
into the widget with no pan or zoom; clicks append polygon points; double-click or Enter
commits; the selected surface's vertices can be dragged; Backspace removes the last point
while drawing. That is the whole interaction.

The problems, in order:

1. **No zoom, no pan.** The polygon *is* the thing the gates test, and it is being drawn on a
   whole-room panorama scaled to fit a ~600 px widget. Vertex precision is roughly a hand-span
   of real counter per pixel. Add scroll-to-zoom, space-drag or two-finger pan, fit-to-window,
   and a **magnifier loupe** near the cursor while placing or dragging a vertex.
2. **Rectangles are the common case and there is no rectangle tool.** Counters, tables, desks
   and shelves are rectangles. Worse, the plane model *wants exactly four corners*:
   `Surface.quad` falls back to `_extreme_quad()` for any polygon with more than four points,
   silently degrading the scale gate. Offer a **4-corner quad mode as the default** — click
   four corners, with a perspective-correct preview of the rectified surface so the user can
   see they got the corners right — and keep freeform polygon as a secondary mode that warns
   it will approximate the plane.
3. **No undo.** Delete is a confirmation dialog; a vertex drag commits on mouse release with
   no way back. Meanwhile the weekly review gives every adjustment its own undo. Add a proper
   `QUndoStack` for the editor: add point, move vertex, delete surface, rename.
4. **Cannot edit an existing shape properly.** No inserting a vertex into an edge, no deleting
   one. Add both (double-click an edge to insert; select a handle and press Delete).
5. **Hit testing is wrong and narrow.** `mousePressEvent` uses `manhattanLength() <= HANDLE_R * 2.4`
   (Manhattan, not Euclidean) and only tests handles on the *already selected* surface. Use
   Euclidean distance in device-independent pixels, and hit-test all visible surfaces.
6. **Brittle parent coupling.** `MapCanvas` calls `self.parent().commit_drawing()` and
   `self.parent().delete_selected()`. Replace with signals.
7. **No feedback about whether the shape will actually work.** While drawing, show live: the
   surface's projected visibility in the current camera view, and its scan coverage from
   `engine.build_plan().coverage()` — which is already computed and currently thrown away.
   A surface the camera can never reach should say so *at the moment it is drawn*, not silently
   report 0%.
8. **Naming happens at the wrong time.** A combo box appears beside the Add button before any
   shape exists, and focus jumps to a second name field after commit. Draw first, then name in
   one place, with the common-name suggestions as a picker.

**Acceptance for Workstream B:** a user can zoom to a counter edge, place four corners to
within a few map pixels, see the rectified preview, undo any step, reopen the surface later
and nudge one corner — all without a modal dialog, and with `tests/test_ui_states.py` extended
to cover the new modes.

---

## Workstream C — continued fine-tuning across two machines

The training loop in `src/surfaceguard/training/` is good and you should not rewrite it: hard
negatives from `not_a_cat`, event-level splits that prevent frame leakage, a frozen backbone,
a **required** general control set, and a ship gate that refuses a regression. Keep all of it.

There is one architectural problem to solve: **the data and the compute are on different
machines.**

- `tools/finetune.py` reads `ActivityLog()` — which lives on *her* Intel Mac, where training
  is not feasible (`training/finetune.py` selects `mps` if available, else `cpu`; on an Intel
  i5, a 24-epoch yolov8m fine-tune on CPU is not a thing that finishes).
- The build machine can train in minutes, and has no data.
- Invariant 3 says her pictures do not leave her Mac without consent.

Design and implement **one** of these, and write down why:

- **(a) Consented export.** A visible "Help improve detection" flow that exports only
  *reviewed* events — the thumbnails and strips already stored, plus their labels and gate
  results — as a single signed archive, with a preview of exactly what is in it and a count,
  and no way to export unreviewed frames. Train on the build machine, ship the model back
  through the existing signed-manifest update channel.
- **(b) On-device tiny training.** Restrict the target-machine path to yolov8n at 416 with the
  backbone frozen and a small head, run it overnight as a low-priority job, and accept that it
  only ever produces small corrections. Measure whether it clears `MIN_IMPROVEMENT` at all
  before committing to this.

Whichever you pick, also fix these two, which are wrong today regardless:

- **Pin the holdout.** `dataset.split()` reshuffles a *growing* pool with a fixed seed, so an
  event that was validation in one round can be training in the next. Record holdout membership
  by event id at the moment an event is first labelled, and never move it.
- **Stage the rollout.** Add the model version to every row in `activity_log`, keep the previous
  ONNX, and auto-revert if the false-alarm rate in the first N days exceeds what the ship gate
  predicted. A gate that fires once before release is an opinion; this makes it a safety net.

Note the licence comment in `packaging/export_model.py`: YOLOv8 is AGPL-3.0. If this is ever
distributed beyond personal use, that is a real constraint — flag it rather than ignoring it.

---

## Workstream D — a proper deterrent tone library

`audio/player.py` synthesises four cues: `chirp` (1.8→3.4 kHz sweep), `clack` (two noise
transients), `hiss` (high-passed noise), `warble` (2.6 kHz ±700 Hz at 11 Hz). The synthesis
framework is clean — `synthesise(name, seconds, seed)` returning mono float32, cached at init.
Extend it; do not replace it.

### What to build

A library of **8–10 cues grouped into families**, each with a plain-language name and a
one-line description of what it sounds like:

- **Startle transients** — short broadband snaps and knocks. Physical-sounding, least
  aversive, best first choice.
- **Hiss family** — shaped broadband noise with energy concentrated ~4–10 kHz, which is the
  band a cat is most sensitive in and which reads as a conspecific warning. Vary the shaping
  and duration across two or three variants.
- **Frequency-modulated tones** — sweeps and warbles in the 2–10 kHz range. Randomise the
  sweep direction, rate and centre frequency per playback so no two are identical.

### Constraints — these are not optional

- **No ultrasonic.** At 44.1 kHz the synthesis ceiling is 22 kHz, a laptop speaker rolls off
  hard well below that, and the README already ruled out the ultrasonic piezo for exactly this
  reason. Do not add a cue that claims a frequency the hardware cannot reproduce. If you raise
  `SAMPLE_RATE` to 48 kHz, first check the actual device capability with
  `sd.check_output_settings` and fall back cleanly.
- **Welfare caps, enforced in code, not in the UI.** Maximum duration per cue (keep under
  ~0.5 s), a hard ceiling on output gain, no sustained tones, and no repeat inside the
  surface's cooldown. The existing person-proximity suppression stays.
- **Anti-habituation is the real feature.** A cat habituates to a fixed sound within days.
  `Deterrent.vary_sound` already rotates cues via `engine._vary`; strengthen it to randomise
  *within* a cue too — seed, duration, centre frequency, and the gap between transients — and
  to avoid repeating the same family twice in a row.
- **Be honest in the copy.** No audible cue reliably repels every cat, and effectiveness
  decays. Say that once, plainly, on the Audio page. Do not write marketing.

### UI

Replace the current Audio page control set — three option buttons (`Chime` / `Voice —
unavailable` / `Custom Sound`) stacked above a separate "Chime style" combo, where choosing
Chime calls `_choose_chime()` and force-selects `"chirp"` regardless of the combo. That is two
controls fighting over one choice. Build one list: the cue families, each expandable to its
variants, each with a preview button, plus "Import a WAV…" at the bottom. Re-sync the selection
when an import is cancelled — today `import_sound` returns early without calling
`_refresh_audio()`, leaving the radio on "Custom Sound" while the sound never changed. Move
"Voice — unavailable" out of the primary list into a note.

Also place `AudioScreen.delay` — it is constructed, read in `_load_surface`, written in
`_push`, and **never added to any layout**, so the response-delay setting is currently
unreachable.

### Verify it

Add a test that renders every cue and asserts: peak amplitude within the cap, duration within
the cap, no DC offset, and spectral energy inside the intended band (FFT, not a listen). A
cue that aliases or clips is worse than no cue.

---

## How to work

- **Measure, then change, then measure.** Workstream A is meaningless without `docs/PERF.md`.
- **One workstream per branch, one logical change per commit.** Do not mix a perf change with
  a UX change.
- `make test` must pass at every commit. Everything runs against the synthetic room, so there
  is no hardware in the loop.
- After a UI change, run the packaged render check:
  `"dist/Surface Guard.app/Contents/MacOS/Surface Guard" --demo --selftest /tmp/check.png`
- **If you find that one of my claims above is wrong, say so and stop.** I read the source; I
  did not run the app. Every file and line reference above is from a static read and may have
  moved. The numbers in `cat_detector.py`'s comment are the author's, measured on the wrong
  machine, and the estimates in this brief are mine and unverified.
- Do not refactor for taste. Do not introduce a dependency without saying what it buys and what
  it costs in bundle size — the app ships as a signed .app with a bundled Node runtime and is
  already ~130 MB.
- Do not touch `bridge/credentials.py`, the updater's signature verification, or the
  Keychain path without flagging it explicitly first.

## Report back, per workstream

1. What you changed, as file paths.
2. The before/after numbers, from the same bench.
3. What you tried that did not work.
4. Anything in this brief you now believe is wrong.
