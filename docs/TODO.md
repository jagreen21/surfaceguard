# What's left

Written 12 Sep 2026, at the end of a long debugging session. Ordered by what
actually blocks the product, not by effort.

Her laptop: 2020 i5 MacBook Pro, 4 cores, no Neural Engine, macOS 15.7.7.
Build: x86_64, yolov8n, cross-built under Rosetta. Current release: **1.1.21**.

---

## P0 — broken now

**1. "All systems online" while there is no camera.**
This violates the design's central invariant: no path may show healthy without a
passing self-check. The video check should fail the moment no frame has arrived
(age is infinite against a 6 s limit), so either the heartbeat is not running or
the shell's status display is not driven by it. Start at `app._refresh_slow` and
`shell.set_inventory` — the plural `ProductViewState` contract has singular
chokepoints there. An app that claims to protect while blind is worse than one
that admits it is broken.

**2. Camera stability is unconfirmed.**
1.1.21 fixed a regression I introduced (the watchdog killed every stream before
P2P could deliver its first frame). Whether the camera now stays up is unknown.
`grep "restarting the video stream"` in the app log distinguishes "watchdog firing
and failing" from "watchdog not firing" — different problems.

**3. A cat on the table does not trigger.**
Detection works; the cat is identified. Two candidate causes and 1.1.20 added
notes naming which: the frame could not be located in the room map (registration
failing, nothing downstream runs at all), or the paw point — the bottom-centre of
the box — fell outside the polygon even though the cat's body covered it. Check
Activity, or `grep "cat seen"` in the log.

---

## P1 — known faults

**4. Broad `except` blocks hide real failures.**
`add_note` shipped in 1.1.19 as dead code: the method did not exist, the
`AttributeError` was swallowed, and the release claimed a fix that did nothing.
Audit the bare `except Exception` sites; most should log at minimum.

**5. RSS is ~528 MB and unexplained.**
`docs/PERF.md` has the baseline. The frame ring buffer was fixed (100 MB → 11 MB)
but was never the cause — the bench does not construct an `Engine`, so it never
measured it. Candidates: the onnxruntime arena, OpenCV allocations, the synthetic
room image. Needs the real app profiled; `tools/bench.py` cannot answer it.

**6. Activity is too quiet in general.**
Two silent paths are patched. The principle is not: anything that stops a decision
should leave a trace.

**7. Registration reliability.**
The room scan completes now, but how reliably frames register afterwards has never
been measured on her camera. `tools/phase0.py --from-settings` reports inliers
across the pan range and has never been run against real hardware.

---

## P2 — the four workstreams

**A. Intel performance.** Done: ROI inference (2× magnification), motion gating
(52 % skipped), bounded inference threads, strip buffer, UI-thread work, response
delay off the engine thread. Not done: `bench.py` does not time UI paint, does not
sweep input sizes or fp16/int8, and has never run on her actual laptop — every
number in PERF.md is Rosetta on a 12-core M2 Max.

**B. Surface editor.** Untouched, and now urgent because the polygon may be why
nothing fires. No zoom, no pan, no undo, no rectangle tool. Hit testing uses
Manhattan distance and only tests the selected surface. `MapCanvas` reaches into
`self.parent()`. Vertex precision is roughly a hand-span of counter per pixel.

**C. Fine-tuning across two machines.** The training loop is good and should not be
rewritten. Unresolved: the data is on her Intel Mac, the compute is on the build
machine, and her pictures may not leave without consent. Pick consented export or
on-device tiny training, and write down why. Regardless: pin the holdout by event
id at label time (`dataset.split()` currently reshuffles a growing pool, so
validation events migrate into training), and stage the rollout with model version
per activity row and auto-revert.

**D. Tone library.** Four cues today. Wanted: 8–10 across startle transients, hiss
and FM tones, randomised per playback for anti-habituation, welfare caps enforced
in code, one honest line on the Audio page about effectiveness decaying. Plus an
FFT test per cue.

---

## Detection quality, ranked

1. **Object tracking.** ~30 lines of IoU tracking unlocks track-aggregated
   confidence, N-of-M dwell, a median-height scale gate that stops rejecting
   mid-leap, and track ids in the log.
2. **Event-level eval.** Nothing measures "did it beep correctly". `ReplayCamera` +
   stored strips + deterministic gates gives false alarms/day and time-to-sound
   p95. That should be the ship gate.
3. **The scale model is self-confirming.** `observe_height` is only called inside
   `_fire` under `if played`, so false positives that fire feed the size model.
   Feed it from review-confirmed verdicts instead.
4. **Night/IR is a separate domain.** `Frame` has no day/night field. Detect IR via
   channel correlation ≈ 1.0, separate thresholds, weight night frames in training.
5. **Thresholds are global constants** applied to every room.
6. **Verify CoreML actually partitions** before trusting `get_providers()`, then
   quantize.

---

## Deployment

- Not notarised. First install needs right-click → Open. A Developer ID
  (£79/$99 a year) removes that and is the only way to fix it.
- `tools/phase0.py` has never run against the real camera. It answers two things
  that change the design rather than the settings: whether the camera's own
  speaker can play the deterrent (which would take her laptop off the critical
  path), and real capture latency.
- The deterrent still comes from the laptop. Lid closed, muted, or in another room
  are all silent failures.
- Nothing has ever been benchmarked on her actual hardware.

---

## Things I got wrong, so nobody repeats them

- Assumed her Mac matched the build machine. Six builds were arm64 and could never
  have run.
- Blamed Local Network permission, 5 GHz band, and App Translocation in turn. The
  real causes were: `start_listening` must precede `driver.connect` or no events
  are delivered at all; `ditto` not `zipfile` for signed bundles; schema 13+
  returns device serials not objects.
- Attributed 528 MB to the ring buffer by reading code, one commit after writing a
  document about measuring instead of reading.
- Shipped `add_note` as dead code and wrote a release note claiming it worked.
- Made the camera worse with a watchdog whose timeout was shorter than P2P's
  startup.

The two tools that worked all week were **her log file** and **running the bridge
directly with its output unfiltered**. Reach for those on the second attempt, not
the sixth.
