# Performance baseline

Measured with `tools/bench.py` against the synthetic room, so there is no camera
in the loop and runs are comparable. p50/p95 in milliseconds per stage.

**Read this caveat before using any number below.** These were taken on an M2 Max
(12 cores) — the x86_64 rows through Rosetta. Her machine is a 2020 i5 with **4
cores and no Neural Engine**. The x86_64 numbers are the right *architecture* and
the wrong *machine*: treat them as a floor, not an estimate. Nothing here was run
on her laptop.

```
arch -x86_64 .venv-x86/bin/python tools/bench.py --frames 200 --model yolov8n.onnx
```

## x86_64, yolov8n — the build that ships to her

| variant | read | register | detect | gates | fps | skipped | magnif | RSS |
|---|---|---|---|---|---|---|---|---|
| full frame, every frame | 61.2 / 62.9 | 4.2 / 4.5 | 21.9 / 22.8 | 0.3 / 0.9 | 11.4 | 0% | 1.00 | 528 MB |
| ROI crop, every frame | 61.0 / 62.5 | 4.2 / 4.5 | 21.9 / 22.5 | 0.3 / 0.9 | 11.4 | 0% | 2.00 | 572 MB |
| ROI crop + motion gate | 79.5 / 87.0 | 4.5 / 4.8 | 22.3 / 23.6 | 0.3 / 0.8 | 11.4 | **52%** | 2.00 | 589 MB |

## x86_64, yolov8m — for comparison only; not shipped to Intel

| variant | detect | RSS |
|---|---|---|
| full frame | 52.4 / 54.0 | 802 MB |
| ROI + motion | 53.3 / 56.0 | 1086 MB |

## arm64, yolov8m — the build machine

| variant | detect | RSS |
|---|---|---|
| full frame | 25.3 / 26.2 | 1020 MB |
| ROI + motion | 25.9 / 26.9 | 1135 MB |

## What the numbers say

**ROI does not make inference faster, and was never going to.** 21.9 ms with and
without the crop. The network's cost is set by its *input* size, and both the
crop and the full frame are letterbox-padded to the same 640 square — so the work
is identical. The brief expected ROI to buy time; it buys **resolution**: 2×
magnification, a 70 px cat reaching the network at ~62 px instead of ~23. That is
what lets a small model stay viable, but the saving shows up as detections that
land, not as milliseconds.

**Motion gating is where the time actually goes.** 52% of frames skipped in a room
with one moving cat, and a real room is still far more of the time than the
synthetic one. That is the change that keeps the fans quiet.

**yolov8n is 2.4× faster than yolov8m on x86_64** (21.9 vs 52.4 ms), and 2× slower
on x86_64 than arm64 (21.9 vs ~9 for n). Shipping n to Intel was right.

**RSS is the outstanding problem: 528–589 MB against a 400 MB target.** The frame
ring buffer copies every frame at full resolution (`engine._remember_frame`), and
at 1080p that is ~6 MB a copy, 16 deep. The strip is displayed at 640 px wide.
This is the next thing to fix and the bench now justifies it.

**`read` at 61 ms is the synthetic camera's simulated latency**, not a real cost —
it is what holds fps near 11 in every row. Ignore it when comparing variants; it
is constant across them.

## Not yet measured

- Her actual laptop. Everything above is Rosetta on a 12-core machine.
- Input sizes other than 640, and fp16/int8 variants.
- UI paint cost, which the brief asked for and this harness does not yet time.
