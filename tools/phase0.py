#!/usr/bin/env python3
"""Phase 0 feasibility harness.

Answers, on the actual camera and firmware, the questions that decide whether the
Surface Guard design holds:

  1. What can this camera actually do? (PTZ, angles, speaker, pet events)
  2. How long does the stream take to start, and does it stay up?
  3. How far off does the camera land when asked to return to the same position?
  4. Does registration work across the whole pan range, and how fast?
  5. Do angle hints matter, or must registration run on features alone?
  6. Does the camera's own speaker work, removing the laptop from the path?
  7. What is end-to-end latency, paw-down to sound?
  8. Does it recover from the stream being interrupted?

Writes a markdown report and a JSON file of the raw numbers. No UI, no app state,
nothing that needs a person watching.

    python tools/phase0.py --source synthetic
    python tools/phase0.py --source eufy --url ws://127.0.0.1:3000 --serial T8417XXXXXXXX
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np

from surfaceguard.audio.player import Player
from surfaceguard.camera.panorama import StitchError, build_room_map
from surfaceguard.camera.registration import Registrar
from surfaceguard.camera.sources.base import CameraSource, SourceError
from surfaceguard.geometry.projection import transform_points

# Thresholds from the design doc's acceptance criteria.
TARGET_P95_LATENCY_MS = 1200.0
TARGET_MIN_INLIERS = 30
TARGET_RECONNECT_S = 30.0


@dataclass
class Finding:
    """One measured answer, with a verdict against the design's own target."""

    name: str
    value: str
    verdict: str = "info"     # pass | fail | warn | info
    detail: str = ""

    def line(self) -> str:
        mark = {"pass": "PASS", "fail": "FAIL", "warn": "WARN", "info": "  · "}[self.verdict]
        out = f"  [{mark}] {self.name}: {self.value}"
        return out + (f"\n          {self.detail}" if self.detail else "")


@dataclass
class Results:
    findings: list[Finding] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    def add(self, *findings: Finding) -> None:
        for f in findings:
            self.findings.append(f)
            print(f.line(), flush=True)

    @property
    def failures(self) -> list[Finding]:
        return [f for f in self.findings if f.verdict == "fail"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.verdict == "warn"]


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}", flush=True)


# --------------------------------------------------------------------- 1. caps


def probe_capabilities(source: CameraSource, r: Results) -> None:
    section("1. What this camera can do")
    caps = source.capabilities
    r.raw["capabilities"] = {k: str(v) for k, v in caps.as_rows()}
    for key, value in caps.as_rows():
        r.add(Finding(key, value))
    for note in caps.notes:
        r.add(Finding("note", note))

    if caps.has_ptz and not caps.reports_angles:
        r.add(Finding(
            "angle read-out", "not available", "warn",
            "Registration must work from image features alone; angle hints cannot be "
            "used to pick a keyframe, which costs time and accuracy (see section 5).",
        ))
    if not caps.has_speaker:
        r.add(Finding(
            "camera speaker", "not reachable", "warn",
            "The laptop stays on the critical path (risk R1): lid closed, muted or "
            "asleep are all silent failures.",
        ))


# ---------------------------------------------------------------- 2. stream up


def probe_stream(source: CameraSource, r: Results, seconds: float) -> list:
    section(f"2. Stream startup and stability ({seconds:.0f}s)")
    startup = getattr(source, "stream_startup_s", None)
    frames, gaps, first = [], [], None
    deadline = time.monotonic() + seconds
    last_at = None

    while time.monotonic() < deadline:
        frame = source.read(timeout=3.0)
        if frame is None:
            gaps.append(("timeout", time.monotonic()))
            continue
        if first is None:
            first = frame
        if last_at is not None:
            gaps.append(("gap", frame.ts_received - last_at))
        last_at = frame.ts_received
        frames.append(frame)

    if not frames:
        r.add(Finding("frames received", "0", "fail",
                      "No video at all — nothing else in this report can be trusted."))
        return frames

    intervals = [g for kind, g in gaps if kind == "gap"]
    fps = 1.0 / statistics.median(intervals) if intervals else 0.0
    r.add(Finding("frames received", str(len(frames))))
    r.add(Finding("frame size", f"{frames[0].size[0]}x{frames[0].size[1]}"))
    r.add(Finding("effective fps", f"{fps:.1f}"))
    if startup is not None:
        r.add(Finding(
            "stream startup", f"{startup:.2f} s",
            "pass" if startup < 3.0 else "warn",
            "This is why the stream is held warm rather than opened on an event (D6).",
        ))
    worst = max(intervals) if intervals else 0.0
    r.add(Finding("worst frame gap", f"{worst * 1000:.0f} ms",
                  "pass" if worst < 2.0 else "warn"))

    if frames[0].ts_capture is not None:
        lags = [(f.ts_received - f.ts_capture) * 1e3 for f in frames if f.ts_capture]
        r.add(Finding("capture-to-arrival", f"median {statistics.median(lags):.0f} ms",
                      "pass" if statistics.median(lags) < 700 else "warn",
                      "Dominant term in the §7 latency budget."))
    else:
        r.add(Finding(
            "capture-to-arrival", "not measurable", "warn",
            "The source relays no capture timestamp, so §7's largest budget line "
            "cannot be measured in-band. Section 7 measures it out of band instead.",
        ))

    r.raw["stream"] = {"frames": len(frames), "fps": round(fps, 2),
                       "startup_s": startup, "worst_gap_s": round(worst, 3)}
    return frames


# ------------------------------------------------------------------- 3. map


def probe_map(source: CameraSource, r: Results, registrar: Registrar):
    section("3. Building the room map")
    caps = source.capabilities
    lo, hi = caps.pan_range or (0.0, 0.0)
    positions = list(np.arange(lo + 5, hi - 4, 20.0)) if caps.has_ptz else [0.0]
    started = time.monotonic()
    try:
        room = build_room_map(
            source, pan_positions=positions, settle_s=1.2 if caps.has_ptz else 0.0,
            registrar=registrar,
            progress=lambda i, n: print(f"        sweep {i}/{n}", flush=True),
        )
    except (StitchError, SourceError) as exc:
        r.add(Finding("stitch", "failed", "fail", str(exc)))
        return None
    elapsed = time.monotonic() - started
    r.add(Finding("map size", f"{room.size[0]}x{room.size[1]} px"))
    r.add(Finding("keyframes", str(len(room.keyframes))))
    r.add(Finding("sweep time", f"{elapsed:.0f} s"))
    features = [kf.features for kf in room.keyframes]
    r.add(Finding("features per keyframe", f"min {min(features)}, median {int(statistics.median(features))}",
                  "pass" if min(features) >= 120 else "warn",
                  "A keyframe with few features is a view registration will struggle on."))
    r.raw["map"] = {"size": list(room.size), "keyframes": len(room.keyframes),
                    "sweep_s": round(elapsed, 1), "features": features}
    return room


# ------------------------------------------------------------ 4. registration


def probe_registration(source: CameraSource, room, registrar: Registrar, r: Results) -> None:
    section("4. Registration across the pan range")
    room.install_into(registrar)
    caps = source.capabilities
    lo, hi = caps.pan_range or (0.0, 0.0)
    targets = list(np.linspace(lo + 8, hi - 8, 9)) if caps.has_ptz else [0.0]

    rows = []
    for pan in targets:
        if caps.has_ptz:
            source.move_to(float(pan), settle_s=1.2)
        frame = source.read(timeout=4.0)
        if frame is None:
            rows.append({"pan": pan, "ok": False, "inliers": 0, "ms": 0.0, "reason": "no frame"})
            continue
        res = registrar.register(frame.image, pan=frame.pan, tilt=frame.tilt)
        rows.append({"pan": round(float(pan), 1), "ok": res.ok, "inliers": res.inliers,
                     "ms": round(res.elapsed_ms, 1), "keyframe": res.keyframe_id,
                     "reason": res.reason})
        print(f"        pan {pan:+7.1f}  inliers {res.inliers:5d}  {res.elapsed_ms:6.1f} ms  "
              f"{'ok' if res.ok else res.reason}", flush=True)

    ok_rows = [x for x in rows if x["ok"]]
    r.raw["registration"] = rows
    if not ok_rows:
        r.add(Finding("registration", "failed everywhere", "fail",
                      "Surfaces cannot be located; the design does not work on this view."))
        return
    inliers = [x["inliers"] for x in ok_rows]
    times = [x["ms"] for x in ok_rows]
    r.add(Finding("positions registered", f"{len(ok_rows)} of {len(rows)}",
                  "pass" if len(ok_rows) == len(rows) else "warn"))
    r.add(Finding("inliers", f"min {min(inliers)}, median {int(statistics.median(inliers))}",
                  "pass" if min(inliers) >= TARGET_MIN_INLIERS else "fail",
                  f"E2 requires at least {TARGET_MIN_INLIERS} across the whole range."))
    r.add(Finding("registration time", f"median {statistics.median(times):.1f} ms, "
                  f"worst {max(times):.1f} ms",
                  "pass" if max(times) <= 15.0 else "warn",
                  "§7 budgets 15 ms."))


def probe_angle_hint_value(source: CameraSource, registrar: Registrar, r: Results) -> None:
    section("5. Do angle hints matter?")
    caps = source.capabilities
    if not caps.has_ptz:
        r.add(Finding("angle hints", "not applicable", "info", "No pan/tilt on this camera."))
        return
    lo, hi = caps.pan_range or (0.0, 0.0)
    with_hint, without = [], []
    for pan in np.linspace(lo + 12, hi - 12, 5):
        source.move_to(float(pan), settle_s=1.0)
        frame = source.read(timeout=4.0)
        if frame is None:
            continue
        a = registrar.register(frame.image, pan=frame.pan, tilt=frame.tilt)
        b = registrar.register(frame.image)  # same pixels, no prior
        if a.ok:
            with_hint.append((a.elapsed_ms, a.inliers))
        if b.ok:
            without.append((b.elapsed_ms, b.inliers))

    if not with_hint or not without:
        r.add(Finding("angle hints", "inconclusive", "warn", "Too few positions registered."))
        return
    ms_a = statistics.median([m for m, _ in with_hint])
    ms_b = statistics.median([m for m, _ in without])
    in_a = statistics.median([i for _, i in with_hint])
    in_b = statistics.median([i for _, i in without])
    r.add(Finding("with angle hint", f"{ms_a:.1f} ms, {int(in_a)} inliers"))
    r.add(Finding("features only", f"{ms_b:.1f} ms, {int(in_b)} inliers",
                  "pass" if ms_b <= 15.0 else "warn",
                  "If this camera reports no angles, this is the real cost." if
                  not caps.reports_angles else ""))
    r.raw["angle_hint"] = {"with_ms": ms_a, "without_ms": ms_b,
                           "with_inliers": in_a, "without_inliers": in_b}


# ------------------------------------------------------------- 6. repeatability


def probe_repeatability(source: CameraSource, room, registrar: Registrar, r: Results,
                        trials: int = 5) -> None:
    section(f"6. Return-to-position repeatability ({trials} trials)")
    caps = source.capabilities
    if not caps.has_ptz:
        r.add(Finding("backlash", "not applicable", "info", "Fixed camera."))
        return
    lo, hi = caps.pan_range or (0.0, 0.0)
    home, away = (lo + hi) / 2.0, hi - 10.0
    centres = []
    for _ in range(trials):
        source.move_to(away, settle_s=1.0)
        source.move_to(home, settle_s=1.5)
        frame = source.read(timeout=4.0)
        if frame is None:
            continue
        res = registrar.register(frame.image, pan=frame.pan, tilt=frame.tilt)
        if not res.ok:
            continue
        w, h = frame.size
        centres.append(transform_points(res.pose.frame_to_map, [(w / 2, h / 2)])[0])

    if len(centres) < 2:
        r.add(Finding("backlash", "could not measure", "warn",
                      "Registration failed on the repeat visits."))
        return
    pts = np.asarray(centres)
    spread = float(np.linalg.norm(pts.std(axis=0)))
    worst = float(np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))
    r.add(Finding("position spread", f"{spread:.1f} px std, {worst:.1f} px worst",
                  "pass" if worst < 120 else "warn",
                  "This is the error registration absorbs. Without it, every polygon "
                  "would shift by this much on each return (D1)."))
    r.raw["repeatability"] = {"std_px": round(spread, 2), "worst_px": round(worst, 2),
                              "trials": len(centres)}


# ------------------------------------------------------------------- 7. sound


def probe_sound(source: CameraSource, r: Results) -> None:
    section("7. Deterrent path and end-to-end latency")
    caps = source.capabilities
    if caps.has_speaker:
        ok = source.play_sound_on_camera("default")
        r.add(Finding("camera speaker", "played" if ok else "refused",
                      "pass" if ok else "warn",
                      "The laptop is off the critical path." if ok else
                      "Falling back to laptop audio (risk R1)."))
    player = Player()
    r.add(Finding("laptop audio backend", player.backend,
                  "pass" if player.available() else "fail", player.last_error[:120]))

    # End-to-end: a frame arrives, we decide immediately, we dispatch a sound.
    # This excludes dwell (a deliberate 500 ms) and capture lag where the source
    # cannot report it, so the report states what it covers.
    samples = []
    for _ in range(5):
        frame = source.read(timeout=3.0)
        if frame is None:
            continue
        origin = frame.ts_capture or frame.ts_received
        outcome = player.play("chirp", volume=0.0)   # silent: measuring dispatch
        if outcome.ok:
            samples.append((time.monotonic() - origin) * 1e3)
        time.sleep(0.2)
    if samples:
        p95 = sorted(samples)[min(len(samples) - 1, int(0.95 * len(samples)))]
        covered = "including capture lag" if frame and frame.ts_capture else \
                  "EXCLUDING capture lag (not reported by this source)"
        r.add(Finding("arrival-to-sound p95", f"{p95:.0f} ms",
                      "pass" if p95 + 500 <= TARGET_P95_LATENCY_MS else "warn",
                      f"{covered}. Add 500 ms dwell for the E1 figure."))
        r.raw["latency"] = {"p95_ms": round(p95, 1), "samples": [round(s, 1) for s in samples],
                            "includes_capture_lag": bool(frame and frame.ts_capture)}


# --------------------------------------------------------------- 8. reconnect


def probe_reconnect(source: CameraSource, r: Results) -> None:
    section("8. Recovery after the stream is interrupted")
    try:
        source.stop()
    except Exception as exc:
        r.add(Finding("teardown", "raised", "warn", str(exc)[:120]))
    time.sleep(1.0)
    started = time.monotonic()
    try:
        source.start()
    except Exception as exc:
        r.add(Finding("reconnect", "failed", "fail", str(exc)[:160]))
        return
    frame = source.read(timeout=TARGET_RECONNECT_S)
    took = time.monotonic() - started
    if frame is None:
        r.add(Finding("reconnect", f"no video within {TARGET_RECONNECT_S:.0f}s", "fail",
                      "E5 requires recovery within 30 s of an interruption."))
    else:
        r.add(Finding("reconnect", f"{took:.1f} s to first frame",
                      "pass" if took <= TARGET_RECONNECT_S else "fail",
                      "E5 target is 30 s."))
    r.raw["reconnect_s"] = round(took, 2)


# ------------------------------------------------------------------- reporting


def write_report(r: Results, room, out_dir: Path, source_label: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if room is not None:
        cv2.imwrite(str(out_dir / "room_map.png"), room.canvas)

    verdict = "GO" if not r.failures else "NO-GO"
    lines = [
        "# Phase 0 feasibility report",
        "",
        f"- **Source**: {source_label}",
        f"- **Run at**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Host**: {platform.platform()}",
        f"- **Verdict**: **{verdict}** — {len(r.failures)} failure(s), {len(r.warnings)} warning(s)",
        "",
        "## Findings",
        "",
        "| Result | Measurement | Value | Notes |",
        "| --- | --- | --- | --- |",
    ]
    for f in r.findings:
        mark = {"pass": "pass", "fail": "**fail**", "warn": "warn", "info": ""}[f.verdict]
        lines.append(f"| {mark} | {f.name} | {f.value} | {f.detail.replace('|', '/')} |")

    if r.failures:
        lines += ["", "## Blocking", ""]
        lines += [f"- **{f.name}**: {f.value} — {f.detail}" for f in r.failures]
    if r.warnings:
        lines += ["", "## Decisions this forces", ""]
        lines += [f"- **{f.name}**: {f.value} — {f.detail}" for f in r.warnings]

    report = out_dir / "phase0-report.md"
    report.write_text("\n".join(lines) + "\n")
    (out_dir / "phase0-raw.json").write_text(json.dumps(r.raw, indent=2, default=str))
    return report


def source_from_settings() -> tuple[CameraSource, str]:
    """Use the camera setup already saved by the app.

    Without this, running against a real camera means assembling a bridge URL and a
    device serial by hand — facts that only exist *after* setup has run, which made
    the harness awkward to reach at exactly the moment it is useful.
    """
    from surfaceguard.bridge.client import BridgeClient
    from surfaceguard.bridge.credentials import EufyAccount
    from surfaceguard.bridge.supervisor import BridgeSupervisor
    from surfaceguard.camera.sources.eufy_bridge import EufyBridgeCamera
    from surfaceguard.storage.preferences import Preferences

    config = Preferences.load().camera or {}
    if config.get("kind") != "eufy":
        raise SystemExit(
            "No Eufy camera is set up yet. Open Surface Guard and finish setup first, "
            "or pass --source and --url explicitly."
        )

    account = EufyAccount(username=str(config.get("username", "")),
                          country=str(config.get("country", "US")))
    supervisor = BridgeSupervisor(account)
    status = supervisor.start(wait=True)
    if not status.listening:
        raise SystemExit(f"The camera service did not start: {status.fatal or status.message}")

    client = BridgeClient(supervisor.url)
    client.connect()
    state = client.connect_driver()
    if state.phase.value != "connected":
        raise SystemExit(f"Could not sign in to Eufy: {state.message}")

    source = EufyBridgeCamera(client=client, serial=str(config.get("serial", "")),
                              model=str(config.get("model", "")),
                              name=str(config.get("name", "")), owns_client=True)
    return source, f"{config.get('name') or 'Eufy camera'} ({config.get('model', '?')})"


def build_source(args) -> tuple[CameraSource, str]:
    if getattr(args, "from_settings", False):
        return source_from_settings()
    if args.source == "synthetic":
        from surfaceguard.camera.sources.synthetic import SyntheticCamera
        return SyntheticCamera(backlash_px=args.backlash, latency_s=0.28), "Synthetic room"
    if args.source == "eufy":
        from surfaceguard.camera.sources.eufy_bridge import EufyBridgeCamera
        return EufyBridgeCamera(url=args.url, serial=args.serial), f"Eufy bridge {args.url}"
    if args.source == "rtsp":
        from surfaceguard.camera.sources.rtsp import RtspCamera
        return RtspCamera(args.url), f"RTSP {args.url}"
    from surfaceguard.camera.sources.replay import ReplayCamera
    return ReplayCamera(args.url, loop=True), f"Replay {args.url}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Surface Guard Phase 0 feasibility harness")
    ap.add_argument("--from-settings", action="store_true",
                    help="use the camera the app already has set up — the usual way "
                         "to run this against a real camera")
    ap.add_argument("--source", default="synthetic",
                    choices=["synthetic", "eufy", "rtsp", "replay"])
    ap.add_argument("--url", default="ws://127.0.0.1:3000",
                    help="bridge URL, RTSP URL, or clip path")
    ap.add_argument("--serial", default=None, help="Eufy device serial number")
    ap.add_argument("--seconds", type=float, default=20.0, help="stream stability window")
    ap.add_argument("--backlash", type=float, default=14.0, help="synthetic backlash, px")
    ap.add_argument("--out", default="phase0-out", help="report directory")
    ap.add_argument("--skip-reconnect", action="store_true")
    args = ap.parse_args()

    source, label = build_source(args)
    results = Results()
    registrar = Registrar()
    room = None

    print(f"Surface Guard — Phase 0 feasibility harness\nSource: {label}")
    try:
        source.start()
    except SourceError as exc:
        print(f"\n  [FAIL] could not start the camera: {exc}")
        results.add(Finding("camera start", "failed", "fail", str(exc)))
        write_report(results, None, Path(args.out), label)
        return 2

    try:
        probe_capabilities(source, results)
        frames = probe_stream(source, results, args.seconds)
        if frames:
            room = probe_map(source, results, registrar)
            if room is not None:
                probe_registration(source, room, registrar, results)
                probe_angle_hint_value(source, registrar, results)
                probe_repeatability(source, room, registrar, results)
            probe_sound(source, results)
        if not args.skip_reconnect:
            probe_reconnect(source, results)
    finally:
        try:
            source.stop()
        except Exception:
            pass

    report = write_report(results, room, Path(args.out), label)
    print(f"\n{'=' * 62}")
    verdict = "GO" if not results.failures else "NO-GO"
    print(f"Verdict: {verdict}  ({len(results.failures)} failures, {len(results.warnings)} warnings)")
    for f in results.failures:
        print(f"  FAIL  {f.name}: {f.value}")
    for f in results.warnings:
        print(f"  WARN  {f.name}: {f.value}")
    print(f"\nReport: {report}")
    return 1 if results.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
