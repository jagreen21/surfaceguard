"""Diagnostics: the numbers the design doc promised would be visible, not just tested.

Latency, registration inliers, inference time and the heartbeat's own verdicts all
live here, so "is it actually working?" has an answer beyond the headline.
"""

from __future__ import annotations

import platform
import subprocess
import time

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..engine import Engine
from ..health.heartbeat import Report
from ..logging_setup import log_dir, redact_support_text, tail
from ..update import build_info
from . import qtutil as Q
from .components import PageHeader

TARGET_P95_MS = 1200.0


class _Tile(QFrame):
    def __init__(self, caption: str, unit: str = "") -> None:
        super().__init__()
        self.setObjectName("panel")
        self.value = QLabel("—")
        self.value.setStyleSheet("font-size: 25px; font-weight: 600;")
        self.unit = QLabel(unit)
        self.unit.setObjectName("dim")
        cap = QLabel(caption)
        cap.setObjectName("dim")
        cap.setWordWrap(True)
        row = QHBoxLayout()
        row.setSpacing(5)
        row.addWidget(self.value)
        row.addWidget(self.unit)
        row.addStretch(1)
        box = QVBoxLayout(self)
        box.setContentsMargins(13, 11, 13, 12)
        box.setSpacing(2)
        box.addLayout(row)
        box.addWidget(cap)

    def set(self, text: str, colour=None) -> None:
        self.value.setText(text)
        self.value.setStyleSheet(
            "font-size: 25px; font-weight: 600;"
            + (f" color: {colour.name()};" if colour is not None else "")
        )


class DiagnosticsScreen(QWidget):
    rescan_requested = Signal()

    def __init__(self, engine: Engine) -> None:
        super().__init__()
        self.engine = engine
        self.header = PageHeader("System Health", "Live camera, detection, and response checks.")

        self.latency = _Tile("End-to-end, p95 — target 1200 ms (E1)", "ms")
        self.inliers = _Tile("Registration inliers — need 30 (E2)", "")
        self.reg_ms = _Tile("Registration time — budget 15 ms", "ms")
        self.infer_ms = _Tile("Detector inference — budget 40 ms", "ms")
        self.fps = _Tile("Frames judged per second", "fps")
        self.missed = _Tile("Pet events the camera saw while video was down", "")

        self.tiles = (self.latency, self.inliers, self.reg_ms, self.infer_ms, self.fps, self.missed)
        self.grid = QGridLayout()
        self.grid.setSpacing(11)
        for i, tile in enumerate(self.tiles):
            self.grid.addWidget(tile, i // 3, i % 3)

        self.checks = QLabel("Waiting for the first check…")
        self.checks.setObjectName("mono")
        self.checks.setWordWrap(True)
        self.checks.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.camera = QLabel("")
        self.camera.setObjectName("mono")
        self.camera.setWordWrap(True)
        self.camera.setAlignment(Qt.AlignmentFlag.AlignTop)

        rescan = QPushButton("Scan the room again")
        rescan.clicked.connect(self.rescan_requested.emit)

        # She will be the one in the room when this breaks, and he will not.
        # These two buttons are how a problem travels from her Mac to him.
        copy_btn = QPushButton("Copy diagnostics")
        copy_btn.clicked.connect(self._copy_diagnostics)
        logs_btn = QPushButton("Show log files")
        logs_btn.clicked.connect(self._reveal_logs)
        self.support_note = QLabel("")
        self.support_note.setObjectName("dim")
        self.support_note.setWordWrap(True)
        support_row = QHBoxLayout()
        support_row.addWidget(copy_btn)
        support_row.addWidget(logs_btn)
        support_row.addStretch(1)

        checks_panel = QFrame()
        checks_panel.setObjectName("panel")
        cbox = QVBoxLayout(checks_panel)
        cbox.setContentsMargins(13, 12, 13, 13)
        cap = QLabel("Self-check")
        cap.setObjectName("h2")
        cbox.addWidget(cap)
        cbox.addWidget(self.checks, 1)

        cam_panel = QFrame()
        cam_panel.setObjectName("panel")
        mbox = QVBoxLayout(cam_panel)
        mbox.setContentsMargins(13, 12, 13, 13)
        cap2 = QLabel("Camera and detector")
        cap2.setObjectName("h2")
        mbox.addWidget(cap2)
        mbox.addWidget(self.camera, 1)
        mbox.addWidget(rescan)
        mbox.addLayout(support_row)
        mbox.addWidget(self.support_note)

        self.checks_panel = checks_panel
        self.camera_panel = cam_panel
        self.panels = QGridLayout()
        self.panels.setSpacing(12)
        self.panels.addWidget(checks_panel, 0, 0)
        self.panels.addWidget(cam_panel, 0, 1)
        self.panels.setColumnStretch(0, 1)
        self.panels.setColumnStretch(1, 1)
        self._narrow = False

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)
        root.addWidget(self.header)
        root.addLayout(self.grid)
        root.addLayout(self.panels, 1)

    def adapt_to_width(self, width: int) -> None:
        narrow = width < 780
        if narrow == self._narrow:
            return
        self._narrow = narrow
        columns = 2 if narrow else 3
        for tile in self.tiles:
            self.grid.removeWidget(tile)
        for index, tile in enumerate(self.tiles):
            self.grid.addWidget(tile, index // columns, index % columns)
        self.panels.removeWidget(self.checks_panel)
        self.panels.removeWidget(self.camera_panel)
        self.panels.addWidget(self.checks_panel, 0, 0)
        self.panels.addWidget(self.camera_panel, 1 if narrow else 0, 0 if narrow else 1)

    def refresh(self) -> None:
        m = self.engine.metrics
        p95 = m.p95_latency_ms()
        if p95 is None:
            self.latency.set("—")
        else:
            self.latency.set(f"{p95:.0f}", Q.GOOD if p95 <= TARGET_P95_MS else Q.BAD)
        self.inliers.set(str(m.inliers), Q.GOOD if m.inliers >= 30 else Q.BAD)
        self.reg_ms.set(f"{m.registration_ms:.1f}", Q.GOOD if m.registration_ms <= 15 else Q.WARN)
        self.infer_ms.set(f"{m.inference_ms:.0f}", Q.GOOD if m.inference_ms <= 40 else Q.WARN)
        self.fps.set(f"{m.fps:.1f}")
        missed = self.engine.missed_pet_events
        self.missed.set(str(missed), Q.BAD if missed else Q.GOOD)

        self._render_report(self.engine.last_report)
        caps = self.engine.source.capabilities
        info = self.engine.detector.info
        lines = [f"{k:26s} {v}" for k, v in caps.as_rows()]
        lines.append(f"{'detector':26s} {info.name} ({info.backend})")
        if info.providers:
            lines.append(f"{'providers':26s} {', '.join(info.providers)}")
        if info.note:
            lines.append(f"{'note':26s} {info.note}")
        for note in caps.notes:
            lines.append(f"{'note':26s} {note}")
        self.camera.setText("\n".join(lines))

    # ------------------------------------------------------------------ support

    def diagnostics_text(self) -> str:
        """One pasteable block: everything needed to debug this from somewhere else."""
        m = self.engine.metrics
        caps = self.engine.source.capabilities
        info = self.engine.detector.info
        report = self.engine.last_report
        lines = [
            f"Surface Guard {build_info.VERSION} ({build_info.UPDATE_CHANNEL})",
            f"macOS {platform.mac_ver()[0]} on {platform.machine()}",
            f"time: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
            "",
            f"state      : {self.engine.state.state().headline} — "
            f"{self.engine.state.state().detail}",
            f"camera     : {caps.name} ({caps.model}) ptz={caps.has_ptz} "
            f"angles={caps.reports_angles} speaker={caps.has_speaker}",
            f"detector   : {info.name} / {info.backend}",
            f"frames     : {m.frames}  fps {m.fps:.1f}  inference {m.inference_ms:.0f} ms",
            f"registered : {m.registered}  inliers {m.inliers}  "
            f"registration {m.registration_ms:.1f} ms",
            f"latency p95: {m.p95_latency_ms() or float('nan'):.0f} ms",
            f"cats seen off every surface: {m.cats_off_surface}"
            + ("   <- if this climbs while nothing fires, the surface is probably "
               "drawn in the wrong place" if m.cats_off_surface else ""),
            f"surfaces   : {len(self.engine.prefs.surfaces)}  "
            f"missed pet events: {self.engine.missed_pet_events}",
        ]
        if self.engine.bridge is not None:
            st = self.engine.bridge.status
            lines.append(
                f"bridge     : running={st.running} listening={st.listening} "
                f"port={st.port} restarts={st.restarts} {st.fatal or st.message}"
            )
        if self.engine.updater is not None:
            us = self.engine.updater.status
            lines.append(f"updates    : {us.state.value} — {us.message}")
            lines.append(f"update auth: {us.token.summary()}")
        if report is not None:
            lines.append("")
            lines.append("self-check:")
            lines += [
                f"  {'ok  ' if c.ok else ('FAIL' if c.required else 'warn')}  {c.detail}"
                for c in report.checks
            ]
        if self.engine.bridge is not None:
            recent = self.engine.bridge.logs(12)
            if recent:
                lines += ["", "camera service (last lines):"]
                lines += [f"  {line[:180]}" for line in recent]
        lines += ["", "app log (last lines):", tail(40)]
        return redact_support_text("\n".join(lines))

    def _copy_diagnostics(self) -> None:
        QApplication.clipboard().setText(self.diagnostics_text())
        self.support_note.setText(
            "Copied. Paste it into a message — it has no passwords or tokens in it."
        )

    def _reveal_logs(self) -> None:
        subprocess.run(["/usr/bin/open", str(log_dir())], check=False)
        self.support_note.setText(f"Logs are in {log_dir()}")

    def _render_report(self, report: Report | None) -> None:
        if report is None:
            self.checks.setText("Waiting for the first check…")
            return
        rows = []
        for c in report.checks:
            colour = Q.GOOD if c.ok else (Q.BAD if c.required else Q.WARN)
            mark = "ok  " if c.ok else ("FAIL" if c.required else "warn")
            rows.append(f"<span style='color:{colour.name()}'>{mark}</span>  {c.detail}")
            if not c.ok and c.remedy:
                rows.append(f"<span style='color:#9aa8b3'>      {c.remedy}</span>")
        self.checks.setText("<pre style='margin:0'>" + "\n".join(rows) + "</pre>")
