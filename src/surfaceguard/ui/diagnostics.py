"""Diagnostics: the numbers the design doc promised would be visible, not just tested.

Latency, registration inliers, inference time and the heartbeat's own verdicts all
live here, so "is it actually working?" has an answer beyond the headline.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
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
from . import qtutil as Q

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

        self.latency = _Tile("End-to-end, p95 — target 1200 ms (E1)", "ms")
        self.inliers = _Tile("Registration inliers — need 30 (E2)", "")
        self.reg_ms = _Tile("Registration time — budget 15 ms", "ms")
        self.infer_ms = _Tile("Detector inference — budget 40 ms", "ms")
        self.fps = _Tile("Frames judged per second", "fps")
        self.missed = _Tile("Pet events the camera saw while video was down", "")

        grid = QGridLayout()
        grid.setSpacing(11)
        for i, tile in enumerate(
            (self.latency, self.inliers, self.reg_ms, self.infer_ms, self.fps, self.missed)
        ):
            grid.addWidget(tile, i // 3, i % 3)

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

        panels = QHBoxLayout()
        panels.setSpacing(12)
        panels.addWidget(checks_panel, 1)
        panels.addWidget(cam_panel, 1)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)
        root.addLayout(grid)
        root.addLayout(panels, 1)

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
