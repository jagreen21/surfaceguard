"""The home screen: one question answered four ways.

Is protection on? Is the camera connected? Which surfaces are protected? What
happened most recently? Everything here is rendered from StateStore, so the screen
cannot claim protection the heartbeat has not confirmed (D4).
"""

from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import QPointF, QRect, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..detection.trigger_policy import Presence
from ..engine import FrameResult
from ..geometry.projection import transform_points
from ..state import AppState, Phase
from . import qtutil as Q


class LiveView(QWidget):
    """The current frame with the surfaces projected onto it.

    This is the honest preview: the polygon drawn here is the one the gates test,
    and the dot is the paw point they test with.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pixmap = QPixmap()
        self._result: FrameResult | None = None
        self._surfaces: list = []
        self._message = "Waiting for video…"

    def update_result(self, result: FrameResult, surfaces: list) -> None:
        self._result = result
        self._surfaces = surfaces
        self._pixmap = Q.bgr_to_pixmap(result.frame.image)
        self.update()

    def set_message(self, text: str) -> None:
        self._message = text
        self._result = None
        self._pixmap = QPixmap()
        self.update()

    def _fit(self) -> tuple[float, int, int]:
        if self._pixmap.isNull():
            return 1.0, 0, 0
        s = min(self.width() / self._pixmap.width(), self.height() / self._pixmap.height())
        return s, int((self.width() - self._pixmap.width() * s) / 2), \
            int((self.height() - self._pixmap.height() * s) / 2)

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#0a0f13"))

        if self._pixmap.isNull() or self._result is None:
            p.setPen(Q.INK_DIM)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._message)
            return

        s, ox, oy = self._fit()
        p.drawPixmap(QRect(ox, oy, int(self._pixmap.width() * s),
                           int(self._pixmap.height() * s)), self._pixmap)

        def to_view(pt) -> tuple[float, float]:
            return (pt[0] * s + ox, pt[1] * s + oy)

        result = self._result
        if result.pose is not None:
            for surface in self._surfaces:
                if not surface.enabled:
                    continue
                poly = transform_points(result.pose.map_to_frame, surface.polygon)
                qpoly = QPolygonF()
                for pt in poly:
                    x, y = to_view(pt)
                    qpoly.append(QPointF(x, y))
                hot = any(v.surface_id == surface.id and v.on_surface for v in result.verdicts)
                p.setPen(QPen(Q.BAD if hot else Q.ACCENT, 2.0))
                p.setBrush(QBrush(QColor(224, 121, 95, 66) if hot else QColor(79, 180, 207, 40)))
                p.drawPolygon(qpoly)
                p.setPen(Q.INK)
                f = p.font(); f.setPointSize(10); p.setFont(f)
                r = qpoly.boundingRect()
                p.drawText(QRect(int(r.x()), int(r.y()) - 18, 220, 16),
                           Qt.AlignmentFlag.AlignLeft, surface.name)
        else:
            p.setPen(Q.WARN)
            p.drawText(QRect(ox + 8, oy + 8, self.width() - 20, 18),
                       Qt.AlignmentFlag.AlignLeft,
                       "Camera view not recognised — " + (result.registration_reason or "no pose"))

        for box, colour, label in (
            *[(b, Q.CAT, f"cat {b.score:.0%}") for b in result.cats],
            *[(b, Q.PERSON, "person") for b in result.people],
        ):
            x1, y1 = to_view((box.x1, box.y1))
            x2, y2 = to_view((box.x2, box.y2))
            p.setPen(QPen(colour, 1.8))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
            px, py = to_view(box.paw_point)
            p.setBrush(QBrush(colour))
            p.drawEllipse(QPointF(px, py), 4, 4)
            p.setPen(colour)
            f = p.font(); f.setPointSize(9); p.setFont(f)
            p.drawText(QRect(int(x1), int(y1) - 15, 140, 14),
                       Qt.AlignmentFlag.AlignLeft, label)


class HomeScreen(QWidget):
    """State, live view, the four controls, and honest per-surface coverage."""

    arm_requested = Signal()
    off_requested = Signal()
    pause_requested = Signal(float)   # seconds
    test_sound_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.live = LiveView()

        self.headline = QLabel("Starting up")
        self.headline.setObjectName("h1")
        self.detail = QLabel("")
        self.detail.setObjectName("dim")
        self.detail.setWordWrap(True)
        self.remedy = QLabel("")
        self.remedy.setWordWrap(True)
        self.remedy.setVisible(False)
        self.dot = QLabel("●")

        self.protect_btn = QPushButton("Turn protection on")
        self.protect_btn.setObjectName("primary")
        self.protect_btn.clicked.connect(self._toggle)
        self.pause30 = QPushButton("Pause 30 minutes")
        self.pause30.clicked.connect(lambda: self.pause_requested.emit(1800.0))
        self.pause_tomorrow = QPushButton("Pause until tomorrow")
        self.pause_tomorrow.clicked.connect(
            lambda: self.pause_requested.emit(_seconds_until_tomorrow())
        )
        self.test_btn = QPushButton("Test sound")
        self.test_btn.clicked.connect(self.test_sound_requested.emit)

        self.coverage = QLabel("")
        self.coverage.setObjectName("mono")
        self.coverage.setTextFormat(Qt.TextFormat.RichText)
        self.coverage.setWordWrap(True)
        self.last_event = QLabel("Nothing yet today.")
        self.last_event.setObjectName("dim")
        self.last_event.setWordWrap(True)

        status = QVBoxLayout()
        status.setSpacing(4)
        top = QHBoxLayout()
        top.setSpacing(9)
        top.addWidget(self.dot)
        top.addWidget(self.headline)
        top.addStretch(1)
        status.addLayout(top)
        status.addWidget(self.detail)
        status.addWidget(self.remedy)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        for b in (self.protect_btn, self.pause30, self.pause_tomorrow, self.test_btn):
            controls.addWidget(b)
        controls.addStretch(1)

        side = QVBoxLayout()
        side.setContentsMargins(14, 14, 14, 14)
        side.setSpacing(9)
        cap = QLabel("What is being watched")
        cap.setObjectName("h2")
        side.addWidget(cap)
        side.addWidget(self.coverage)
        side.addStretch(1)
        recent = QLabel("Most recent")
        recent.setObjectName("h2")
        side.addWidget(recent)
        side.addWidget(self.last_event)
        panel = QFrame()
        panel.setObjectName("panel")
        panel.setFixedWidth(300)
        panel.setLayout(side)

        body = QHBoxLayout()
        body.setSpacing(14)
        body.addWidget(self.live, 1)
        body.addWidget(panel)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)
        root.addLayout(status)
        root.addLayout(controls)
        root.addLayout(body, 1)

        self._state: AppState | None = None

    # ------------------------------------------------------------------ render

    def render_state(self, state: AppState) -> None:
        self._state = state
        colour = Q.SEVERITY_COLOUR[state.severity.value]
        self.dot.setStyleSheet(f"color: {colour.name()}; font-size: 17px;")
        self.headline.setText(state.headline)
        self.detail.setText(state.detail)
        self.remedy.setText(state.remedy)
        self.remedy.setVisible(bool(state.remedy))
        self.remedy.setStyleSheet(f"color: {Q.WARN.name()};")

        guarding = state.phase in (Phase.GUARDING, Phase.ALERTING, Phase.STARTING, Phase.PROBLEM)
        if state.phase is Phase.NEEDS_SETUP:
            # Stays enabled: it routes to the step that is actually missing, which
            # is more use than a dead button on an otherwise empty screen.
            self.protect_btn.setText("Finish setup")
        else:
            self.protect_btn.setText("Turn protection off" if guarding else "Turn protection on")
        can_pause = state.phase in (Phase.GUARDING, Phase.ALERTING, Phase.PROBLEM, Phase.STARTING)
        self.pause30.setEnabled(can_pause)
        self.pause_tomorrow.setEnabled(can_pause)

    def render_coverage(self, surfaces: list, plan, presence: dict[str, Presence]) -> None:
        if not surfaces:
            self.coverage.setText("<span style='color:#9aa8b3'>No surfaces yet. "
                                  "Open Surfaces to draw one.</span>")
            return
        cov = plan.coverage() if plan is not None else {}
        rows = []
        for s in surfaces:
            frac = cov.get(s.id, 0.0)
            if not s.enabled:
                chip, colour = "paused", Q.INK_DIM
            elif frac <= 0.0:
                chip, colour = "not watched", Q.BAD
            elif frac >= 0.999:
                chip, colour = "watched", Q.GOOD
            else:
                chip, colour = f"{frac:.0%} of the time", Q.WARN
            here = presence.get(s.id)
            note = " · cat here now" if here is Presence.PRESENT else ""
            rows.append(
                f"<div style='margin-bottom:6px'>{s.name}<br>"
                f"<span style='color:{colour.name()}'>{chip}</span>"
                f"<span style='color:#9aa8b3'>{note}</span></div>"
            )
        if plan is not None and plan.single_view:
            rows.append("<div style='color:#5fbf94'>All surfaces visible at once — "
                        "no scanning needed.</div>")
        self.coverage.setText("".join(rows))

    def render_last_event(self, text: str) -> None:
        self.last_event.setText(text)

    def _toggle(self) -> None:
        state = self._state
        if state is not None and state.phase in (
            Phase.GUARDING, Phase.ALERTING, Phase.STARTING, Phase.PROBLEM
        ):
            self.off_requested.emit()
        else:
            self.arm_requested.emit()


def _seconds_until_tomorrow() -> float:
    lt = time.localtime()
    # 07:00 tomorrow: "until tomorrow" should end when the kitchen wakes up.
    seconds_today = lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec
    return max(60.0, (24 * 3600 - seconds_today) + 7 * 3600)
