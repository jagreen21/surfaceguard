"""The home screen: one question answered four ways.

Is protection on? Is the camera connected? Which surfaces are protected? What
happened most recently? Everything here is rendered from StateStore, so the screen
cannot claim protection the heartbeat has not confirmed (D4).
"""

from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import QPointF, QRect, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..detection.trigger_policy import Presence
from ..engine import FrameResult
from ..geometry.projection import transform_points
from ..state import AppState, Phase
from . import qtutil as Q
from .components import ActionCard, GlassCard
from .review import ReviewInvite


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
        self._show_zones = True
        self._show_boxes = True
        self._show_labels = True
        self._room_name = "Kitchen"
        self._camera_name = ""

    def set_overlays(self, zones: bool, boxes: bool, labels: bool) -> None:
        self._show_zones = zones
        self._show_boxes = boxes
        self._show_labels = labels
        self.update()

    def set_room_name(self, name: str) -> None:
        self._room_name = name
        self.update()

    def set_camera_name(self, name: str) -> None:
        self._camera_name = name
        self.update()

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
        violation = False
        if result.pose is not None and self._show_zones:
            for surface in self._surfaces:
                if not surface.enabled:
                    continue
                poly = transform_points(result.pose.map_to_frame, surface.polygon)
                qpoly = QPolygonF()
                for pt in poly:
                    x, y = to_view(pt)
                    qpoly.append(QPointF(x, y))
                hot = any(v.surface_id == surface.id and v.on_surface for v in result.verdicts)
                violation = violation or hot
                p.setPen(QPen(Q.BAD, 2.2))
                p.setBrush(QBrush(QColor(255, 92, 87, 66)))
                p.drawPolygon(qpoly)
                p.setBrush(QBrush(Qt.GlobalColor.white))
                for point in qpoly:
                    p.drawEllipse(point, 3.5, 3.5)
                if self._show_labels:
                    p.setPen(Q.INK)
                    f = p.font(); f.setPointSize(10); p.setFont(f)
                    r = qpoly.boundingRect()
                    p.drawText(QRect(int(r.x()), int(r.y()) - 18, 220, 16),
                               Qt.AlignmentFlag.AlignLeft, surface.name)
        elif self._surfaces and self._show_zones:
            p.setPen(Q.WARN)
            p.drawText(QRect(ox + 8, oy + 8, self.width() - 20, 18),
                       Qt.AlignmentFlag.AlignLeft,
                       _friendly_registration_message(result.registration_reason))

        if self._show_boxes:
            for box, colour, label in (
                *[(b, Q.CAT, "Cat detected") for b in result.cats],
                *[(b, Q.PERSON, "Person nearby") for b in result.people],
            ):
                x1, y1 = to_view((box.x1, box.y1))
                x2, y2 = to_view((box.x2, box.y2))
                p.setPen(QPen(colour, 2.0))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawRoundedRect(QRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1)), 5, 5)
                px, py = to_view(box.paw_point)
                p.setBrush(QBrush(colour))
                p.drawEllipse(QPointF(px, py), 4, 4)
                p.setPen(colour)
                f = p.font(); f.setPointSize(9); p.setFont(f)
                p.drawText(QRect(int(x1), int(y1) - 17, 150, 15),
                           Qt.AlignmentFlag.AlignLeft, label)

        # Camera identity and detection context stay small and local to the video.
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(8, 16, 22, 210))
        p.drawRoundedRect(QRect(12, 12, max(112, len(self._room_name) * 8 + 34), 28), 8, 8)
        p.setPen(Q.INK)
        f = p.font(); f.setPointSize(10); p.setFont(f)
        p.drawText(QRect(23, 12, 190, 28), Qt.AlignmentFlag.AlignVCenter,
                   "●  " + self._room_name)
        if self._camera_name:
            p.setPen(Qt.GlobalColor.white)
            p.drawText(QRect(self.width() - 180, self.height() - 30, 166, 20),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       self._camera_name)
        if violation:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(171, 48, 47, 224))
            p.drawRoundedRect(QRect(self.width() - 132, 12, 120, 28), 8, 8)
            p.setPen(Qt.GlobalColor.white)
            p.drawText(QRect(self.width() - 123, 12, 108, 28),
                       Qt.AlignmentFlag.AlignVCenter, "Cat detected")


class HomeScreen(QWidget):
    """Camera-first dashboard with calm status and the common actions."""

    arm_requested = Signal()
    off_requested = Signal()
    pause_requested = Signal(float)   # seconds
    test_sound_requested = Signal()
    navigate_requested = Signal(str)
    activity_requested = Signal()
    review_requested = Signal()
    review_declined = Signal()
    review_silenced = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.live = LiveView()
        self.live.setObjectName("heroView")

        self.headline = QLabel("Starting up")
        self.headline.setObjectName("cardTitle")
        self.detail = QLabel("")
        self.detail.setObjectName("dim")
        self.detail.setWordWrap(True)
        self.remedy = QLabel("")
        self.remedy.setObjectName("dim")
        self.remedy.setWordWrap(True)
        self.dot = QLabel("●")
        self.attention_action = QPushButton("Continue")
        self.attention_action.setObjectName("primary")
        self.attention_action.clicked.connect(self._toggle)
        self.attention_card = GlassCard(compact=True)
        self.attention_card.setObjectName("statusCard")
        attention_top = QHBoxLayout()
        attention_top.addWidget(self.dot)
        attention_top.addWidget(self.headline)
        attention_top.addStretch(1)
        attention_top.addWidget(self.attention_action)
        self.attention_card.box.addLayout(attention_top)
        self.attention_card.box.addWidget(self.detail)
        self.attention_card.box.addWidget(self.remedy)
        self.attention_card.setVisible(False)

        self.protect_btn = QPushButton(" ")
        self.protect_btn.setObjectName("toggleButton")
        self.protect_btn.setCheckable(True)
        self.protect_btn.setToolTip("Turn protection on or off")
        self.protect_btn.setAccessibleName("Protection")
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
        self.coverage.setTextFormat(Qt.TextFormat.RichText)
        self.coverage.setWordWrap(True)

        self.invite = ReviewInvite()
        self.invite.accepted.connect(self.review_requested.emit)
        self.invite.declined.connect(self.review_declined.emit)
        self.invite.silenced.connect(self.review_silenced.emit)

        hero = QFrame()
        hero.setObjectName("heroFrame")
        hero_box = QVBoxLayout(hero)
        hero_box.setContentsMargins(1, 1, 1, 1)
        hero_box.addWidget(self.live)

        self.protection_card = ActionCard("Protection", "Active", "Kitchen Counter  ·  24/7")
        self.protection_card.add_trailing(self.protect_btn)
        self.detection_card = ActionCard("Detection sensitivity", "Balanced", "Medium")
        self.audio_card = ActionCard("Sound response", "Short hiss", "This Mac")
        chevron = QLabel("›")
        chevron.setStyleSheet("font-size: 24px; color: #96a4ad")
        self.audio_card.add_trailing(chevron)
        self.protection_card.activated.connect(lambda: self.navigate_requested.emit("Rooms"))
        self.detection_card.activated.connect(lambda: self.navigate_requested.emit("Detection"))
        self.audio_card.activated.connect(lambda: self.navigate_requested.emit("Audio"))
        self.quick = QGridLayout()
        self.quick.setSpacing(12)
        self.quick_cards = (self.protection_card, self.detection_card, self.audio_card)
        self._quick_columns = 3
        for i, card in enumerate(self.quick_cards):
            self.quick.addWidget(card, 0, i)

        protection_actions = QHBoxLayout()
        protection_actions.setSpacing(8)
        protection_actions.addWidget(self.pause30)
        protection_actions.addWidget(self.pause_tomorrow)
        protection_actions.addWidget(self.test_btn)
        protection_actions.addStretch(1)

        coverage_card = GlassCard(compact=True)
        coverage_card.setObjectName("sectionCard")
        coverage_title = QLabel("Coverage")
        coverage_title.setObjectName("cardTitle")
        coverage_card.box.addWidget(coverage_title)
        coverage_card.box.addWidget(self.coverage)

        recent_card = GlassCard(compact=True)
        recent_card.setObjectName("sectionCard")
        recent_row = QHBoxLayout()
        recent = QLabel("Events")
        recent.setObjectName("cardTitle")
        see_all = QPushButton("View all")
        see_all.setObjectName("secondary")
        see_all.clicked.connect(self.activity_requested.emit)
        recent_row.addWidget(recent)
        recent_row.addStretch(1)
        recent_row.addWidget(see_all)
        recent_card.box.addLayout(recent_row)
        self.event_rows: list[QLabel] = []
        self.event_thumbs: list[QLabel] = []
        for _ in range(3):
            row = QLabel("")
            row.setObjectName("eventRow")
            row.setTextFormat(Qt.TextFormat.RichText)
            row.setWordWrap(True)
            thumb = QLabel("")
            thumb.setFixedSize(76, 32)
            thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
            line = QHBoxLayout()
            line.setContentsMargins(0, 0, 0, 0)
            line.addWidget(row, 1)
            line.addWidget(thumb)
            self.event_rows.append(row)
            self.event_thumbs.append(thumb)
            recent_card.box.addLayout(line)
        self.last_event = self.event_rows[0]

        content = QWidget()
        body = QVBoxLayout(content)
        body.setContentsMargins(12, 12, 12, 12)
        body.setSpacing(10)
        body.addWidget(self.attention_card)
        body.addWidget(self.invite)
        body.addWidget(hero, 1)
        body.addLayout(self.quick)
        body.addLayout(protection_actions)
        body.addWidget(coverage_card)
        body.addWidget(recent_card)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scroll)

        self._state: AppState | None = None

    def adapt_to_width(self, width: int) -> None:
        """Reflow dashboard cards instead of squeezing three columns unreadably."""
        columns = 1 if width < 720 else 3
        if columns == self._quick_columns:
            return
        self._quick_columns = columns
        for card in self.quick_cards:
            self.quick.removeWidget(card)
        for index, card in enumerate(self.quick_cards):
            self.quick.addWidget(card, index // columns, index % columns)

    def set_context(self, room_name: str, surface_summary: str, sensitivity: str,
                    audio_name: str, audio_target: str) -> None:
        self.live.set_room_name(room_name)
        self.protection_card.detail.setText(surface_summary)
        self.detection_card.title.setText(sensitivity)
        self.detection_card.detail.setText({
            "Calm": "Low", "Balanced": "Medium", "Sensitive": "High"
        }.get(sensitivity, sensitivity))
        self.audio_card.title.setText(audio_name)
        self.audio_card.detail.setText(audio_target)

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
        self.protect_btn.blockSignals(True)
        self.protect_btn.setChecked(guarding)
        self.protect_btn.blockSignals(False)
        show_attention = state.phase in (Phase.NEEDS_SETUP, Phase.PROBLEM, Phase.PAUSED)
        self.attention_card.setVisible(show_attention)
        if state.phase is Phase.NEEDS_SETUP:
            self.attention_action.setText("Finish setup")
        elif state.phase is Phase.PROBLEM:
            self.attention_action.setText("Turn protection off")
        elif state.phase is Phase.PAUSED:
            self.attention_action.setText("Resume")
        else:
            self.attention_action.setText("Continue")
        can_pause = state.phase in (Phase.GUARDING, Phase.ALERTING, Phase.PROBLEM, Phase.STARTING)
        self.pause30.setEnabled(can_pause)
        self.pause_tomorrow.setEnabled(can_pause)
        self.protection_card.title.setText({
            Phase.GUARDING: "Active", Phase.ALERTING: "Responding",
            Phase.PAUSED: "Paused", Phase.OFF: "Off", Phase.PROBLEM: "Needs attention",
            Phase.STARTING: "Starting", Phase.NEEDS_SETUP: "Finish setup",
        }.get(state.phase, state.headline))

    def render_coverage(self, surfaces: list, plan, presence: dict[str, Presence]) -> None:
        if not surfaces:
            self.coverage.setText("<span style='color:#9caaa1'>Nothing is protected yet. "
                                  "Open Rooms to draw a surface.</span>")
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
                chip, colour = "protected", Q.GOOD
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
        self.render_events([text])

    def render_events(self, rows: list) -> None:
        if not rows:
            rows = ["No activity yet. Recent detections will appear here."]
        for index, (label, thumb) in enumerate(zip(self.event_rows, self.event_thumbs)):
            visible = index < len(rows)
            item = rows[index] if visible else ""
            text, path = item if isinstance(item, tuple) else (item, None)
            label.setText(text)
            label.setVisible(visible)
            pixmap = QPixmap(path) if path else QPixmap()
            thumb.setPixmap(pixmap.scaled(
                thumb.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            ) if not pixmap.isNull() else QPixmap())
            thumb.setVisible(visible and not pixmap.isNull())

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


def _friendly_registration_message(reason: str) -> str:
    """Translate map-localisation internals into an actionable camera message."""
    if "keyframe" in reason.lower():
        return "Finish the room scan to show surfaces here."
    return "Surfaces are temporarily unavailable in this camera view."
