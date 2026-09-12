"""Reusable consumer-facing pieces for the Surface Guard interface."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class BrandMark(QWidget):
    """Small shield-and-cat mark drawn natively so it stays crisp at any scale."""

    def __init__(self) -> None:
        super().__init__()
        self.setFixedSize(30, 30)

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#2a3540"))
        painter.drawRoundedRect(self.rect(), 7, 7)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#eef5f8"), 1.25))
        shield = QPainterPath()
        shield.moveTo(7, 7); shield.lineTo(15, 5); shield.lineTo(23, 7)
        shield.lineTo(22, 18); shield.quadTo(20, 23, 15, 25)
        shield.quadTo(10, 23, 8, 18); shield.closeSubpath()
        painter.drawPath(shield)
        cat = QPainterPath()
        cat.moveTo(10, 13); cat.lineTo(10, 10); cat.lineTo(13, 12)
        cat.quadTo(15, 11, 17, 12); cat.lineTo(20, 10); cat.lineTo(20, 14)
        cat.quadTo(20, 19, 15, 19); cat.quadTo(10, 19, 10, 14)
        painter.drawPath(cat)
        painter.drawPoint(13, 15); painter.drawPoint(17, 15)


class GlassCard(QFrame):
    """The standard content surface used throughout the app."""

    def __init__(self, parent: QWidget | None = None, *, compact: bool = False) -> None:
        super().__init__(parent)
        self.setObjectName("glassCard")
        self.box = QVBoxLayout(self)
        margin = 14 if compact else 18
        self.box.setContentsMargins(margin, margin, margin, margin)
        self.box.setSpacing(10)


class PageHeader(QWidget):
    def __init__(self, title: str, subtitle: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.title = QLabel(title)
        self.title.setObjectName("pageTitle")
        self.subtitle = QLabel(subtitle)
        self.subtitle.setObjectName("pageSubtitle")
        self.subtitle.setWordWrap(True)
        copy = QVBoxLayout()
        copy.setContentsMargins(0, 0, 0, 0)
        copy.setSpacing(3)
        copy.addWidget(self.title)
        if subtitle:
            copy.addWidget(self.subtitle)
        self.actions = QHBoxLayout()
        self.actions.setContentsMargins(0, 0, 0, 0)
        self.actions.setSpacing(8)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addLayout(copy, 1)
        row.addLayout(self.actions)

    def add_action(self, button: QWidget) -> None:
        self.actions.addWidget(button)


class StatusPill(QLabel):
    def __init__(self, text: str = "", tone: str = "neutral") -> None:
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.set_status(text, tone)

    def set_status(self, text: str, tone: str = "neutral") -> None:
        self.setText(text)
        self.setProperty("tone", tone)
        self.style().unpolish(self)
        self.style().polish(self)


class NavButton(QPushButton):
    """Sidebar destination with a stable compact form for narrow windows."""

    def __init__(self, symbol: str, label: str) -> None:
        super().__init__(f"{symbol}   {label}")
        self.symbol = symbol
        self.label = label
        self.setCheckable(True)
        self.setObjectName("navButton")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(label)
        self.setAccessibleName(label)

    def set_compact(self, compact: bool) -> None:
        self.setText(self.symbol if compact else f"{self.symbol}   {self.label}")
        self.setProperty("compact", compact)
        self.style().unpolish(self)
        self.style().polish(self)


class ActionCard(GlassCard):
    """Clickable dashboard summary with one primary value."""

    activated = Signal()

    def __init__(self, eyebrow: str, title: str, detail: str = "") -> None:
        super().__init__(compact=True)
        self.setObjectName("actionCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName(title)
        self.setAccessibleDescription(detail)
        self.eyebrow = QLabel(eyebrow)
        self.eyebrow.setObjectName("eyebrow")
        self.title = QLabel(title)
        self.title.setObjectName("cardValue")
        self.detail = QLabel(detail)
        self.detail.setObjectName("muted")
        self.detail.setWordWrap(True)
        copy = QVBoxLayout()
        copy.setContentsMargins(0, 0, 0, 0)
        copy.setSpacing(3)
        copy.addWidget(self.eyebrow)
        copy.addWidget(self.title)
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.addLayout(copy, 1)
        self.trailing = QHBoxLayout()
        self.trailing.setContentsMargins(0, 0, 0, 0)
        head.addLayout(self.trailing)
        self.box.addLayout(head)
        self.box.addWidget(self.detail)

    def add_trailing(self, widget: QWidget) -> None:
        self.trailing.addWidget(widget)

    def add_control(self, widget: QWidget) -> None:
        self.box.addWidget(widget)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.activated.emit()
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.activated.emit()
            return
        super().keyPressEvent(event)


class EmptyState(GlassCard):
    action_requested = Signal()

    def __init__(self, title: str, detail: str, action: str = "") -> None:
        super().__init__()
        self.box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        heading = QLabel(title)
        heading.setObjectName("cardTitle")
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        copy = QLabel(detail)
        copy.setObjectName("muted")
        copy.setWordWrap(True)
        copy.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.box.addStretch(1)
        self.box.addWidget(heading)
        self.box.addWidget(copy)
        if action:
            button = QPushButton(action)
            button.setObjectName("primary")
            button.clicked.connect(self.action_requested.emit)
            self.box.addWidget(button, 0, Qt.AlignmentFlag.AlignCenter)
        self.box.addStretch(1)
