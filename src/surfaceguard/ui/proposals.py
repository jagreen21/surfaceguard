"""Confirming the boxes proposed for cats the detector missed.

``training/propose.py`` says a suggestion "is shown, confirmed, and only then
written". This is the showing. Without it the ``missed`` verdict — the most
valuable label there is, because it marks what the detector cannot currently see —
could never become training data, and the pipeline had no way to finish.

The box can be redrawn, not just accepted or rejected. A proposal that is roughly
right is the common case, and forcing a yes/no on a box that is close but wrong
either loses the example or poisons it.
"""

from __future__ import annotations

from pathlib import Path

import cv2
from PySide6.QtCore import QPoint, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from . import qtutil as Q

HANDLE = 7
MIN_BOX = 8.0


class BoxCanvas(QWidget):
    """One frame with one box on it, redrawable by dragging."""

    box_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(360)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._pixmap = QPixmap()
        self._size = (0, 0)                 # source image size
        self.box: tuple[float, float, float, float] | None = None
        self._drag_from: tuple[float, float] | None = None

    def clear(self) -> None:
        self._pixmap = QPixmap()
        self._size = (0, 0)
        self.box = None
        self.update()

    def load(self, path: Path, box: tuple[float, float, float, float]) -> None:
        image = cv2.imread(str(path)) if path.exists() else None
        if image is None:
            self._pixmap = QPixmap()
            self._size = (0, 0)
        else:
            self._pixmap = Q.bgr_to_pixmap(image)
            self._size = (image.shape[1], image.shape[0])
        self.box = tuple(box)
        self._drag_from = None
        self.update()

    # ---------------------------------------------------------------- mapping

    def _fit(self) -> tuple[float, float, float]:
        if self._pixmap.isNull():
            return 1.0, 0.0, 0.0
        s = min(self.width() / self._pixmap.width(), self.height() / self._pixmap.height())
        return (s,
                (self.width() - self._pixmap.width() * s) / 2,
                (self.height() - self._pixmap.height() * s) / 2)

    def _to_view(self, x: float, y: float) -> tuple[float, float]:
        s, ox, oy = self._fit()
        return x * s + ox, y * s + oy

    def _to_image(self, x: float, y: float) -> tuple[float, float]:
        s, ox, oy = self._fit()
        s = s or 1.0
        w, h = self._size
        return (min(max((x - ox) / s, 0.0), float(w)),
                min(max((y - oy) / s, 0.0), float(h)))

    # ------------------------------------------------------------------ paint

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#0a0f13"))
        if self._pixmap.isNull():
            p.setPen(Q.INK_DIM)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "picture missing")
            return
        s, ox, oy = self._fit()
        p.drawPixmap(QRect(int(ox), int(oy), int(self._pixmap.width() * s),
                           int(self._pixmap.height() * s)), self._pixmap)
        if self.box is None:
            return
        x1, y1 = self._to_view(self.box[0], self.box[1])
        x2, y2 = self._to_view(self.box[2], self.box[3])
        rect = QRectF(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        p.setPen(QPen(Q.CAT, 2.2))
        p.setBrush(QBrush(QColor(240, 193, 75, 40)))
        p.drawRect(rect)
        p.setBrush(QBrush(Q.GROUND))
        for corner in (rect.topLeft(), rect.topRight(), rect.bottomLeft(), rect.bottomRight()):
            p.drawEllipse(corner, HANDLE / 2, HANDLE / 2)
        p.setPen(Q.INK_DIM)
        p.drawText(QRect(10, self.height() - 26, self.width() - 20, 20),
                   Qt.AlignmentFlag.AlignLeft, "drag to redraw the box")

    # ------------------------------------------------------------------ mouse

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            self._drag_from = self._to_image(pos.x(), pos.y())

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_from is None:
            return
        pos = event.position()
        x, y = self._to_image(pos.x(), pos.y())
        x0, y0 = self._drag_from
        self.box = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))
        self.update()

    def mouseReleaseEvent(self, _event) -> None:  # noqa: N802
        if self._drag_from is None:
            return
        self._drag_from = None
        if self.box and (self.box[2] - self.box[0] < MIN_BOX
                         or self.box[3] - self.box[1] < MIN_BOX):
            self.box = None          # a stray click must not become a 2 px label
        self.box_changed.emit()
        self.update()


class ProposalReviewDialog(QDialog):
    """Walk the proposals, confirming, redrawing or rejecting each one."""

    def __init__(self, proposals: list, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Check these before training")
        self.setMinimumSize(720, 560)
        self.proposals = proposals
        self.index = 0

        self.step = QLabel("")
        self.step.setObjectName("mono")
        self.heading = QLabel("Was there a cat here?")
        self.heading.setObjectName("h1")
        self.subtitle = QLabel(
            "These are the cats the detector missed. The box is a guess — accept it, "
            "redraw it, or say there was no cat after all."
        )
        self.subtitle.setObjectName("dim")
        self.subtitle.setWordWrap(True)

        self.canvas = BoxCanvas()
        self.detail = QLabel("")
        self.detail.setObjectName("mono")

        self.yes = QPushButton("Yes, that's the cat  (Y)")
        self.yes.setObjectName("primary")
        self.yes.clicked.connect(lambda: self._answer(True))
        self.no = QPushButton("No cat here  (N)")
        self.no.setObjectName("danger")
        self.no.clicked.connect(lambda: self._answer(False))
        self.skip = QPushButton("Skip")
        self.skip.clicked.connect(lambda: self._answer(None))
        self.done_btn = QPushButton("Finish")
        self.done_btn.clicked.connect(self.accept)

        buttons = QHBoxLayout()
        buttons.addWidget(self.yes)
        buttons.addWidget(self.no)
        buttons.addWidget(self.skip)
        buttons.addStretch(1)
        buttons.addWidget(self.done_btn)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 18)
        root.setSpacing(12)
        root.addWidget(self.step)
        root.addWidget(self.heading)
        root.addWidget(self.subtitle)
        root.addWidget(self.canvas, 1)
        root.addWidget(self.detail)
        root.addLayout(buttons)
        self._show_current()

    # ------------------------------------------------------------------ state

    @property
    def current(self):
        return self.proposals[self.index] if 0 <= self.index < len(self.proposals) else None

    def _show_current(self) -> None:
        item = self.current
        if item is None:
            self.step.setText("ALL DONE")
            self.heading.setText("That's all of them")
            self.canvas.clear()
            self.detail.setText(self._tally())
            for b in (self.yes, self.no, self.skip):
                b.setEnabled(False)
            self.done_btn.setDefault(True)
            return
        self.step.setText(f"{self.index + 1} OF {len(self.proposals)}")
        self.canvas.load(item.image, item.box)
        self.detail.setText(
            f"the detector scored this {item.score:.0%} — below the {0.35:.0%} it acts on"
        )
        for b in (self.yes, self.no, self.skip):
            b.setEnabled(True)

    def _tally(self) -> str:
        yes = sum(1 for p in self.proposals if p.confirmed is True)
        no = sum(1 for p in self.proposals if p.confirmed is False)
        pending = sum(1 for p in self.proposals if p.confirmed is None)
        return f"{yes} confirmed, {no} rejected, {pending} skipped"

    def _answer(self, verdict: bool | None) -> None:
        item = self.current
        if item is not None:
            if verdict and self.canvas.box is None:
                return                      # nothing to confirm; they erased the box
            if verdict:
                item.box = tuple(self.canvas.box)
            item.confirmed = verdict
        self.index += 1
        self._show_current()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        if key in (Qt.Key.Key_Y, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._answer(True)
        elif key == Qt.Key.Key_N:
            self._answer(False)
        elif key == Qt.Key.Key_Space:
            self._answer(None)
        elif key == Qt.Key.Key_Left and self.index > 0:
            self.index -= 1
            self._show_current()
        elif key == Qt.Key.Key_Escape:
            self.accept()
        else:
            super().keyPressEvent(event)


def review_proposals(proposals: list) -> int:
    """Run the dialog standalone. Returns how many were confirmed."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(Q.STYLESHEET)
    dialog = ProposalReviewDialog(proposals)
    dialog.exec()
    return sum(1 for p in proposals if p.confirmed is True)
