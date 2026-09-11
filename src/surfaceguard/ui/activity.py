"""Activity history, and the feedback loop that makes tuning possible.

"Not a cat" does two things: it drops the sample that detection fed into the
surface's scale model, and it marks the clip as a regression case (§9). That is the
difference between a tally and a dataset.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..storage.activity_log import ActivityLog, Event
from . import qtutil as Q

COLUMNS = ("When", "Surface", "Sound", "Why", "Latency", "Your call")


class ActivityScreen(QWidget):
    feedback_given = Signal(str, str)      # event_id (as str), verdict
    retention_changed = Signal(int, bool)  # days, save_thumbnails

    def __init__(self, log: ActivityLog) -> None:
        super().__init__()
        self.log = log
        self._events: list[Event] = []

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        head = self.table.horizontalHeader()
        for i in range(len(COLUMNS) - 1):
            head.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        head.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.currentCellChanged.connect(lambda *_: self._show_selected())

        self.thumb = QLabel("Select an event to see the picture.")
        self.thumb.setObjectName("dim")
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb.setMinimumHeight(190)
        self.gates = QLabel("")
        self.gates.setObjectName("mono")
        self.gates.setWordWrap(True)
        self.gates.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.correct_btn = QPushButton("That was a cat")
        self.correct_btn.clicked.connect(lambda: self._feedback("correct"))
        self.wrong_btn = QPushButton("Not a cat")
        self.wrong_btn.setObjectName("danger")
        self.wrong_btn.clicked.connect(lambda: self._feedback("not_a_cat"))

        self.summary = QLabel("")
        self.summary.setObjectName("dim")
        self.keep_days = QSpinBox()
        self.keep_days.setRange(0, 365)
        self.keep_days.setSuffix(" days")
        self.save_thumbs = QCheckBox("Save pictures")
        self.keep_days.valueChanged.connect(self._push_retention)
        self.save_thumbs.toggled.connect(self._push_retention)
        forget = QPushButton("Delete all pictures now")
        forget.setObjectName("danger")
        forget.clicked.connect(self._forget)

        side = QVBoxLayout()
        side.setContentsMargins(12, 12, 12, 12)
        side.setSpacing(8)
        side.addWidget(self.thumb)
        buttons = QHBoxLayout()
        buttons.addWidget(self.correct_btn)
        buttons.addWidget(self.wrong_btn)
        side.addLayout(buttons)
        self.advanced_btn = QPushButton("Show Advanced Details")
        self.advanced_btn.setCheckable(True)
        self.advanced_btn.toggled.connect(self._toggle_advanced)
        side.addWidget(self.advanced_btn)
        self.advanced_caption = QLabel("Detection checks")
        self.advanced_caption.setObjectName("h2")
        side.addWidget(self.advanced_caption)
        side.addWidget(self.gates, 1)
        privacy = QLabel("Privacy")
        privacy.setObjectName("h2")
        side.addWidget(privacy)
        side.addWidget(self.save_thumbs)
        row = QHBoxLayout()
        row.addWidget(QLabel("Keep pictures for"))
        row.addWidget(self.keep_days)
        side.addLayout(row)
        side.addWidget(forget)
        panel = QFrame()
        panel.setObjectName("panel")
        panel.setFixedWidth(330)
        panel.setLayout(side)

        left = QVBoxLayout()
        left.setSpacing(8)
        left.addWidget(self.summary)
        left.addWidget(self.table, 1)

        root = QHBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(14)
        root.addLayout(left, 1)
        root.addWidget(panel)
        self._toggle_advanced(False)
        self._set_feedback_enabled(False)

    # ------------------------------------------------------------------ render

    def set_retention(self, days: int, save: bool) -> None:
        self.keep_days.blockSignals(True)
        self.save_thumbs.blockSignals(True)
        self.keep_days.setValue(days)
        self.save_thumbs.setChecked(save)
        self.keep_days.blockSignals(False)
        self.save_thumbs.blockSignals(False)

    def refresh(self) -> None:
        keep_row = self.table.currentRow()
        self._events = self.log.recent(limit=200, fired_only=True)
        counts = self.log.counts()
        self.summary.setText(
            f"Last 24 hours: {counts['events']} checks, {counts['fired']} sounds played, "
            f"{counts['false_positives']} marked not a cat."
        )
        self.table.setRowCount(len(self._events))
        for row, e in enumerate(self._events):
            cells = (
                e.when,
                e.surface_name,
                "played" if e.fired else "—",
                e.reason,
                f"{e.latency_ms:.0f} ms" if e.latency_ms else "",
                {"correct": "a cat", "not_a_cat": "not a cat"}.get(e.feedback or "", ""),
            )
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col == 2 and e.fired:
                    item.setForeground(Q.BAD)
                if col == 5 and e.feedback == "not_a_cat":
                    item.setForeground(Q.WARN)
                self.table.setItem(row, col, item)
        if 0 <= keep_row < len(self._events):
            self.table.setCurrentCell(keep_row, 0)
        self._show_selected()

    def _toggle_advanced(self, shown: bool) -> None:
        self.advanced_btn.setText("Hide Advanced Details" if shown else "Show Advanced Details")
        self.advanced_caption.setVisible(shown)
        self.gates.setVisible(shown)

    def _selected(self) -> Event | None:
        row = self.table.currentRow()
        return self._events[row] if 0 <= row < len(self._events) else None

    def _show_selected(self) -> None:
        event = self._selected()
        self._set_feedback_enabled(event is not None)
        if event is None:
            self.thumb.setText("Select an event to see the picture.")
            self.gates.setText("")
            return
        if event.thumbnail:
            path = self.log.thumb_dir / event.thumbnail
            pix = QPixmap(str(path)) if Path(path).exists() else QPixmap()
            if not pix.isNull():
                self.thumb.setPixmap(pix.scaledToWidth(
                    300, Qt.TransformationMode.SmoothTransformation))
            else:
                self.thumb.setText("Picture has been deleted.")
        else:
            self.thumb.setPixmap(QPixmap())
            self.thumb.setText("No picture saved for this one.")
        self.gates.setText("\n".join(
            f"{g['name']:10s} {g['status']:12s} {g.get('detail', '')}" for g in event.gates
        ) or "no checks recorded")

    def _set_feedback_enabled(self, on: bool) -> None:
        self.correct_btn.setEnabled(on)
        self.wrong_btn.setEnabled(on)

    # ------------------------------------------------------------------ actions

    def _feedback(self, verdict: str) -> None:
        event = self._selected()
        if event is None:
            return
        self.log.set_feedback(event.id, verdict)
        self.feedback_given.emit(event.surface_id, verdict)
        self.refresh()

    def _push_retention(self) -> None:
        self.retention_changed.emit(self.keep_days.value(), self.save_thumbs.isChecked())

    def _forget(self) -> None:
        self.log.forget_all_thumbnails()
        self.refresh()
