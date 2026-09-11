"""Room-centric home for cameras, surfaces, audio outputs, and activity."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .components import GlassCard, PageHeader, StatusPill
from .surface_editor import SurfaceEditor


class RoomsScreen(QWidget):
    room_name_changed = Signal(str)

    def __init__(self, editor: SurfaceEditor, room_name: str = "Kitchen") -> None:
        super().__init__()
        self.editor = editor
        self.header = PageHeader(
            "Rooms", "Protected surfaces belong to the room, even when more cameras are added."
        )

        room_card = GlassCard(compact=True)
        self.room_name = QLineEdit(room_name)
        self.room_name.setObjectName("roomName")
        self.room_name.editingFinished.connect(self._push_name)
        self.room_status = StatusPill("Idle", "neutral")
        self.room_summary = QLabel("No protected surfaces yet")
        self.room_summary.setObjectName("muted")
        self.camera_summary = QLabel("No camera connected")
        self.camera_summary.setObjectName("muted")
        title_row = QHBoxLayout()
        title_row.addWidget(self.room_name, 1)
        title_row.addWidget(self.room_status)
        room_card.box.addLayout(title_row)
        room_card.box.addWidget(self.room_summary)
        room_card.box.addWidget(self.camera_summary)

        surfaces_head = QHBoxLayout()
        surfaces_title = QLabel("Protected surfaces")
        surfaces_title.setObjectName("cardTitle")
        self.add_button = QPushButton("Add Protection Zone")
        self.add_button.setObjectName("primary")
        self.add_button.clicked.connect(self.editor.begin_drawing)
        surfaces_head.addWidget(surfaces_title)
        surfaces_head.addStretch(1)
        surfaces_head.addWidget(self.add_button)

        content = QWidget()
        box = QVBoxLayout(content)
        box.setContentsMargins(22, 20, 22, 22)
        box.setSpacing(14)
        box.addWidget(self.header)
        box.addWidget(room_card)
        box.addLayout(surfaces_head)
        box.addWidget(editor, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scroll)

    def _push_name(self) -> None:
        name = self.room_name.text().strip() or "Room"
        self.room_name.setText(name)
        self.room_name_changed.emit(name)

    def refresh_summary(self, surface_count: int, camera_name: str, active: bool) -> None:
        self.room_summary.setText(
            "Nothing is protected yet" if surface_count == 0 else
            f"{surface_count} protection zone{'s' if surface_count != 1 else ''}"
        )
        self.camera_summary.setText(camera_name or "No camera connected")
        self.room_status.set_status("Active" if active else "Idle", "good" if active else "neutral")

