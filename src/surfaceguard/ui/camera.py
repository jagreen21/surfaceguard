"""Live monitoring, honest overlays, and only supported camera actions."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .components import GlassCard, PageHeader, StatusPill
from .home import LiveView


class CameraScreen(QWidget):
    connect_requested = Signal()
    snapshot_requested = Signal()
    camera_sound_requested = Signal()
    overlays_changed = Signal(bool, bool, bool)

    def __init__(self) -> None:
        super().__init__()
        self.live = LiveView()
        self.header = PageHeader("Live view", "Watch the room and use supported camera controls.")
        self.connect = QPushButton("Connect camera")
        self.connect.clicked.connect(self.connect_requested.emit)
        self.header.add_action(self.connect)

        self.camera_name = QLabel("Camera")
        self.camera_name.setObjectName("cardTitle")
        self.camera_detail = QLabel("Not connected")
        self.camera_detail.setObjectName("muted")
        self.camera_status = StatusPill("Offline", "bad")
        cam_row = QHBoxLayout()
        copy = QVBoxLayout(); copy.addWidget(self.camera_name); copy.addWidget(self.camera_detail)
        cam_row.addLayout(copy); cam_row.addStretch(1); cam_row.addWidget(self.camera_status)

        hero = QFrame(); hero.setObjectName("heroFrame")
        hbox = QVBoxLayout(hero); hbox.setContentsMargins(1, 1, 1, 1); hbox.addWidget(self.live)

        controls = GlassCard(compact=True)
        action_row = QHBoxLayout()
        self.snapshot = QPushButton("Take snapshot")
        self.snapshot.setObjectName("primary")
        self.snapshot.clicked.connect(self.snapshot_requested.emit)
        self.camera_sound = QPushButton("Test sound")
        self.camera_sound.clicked.connect(self.camera_sound_requested.emit)
        action_row.addWidget(self.snapshot); action_row.addWidget(self.camera_sound); action_row.addStretch(1)
        controls.box.addLayout(action_row)
        overlays = QHBoxLayout()
        self.zones = QCheckBox("Surfaces")
        self.boxes = QCheckBox("Detection boxes")
        self.labels = QCheckBox("Surface labels")
        for check in (self.zones, self.boxes, self.labels):
            check.toggled.connect(self._emit_overlays)
            overlays.addWidget(check)
        overlays.addStretch(1)
        controls.box.addLayout(overlays)

        root = QVBoxLayout(self)
        root.setContentsMargins(22, 20, 22, 22); root.setSpacing(14)
        root.addWidget(self.header); root.addLayout(cam_row); root.addWidget(hero, 1); root.addWidget(controls)

    def set_camera(self, name: str, model: str, *, online: bool, has_speaker: bool,
                   zones: bool, boxes: bool, labels: bool) -> None:
        self.camera_name.setText(name or "Camera")
        self.camera_detail.setText(model or "Camera model unavailable")
        self.camera_status.set_status("Online" if online else "Offline", "good" if online else "bad")
        self.live.set_camera_name(model or name)
        self.snapshot.setEnabled(online)
        self.camera_sound.setVisible(has_speaker)
        self.connect.setText("Change camera" if name else "Connect camera")
        for control, value in ((self.zones, zones), (self.boxes, boxes), (self.labels, labels)):
            control.blockSignals(True); control.setChecked(value); control.blockSignals(False)
        self.live.set_overlays(zones, boxes, labels)

    def _emit_overlays(self) -> None:
        values = (self.zones.isChecked(), self.boxes.isChecked(), self.labels.isChecked())
        self.live.set_overlays(*values)
        self.overlays_changed.emit(*values)
