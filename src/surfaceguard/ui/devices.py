"""Capability-aware device inventory with contextual health and battery UI."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..state import DeviceKind, DeviceViewState, ProductViewState
from .components import EmptyState, GlassCard, PageHeader, StatusPill


class _DeviceCard(GlassCard):
    def __init__(self, device: DeviceViewState) -> None:
        super().__init__()
        row = QHBoxLayout()
        name = QLabel(device.name); name.setObjectName("cardTitle")
        status = StatusPill("Online" if device.online else "Offline",
                            "good" if device.online else "bad")
        row.addWidget(name); row.addStretch(1); row.addWidget(status)
        self.box.addLayout(row)
        if device.detail:
            detail = QLabel(device.detail); detail.setObjectName("muted"); detail.setWordWrap(True)
            self.box.addWidget(detail)
        traits = []
        caps = device.capabilities
        if caps.camera: traits.append("Camera")
        if caps.microphone: traits.append("Microphone")
        if caps.speaker: traits.append("Speaker")
        if traits:
            label = QLabel("  ·  ".join(traits)); label.setObjectName("muted")
            self.box.addWidget(label)
        if device.battery_percent is not None:
            battery = device.battery_percent
            charging = " · Charging" if device.charging else ""
            text = f"Low Battery · {battery}%" if battery < 15 else f"Battery {battery}%{charging}"
            tone = "bad" if battery < 15 else ("warning" if battery < 30 else "neutral")
            self.box.addWidget(StatusPill(text, tone))
        if not device.online:
            action = QPushButton("Check Device")
            action.setEnabled(False)
            action.setToolTip("Automatic reconnect is already in progress.")
            self.box.addWidget(action)


class DevicesScreen(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._signature: tuple[DeviceViewState, ...] | None = None
        self.header = PageHeader("Devices", "Cameras and audio outputs that help protect your rooms.")
        self.content = QVBoxLayout()
        self.content.setSpacing(12)
        host = QWidget(); host.setLayout(self.content)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(host)
        root = QVBoxLayout(self); root.setContentsMargins(22, 20, 22, 22); root.setSpacing(14)
        root.addWidget(self.header); root.addWidget(scroll, 1)

    def render(self, state: ProductViewState) -> None:
        if state.devices == self._signature:
            return
        self._signature = state.devices
        while self.content.count():
            item = self.content.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        if not state.devices:
            self.content.addWidget(EmptyState(
                "No devices yet", "Connect a camera to start protecting a room.", ""
            ))
            self.content.addStretch(1)
            return
        groups = (
            (DeviceKind.MAIN_UNIT, "Main Unit"),
            (DeviceKind.CAMERA, "Cameras"),
            (DeviceKind.SPEAKER, "Audio Outputs"),
        )
        for kind, title in groups:
            devices = [d for d in state.devices if d.kind is kind]
            if not devices:
                continue
            heading = QLabel(title); heading.setObjectName("cardTitle")
            self.content.addWidget(heading)
            grid = QGridLayout(); grid.setSpacing(12)
            cols = 1 if len(devices) == 1 else 2
            for i, device in enumerate(devices):
                grid.addWidget(_DeviceCard(device), i // cols, i % cols)
            holder = QWidget(); holder.setLayout(grid); self.content.addWidget(holder)
        self.content.addStretch(1)
