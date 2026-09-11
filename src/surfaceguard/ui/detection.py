"""Plain-language detection controls with technical tuning disclosed on demand."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..state import AppState, Phase
from .components import GlassCard, PageHeader, StatusPill


class DetectionScreen(QWidget):
    protection_requested = Signal(bool)
    sensitivity_changed = Signal(str)
    cooldown_changed = Signal(float)

    def __init__(self) -> None:
        super().__init__()
        self.header = PageHeader("Detection", "Choose how Surface Guard decides when to respond.")

        protection = GlassCard()
        title = QLabel("Protection")
        title.setObjectName("cardTitle")
        self.status = StatusPill("Off", "neutral")
        self.toggle = QPushButton("Turn protection on")
        self.toggle.setObjectName("primary")
        self.toggle.clicked.connect(self._toggle)
        row = QHBoxLayout()
        row.addWidget(title)
        row.addStretch(1)
        row.addWidget(self.status)
        row.addWidget(self.toggle)
        protection.box.addLayout(row)
        note = QLabel("Surface Guard only responds when the camera, detector, and sound are ready.")
        note.setObjectName("muted")
        note.setWordWrap(True)
        protection.box.addWidget(note)

        behaviour = GlassCard()
        bh = QLabel("Detection behavior")
        bh.setObjectName("cardTitle")
        behaviour.box.addWidget(bh)
        self.sensitivity = QComboBox()
        self.sensitivity.addItems(["Calm", "Balanced", "Sensitive"])
        self.sensitivity.currentTextChanged.connect(self.sensitivity_changed.emit)
        self.cooldown = QDoubleSpinBox()
        self.cooldown.setRange(2.0, 300.0)
        self.cooldown.setSuffix(" seconds")
        self.cooldown.setSingleStep(5.0)
        self.cooldown.valueChanged.connect(self.cooldown_changed.emit)
        for label, control, detail in (
            ("Sensitivity", self.sensitivity, "Balanced works well in most rooms."),
            ("Cooldown", self.cooldown, "Minimum wait before another response."),
        ):
            cap = QLabel(label)
            cap.setObjectName("eyebrow")
            help_ = QLabel(detail)
            help_.setObjectName("muted")
            row = QHBoxLayout()
            copy = QVBoxLayout(); copy.addWidget(cap); copy.addWidget(help_)
            row.addLayout(copy, 1); row.addWidget(control)
            behaviour.box.addLayout(row)

        self.advanced_button = QPushButton("Advanced")
        self.advanced_button.setCheckable(True)
        self.advanced_button.clicked.connect(self._show_advanced)
        self.advanced = QFrame()
        self.advanced.setObjectName("glassCard")
        abox = QVBoxLayout(self.advanced)
        self.detector_info = QLabel("Detector information will appear here.")
        self.detector_info.setObjectName("muted")
        self.detector_info.setWordWrap(True)
        abox.addWidget(self.detector_info)
        self.advanced.setVisible(False)

        content = QWidget()
        box = QVBoxLayout(content)
        box.setContentsMargins(22, 20, 22, 22)
        box.setSpacing(14)
        box.addWidget(self.header)
        box.addWidget(protection)
        box.addWidget(behaviour)
        box.addWidget(self.advanced_button)
        box.addWidget(self.advanced)
        box.addStretch(1)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(content)
        root = QVBoxLayout(self); root.setContentsMargins(0, 0, 0, 0); root.addWidget(scroll)
        self._state: AppState | None = None

    def set_values(self, sensitivity: str, cooldown: float, detector_text: str) -> None:
        self.sensitivity.blockSignals(True)
        self.sensitivity.setCurrentText(sensitivity)
        self.sensitivity.blockSignals(False)
        self.cooldown.blockSignals(True)
        self.cooldown.setValue(cooldown)
        self.cooldown.blockSignals(False)
        self.detector_info.setText(detector_text)

    def render_state(self, state: AppState) -> None:
        self._state = state
        on = state.phase in (Phase.GUARDING, Phase.ALERTING, Phase.STARTING, Phase.PROBLEM)
        tone = "bad" if state.phase is Phase.PROBLEM else ("good" if on else "neutral")
        self.status.set_status(state.headline, tone)
        self.toggle.setText("Turn protection off" if on else "Turn protection on")

    def _toggle(self) -> None:
        on = self._state is not None and self._state.phase in (
            Phase.GUARDING, Phase.ALERTING, Phase.STARTING, Phase.PROBLEM
        )
        self.protection_requested.emit(not on)

    def _show_advanced(self, shown: bool) -> None:
        self.advanced.setVisible(shown)
        self.advanced_button.setText("Hide Advanced" if shown else "Advanced")

