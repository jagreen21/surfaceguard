"""Response and output routing without exposing audio backend details."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..geometry.surface import Surface
from .components import GlassCard, PageHeader, StatusPill


class AudioScreen(QWidget):
    settings_changed = Signal()
    test_requested = Signal()
    output_changed = Signal(bool)  # prefer camera speaker
    import_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._surfaces: list[Surface] = []
        self._loading = False
        self.header = PageHeader("Audio", "Choose the sound and where it should play.")

        response = GlassCard()
        rh = QHBoxLayout()
        title = QLabel("Sound response"); title.setObjectName("cardTitle")
        self.ready = StatusPill("Ready", "good")
        rh.addWidget(title); rh.addStretch(1); rh.addWidget(self.ready)
        response.box.addLayout(rh)

        self.surface = QComboBox()
        self.surface.setAccessibleName("Surface")
        self.surface.currentIndexChanged.connect(self._load_surface)
        self.sound = QComboBox()
        self.sound.setAccessibleName("Sound response")
        self.sound.currentTextChanged.connect(self._push)
        self.import_button = QPushButton("Import custom sound…")
        self.import_button.clicked.connect(self.import_requested.emit)
        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setAccessibleName("Sound volume")
        self.volume.setRange(0, 100); self.volume.valueChanged.connect(self._push)
        self.delay = QComboBox()
        self.delay.setAccessibleName("Response delay")
        self.delay.addItems(["Immediately", "After 1 second", "After 2 seconds", "After 5 seconds"])
        self.delay.currentIndexChanged.connect(self._push)
        cap = QLabel("Surface"); cap.setObjectName("eyebrow")
        response.box.addWidget(cap); response.box.addWidget(self.surface)
        for label, widget in (("Sound", self.sound), ("Volume", self.volume),
                              ("Play the sound", self.delay)):
            cap = QLabel(label); cap.setObjectName("eyebrow")
            response.box.addWidget(cap); response.box.addWidget(widget)
        response.box.addWidget(self.import_button)
        actions = QHBoxLayout()
        test = QPushButton("Test sound"); test.setObjectName("primary")
        test.clicked.connect(self.test_requested.emit)
        actions.addStretch(1); actions.addWidget(test)
        response.box.addLayout(actions)

        output = GlassCard()
        oh = QLabel("Audio output"); oh.setObjectName("cardTitle")
        output.box.addWidget(oh)
        self.destination = QComboBox()
        self.destination.setAccessibleName("Audio output")
        self.destination.currentIndexChanged.connect(self._push_output)
        output.box.addWidget(self.destination)
        self.output_note = QLabel("This Mac is the dependable default output.")
        self.output_note.setObjectName("muted"); self.output_note.setWordWrap(True)
        output.box.addWidget(self.output_note)

        content = QWidget(); box = QVBoxLayout(content)
        box.setContentsMargins(22, 20, 22, 22); box.setSpacing(14)
        box.addWidget(self.header); box.addWidget(response); box.addWidget(output)
        box.addStretch(1)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(content)
        root = QVBoxLayout(self); root.setContentsMargins(0, 0, 0, 0); root.addWidget(scroll)

    def bind(self, surfaces: list[Surface], sounds: list[str], *, camera_speaker: bool,
             prefer_camera: bool, output_available: bool = True) -> None:
        current = self.surface.currentIndex()
        self._surfaces = surfaces
        self._loading = True
        self.surface.blockSignals(True)
        self.sound.blockSignals(True)
        self.destination.blockSignals(True)
        self.surface.clear()
        self.surface.addItems([s.name for s in surfaces] or ["No surfaces"])
        self.surface.setEnabled(bool(surfaces))
        self.sound.clear(); self.sound.addItems(sounds)
        self.destination.clear()
        self.destination.addItem("This Mac", False)
        if camera_speaker:
            self.destination.addItem("Camera speaker", True)
        idx = self.destination.findData(prefer_camera and camera_speaker)
        self.destination.setCurrentIndex(max(0, idx))
        if not surfaces:
            self.ready.set_status("Add a surface", "neutral")
        elif not output_available:
            self.ready.set_status("Output unavailable", "bad")
        else:
            self.ready.set_status("Ready", "good")
        self.output_note.setText(
            "Responses will play through the camera speaker."
            if self.destination.currentData() else
            "This Mac is the dependable default output."
        )
        if surfaces:
            self.surface.setCurrentIndex(min(max(0, current), len(surfaces) - 1))
            self._load_surface()
        self.surface.blockSignals(False)
        self.sound.blockSignals(False)
        self.destination.blockSignals(False)
        self._loading = False

    def _load_surface(self) -> None:
        i = self.surface.currentIndex()
        if not (0 <= i < len(self._surfaces)):
            return
        s = self._surfaces[i]
        was_loading = self._loading
        self._loading = True
        self.sound.setCurrentText(s.deterrent.sound)
        self.volume.setValue(round(s.deterrent.volume * 100))
        self.delay.setCurrentIndex({0.0: 0, 1.0: 1, 2.0: 2, 5.0: 3}.get(s.deterrent.delay_s, 0))
        self._loading = was_loading

    def _push(self) -> None:
        if self._loading:
            return
        i = self.surface.currentIndex()
        if not (0 <= i < len(self._surfaces)):
            return
        s = self._surfaces[i]
        s.deterrent.sound = self.sound.currentText()
        s.deterrent.volume = self.volume.value() / 100.0
        s.deterrent.delay_s = [0.0, 1.0, 2.0, 5.0][self.delay.currentIndex()]
        self.settings_changed.emit()

    def _push_output(self) -> None:
        if not self._loading and self.destination.currentIndex() >= 0:
            self.output_changed.emit(bool(self.destination.currentData()))

    @property
    def current_surface(self) -> Surface | None:
        index = self.surface.currentIndex()
        return self._surfaces[index] if 0 <= index < len(self._surfaces) else None
