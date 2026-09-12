"""Application chrome: adaptive sidebar, system health, and page stack."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .components import BrandMark, NavButton


DESTINATIONS = (
    ("⌂", "Home"),
    ("◇", "Rooms"),
    ("◎", "Detection"),
    ("◖", "Audio"),
    ("▣", "Camera"),
    ("◫", "Devices"),
    ("◷", "Activity"),
    ("⚙", "Settings"),
)


class AppShell(QWidget):
    page_changed = Signal(str)
    health_requested = Signal()

    def __init__(self, pages: dict[str, QWidget]) -> None:
        super().__init__()
        self.pages = pages
        self.stack = QStackedWidget()
        self.buttons: dict[str, NavButton] = {}
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(210)
        nav = QVBoxLayout(self.sidebar)
        nav.setContentsMargins(12, 16, 12, 14)
        nav.setSpacing(5)

        self.brand_mark = BrandMark()
        self.brand_name = QLabel("Surface Guard")
        self.brand_name.setObjectName("brandName")
        brand = QHBoxLayout()
        brand.setSpacing(10)
        brand.addWidget(self.brand_mark)
        brand.addWidget(self.brand_name)
        brand.addStretch(1)
        nav.addLayout(brand)
        nav.addSpacing(14)

        for index, (symbol, name) in enumerate(DESTINATIONS):
            if name not in pages:
                continue
            button = NavButton(symbol, name)
            button.clicked.connect(lambda _checked=False, n=name: self.show_page(n))
            self.group.addButton(button, index)
            self.buttons[name] = button
            page = pages[name]
            self.stack.addWidget(page)
            nav.addWidget(button)
        for name, page in pages.items():
            if name not in self.buttons:
                self.stack.addWidget(page)
        nav.addStretch(1)

        self.status_panel = QFrame()
        self.status_panel.setObjectName("sideStatus")
        self.status_box = QVBoxLayout(self.status_panel)
        self.status_box.setContentsMargins(8, 8, 8, 8)
        self.status_box.setSpacing(2)
        self.health_button = QPushButton("Checking systems…")
        self.health_button.setObjectName("healthButton")
        self.health_button.clicked.connect(self.health_requested.emit)
        self._health_text = "Checking systems…"
        self._health_tone = "neutral"
        self._compact = False
        self.status_box.addWidget(self.health_button)
        self.device_labels: list[QLabel] = []
        nav.addWidget(self.status_panel)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.sidebar)
        layout.addWidget(self.stack, 1)
        self.show_page("Home")

    @property
    def current_name(self) -> str:
        for name, page in self.pages.items():
            if page is self.stack.currentWidget():
                return name
        return "Home"

    def show_page(self, name: str) -> None:
        if name not in self.pages:
            return
        self.stack.setCurrentWidget(self.pages[name])
        if name in self.buttons:
            self.buttons[name].setChecked(True)
        else:
            self.group.setExclusive(False)
            for button in self.buttons.values():
                button.setChecked(False)
            self.group.setExclusive(True)
        self.page_changed.emit(name)

    def set_health(self, text: str, tone: str) -> None:
        self._health_text = text
        self._health_tone = tone
        self.health_button.setText("Health" if self._compact else text)
        self.health_button.setProperty("tone", tone)
        self.health_button.style().unpolish(self.health_button)
        self.health_button.style().polish(self.health_button)

    def set_inventory(self, devices: list) -> None:
        """Render every current device instead of assuming a four-item system."""
        while len(self.device_labels) < len(devices):
            label = QLabel("")
            label.setObjectName("sideDevice")
            self.device_labels.append(label)
            self.status_box.addWidget(label)
        for label, device in zip(self.device_labels, devices):
            status = "Online" if device.online else "Offline"
            if device.battery_percent is not None:
                status = f"{device.battery_percent}%"
            label.setText(f"{device.name}  ·  {status}")
            label.setToolTip(f"{device.name}: {status}")
            label.setVisible(not self._compact)
        for label in self.device_labels[len(devices):]:
            label.setText("")
            label.setVisible(False)

    def adapt_to_width(self, width: int) -> None:
        compact = width < 900
        target = 72 if compact else 210
        if self._compact == compact and self.sidebar.width() == target:
            return
        self._compact = compact
        self.sidebar.setFixedWidth(target)
        self.brand_name.setVisible(not compact)
        for button in self.buttons.values():
            button.set_compact(compact)
        for label in self.device_labels:
            label.setVisible(not compact and bool(label.text()))
        self.health_button.setText("Health" if compact else self._health_text)
        self.health_button.setToolTip(f"System Health: {self._health_text}")
        available = max(0, width - target)
        for page in self.pages.values():
            adapt = getattr(page, "adapt_to_width", None)
            if callable(adapt):
                adapt(available)
