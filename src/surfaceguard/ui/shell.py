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

from .components import NavButton


DESTINATIONS = (
    ("⌂", "Home"),
    ("◇", "Rooms"),
    ("◎", "Detection"),
    ("◖", "Audio"),
    ("▣", "Camera"),
    ("◫", "Devices"),
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

        self.brand_mark = QLabel("◒")
        self.brand_mark.setObjectName("brandMark")
        self.brand_name = QLabel("Surface Guard")
        self.brand_name.setObjectName("brandName")
        brand = QHBoxLayout()
        brand.setSpacing(10)
        brand.addWidget(self.brand_mark)
        brand.addWidget(self.brand_name)
        brand.addStretch(1)
        nav.addLayout(brand)
        nav.addSpacing(18)

        for index, (symbol, name) in enumerate(DESTINATIONS):
            button = NavButton(symbol, name)
            button.clicked.connect(lambda _checked=False, n=name: self.show_page(n))
            self.group.addButton(button, index)
            self.buttons[name] = button
            page = pages[name]
            self.stack.addWidget(page)
            nav.addWidget(button)
        nav.addStretch(1)

        self.health_button = QPushButton("●  Checking systems…")
        self.health_button.setObjectName("healthButton")
        self.health_button.clicked.connect(self.health_requested.emit)
        nav.addWidget(self.health_button)

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
        self.buttons[name].setChecked(True)
        self.page_changed.emit(name)

    def set_health(self, text: str, tone: str) -> None:
        self.health_button.setText(f"●  {text}")
        self.health_button.setProperty("tone", tone)
        self.health_button.style().unpolish(self.health_button)
        self.health_button.style().polish(self.health_button)

    def adapt_to_width(self, width: int) -> None:
        compact = width < 900
        target = 72 if compact else 210
        if self.sidebar.width() == target:
            return
        self.sidebar.setFixedWidth(target)
        self.brand_name.setVisible(not compact)
        for button in self.buttons.values():
            button.set_compact(compact)
        self.health_button.setText("●" if compact else self.health_button.text())
        self.health_button.setToolTip("System Health")

