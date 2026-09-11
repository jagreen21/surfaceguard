"""Small shared helpers: image conversion and the app's palette."""

from __future__ import annotations

import numpy as np
from PySide6.QtGui import QColor, QFont, QImage, QPixmap

# The approved render uses a near-black blue canvas, cool translucent cards, a
# restrained blue action colour, mint health, and coral detection overlays.
INK = QColor("#edf5f0")
INK_DIM = QColor("#96a4ad")
PANEL = QColor("#17222a")
PANEL_2 = QColor("#202c35")
GROUND = QColor("#081117")
SIDEBAR = QColor("#0d171e")
RULE = QColor("#2a3943")
ACCENT = QColor("#6ea7ff")
ACCENT_DIM = QColor(88, 166, 231, 54)
GOOD = QColor("#62e6a1")
WARN = QColor("#d2a64f")
BAD = QColor("#ff5c57")
CAT = QColor("#ff5c57")
PERSON = QColor("#6fa2d8")

SEVERITY_COLOUR = {"good": GOOD, "waiting": INK_DIM, "problem": BAD}

STYLESHEET = f"""
QWidget {{
    background: {GROUND.name()}; color: {INK.name()};
    font-family: ".AppleSystemUIFont", "SF Pro Text", "Helvetica Neue", sans-serif;
    font-size: 13px;
}}
QLabel, QCheckBox, QRadioButton {{ background: transparent; }}
QFrame#sidebar {{ background: {SIDEBAR.name()}; border-right: 1px solid {RULE.name()}; }}
QFrame#glassCard, QFrame#panel {{
    background: rgba(25, 36, 44, 238);
    border: 1px solid {RULE.name()};
    border-radius: 11px;
}}
QFrame#heroFrame {{
    background: #050b0f;
    border: 1px solid {RULE.name()};
    border-radius: 11px;
}}
QLabel#brandMark {{
    background: #2a3540; color: white; border-radius: 7px;
    min-width: 30px; min-height: 30px; qproperty-alignment: AlignCenter;
    font-size: 17px; font-weight: 600;
}}
QLabel#brandName {{ font-size: 15px; font-weight: 600; }}
QLabel#pageTitle, QLabel#h1 {{ font-size: 27px; font-weight: 600; }}
QLabel#pageSubtitle, QLabel#muted, QLabel#dim {{ color: {INK_DIM.name()}; }}
QLabel#cardTitle, QLabel#h2 {{ font-size: 16px; font-weight: 600; }}
QLabel#cardValue {{ font-size: 19px; font-weight: 600; }}
QLabel#eyebrow {{ color: {INK_DIM.name()}; font-size: 11px; font-weight: 600; }}
QLabel#mono {{
    font-family: Menlo, Monaco, monospace; font-size: 11px; color: {INK_DIM.name()};
}}
QLabel[tone="good"] {{ color: {GOOD.name()}; background: rgba(73,184,121,38); border-radius: 8px; padding: 4px 8px; }}
QLabel[tone="warning"] {{ color: {WARN.name()}; background: rgba(210,166,79,34); border-radius: 8px; padding: 4px 8px; }}
QLabel[tone="bad"] {{ color: {BAD.name()}; background: rgba(237,98,93,36); border-radius: 8px; padding: 4px 8px; }}
QLabel[tone="neutral"] {{ color: {INK_DIM.name()}; background: rgba(156,170,161,24); border-radius: 8px; padding: 4px 8px; }}
QPushButton {{
    min-height: 34px; background: {PANEL_2.name()}; border: 1px solid {RULE.name()};
    border-radius: 9px; padding: 4px 13px;
}}
QPushButton:hover {{ background: #2b3943; border-color: #465762; }}
QPushButton:pressed {{ background: #18242c; }}
QPushButton:disabled {{ color: #65727a; background: #141e25; border-color: #25323a; }}
QPushButton#primary {{
    background: #4f83d9; color: white; font-weight: 600; border: none; padding: 5px 16px;
}}
QPushButton#primary:hover {{ background: #6193e5; }}
QPushButton#secondary {{ color: {ACCENT.name()}; }}
QPushButton#danger {{ color: {BAD.name()}; }}
QPushButton#navButton {{
    min-height: 42px; border: none; border-radius: 10px; background: transparent;
    color: {INK_DIM.name()}; text-align: left; padding: 2px 12px; font-size: 14px;
}}
QPushButton#navButton:hover {{ background: rgba(255,255,255,10); color: {INK.name()}; }}
QPushButton#navButton:checked {{ background: #394650; color: white; }}
QPushButton#navButton[compact="true"] {{ text-align: center; padding: 2px; font-size: 19px; }}
QPushButton#healthButton {{
    min-height: 40px; border: none; background: transparent; text-align: left;
    color: {INK_DIM.name()}; padding: 2px 10px;
}}
QPushButton#healthButton[tone="good"] {{ color: {GOOD.name()}; }}
QPushButton#healthButton[tone="warning"] {{ color: {WARN.name()}; }}
QPushButton#healthButton[tone="bad"] {{ color: {BAD.name()}; }}
QFrame#sideStatus {{
    background: rgba(24, 35, 43, 230); border: 1px solid {RULE.name()};
    border-radius: 11px;
}}
QLabel#sideDevice {{ color: {INK_DIM.name()}; font-size: 11px; }}
QPushButton#toggleButton {{
    min-width: 42px; max-width: 42px; min-height: 24px; max-height: 24px;
    border-radius: 12px; padding: 0; background: #394650; color: white;
}}
QPushButton#toggleButton:checked {{ background: {GOOD.name()}; color: #092018; }}
QLabel#eventRow {{
    min-height: 28px; padding: 3px 5px; border-bottom: 1px solid #26343d;
}}
QPushButton#optionButton {{
    min-height: 48px; text-align: left; padding: 7px 12px; border-radius: 9px;
}}
QPushButton#optionButton:checked {{
    background: #263b4e; border-color: {ACCENT.name()};
}}
QListWidget, QTableWidget, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background: {PANEL_2.name()}; border: 1px solid {RULE.name()}; border-radius: 8px;
    padding: 6px; selection-background-color: #365c8d; selection-color: white;
}}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{ min-height: 30px; }}
QListWidget::item {{ padding: 9px 7px; border-radius: 7px; }}
QListWidget::item:selected {{ background: #334b60; }}
QTableWidget {{ gridline-color: transparent; }}
QTableWidget::item {{ padding: 7px; border-bottom: 1px solid #27322b; }}
QHeaderView::section {{
    background: {PANEL.name()}; border: none; border-bottom: 1px solid {RULE.name()};
    padding: 8px; color: {INK_DIM.name()}; font-size: 11px;
}}
QSlider::groove:horizontal {{ height: 5px; background: {RULE.name()}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT.name()}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    width: 16px; margin: -6px 0; border-radius: 8px; background: white;
}}
QCheckBox {{ spacing: 8px; min-height: 28px; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 9px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #3a4852; border-radius: 4px; min-height: 28px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QToolTip {{ background: #28332c; color: {INK.name()}; border: 1px solid #445047; padding: 5px; }}
"""


def mono(size: int = 12) -> QFont:
    f = QFont("Menlo")
    f.setStyleHint(QFont.StyleHint.Monospace)
    f.setPointSize(size)
    return f


def bgr_to_pixmap(image: np.ndarray) -> QPixmap:
    """OpenCV BGR ndarray -> QPixmap, with the buffer copied so Qt owns it."""
    if image is None or image.size == 0:
        return QPixmap()
    rgb = np.ascontiguousarray(image[:, :, ::-1])
    h, w = rgb.shape[:2]
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)
