"""Small shared helpers: image conversion and the app's palette."""

from __future__ import annotations

import numpy as np
from PySide6.QtGui import QColor, QFont, QImage, QPixmap

# The visual source of truth uses a warm green-black canvas, soft glass panels,
# and three semantic accents. Keep every custom-painted view on this palette.
INK = QColor("#edf5f0")
INK_DIM = QColor("#9caaa1")
PANEL = QColor("#19221c")
PANEL_2 = QColor("#202b24")
GROUND = QColor("#101612")
SIDEBAR = QColor("#151c17")
RULE = QColor("#303b33")
ACCENT = QColor("#58a6e7")
ACCENT_DIM = QColor(88, 166, 231, 54)
GOOD = QColor("#49b879")
WARN = QColor("#d2a64f")
BAD = QColor("#ed625d")
CAT = QColor("#ed625d")
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
    background: rgba(27, 37, 31, 232);
    border: 1px solid {RULE.name()};
    border-radius: 14px;
}}
QFrame#heroFrame {{
    background: #0b100d;
    border: 1px solid {RULE.name()};
    border-radius: 16px;
}}
QLabel#brandMark {{
    background: #2f7d54; color: white; border-radius: 8px;
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
QPushButton:hover {{ background: #28352d; border-color: #44534a; }}
QPushButton:pressed {{ background: #1b251f; }}
QPushButton:disabled {{ color: #657168; background: #18201b; border-color: #263129; }}
QPushButton#primary {{
    background: #2f7d54; color: white; font-weight: 600; border: none; padding: 5px 16px;
}}
QPushButton#primary:hover {{ background: #388d61; }}
QPushButton#secondary {{ color: {ACCENT.name()}; }}
QPushButton#danger {{ color: {BAD.name()}; }}
QPushButton#navButton {{
    min-height: 42px; border: none; border-radius: 10px; background: transparent;
    color: {INK_DIM.name()}; text-align: left; padding: 2px 12px; font-size: 14px;
}}
QPushButton#navButton:hover {{ background: rgba(255,255,255,10); color: {INK.name()}; }}
QPushButton#navButton:checked {{ background: rgba(73,184,121,31); color: #8bdfb2; }}
QPushButton#navButton[compact="true"] {{ text-align: center; padding: 2px; font-size: 19px; }}
QPushButton#healthButton {{
    min-height: 40px; border: none; background: transparent; text-align: left;
    color: {INK_DIM.name()}; padding: 2px 10px;
}}
QPushButton#healthButton[tone="good"] {{ color: {GOOD.name()}; }}
QPushButton#healthButton[tone="warning"] {{ color: {WARN.name()}; }}
QPushButton#healthButton[tone="bad"] {{ color: {BAD.name()}; }}
QListWidget, QTableWidget, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background: {PANEL_2.name()}; border: 1px solid {RULE.name()}; border-radius: 8px;
    padding: 6px; selection-background-color: #315a43; selection-color: white;
}}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{ min-height: 30px; }}
QListWidget::item {{ padding: 9px 7px; border-radius: 7px; }}
QListWidget::item:selected {{ background: #294434; }}
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
QScrollBar::handle:vertical {{ background: #3b483f; border-radius: 4px; min-height: 28px; }}
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
