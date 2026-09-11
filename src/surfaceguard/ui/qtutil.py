"""Small shared helpers: image conversion and the app's palette."""

from __future__ import annotations

import numpy as np
from PySide6.QtGui import QColor, QFont, QImage, QPixmap

# One place for colour, so the editor overlay, the home screen and the
# diagnostics view agree on what "good", "warning" and "problem" look like.
INK = QColor("#e6edf2")
INK_DIM = QColor("#9aa8b3")
PANEL = QColor("#161f26")
PANEL_2 = QColor("#1d2831")
GROUND = QColor("#0e1419")
RULE = QColor("#2a363f")
ACCENT = QColor("#4fb4cf")
ACCENT_DIM = QColor(79, 180, 207, 60)
GOOD = QColor("#5fbf94")
WARN = QColor("#d6a441")
BAD = QColor("#e0795f")
CAT = QColor("#f0c14b")
PERSON = QColor("#6f9ad6")

SEVERITY_COLOUR = {"good": GOOD, "waiting": INK_DIM, "problem": BAD}

STYLESHEET = f"""
QWidget {{ background: {GROUND.name()}; color: {INK.name()};
           font-family: ".AppleSystemUIFont", "Helvetica Neue", Helvetica, sans-serif;
           font-size: 13px; }}
/* Labels inherit the panel behind them instead of painting the window ground. */
QLabel, QCheckBox {{ background: transparent; }}
QFrame#panel {{ background: {PANEL.name()}; border: 1px solid {RULE.name()}; border-radius: 6px; }}
QLabel#h1 {{ font-size: 26px; font-weight: 600; }}
QLabel#h2 {{ font-size: 16px; font-weight: 600; }}
QLabel#dim {{ color: {INK_DIM.name()}; }}
QLabel#mono {{ font-family: Menlo, Monaco, monospace; font-size: 12px; color: {INK_DIM.name()}; }}
QPushButton {{ background: {PANEL_2.name()}; border: 1px solid {RULE.name()};
               border-radius: 5px; padding: 7px 14px; }}
QPushButton:hover {{ border-color: {ACCENT.name()}; }}
QPushButton:disabled {{ color: {INK_DIM.name()}; border-color: {RULE.name()}; }}
QPushButton#primary {{ background: {ACCENT.name()}; color: #0b1418; font-weight: 600;
                       border: none; padding: 9px 18px; }}
QPushButton#danger {{ color: {BAD.name()}; }}
QListWidget, QTableWidget, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background: {PANEL_2.name()}; border: 1px solid {RULE.name()}; border-radius: 5px;
    padding: 4px; selection-background-color: {ACCENT.name()}; selection-color: #0b1418; }}
QListWidget::item {{ padding: 6px 4px; }}
QTabBar::tab {{ background: transparent; padding: 9px 16px; color: {INK_DIM.name()};
                border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {INK.name()}; border-bottom: 2px solid {ACCENT.name()}; }}
QTabWidget::pane {{ border: none; }}
QSlider::groove:horizontal {{ height: 4px; background: {RULE.name()}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 14px; margin: -6px 0; border-radius: 7px;
                              background: {ACCENT.name()}; }}
QCheckBox {{ spacing: 7px; }}
QHeaderView::section {{ background: {PANEL.name()}; border: none;
                        border-bottom: 1px solid {RULE.name()}; padding: 6px;
                        color: {INK_DIM.name()}; font-size: 11px; }}
QScrollBar:vertical {{ background: transparent; width: 9px; }}
QScrollBar::handle:vertical {{ background: {RULE.name()}; border-radius: 4px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
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
