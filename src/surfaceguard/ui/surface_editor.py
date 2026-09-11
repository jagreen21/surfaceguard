"""The visual surface editor — the core feature.

Surfaces are drawn on the stitched room map, not on a live frame, so a shape is
drawn once and stays correct from every angle the camera can reach (D2). The live
preview then shows the same polygon projected into the current frame, which is
literally what the gates will test.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..geometry.projection import Pose, transform_points
from ..geometry.surface import Surface
from . import qtutil as Q

HANDLE_R = 6
MIN_POINTS = 3


class MapCanvas(QWidget):
    """Pans/zooms nothing — fits the map to the widget and edits polygons on it.

    Deliberately simple: the map is shown whole, so the user never has to navigate
    to find their counter.
    """

    surfaces_changed = Signal()
    selection_changed = Signal(object)  # Surface | None

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(300)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._map: QPixmap = QPixmap()
        self.surfaces: list[Surface] = []
        self.selected: Surface | None = None
        self.drawing: list[tuple[float, float]] = []
        self.draw_mode = False
        self._drag: tuple[Surface, int] | None = None
        self._hover_point: QPointF | None = None
        # Live overlay: the current frame's pose, so the editor can show which
        # part of the map the camera is looking at right now.
        self.live_pose: Pose | None = None

    # ------------------------------------------------------------------ data

    def set_map(self, canvas: np.ndarray | None) -> None:
        self._map = Q.bgr_to_pixmap(canvas) if canvas is not None else QPixmap()
        self.update()

    def set_surfaces(self, surfaces: list[Surface]) -> None:
        self.surfaces = surfaces
        if self.selected is not None and self.selected not in surfaces:
            self.select(None)
        self.update()

    def select(self, surface: Surface | None) -> None:
        self.selected = surface
        self.selection_changed.emit(surface)
        self.update()

    def start_drawing(self) -> None:
        self.draw_mode = True
        self.drawing = []
        self.select(None)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()

    def cancel_drawing(self) -> None:
        self.draw_mode = False
        self.drawing = []
        self.unsetCursor()
        self.update()

    def finish_drawing(self, name: str) -> Surface | None:
        if len(self.drawing) < MIN_POINTS:
            return None
        surface = Surface(name=name, polygon=np.asarray(self.drawing, float))
        self.surfaces.append(surface)
        self.cancel_drawing()
        self.select(surface)
        self.surfaces_changed.emit()
        return surface

    # ------------------------------------------------------------- transform

    def _fit(self) -> tuple[float, QPoint]:
        """Scale and top-left offset that fits the map inside this widget."""
        if self._map.isNull():
            return 1.0, QPoint(0, 0)
        sx = self.width() / self._map.width()
        sy = self.height() / self._map.height()
        s = min(sx, sy)
        w, h = self._map.width() * s, self._map.height() * s
        return s, QPoint(int((self.width() - w) / 2), int((self.height() - h) / 2))

    def map_to_view(self, pt) -> QPointF:
        s, off = self._fit()
        return QPointF(pt[0] * s + off.x(), pt[1] * s + off.y())

    def view_to_map(self, pos: QPointF) -> tuple[float, float]:
        s, off = self._fit()
        s = s or 1.0
        return ((pos.x() - off.x()) / s, (pos.y() - off.y()) / s)

    # ------------------------------------------------------------------ paint

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), Q.GROUND)

        if self._map.isNull():
            p.setPen(Q.INK_DIM)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "No room map yet.\nRun setup to scan the room.")
            return

        s, off = self._fit()
        p.drawPixmap(QRect(off.x(), off.y(), int(self._map.width() * s),
                           int(self._map.height() * s)), self._map)

        if self.live_pose is not None:
            self._paint_live_footprint(p)
        for surface in self.surfaces:
            self._paint_surface(p, surface)
        if self.draw_mode:
            self._paint_in_progress(p)

    def _paint_live_footprint(self, p: QPainter) -> None:
        """Where the camera is pointing right now, as a box on the map."""
        pose = self.live_pose
        assert pose is not None
        w, h = pose.frame_size
        corners = transform_points(pose.frame_to_map, [(0, 0), (w, 0), (w, h), (0, h)])
        poly = QPolygonF([self.map_to_view(c) for c in corners])
        p.setPen(QPen(Q.ACCENT, 1.4, Qt.PenStyle.DashLine))
        p.setBrush(QBrush(QColor(79, 180, 207, 26)))
        p.drawPolygon(poly)
        p.setPen(Q.ACCENT)
        f = p.font(); f.setPointSize(9); p.setFont(f)
        p.drawText(poly.boundingRect().adjusted(6, 4, 0, 0), Qt.AlignmentFlag.AlignTop,
                   "camera is here")

    def _paint_surface(self, p: QPainter, surface: Surface) -> None:
        selected = surface is self.selected
        poly = QPolygonF([self.map_to_view(pt) for pt in surface.polygon])
        if surface.enabled:
            edge = Q.ACCENT if selected else Q.INK
            fill = QColor(79, 180, 207, 70) if selected else QColor(230, 237, 242, 34)
        else:
            edge, fill = Q.INK_DIM, QColor(154, 168, 179, 22)

        p.setPen(QPen(edge, 2.2 if selected else 1.6,
                      Qt.PenStyle.SolidLine if surface.enabled else Qt.PenStyle.DashLine))
        p.setBrush(QBrush(fill))
        p.drawPolygon(poly)

        label = surface.name if surface.enabled else f"{surface.name} (off)"
        p.setPen(Q.INK if surface.enabled else Q.INK_DIM)
        f = p.font(); f.setPointSize(11); f.setBold(selected); p.setFont(f)
        centre = poly.boundingRect().center()
        p.drawText(QRect(int(centre.x()) - 90, int(centre.y()) - 9, 180, 18),
                   Qt.AlignmentFlag.AlignCenter, label)

        if selected:
            p.setPen(QPen(Q.ACCENT, 1.5))
            p.setBrush(QBrush(Q.GROUND))
            for pt in surface.polygon:
                v = self.map_to_view(pt)
                p.drawEllipse(v, HANDLE_R, HANDLE_R)

    def _paint_in_progress(self, p: QPainter) -> None:
        pts = [self.map_to_view(pt) for pt in self.drawing]
        if self._hover_point is not None and pts:
            pts_preview = pts + [self._hover_point]
        else:
            pts_preview = pts
        if len(pts_preview) >= 2:
            p.setPen(QPen(Q.CAT, 2.0, Qt.PenStyle.DashLine))
            p.setBrush(QBrush(QColor(240, 193, 75, 46)))
            p.drawPolygon(QPolygonF(pts_preview))
        p.setPen(QPen(Q.CAT, 1.5))
        p.setBrush(QBrush(Q.GROUND))
        for v in pts:
            p.drawEllipse(v, HANDLE_R - 1, HANDLE_R - 1)
        if pts:
            p.setPen(Q.INK_DIM)
            p.drawText(QRect(10, self.height() - 30, self.width() - 20, 20),
                       Qt.AlignmentFlag.AlignLeft,
                       f"{len(self.drawing)} corners — click to add, "
                       "double-click or press Enter to finish")

    # ------------------------------------------------------------------ mouse

    def mousePressEvent(self, event) -> None:  # noqa: N802
        pos = event.position()
        if self.draw_mode:
            if event.button() == Qt.MouseButton.LeftButton:
                self.drawing.append(self.view_to_map(pos))
                self.update()
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return
        # A handle beats a body hit, so corners stay draggable inside the shape.
        if self.selected is not None:
            for i, pt in enumerate(self.selected.polygon):
                if (self.map_to_view(pt) - pos).manhattanLength() <= HANDLE_R * 2.4:
                    self._drag = (self.selected, i)
                    return
        for surface in reversed(self.surfaces):
            poly = QPolygonF([self.map_to_view(pt) for pt in surface.polygon])
            if poly.containsPoint(pos, Qt.FillRule.OddEvenFill):
                self.select(surface)
                return
        self.select(None)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self.draw_mode:
            self._hover_point = event.position()
            self.update()
            return
        if self._drag is None:
            return
        surface, index = self._drag
        surface.polygon[index] = self.view_to_map(event.position())
        self.update()

    def mouseReleaseEvent(self, _event) -> None:  # noqa: N802
        if self._drag is not None:
            self._drag = None
            self.surfaces_changed.emit()

    def mouseDoubleClickEvent(self, _event) -> None:  # noqa: N802
        if self.draw_mode and len(self.drawing) >= MIN_POINTS:
            self.parent().commit_drawing()  # type: ignore[attr-defined]

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        if self.draw_mode:
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.parent().commit_drawing()  # type: ignore[attr-defined]
            elif key == Qt.Key.Key_Escape:
                self.cancel_drawing()
            elif key == Qt.Key.Key_Backspace and self.drawing:
                self.drawing.pop()
                self.update()
            return
        if key in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete) and self.selected is not None:
            self.parent().delete_selected()  # type: ignore[attr-defined]


class SurfaceEditor(QWidget):
    """Map canvas, surface list, and the per-surface deterrent settings."""

    surfaces_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.canvas = MapCanvas(self)
        self.canvas.surfaces_changed.connect(self._on_canvas_changed)
        self.canvas.selection_changed.connect(self._on_selected)

        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._on_row)

        self.add_btn = QPushButton("Draw a new surface")
        self.add_btn.setObjectName("primary")
        self.add_btn.clicked.connect(self.begin_drawing)
        self.done_btn = QPushButton("Finish shape")
        self.done_btn.clicked.connect(self.commit_drawing)
        self.done_btn.setVisible(False)
        self.del_btn = QPushButton("Delete")
        self.del_btn.setObjectName("danger")
        self.del_btn.clicked.connect(self.delete_selected)

        self.settings = _DeterrentPanel()
        self.settings.changed.connect(self._on_canvas_changed)

        left = QVBoxLayout()
        left.setSpacing(8)
        hint = QLabel("Surfaces are drawn on the room, so they keep working when the "
                      "camera turns.")
        hint.setObjectName("dim")
        hint.setWordWrap(True)
        left.addWidget(hint)
        left.addWidget(self.canvas, 1)

        right = QVBoxLayout()
        right.setSpacing(8)
        title = QLabel("Protected surfaces")
        title.setObjectName("h2")
        right.addWidget(title)
        right.addWidget(self.list, 1)
        row = QHBoxLayout()
        row.addWidget(self.add_btn)
        row.addWidget(self.done_btn)
        row.addWidget(self.del_btn)
        right.addLayout(row)
        right.addWidget(self.settings)

        holder = QFrame()
        holder.setObjectName("panel")
        holder.setFixedWidth(320)
        holder.setLayout(right)

        root = QHBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(14)
        root.addLayout(left, 1)
        root.addWidget(holder)
        self._refresh_buttons()

    # ------------------------------------------------------------------ data

    def load(self, canvas: np.ndarray | None, surfaces: list[Surface]) -> None:
        self.canvas.set_map(canvas)
        self.canvas.set_surfaces(surfaces)
        self._rebuild_list()

    @property
    def surfaces(self) -> list[Surface]:
        return self.canvas.surfaces

    def set_live_pose(self, pose: Pose | None) -> None:
        self.canvas.live_pose = pose
        self.canvas.update()

    # -------------------------------------------------------------- drawing

    def begin_drawing(self) -> None:
        self.canvas.start_drawing()
        self.canvas.setFocus()
        self._refresh_buttons()

    def commit_drawing(self) -> None:
        if len(self.canvas.drawing) < MIN_POINTS:
            QMessageBox.information(
                self, "Keep going",
                "A surface needs at least three corners. Click around the edge of the "
                "counter or table, then press Enter.",
            )
            return
        name = f"Surface {len(self.canvas.surfaces) + 1}"
        surface = self.canvas.finish_drawing(name)
        self._rebuild_list()
        self._refresh_buttons()
        if surface is not None:
            self.settings.name.setFocus()
            self.settings.name.selectAll()

    def delete_selected(self) -> None:
        surface = self.canvas.selected
        if surface is None:
            return
        confirm = QMessageBox.question(
            self, "Delete this surface?",
            f"“{surface.name}” will stop being protected.",
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self.canvas.surfaces.remove(surface)
        self.canvas.select(None)
        self._rebuild_list()
        self.surfaces_changed.emit()

    # --------------------------------------------------------------- plumbing

    def _rebuild_list(self) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        for surface in self.canvas.surfaces:
            state = "" if surface.enabled else "  (paused)"
            calib = surface.calibration_samples
            note = "learning sizes" if calib < 8 else "size check on"
            item = QListWidgetItem(f"{surface.name}{state}\n{note} · {calib} samples")
            self.list.addItem(item)
        if self.canvas.selected in self.canvas.surfaces:
            self.list.setCurrentRow(self.canvas.surfaces.index(self.canvas.selected))
        self.list.blockSignals(False)

    def _on_row(self, row: int) -> None:
        if 0 <= row < len(self.canvas.surfaces):
            self.canvas.select(self.canvas.surfaces[row])

    def _on_selected(self, surface: Surface | None) -> None:
        self.settings.bind(surface)
        self._refresh_buttons()
        if surface in self.canvas.surfaces:
            row = self.canvas.surfaces.index(surface)
            if self.list.currentRow() != row:
                self.list.blockSignals(True)
                self.list.setCurrentRow(row)
                self.list.blockSignals(False)

    def _on_canvas_changed(self) -> None:
        self._rebuild_list()
        self.surfaces_changed.emit()

    def _refresh_buttons(self) -> None:
        drawing = self.canvas.draw_mode
        self.add_btn.setVisible(not drawing)
        self.done_btn.setVisible(drawing)
        self.del_btn.setEnabled(self.canvas.selected is not None and not drawing)


class _DeterrentPanel(QFrame):
    """Per-surface settings, in the user's vocabulary rather than the model's."""

    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("panel")
        self.surface: Surface | None = None

        self.name = QLineEdit()
        self.name.setPlaceholderText("Kitchen counter")
        self.enabled = QCheckBox("Protect this surface")
        self.sound = QComboBox()
        self.sound.addItems(["chirp", "clack", "hiss", "warble"])
        self.test = QPushButton("Test sound")
        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setRange(0, 100)
        self.delay = QComboBox()
        self.delay.addItems(["Right away", "After 1 second", "After 2 seconds", "After 5 seconds"])
        self.cooldown = QDoubleSpinBox()
        self.cooldown.setRange(2.0, 300.0)
        self.cooldown.setSuffix(" s")
        self.cooldown.setSingleStep(5.0)
        self.vary = QCheckBox("Change the sound each time")

        form = QVBoxLayout(self)
        form.setContentsMargins(12, 12, 12, 12)
        form.setSpacing(7)
        for label, widget in (
            ("Name", self.name),
            ("", self.enabled),
            ("Sound", self.sound),
            ("Volume", self.volume),
            ("Play the sound", self.delay),
            ("Wait before repeating", self.cooldown),
            ("", self.vary),
        ):
            if label:
                cap = QLabel(label)
                cap.setObjectName("dim")
                form.addWidget(cap)
            form.addWidget(widget)
        form.addWidget(self.test)
        self.status = QLabel("")
        self.status.setObjectName("mono")
        self.status.setWordWrap(True)
        form.addWidget(self.status)

        self.name.editingFinished.connect(self._push)
        self.enabled.toggled.connect(self._push)
        self.sound.currentTextChanged.connect(self._push)
        self.volume.valueChanged.connect(self._push)
        self.delay.currentIndexChanged.connect(self._push)
        self.cooldown.valueChanged.connect(self._push)
        self.vary.toggled.connect(self._push)
        self.bind(None)

    def bind(self, surface: Surface | None) -> None:
        self.surface = None      # suppress _push while loading
        self.setEnabled(surface is not None)
        if surface is None:
            self.name.setText("")
            self.status.setText("Select a surface, or draw a new one.")
            return
        self.name.setText(surface.name)
        self.enabled.setChecked(surface.enabled)
        self.sound.setCurrentText(surface.deterrent.sound)
        self.volume.setValue(int(surface.deterrent.volume * 100))
        self.delay.setCurrentIndex({0.0: 0, 1.0: 1, 2.0: 2, 5.0: 3}.get(surface.deterrent.delay_s, 0))
        self.cooldown.setValue(surface.deterrent.cooldown_s)
        self.vary.setChecked(surface.deterrent.vary_sound)
        k = surface.size_scale
        self.status.setText(
            f"size check: {surface.calibration_samples} samples"
            + (f", k={k:.0f}" if k else ", still learning")
        )
        self.surface = surface

    def _push(self) -> None:
        s = self.surface
        if s is None:
            return
        s.name = self.name.text().strip() or s.name
        s.enabled = self.enabled.isChecked()
        s.deterrent.sound = self.sound.currentText()
        s.deterrent.volume = self.volume.value() / 100.0
        s.deterrent.delay_s = [0.0, 1.0, 2.0, 5.0][self.delay.currentIndex()]
        s.deterrent.cooldown_s = float(self.cooldown.value())
        s.deterrent.vary_sound = self.vary.isChecked()
        self.changed.emit()
