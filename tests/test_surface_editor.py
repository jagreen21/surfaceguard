"""The surface editor.

The polygon is what every gate tests against, so how accurately it can be drawn
is not a cosmetic question. These cover the parts that decide accuracy: exactly
four corners, so the plane model is exact rather than approximated; grabbing the
corner you aimed at; and repairing a shape instead of redrawing it.
"""

import numpy as np
import pytest
from PySide6.QtCore import QPointF

from surfaceguard.geometry.projection import polygon_area
from surfaceguard.geometry.surface import Surface
from surfaceguard.ui.surface_editor import HANDLE_R, QUAD_POINTS, MapCanvas


@pytest.fixture
def canvas(qt_app):
    c = MapCanvas()
    c.resize(800, 600)
    c.set_map(np.zeros((600, 800, 3), np.uint8))
    return c


def square(x=100.0, y=100.0, size=200.0, name="Counter") -> Surface:
    return Surface(name, np.array([[x, y], [x + size, y], [x + size, y + size],
                                   [x, y + size]], float))


# --------------------------------------------------------------- quad drawing


def test_four_corners_are_enough_for_an_exact_plane_model():
    """Surface.quad falls back to _extreme_quad for anything but four points, and
    that moves every corner — so the homography predicting how tall a cat should
    look is built from a shape the user never drew."""
    exact = square()
    assert np.allclose(exact.quad, exact.polygon), "four points should be used as-is"

    drawn = np.array([[100, 100], [300, 98], [500, 100], [520, 180],
                      [300, 190], [80, 180]], float)
    approximated = Surface("Counter", drawn)
    assert not np.allclose(approximated.quad[:4], drawn[:4])
    assert polygon_area(approximated.quad) != pytest.approx(polygon_area(drawn), rel=0.01)


def test_quad_mode_announces_when_four_corners_are_placed(canvas):
    fired = []
    canvas.quad_complete.connect(lambda: fired.append(True))
    canvas.start_drawing(quad=True)
    for point in ((100, 100), (300, 100), (300, 250), (100, 250)):
        canvas.mousePressEvent(_click(QPointF(*point)))
    assert len(canvas.drawing) == QUAD_POINTS
    assert fired, "placing the fourth corner should offer to finish"


def test_freeform_mode_keeps_collecting_points(canvas):
    fired = []
    canvas.quad_complete.connect(lambda: fired.append(True))
    canvas.start_drawing(quad=False)
    for point in ((100, 100), (300, 100), (300, 250), (100, 250), (90, 180)):
        canvas.mousePressEvent(_click(QPointF(*point)))
    assert len(canvas.drawing) == 5
    assert not fired


# ------------------------------------------------------------- hit testing


def test_a_corner_is_grabbed_by_straight_line_distance(canvas):
    """Manhattan distance makes the grab area a diamond, so a corner is catchable
    from further away diagonally than straight on."""
    surface = square()
    canvas.set_surfaces([surface])
    canvas.select(surface)

    corner = canvas.map_to_view(surface.polygon[0])
    near = QPointF(corner.x() + HANDLE_R, corner.y() + HANDLE_R)   # diagonal
    far = QPointF(corner.x() + HANDLE_R * 4, corner.y())
    assert canvas._handle_at(corner) is not None
    assert canvas._handle_at(near) is not None
    assert canvas._handle_at(far) is None


def test_corners_of_unselected_surfaces_are_grabbable(canvas):
    """Otherwise adjusting a neighbour means selecting it first, which is not
    discoverable."""
    a, b = square(name="Counter"), square(x=400.0, name="Table")
    canvas.set_surfaces([a, b])
    canvas.select(a)
    found = canvas._handle_at(canvas.map_to_view(b.polygon[0]))
    assert found is not None and found[0] is b


# --------------------------------------------------------- repairing a shape


def test_a_corner_can_be_added_to_an_edge(canvas):
    surface = square()
    canvas.set_surfaces([surface])
    canvas.select(surface)
    before = len(surface.polygon)
    midpoint = (surface.polygon[0] + surface.polygon[1]) / 2.0
    assert canvas.insert_vertex_at(canvas.map_to_view(midpoint))
    assert len(surface.polygon) == before + 1


def test_adding_a_corner_away_from_any_edge_does_nothing(canvas):
    surface = square()
    canvas.set_surfaces([surface])
    canvas.select(surface)
    centre = surface.polygon.mean(axis=0)
    assert not canvas.insert_vertex_at(canvas.map_to_view(centre))
    assert len(surface.polygon) == 4


def test_a_corner_can_be_removed_but_never_below_a_triangle(canvas):
    surface = square()
    canvas.set_surfaces([surface])
    canvas.select(surface)
    assert canvas.delete_vertex(surface, 0)
    assert len(surface.polygon) == 3
    assert not canvas.delete_vertex(surface, 0), "a polygon needs three corners"
    assert len(surface.polygon) == 3


def test_every_shape_change_is_undoable(canvas):
    surface = square()
    canvas.set_surfaces([surface])
    canvas.select(surface)
    original = surface.polygon.copy()

    midpoint = (surface.polygon[0] + surface.polygon[1]) / 2.0
    canvas.insert_vertex_at(canvas.map_to_view(midpoint))
    assert len(surface.polygon) == 5
    canvas.undo_last_edit()
    assert np.allclose(surface.polygon, original), "inserting a corner was not undone"

    canvas.delete_vertex(surface, 0)
    canvas.undo_last_edit()
    assert np.allclose(surface.polygon, original), "deleting a corner was not undone"


# ------------------------------------------------------------------ coupling


def test_the_canvas_does_not_reach_into_its_parent():
    """self.parent().commit_drawing() breaks silently the moment the canvas is
    reparented, and cannot be tested in isolation."""
    import inspect

    from surfaceguard.ui import surface_editor

    source = inspect.getsource(surface_editor.MapCanvas)
    # Strip comments, or this trips on the comment explaining why the calls went.
    code = "\n".join(line.split("#")[0] for line in source.splitlines())
    assert "self.parent()" not in code
    for signal in ("commit_requested", "delete_requested", "quad_complete"):
        assert hasattr(surface_editor.MapCanvas, signal)


class _Click:
    """The two things mousePressEvent asks of an event.

    A real QMouseEvent needs a QPointingDevice and its other constructors are
    deprecated; the handler only calls position() and button().
    """

    def __init__(self, pos: QPointF):
        from PySide6.QtCore import Qt

        self._pos = pos
        self._button = Qt.MouseButton.LeftButton

    def position(self) -> QPointF:
        return self._pos

    def button(self):
        return self._button


def _click(pos: QPointF) -> _Click:
    return _Click(pos)
