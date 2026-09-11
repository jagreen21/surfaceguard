"""The weekly review: the app asks to be graded, and shows what the grade changed.

The shape is borrowed from a CAPTCHA — quick image judgements, batched, almost no
reading — and the power dynamic is inverted. A CAPTCHA blocks you until you prove
something about yourself. This blocks nothing: it is not modal, it never suspends
protection, and quitting halfway is a legitimate ending that keeps every answer
already given. The user is the examiner; the app is the one being marked.

Two acts, because the two kinds of question deserve different interfaces.

*Act one* is the alerts, one at a time. Each played a sound at somebody, so each
gets room, context and a strip of frames rather than a still.

*Act two* is the silent calls, as a grid. They are cheap, individually low-stakes
and scanned rather than studied, which is exactly the interaction the grid form is
good at. "None of these" sits at the same weight as "Done" on purpose — hide it,
and people invent cats to be agreeable, which poisons the data the whole feature
exists to collect.

The third screen is the one that decides whether anyone ever does this twice: what
their answers changed, in their words, each item individually reversible.
"""

from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..geometry.surface import Surface
from ..storage.activity_log import ActivityLog
from ..storage.review_policy import (
    Adjustment,
    Candidate,
    Deck,
    Outcome,
    plan_adjustments,
    weekly_accuracy,
)
from . import qtutil as Q
from .components import GlassCard, StatusPill

STRIP_MS = 260  # how fast the frames cycle; slow enough to read, fast enough to move


class StripView(QWidget):
    """The evidence: a few frames around the moment, cycling by themselves.

    Autoplay rather than a scrubber the user has to discover. The cases this
    review selects are the ambiguous ones, and ambiguity is usually resolved by
    motion — asking someone to find and drag a control first would mean most
    people judge from a single still, which is the thing we were avoiding.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._frames: list[QPixmap] = []
        self._index = 0
        self._playing = True
        self._message = ""
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

    def set_frames(self, paths: list[Path], message: str = "") -> None:
        self._frames = [p for p in (QPixmap(str(x)) for x in paths) if not p.isNull()]
        self._index = 0
        self._message = message or ("No picture was kept for this one."
                                    if not self._frames else "")
        self._timer.stop()
        if len(self._frames) > 1 and self._playing:
            self._timer.start(STRIP_MS)
        self.update()

    def toggle_play(self) -> None:
        self._playing = not self._playing
        if self._playing and len(self._frames) > 1:
            self._timer.start(STRIP_MS)
        else:
            self._timer.stop()
        self.update()

    @property
    def playing(self) -> bool:
        return self._playing and len(self._frames) > 1

    def _advance(self) -> None:
        if self._frames:
            self._index = (self._index + 1) % len(self._frames)
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if len(self._frames) > 1:
            self.toggle_play()
        super().mouseReleaseEvent(event)

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor("#0b100d"))
        if not self._frames:
            p.setPen(Q.INK_DIM)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._message)
            return

        pix = self._frames[self._index]
        scaled = pix.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        p.drawPixmap(x, y, scaled)

        if len(self._frames) > 1:
            # Position dots, bottom centre: enough to show that this is moving
            # footage and roughly where in it we are, without a scrubber's chrome.
            total = len(self._frames)
            width = total * 14
            ox = (self.width() - width) // 2
            oy = self.height() - 18
            for i in range(total):
                on = i == self._index
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(Q.INK if on else QColor(255, 255, 255, 70))
                p.drawEllipse(ox + i * 14, oy, 7 if on else 5, 7 if on else 5)
            if not self._playing:
                p.setPen(Q.INK_DIM)
                p.drawText(self.rect().adjusted(0, 0, -12, -8),
                           Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
                           "paused — click to play")


class Sparkline(QWidget):
    """Five weeks of "how often was I right", as a shape rather than a table."""

    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(26)
        self.setMinimumWidth(80)
        self._values: list[float | None] = []

    def set_values(self, values: list[float | None]) -> None:
        self._values = values
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        if not self._values:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        n = len(self._values)
        slot = self.width() / max(1, n)
        bar = max(4.0, slot - 5)
        for i, value in enumerate(self._values):
            x = i * slot
            if value is None:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(255, 255, 255, 26))
                p.drawRoundedRect(int(x), self.height() - 4, int(bar), 3, 1.5, 1.5)
                continue
            h = max(3.0, value * (self.height() - 3))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(Q.GOOD if value >= 0.8 else (Q.WARN if value >= 0.5 else Q.BAD))
            p.drawRoundedRect(int(x), int(self.height() - h), int(bar), int(h), 2, 2)


TILE_SIZE = QSize(216, 150)


class GridTile(QWidget):
    """One picture in act two. Selected means "yes, there is a cat on it".

    A plain widget rather than a QPushButton: the app's stylesheet gives buttons a
    ``min-height``, and QStyleSheetStyle's polish writes that back over anything
    ``setFixedSize`` asked for, which flattens a 150 px tile to a 44 px sliver.
    Nothing of the button is used here anyway — every pixel is painted below.
    """

    toggled = Signal()

    def __init__(self, candidate: Candidate, thumb_dir: Path, number: int) -> None:
        super().__init__()
        self.candidate = candidate
        self.number = number
        self._checked = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(TILE_SIZE)
        self.setToolTip(f"{candidate.event.long_when} · {candidate.reason}")
        source = candidate.event.strip or ([candidate.event.thumbnail]
                                           if candidate.event.thumbnail else [])
        self._pix = QPixmap(str(thumb_dir / source[-1])) if source else QPixmap()

    def sizeHint(self) -> QSize:  # noqa: N802
        return TILE_SIZE

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return TILE_SIZE

    def isChecked(self) -> bool:  # noqa: N802
        return self._checked

    def setChecked(self, on: bool) -> None:  # noqa: N802
        if on != self._checked:
            self._checked = on
            self.toggled.emit()
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.pos()):
            self.setChecked(not self._checked)
        super().mouseReleaseEvent(event)

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect().adjusted(1, 1, -1, -1)
        p.fillRect(r, QColor("#0b100d"))
        # Clip first, so "expanding" genuinely crops to a filled thumbnail
        # instead of painting past the tile's rounded edge.
        clip = QPainterPath()
        clip.addRoundedRect(float(r.x()), float(r.y()), float(r.width()),
                            float(r.height()), 10, 10)
        p.setClipPath(clip)
        if not self._pix.isNull():
            scaled = self._pix.scaled(r.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                      Qt.TransformationMode.SmoothTransformation)
            p.drawPixmap(r.x() + (r.width() - scaled.width()) // 2,
                         r.y() + (r.height() - scaled.height()) // 2, scaled)
        else:
            p.setPen(Q.INK_DIM)
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "picture gone")
        p.setClipping(False)

        if self.candidate.count > 1:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, 165))
            p.drawRoundedRect(r.x() + 6, r.y() + 6, 74, 19, 9, 9)
            p.setPen(Q.INK)
            f = p.font(); f.setPointSize(9); p.setFont(f)
            p.drawText(r.x() + 6, r.y() + 6, 74, 19, Qt.AlignmentFlag.AlignCenter,
                       f"×{self.candidate.count} times")

        # The number is the keyboard shortcut. Shown always, so the shortcut is
        # discoverable rather than a thing power users find in a changelog.
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 150))
        p.drawEllipse(r.right() - 26, r.bottom() - 26, 20, 20)
        p.setPen(Q.INK_DIM)
        p.drawText(r.right() - 26, r.bottom() - 26, 20, 20,
                   Qt.AlignmentFlag.AlignCenter, str(self.number))

        pen = QPen(Q.GOOD if self.isChecked() else QColor("#39453d"),
                   3 if self.isChecked() else 1)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(r, 10, 10)
        if self.isChecked():
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(Q.GOOD)
            p.drawEllipse(r.right() - 32, r.y() + 8, 24, 24)
            p.setPen(QPen(QColor("white"), 2.4))
            p.drawLine(r.right() - 27, r.y() + 20, r.right() - 23, r.y() + 24)
            p.drawLine(r.right() - 23, r.y() + 24, r.right() - 15, r.y() + 13)


class AlertPage(QWidget):
    """Act one: one alert at a time, with the context to judge it."""

    answered = Signal(str)     # verdict
    undo_requested = Signal()

    # The three follow-ups map onto the gates, so "no" becomes attributable to a
    # named condition instead of a shrug. They appear only after "no": asking
    # everyone to categorise every alert up front would be four decisions where
    # one will do.
    FOLLOW_UPS = (
        ("Cat, but not on it", "not_on_surface"),
        ("A person", "person"),
        ("Not a cat at all", "not_a_cat"),
    )

    def __init__(self, thumb_dir: Path) -> None:
        super().__init__()
        self.thumb_dir = thumb_dir
        self._candidate: Candidate | None = None

        self.eyebrow = QLabel("")
        self.eyebrow.setObjectName("eyebrow")
        self.progress = QLabel("")
        self.progress.setObjectName("muted")
        self.repeat = StatusPill("", "warning")
        self.repeat.setVisible(False)
        self.strip = StripView()

        self.question = QLabel("I played the sound. Was that right?")
        self.question.setObjectName("cardTitle")

        self.yes = QPushButton("✓  Yes, cat on it")
        self.yes.setObjectName("primary")
        self.yes.clicked.connect(lambda: self.answered.emit("correct"))
        self.no = QPushButton("✗  No")
        self.no.clicked.connect(self._show_follow_ups)
        self.unsure = QPushButton("Not sure")
        self.unsure.setObjectName("secondary")
        self.unsure.clicked.connect(lambda: self.answered.emit("unsure"))

        self.follow_row = QWidget()
        self.follow_row.setStyleSheet("background: transparent;")
        follow = QHBoxLayout(self.follow_row)
        follow.setContentsMargins(0, 0, 0, 0)
        follow.setSpacing(8)
        what = QLabel("What was it?")
        what.setObjectName("muted")
        follow.addWidget(what)
        for label, verdict in self.FOLLOW_UPS:
            button = QPushButton(label)
            button.clicked.connect(lambda _=False, v=verdict: self.answered.emit(v))
            follow.addWidget(button)
        follow.addStretch(1)
        self.follow_row.setVisible(False)

        self.undo = QPushButton("↩  Undo")
        self.undo.setObjectName("secondary")
        self.undo.clicked.connect(self.undo_requested.emit)
        self.why = QPushButton("Why am I being asked this?")
        self.why.setObjectName("secondary")
        self.why.setCheckable(True)
        self.why.toggled.connect(self._toggle_why)
        self.why_text = QLabel("")
        self.why_text.setObjectName("mono")
        self.why_text.setWordWrap(True)
        self.why_text.setVisible(False)

        card = GlassCard()
        head = QHBoxLayout()
        head.addWidget(self.eyebrow)
        head.addStretch(1)
        head.addWidget(self.repeat)
        head.addWidget(self.progress)
        card.box.addLayout(head)
        card.box.addWidget(self.strip, 1)
        card.box.addWidget(self.question)
        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addWidget(self.yes)
        buttons.addWidget(self.no)
        buttons.addWidget(self.unsure)
        buttons.addStretch(1)
        card.box.addLayout(buttons)
        card.box.addWidget(self.follow_row)
        footer = QHBoxLayout()
        footer.addWidget(self.why)
        footer.addStretch(1)
        footer.addWidget(self.undo)
        card.box.addLayout(footer)
        card.box.addWidget(self.why_text)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(card)

    def show_candidate(self, candidate: Candidate, index: int, total: int,
                       can_undo: bool) -> None:
        self._candidate = candidate
        event = candidate.event
        self.eyebrow.setText(f"{event.long_when}  ·  {event.surface_name}".upper())
        self.progress.setText(f"{index + 1} / {total}")
        self.repeat.setVisible(candidate.count > 1)
        self.repeat.set_status(candidate.repeat_note, "warning")
        paths = [self.thumb_dir / n for n in event.strip]
        if not paths and event.thumbnail:
            paths = [self.thumb_dir / event.thumbnail]
        self.strip.set_frames(paths)
        self.follow_row.setVisible(False)
        self.why.setChecked(False)
        self.undo.setVisible(can_undo)
        gates = "\n".join(
            f"{g['name']:18s} {g['status']:12s} {g.get('detail', '')}" for g in event.gates
        )
        self.why_text.setText(f"{candidate.reason}.\n\n{gates or 'no checks recorded'}")

    def _show_follow_ups(self) -> None:
        self.follow_row.setVisible(True)

    def _toggle_why(self, on: bool) -> None:
        self.why_text.setVisible(on)


class GridPage(QWidget):
    """Act two: the silent calls, scanned rather than studied."""

    # Signal(object), not Signal(dict): a dict signal marshals through
    # QVariantMap, which silently drops integer keys — and these are event ids.
    submitted = Signal(object)   # dict of event_id -> verdict
    _COLUMNS = 3

    def __init__(self, thumb_dir: Path) -> None:
        super().__init__()
        self.thumb_dir = thumb_dir
        self.tiles: list[GridTile] = []

        self.prompt = QLabel("")
        self.prompt.setObjectName("cardTitle")
        self.counter = QLabel("")
        self.counter.setObjectName("muted")

        self.grid = QGridLayout()
        self.grid.setSpacing(12)
        holder = QWidget()
        holder.setStyleSheet("background: transparent;")
        holder.setLayout(self.grid)

        self.none_btn = QPushButton("None of these")
        self.none_btn.clicked.connect(self._submit_none)
        self.done_btn = QPushButton("Done")
        self.done_btn.setObjectName("primary")
        self.done_btn.clicked.connect(self._submit)

        card = GlassCard()
        head = QHBoxLayout()
        head.addWidget(self.prompt)
        head.addStretch(1)
        head.addWidget(self.counter)
        card.box.addLayout(head)
        card.box.addWidget(holder)
        card.box.addStretch(1)
        row = QHBoxLayout()
        row.setSpacing(8)
        # Equal weight, deliberately. A quiet "none of these" is the answer that
        # keeps the data honest, and burying it teaches people to find cats that
        # were never there.
        row.addWidget(self.none_btn)
        row.addStretch(1)
        row.addWidget(self.done_btn)
        card.box.addLayout(row)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(card)

    def load(self, candidates: list[Candidate]) -> None:
        for tile in self.tiles:
            tile.setParent(None)
        self.tiles = []
        names = {c.event.surface_name for c in candidates}
        where = f"the {names.pop().lower()}" if len(names) == 1 else "a surface you protect"
        self.prompt.setText(f"Tap every picture with a cat on {where}.")
        self.counter.setText(f"{len(candidates)} to check")
        for i, candidate in enumerate(candidates):
            tile = GridTile(candidate, self.thumb_dir, i + 1)
            self.grid.addWidget(tile, i // self._COLUMNS, i % self._COLUMNS)
            self.tiles.append(tile)

    def toggle(self, number: int) -> None:
        if 1 <= number <= len(self.tiles):
            tile = self.tiles[number - 1]
            tile.setChecked(not tile.isChecked())

    def _submit_none(self) -> None:
        for tile in self.tiles:
            tile.setChecked(False)
        self._submit()

    def _submit(self) -> None:
        labels: dict[int, str] = {}
        for tile in self.tiles:
            # Not selected is a real answer, not a skipped one: it says the app
            # was right to stay quiet, which is half of what the score measures.
            verdict = "missed" if tile.isChecked() else "not_on_surface"
            for event_id in tile.candidate.ids:
                labels[event_id] = verdict
        self.submitted.emit(labels)


class SummaryPage(QWidget):
    """Act three, and the only reason anyone does act one twice.

    People abandon labelling loops because nothing visibly happens. So this screen
    is specific about what moved, attaches an undo to each item, and never invents
    a change to look busy — "you agreed with all of them" is a real result and is
    said as one.
    """

    tuning_changed = Signal()
    done_clicked = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._outcome: Outcome | None = None
        self._surfaces: dict[str, Surface] = {}

        self.headline = QLabel("")
        self.headline.setObjectName("pageTitle")
        self.headline.setWordWrap(True)

        self.items = QVBoxLayout()
        self.items.setSpacing(10)
        items_holder = QWidget()
        items_holder.setStyleSheet("background: transparent;")
        items_holder.setLayout(self.items)

        self.score = QLabel("")
        self.score.setObjectName("muted")
        self.spark = Sparkline()
        self.trend_note = QLabel("five weeks")
        self.trend_note.setObjectName("eyebrow")

        self.detail_btn = QPushButton("What changed, exactly")
        self.detail_btn.setObjectName("secondary")
        self.detail_btn.setCheckable(True)
        self.detail_btn.toggled.connect(self._toggle_detail)
        self.detail_text = QLabel("")
        self.detail_text.setObjectName("mono")
        self.detail_text.setWordWrap(True)
        self.detail_text.setVisible(False)

        done = QPushButton("Done")
        done.setObjectName("primary")
        done.clicked.connect(self.done_clicked.emit)

        card = GlassCard()
        card.box.addWidget(self.headline)
        card.box.addWidget(items_holder)
        card.box.addStretch(1)
        trend = QHBoxLayout()
        trend.setSpacing(10)
        trend.addWidget(self.score)
        trend.addWidget(self.spark)
        trend.addWidget(self.trend_note)
        trend.addStretch(1)
        card.box.addLayout(trend)
        card.box.addWidget(self.detail_text)
        footer = QHBoxLayout()
        footer.addWidget(self.detail_btn)
        footer.addStretch(1)
        footer.addWidget(done)
        card.box.addLayout(footer)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(card)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scroll)

    def show_outcome(self, outcome: Outcome, surfaces: list[Surface],
                     trend: list[tuple[int, int]]) -> None:
        self._outcome = outcome
        self._surfaces = {s.id: s for s in surfaces}
        self.headline.setText(outcome.headline())

        while self.items.count():
            item = self.items.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        for adjustment in outcome.adjustments:
            surface = self._surfaces.get(adjustment.surface_id)
            if surface is not None:
                adjustment.apply(surface)
            self.items.addWidget(self._row(adjustment))
        if outcome.adjustments:
            self.tuning_changed.emit()

        if outcome.regression_cases:
            note = QLabel(f"{outcome.regression_cases} saved as test cases — "
                          f"I can be replayed against them after any change.")
            note.setObjectName("muted")
            note.setWordWrap(True)
            self.items.addWidget(note)

        if outcome.graded:
            self.score.setText(
                f"Right {outcome.right} of {outcome.graded} this week."
                + (f"  {outcome.unsure} you couldn't call." if outcome.unsure else "")
            )
        else:
            self.score.setText("Nothing graded this time.")
        self.spark.set_values([None if g == 0 else r / g for r, g in trend])
        self.detail_text.setText(self._detail_text(outcome))

    def _row(self, adjustment: Adjustment) -> QWidget:
        frame = QFrame()
        frame.setStyleSheet("background: transparent;")
        box = QHBoxLayout(frame)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(10)
        dot = QLabel("●")
        dot.setStyleSheet(
            f"color: {(Q.GOOD if adjustment.reversible else Q.INK_DIM).name()};")
        dot.setAlignment(Qt.AlignmentFlag.AlignTop)
        copy = QVBoxLayout()
        copy.setSpacing(2)
        title = QLabel(adjustment.title)
        title.setObjectName("cardTitle")
        title.setWordWrap(True)
        detail = QLabel(adjustment.detail)
        detail.setObjectName("muted")
        detail.setWordWrap(True)
        copy.addWidget(title)
        copy.addWidget(detail)
        box.addWidget(dot)
        box.addLayout(copy, 1)

        if adjustment.reversible:
            button = QPushButton("Undo")
            button.setObjectName("secondary")
            button.clicked.connect(lambda _=False, a=adjustment, b=button: self._toggle(a, b))
            box.addWidget(button, 0, Qt.AlignmentFlag.AlignTop)
        return frame

    def _toggle(self, adjustment: Adjustment, button: QPushButton) -> None:
        surface = self._surfaces.get(adjustment.surface_id)
        if surface is None:
            return
        if adjustment.applied:
            adjustment.undo(surface)
            button.setText("Redo")
        else:
            adjustment.apply(surface)
            button.setText("Undo")
        self.tuning_changed.emit()
        if self._outcome is not None:
            self.detail_text.setText(self._detail_text(self._outcome))

    def _toggle_detail(self, on: bool) -> None:
        self.detail_text.setVisible(on)

    @staticmethod
    def _detail_text(outcome: Outcome) -> str:
        lines = []
        for a in outcome.adjustments:
            if not a.reversible:
                lines.append(f"{a.surface_id}  no change  ({len(a.event_ids)} events)")
                continue
            state = "applied" if a.applied else "undone"
            lines.append(
                f"{a.surface_id}  {a.kind}  {state}\n"
                f"    was:  {a.previous or '—'}\n"
                f"    now:  {a.payload}"
            )
        return "\n".join(lines) or "No parameters changed."


class ReviewDialog(QDialog):
    """The whole flow. Deliberately not modal, and cheap to abandon."""

    tuning_changed = Signal()
    finished_review = Signal()

    def __init__(
        self,
        deck: Deck,
        surfaces: list[Surface],
        log: ActivityLog,
        pictures_are_temporary: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.deck = deck
        self.surfaces = surfaces
        self.log = log
        self.pictures_are_temporary = pictures_are_temporary
        self.labels: dict[int, str] = {}
        self.outcome: Outcome | None = None
        self._index = 0
        self._history: list[Candidate] = []
        self._batch = log.next_batch_id()

        self.setWindowTitle("Week in review")
        self.setModal(False)      # protection is never suspended behind this
        self.resize(760, 660)

        self.banner = StatusPill("", "bad")
        self.banner.setVisible(False)
        self.alert_page = AlertPage(log.thumb_dir)
        self.grid_page = GridPage(log.thumb_dir)
        self.summary_page = SummaryPage()
        self.stack = QStackedWidget()
        for page in (self.alert_page, self.grid_page, self.summary_page):
            self.stack.addWidget(page)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 16)
        root.setSpacing(10)
        root.addWidget(self.banner, 0, Qt.AlignmentFlag.AlignHCenter)
        root.addWidget(self.stack, 1)

        self.alert_page.answered.connect(self._answer_alert)
        self.alert_page.undo_requested.connect(self._undo_alert)
        self.grid_page.submitted.connect(self._answer_grid)
        self.summary_page.tuning_changed.connect(self.tuning_changed.emit)
        self.summary_page.done_clicked.connect(self.accept)

        self._start()

    # ------------------------------------------------------------------- flow

    def _start(self) -> None:
        if self.deck.alerts:
            self._show_alert()
        else:
            self._begin_grid()

    def _show_alert(self) -> None:
        self.stack.setCurrentWidget(self.alert_page)
        self.alert_page.show_candidate(
            self.deck.alerts[self._index], self._index, len(self.deck.alerts),
            can_undo=bool(self._history),
        )

    def _answer_alert(self, verdict: str) -> None:
        candidate = self.deck.alerts[self._index]
        self._label(candidate, verdict)
        self._history.append(candidate)
        self._index += 1
        if self._index < len(self.deck.alerts):
            self._show_alert()
        else:
            self._begin_grid()

    def _undo_alert(self) -> None:
        if not self._history:
            return
        candidate = self._history.pop()
        for event_id in candidate.ids:
            self.labels.pop(event_id, None)
            self.log.set_feedback(event_id, "", batch=None)
        self._index = max(0, self._index - 1)
        self._show_alert()

    def _begin_grid(self) -> None:
        if not self.deck.grid:
            self._finish()
            return
        self.grid_page.load(self.deck.grid)
        self.stack.setCurrentWidget(self.grid_page)

    def _answer_grid(self, labels: dict[int, str]) -> None:
        for event_id, verdict in labels.items():
            self.labels[event_id] = verdict
            self.log.set_feedback(event_id, verdict, batch=self._batch)
        self._finish()

    def _label(self, candidate: Candidate, verdict: str) -> None:
        # One answer covers every event the card stands for. That is the promise
        # "this happened 14 times" makes, and the reason the deck clusters at all.
        for event_id in candidate.ids:
            self.labels[event_id] = verdict
            self.log.set_feedback(event_id, verdict, batch=self._batch)

    def _finish(self) -> None:
        # Note what is deliberately *not* done here: the per-event correction in
        # the Activity tab drops the newest size sample, which is right for a
        # correction made seconds later and wrong for one made a week later — by
        # then the newest sample belongs to some other visit. A week-old false
        # alarm earns a blind spot and a test case instead of corrupting the
        # scale model on a guess about which sample it was.
        self.outcome = plan_adjustments(self.labels, self.deck, self.surfaces)
        history = self.log.since(time.time() - 5 * 7 * 86_400.0)
        self.summary_page.show_outcome(self.outcome, self.surfaces,
                                       weekly_accuracy(history, weeks=5))
        self.stack.setCurrentWidget(self.summary_page)

    # -------------------------------------------------------------- interrupts

    def note_alert(self, text: str) -> None:
        """A real alert fired while the review was open. Say so; change nothing."""
        self.banner.set_status(f"Live: {text}", "bad")
        self.banner.setVisible(True)
        QTimer.singleShot(6000, lambda: self.banner.setVisible(False))

    # ---------------------------------------------------------------- keyboard

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        page = self.stack.currentWidget()
        if key == Qt.Key.Key_Escape:
            # Quitting halfway keeps every answer already given. There is nothing
            # to confirm and nothing to lose, so do not ask.
            self.reject()
            return
        if page is self.alert_page:
            if key in (Qt.Key.Key_Y, Qt.Key.Key_Right):
                self.alert_page.yes.click()
            elif key in (Qt.Key.Key_N, Qt.Key.Key_Left):
                self.alert_page.no.click()
            elif key == Qt.Key.Key_U:
                self.alert_page.undo.click()
            elif key == Qt.Key.Key_Space:
                self.alert_page.strip.toggle_play()
            elif Qt.Key.Key_1 <= key <= Qt.Key.Key_3 and self.alert_page.follow_row.isVisible():
                self.alert_page.answered.emit(AlertPage.FOLLOW_UPS[key - Qt.Key.Key_1][1])
            else:
                super().keyPressEvent(event)
            return
        if page is self.grid_page:
            if Qt.Key.Key_1 <= key <= Qt.Key.Key_9:
                self.grid_page.toggle(key - Qt.Key.Key_0)
            elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.grid_page.done_btn.click()
            else:
                super().keyPressEvent(event)
            return
        super().keyPressEvent(event)

    # ---------------------------------------------------------------- teardown

    def done(self, result: int) -> None:  # noqa: D102
        if self.pictures_are_temporary and self.labels:
            # The user agreed to keep pictures for the review, and only for it.
            self.log.forget_pictures_for(list(self.labels))
        self.finished_review.emit()
        super().done(result)


class ReviewInvite(GlassCard):
    """The home-screen invitation. Passive, dismissible, and easy to ignore.

    It sits below the state block and never above it: whether protection is on is
    always the more important thing on this screen. It is a card rather than a
    prompt because a modal here would be the app taxing the user for its own
    uncertainty, which is the failure the whole design is arranged against.
    """

    accepted = Signal()
    declined = Signal()
    silenced = Signal()

    def __init__(self) -> None:
        super().__init__(compact=True)
        self.setVisible(False)
        self.headline = QLabel("")
        self.headline.setObjectName("cardTitle")
        self.headline.setWordWrap(True)
        self.body = QLabel("")
        self.body.setObjectName("muted")
        self.body.setWordWrap(True)

        start = QPushButton("Start review  ·  ~1 min")
        start.setObjectName("primary")
        start.clicked.connect(self.accepted.emit)
        later = QPushButton("Not now")
        later.setObjectName("secondary")
        later.clicked.connect(self.declined.emit)
        never = QPushButton("Don't ask me")
        never.setObjectName("secondary")
        never.clicked.connect(self.silenced.emit)

        self.box.addWidget(self.headline)
        self.box.addWidget(self.body)
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(start)
        row.addStretch(1)
        row.addWidget(later)
        row.addWidget(never)
        self.box.addLayout(row)

    def offer(self, headline: str, body: str) -> None:
        self.headline.setText(headline)
        # The time estimate on the button is a promise the deck size keeps; see
        # review_policy.MAX_CARDS.
        self.body.setText(body)
        self.setVisible(True)
