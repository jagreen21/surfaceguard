"""Guided setup: seven steps, no jargon.

The wizard never mentions WebSockets, serial numbers, homographies or Docker. It
does mention the one physical thing that matters — where the camera is pointing —
because that is the only part the user has to get right.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..camera.panorama import RoomMap, StitchError, build_room_map
from ..camera.sources.base import CameraSource, SourceError
from .home import LiveView
from . import qtutil as Q


@dataclass
class SetupChoice:
    kind: str = "eufy"      # eufy | rtsp | demo
    url: str = "ws://127.0.0.1:3000"
    serial: str = ""


class _ScanWorker(QObject):
    progress = Signal(int, int)
    finished = Signal(object, str)   # RoomMap | None, error message

    def __init__(self, source: CameraSource) -> None:
        super().__init__()
        self.source = source

    def run(self) -> None:
        try:
            room = build_room_map(
                self.source,
                settle_s=1.2 if self.source.capabilities.has_ptz else 0.0,
                progress=lambda i, n: self.progress.emit(i, n),
            )
        except (StitchError, SourceError) as exc:
            self.finished.emit(None, str(exc))
            return
        except Exception as exc:                       # unexpected, still the user's problem
            self.finished.emit(None, f"The room scan failed: {exc}")
            return
        self.finished.emit(room, "")


class _Page(QWidget):
    def __init__(self, title: str, body: str) -> None:
        super().__init__()
        self.heading = QLabel(title)
        self.heading.setObjectName("h1")
        self.heading.setWordWrap(True)
        self.body = QLabel(body)
        self.body.setObjectName("dim")
        self.body.setWordWrap(True)
        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.layout_.setSpacing(10)
        self.layout_.addWidget(self.heading)
        self.layout_.addWidget(self.body)


class OnboardingDialog(QDialog):
    """Runs setup and hands back a connected source and a stitched room map."""

    def __init__(
        self,
        parent: QWidget | None = None,
        allow_demo: bool = True,
        preselect: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Set up Surface Guard")
        self.setMinimumSize(720, 520)
        self.source: CameraSource | None = None
        self.room: RoomMap | None = None
        self._thread: QThread | None = None
        self._worker: _ScanWorker | None = None

        # --- page 1: connect ------------------------------------------------
        self.connect_page = _Page(
            "Let’s connect your camera",
            "Surface Guard watches the video from a camera you already have.",
        )
        self.kind = QComboBox()
        self.kind.addItem("Eufy camera (recommended)", "eufy")
        self.kind.addItem("Another camera with a video address", "rtsp")
        if allow_demo:
            self.kind.addItem("Try it without a camera", "demo")
        self.url = QLineEdit("ws://127.0.0.1:3000")
        self.serial = QLineEdit()
        self.serial.setPlaceholderText("Leave empty to use the first camera found")
        self.url_label = QLabel("Camera bridge address")
        self.url_label.setObjectName("dim")
        self.serial_label = QLabel("Camera ID (optional)")
        self.serial_label.setObjectName("dim")
        self.connect_error = QLabel("")
        self.connect_error.setWordWrap(True)
        self.connect_error.setStyleSheet(f"color: {Q.BAD.name()};")
        for w in (self.kind, self.url_label, self.url, self.serial_label, self.serial,
                  self.connect_error):
            self.connect_page.layout_.addWidget(w)
        self.connect_page.layout_.addStretch(1)
        self.kind.currentIndexChanged.connect(self._sync_connect_fields)

        # --- page 2: confirm the view ---------------------------------------
        self.confirm_page = _Page(
            "Is this the right camera?",
            "Point it at the counter, table or shelf you want protected, then leave it "
            "there. Surface Guard will remember this view.",
        )
        self.preview = LiveView()
        self.confirm_page.layout_.addWidget(self.preview, 1)

        # --- page 3: scan ---------------------------------------------------
        self.scan_page = _Page(
            "Looking around the room",
            "The camera turns slowly to build one wide picture. You only have to draw "
            "your surfaces once on it.",
        )
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.scan_status = QLabel("Ready when you are.")
        self.scan_status.setObjectName("mono")
        self.scan_status.setWordWrap(True)
        self.scan_page.layout_.addWidget(self.progress)
        self.scan_page.layout_.addWidget(self.scan_status)
        self.scan_page.layout_.addStretch(1)

        # --- page 4: sound --------------------------------------------------
        self.sound_page = _Page(
            "Check you can hear it",
            "This is the sound that plays when a cat gets on a surface. If you cannot "
            "hear it, turn up the volume or move this computer closer.",
        )
        self.test_btn = QPushButton("Play the sound")
        self.test_btn.setObjectName("primary")
        self.sound_note = QLabel("")
        self.sound_note.setObjectName("dim")
        self.sound_note.setWordWrap(True)
        self.sound_page.layout_.addWidget(self.test_btn)
        self.sound_page.layout_.addWidget(self.sound_note)
        self.sound_page.layout_.addStretch(1)

        # --- page 5: done ---------------------------------------------------
        self.done_page = _Page(
            "Now draw your first surface",
            "Click around the edge of the counter or table you want protected. You can "
            "add more later, and change them any time.",
        )
        self.done_page.layout_.addStretch(1)

        self.stack = QStackedWidget()
        for page in (self.connect_page, self.confirm_page, self.scan_page,
                     self.sound_page, self.done_page):
            self.stack.addWidget(page)

        self.back_btn = QPushButton("Back")
        self.back_btn.clicked.connect(self._back)
        self.next_btn = QPushButton("Continue")
        self.next_btn.setObjectName("primary")
        self.next_btn.clicked.connect(self._next)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)

        nav = QHBoxLayout()
        nav.addWidget(self.cancel_btn)
        nav.addStretch(1)
        nav.addWidget(self.back_btn)
        nav.addWidget(self.next_btn)

        root = QVBoxLayout(self)
        root.setContentsMargins(26, 24, 26, 20)
        root.setSpacing(16)
        root.addWidget(self.stack, 1)
        root.addLayout(nav)

        if preselect:
            index = self.kind.findData(preselect)
            if index >= 0:
                self.kind.setCurrentIndex(index)
        self._sync_connect_fields()
        self._sync_nav()

    # ---------------------------------------------------------------- helpers

    def choice(self) -> SetupChoice:
        return SetupChoice(
            kind=str(self.kind.currentData()),
            url=self.url.text().strip(),
            serial=self.serial.text().strip(),
        )

    def bind_player(self, play) -> None:
        """``play`` is called with no arguments and returns a PlaybackResult."""
        def run() -> None:
            outcome = play()
            self.sound_note.setText(
                f"Played through {outcome.backend}." if outcome.ok
                else f"Could not play: {outcome.error}"
            )
            self.sound_note.setStyleSheet(
                f"color: {(Q.GOOD if outcome.ok else Q.BAD).name()};"
            )
        self.test_btn.clicked.connect(run)

    def _sync_connect_fields(self) -> None:
        kind = self.kind.currentData()
        eufy, rtsp = kind == "eufy", kind == "rtsp"
        self.url_label.setVisible(eufy or rtsp)
        self.url.setVisible(eufy or rtsp)
        self.url_label.setText("Camera bridge address" if eufy else "Video address")
        if rtsp and self.url.text().startswith("ws://"):
            self.url.setText("rtsp://")
        if eufy and self.url.text().startswith("rtsp://"):
            self.url.setText("ws://127.0.0.1:3000")
        self.serial_label.setVisible(eufy)
        self.serial.setVisible(eufy)

    def _sync_nav(self) -> None:
        index = self.stack.currentIndex()
        self.back_btn.setEnabled(index > 0 and index != 2)
        labels = {0: "Connect", 1: "Yes, that’s right", 2: "Start looking around",
                  3: "Continue", 4: "Draw a surface"}
        self.next_btn.setText(labels.get(index, "Continue"))

    def _back(self) -> None:
        self.stack.setCurrentIndex(max(0, self.stack.currentIndex() - 1))
        self._sync_nav()

    def _next(self) -> None:
        index = self.stack.currentIndex()
        if index == 0:
            self._do_connect()
        elif index == 1:
            self.stack.setCurrentIndex(2)
            self._sync_nav()
        elif index == 2:
            self._do_scan()
        elif index == 3:
            self.stack.setCurrentIndex(4)
            self._sync_nav()
        else:
            self.accept()

    # ---------------------------------------------------------------- actions

    def _do_connect(self) -> None:
        self.connect_error.setText("")
        self.next_btn.setEnabled(False)
        try:
            source = _build_source(self.choice())
            source.start()
        except SourceError as exc:
            self.connect_error.setText(str(exc))
            self.next_btn.setEnabled(True)
            return
        except Exception as exc:
            self.connect_error.setText(f"Could not connect: {exc}")
            self.next_btn.setEnabled(True)
            return

        frame = source.read(timeout=12.0)
        self.next_btn.setEnabled(True)
        if frame is None:
            source.stop()
            self.connect_error.setText(
                "Connected, but no video arrived. Check the camera is switched on and "
                "not being viewed in another app."
            )
            return
        self.source = source
        self.preview.set_message("")
        self.stack.setCurrentIndex(1)
        self._sync_nav()
        self._pump_preview()

    def _pump_preview(self) -> None:
        """One-shot still: enough to confirm the camera, without a video loop here."""
        from ..engine import FrameResult
        if self.source is None:
            return
        frame = self.source.read(timeout=4.0)
        if frame is not None:
            self.preview.update_result(FrameResult(frame=frame, pose=None), [])

    def _do_scan(self) -> None:
        if self.source is None:
            return
        self.next_btn.setEnabled(False)
        self.back_btn.setEnabled(False)
        self.progress.setRange(0, 0)
        self.scan_status.setText("Turning the camera…")

        self._thread = QThread(self)
        self._worker = _ScanWorker(self.source)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_scan_progress)
        self._worker.finished.connect(self._on_scan_finished)
        self._thread.start()

    def _on_scan_progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self.scan_status.setText(f"Looked at {done} of {total} positions")

    def _on_scan_finished(self, room: object, error: str) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
            self._thread = None
        self.next_btn.setEnabled(True)
        self.back_btn.setEnabled(True)
        if room is None:
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.scan_status.setText(error)
            self.scan_status.setStyleSheet(f"color: {Q.BAD.name()};")
            self.next_btn.setText("Try again")
            return
        self.room = room  # type: ignore[assignment]
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.scan_status.setStyleSheet(f"color: {Q.GOOD.name()};")
        w, h = room.size  # type: ignore[attr-defined]
        self.scan_status.setText(f"Done — the room picture is {w} by {h}.")
        self.stack.setCurrentIndex(3)
        self._sync_nav()


def _build_source(choice: SetupChoice) -> CameraSource:
    if choice.kind == "demo":
        from ..camera.sources.synthetic import SyntheticCamera
        return SyntheticCamera()
    if choice.kind == "rtsp":
        from ..camera.sources.rtsp import RtspCamera
        return RtspCamera(choice.url)
    from ..camera.sources.eufy_bridge import EufyBridgeCamera
    return EufyBridgeCamera(url=choice.url, serial=choice.serial or None)
