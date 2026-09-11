"""First-run setup: one flow, from "which camera" to "draw a surface".

Signing in to Eufy is part of this, not a separate dialog that runs first. When it
was separate, setup asked how to connect a camera, took the sign-in, and then
asked how to connect a camera again — and the fix for that was a flag threaded
between two dialogs. Folding the sign-in in makes the double-ask impossible rather
than patched: there is one stack, one step counter, and one route through it.

The steps are conditional, so the counter tells the truth. A Eufy camera needs a
sign-in and a camera choice; an RTSP address needs neither, and skips straight to
confirming the view. Answering a two-factor prompt adds a step only when Eufy
actually asks for one.

No jargon reaches the screen. The only physical thing she has to get right — where
the camera is pointing — is the one thing the flow dwells on.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from enum import IntEnum

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..bridge.client import BridgeClient, BridgeError, DriverPhase, looks_like_camera
from ..bridge.credentials import EufyAccount
from ..bridge.supervisor import BridgeSupervisor
from ..camera.panorama import RoomMap, StitchError, build_room_map
from ..camera.sources.base import CameraSource, SourceError
from . import qtutil as Q
from .home import LiveView

# Eufy accounts are region-locked, and the wrong region is a common silent failure.
COUNTRIES = [
    ("United States", "US"), ("United Kingdom", "GB"), ("Canada", "CA"),
    ("Germany", "DE"), ("France", "FR"), ("Netherlands", "NL"), ("Spain", "ES"),
    ("Italy", "IT"), ("Sweden", "SE"), ("Norway", "NO"), ("Denmark", "DK"),
    ("Australia", "AU"), ("New Zealand", "NZ"), ("Japan", "JP"),
]


class Page(IntEnum):
    KIND = 0
    CREDENTIALS = 1
    CHALLENGE = 2
    PICKER = 3
    CONFIRM = 4
    SCAN = 5
    SOUND = 6
    DONE = 7


@dataclass
class SetupChoice:
    kind: str = "eufy"          # eufy | rtsp | demo
    url: str = "ws://127.0.0.1:3000"
    serial: str = ""


class _SignInWorker(QObject):
    settled = Signal(object)
    failed = Signal(str)

    def __init__(self, supervisor: BridgeSupervisor, client: BridgeClient | None) -> None:
        super().__init__()
        self.supervisor = supervisor
        self.client = client
        self.answer: tuple[str, str] | None = None

    def run(self) -> None:
        try:
            if self.answer is None:
                status = self.supervisor.start(wait=True)
                if not status.listening:
                    self.failed.emit(status.fatal or status.message
                                     or "The camera service did not start.")
                    return
                if self.client is None or not self.client.connected:
                    self.client = BridgeClient(self.supervisor.url)
                    self.client.connect()
                state = self.client.connect_driver()
            elif self.answer[0] == "code":
                state = self.client.submit_verify_code(self.answer[1])
            else:
                state = self.client.submit_captcha(self.answer[1])
            self.settled.emit(state)
        except BridgeError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(f"Something went wrong signing in: {exc}")


class _ScanWorker(QObject):
    progress = Signal(int, int)
    finished = Signal(object, str)

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
        except Exception as exc:
            self.finished.emit(None, f"The room scan failed: {exc}")
            return
        self.finished.emit(room, "")


class OnboardingDialog(QDialog):
    """The whole of setup. ``sign_in_only`` stops once a camera has been chosen."""

    HEADINGS = {
        Page.KIND: ("Connect your camera",
                    "Surface Guard watches video from a camera you already have."),
        Page.CREDENTIALS: ("Sign in to Eufy",
                           "Use the same account you use in the Eufy app."),
        Page.CHALLENGE: ("One more check", "Eufy wants to be sure it is really you."),
        Page.PICKER: ("Choose your camera",
                      "Pick the one pointing at the surfaces you want protected."),
        Page.CONFIRM: ("Is this the right camera?",
                       "Point it at the counter, table or shelf you want protected, "
                       "then leave it there."),
        Page.SCAN: ("Looking around the room",
                    "The camera turns slowly to build one wide picture. You only draw "
                    "your surfaces once on it."),
        Page.SOUND: ("Check you can hear it",
                     "This is what plays when a cat gets on a surface."),
        Page.DONE: ("Now draw your first surface",
                    "Click around the edge of the counter or table you want protected."),
    }

    def __init__(
        self,
        parent: QWidget | None = None,
        allow_demo: bool = True,
        preselect: str = "",
        existing_source: CameraSource | None = None,
        account: EufyAccount | None = None,
        sign_in_only: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Set up Surface Guard")
        self.setMinimumSize(700, 540)

        self.account = account or EufyAccount(username="")
        self.sign_in_only = sign_in_only
        self.source: CameraSource | None = None
        self.supervisor: BridgeSupervisor | None = None
        self.client: BridgeClient | None = None
        self.chosen: dict | None = None
        self.room: RoomMap | None = None
        self.adopted = False
        self._thread: QThread | None = None
        self._worker: QObject | None = None
        self._challenged = False

        self.stack = QStackedWidget()
        for build in (self._page_kind, self._page_credentials, self._page_challenge,
                      self._page_picker, self._page_confirm, self._page_scan,
                      self._page_sound, self._page_done):
            self.stack.addWidget(build(allow_demo))

        self.step = QLabel("")
        self.step.setObjectName("mono")
        self.heading = QLabel("")
        self.heading.setObjectName("h1")
        self.heading.setWordWrap(True)
        self.subtitle = QLabel("")
        self.subtitle.setObjectName("dim")
        self.subtitle.setWordWrap(True)
        self.status = QLabel("")
        self.status.setWordWrap(True)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        self.back_btn = QPushButton("Back")
        self.back_btn.clicked.connect(self._back)
        self.next_btn = QPushButton("Continue")
        self.next_btn.setObjectName("primary")
        self.next_btn.clicked.connect(self._advance)
        nav = QHBoxLayout()
        nav.addWidget(self.cancel_btn)
        nav.addStretch(1)
        nav.addWidget(self.back_btn)
        nav.addWidget(self.next_btn)

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 26, 28, 20)
        root.setSpacing(14)
        root.addWidget(self.step)
        root.addWidget(self.heading)
        root.addWidget(self.subtitle)
        root.addWidget(self.stack, 1)
        root.addWidget(self.status)
        root.addLayout(nav)

        if preselect:
            index = self.kind.findData(preselect)
            if index >= 0:
                self.kind.setCurrentIndex(index)
        self._sync_kind_fields()

        if existing_source is not None:
            self.source = existing_source
            self.adopted = True
            self._go(Page.CONFIRM)
            self._pump_preview()
        elif sign_in_only:
            self._go(Page.CREDENTIALS)
        else:
            self._go(Page.KIND)

    # -------------------------------------------------------------- the pages

    def _page_kind(self, allow_demo: bool) -> QWidget:
        page = QWidget()
        box = QVBoxLayout(page)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(10)
        self.kind = QComboBox()
        self.kind.addItem("A Eufy camera", "eufy")
        self.kind.addItem("Another camera with a video address", "rtsp")
        if allow_demo:
            self.kind.addItem("Try it without a camera", "demo")
        self.kind.currentIndexChanged.connect(self._sync_kind_fields)
        self.url_label = QLabel("Video address")
        self.url_label.setObjectName("dim")
        self.url = QLineEdit("ws://127.0.0.1:3000")
        box.addWidget(self.kind)
        box.addWidget(self.url_label)
        box.addWidget(self.url)
        box.addStretch(1)
        return page

    def _page_credentials(self, _allow_demo: bool) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(11)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.email = QLineEdit(self.account.username)
        self.email.setPlaceholderText("the email you use for the Eufy app")
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        self.password.setPlaceholderText("your Eufy password")
        self.password.returnPressed.connect(self._advance)
        self.country = QComboBox()
        for label, code in COUNTRIES:
            self.country.addItem(label, code)
        self.country.setCurrentIndex(max(0, self.country.findData(self.account.country)))
        form.addRow("Email", self.email)
        form.addRow("Password", self.password)
        form.addRow("Region", self.country)
        note = QLabel("Your password is stored in the Mac Keychain and is never saved "
                      "in the app's own files.")
        note.setObjectName("dim")
        note.setWordWrap(True)
        form.addRow(note)
        return page

    def _page_challenge(self, _allow_demo: bool) -> QWidget:
        page = QWidget()
        box = QVBoxLayout(page)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(10)
        self.challenge_text = QLabel("")
        self.challenge_text.setWordWrap(True)
        self.captcha_image = QLabel()
        self.captcha_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.captcha_image.setVisible(False)
        self.challenge_input = QLineEdit()
        self.challenge_input.returnPressed.connect(self._advance)
        box.addWidget(self.challenge_text)
        box.addWidget(self.captcha_image)
        box.addWidget(self.challenge_input)
        box.addStretch(1)
        return page

    def _page_picker(self, _allow_demo: bool) -> QWidget:
        page = QWidget()
        box = QVBoxLayout(page)
        box.setContentsMargins(0, 0, 0, 0)
        self.devices = QListWidget()
        self.devices.itemDoubleClicked.connect(lambda _: self._advance())
        box.addWidget(self.devices, 1)
        return page

    def _page_confirm(self, _allow_demo: bool) -> QWidget:
        page = QWidget()
        box = QVBoxLayout(page)
        box.setContentsMargins(0, 0, 0, 0)
        self.preview = LiveView()
        box.addWidget(self.preview, 1)
        return page

    def _page_scan(self, _allow_demo: bool) -> QWidget:
        page = QWidget()
        box = QVBoxLayout(page)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(10)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.scan_status = QLabel("Ready when you are.")
        self.scan_status.setObjectName("mono")
        self.scan_status.setWordWrap(True)
        box.addWidget(self.progress)
        box.addWidget(self.scan_status)
        box.addStretch(1)
        return page

    def _page_sound(self, _allow_demo: bool) -> QWidget:
        page = QWidget()
        box = QVBoxLayout(page)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(10)
        self.test_btn = QPushButton("Play the sound")
        self.sound_note = QLabel("")
        self.sound_note.setObjectName("dim")
        self.sound_note.setWordWrap(True)
        box.addWidget(self.test_btn)
        box.addWidget(self.sound_note)
        box.addStretch(1)
        return page

    def _page_done(self, _allow_demo: bool) -> QWidget:
        page = QWidget()
        box = QVBoxLayout(page)
        box.setContentsMargins(0, 0, 0, 0)
        box.addStretch(1)
        return page

    # ------------------------------------------------------------- the route

    def route(self) -> list[Page]:
        """Which steps this particular setup actually needs.

        Conditional, so the counter never promises a step that will not happen or
        hides one that will.
        """
        if self.adopted:
            steps = [Page.CONFIRM, Page.SCAN, Page.SOUND, Page.DONE]
        elif self.sign_in_only:
            steps = [Page.CREDENTIALS, Page.PICKER]
        elif self.choice().kind == "eufy":
            steps = [Page.KIND, Page.CREDENTIALS, Page.PICKER, Page.CONFIRM,
                     Page.SCAN, Page.SOUND, Page.DONE]
        else:
            steps = [Page.KIND, Page.CONFIRM, Page.SCAN, Page.SOUND, Page.DONE]
        if self._challenged:
            insert_at = steps.index(Page.PICKER) if Page.PICKER in steps else len(steps)
            steps.insert(insert_at, Page.CHALLENGE)
        return steps

    def _go(self, page: Page) -> None:
        self.stack.setCurrentIndex(int(page))
        self._sync_nav()

    @property
    def page(self) -> Page:
        return Page(self.stack.currentIndex())

    def _sync_nav(self) -> None:
        steps = self.route()
        page = self.page
        if page in steps:
            self.step.setText(f"STEP {steps.index(page) + 1} OF {len(steps)}")
        else:
            self.step.setText("")
        heading, subtitle = self.HEADINGS[page]
        self.heading.setText(heading)
        self.subtitle.setText(subtitle)

        labels = {
            Page.KIND: "Continue", Page.CREDENTIALS: "Sign in",
            Page.CHALLENGE: "Continue", Page.PICKER: "Use this camera",
            Page.CONFIRM: "Yes, that's right", Page.SCAN: "Start looking around",
            Page.SOUND: "Continue", Page.DONE: "Draw a surface",
        }
        self.next_btn.setText(labels[page])
        first = steps[0] if steps else page
        self.back_btn.setEnabled(page != first and page != Page.SCAN)

    def _back(self) -> None:
        steps = self.route()
        if self.page in steps:
            index = steps.index(self.page)
            if index > 0:
                self._go(steps[index - 1])

    def _next_page(self) -> Page | None:
        steps = self.route()
        if self.page not in steps:
            return None
        index = steps.index(self.page)
        return steps[index + 1] if index + 1 < len(steps) else None

    def choice(self) -> SetupChoice:
        return SetupChoice(kind=str(self.kind.currentData()),
                           url=self.url.text().strip(), serial="")

    def _sync_kind_fields(self) -> None:
        kind = self.kind.currentData()
        needs_url = kind == "rtsp"
        self.url_label.setVisible(needs_url)
        self.url.setVisible(needs_url)
        if needs_url and self.url.text().startswith("ws://"):
            self.url.setText("rtsp://")
        if self.page == Page.KIND:
            self._sync_nav()

    # ------------------------------------------------------------ advancing

    def _advance(self) -> None:
        page = self.page
        if page is Page.KIND:
            self._leave_kind()
        elif page is Page.CREDENTIALS:
            self._start_sign_in()
        elif page is Page.CHALLENGE:
            self._submit_challenge()
        elif page is Page.PICKER:
            self._choose_device()
        elif page is Page.CONFIRM:
            self._go(Page.SCAN)
        elif page is Page.SCAN:
            self._start_scan()
        elif page is Page.SOUND:
            self._go(Page.DONE)
        else:
            self.accept()

    def _leave_kind(self) -> None:
        if self.choice().kind == "eufy":
            self._go(Page.CREDENTIALS)
            self.email.setFocus()
            return
        self._say("Connecting…")
        self.next_btn.setEnabled(False)
        try:
            source = _build_source(self.choice())
            source.start()
        except (SourceError, Exception) as exc:
            self.next_btn.setEnabled(True)
            self._say(str(exc), bad=True)
            return
        self.next_btn.setEnabled(True)
        if source.read(timeout=12.0) is None:
            source.stop()
            self._say("Connected, but no video arrived. Check the camera is switched on "
                      "and not being viewed in another app.", bad=True)
            return
        self.source = source
        self._say("")
        self._go(Page.CONFIRM)
        self._pump_preview()

    def _start_sign_in(self) -> None:
        email = self.email.text().strip()
        if not email or not self.password.text():
            self._say("Enter the email and password you use for the Eufy app.", bad=True)
            return
        self.account.username = email
        self.account.country = str(self.country.currentData())
        try:
            self.account.save_password(self.password.text())
        except Exception as exc:
            self._say(str(exc), bad=True)
            return
        self.password.clear()          # the Keychain has it now
        self.supervisor = BridgeSupervisor(self.account)
        self._say("Starting the camera service and signing in…")
        self._run_signin(None)

    def _submit_challenge(self) -> None:
        value = self.challenge_input.text().strip()
        if not value:
            self._say("Enter the code first.", bad=True)
            return
        kind = "captcha" if not self.captcha_image.isHidden() else "code"
        self._say("Checking…")
        self._run_signin((kind, value))

    def _run_signin(self, answer: tuple[str, str] | None) -> None:
        if self.supervisor is None:
            return
        self.next_btn.setEnabled(False)
        self._thread = QThread(self)
        worker = _SignInWorker(self.supervisor, self.client)
        worker.answer = answer
        self._worker = worker
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        worker.settled.connect(self._on_signin_settled)
        worker.failed.connect(self._on_signin_failed)
        self._thread.start()

    def _end_thread(self) -> None:
        if isinstance(self._worker, _SignInWorker):
            self.client = self._worker.client
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(4000)
            self._thread = None
        self._worker = None
        self.next_btn.setEnabled(True)

    def _on_signin_failed(self, message: str) -> None:
        self._end_thread()
        self._say(message, bad=True)
        self._go(Page.CREDENTIALS)

    def _on_signin_settled(self, state) -> None:
        self._end_thread()
        if state.phase is DriverPhase.CONNECTED:
            self._say("Signed in. Looking for cameras…")
            self._load_devices()
            return
        if state.phase is DriverPhase.NEEDS_CODE:
            self._challenged = True
            self.captcha_image.setVisible(False)
            self.challenge_text.setText(
                f"Eufy emailed a verification code to {self.account.username}."
            )
            self.challenge_input.clear()
            self.challenge_input.setPlaceholderText("6-digit code")
            self._go(Page.CHALLENGE)
            self.challenge_input.setFocus()
            self._say("")
            return
        if state.phase is DriverPhase.NEEDS_CAPTCHA:
            self._challenged = True
            self.challenge_text.setText("Type the characters shown in the picture.")
            self._show_captcha(state.captcha_image)
            self.challenge_input.clear()
            self.challenge_input.setPlaceholderText("characters from the picture")
            self._go(Page.CHALLENGE)
            self.challenge_input.setFocus()
            self._say("")
            return
        self._say(state.message or "Eufy refused the sign-in.", bad=True)
        self._go(Page.CREDENTIALS)

    def _show_captcha(self, data_uri: str) -> None:
        pix = QPixmap()
        if data_uri.startswith("data:") and "," in data_uri:
            try:
                pix.loadFromData(base64.b64decode(data_uri.split(",", 1)[1]))
            except Exception:
                pix = QPixmap()
        if pix.isNull():
            self.captcha_image.setText("(could not show the picture — try again)")
        else:
            self.captcha_image.setPixmap(pix)
        self.captcha_image.setVisible(True)

    def _load_devices(self) -> None:
        if self.client is None:
            return
        try:
            found = self.client.devices()
        except BridgeError as exc:
            self._say(str(exc), bad=True)
            return
        cameras = [d for d in found if looks_like_camera(d)]
        self.devices.clear()
        for device in cameras:
            item = QListWidgetItem(
                f"{device.get('name') or device.get('serialNumber', 'camera')}\n"
                f"{device.get('model') or 'camera'}"
            )
            item.setToolTip(str(device.get("serialNumber", "")))
            item.setData(Qt.ItemDataRole.UserRole, device)
            self.devices.addItem(item)
        if not cameras:
            self._say("Signed in, but this account has no cameras on it. Check you used "
                      "the account the camera is registered to.", bad=True)
            return
        self.devices.setCurrentRow(0)
        self._go(Page.PICKER)
        self._say(f"Found {len(cameras)} camera{'s' if len(cameras) != 1 else ''}.")

    def _choose_device(self) -> None:
        item = self.devices.currentItem()
        if item is None:
            self._say("Choose a camera first.", bad=True)
            return
        self.chosen = item.data(Qt.ItemDataRole.UserRole)
        if self.sign_in_only:
            self.accept()
            return
        from ..camera.sources.eufy_bridge import EufyBridgeCamera

        self.source = EufyBridgeCamera(
            client=self.client, serial=str(self.chosen.get("serialNumber", "")),
            model=str(self.chosen.get("model", "")), name=str(self.chosen.get("name", "")),
        )
        try:
            self.source.start()
        except SourceError as exc:
            self.source = None
            self._say(str(exc), bad=True)
            return
        self._say("")
        self._go(Page.CONFIRM)
        self._pump_preview()

    # ------------------------------------------------------------------ scan

    def _pump_preview(self) -> None:
        from ..engine import FrameResult

        if self.source is None:
            return
        frame = self.source.read(timeout=6.0)
        if frame is not None:
            self.preview.update_result(FrameResult(frame=frame, pose=None), [])

    def _start_scan(self) -> None:
        if self.source is None:
            return
        self.next_btn.setEnabled(False)
        self.back_btn.setEnabled(False)
        self.progress.setRange(0, 0)
        self.scan_status.setText("Turning the camera…")
        self._thread = QThread(self)
        worker = _ScanWorker(self.source)
        self._worker = worker
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        worker.progress.connect(self._on_scan_progress)
        worker.finished.connect(self._on_scan_finished)
        self._thread.start()

    def _on_scan_progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self.scan_status.setText(f"Looked at {done} of {total} positions")

    def _on_scan_finished(self, room: object, error: str) -> None:
        self._end_thread()
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
        self._go(Page.SOUND)

    def bind_player(self, play) -> None:
        def run() -> None:
            outcome = play()
            self.sound_note.setText(f"Played through {outcome.backend}." if outcome.ok
                                    else f"Could not play: {outcome.error}")
            self.sound_note.setStyleSheet(
                f"color: {(Q.GOOD if outcome.ok else Q.BAD).name()};")
        self.test_btn.clicked.connect(run)

    # -------------------------------------------------------------- teardown

    def _say(self, text: str, bad: bool = False) -> None:
        self.status.setText(text)
        self.status.setStyleSheet(f"color: {(Q.BAD if bad else Q.INK_DIM).name()};")

    def reject(self) -> None:
        self._end_thread()
        if self.adopted:
            self.source = None          # the camera belongs to the app, not here
        super().reject()


def _build_source(choice: SetupChoice) -> CameraSource:
    if choice.kind == "demo":
        from ..camera.sources.synthetic import SyntheticCamera
        return SyntheticCamera()
    if choice.kind == "rtsp":
        from ..camera.sources.rtsp import RtspCamera
        return RtspCamera(choice.url)
    from ..camera.sources.eufy_bridge import EufyBridgeCamera
    return EufyBridgeCamera(url=choice.url, serial=choice.serial or None)
