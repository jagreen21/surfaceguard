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
import platform
import time
from dataclasses import dataclass
from enum import IntEnum

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
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
from ..logging_setup import redact_support_text, tail
from ..update import build_info
from . import qtutil as Q
from .home import LiveView

# Eufy accounts are region-locked, and the wrong region is a common silent failure.
# Friendly names for the models this is likely to meet, so the picker does not
# just show three part numbers.
# How long to wait for a first frame before suggesting the camera is busy.
PREVIEW_PATIENCE_S = 12.0

MODEL_NAMES = {
    "T8417": "Indoor Cam E30",
    "T8416": "Indoor Cam E220",
    "T8410": "Indoor Cam Pan & Tilt",
    "T8414": "Indoor Cam C210",
    "T8441": "SoloCam",
    "T8600": "eufyCam",
}

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
    devices_found = Signal(object)

    def __init__(self, supervisor: BridgeSupervisor, client: BridgeClient | None) -> None:
        super().__init__()
        self.supervisor = supervisor
        self.client = client
        self.answer: tuple[str, str] | None = None

    def run(self) -> None:
        try:
            if self.answer == ("devices", ""):
                # Off the GUI thread: finding devices takes seconds, and doing it
                # in a signal handler froze the window on "Looking for cameras…".
                self.devices_found.emit(self.client.wait_for_devices())
                return
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
                      "Pick the one pointing at the surfaces you want protected — the "
                      "names are the ones you gave them in the Eufy app. You will see "
                      "its picture next, and can come back if it is the wrong one."),
        Page.CONFIRM: ("Is this the right camera?",
                       "Use the Eufy app to point it at the counter, table or shelf "
                       "you want protected, then leave it there."),
        Page.SCAN: ("Looking around the room",
                    "The camera looks a little left and right to build one wide "
                    "picture, then returns to its starting view."),
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
        supervisor: BridgeSupervisor | None = None,
        client: BridgeClient | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Set up Surface Guard")
        self.setMinimumSize(700, 540)

        self.account = account or EufyAccount(username="")
        self.sign_in_only = sign_in_only
        self.source: CameraSource | None = None
        # A bridge and sign-in handed in from a previous run of this dialog. Closing
        # setup used to throw the signed-in session away, so coming back meant
        # typing the password again even though the bridge was still connected.
        self.supervisor = supervisor
        self.client = client
        self.resumed = False
        self.chosen: dict | None = None
        self.room: RoomMap | None = None
        self.adopted = False
        self._thread: QThread | None = None
        self._worker: QObject | None = None
        self._challenged = False
        self._preview_timer: QTimer | None = None
        self._preview_frames = 0
        self._preview_started = 0.0

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
        elif self._already_signed_in():
            # Straight back to choosing a camera; the sign-in still stands.
            self.resumed = True
            self._say("Still signed in to Eufy.")
            self._go(Page.PICKER)
            self._run_signin(("devices", ""))
        elif sign_in_only:
            self._go(Page.CREDENTIALS)
        else:
            self._go(Page.KIND)

    def _already_signed_in(self) -> bool:
        """Is there a live bridge with a connected driver from a previous attempt?"""
        return (
            self.client is not None
            and self.client.connected
            and self.client.driver.phase is DriverPhase.CONNECTED
            and self.supervisor is not None
            and self.supervisor.status.healthy
        )

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
        support = QHBoxLayout()
        self.copy_support_btn = QPushButton("Copy support report")
        self.copy_support_btn.clicked.connect(self._copy_support_report)
        self.copy_support_btn.setVisible(False)
        self.copy_support_note = QLabel("")
        self.copy_support_note.setObjectName("dim")
        support.addWidget(self.copy_support_btn)
        support.addWidget(self.copy_support_note)
        support.addStretch(1)
        box.addLayout(support)
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
        if page is not Page.CONFIRM:
            self._stop_preview()
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
        if page is Page.CONFIRM and self.chosen:
            # With three cameras on the account, "is this the right one?" is only
            # answerable if she can see which one she is looking at.
            name = str(self.chosen.get("name") or "").strip()
            if name:
                heading = f"Is this {name}?"
                subtitle = ("If this is the wrong camera, go Back and pick another. "
                            + subtitle)
        self.heading.setText(heading)
        self.subtitle.setText(subtitle)

        labels = {
            Page.KIND: "Continue", Page.CREDENTIALS: "Sign in",
            Page.CHALLENGE: "Continue", Page.PICKER: "Use this camera",
            Page.CONFIRM: "Yes, that's right", Page.SCAN: "Start looking around",
            Page.SOUND: "Continue", Page.DONE: "Draw a surface",
        }
        self.next_btn.setText(labels[page])
        if page is Page.CONFIRM:
            # PTZ control and video use separate paths. A working camera motor is
            # not proof that a picture is available for the room scan.
            self.next_btn.setEnabled(self._preview_frames > 0)
        first = steps[0] if steps else page
        self.back_btn.setEnabled(page != first and page != Page.SCAN)

    def _back(self) -> None:
        steps = self.route()
        if self.page not in steps:
            return
        index = steps.index(self.page)
        if index <= 0:
            return
        previous = steps[index - 1]
        if self.page is Page.CONFIRM and previous is Page.PICKER and not self.adopted:
            # Release the camera on the way back, so only ever one stream is open.
            if self.source is not None:
                try:
                    self.source.stop()
                except Exception:
                    pass
                self.source = None
            self.chosen = None
        self._go(previous)

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
            if self._preview_frames == 0:
                self._say(
                    "Wait until the camera picture appears before continuing. If it "
                    "doesn't, go Back and try the camera again.",
                    bad=True,
                )
                return
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
        # Reuse the running bridge rather than starting another. Every retry used
        # to leak one: her log showed five of them, on ports 3050 through 3054,
        # each still listening because nothing ever stopped the previous.
        if self.supervisor is not None:
            self.supervisor.stop()
            self.supervisor = None
        if self.client is not None:
            self.client.close()
            self.client = None
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
        worker.devices_found.connect(self._on_devices_found)
        # The worker emits exactly one terminal signal. Stop its event loop at
        # the source as well as in the UI callback, so closing setup during that
        # queued handoff cannot destroy a still-running QThread.
        worker.settled.connect(self._thread.quit, Qt.ConnectionType.DirectConnection)
        worker.failed.connect(self._thread.quit, Qt.ConnectionType.DirectConnection)
        worker.devices_found.connect(self._thread.quit, Qt.ConnectionType.DirectConnection)
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
            # Remember the account the moment it works, not when setup finishes.
            # The password is already in the Keychain; without the email and region
            # beside it, an update or a quit before choosing a camera meant typing
            # the whole thing again.
            self._remember_account()
            self._say("Signed in. Looking for your cameras…")
            self._run_signin(("devices", ""))
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

    def _remember_account(self) -> None:
        from ..storage.preferences import Preferences

        try:
            prefs = Preferences.load()
            camera = dict(prefs.camera or {})
            camera.update({"kind": "eufy", "username": self.account.username,
                           "country": self.account.country})
            prefs.camera = camera
            prefs.save()
        except Exception:
            # Not being able to remember it is not a reason to stop signing in.
            pass

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

    def _on_devices_found(self, found: list) -> None:
        self._end_thread()
        self._show_devices(found)

    def _load_devices(self) -> None:
        """Synchronous path, used by tests and by Settings."""
        if self.client is None:
            return
        try:
            self._show_devices(self.client.devices())
        except BridgeError as exc:
            self._say(str(exc), bad=True)

    def _show_devices(self, found: list) -> None:
        cameras = [d for d in found if looks_like_camera(d)]
        self.devices.clear()
        for device in cameras:
            serial = str(device.get("serialNumber", ""))
            name = str(device.get("name") or "").strip()
            model = MODEL_NAMES.get(str(device.get("model", "")).upper(),
                                    str(device.get("model") or "Eufy camera"))
            # The name she gave it in the Eufy app is what tells three cameras
            # apart; the model and serial are only there to break a tie.
            label = name or "(unnamed camera)"
            detail = f"{model}  ·  {serial[-6:] if serial else '?'}"
            item = QListWidgetItem(f"{label}\n{detail}")
            item.setToolTip(serial)
            item.setData(Qt.ItemDataRole.UserRole, device)
            self.devices.addItem(item)
        if not cameras:
            self._say(
                "Signed in, but no cameras turned up on this account. Check the "
                "camera is switched on and set up in the Eufy app, and that this is "
                "the account it is registered to.", bad=True,
            )
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

        # Coming back to try a different camera has to stop the first one. Eufy
        # serves one livestream at a time, so starting a second without stopping
        # the first leaves her looking at a picture that never arrives.
        if self.source is not None and not self.adopted:
            try:
                self.source.stop()
            except Exception:
                pass
            self.source = None

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
        self._say("Starting the picture…")
        self._go(Page.CONFIRM)
        self._pump_preview()

    # ------------------------------------------------------------------ scan

    def _pump_preview(self) -> None:
        """Start pulling frames continuously while the preview is on screen.

        This used to read a single frame, once. A P2P stream takes two to five
        seconds to produce its first picture, so the one read usually timed out and
        the page stayed black — with nothing to say why, on the one screen whose
        entire job is "does this look right?".
        """
        if self._preview_timer is None:
            self._preview_timer = QTimer(self)
            self._preview_timer.setInterval(120)
            self._preview_timer.timeout.connect(self._preview_tick)
        self._preview_frames = 0
        self._preview_started = time.monotonic()
        self.copy_support_btn.setVisible(False)
        self.copy_support_note.clear()
        self.next_btn.setEnabled(False)
        self._preview_timer.start()
        self._preview_tick()

    def _preview_tick(self) -> None:
        from ..engine import FrameResult

        if self.source is None or self.page is not Page.CONFIRM:
            self._stop_preview()
            return
        frame = self.source.read(timeout=0.05)
        if frame is not None:
            self._preview_frames += 1
            self.preview.update_result(FrameResult(frame=frame, pose=None), [])
            if self._preview_frames == 1:
                self._say("")
                self.copy_support_btn.setVisible(False)
                self.next_btn.setEnabled(True)
            return
        waited = time.monotonic() - self._preview_started
        if self._preview_frames == 0:
            stream_error = str(getattr(self.source, "last_error", "") or "").strip()
            if stream_error:
                self._say(stream_error + " Go Back and try this camera again.", bad=True)
                self.next_btn.setEnabled(False)
                self.copy_support_btn.setVisible(True)
                return
            snapshot = getattr(self.source, "video_diagnostics", None)
            diagnostics = snapshot() if callable(snapshot) else {}
            if (
                waited >= 1.0
                and int(diagnostics.get("chunks", 0)) > 0
                and int(diagnostics.get("decode_errors", 0)) > 0
            ):
                self._say(
                    "The camera is sending video, but Surface Guard cannot decode "
                    "its picture yet. Copy the support report and send it to us.",
                    bad=True,
                )
                self.next_btn.setEnabled(False)
                self.copy_support_btn.setVisible(True)
                return
            if waited < PREVIEW_PATIENCE_S:
                self._say(f"Waiting for the picture… ({waited:.0f}s)")
            else:
                self._say(
                    "No video arrived, so Surface Guard will not move the camera. "
                    "Close the Eufy app on any phone, then go Back and try this "
                    "camera again.",
                    bad=True,
                )
                self.copy_support_btn.setVisible(True)

    def _copy_support_report(self) -> None:
        diagnostics = {}
        if self.source is not None:
            snapshot = getattr(self.source, "video_diagnostics", None)
            if callable(snapshot):
                diagnostics = snapshot()
        lines = [
            f"Surface Guard {build_info.VERSION} ({build_info.UPDATE_CHANNEL})",
            f"macOS {platform.mac_ver()[0]} on {platform.machine()}",
            f"camera model: {str((self.chosen or {}).get('model') or 'unknown')}",
            "video: " + " ".join(f"{key}={value}" for key, value in diagnostics.items()),
        ]
        if self.supervisor is not None:
            status = self.supervisor.status
            lines.append(
                f"bridge: running={status.running} listening={status.listening} "
                f"lan_unreachable={status.lan_unreachable} restarts={status.restarts}"
            )
            recent = self.supervisor.logs(30)
            if recent:
                lines += ["", "camera service:", *recent]
        lines += ["", "app log:", tail(100)]
        QApplication.clipboard().setText(redact_support_text("\n".join(lines)))
        self.copy_support_note.setText("Copied — paste it into your support message.")

    def _stop_preview(self) -> None:
        if self._preview_timer is not None:
            self._preview_timer.stop()

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
        worker.finished.connect(self._thread.quit, Qt.ConnectionType.DirectConnection)
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
        # A signed-in bridge is deliberately left running when setup is closed: it
        # is handed back on the next attempt so she does not sign in twice. The app
        # owns it from here and stops it on quit.
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
