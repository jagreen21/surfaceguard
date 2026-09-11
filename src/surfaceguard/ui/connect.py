"""Signing in to the camera account, including two-factor and captcha.

Everything Eufy needs is asked for once, in plain words. The password goes
straight to the Keychain and is never written to preferences, logs or the activity
history. Error text comes from Eufy itself, so "Email address or password
incorrect" reaches her as that, not as code 26006.
"""

from __future__ import annotations

import base64

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
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..bridge.client import BridgeClient, BridgeError, DriverPhase, looks_like_camera
from ..bridge.credentials import EufyAccount
from ..bridge.supervisor import BridgeSupervisor
from . import qtutil as Q

# Eufy accounts are region-locked; the wrong country is a common silent failure.
COUNTRIES = [
    ("United States", "US"), ("United Kingdom", "GB"), ("Canada", "CA"),
    ("Germany", "DE"), ("France", "FR"), ("Netherlands", "NL"), ("Spain", "ES"),
    ("Italy", "IT"), ("Sweden", "SE"), ("Norway", "NO"), ("Denmark", "DK"),
    ("Australia", "AU"), ("New Zealand", "NZ"), ("Japan", "JP"),
]


class _SignInWorker(QObject):
    """Runs the bridge start and driver sign-in off the GUI thread."""

    settled = Signal(object)     # DriverState
    failed = Signal(str)

    def __init__(self, supervisor: BridgeSupervisor, client: BridgeClient | None) -> None:
        super().__init__()
        self.supervisor = supervisor
        self.client = client
        self.answer: tuple[str, str] | None = None   # (kind, value)

    def run(self) -> None:
        try:
            if self.answer is None:
                status = self.supervisor.start(wait=True)
                if not status.listening:
                    self.failed.emit(
                        status.fatal or status.message or "The camera service did not start."
                    )
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


class EufySignInDialog(QDialog):
    """Sign in, answer any challenge, then choose which camera to watch."""

    def __init__(self, account: EufyAccount | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Connect your camera")
        self.setMinimumSize(560, 440)
        self.setMaximumWidth(720)        # a form has no business being 1000 px wide

        self.account = account or EufyAccount(username="")
        self.supervisor: BridgeSupervisor | None = None
        self.client: BridgeClient | None = None
        self.chosen: dict | None = None
        self._thread: QThread | None = None
        self._worker: _SignInWorker | None = None

        # --- credentials ---------------------------------------------------
        self.email = QLineEdit(self.account.username)
        self.email.setPlaceholderText("the email you use for the Eufy app")
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        self.password.setPlaceholderText("your Eufy password")
        self.country = QComboBox()
        for label, code in COUNTRIES:
            self.country.addItem(label, code)
        index = self.country.findData(self.account.country)
        self.country.setCurrentIndex(max(0, index))

        creds = QWidget()
        form = QFormLayout(creds)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(11)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        # Without this the fields stay at their size hint and the dialog is mostly
        # empty, which reads as broken rather than minimal.
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.addRow("Email", self.email)
        form.addRow("Password", self.password)
        form.addRow("Region", self.country)
        reassurance = QLabel(
            "Your password is stored in the Mac Keychain and is never saved in the "
            "app's own files."
        )
        reassurance.setObjectName("dim")
        reassurance.setWordWrap(True)
        form.addRow(reassurance)

        # --- challenge -----------------------------------------------------
        challenge = QWidget()
        cbox = QVBoxLayout(challenge)
        cbox.setContentsMargins(0, 0, 0, 0)
        cbox.setSpacing(9)
        self.challenge_text = QLabel("")
        self.challenge_text.setWordWrap(True)
        self.captcha_image = QLabel()
        self.captcha_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.captcha_image.setVisible(False)
        self.challenge_input = QLineEdit()
        self.challenge_input.setPlaceholderText("code")
        self.challenge_input.returnPressed.connect(self._submit_challenge)
        cbox.addWidget(self.challenge_text)
        cbox.addWidget(self.captcha_image)
        cbox.addWidget(self.challenge_input)
        cbox.addStretch(1)

        # --- camera picker -------------------------------------------------
        picker = QWidget()
        pbox = QVBoxLayout(picker)
        pbox.setContentsMargins(0, 0, 0, 0)
        pbox.setSpacing(9)
        # No heading here: the dialog header already asks the question.
        self.devices = QListWidget()
        self.devices.itemDoubleClicked.connect(lambda _: self._advance())
        pbox.addWidget(self.devices, 1)

        self.stack = QStackedWidget()
        for page in (creds, challenge, picker):
            self.stack.addWidget(page)

        self.step = QLabel("")
        self.step.setObjectName("mono")
        self.heading = QLabel("")
        self.heading.setObjectName("h1")
        self.heading.setWordWrap(True)
        self.subtitle = QLabel("")
        self.subtitle.setObjectName("dim")
        self.subtitle.setWordWrap(True)
        header = QVBoxLayout()
        header.setSpacing(4)
        header.addWidget(self.step)
        header.addWidget(self.heading)
        header.addWidget(self.subtitle)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        self.next_btn = QPushButton("Sign in")
        self.next_btn.setObjectName("primary")
        self.next_btn.clicked.connect(self._advance)

        nav = QHBoxLayout()
        nav.addWidget(self.cancel_btn)
        nav.addStretch(1)
        nav.addWidget(self.next_btn)

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 26, 28, 20)
        root.setSpacing(16)
        root.addLayout(header)
        root.addWidget(self.stack, 1)
        root.addWidget(self.status)
        root.addLayout(nav)
        self._sync_header()

    # -------------------------------------------------------------- stepping

    HEADINGS = {
        0: ("Step 1 of 3", "Connect your camera",
            "Sign in with the same account you use in the Eufy app."),
        1: ("Step 2 of 3", "One more check",
            "Eufy wants to be sure it is really you."),
        2: ("Step 3 of 3", "Choose your camera",
            "Pick the one pointing at the surfaces you want protected."),
    }

    def _sync_header(self) -> None:
        step, heading, subtitle = self.HEADINGS[self.stack.currentIndex()]
        self.step.setText(step.upper())
        self.heading.setText(heading)
        self.subtitle.setText(subtitle)

    def _advance(self) -> None:
        page = self.stack.currentIndex()
        if page == 0:
            self._start_sign_in()
        elif page == 1:
            self._submit_challenge()
        else:
            self._finish()

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
        # Drop the plaintext from the widget now that the Keychain has it.
        self.password.clear()

        self.supervisor = BridgeSupervisor(self.account)
        self._say("Starting the camera service and signing in…")
        self._run_worker(None)

    def _submit_challenge(self) -> None:
        value = self.challenge_input.text().strip()
        if not value:
            self._say("Enter the code first.", bad=True)
            return
        kind = "captcha" if self.captcha_image.isVisible() else "code"
        self._say("Checking…")
        self._run_worker((kind, value))

    def _run_worker(self, answer: tuple[str, str] | None) -> None:
        assert self.supervisor is not None
        self.next_btn.setEnabled(False)
        self._thread = QThread(self)
        self._worker = _SignInWorker(self.supervisor, self.client)
        self._worker.answer = answer
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.settled.connect(self._on_settled)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    def _end_worker(self) -> None:
        if self._worker is not None:
            self.client = self._worker.client
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(4000)
            self._thread = None
        self.next_btn.setEnabled(True)

    # ------------------------------------------------------------- outcomes

    def _on_failed(self, message: str) -> None:
        self._end_worker()
        self._say(message, bad=True)

    def _on_settled(self, state) -> None:
        self._end_worker()
        if state.phase is DriverPhase.CONNECTED:
            self._say("Signed in. Looking for cameras…")
            self._load_devices()
            return
        if state.phase is DriverPhase.NEEDS_CODE:
            self.captcha_image.setVisible(False)
            self.challenge_text.setText(
                "Eufy emailed a verification code to "
                f"{self.account.username}. Enter it here."
            )
            self.challenge_input.clear()
            self.challenge_input.setPlaceholderText("6-digit code")
            self.stack.setCurrentIndex(1)
            self._sync_header()
            self.next_btn.setText("Continue")
            self.challenge_input.setFocus()
            self._say("")
            return
        if state.phase is DriverPhase.NEEDS_CAPTCHA:
            self.challenge_text.setText("Type the characters shown in the picture.")
            self._show_captcha(state.captcha_image)
            self.challenge_input.clear()
            self.challenge_input.setPlaceholderText("characters from the picture")
            self.stack.setCurrentIndex(1)
            self._sync_header()
            self.next_btn.setText("Continue")
            self.challenge_input.setFocus()
            self._say("")
            return
        self._say(state.message or "Eufy refused the sign-in.", bad=True)
        self.stack.setCurrentIndex(0)
        self._sync_header()
        self.next_btn.setText("Sign in")

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
        assert self.client is not None
        try:
            found = self.client.devices()
        except BridgeError as exc:
            self._say(str(exc), bad=True)
            return
        cameras = [d for d in found if looks_like_camera(d)]
        self.devices.clear()
        for device in cameras:
            label = device.get("name") or device.get("serialNumber", "camera")
            model = device.get("model", "")
            item = QListWidgetItem(f"{label}\n{model or 'camera'}")
            item.setToolTip(str(device.get("serialNumber", "")))
            item.setData(Qt.ItemDataRole.UserRole, device)
            self.devices.addItem(item)
        if not cameras:
            self._say(
                "Signed in, but this account has no cameras on it. Check you used the "
                "account the camera is registered to.", bad=True,
            )
            return
        self.devices.setCurrentRow(0)
        self.stack.setCurrentIndex(2)
        self._sync_header()
        self.next_btn.setText("Use this camera")
        self._say(f"Found {len(cameras)} camera{'s' if len(cameras) != 1 else ''}.")

    def _finish(self) -> None:
        item = self.devices.currentItem()
        if item is None:
            self._say("Choose a camera first.", bad=True)
            return
        self.chosen = item.data(Qt.ItemDataRole.UserRole)
        self.accept()

    def _say(self, text: str, bad: bool = False) -> None:
        self.status.setText(text)
        self.status.setStyleSheet(f"color: {(Q.BAD if bad else Q.INK_DIM).name()};")

    # ------------------------------------------------------------- teardown

    def reject(self) -> None:
        self._end_worker()
        if self.client is not None:
            self.client.close()
            self.client = None
        if self.supervisor is not None:
            self.supervisor.stop()
            self.supervisor = None
        super().reject()
