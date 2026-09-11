"""Settings: the camera account, automatic updates, and starting at login.

The updates panel exists because the update path can fail silently. An access
token that quietly expires would stop her receiving fixes with no symptom at all,
so its state is shown here in words and checked once a day in the background.
"""

from __future__ import annotations

import time

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..bridge.credentials import get_update_token, set_update_token
from ..update.updater import UpdateState, Updater
from . import qtutil as Q


class _UpdateWorker(QObject):
    done = Signal(str)          # '' on success, else a message

    def __init__(self, updater: Updater, action: str) -> None:
        super().__init__()
        self.updater = updater
        self.action = action

    def run(self) -> None:
        try:
            if self.action == "check":
                self.updater.refresh_token_health(force=True)
                self.updater.check()
            elif self.action == "download":
                self.updater.download()
            self.done.emit("")
        except Exception as exc:
            self.done.emit(str(exc))


class _Section(QFrame):
    def __init__(self, title: str, subtitle: str = "") -> None:
        super().__init__()
        self.setObjectName("panel")
        self.box = QVBoxLayout(self)
        self.box.setContentsMargins(15, 14, 15, 15)
        self.box.setSpacing(9)
        head = QLabel(title)
        head.setObjectName("h2")
        self.box.addWidget(head)
        if subtitle:
            sub = QLabel(subtitle)
            sub.setObjectName("dim")
            sub.setWordWrap(True)
            self.box.addWidget(sub)


class SettingsScreen(QWidget):
    sign_in_requested = Signal()
    launch_at_login_changed = Signal(bool)

    def __init__(self, updater: Updater) -> None:
        super().__init__()
        self.updater = updater
        self._thread: QThread | None = None
        self._worker: _UpdateWorker | None = None

        # --- camera account -------------------------------------------------
        camera = _Section(
            "Camera account",
            "Surface Guard signs in to Eufy the way the Eufy app does. Your password "
            "is kept in the Mac Keychain, never in the app's own files.",
        )
        self.account_label = QLabel("Not connected")
        self.bridge_label = QLabel("")
        self.bridge_label.setObjectName("mono")
        self.bridge_label.setWordWrap(True)
        sign_in = QPushButton("Sign in to the camera account…")
        sign_in.clicked.connect(self.sign_in_requested.emit)
        camera.box.addWidget(self.account_label)
        camera.box.addWidget(self.bridge_label)
        camera.box.addWidget(sign_in)

        # --- updates --------------------------------------------------------
        updates = _Section(
            "Automatic updates",
            "Updates are downloaded from a private repository and are only installed "
            "if they were signed by the build machine. A bad update host still cannot "
            "put code on this Mac.",
        )
        self.version_label = QLabel("")
        self.update_status = QLabel("Not checked yet")
        self.update_status.setWordWrap(True)
        self.token_status = QLabel("")
        self.token_status.setWordWrap(True)

        self.token_input = QLineEdit()
        self.token_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.token_input.setPlaceholderText("GitHub access token (read-only)")
        save_token = QPushButton("Save token")
        save_token.clicked.connect(self._save_token)
        token_row = QHBoxLayout()
        token_row.addWidget(self.token_input, 1)
        token_row.addWidget(save_token)

        self.check_btn = QPushButton("Check for updates now")
        self.check_btn.clicked.connect(lambda: self._run("check"))
        self.install_btn = QPushButton("Download and install")
        self.install_btn.setObjectName("primary")
        self.install_btn.clicked.connect(self._install)
        self.install_btn.setVisible(False)
        button_row = QHBoxLayout()
        button_row.addWidget(self.check_btn)
        button_row.addWidget(self.install_btn)
        button_row.addStretch(1)

        updates.box.addWidget(self.version_label)
        updates.box.addWidget(self.update_status)
        updates.box.addWidget(self.token_status)
        updates.box.addLayout(token_row)
        updates.box.addLayout(button_row)

        # --- startup --------------------------------------------------------
        startup = _Section("Starting up")
        self.launch = QCheckBox("Open Surface Guard automatically when this Mac starts")
        self.launch.toggled.connect(self.launch_at_login_changed.emit)
        startup.box.addWidget(self.launch)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(13)
        for section in (camera, updates, startup):
            root.addWidget(section)
        root.addStretch(1)

    # ------------------------------------------------------------------ state

    def set_account(self, username: str, bridge_message: str) -> None:
        self.account_label.setText(
            f"Signed in as {username}" if username else "No camera account yet"
        )
        self.bridge_label.setText(bridge_message)

    def set_launch_at_login(self, on: bool) -> None:
        self.launch.blockSignals(True)
        self.launch.setChecked(on)
        self.launch.blockSignals(False)

    def refresh(self) -> None:
        from ..update import build_info

        self.version_label.setText(f"This is version {build_info.VERSION}")
        status = self.updater.status
        self.update_status.setText(status.message)
        colour = {
            UpdateState.FAILED: Q.BAD,
            UpdateState.AVAILABLE: Q.WARN,
            UpdateState.READY: Q.GOOD,
        }.get(status.state, Q.INK_DIM)
        self.update_status.setStyleSheet(f"color: {colour.name()};")
        self.install_btn.setVisible(status.actionable)
        self.install_btn.setText(
            "Install and restart" if status.state is UpdateState.READY
            else "Download and install"
        )

        token = status.token
        if token.checked_at == 0.0:
            self.token_status.setText("Update access has not been checked yet.")
            self.token_status.setStyleSheet(f"color: {Q.INK_DIM.name()};")
        else:
            self.token_status.setText(
                token.summary() + (f"  {token.remedy}" if not token.ok else "")
            )
            good = token.ok and not token.expiring_soon
            self.token_status.setStyleSheet(
                f"color: {(Q.GOOD if good else (Q.WARN if token.ok else Q.BAD)).name()};"
            )
        self.token_input.setPlaceholderText(
            "GitHub access token (a token is saved)" if get_update_token()
            else "GitHub access token (read-only)"
        )

    # ---------------------------------------------------------------- actions

    def _save_token(self) -> None:
        token = self.token_input.text().strip()
        if not token:
            return
        try:
            set_update_token(token)
        except Exception as exc:
            QMessageBox.warning(self, "Could not save the token", str(exc))
            return
        self.token_input.clear()
        self._run("check")

    def _run(self, action: str) -> None:
        if self._thread is not None:
            return
        self.check_btn.setEnabled(False)
        self.install_btn.setEnabled(False)
        self.update_status.setText(
            "Checking…" if action == "check" else "Downloading…"
        )
        self._thread = QThread(self)
        self._worker = _UpdateWorker(self.updater, action)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._on_done)
        self._thread.start()

    def _on_done(self, error: str) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(4000)
            self._thread = None
        self.check_btn.setEnabled(True)
        self.install_btn.setEnabled(True)
        if error:
            self.update_status.setText(error)
            self.update_status.setStyleSheet(f"color: {Q.BAD.name()};")
        self.refresh()

    def _install(self) -> None:
        status = self.updater.status
        if status.state is UpdateState.AVAILABLE:
            self._run("download")
            return
        ok, why = self.updater.can_install()
        if not ok:
            QMessageBox.warning(self, "Cannot install the update", why)
            return
        release = status.release
        confirm = QMessageBox.question(
            self, "Install and restart?",
            f"Surface Guard will restart into version {release.version if release else 'the new build'}.\n\n"
            "Protection stops for a few seconds while it restarts.",
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        error = self.updater.install_and_relaunch()
        if error:
            QMessageBox.warning(self, "Update failed", error)
            return
        # The replacement is launching; this copy must get out of the way.
        from PySide6.QtWidgets import QApplication
        QApplication.quit()
