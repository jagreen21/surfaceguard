"""Application lifecycle: window, tray, engine wiring, and staying awake.

Engine callbacks arrive on the engine thread, so everything crosses into Qt through
_Bridge's signals rather than touching widgets directly. The app keeps running from
the menu bar when its window is closed, and holds a power assertion while armed —
a sleeping laptop is the most common way "forget about it" quietly stops being true.
"""

from __future__ import annotations

import argparse
import html
import os
import plistlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QMenu,
    QMessageBox,
    QSystemTrayIcon,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

if __package__ in (None, ""):  # allow `python src/surfaceguard/app.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "surfaceguard"

from .audio.player import Player
from .bridge.credentials import EufyAccount
from .logging_setup import get as get_logger, log_dir, setup as setup_logging
from .bridge.supervisor import BridgeSupervisor
from .camera.sources.base import CameraSource, SourceError
from .detection.cat_detector import Detector, SyntheticDetector, load_detector
from .engine import Engine, FrameResult
from .health.heartbeat import Report
from .state import (
    DeviceCapabilities,
    DeviceKind,
    DeviceViewState,
    Phase,
    ProductViewState,
    RoomViewState,
    StateStore,
)
from .storage.activity_log import ActivityLog
from .storage.preferences import Preferences, support_dir
from .storage.review_policy import REVIEW_INTERVAL_S, build_deck, consider
from .storage.room_map import load_room_map, save_room_map
from .ui import qtutil as Q
from .ui.activity import ActivityScreen
from .ui.audio import AudioScreen
from .ui.camera import CameraScreen
from .ui.connect import EufySignInDialog
from .ui.detection import DetectionScreen
from .ui.devices import DevicesScreen
from .ui.diagnostics import DiagnosticsScreen
from .ui.home import HomeScreen
from .ui.review import ReviewDialog
from .ui.onboarding import OnboardingDialog, _build_source
from .ui.rooms import RoomsScreen
from .ui.settings import SettingsScreen
from .ui.shell import AppShell
from .ui.surface_editor import SurfaceEditor
from .update import build_info
from .update.updater import Updater, cleanup_previous

logger = get_logger("app")

LAUNCH_AGENT = Path.home() / "Library" / "LaunchAgents" / "com.surfaceguard.app.plist"
UI_REFRESH_MS = 120
SLOW_REFRESH_MS = 1500
# The invitation is not urgent, and building a deck reads a week of history.
REVIEW_CHECK_EVERY_S = 300.0


class ReviewFlow:
    """The weekly review, shared by the shell and the window it replaces.

    Kept as a mixin rather than copied: the invitation's rules about when to ask
    are the feature, and two drifting copies of them is exactly how an app ends up
    nagging someone it had already agreed to stop asking.
    """

    def _init_review(self) -> None:
        self._review: ReviewDialog | None = None
        self._last_review_check = 0.0

    def _connect_review(self) -> None:
        self.home.review_requested.connect(self.start_review)
        self.home.review_declined.connect(self._decline_review)
        self.home.review_silenced.connect(self._silence_review)

    def _maybe_offer_review(self, now: float) -> None:
        """Offer the weekly review, but only when there is something worth asking.

        Deck quality gates this, not the calendar: an invitation that arrives on
        schedule with three dull cards in it is the one that teaches the user to
        dismiss the card without reading it.
        """
        if self._review is not None and self._review.isVisible():
            return
        if now - self._last_review_check < REVIEW_CHECK_EVERY_S:
            return
        self._last_review_check = now
        deck = self._build_deck()
        invitation = consider(
            deck, self.prefs.last_review_at, enabled=self.prefs.review_enabled,
            declines=self.prefs.review_declines,
        )
        if invitation.offer:
            self.home.invite.offer(invitation.headline, invitation.body)
        else:
            self.home.invite.setVisible(False)
            logger.debug("no review offered: %s", invitation.reason)

    def _build_deck(self):
        window = max(self.prefs.last_review_at, time.time() - REVIEW_INTERVAL_S)
        return build_deck(self.log.since(window), window_start=window)

    def start_review(self) -> None:
        deck = self._build_deck()
        if deck.is_empty:
            QMessageBox.information(
                self, "Nothing to review",
                "There is nothing I am unsure about this week. That is the good "
                "outcome — I will ask again when there is.",
            )
            self.home.invite.setVisible(False)
            return
        self.home.invite.setVisible(False)
        dialog = ReviewDialog(
            deck, self.prefs.surfaces, self.log,
            pictures_are_temporary=(self.prefs.review_pictures_only
                                    and not self.prefs.save_thumbnails),
            parent=self,
        )
        dialog.tuning_changed.connect(self._on_tuning_changed)
        dialog.finished_review.connect(self._on_review_finished)
        self._review = dialog
        dialog.show()

    def _on_tuning_changed(self) -> None:
        """An adjustment was applied or taken back: persist it and use it now."""
        self.prefs.save()
        self.engine.set_surfaces(self.prefs.surfaces)

    def _on_review_finished(self) -> None:
        self.prefs.last_review_at = time.time()
        self.prefs.review_declines = 0
        self.prefs.save()
        self.activity.refresh()
        self._last_review_check = 0.0

    def _decline_review(self) -> None:
        self.prefs.review_declines += 1
        self.prefs.last_review_offered_at = time.time()
        self.prefs.save()
        self.home.invite.setVisible(False)

    def _silence_review(self) -> None:
        self.prefs.review_enabled = False
        self.prefs.save()
        self.home.invite.setVisible(False)


class _Bridge(QObject):
    """Marshals engine-thread callbacks onto the GUI thread."""

    frame = Signal(object)
    trigger = Signal(object, object)
    heartbeat = Signal(object)


class _LegacyMainWindow(ReviewFlow, QWidget):
    def __init__(
        self,
        engine: Engine,
        prefs: Preferences,
        log: ActivityLog,
        updater: Updater | None = None,
        supervisor: BridgeSupervisor | None = None,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.prefs = prefs
        self.log = log
        self.updater = updater or Updater()
        self.supervisor = supervisor
        engine.updater = self.updater
        engine.bridge = supervisor
        self._quitting = False
        self._caffeinate: subprocess.Popen | None = None
        self._init_review()

        self.setWindowTitle("Surface Guard")
        self.resize(1160, 740)

        self.home = HomeScreen()
        self.editor = SurfaceEditor()
        self.activity = ActivityScreen(log)
        self.diagnostics = DiagnosticsScreen(engine)
        self.settings = SettingsScreen(self.updater)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.home, "Home")
        self.tabs.addTab(self.editor, "Surfaces")
        self.tabs.addTab(self.activity, "Activity")
        self.tabs.addTab(self.diagnostics, "Diagnostics")
        self.tabs.addTab(self.settings, "Settings")
        self.tabs.currentChanged.connect(self._on_tab)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self.tabs)

        self.home.arm_requested.connect(self.arm)
        self.home.off_requested.connect(self.turn_off)
        self.home.pause_requested.connect(self.pause_for)
        self.home.test_sound_requested.connect(self.test_sound)
        self.editor.surfaces_changed.connect(self._on_surfaces_changed)
        self._connect_review()
        self.activity.feedback_given.connect(self._on_feedback)
        self.activity.retention_changed.connect(self._on_retention)
        self.diagnostics.rescan_requested.connect(self.rescan_room)
        self.settings.sign_in_requested.connect(self.connect_camera)
        self.settings.launch_at_login_changed.connect(self._set_launch_at_login)

        self.bridge = _Bridge()
        self.bridge.frame.connect(self._on_frame)
        self.bridge.trigger.connect(self._on_trigger)
        self.bridge.heartbeat.connect(self._on_heartbeat)
        engine.on_frame = self.bridge.frame.emit
        engine.on_trigger = lambda d, r: self.bridge.trigger.emit(d, r)
        engine.on_heartbeat = self.bridge.heartbeat.emit

        self._fast = QTimer(self)
        self._fast.timeout.connect(self._refresh_state)
        self._fast.start(UI_REFRESH_MS)
        self._slow = QTimer(self)
        self._slow.timeout.connect(self._refresh_slow)
        self._slow.start(SLOW_REFRESH_MS)

        self.tray = self._build_tray()
        self.reload_from_prefs()
        self.activity.set_retention(prefs.keep_thumbnails_days, prefs.save_thumbnails)
        self.activity.refresh()
        self.settings.set_launch_at_login(prefs.launch_at_login)
        self._refresh_settings()
        self.updater.start()

    # -------------------------------------------------------------------- tray

    def _build_tray(self) -> QSystemTrayIcon:
        tray = QSystemTrayIcon(self)
        tray.setIcon(_dot_icon(Q.INK_DIM))
        menu = QMenu()
        self.act_show = QAction("Open Surface Guard", self)
        self.act_show.triggered.connect(self._show_window)
        self.act_toggle = QAction("Turn protection on", self)
        self.act_toggle.triggered.connect(self._tray_toggle)
        self.act_pause = QAction("Pause 30 minutes", self)
        self.act_pause.triggered.connect(lambda: self.pause_for(1800.0))
        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self.quit)
        for a in (self.act_show, self.act_toggle, self.act_pause):
            menu.addAction(a)
        menu.addSeparator()
        menu.addAction(act_quit)
        tray.setContextMenu(menu)
        tray.activated.connect(
            lambda reason: self._show_window()
            if reason == QSystemTrayIcon.ActivationReason.Trigger else None
        )
        tray.show()
        return tray

    def _show_window(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _tray_toggle(self) -> None:
        if self.engine.state.state().is_guarding:
            self.turn_off()
        else:
            self.arm()

    # ---------------------------------------------------------------- controls

    def arm(self) -> None:
        state = self.engine.state
        if not (self.prefs.camera or {}).get("kind"):
            self.connect_camera()
            return
        if not state.has_map:
            self.run_setup()
            return
        if not state.has_surfaces:
            self.tabs.setCurrentWidget(self.editor)
            self.editor.begin_drawing()
            return
        state.arm()
        if not self.engine.running:
            try:
                self.engine.start()
            except SourceError as exc:
                QMessageBox.warning(self, "Camera trouble", str(exc))
                state.turn_off()
                return
        self._hold_awake(True)

    def turn_off(self) -> None:
        self.engine.state.turn_off()
        self._hold_awake(False)

    def pause_for(self, seconds: float) -> None:
        self.engine.state.pause_for(seconds)
        self._hold_awake(False)

    def test_sound(self) -> None:
        surfaces = self.engine.prefs.surfaces
        sound = surfaces[0].deterrent.sound if surfaces else "chirp"
        volume = surfaces[0].deterrent.volume if surfaces else self.prefs.master_volume
        outcome = self.engine.player.play(sound, volume)
        if not outcome.ok:
            QMessageBox.warning(
                self, "Sound could not play",
                f"{outcome.error}\n\nCheck the output device in System Settings > Sound.",
            )

    # ---------------------------------------------------------- camera account

    def connect_camera(self) -> bool:
        """Sign in to Eufy and adopt the chosen camera as the live source."""
        account = EufyAccount(
            username=str((self.prefs.camera or {}).get("username", "")),
            country=str((self.prefs.camera or {}).get("country", "US")),
        )
        dialog = EufySignInDialog(account, self)
        if dialog.exec() != EufySignInDialog.DialogCode.Accepted or dialog.chosen is None:
            return False

        device = dialog.chosen
        from .camera.sources.eufy_bridge import EufyBridgeCamera

        was_running = self.engine.running
        if was_running:
            self.engine.stop()
        if self.supervisor is not None and self.supervisor is not dialog.supervisor:
            self.supervisor.stop()
        self.supervisor = dialog.supervisor
        self.engine.bridge = self.supervisor

        source = EufyBridgeCamera(
            client=dialog.client,
            serial=str(device.get("serialNumber", "")),
            model=str(device.get("model", "")),
            name=str(device.get("name", "")),
        )
        self.engine.source = source
        self.prefs.camera = {
            "kind": "eufy",
            "username": dialog.account.username,
            "country": dialog.account.country,
            "serial": source.serial,
            "model": source.model,
            "name": source.device_name,
        }
        self.prefs.save()
        self.engine.detector = _make_detector(self.prefs, source)
        self._refresh_settings()
        try:
            self.engine.start()
        except SourceError as exc:
            QMessageBox.warning(self, "Camera trouble", str(exc))
            return False
        if self.engine.room_map is None:
            QMessageBox.information(
                self, "Camera connected",
                "Next, Surface Guard will look around the room so you can draw the "
                "surfaces you want protected.",
            )
            return self.run_setup()
        return True

    def _refresh_settings(self) -> None:
        username = str((self.prefs.camera or {}).get("username", ""))
        message = self.supervisor.status.message if self.supervisor else "Not started"
        self.settings.set_account(username, message)
        self.settings.refresh()

    def _set_launch_at_login(self, on: bool) -> None:
        self.prefs.launch_at_login = set_launch_at_login(on)
        self.prefs.save()

    # ------------------------------------------------------------------- setup

    def run_setup(self) -> bool:
        # Re-running setup should default to however the camera is already set up.
        dialog = OnboardingDialog(
            self, allow_demo=True, preselect=str((self.prefs.camera or {}).get("kind", "")),
        )
        dialog.bind_player(lambda: self.engine.player.play("chirp", 0.6))
        if dialog.exec() != OnboardingDialog.DialogCode.Accepted or dialog.room is None:
            if dialog.source is not None and dialog.source is not self.engine.source:
                dialog.source.stop()
            return False

        was_running = self.engine.running
        if was_running:
            self.engine.stop()
        if dialog.source is not None:
            self.engine.source = dialog.source
            self.prefs.camera = {
                "kind": dialog.choice().kind,
                "url": dialog.choice().url,
                "serial": dialog.choice().serial,
            }
            if isinstance(self.engine.detector, SyntheticDetector):
                self.engine.detector = _make_detector(self.prefs, dialog.source)
        self.engine.set_room_map(dialog.room)
        save_room_map(dialog.room)
        self.prefs.save()
        self.reload_from_prefs()
        self.tabs.setCurrentWidget(self.editor)
        self.editor.begin_drawing()
        return True

    def rescan_room(self) -> None:
        confirm = QMessageBox.question(
            self, "Scan the room again?",
            "The surfaces you have drawn are tied to the current room picture. "
            "Scanning again will keep them, but they may need nudging into place.",
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.run_setup()

    def reload_from_prefs(self) -> None:
        canvas = self.engine.room_map.canvas if self.engine.room_map else None
        self.editor.load(canvas, self.prefs.surfaces)
        self.engine.state.has_surfaces = bool(self.prefs.surfaces)

    # ------------------------------------------------------------------ signals

    def _on_frame(self, result: FrameResult) -> None:
        self.home.live.update_result(result, self.prefs.surfaces)
        if self.tabs.currentWidget() is self.editor:
            self.editor.set_live_pose(result.pose)

    def _on_trigger(self, decision, _result) -> None:
        self.tray.showMessage(
            "Cat on the " + decision.surface_name,
            "Deterrent played.", QSystemTrayIcon.MessageIcon.Information, 4000,
        )
        self.activity.refresh()
        if self._review is not None and self._review.isVisible():
            self._review.note_alert("cat on the " + decision.surface_name)

    def _on_heartbeat(self, report: Report) -> None:
        if not report.ok:
            first = report.failures[0]
            self.tray.showMessage(
                "Surface Guard is not protecting",
                f"{first.detail}\n{first.remedy}",
                QSystemTrayIcon.MessageIcon.Warning, 8000,
            )

    def _on_surfaces_changed(self) -> None:
        self.engine.set_surfaces(self.editor.surfaces)
        self.prefs.surfaces = self.editor.surfaces
        self.prefs.save()

    def _on_feedback(self, surface_id: str, verdict: str) -> None:
        if verdict != "not_a_cat":
            return
        surface = self.prefs.surface_by_id(surface_id)
        if surface is not None:
            # Drop the sample that this false positive contributed to the size
            # model, so one bad trigger does not skew the gate that should catch it.
            surface.forget_last_observation()
            self.prefs.save()
            self.editor._rebuild_list()

    def _on_retention(self, days: int, save: bool) -> None:
        self.prefs.keep_thumbnails_days = days
        self.prefs.save_thumbnails = save
        self.prefs.save()
        if not save:
            self.log.forget_all_thumbnails()
        else:
            self.log.prune(days)
        self.activity.refresh()

    def _on_tab(self, _index: int) -> None:
        if self.tabs.currentWidget() is self.activity:
            self.activity.refresh()

    # ------------------------------------------------------------------ refresh

    def _refresh_state(self) -> None:
        state = self.engine.state.state()
        self.home.render_state(state)
        colour = Q.SEVERITY_COLOUR[state.severity.value]
        self.tray.setIcon(_dot_icon(colour))
        self.tray.setToolTip(f"Surface Guard — {state.headline}")
        self.act_toggle.setText(
            "Turn protection off" if state.is_guarding else "Turn protection on"
        )
        self.act_pause.setEnabled(state.is_guarding)
        if state.phase is Phase.OFF:
            self.home.live.set_message("Protection is off.")

    def _refresh_slow(self) -> None:
        plan = self.engine.build_plan() if self.engine.room_map else None
        presence = {s.id: self.engine.policy.state_of(s.id) for s in self.prefs.surfaces}
        self.home.render_coverage(self.prefs.surfaces, plan, presence)
        recent = self.log.recent(limit=1)
        self.home.render_last_event(
            f"{recent[0].when} — {recent[0].surface_name}: {recent[0].reason}"
            if recent else "Nothing yet."
        )
        self._maybe_offer_review(time.time())
        if self.tabs.currentWidget() is self.diagnostics:
            self.diagnostics.refresh()
        elif self.tabs.currentWidget() is self.settings:
            self._refresh_settings()

    # ------------------------------------------------------------------ awake

    def _hold_awake(self, on: bool) -> None:
        """`caffeinate -i` while armed: a napping Mac is a silent failure."""
        if on and self._caffeinate is None and sys.platform == "darwin":
            try:
                self._caffeinate = subprocess.Popen(
                    ["caffeinate", "-i", "-w", str(os.getpid())],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except OSError:
                self._caffeinate = None
        elif not on and self._caffeinate is not None:
            self._caffeinate.terminate()
            self._caffeinate = None

    # ---------------------------------------------------------------- lifecycle

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._quitting:
            event.accept()
            return
        event.ignore()
        self.hide()
        self.tray.showMessage(
            "Still watching",
            "Surface Guard keeps running in the menu bar.",
            QSystemTrayIcon.MessageIcon.Information, 2500,
        )

    def quit(self) -> None:
        self._quitting = True
        self.prefs.surfaces = self.editor.surfaces
        self.prefs.save()
        self._hold_awake(False)
        self.updater.stop()
        self.engine.stop()
        if self.supervisor is not None:
            self.supervisor.stop()
        self.log.close()
        QApplication.quit()


class MainWindow(ReviewFlow, QWidget):
    """Consumer shell around the existing camera, detection, and audio engine."""

    def __init__(self, engine: Engine, prefs: Preferences, log: ActivityLog,
                 updater: Updater | None = None,
                 supervisor: BridgeSupervisor | None = None) -> None:
        super().__init__()
        self.engine, self.prefs, self.log = engine, prefs, log
        self.updater = updater or Updater()
        self.supervisor = supervisor
        engine.updater, engine.bridge = self.updater, supervisor
        self._quitting = False
        self._caffeinate: subprocess.Popen | None = None
        self._init_review()
        self.setWindowTitle("Surface Guard")
        self.setMinimumSize(760, 560)
        self.resize(1180, 760)

        self.editor = SurfaceEditor()
        self.home = HomeScreen()
        self.rooms = RoomsScreen(self.editor, prefs.room_name)
        self.detection = DetectionScreen()
        self.audio_screen = AudioScreen()
        self.camera_screen = CameraScreen()
        self.devices = DevicesScreen()
        self.activity = ActivityScreen(log)
        self.diagnostics = DiagnosticsScreen(engine)
        self.settings = SettingsScreen(self.updater)
        self.shell = AppShell({
            "Home": self.home, "Rooms": self.rooms, "Detection": self.detection,
            "Audio": self.audio_screen, "Camera": self.camera_screen,
            "Devices": self.devices, "Settings": self.settings,
        })
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self.shell)
        self._wire_ui()

        self.bridge = _Bridge()
        self.bridge.frame.connect(self._on_frame)
        self.bridge.trigger.connect(self._on_trigger)
        self.bridge.heartbeat.connect(self._on_heartbeat)
        engine.on_frame = self.bridge.frame.emit
        engine.on_trigger = lambda d, r: self.bridge.trigger.emit(d, r)
        engine.on_heartbeat = self.bridge.heartbeat.emit

        self._fast = QTimer(self)
        self._fast.timeout.connect(self._refresh_state)
        self._fast.start(UI_REFRESH_MS)
        self._slow = QTimer(self)
        self._slow.timeout.connect(self._refresh_slow)
        self._slow.start(SLOW_REFRESH_MS)
        self.tray = self._build_tray()
        self.reload_from_prefs()
        self.activity.set_retention(prefs.keep_thumbnails_days, prefs.save_thumbnails)
        self.activity.refresh()
        self.settings.set_launch_at_login(prefs.launch_at_login)
        self._refresh_settings()
        self._refresh_product_ui()
        self.updater.start()

    def _wire_ui(self) -> None:
        self.home.arm_requested.connect(self.arm)
        self.home.off_requested.connect(self.turn_off)
        self.home.pause_requested.connect(self.pause_for)
        self.home.test_sound_requested.connect(self.test_sound)
        self.home.navigate_requested.connect(self.shell.show_page)
        self.home.activity_requested.connect(self.show_activity)
        self._connect_review()
        self.editor.surfaces_changed.connect(self._on_surfaces_changed)
        self.editor.test_sound_requested.connect(
            lambda: self.test_sound(self.editor.canvas.selected)
        )
        self.rooms.room_name_changed.connect(self._on_room_name_changed)
        self.detection.protection_requested.connect(lambda on: self.arm() if on else self.turn_off())
        self.detection.sensitivity_changed.connect(self._on_sensitivity_changed)
        self.detection.cooldown_changed.connect(self._on_global_cooldown)
        self.audio_screen.settings_changed.connect(self._on_audio_settings_changed)
        self.audio_screen.test_requested.connect(
            lambda: self.test_sound(self.audio_screen.current_surface)
        )
        self.audio_screen.output_changed.connect(self._on_output_changed)
        self.audio_screen.import_requested.connect(self.import_sound)
        self.camera_screen.connect_requested.connect(self.connect_camera)
        self.camera_screen.snapshot_requested.connect(self.save_snapshot)
        self.camera_screen.camera_sound_requested.connect(self.test_camera_speaker)
        self.camera_screen.overlays_changed.connect(self._on_overlays_changed)
        self.activity.feedback_given.connect(self._on_feedback)
        self.activity.retention_changed.connect(self._on_retention)
        self.diagnostics.rescan_requested.connect(self.rescan_room)
        self.settings.sign_in_requested.connect(self.connect_camera)
        self.settings.launch_at_login_changed.connect(self._set_launch_at_login)
        self.settings.system_health_requested.connect(self.show_system_health)
        self.shell.health_requested.connect(self.show_system_health)
        self.shell.page_changed.connect(self._on_page)

    def _build_tray(self) -> QSystemTrayIcon:
        tray = QSystemTrayIcon(self)
        tray.setIcon(_dot_icon(Q.INK_DIM))
        menu = QMenu()
        self.act_show = QAction("Open Surface Guard", self)
        self.act_show.triggered.connect(self._show_window)
        self.act_toggle = QAction("Turn protection on", self)
        self.act_toggle.triggered.connect(lambda: self.turn_off()
                                          if self.engine.state.state().is_guarding else self.arm())
        self.act_pause = QAction("Pause 30 minutes", self)
        self.act_pause.triggered.connect(lambda: self.pause_for(1800.0))
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.quit)
        for action in (self.act_show, self.act_toggle, self.act_pause):
            menu.addAction(action)
        menu.addSeparator(); menu.addAction(quit_action)
        tray.setContextMenu(menu)
        tray.activated.connect(lambda reason: self._show_window()
                               if reason == QSystemTrayIcon.ActivationReason.Trigger else None)
        tray.show()
        return tray

    def _show_window(self) -> None:
        self.showNormal(); self.raise_(); self.activateWindow()

    def arm(self) -> None:
        state = self.engine.state
        if not (self.prefs.camera or {}).get("kind"):
            self.connect_camera(); return
        if not state.has_map:
            self.run_setup(); return
        if not state.has_surfaces:
            self.shell.show_page("Rooms"); self.editor.begin_drawing(); return
        state.arm()
        if not self.engine.running:
            try:
                self.engine.start()
            except SourceError as exc:
                QMessageBox.warning(self, "Camera trouble", str(exc))
                state.turn_off(); return
        self._hold_awake(True)

    def turn_off(self) -> None:
        self.engine.state.turn_off(); self._hold_awake(False)

    def pause_for(self, seconds: float) -> None:
        self.engine.state.pause_for(seconds); self._hold_awake(False)

    def test_sound(self, surface=None) -> None:
        surfaces = self.engine.prefs.surfaces
        chosen = surface or (surfaces[0] if surfaces else None)
        sound = chosen.deterrent.sound if chosen is not None else "chirp"
        volume = chosen.deterrent.volume if chosen is not None else self.prefs.master_volume
        if (self.prefs.prefer_camera_speaker and
                self.engine.source.capabilities.has_speaker and
                self.engine.source.play_sound_on_camera(sound)):
            return
        outcome = self.engine.player.play(sound, volume)
        if not outcome.ok:
            QMessageBox.warning(self, "Sound could not play",
                                f"{outcome.error}\n\nCheck the output device in System Settings > Sound.")

    def test_camera_speaker(self) -> None:
        if not self.engine.source.play_sound_on_camera("chirp"):
            QMessageBox.information(self, "Camera speaker unavailable",
                                    "This camera connection cannot play a sound directly. Use This Mac instead.")

    def import_sound(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import a sound", "", "Wave audio (*.wav)")
        if not path:
            return
        source = Path(path)
        sounds = support_dir() / "sounds"
        sounds.mkdir(parents=True, exist_ok=True)
        target = sounds / source.name
        try:
            shutil.copy2(source, target)
        except OSError as exc:
            QMessageBox.warning(self, "Could not import sound", str(exc)); return
        self.engine.player.custom[source.stem] = target
        self.prefs.custom_sounds[source.stem] = str(target)
        self.prefs.save(); self._refresh_audio()

    def save_snapshot(self) -> None:
        result = self.engine.last_result
        if result is None:
            QMessageBox.information(self, "No picture yet",
                                    "Wait for the camera picture to appear, then try again.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Snapshot", str(Path.home() / "Desktop" / "Surface Guard Snapshot.jpg"),
            "JPEG image (*.jpg);;PNG image (*.png)")
        if not path:
            return
        import cv2
        if not cv2.imwrite(path, result.frame.image):
            QMessageBox.warning(self, "Could not save snapshot", "Choose another location and try again.")

    def connect_camera(self) -> bool:
        account = EufyAccount(username=str((self.prefs.camera or {}).get("username", "")),
                              country=str((self.prefs.camera or {}).get("country", "US")))
        dialog = EufySignInDialog(account, self)
        if dialog.exec() != EufySignInDialog.DialogCode.Accepted or dialog.chosen is None:
            return False
        device = dialog.chosen
        from .camera.sources.eufy_bridge import EufyBridgeCamera
        if self.engine.running:
            self.engine.stop()
        if self.supervisor is not None and self.supervisor is not dialog.supervisor:
            self.supervisor.stop()
        self.supervisor = dialog.supervisor
        self.engine.bridge = self.supervisor
        source = EufyBridgeCamera(client=dialog.client,
                                  serial=str(device.get("serialNumber", "")),
                                  model=str(device.get("model", "")),
                                  name=str(device.get("name", "")))
        self.engine.source = source
        self.prefs.camera = {"kind": "eufy", "username": dialog.account.username,
                             "country": dialog.account.country, "serial": source.serial,
                             "model": source.model, "name": source.device_name}
        self.prefs.save()
        self.engine.detector = _make_detector(self.prefs, source)
        self._refresh_settings()
        try:
            self.engine.start()
        except SourceError as exc:
            QMessageBox.warning(self, "Camera trouble", str(exc)); return False
        if self.engine.room_map is None:
            QMessageBox.information(self, "Camera connected",
                                    "Next, Surface Guard will look around the room so you can draw protected surfaces.")
            return self.run_setup()
        self._refresh_product_ui()
        return True

    def _refresh_settings(self) -> None:
        username = str((self.prefs.camera or {}).get("username", ""))
        message = self.supervisor.status.message if self.supervisor else "Not started"
        self.settings.set_account(username, message); self.settings.refresh()

    def _set_launch_at_login(self, on: bool) -> None:
        self.prefs.launch_at_login = set_launch_at_login(on); self.prefs.save()

    def run_setup(self, existing_source=None) -> bool:
        dialog = OnboardingDialog(self, allow_demo=True,
                                  preselect=str((self.prefs.camera or {}).get("kind", "")),
                                  existing_source=existing_source)
        dialog.bind_player(lambda: self.engine.player.play("chirp", 0.6))
        if dialog.exec() != OnboardingDialog.DialogCode.Accepted or dialog.room is None:
            if dialog.source is not None and dialog.source is not self.engine.source:
                dialog.source.stop()
            return False
        if self.engine.running:
            self.engine.stop()
        if dialog.source is not None:
            self.engine.source = dialog.source
            self.prefs.camera = {"kind": dialog.choice().kind, "url": dialog.choice().url,
                                 "serial": dialog.choice().serial}
            self.engine.detector = _make_detector(self.prefs, dialog.source)
        self.engine.set_room_map(dialog.room)
        save_room_map(dialog.room); self.prefs.save(); self.reload_from_prefs()
        self.shell.show_page("Rooms"); self.editor.begin_drawing()
        return True

    def rescan_room(self) -> None:
        answer = QMessageBox.question(
            self, "Scan the room again?",
            "Your protected surfaces will remain, but they may need nudging into place.",
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes)
        if answer == QMessageBox.StandardButton.Yes:
            self.run_setup()

    def reload_from_prefs(self) -> None:
        canvas = self.engine.room_map.canvas if self.engine.room_map else None
        self.editor.load(canvas, self.prefs.surfaces)
        self.engine.state.has_surfaces = bool(self.prefs.surfaces)
        for name, path in self.prefs.custom_sounds.items():
            custom = Path(path)
            if custom.exists():
                self.engine.player.custom[name] = custom
        self._refresh_audio()

    def _on_frame(self, result: FrameResult) -> None:
        self.home.live.update_result(result, self.prefs.surfaces)
        self.camera_screen.live.update_result(result, self.prefs.surfaces)
        if self.shell.current_name == "Rooms":
            self.editor.set_live_pose(result.pose)

    def _on_trigger(self, decision, _result) -> None:
        self.tray.showMessage("Cat on the " + decision.surface_name, "Audio response played.",
                              QSystemTrayIcon.MessageIcon.Information, 4000)
        self.activity.refresh()
        if self._review is not None and self._review.isVisible():
            self._review.note_alert("cat on the " + decision.surface_name)

    def _on_heartbeat(self, report: Report) -> None:
        if not report.ok:
            first = report.failures[0]
            self.tray.showMessage("Surface Guard is not protecting",
                                  f"{first.detail}\n{first.remedy}",
                                  QSystemTrayIcon.MessageIcon.Warning, 8000)

    def _on_surfaces_changed(self) -> None:
        self.engine.set_surfaces(self.editor.surfaces)
        self.prefs.surfaces = self.editor.surfaces
        self.prefs.save(); self._refresh_audio(); self._refresh_product_ui()

    def _on_room_name_changed(self, name: str) -> None:
        self.prefs.room_name = name; self.prefs.save(); self._refresh_product_ui()

    def _on_sensitivity_changed(self, value: str) -> None:
        self.prefs.detection_sensitivity = value
        if hasattr(self.engine.detector, "conf"):
            self.engine.detector.conf = {"Calm": 0.52, "Balanced": 0.35, "Sensitive": 0.22}[value]
        self.prefs.save(); self._refresh_product_ui()

    def _on_global_cooldown(self, seconds: float) -> None:
        for surface in self.prefs.surfaces:
            surface.deterrent.cooldown_s = seconds
        self.prefs.save(); self.editor._rebuild_list(); self._refresh_audio()

    def _on_audio_settings_changed(self) -> None:
        self.prefs.save(); self.editor._rebuild_list(); self._refresh_product_ui()

    def _on_output_changed(self, prefer_camera: bool) -> None:
        self.prefs.prefer_camera_speaker = prefer_camera
        self.prefs.save(); self._refresh_product_ui()

    def _on_overlays_changed(self, zones: bool, boxes: bool, labels: bool) -> None:
        self.prefs.show_protected_zones, self.prefs.show_detection_boxes = zones, boxes
        self.prefs.show_surface_labels = labels
        self.home.live.set_overlays(zones, boxes, labels); self.prefs.save()

    def _on_feedback(self, surface_id: str, verdict: str) -> None:
        if verdict == "not_a_cat":
            surface = self.prefs.surface_by_id(surface_id)
            if surface is not None:
                surface.forget_last_observation(); self.prefs.save(); self.editor._rebuild_list()

    def _on_retention(self, days: int, save: bool) -> None:
        self.prefs.keep_thumbnails_days, self.prefs.save_thumbnails = days, save
        self.prefs.save()
        self.log.prune(days) if save else self.log.forget_all_thumbnails()
        self.activity.refresh()

    def _on_page(self, name: str) -> None:
        if name == "Settings": self._refresh_settings()
        elif name == "Audio": self._refresh_audio()
        elif name in ("Devices", "Rooms", "Camera", "Detection"): self._refresh_product_ui()

    def show_activity(self) -> None:
        self.activity.refresh(); self._show_auxiliary("Recent Activity", self.activity, 980, 620)

    def show_system_health(self) -> None:
        self.diagnostics.refresh(); self._show_auxiliary("System Health", self.diagnostics, 960, 610)

    def _show_auxiliary(self, title: str, widget: QWidget, width: int, height: int) -> None:
        dialog = QDialog(self); dialog.setWindowTitle(title); dialog.resize(width, height)
        widget.setParent(dialog)
        box = QVBoxLayout(dialog); box.setContentsMargins(0, 0, 0, 0); box.addWidget(widget)
        dialog.exec(); widget.setParent(self)

    def _refresh_state(self) -> None:
        state = self.engine.state.state()
        self.home.render_state(state); self.detection.render_state(state)
        text, tone = self._health_copy(state); self.shell.set_health(text, tone)
        colour = Q.SEVERITY_COLOUR[state.severity.value]
        self.tray.setIcon(_dot_icon(colour)); self.tray.setToolTip(f"Surface Guard — {state.headline}")
        self.act_toggle.setText("Turn protection off" if state.is_guarding else "Turn protection on")
        self.act_pause.setEnabled(state.is_guarding)
        if state.phase is Phase.OFF:
            self.home.live.set_message("Protection is off.")

    def _refresh_slow(self) -> None:
        plan = self.engine.build_plan() if self.engine.room_map else None
        presence = {s.id: self.engine.policy.state_of(s.id) for s in self.prefs.surfaces}
        self.home.render_coverage(self.prefs.surfaces, plan, presence)
        recent = self.log.recent(limit=3, fired_only=True)
        self.home.render_events([(
            f"<span style='color:#96a4ad'>{event.when}</span>    "
            f"<span style='color:#ff5c57'>●</span>  "
            f"<b>Cat detected on {html.escape(event.surface_name)}</b>    "
            "<span style='color:#96a4ad'>Audio response played</span>",
            str(self.log.thumb_dir / event.thumbnail) if event.thumbnail else None,
        ) for event in recent])
        self._refresh_product_ui()
        self._maybe_offer_review(time.time())
        if self.shell.current_name == "Settings": self._refresh_settings()

    def _refresh_audio(self) -> None:
        caps = self.engine.source.capabilities
        self.audio_screen.bind(self.prefs.surfaces, self.engine.player.sounds(),
                               camera_speaker=caps.has_speaker,
                               prefer_camera=self.prefs.prefer_camera_speaker,
                               output_available=self.engine.player.available())

    def _refresh_product_ui(self) -> None:
        caps = self.engine.source.capabilities
        configured = bool((self.prefs.camera or {}).get("kind"))
        self.engine.state.has_camera = configured
        online = self.engine.running and (self.engine.metrics.last_frame_at == 0.0 or
                                          time.monotonic() - self.engine.metrics.last_frame_at <= 6.0)
        camera_name = caps.name if configured else ""
        state = self.engine.state.state()
        count = len(self.prefs.surfaces)
        self.rooms.refresh_summary(count, camera_name, state.is_guarding)
        summary = "Nothing protected yet" if not count else ", ".join(s.name for s in self.prefs.surfaces[:2])
        if count > 2: summary += f" +{count - 2} more"
        sound = self.prefs.surfaces[0].deterrent.sound if count else "Short chirp"
        target = "Camera speaker" if self.prefs.prefer_camera_speaker and caps.has_speaker else "Plays through this Mac"
        self.home.set_context(self.prefs.room_name, summary, self.prefs.detection_sensitivity,
                              sound.title(), target)
        overlays = (self.prefs.show_protected_zones, self.prefs.show_detection_boxes,
                    self.prefs.show_surface_labels)
        self.home.live.set_overlays(*overlays)
        self.home.live.set_camera_name(caps.model or camera_name)
        self.camera_screen.live.set_room_name(self.prefs.room_name)
        self.camera_screen.set_camera(camera_name, caps.model, online=online and configured,
                                      has_speaker=caps.has_speaker, zones=overlays[0],
                                      boxes=overlays[1], labels=overlays[2])
        cooldown = self.prefs.surfaces[0].deterrent.cooldown_s if count else 20.0
        info = self.engine.detector.info
        detector_text = "Cat detection is ready." if info.available else \
            "Cat detection needs attention. " + (info.note or "No model is configured.")
        self.detection.set_values(self.prefs.detection_sensitivity, cooldown, detector_text)
        devices: list[DeviceViewState] = []
        if configured:
            devices.append(DeviceViewState(
                id=str((self.prefs.camera or {}).get("serial", "camera")),
                name=camera_name or "Camera", kind=DeviceKind.CAMERA, room_id="primary-room",
                online=online, detail=caps.model,
                capabilities=DeviceCapabilities(camera=True, speaker=caps.has_speaker)))
        devices.append(DeviceViewState(
            id="local-audio", name="This Mac", kind=DeviceKind.SPEAKER,
            room_id="primary-room", online=self.engine.player.available(),
            detail="Default audio output", capabilities=DeviceCapabilities(speaker=True)))
        room = RoomViewState(id="primary-room", name=self.prefs.room_name,
                             camera_ids=tuple(d.id for d in devices if d.kind is DeviceKind.CAMERA),
                             speaker_ids=("local-audio",),
                             surface_ids=tuple(s.id for s in self.prefs.surfaces),
                             protection="active" if state.is_guarding else "idle")
        product_state = ProductViewState(
            rooms=(room,), devices=tuple(devices), selected_room_id=room.id,
            selected_camera_id=room.camera_ids[0] if room.camera_ids else None,
            system_health=self._health_copy(state)[0])
        self.devices.render(product_state)
        self.shell.set_inventory(devices)

    @staticmethod
    def _health_copy(state) -> tuple[str, str]:
        if state.phase is Phase.PROBLEM: return "Protection Interrupted", "bad"
        if state.phase is Phase.NEEDS_SETUP: return "Setup Needed", "warning"
        if state.phase is Phase.PAUSED: return "Protection Paused", "warning"
        if state.phase is Phase.OFF: return "Protection Off", "neutral"
        return "All Systems Online", "good"

    def resizeEvent(self, event) -> None:  # noqa: N802
        self.shell.adapt_to_width(event.size().width())
        super().resizeEvent(event)

    def _hold_awake(self, on: bool) -> None:
        if on and self._caffeinate is None and sys.platform == "darwin":
            try:
                self._caffeinate = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())],
                                                     stdout=subprocess.DEVNULL,
                                                     stderr=subprocess.DEVNULL)
            except OSError:
                self._caffeinate = None
        elif not on and self._caffeinate is not None:
            self._caffeinate.terminate(); self._caffeinate = None

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._quitting:
            event.accept(); return
        event.ignore(); self.hide()
        self.tray.showMessage("Still watching", "Surface Guard keeps running in the menu bar.",
                              QSystemTrayIcon.MessageIcon.Information, 2500)

    def quit(self) -> None:
        self._quitting = True
        self.prefs.surfaces = self.editor.surfaces; self.prefs.save()
        self._hold_awake(False); self.updater.stop(); self.engine.stop()
        if self.supervisor is not None: self.supervisor.stop()
        self.log.close(); QApplication.quit()


def _dot_icon(colour) -> QIcon:
    """Menu-bar dot in the state's colour — the status is readable without opening the app."""
    pix = QPixmap(20, 20)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(colour)
    p.setPen(colour)
    p.drawEllipse(4, 4, 12, 12)
    p.end()
    return QIcon(pix)


# ------------------------------------------------------------- launch at login


def set_launch_at_login(enabled: bool, executable: str | None = None) -> bool:
    """Install or remove a macOS LaunchAgent. Returns the resulting state."""
    if sys.platform != "darwin":
        return False
    if not enabled:
        LAUNCH_AGENT.unlink(missing_ok=True)
        subprocess.run(["launchctl", "unload", str(LAUNCH_AGENT)],
                       capture_output=True, check=False)
        return False
    LAUNCH_AGENT.parent.mkdir(parents=True, exist_ok=True)
    argv = [executable or sys.executable, "-m", "surfaceguard.app", "--background"]
    LAUNCH_AGENT.write_bytes(plistlib.dumps({
        "Label": "com.surfaceguard.app",
        "ProgramArguments": argv,
        "RunAtLoad": True,
        "KeepAlive": False,
        "WorkingDirectory": str(Path(__file__).resolve().parent.parent),
    }))
    subprocess.run(["launchctl", "load", str(LAUNCH_AGENT)], capture_output=True, check=False)
    return True


# ---------------------------------------------------------------------- startup


def _make_source(prefs: Preferences) -> tuple[CameraSource | None, BridgeSupervisor | None]:
    """Rebuild the configured camera at startup, or nothing if setup never ran."""
    cfg = prefs.camera or {}
    kind = cfg.get("kind")
    if not kind:
        return None, None

    if kind == "eufy":
        from .bridge.client import BridgeClient
        from .camera.sources.eufy_bridge import EufyBridgeCamera

        account = EufyAccount(
            username=str(cfg.get("username", "")), country=str(cfg.get("country", "US"))
        )
        supervisor = BridgeSupervisor(account)
        status = supervisor.start(wait=True)
        if not status.listening:
            logger.error("camera service did not start: %s", status.fatal or status.message)
            return None, supervisor
        client = BridgeClient(supervisor.url)
        try:
            client.connect()
            state = client.connect_driver()
        except Exception as exc:
            logger.error("could not sign in to Eufy: %s", exc)
            return None, supervisor
        if state.phase.value != "connected":
            # A code or captcha cannot be answered without her; the UI will ask.
            logger.warning("Eufy sign-in needs attention: %s", state.message)
            return None, supervisor
        source = EufyBridgeCamera(
            client=client, serial=str(cfg.get("serial", "")),
            model=str(cfg.get("model", "")), name=str(cfg.get("name", "")),
            owns_client=True,
        )
        return source, supervisor

    from .ui.onboarding import SetupChoice
    try:
        return _build_source(SetupChoice(
            kind=kind, url=cfg.get("url", ""), serial=cfg.get("serial", "")
        )), None
    except Exception:
        return None, None


def _make_detector(prefs: Preferences, source: CameraSource) -> Detector:
    try:
        detector = load_detector(prefs.model_path, camera=source)
    except Exception as exc:
        logger.warning("no detection model (%s); using the test detector", exc)
        detector = SyntheticDetector(source) if hasattr(source, "ground_truth_boxes") else _Null()
    if hasattr(detector, "conf"):
        detector.conf = {"Calm": .52, "Balanced": .35, "Sensitive": .22}.get(
            prefs.detection_sensitivity, .35
        )
    return detector


class _Null(Detector):
    """Detects nothing, and says so. Keeps the app usable with no model installed."""

    def detect(self, image):  # noqa: D102
        return []

    @property
    def info(self):  # noqa: D102
        from .detection.cat_detector import DetectorInfo
        return DetectorInfo("none", "not configured", 0, available=False,
                            note="No detection model installed — nothing will be detected.")


def _selftest(app: QApplication, window: "MainWindow", engine: Engine, out: Path) -> int:
    """Render the window, save it, and report whether it actually drew anything."""
    result = {"code": 1}

    def capture() -> None:
        try:
            engine.start()
        except Exception as exc:
            logger.error("selftest: engine did not start: %s", exc)
        for _ in range(40):                      # ~2 s of event loop
            app.processEvents()
            time.sleep(0.05)
        pixmap = window.grab()
        out.parent.mkdir(parents=True, exist_ok=True)
        saved = pixmap.save(str(out))
        blank = pixmap.isNull() or pixmap.width() < 200 or pixmap.height() < 200
        state = engine.state.state()
        page_count = len(window.shell.pages)
        logger.info("selftest: %dx%d saved=%s state=%s pages=%d",
                    pixmap.width(), pixmap.height(), saved, state.phase.value,
                    page_count)
        print(f"selftest: window {pixmap.width()}x{pixmap.height()}, "
              f"pages={page_count}, state={state.phase.value}, saved={saved} -> {out}")
        result["code"] = 0 if (saved and not blank and page_count >= 7) else 1
        engine.stop()
        app.quit()

    QTimer.singleShot(400, capture)
    app.exec()
    return result["code"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="surfaceguard")
    ap.add_argument("--background", action="store_true", help="start hidden in the menu bar")
    ap.add_argument("--demo", action="store_true", help="run against the synthetic room")
    ap.add_argument("--model", default=None, help="path to a YOLOv8 ONNX model")
    ap.add_argument("--verbose", action="store_true", help="debug logging")
    ap.add_argument("--selftest", metavar="PNG", default=None,
                    help="start, screenshot the window to PNG, then exit (build check)")
    args = ap.parse_args(argv)

    setup_logging(verbose=args.verbose)
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Surface Guard")
    app.setQuitOnLastWindowClosed(False)
    app.setStyleSheet(Q.STYLESHEET)

    prefs = Preferences.load()
    if args.model:
        prefs.model_path = args.model
    if args.demo:
        prefs.camera = {"kind": "demo", "url": "", "serial": ""}

    # Remove the bundle a previous update left behind, before anything else runs.
    cleanup_previous()

    source, supervisor = _make_source(prefs)
    if source is None:
        from .camera.sources.synthetic import SyntheticCamera
        source = SyntheticCamera()

    log = ActivityLog()
    log.prune(prefs.keep_thumbnails_days)
    detector = _make_detector(prefs, source)
    room = load_room_map()
    state = StateStore()
    engine = Engine(source, detector, prefs, room_map=room,
                    player=Player(volume=prefs.master_volume), log=log, state=state)

    window = MainWindow(engine, prefs, log, updater=Updater(), supervisor=supervisor)
    if not args.background:
        window.show()

    if args.selftest:
        # Proves a *packaged* build really renders: a frozen app that starts but
        # draws nothing looks identical to a healthy one from the outside.
        return _selftest(app, window, engine, Path(args.selftest))

    if room is None:
        QTimer.singleShot(300, window.run_setup)
    else:
        try:
            engine.start()
        except SourceError as exc:
            logger.error("camera unavailable at startup: %s", exc)

    logger.info("%s", build_info.describe())
    logger.info("settings: %s", support_dir())
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
