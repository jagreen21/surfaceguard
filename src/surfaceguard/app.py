"""Application lifecycle: window, tray, engine wiring, and staying awake.

Engine callbacks arrive on the engine thread, so everything crosses into Qt through
_Bridge's signals rather than touching widgets directly. The app keeps running from
the menu bar when its window is closed, and holds a power assertion while armed —
a sleeping laptop is the most common way "forget about it" quietly stops being true.
"""

from __future__ import annotations

import argparse
import os
import plistlib
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
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
from .camera.sources.base import CameraSource, SourceError
from .detection.cat_detector import Detector, SyntheticDetector, load_detector
from .engine import Engine, FrameResult
from .health.heartbeat import Report
from .state import Phase, StateStore
from .storage.activity_log import ActivityLog
from .storage.preferences import Preferences, support_dir
from .storage.room_map import load_room_map, save_room_map
from .ui import qtutil as Q
from .ui.activity import ActivityScreen
from .ui.diagnostics import DiagnosticsScreen
from .ui.home import HomeScreen
from .ui.onboarding import OnboardingDialog, _build_source
from .ui.surface_editor import SurfaceEditor

LAUNCH_AGENT = Path.home() / "Library" / "LaunchAgents" / "com.surfaceguard.app.plist"
UI_REFRESH_MS = 120
SLOW_REFRESH_MS = 1500


class _Bridge(QObject):
    """Marshals engine-thread callbacks onto the GUI thread."""

    frame = Signal(object)
    trigger = Signal(object, object)
    heartbeat = Signal(object)


class MainWindow(QWidget):
    def __init__(self, engine: Engine, prefs: Preferences, log: ActivityLog) -> None:
        super().__init__()
        self.engine = engine
        self.prefs = prefs
        self.log = log
        self._quitting = False
        self._caffeinate: subprocess.Popen | None = None

        self.setWindowTitle("Surface Guard")
        self.resize(1160, 740)

        self.home = HomeScreen()
        self.editor = SurfaceEditor()
        self.activity = ActivityScreen(log)
        self.diagnostics = DiagnosticsScreen(engine)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.home, "Home")
        self.tabs.addTab(self.editor, "Surfaces")
        self.tabs.addTab(self.activity, "Activity")
        self.tabs.addTab(self.diagnostics, "Diagnostics")
        self.tabs.currentChanged.connect(self._on_tab)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self.tabs)

        self.home.arm_requested.connect(self.arm)
        self.home.off_requested.connect(self.turn_off)
        self.home.pause_requested.connect(self.pause_for)
        self.home.test_sound_requested.connect(self.test_sound)
        self.editor.surfaces_changed.connect(self._on_surfaces_changed)
        self.activity.feedback_given.connect(self._on_feedback)
        self.activity.retention_changed.connect(self._on_retention)
        self.diagnostics.rescan_requested.connect(self.rescan_room)

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

    # ------------------------------------------------------------------- setup

    def run_setup(self) -> bool:
        dialog = OnboardingDialog(self, allow_demo=True)
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
        if self.tabs.currentWidget() is self.diagnostics:
            self.diagnostics.refresh()

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
        self.engine.stop()
        self.log.close()
        QApplication.quit()


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


def _make_source(prefs: Preferences) -> CameraSource | None:
    from .ui.onboarding import SetupChoice
    cfg = prefs.camera or {}
    if not cfg.get("kind"):
        return None
    try:
        return _build_source(SetupChoice(
            kind=cfg["kind"], url=cfg.get("url", ""), serial=cfg.get("serial", "")
        ))
    except Exception:
        return None


def _make_detector(prefs: Preferences, source: CameraSource) -> Detector:
    try:
        return load_detector(prefs.model_path, camera=source)
    except Exception as exc:
        print(f"[app] no detection model ({exc}); using the test detector")
        return SyntheticDetector(source) if hasattr(source, "ground_truth_boxes") else _Null()


class _Null(Detector):
    """Detects nothing, and says so. Keeps the app usable with no model installed."""

    def detect(self, image):  # noqa: D102
        return []

    @property
    def info(self):  # noqa: D102
        from .detection.cat_detector import DetectorInfo
        return DetectorInfo("none", "not configured", 0, available=False,
                            note="No detection model installed — nothing will be detected.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="surfaceguard")
    ap.add_argument("--background", action="store_true", help="start hidden in the menu bar")
    ap.add_argument("--demo", action="store_true", help="run against the synthetic room")
    ap.add_argument("--model", default=None, help="path to a YOLOv8 ONNX model")
    args = ap.parse_args(argv)

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Surface Guard")
    app.setQuitOnLastWindowClosed(False)
    app.setStyleSheet(Q.STYLESHEET)

    prefs = Preferences.load()
    if args.model:
        prefs.model_path = args.model
    if args.demo:
        prefs.camera = {"kind": "demo", "url": "", "serial": ""}

    source = _make_source(prefs)
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

    window = MainWindow(engine, prefs, log)
    if not args.background:
        window.show()
    if room is None:
        QTimer.singleShot(300, window.run_setup)
    else:
        try:
            engine.start()
        except SourceError as exc:
            print(f"[app] camera unavailable at startup: {exc}")

    print(f"[app] settings: {support_dir()}")
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
