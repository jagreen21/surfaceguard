import numpy as np

from surfaceguard.state import DeviceKind, DeviceViewState, StateStore
from surfaceguard.ui.mock_states import configurations


def test_required_hardware_configurations_cover_sparse_and_dense_states():
    states = configurations()
    assert set(states) == set("ABCDEFGH")
    assert len(states["A"].rooms) == 1
    assert len(states["E"].rooms) == 3
    assert len(states["F"].rooms) == 5
    assert sum(d.kind is DeviceKind.CAMERA for d in states["F"].devices) == 8
    assert any(not d.online for d in states["F"].devices)


def test_capabilities_do_not_invent_camera_ui():
    state = configurations()["C"]
    main = next(d for d in state.devices if d.kind is DeviceKind.MAIN_UNIT)
    assert not main.capabilities.camera
    assert not any(d.kind is DeviceKind.CAMERA for d in state.devices)


def test_attention_and_alert_states_are_representable():
    states = configurations()
    assert states["G"].system_health == "Protection Interrupted"
    assert any(d.battery_percent == 10 for d in states["G"].devices)
    assert states["H"].rooms[0].protection == "alerting"


def test_setup_copy_distinguishes_camera_connection_from_room_scan():
    assert "Connect your camera" in StateStore().state().detail
    assert "Scan this room" in StateStore(has_camera=True).state().detail


def test_room_map_can_be_zoomed_and_reset(qt_app):
    from surfaceguard.ui.surface_editor import MapCanvas

    canvas = MapCanvas()
    canvas.resize(800, 600)
    canvas.set_map(np.zeros((900, 1600, 3), np.uint8))
    canvas.show()
    qt_app.processEvents()
    assert not canvas.grab().isNull()
    fitted, _ = canvas._fit()
    canvas.zoom_in()
    zoomed, _ = canvas._fit()
    assert zoomed > fitted
    canvas.reset_view()
    reset, _ = canvas._fit()
    assert reset == fitted


def test_home_exposes_pause_test_and_coverage_controls(qt_app):
    from surfaceguard.ui.home import HomeScreen

    home = HomeScreen()
    for widget in (home.pause30, home.pause_tomorrow, home.test_btn, home.coverage):
        assert home.isAncestorOf(widget)

    home.adapt_to_width(600)
    assert home.quick.indexOf(home.protection_card) >= 0
    assert home.quick.getItemPosition(home.quick.indexOf(home.audio_card))[0] == 2


def test_audio_exposes_response_delay_without_duplicate_cooldown(qt_app):
    from surfaceguard.ui.audio import AudioScreen

    audio = AudioScreen()
    assert audio.isAncestorOf(audio.delay)
    assert not hasattr(audio, "cooldown")
    assert not hasattr(audio, "chime_choice")


def test_shell_has_activity_and_dynamic_device_inventory(qt_app):
    from PySide6.QtWidgets import QWidget

    from surfaceguard.ui.shell import AppShell, DESTINATIONS

    pages = {name: QWidget() for _symbol, name in DESTINATIONS}
    pages["System Health"] = QWidget()
    shell = AppShell(pages)
    devices = [
        DeviceViewState(str(i), f"Device {i}", DeviceKind.SPEAKER, online=True)
        for i in range(7)
    ]
    shell.set_inventory(devices)
    assert "Activity" in shell.buttons
    assert len(shell.device_labels) == 7
    shell.show_page("System Health")
    assert shell.current_name == "System Health"


def test_surface_vertex_edits_can_be_undone(qt_app):
    from surfaceguard.geometry.surface import Surface
    from surfaceguard.ui.surface_editor import MapCanvas

    original = np.asarray([(10.0, 10.0), (80.0, 10.0), (80.0, 80.0)])
    surface = Surface("Counter", original.copy())
    canvas = MapCanvas()
    canvas.set_surfaces([surface])
    canvas._undo_stack.append((surface, original.copy()))
    surface.polygon[0] = (30.0, 40.0)
    canvas.undo_last_edit()
    assert np.array_equal(surface.polygon, original)
