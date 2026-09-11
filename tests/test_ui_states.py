from surfaceguard.state import DeviceKind, StateStore
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
