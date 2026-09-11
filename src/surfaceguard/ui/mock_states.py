"""Development-only hardware configurations used to exercise adaptive UI.

These are presentation states, not simulated integrations.  They let designers
render absent, singular, crowded, offline, battery, and alerting layouts without
pretending unsupported hardware can perform real actions.
"""

from __future__ import annotations

from ..state import (
    DeviceCapabilities,
    DeviceKind,
    DeviceViewState,
    ProductViewState,
    RoomViewState,
)


def _device(identifier: str, name: str, kind: DeviceKind, room: str,
            *, online: bool = True, camera: bool = False, speaker: bool = False,
            battery: int | None = None, detail: str = "") -> DeviceViewState:
    return DeviceViewState(
        id=identifier, name=name, kind=kind, room_id=room, online=online,
        battery_percent=battery, detail=detail,
        capabilities=DeviceCapabilities(
            camera=camera, microphone=camera, speaker=speaker,
            battery=battery is not None,
        ),
    )


def _state(room_names: tuple[str, ...], cameras_per_room: tuple[int, ...],
           speakers_per_room: tuple[int, ...], *, main_camera: bool | None = None,
           offline: frozenset[str] = frozenset(), low_battery: str | None = None,
           alert_room: int | None = None) -> ProductViewState:
    devices: list[DeviceViewState] = []
    rooms: list[RoomViewState] = []
    if main_camera is not None:
        devices.append(_device(
            "main", "Surface Guard Main Unit", DeviceKind.MAIN_UNIT, "room-0",
            camera=main_camera, speaker=True, detail="Main Unit",
        ))
    for room_index, name in enumerate(room_names):
        room_id = f"room-{room_index}"
        camera_ids: list[str] = []
        speaker_ids: list[str] = []
        for index in range(cameras_per_room[room_index]):
            identifier = f"cam-{room_index}-{index}"
            camera_ids.append(identifier)
            devices.append(_device(
                identifier, f"{name} Camera {index + 1}", DeviceKind.CAMERA, room_id,
                online=identifier not in offline, camera=True, speaker=True,
                detail="Eufy Indoor Cam E30" if identifier == "cam-0-0" else "Camera",
            ))
        for index in range(speakers_per_room[room_index]):
            identifier = f"speaker-{room_index}-{index}"
            speaker_ids.append(identifier)
            devices.append(_device(
                identifier, f"{name} Speaker {index + 1}", DeviceKind.SPEAKER, room_id,
                online=identifier not in offline, speaker=True,
                battery=10 if identifier == low_battery else 76,
                detail="Satellite speaker",
            ))
        rooms.append(RoomViewState(
            id=room_id, name=name, camera_ids=tuple(camera_ids),
            speaker_ids=tuple(speaker_ids), surface_ids=(f"surface-{room_index}",),
            protection="alerting" if alert_room == room_index else "active",
        ))
    return ProductViewState(
        rooms=tuple(rooms), devices=tuple(devices),
        selected_room_id=rooms[0].id if rooms else None,
        selected_camera_id=rooms[0].camera_ids[0] if rooms and rooms[0].camera_ids else None,
        system_health="Protection Interrupted" if offline else "All Systems Online",
    )


def configurations() -> dict[str, ProductViewState]:
    """The eight configurations required by the UI specification."""
    return {
        "A": _state(("Kitchen",), (1,), (0,)),
        "B": _state(("Kitchen",), (0,), (0,), main_camera=True),
        "C": _state(("Kitchen",), (0,), (2,), main_camera=False),
        "D": _state(("Kitchen",), (2,), (2,), main_camera=False),
        "E": _state(("Kitchen", "Living Room", "Office"), (2, 1, 1), (1, 1, 0),
                    main_camera=False),
        "F": _state(("Kitchen", "Living Room", "Dining Room", "Office", "Bedroom"),
                    (2, 2, 2, 1, 1), (2, 1, 1, 1, 1), main_camera=False,
                    offline=frozenset({"cam-1-1", "speaker-3-0"})),
        "G": _state(("Kitchen",), (1,), (2,), main_camera=False,
                    offline=frozenset({"cam-0-0", "speaker-0-1"}),
                    low_battery="speaker-0-0"),
        "H": _state(("Kitchen",), (1,), (1,), alert_room=0),
    }

