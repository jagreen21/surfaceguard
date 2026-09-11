# Surface Guard UI audit

This audit compares the implemented desktop interface with the approved visualizer
and the current backend. The visualizer is the visual source of truth; the camera,
detection, mapping, audio, health, history, and update code remain the functional
source of truth.

## Implemented

- Adaptive seven-destination sidebar: Home, Rooms, Detection, Audio, Camera,
  Devices, and Settings. System Health is contextual rather than a primary page.
- Camera-first Home with a single prominent live view, honest protection status,
  protected-zone and detection overlays, quick controls, surfaces, and recent
  activity.
- Room-level polygon editor with approachable handles, translucent protected
  zones, common-name suggestions, Finish/Cancel actions, and no raw coordinates.
- Plain-language detection controls with Calm, Balanced, and Sensitive presets;
  detector internals are behind Advanced.
- Per-surface audio response, volume, delay, cooldown, destination, test, and WAV
  import. Camera-speaker controls only appear when the source reports that ability.
- Camera page with live view, snapshot, supported speaker test, and optional
  overlays.
- Capability-driven device cards, persistent offline state, and contextual battery
  and charging presentation when those values exist.
- Activity and System Health are focused secondary dialogs. Detection gate details
  are hidden until Advanced Details is opened.
- Wide and compact window renders were compared with the approved visualizer.
  The compact shell preserves the camera as the dominant object.
- Development states A–H cover sparse and dense rooms/devices, camera-less main
  hardware, multiple cameras/speakers, mixed connectivity, low battery, and an
  active protected-surface alert.

## Preserved from the existing program

- Eufy and RTSP setup, room scan/registration, room-coordinate surfaces, detection
  gates and policy, audible deterrents, heartbeat-derived protection state,
  activity feedback, weekly review, menu-bar operation, launch at login,
  diagnostics, and signed updates remain connected to the new shell.

## Intentionally not presented as working

- Voice phrases are visibly unavailable because the audio backend only supports
  generated cues and imported WAV files.
- Satellite pairing, stereo grouping, charging docks, and main-unit controls are
  absent because the current backend exposes none of those devices.
- The backend currently owns one live camera source, one room map, and one surface
  collection. The UI state contract and device inventory are collection-driven,
  but simultaneous multi-room and multi-camera live grids cannot be made real
  until the engine exposes multiple streams/maps. The current Eufy E30 workflow
  therefore uses the correct one-camera hero layout and does not show fake feeds.

## Current-product fit

For the intended installation—a Mac laptop, one Eufy Indoor Cam E30, laptop or
camera audio, and user-drawn protected surfaces—the normal workflow is complete:
connect the camera, scan the room, draw or adjust surfaces, choose detection and
audio behavior, turn protection on, and review activity without using a terminal.
