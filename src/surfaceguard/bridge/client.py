"""The eufy-security-ws protocol, in one place.

Learned from probing a live bridge rather than from the docs, because the two
disagree in ways that matter:

* The bridge sends an unsolicited ``{"type":"version"}`` greeting before anything
  is asked of it.
* ``start_listening`` returns an *empty* device list until the driver has signed
  in. Reading devices before ``driver.connect`` succeeds always finds nothing.
* ``driver.connect`` answers ``success: true`` even when the sign-in fails. The
  real outcome arrives later as a ``driver/connectionError`` event, which carries a
  human-readable message — so the UI shows Eufy's own words rather than a code.
* Two-factor and captcha arrive as events mid-connect and must be answered before
  the driver will finish connecting.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

SCHEMA_VERSION = 21
DEFAULT_TIMEOUT = 15.0
CONNECT_TIMEOUT = 75.0


class BridgeError(RuntimeError):
    """A failure worth showing the user verbatim."""


class DriverPhase(Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    NEEDS_CAPTCHA = "needs_captcha"
    NEEDS_CODE = "needs_code"
    CONNECTED = "connected"
    FAILED = "failed"


@dataclass
class DriverState:
    phase: DriverPhase = DriverPhase.IDLE
    message: str = ""
    captcha_id: str = ""
    captcha_image: str = ""      # data: URI, as the bridge sends it
    error_code: int | None = None

    @property
    def needs_input(self) -> bool:
        return self.phase in (DriverPhase.NEEDS_CAPTCHA, DriverPhase.NEEDS_CODE)


@dataclass
class _Pending:
    event: threading.Event = field(default_factory=threading.Event)
    result: dict | None = None


class BridgeClient:
    """One WebSocket to the bridge, shared by sign-in and by the camera source."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.server_version = ""
        self.driver = DriverState()
        self._ws = None
        self._reader: threading.Thread | None = None
        self._stop = threading.Event()
        self._pending: dict[str, _Pending] = {}
        self._lock = threading.Lock()
        self._handlers: list[Callable[[dict], None]] = []
        self._driver_settled = threading.Event()

    # ------------------------------------------------------------- lifecycle

    def connect(self, timeout: float = 12.0) -> None:
        from websockets.sync.client import connect as ws_connect

        try:
            self._ws = ws_connect(self.url, open_timeout=timeout, max_size=None)
        except Exception as exc:
            raise BridgeError(
                f"Could not reach the camera service at {self.url}. ({exc})"
            ) from exc
        self._stop.clear()
        self._reader = threading.Thread(target=self._read_loop, name="sg-bridge-ws", daemon=True)
        self._reader.start()
        self.send_wait("set_api_schema", schemaVersion=SCHEMA_VERSION)

    def close(self) -> None:
        self._stop.set()
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        if self._reader:
            self._reader.join(timeout=2.0)
            self._reader = None

    @property
    def connected(self) -> bool:
        return self._ws is not None and not self._stop.is_set()

    def add_handler(self, handler: Callable[[dict], None]) -> None:
        self._handlers.append(handler)

    # -------------------------------------------------------------- messaging

    def send(self, command: str, **payload) -> str:
        if self._ws is None:
            raise BridgeError("Not connected to the camera service")
        message_id = uuid.uuid4().hex[:12]
        self._ws.send(json.dumps({"messageId": message_id, "command": command, **payload}))
        return message_id

    def send_wait(self, command: str, timeout: float = DEFAULT_TIMEOUT, **payload) -> dict:
        pending = _Pending()
        with self._lock:
            message_id = self.send(command, **payload)
            self._pending[message_id] = pending
        if not pending.event.wait(timeout):
            with self._lock:
                self._pending.pop(message_id, None)
            raise BridgeError(f"The camera service did not answer '{command}' in {timeout:.0f}s")
        result = pending.result or {}
        if not result.get("success", True):
            raise BridgeError(_describe_error(command, result))
        return result.get("result", {}) or {}

    # ------------------------------------------------------------ driver flow

    def connect_driver(self, timeout: float = CONNECT_TIMEOUT) -> DriverState:
        """Sign the bridge in to the Eufy account and wait for a real outcome.

        Returns when the driver connects, fails, or asks for a code or captcha.
        """
        self.driver = DriverState(phase=DriverPhase.CONNECTING, message="Signing in…")
        self._driver_settled.clear()
        self.send_wait("driver.connect", timeout=DEFAULT_TIMEOUT)
        # success:true here only means "the request was accepted"; the outcome is
        # an event, so wait for that rather than believing the acknowledgement.
        self._driver_settled.wait(timeout=timeout)
        if self.driver.phase is DriverPhase.CONNECTING:
            self.driver.phase = DriverPhase.FAILED
            self.driver.message = (
                "Signing in to Eufy timed out. Check the internet connection and try again."
            )
        return self.driver

    def submit_captcha(self, text: str, timeout: float = CONNECT_TIMEOUT) -> DriverState:
        captcha_id = self.driver.captcha_id
        self.driver = DriverState(phase=DriverPhase.CONNECTING, message="Checking…")
        self._driver_settled.clear()
        self.send_wait("driver.set_captcha", captchaId=captcha_id, captcha=text)
        self._driver_settled.wait(timeout=timeout)
        return self.driver

    def submit_verify_code(self, code: str, timeout: float = CONNECT_TIMEOUT) -> DriverState:
        self.driver = DriverState(phase=DriverPhase.CONNECTING, message="Checking…")
        self._driver_settled.clear()
        self.send_wait("driver.set_verify_code", verifyCode=code)
        self._driver_settled.wait(timeout=timeout)
        return self.driver

    def devices(self) -> list[dict]:
        """Cameras the account can see. Empty until the driver has connected."""
        state = self.send_wait("start_listening", timeout=20.0)
        return list((state.get("state") or {}).get("devices") or [])

    def device_properties(self, serial: str) -> dict:
        result = self.send_wait("device.get_properties", serialNumber=serial)
        return result.get("properties", {}) or {}

    # ------------------------------------------------------------------ events

    def _read_loop(self) -> None:
        while not self._stop.is_set() and self._ws is not None:
            try:
                raw = self._ws.recv(timeout=1.0)
            except TimeoutError:
                continue
            except Exception:
                break
            try:
                message = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            self._dispatch(message)
        if not self._stop.is_set():
            # The bridge went away underneath us; unblock anyone waiting.
            self.driver.phase = DriverPhase.FAILED
            self.driver.message = "The camera service stopped unexpectedly."
            self._driver_settled.set()

    def _dispatch(self, message: dict) -> None:
        kind = message.get("type")

        if kind == "version":
            self.server_version = str(message.get("serverVersion", ""))
            return

        if kind == "result":
            with self._lock:
                pending = self._pending.pop(message.get("messageId", ""), None)
            if pending is not None:
                pending.result = message
                pending.event.set()
            return

        if kind != "event":
            return

        event = message.get("event", {})
        if event.get("source") == "driver":
            self._handle_driver_event(event)
        for handler in list(self._handlers):
            try:
                handler(event)
            except Exception:
                pass  # a bad handler must not take the socket down

    def _handle_driver_event(self, event: dict) -> None:
        name = event.get("event")
        if name == "connected":
            self.driver = DriverState(phase=DriverPhase.CONNECTED, message="Signed in to Eufy")
            self._driver_settled.set()
        elif name == "connectionError":
            context = (event.get("error") or {}).get("context") or {}
            message = str(context.get("message") or "").strip()
            code = context.get("code")
            if not message and not code:
                return  # the bridge emits a trailing empty error; ignore it
            self.driver = DriverState(
                phase=DriverPhase.FAILED,
                message=message or f"Eufy refused the sign-in (code {code}).",
                error_code=int(code) if isinstance(code, int) else None,
            )
            self._driver_settled.set()
        elif name == "captcha request":
            self.driver = DriverState(
                phase=DriverPhase.NEEDS_CAPTCHA,
                message="Eufy wants you to read a short code from a picture.",
                captcha_id=str(event.get("captchaId") or ""),
                captcha_image=str(event.get("captcha") or ""),
            )
            self._driver_settled.set()
        elif name == "verify code":
            self.driver = DriverState(
                phase=DriverPhase.NEEDS_CODE,
                message="Eufy sent a verification code to your email.",
            )
            self._driver_settled.set()
        elif name == "disconnected":
            if self.driver.phase is DriverPhase.CONNECTED:
                self.driver = DriverState(
                    phase=DriverPhase.FAILED, message="Eufy signed this app out."
                )


def _describe_error(command: str, result: dict) -> str:
    code = result.get("errorCode") or result.get("message") or result
    hints = {
        "device_not_found": "That camera is not on this Eufy account any more.",
        "schema_incompatible": "This version of the camera service is too old for the app.",
        "driver_not_connected": "Not signed in to Eufy yet.",
    }
    return hints.get(str(code), f"The camera service refused '{command}' ({code}).")


def looks_like_camera(device: dict) -> bool:
    """Cameras only: the account may also hold locks, sensors and doorbells."""
    model = str(device.get("model") or "").upper()
    name = str(device.get("name") or "")
    if model.startswith("T8") and not model.startswith("T85"):   # T85xx are sensors
        return True
    return "cam" in name.lower()


def is_pan_tilt(model: str) -> bool:
    """T8417 is the Indoor Cam E30; the rest are its pan/tilt siblings."""
    return any(str(model).upper().startswith(p) for p in
               ("T8410", "T8414", "T8416", "T8417", "T8441"))
