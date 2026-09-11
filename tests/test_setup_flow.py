"""First-run setup: one flow, from "which camera" to "draw a surface".

This file exists because of a specific escape: a patch left ``app.py`` calling
``run_setup(existing_source=...)`` against a dialog that did not accept it, and the
whole suite still passed, because nothing touched this path. The first thing a new
user does was the least tested thing in the app.

The route tests matter most. Setup used to ask how to connect a camera, take a
sign-in, then ask how to connect a camera again; the sign-in is now part of the
same flow, so that is impossible rather than patched — and these assert it.

Nothing here reaches the Keychain, a bridge, or a camera.
"""

import pytest

from surfaceguard.bridge.client import DriverPhase, DriverState
from surfaceguard.bridge.credentials import EufyAccount
from surfaceguard.camera.sources.synthetic import SyntheticCamera
from surfaceguard.ui.connect import EufySignInDialog
from surfaceguard.ui.onboarding import OnboardingDialog, Page


@pytest.fixture
def no_keychain(monkeypatch):
    """A test must never write a password to the real Keychain."""
    saved = {}
    monkeypatch.setattr(EufyAccount, "save_password",
                        lambda self, password: saved.__setitem__("password", password))
    monkeypatch.setattr(EufyAccount, "password",
                        property(lambda self: saved.get("password")))
    return saved


@pytest.fixture
def setup(qt_app, no_keychain):
    return OnboardingDialog(allow_demo=True)


@pytest.fixture
def camera():
    cam = SyntheticCamera()
    cam.start()
    yield cam
    cam.stop()


class FakeClient:
    connected = True

    def __init__(self, devices=None):
        self._devices = devices if devices is not None else [
            {"name": "Kitchen", "model": "T8417", "serialNumber": "T8417P1"},
            {"name": "Living room", "model": "T8416", "serialNumber": "T8416P2"},
        ]

    def devices(self):
        return self._devices

    def close(self):
        pass


# ---------------------------------------------------------------- the route


def test_a_eufy_setup_signs_in_inside_the_same_flow(setup):
    """The whole point: no separate sign-in dialog before this one."""
    setup.kind.setCurrentIndex(setup.kind.findData("eufy"))
    route = setup.route()
    assert Page.CREDENTIALS in route
    assert route.index(Page.KIND) < route.index(Page.CREDENTIALS) < route.index(Page.SCAN)


def test_a_non_eufy_setup_skips_the_sign_in_steps(setup):
    setup.kind.setCurrentIndex(setup.kind.findData("demo"))
    route = setup.route()
    assert Page.CREDENTIALS not in route and Page.PICKER not in route
    assert route == [Page.KIND, Page.CONFIRM, Page.SCAN, Page.SOUND, Page.DONE]


def test_the_step_counter_matches_the_route_it_is_on(setup):
    setup.kind.setCurrentIndex(setup.kind.findData("demo"))
    setup._go(Page.KIND)
    assert setup.step.text() == f"STEP 1 OF {len(setup.route())}"
    setup.kind.setCurrentIndex(setup.kind.findData("eufy"))
    setup._go(Page.KIND)
    assert setup.step.text() == f"STEP 1 OF {len(setup.route())}"
    assert "OF 7" in setup.step.text(), "a Eufy setup has more steps than a demo"


def test_a_two_factor_prompt_adds_a_step_only_when_eufy_asks(setup):
    setup.kind.setCurrentIndex(setup.kind.findData("eufy"))
    before = len(setup.route())
    setup._on_signin_settled(DriverState(phase=DriverPhase.NEEDS_CODE, message="code"))
    assert len(setup.route()) == before + 1
    assert Page.CHALLENGE in setup.route()


def test_back_cannot_leave_the_first_step_of_the_route(setup):
    setup._go(Page.KIND)
    for _ in range(3):
        setup._back()
    assert setup.page is Page.KIND
    assert not setup.back_btn.isEnabled()


# ------------------------------------------------------------------ sign-in


def test_empty_credentials_are_refused_before_anything_starts(setup):
    setup._go(Page.CREDENTIALS)
    setup.email.setText("")
    setup.password.setText("")
    setup._advance()
    assert setup.page is Page.CREDENTIALS
    assert "email" in setup.status.text().lower()


def test_the_password_field_is_masked(setup):
    from PySide6.QtWidgets import QLineEdit

    assert setup.password.echoMode() == QLineEdit.EchoMode.Password


def test_a_code_prompt_shows_no_picture(setup):
    setup._on_signin_settled(DriverState(phase=DriverPhase.NEEDS_CODE, message="code"))
    assert setup.page is Page.CHALLENGE
    assert setup.captcha_image.isHidden(), "a code prompt must not show a captcha"


def test_a_captcha_prompt_shows_the_picture(setup):
    setup._on_signin_settled(DriverState(
        phase=DriverPhase.NEEDS_CAPTCHA, message="read it", captcha_id="c1",
        captcha_image="data:image/png;base64,bm90YW5pbWFnZQ=="))
    assert setup.page is Page.CHALLENGE
    assert not setup.captcha_image.isHidden()


def test_a_failed_sign_in_returns_to_the_form_with_eufys_own_words(setup):
    setup._on_signin_settled(DriverState(
        phase=DriverPhase.FAILED, message="Email address or password incorrect."))
    assert setup.page is Page.CREDENTIALS
    assert "incorrect" in setup.status.text()


def test_cameras_are_listed_and_one_is_preselected(setup):
    setup.client = FakeClient()
    setup._load_devices()
    assert setup.page is Page.PICKER
    assert setup.devices.count() == 2
    assert setup.devices.currentRow() == 0, "she should not have to click before continuing"


def test_non_cameras_on_the_account_are_filtered_out(setup):
    setup.client = FakeClient([
        {"name": "Kitchen", "model": "T8417", "serialNumber": "T8417P1"},
        {"name": "Front door lock", "model": "T8520", "serialNumber": "T8520P9"},
    ])
    setup._load_devices()
    assert setup.devices.count() == 1


def test_an_account_with_no_cameras_says_so_plainly(setup):
    setup.client = FakeClient([])
    setup._load_devices()
    assert setup.page is not Page.PICKER
    assert "no cameras" in setup.status.text().lower()


# -------------------------------------------------------- the standalone form


def test_settings_sign_in_stops_after_choosing_a_camera(qt_app, no_keychain):
    dialog = EufySignInDialog(EufyAccount(username="her@example.com"))
    assert dialog.sign_in_only
    assert dialog.route() == [Page.CREDENTIALS, Page.PICKER]
    assert dialog.page is Page.CREDENTIALS, "it should open on the form, not on 'which kind'"


def test_settings_sign_in_exposes_what_the_app_reads(qt_app, no_keychain):
    """app.py reads .chosen, .client, .supervisor and .account off this."""
    dialog = EufySignInDialog(EufyAccount(username="a@b.c"))
    for attribute in ("chosen", "client", "supervisor", "account"):
        assert hasattr(dialog, attribute), f"app.py reads .{attribute}"
    dialog.client = FakeClient()
    dialog._load_devices()
    dialog._choose_device()
    assert dialog.chosen["serialNumber"] == "T8417P1"


# ----------------------------------------------------------------- room scan


def test_the_scan_accepts_a_camera_that_is_already_connected(qt_app, camera):
    dialog = OnboardingDialog(existing_source=camera)
    assert dialog.adopted and dialog.source is camera
    assert dialog.page is Page.CONFIRM
    assert Page.KIND not in dialog.route(), "still offering to connect a connected camera"


def test_cancelling_does_not_stop_a_camera_the_app_is_using(qt_app, camera):
    dialog = OnboardingDialog(existing_source=camera)
    dialog.reject()
    assert dialog.source is None, "the dialog kept a camera it does not own"
    assert camera._running, "cancelling stopped the app's camera"


def test_the_demo_option_is_offered_or_withheld_as_asked(qt_app):
    assert OnboardingDialog(allow_demo=True).kind.count() == 3
    assert OnboardingDialog(allow_demo=False).kind.count() == 2


def test_the_address_field_appears_only_when_an_address_is_needed(qt_app):
    dialog = OnboardingDialog(allow_demo=True)
    dialog.kind.setCurrentIndex(dialog.kind.findData("rtsp"))
    assert dialog.url.text().startswith("rtsp://")
    assert not dialog.url.isHidden()
    dialog.kind.setCurrentIndex(dialog.kind.findData("eufy"))
    assert dialog.url.isHidden(), "a Eufy camera is found by signing in, not by address"
    dialog.kind.setCurrentIndex(dialog.kind.findData("demo"))
    assert dialog.url.isHidden()


# -------------------------------------------------------- the wiring itself


def test_app_calls_run_setup_the_way_the_dialog_accepts_it():
    """The exact break that got through: a call site and a signature that disagree."""
    import inspect

    from surfaceguard import app

    if "existing_source=" not in inspect.getsource(app):
        pytest.skip("this build does not hand the camera to the room scan")
    assert "existing_source" in inspect.signature(OnboardingDialog.__init__).parameters


def test_every_heading_and_label_exists_for_every_page(qt_app):
    """A page with no heading is a screen that reads as broken."""
    dialog = OnboardingDialog(allow_demo=True)
    for page in Page:
        dialog._go(page)
        assert dialog.heading.text(), f"{page.name} has no heading"
        assert dialog.subtitle.text(), f"{page.name} has no subtitle"
        assert dialog.next_btn.text(), f"{page.name} has no button label"


def test_sign_in_does_not_wait_on_the_meaningless_acknowledgement(monkeypatch):
    """driver.connect answers success:true even when the sign-in fails, so waiting
    on that result only added a 15s deadline to a login that legitimately takes
    longer — abandoning a sign-in that was still working. The outcome must come
    from the event, and only the event."""
    import inspect

    from surfaceguard.bridge.client import BridgeClient

    source = inspect.getsource(BridgeClient.connect_driver)
    assert 'send_wait("driver.connect"' not in source
    assert 'self.send("driver.connect")' in source


def test_a_captcha_challenge_reaches_the_user(qt_app, no_keychain):
    """The failure that cost a whole evening: Eufy demanded a captcha, the bridge
    forwarded it, and it had to show up as a step rather than a timeout."""
    from surfaceguard.bridge.client import BridgeClient

    dialog = OnboardingDialog(allow_demo=True)
    client = BridgeClient("ws://127.0.0.1:1")
    client.add_handler(lambda e: None)
    # Exactly the shape eufy-security-ws sends.
    client._dispatch({
        "type": "event",
        "event": {
            "source": "driver",
            "event": "captcha request",
            "captchaId": "abc123",
            "captcha": "data:image/png;base64,aGVsbG8=",
        },
    })
    assert client.driver.phase.value == "needs_captcha"
    assert client.driver.captcha_id == "abc123"

    dialog._on_signin_settled(client.driver)
    assert dialog.page is Page.CHALLENGE
    assert not dialog.captcha_image.isHidden()
    assert Page.CHALLENGE in dialog.route(), "the captcha must count as a step"


def test_the_client_subscribes_to_events_before_it_signs_in():
    """The bug that cost an evening. The bridge gates every event on a per-client
    receiveEvents flag that only start_listening sets, so calling driver.connect
    first means the captcha request, the connected event and the connection error
    are all delivered to nobody — and sign-in can only ever time out."""
    import inspect

    from surfaceguard.bridge.client import BridgeClient

    connect = inspect.getsource(BridgeClient.connect)
    assert "_start_listening" in connect, "connect() must subscribe to events"
    assert connect.index("set_api_schema") < connect.index("_start_listening")

    # And nothing may send driver.connect before that subscription exists.
    driver = inspect.getsource(BridgeClient.connect_driver)
    assert "start_listening" not in driver


def test_a_successful_sign_in_is_remembered_before_setup_finishes(qt_app, no_keychain,
                                                                  tmp_path, monkeypatch):
    """The password lives in the Keychain and survives updates, but without the
    email and region stored beside it an update before choosing a camera meant
    typing everything again."""
    from surfaceguard.bridge.client import DriverPhase, DriverState
    from surfaceguard.storage import preferences as prefs_mod

    monkeypatch.setattr(prefs_mod, "support_dir", lambda: tmp_path)
    dialog = OnboardingDialog(allow_demo=True)
    dialog.account.username = "her@example.com"
    dialog.account.country = "GB"
    dialog._remember_account()

    saved = prefs_mod.Preferences.load().camera
    assert saved["kind"] == "eufy"
    assert saved["username"] == "her@example.com"
    assert saved["country"] == "GB"


class _LiveClient:
    """A bridge client that is already connected and signed in."""

    connected = True

    def __init__(self):
        from surfaceguard.bridge.client import DriverPhase, DriverState

        self.driver = DriverState(phase=DriverPhase.CONNECTED, message="Signed in")
        self.closed = False

    def devices(self):
        return [{"name": "Kitchen", "model": "T8417", "serialNumber": "T8417P1"}]

    def wait_for_devices(self, timeout=60.0, poll=1.5):
        return self.devices()

    def close(self):
        self.closed = True


class _LiveSupervisor:
    def __init__(self):
        from surfaceguard.bridge.supervisor import BridgeStatus

        self.status = BridgeStatus(running=True, listening=True, port=3050)
        self.stopped = False

    def stop(self):
        self.stopped = True


def test_reopening_setup_does_not_ask_her_to_sign_in_again(qt_app, no_keychain):
    """Closing setup used to discard the signed-in session, so coming back meant
    typing the password again while the bridge was still connected."""
    dialog = OnboardingDialog(allow_demo=True, supervisor=_LiveSupervisor(),
                              client=_LiveClient())
    assert dialog.resumed
    assert dialog.page is Page.PICKER, "it should resume at choosing a camera"
    assert Page.CREDENTIALS not in dialog.route() or dialog.page is not Page.CREDENTIALS


def test_closing_setup_leaves_a_signed_in_bridge_running(qt_app, no_keychain):
    """The app hands it back next time; stopping it here is what forced the
    second sign-in."""
    supervisor = _LiveSupervisor()
    dialog = OnboardingDialog(allow_demo=True, supervisor=supervisor, client=_LiveClient())
    dialog.reject()
    assert not supervisor.stopped, "a live sign-in was thrown away on close"


def test_a_dead_bridge_is_not_treated_as_signed_in(qt_app, no_keychain):
    """Only a genuinely healthy bridge may skip the sign-in step."""
    from surfaceguard.bridge.supervisor import BridgeStatus

    supervisor = _LiveSupervisor()
    supervisor.status = BridgeStatus(running=False, listening=False)
    dialog = OnboardingDialog(allow_demo=True, supervisor=supervisor, client=_LiveClient())
    assert not dialog.resumed
    assert dialog.page is Page.KIND


class _FakeCamera:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True

    def read(self, timeout=0):
        return None


def test_picking_a_different_camera_stops_the_first_stream(qt_app, no_keychain):
    """Eufy serves one livestream at a time. Starting a second without stopping the
    first leaves her watching a picture that never arrives — which is exactly the
    "try each until I find the right one" flow."""
    dialog = OnboardingDialog(allow_demo=True)
    first = _FakeCamera()
    dialog.source = first
    dialog.chosen = {"name": "Kitchen", "serialNumber": "T8417P1"}
    dialog._go(Page.CONFIRM)
    dialog._back()
    assert first.stopped, "the first camera kept streaming"
    assert dialog.source is None and dialog.chosen is None
    assert dialog.page is Page.PICKER


def test_the_preview_says_which_camera_it_is(qt_app, no_keychain):
    """With three cameras on the account, "is this the right one?" needs a name."""
    dialog = OnboardingDialog(allow_demo=True)
    dialog.chosen = {"name": "Kitchen", "serialNumber": "T8417P1", "model": "T8417"}
    dialog._go(Page.CONFIRM)
    assert "Kitchen" in dialog.heading.text()
    assert "Back" in dialog.subtitle.text()


def test_the_picker_shows_friendly_model_names(qt_app, no_keychain):
    dialog = OnboardingDialog(allow_demo=True)
    dialog._show_devices([
        {"name": "Kitchen", "model": "T8417", "serialNumber": "T8417P1234567890"},
        {"name": "", "model": "T8416", "serialNumber": "T8416P0987654321"},
    ])
    rows = [dialog.devices.item(i).text() for i in range(dialog.devices.count())]
    assert any("Kitchen" in r and "Indoor Cam E30" in r for r in rows)
    assert any("(unnamed camera)" in r for r in rows), "an unnamed camera must still be pickable"
