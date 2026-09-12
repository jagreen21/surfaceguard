"""Codec handling at the Python/Node livestream boundary."""

from surfaceguard.camera.sources.eufy_bridge import _decoder_name, _sniff_decoder_name


def test_bridge_h264_name_maps_to_avc_decoder():
    assert _decoder_name("H264") == "h264"


def test_bridge_h265_name_maps_to_hevc_decoder():
    assert _decoder_name("H265") == "hevc"
    assert _decoder_name("HEVC") == "hevc"
    assert _decoder_name(1) == "hevc"


def test_missing_codec_keeps_the_bridge_default():
    assert _decoder_name(None) == "h264"


def test_payload_parameter_sets_override_incorrect_codec_metadata():
    assert _sniff_decoder_name(b"junk\x00\x00\x00\x01\x67\x64") == "h264"
    assert _sniff_decoder_name(b"\x00\x00\x00\x01\x40\x01") == "hevc"
    assert _sniff_decoder_name(b"\x00\x00\x01\x42\x01") == "hevc"
    assert _sniff_decoder_name(b"not annex b") == ""


def test_first_h265_chunk_replaces_the_initial_h264_decoder():
    from surfaceguard.camera.sources.eufy_bridge import EufyBridgeCamera

    camera = EufyBridgeCamera(client=None, serial="T8417P1")
    opened = []

    class Decoder:
        def parse(self, _chunk):
            return []

    def open_decoder(codec):
        opened.append(codec)
        camera._decoder_codec = codec
        camera._decoder = Decoder()

    camera._decoder = Decoder()
    camera._decoder_codec = "h264"
    camera._open_decoder = open_decoder
    camera._feed(b"video", {"videoCodec": "H265", "videoWidth": 3840})

    assert opened == ["hevc"]
    assert camera._decoder_codec == "hevc"


def test_hevc_payload_overrides_an_incorrect_h264_label():
    from surfaceguard.camera.sources.eufy_bridge import EufyBridgeCamera

    camera = EufyBridgeCamera(client=None, serial="T8417P1")
    opened = []

    class Decoder:
        def parse(self, _chunk):
            return []

    def open_decoder(codec):
        opened.append(codec)
        camera._decoder_codec = codec
        camera._decoder = Decoder()

    camera._decoder = Decoder()
    camera._decoder_codec = "h264"
    camera._open_decoder = open_decoder
    camera._feed(
        b"\x00\x00\x00\x01\x40\x01payload",
        {"videoCodec": "H264", "videoWidth": 1920},
    )

    assert opened == ["hevc"]
    assert camera._decoder_codec == "hevc"
    assert camera.video_diagnostics()["reported_codec"] == "h264"
    assert camera.video_diagnostics()["sniffed_codec"] == "hevc"


def test_unknown_payload_gets_one_alternate_decoder_attempt():
    from surfaceguard.camera.sources.eufy_bridge import EufyBridgeCamera

    camera = EufyBridgeCamera(client=None, serial="T8417P1")
    opened = []

    class RejectingDecoder:
        def parse(self, _chunk):
            raise ValueError("wrong codec")

    class AcceptingDecoder:
        def parse(self, _chunk):
            return []

    def open_decoder(codec):
        opened.append(codec)
        camera._decoder_codec = codec
        camera._decoder = AcceptingDecoder() if codec == "hevc" else RejectingDecoder()

    camera._decoder = RejectingDecoder()
    camera._decoder_codec = "h264"
    camera._open_decoder = open_decoder
    camera._feed(b"payload without parameter sets", {"videoCodec": "H264"})

    assert opened == ["hevc"]
    assert camera.video_diagnostics()["codec_override"] == "hevc"


def test_room_scan_pauses_and_restores_eufy_motion_tracking(monkeypatch):
    import surfaceguard.camera.sources.eufy_bridge as mod
    from surfaceguard.camera.sources.eufy_bridge import EufyBridgeCamera

    calls = []

    class Client:
        def send_wait(self, command, **payload):
            calls.append((command, payload))
            return {}

    monkeypatch.setattr(mod.time, "sleep", lambda _seconds: None)
    camera = EufyBridgeCamera(client=Client(), serial="T8417P1")
    camera._properties = {"motionTracking": True}

    assert camera.capabilities.auto_tracks_motion
    token = camera.suspend_auto_tracking()
    assert token is True and not camera.capabilities.auto_tracks_motion
    camera.restore_auto_tracking(token)
    assert camera.capabilities.auto_tracks_motion
    assert [call[1]["value"] for call in calls] == [False, True]
