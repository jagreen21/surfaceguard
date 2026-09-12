"""Codec handling at the Python/Node livestream boundary."""

from surfaceguard.camera.sources.eufy_bridge import _decoder_name


def test_bridge_h264_name_maps_to_avc_decoder():
    assert _decoder_name("H264") == "h264"


def test_bridge_h265_name_maps_to_hevc_decoder():
    assert _decoder_name("H265") == "hevc"
    assert _decoder_name("HEVC") == "hevc"
    assert _decoder_name(1) == "hevc"


def test_missing_codec_keeps_the_bridge_default():
    assert _decoder_name(None) == "h264"


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
