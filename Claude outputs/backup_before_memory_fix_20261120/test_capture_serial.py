from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from ioscan.capture.logger import SerialIMUReader


class _FakeSerialPort:
    def __init__(self, lines, stop_event):
        self._lines = list(lines)
        self._stop_event = stop_event

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def readline(self):
        if self._lines:
            return self._lines.pop(0)
        self._stop_event.set()
        return b""


def _run_reader(monkeypatch, lines):
    reader = SerialIMUReader("COM_TEST")
    fake_serial = types.SimpleNamespace(
        Serial=lambda *_args, **_kwargs: _FakeSerialPort(lines, reader._stop)
    )
    monkeypatch.setitem(sys.modules, "serial", fake_serial)
    reader._run()
    return reader


def test_serial_reader_queues_each_event_and_extends_micros_wrap(monkeypatch):
    lines = [
        b"4294967290,0,1,0,0,0,41\n",
        b"10,1,0.999,0.01,0,0,42\n",
    ]
    reader = _run_reader(monkeypatch, lines)

    first = reader.get_next(timeout=0.01)
    second = reader.get_next(timeout=0.01)

    assert first[0] == 4_294_967_290
    assert first[1] is False
    assert first[3] == 41
    assert second[0] == 4_294_967_306
    assert second[1] is True
    assert second[3] == 42
    assert np.isclose(np.linalg.norm(second[2]), 1.0)
    assert [t for t, _ in reader.packets()] == [4_294_967_290, 4_294_967_306]


def test_serial_reader_reports_dropped_strobe_sequence(monkeypatch):
    lines = [
        b"1000,0,1,0,0,0,7\n",
        b"3000,1,1,0,0,0,9\n",
    ]
    reader = _run_reader(monkeypatch, lines)
    with pytest.raises(RuntimeError, match="sequence gap"):
        reader.get_next(timeout=0.01)


def test_serial_reader_surfaces_firmware_bno_error(monkeypatch):
    reader = _run_reader(
        monkeypatch,
        [b"# ERROR: BNO086 I2C initialization failed; scan packets are disabled\n"],
    )
    with pytest.raises(RuntimeError, match="BNO086 I2C initialization failed"):
        reader.get_next(timeout=0.01)


def test_six_field_legacy_firmware_packets_remain_supported(monkeypatch):
    reader = _run_reader(monkeypatch, [b"2000,1,1,0,0,0\n"])
    packet = reader.get_next(timeout=0.01)
    assert packet[:2] == (2000, True)
    assert packet[3] is None


def test_capture_consumes_one_ordered_strobe_event_per_camera_frame(monkeypatch, tmp_path):
    from ioscan.capture import logger
    from ioscan.capture.logger import CaptureConfig, record_hardware_session

    frames = [np.full((48, 64), i, dtype=np.uint8) for i in range(4)]

    class FakeCamera:
        def __init__(self):
            self._frames = list(frames)
            self.released = False

        def isOpened(self):
            return True

        def set(self, _prop, _value):
            return True

        def get(self, prop):
            values = {
                logger.cv2.CAP_PROP_FRAME_WIDTH: 64,
                logger.cv2.CAP_PROP_FRAME_HEIGHT: 48,
                logger.cv2.CAP_PROP_FPS: 50,
                logger.cv2.CAP_PROP_EXPOSURE: -7,
            }
            return values.get(prop, 0)

        def read(self):
            if not self._frames:
                return False, None
            return True, self._frames.pop(0)

        def release(self):
            self.released = True

    class FakeReader:
        def __init__(self, _port, _baudrate):
            self.events = [
                (1_000_000 + i * 20_000, bool(i % 2), np.array([1.0, 0, 0, 0]), i)
                for i in range(4)
            ]
            self._all = list(self.events)
            self.error = None
            self.warnings = []

        def start(self):
            pass

        def get_next(self, timeout=1.0):
            return self.events.pop(0)

        def packets(self):
            return [(t, q) for t, _on, q, _seq in self._all]

        def stop(self):
            pass

    camera = FakeCamera()
    camera_open_calls = {}
    def fake_open(camera_id, width, height, fps, **kwargs):
        camera_open_calls.update(camera_id=camera_id, width=width, height=height, fps=fps, **kwargs)
        return camera, frames[0].copy(), {
            "backend": "DirectShow", "requested_format": "MJPG", "actual_fourcc_text": "MJPG",
            "width": width, "height": height, "fps": 50.0, "measured_fps": 50.0,
            "driver_reported_fps": 50.0, "set_results": {},
        }
    monkeypatch.setattr(logger, "open_usb_camera", fake_open)
    monkeypatch.setattr(logger, "SerialIMUReader", FakeReader)
    callback_rows = []

    calibration_camera = {
        "width": 64, "height": 48, "fx": 100.0, "fy": 100.0,
        "cx": 32.0, "cy": 24.0, "dist_coeffs": [0, 0, 0, 0, 0],
    }
    calibration_laser = {"normal": [0, -0.8, -0.6], "d": 7.5}
    out = record_hardware_session(
        tmp_path / "capture",
        cfg=CaptureConfig(
            camera_id=0, width=64, height=48, fps=50, serial_port="COM1", max_frames=4,
        ),
        camera_cal_dict=calibration_camera,
        laser_cal_dict=calibration_laser,
        frame_callback=lambda image, idx, laser_on, t_ns, q: callback_rows.append((idx, laser_on, t_ns, q)),
    )

    assert len(callback_rows) == 4
    assert [r[1] for r in callback_rows] == [False, True, False, True]
    assert [r[2] for r in callback_rows] == [1_000_000_000, 1_020_000_000, 1_040_000_000, 1_060_000_000]
    assert all(np.allclose(r[3], [1, 0, 0, 0]) for r in callback_rows)
    assert camera.released
    assert camera_open_calls["required_fourcc"] == "MJPG"
    assert camera_open_calls["read_attempts"] >= 16
    assert camera_open_calls["min_fps"] == 42.5
    assert (out / "frames.csv").is_file()
    assert (out / "imu.csv").is_file()
    assert (out / "meta.json").is_file()
    assert all((out / f"frames/{idx:06d}.png").is_file() for idx in range(4))
    meta = __import__("json").loads((out / "meta.json").read_text())
    assert meta["requested_fps"] == 50.0
    assert meta["fps"] == 50.0
    assert meta["camera_runtime"]["actual_fourcc_text"] == "MJPG"


def _framed_packet(payload: str) -> bytes:
    from ioscan.capture.logger import _crc16_ccitt
    data = payload.encode("ascii")
    return b"@" + data + f"*{_crc16_ccitt(data):04X}\n".encode("ascii")


def test_serial_reader_accepts_crc_framed_packet_and_resyncs_after_garbage(monkeypatch):
    payload = "123456,1,0.978333,0.035217,-0.052063,-0.197449,1211"
    reader = _run_reader(monkeypatch, [b"\x00\x00\x00garbage" + _framed_packet(payload)])
    packet = reader.get_next(timeout=0.01)
    assert packet[0] == 123456
    assert packet[1] is True
    assert packet[3] == 1211
    assert any("stray serial byte" in warning for warning in reader.warnings)


def test_serial_reader_rejects_bad_crc_and_detects_sequence_gap(monkeypatch):
    first = _framed_packet("1000,0,1,0,0,0,10")
    bad = bytearray(_framed_packet("2000,1,1,0,0,0,11"))
    bad[-3] = ord("0") if bad[-3] != ord("0") else ord("1")
    third = _framed_packet("3000,0,1,0,0,0,12")
    reader = _run_reader(monkeypatch, [first, bytes(bad), third])
    with pytest.raises(RuntimeError, match="sequence gap"):
        reader.get_next(timeout=0.01)
    assert any("CRC mismatch" in warning for warning in reader.warnings)


def test_serial_reader_discards_corrupt_legacy_null_line(monkeypatch):
    reader = _run_reader(
        monkeypatch,
        [b"\x00" * 32 + b",0,0.978333,0.035217,-0.052063,-0.197449,1211\n",
         b"123456,1,1,0,0,0,1212\n"],
    )
    packet = reader.get_next(timeout=0.01)
    assert packet[0] == 123456
    assert packet[3] == 1212
    assert any("NUL bytes" in warning for warning in reader.warnings)


def test_crc16_ccitt_false_standard_check_value():
    from ioscan.capture.logger import _crc16_ccitt
    assert _crc16_ccitt(b"123456789") == 0x29B1


def test_serial_reader_preserves_sequence_across_stale_imu_warning(monkeypatch):
    # Firmware now repeats its latest valid quaternion during a stale IMU
    # interval instead of omitting the physical STROBE event. The diagnostic
    # warning must not affect framed packet parsing or sequence continuity.
    lines = [
        b"# WARN: IMU orientation is stale; repeating latest valid quaternion to preserve STROBE sequence\n",
        _framed_packet("1000,0,1,0,0,0,50"),
        _framed_packet("21000,1,1,0,0,0,51"),
    ]
    reader = _run_reader(monkeypatch, lines)
    first = reader.get_next(timeout=0.01)
    second = reader.get_next(timeout=0.01)
    assert first[3] == 50
    assert second[3] == 51
    assert any("IMU orientation is stale" in warning for warning in reader.warnings)


def test_serial_reader_resynchronizes_when_multiple_framed_packets_share_one_line(monkeypatch):
    from ioscan.capture.logger import _crc16_ccitt

    def framed(payload: bytes) -> bytes:
        return b"@" + payload + b"*" + f"{_crc16_ccitt(payload):04X}".encode("ascii")

    packet1 = framed(b"1000,0,1.000000,0.000000,0.000000,0.000000,10")
    packet2 = framed(b"2000,1,0.999000,0.010000,0.000000,0.000000,11")
    # Simulate a lost line ending between packets. Both frames still have their
    # own start marker and CRC; the reader should recover both independently.
    reader = _run_reader(monkeypatch, [packet1 + packet2 + b"\n"])

    first = reader.get_next(timeout=0.01)
    second = reader.get_next(timeout=0.01)
    assert first[3] == 10
    assert second[3] == 11
    assert first[1] is False
    assert second[1] is True


def test_serial_reader_does_not_accept_corrupt_framed_packet(monkeypatch):
    from ioscan.capture.logger import _crc16_ccitt

    payload = b"1000,0,1.000000,0.000000,0.000000,0.000000,10"
    crc = _crc16_ccitt(payload)
    # Alter the quaternion after calculating CRC; it must be rejected.
    corrupt = b"@" + payload.replace(b"1.000000", b"9.000000") + b"*" + f"{crc:04X}".encode() + b"\n"
    reader = _run_reader(monkeypatch, [corrupt])
    assert reader.error is None
    assert reader.latest() is None
    assert any("CRC mismatch" in warning for warning in reader.warnings)


def test_ffmpeg_capture_decodes_only_sampled_matched_pairs(monkeypatch, tmp_path):
    """The synchronized acquisition loop must not grayscale-decode every MJPEG frame."""
    import cv2
    from ioscan.capture import logger
    from ioscan.capture.logger import CaptureConfig, record_hardware_session

    class FakeFFmpegCamera:
        def __init__(self):
            self.frames = []
            for idx in range(8):
                pixels = np.full((48, 64), 20 if idx % 2 == 0 else 220, dtype=np.uint8)
                ok, encoded = cv2.imencode(".jpg", pixels)
                assert ok
                self.frames.append(encoded.tobytes())
            self.released = False

        def isOpened(self):
            return True

        def read_encoded(self):
            if not self.frames:
                return False, None
            return True, self.frames.pop(0)

        def get(self, prop):
            values = {
                logger.cv2.CAP_PROP_FRAME_WIDTH: 64,
                logger.cv2.CAP_PROP_FRAME_HEIGHT: 48,
                logger.cv2.CAP_PROP_FPS: 50,
                logger.cv2.CAP_PROP_EXPOSURE: -7,
            }
            return values.get(prop, 0)

        def release(self):
            self.released = True

    class FakeReader:
        def __init__(self, *_args):
            self.events = [
                (1_000_000 + idx * 20_000, bool(idx % 2), np.array([1., 0., 0., 0.]), idx + 1)
                for idx in range(8)
            ]
            self._all = list(self.events)
            self.warnings = []

        def start(self):
            pass

        def get_next(self, timeout=1.0):
            return self.events.pop(0)

        def packets(self):
            return [(t, q) for t, _on, q, _seq in self._all]

        def stats(self):
            return {"events_received": 8, "max_event_queue_depth": 1,
                    "current_event_queue_depth": 0, "event_queue_capacity": 256}

        def stop(self):
            pass

    camera = FakeFFmpegCamera()
    monkeypatch.setattr(
        logger, "open_usb_camera",
        lambda *_a, **_k: (camera, None, {
            "capture_engine": "ffmpeg", "backend": "DirectShow",
            "actual_fourcc_text": "MJPG", "width": 64, "height": 48,
            "fps": 50.0, "measured_fps": 50.0, "set_results": {},
        }),
    )
    monkeypatch.setattr(logger, "SerialIMUReader", FakeReader)
    decode_calls = []
    real_imdecode = logger.cv2.imdecode

    def count_decode(*args, **kwargs):
        decode_calls.append(1)
        return real_imdecode(*args, **kwargs)

    monkeypatch.setattr(logger.cv2, "imdecode", count_decode)
    callbacks = []
    progress = []
    out = record_hardware_session(
        tmp_path / "ffmpeg_capture",
        cfg=CaptureConfig(width=64, height=48, fps=50, serial_port="COM_TEST",
                          max_frames=8, live_preview_profile_stride=2),
        camera_cal_dict={"width": 64, "height": 48, "fx": 100, "fy": 100,
                         "cx": 32, "cy": 24, "dist_coeffs": [0, 0, 0, 0, 0]},
        laser_cal_dict={"normal": [0, -0.8, -0.6], "d": 7.5},
        frame_callback=lambda image, idx, laser_on, t_ns, q: callbacks.append((idx, laser_on, image.shape)),
        progress_callback=lambda event: progress.append(event["frame"]),
    )

    # Four ON profiles, stride two => two OFF/ON preview pairs, four decodes total.
    assert len(decode_calls) == 4
    assert [(idx, on) for idx, on, _shape in callbacks] == [(2, False), (3, True), (6, False), (7, True)]
    assert len(progress) == 8
    assert len(list((out / "frames").glob("*.jpg"))) == 8
    assert camera.released
