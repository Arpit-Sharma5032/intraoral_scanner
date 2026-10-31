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
        [b"# ERROR: BNO086 SPI initialization failed; scan packets are disabled\n"],
    )
    with pytest.raises(RuntimeError, match="BNO086 SPI initialization failed"):
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
    monkeypatch.setattr(logger.cv2, "VideoCapture", lambda *_args: camera)
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
    assert (out / "frames.csv").is_file()
    assert (out / "imu.csv").is_file()
    assert (out / "meta.json").is_file()
