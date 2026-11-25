from __future__ import annotations

import numpy as np
import pytest

from ioscan.capture import camera as camera_module


class FakeCapture:
    def __init__(self, fourcc: int, frame_shape=(1200, 1600), accept_fourcc_set=True):
        self.fourcc = fourcc
        self.frame_shape = frame_shape
        self.accept_fourcc_set = accept_fourcc_set
        self.released = False

    def isOpened(self):
        return True

    def set(self, prop, value):
        if prop == camera_module.cv2.CAP_PROP_FOURCC and self.accept_fourcc_set:
            self.fourcc = int(value)
        return True

    def get(self, prop):
        values = {
            camera_module.cv2.CAP_PROP_FOURCC: self.fourcc,
            camera_module.cv2.CAP_PROP_FRAME_WIDTH: self.frame_shape[1],
            camera_module.cv2.CAP_PROP_FRAME_HEIGHT: self.frame_shape[0],
            camera_module.cv2.CAP_PROP_FPS: 50.0,
        }
        return values.get(prop, 0.0)

    def read(self):
        return True, np.full(self.frame_shape, 80, dtype=np.uint8)

    def release(self):
        self.released = True


def test_scan_camera_requires_mjpg_and_reports_measured_rate(monkeypatch):
    mjpg = camera_module.cv2.VideoWriter_fourcc(*"MJPG")
    capture = FakeCapture(mjpg)
    monkeypatch.setattr(camera_module.cv2, "VideoCapture", lambda *_args: capture)

    cap, frame, info = camera_module.open_usb_camera(0, 1600, 1200, 50, read_attempts=8,
        backend_name="DirectShow", required_fourcc="MJPG")

    assert cap is capture
    assert frame.shape == (1200, 1600)
    assert info["actual_fourcc_text"] == "MJPG"
    assert info["measured_fps"] > 0
    assert info["driver_reported_fps"] == 50.0


def test_scan_camera_rejects_yuy2_when_mjpg_is_required(monkeypatch):
    yuy2 = camera_module.cv2.VideoWriter_fourcc(*"YUY2")
    captures = []

    def factory(*_args):
        cap = FakeCapture(yuy2, accept_fourcc_set=False)
        captures.append(cap)
        return cap

    monkeypatch.setattr(camera_module.cv2, "VideoCapture", factory)
    with pytest.raises(RuntimeError, match="Required format: MJPG"):
        camera_module.open_usb_camera(0, 1600, 1200, 50, read_attempts=8,
            backend_name="DirectShow", required_fourcc="MJPG")
    assert captures and all(cap.released for cap in captures)


def test_camera_refuses_zero_read_attempt_configuration():
    with pytest.raises(ValueError, match="at least 3"):
        camera_module.open_usb_camera(0, read_attempts=0)


def test_reconstruction_start_clears_stale_segmentation_artifacts(tmp_path):
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_pipeline.py"
    spec = importlib.util.spec_from_file_location("ioscan_run_pipeline_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    out = tmp_path / "results"
    (out / "segmented_teeth_by_class").mkdir(parents=True)
    for name in ("pipeline_report.json", "segmented_full_colored.ply", "segmentation_labels.npz"):
        (out / name).write_text("stale")
    (out / "segmented_teeth_by_class" / "tooth_class_01.ply").write_text("stale")
    (out / "keep-me.txt").write_text("operator note")

    module._clear_stale_scan_outputs(out)

    assert not (out / "pipeline_report.json").exists()
    assert not (out / "segmented_full_colored.ply").exists()
    assert not (out / "segmentation_labels.npz").exists()
    assert not (out / "segmented_teeth_by_class").exists()
    assert (out / "keep-me.txt").read_text() == "operator note"


def test_ffmpeg_dshow_capture_decodes_jpeg_frames(monkeypatch):
    import io

    frame = np.full((24, 32, 3), 90, dtype=np.uint8)
    ok, encoded = camera_module.cv2.imencode('.jpg', frame)
    assert ok

    class FakeStdout(io.BytesIO):
        def read1(self, n=-1):
            return self.read(n)

    class FakeProcess:
        def __init__(self, payload):
            self.stdout = FakeStdout(payload)
            self.returncode = None
        def poll(self):
            return self.returncode
        def terminate(self):
            self.returncode = 0
        def wait(self, timeout=None):
            self.returncode = 0
            return 0
        def kill(self):
            self.returncode = -9

    jpg = encoded.tobytes()
    process = FakeProcess(jpg + jpg + jpg)
    cap = camera_module.FFmpegDShowCapture(process, 32, 24, 50)
    got1, decoded1 = cap.read()
    got2, raw2 = cap.read_encoded()
    got3, decoded3 = cap.read()
    assert got1 and got2 and got3
    assert decoded1.shape == (24, 32)
    assert raw2 == jpg
    assert decoded3.shape == (24, 32)
    assert cap.get(camera_module.cv2.CAP_PROP_FOURCC) == float(
        camera_module.cv2.VideoWriter_fourcc(*'MJPG')
    )
    cap.release()


def test_ffmpeg_command_uses_exact_directshow_mjpg_mode(monkeypatch):
    import io

    frame = np.full((24, 32, 3), 90, dtype=np.uint8)
    ok, encoded = camera_module.cv2.imencode('.jpg', frame)
    assert ok

    class FakeStdout(io.BytesIO):
        def read1(self, n=-1):
            return self.read(n)

    class FakeProcess:
        def __init__(self, cmd, **kwargs):
            self.cmd = cmd
            self.stdout = FakeStdout(encoded.tobytes() * 20)
            self.returncode = None
        def poll(self):
            return self.returncode
        def terminate(self):
            self.returncode = 0
        def wait(self, timeout=None):
            self.returncode = 0
            return 0
        def kill(self):
            self.returncode = -9

    made = {}
    def fake_popen(cmd, **kwargs):
        process = FakeProcess(cmd, **kwargs)
        made['process'] = process
        return process

    monkeypatch.setattr(camera_module.shutil, 'which', lambda _name: '/usr/bin/ffmpeg')
    monkeypatch.setattr(camera_module.subprocess, 'Popen', fake_popen)
    monkeypatch.setattr(camera_module, '_prime_directshow_controls', lambda *_a: {'attempted': True})
    result = camera_module._open_ffmpeg_dshow(
        'Arducam OV2311 USB Camera', 32, 24, 50, 16, 42.5, [], 0, -7
    )
    assert result is not None
    _cap, first_frame, info = result
    command = made['process'].cmd
    assert command[command.index('-video_size') + 1] == '32x24'
    assert command[command.index('-framerate') + 1] == '50'
    assert command[command.index('-vcodec') + 1] == 'mjpeg'
    assert command[command.index('-i') + 1] == 'video=Arducam OV2311 USB Camera'
    assert info['actual_fourcc_text'] == 'MJPG'
    assert info['capture_engine'] == 'ffmpeg'
    assert first_frame.shape == (24, 32)
    _cap.release()


def test_prefer_ffmpeg_bypasses_open_cv_for_directshow_mjpg(monkeypatch):
    monkeypatch.setattr(camera_module.os, "name", "nt")
    called = {}
    expected = (FakeCapture(camera_module.cv2.VideoWriter_fourcc(*"MJPG")),
                np.zeros((1200, 1600), dtype=np.uint8),
                {"backend": "FFmpeg/DirectShow", "capture_engine": "ffmpeg",
                 "actual_fourcc_text": "MJPG", "measured_fps": 49.0})

    def fake_ffmpeg(*args, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs
        return expected

    def forbidden_opencv(*_args, **_kwargs):
        raise AssertionError("OpenCV must not be opened when FFmpeg is preferred")

    monkeypatch.setattr(camera_module, "_open_ffmpeg_dshow", fake_ffmpeg)
    monkeypatch.setattr(camera_module.cv2, "VideoCapture", forbidden_opencv)
    result = camera_module.open_usb_camera(
        0, 1600, 1200, 50, exposure=None, read_attempts=8,
        backend_name="DirectShow", required_fourcc="MJPG", min_fps=42.5,
        camera_name="Arducam OV2311 USB Camera", prefer_ffmpeg=True,
    )
    assert result == expected
    assert called["args"][0] == "Arducam OV2311 USB Camera"
    assert called["kwargs"]["prime_controls"] is False


def test_preferred_ffmpeg_failure_does_not_fall_through_to_blocking_opencv(monkeypatch):
    monkeypatch.setattr(camera_module.os, "name", "nt")
    monkeypatch.setattr(camera_module, "_open_ffmpeg_dshow", lambda *a, **k: None)

    def forbidden_opencv(*_args, **_kwargs):
        raise AssertionError("OpenCV must not be tried after preferred FFmpeg failure")

    monkeypatch.setattr(camera_module.cv2, "VideoCapture", forbidden_opencv)
    with pytest.raises(RuntimeError, match="OpenCV was deliberately not tried"):
        camera_module.open_usb_camera(
            0, 1600, 1200, 50, read_attempts=8, backend_name="DirectShow",
            required_fourcc="MJPG", min_fps=42.5,
            camera_name="Arducam OV2311 USB Camera", prefer_ffmpeg=True,
        )
