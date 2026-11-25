import threading

import numpy as np

from ioscan.dashboard import server as dashboard_server
from ioscan.dashboard.server import ScannerDashboard


class FakeCamera:
    def __init__(self, frame):
        self.frame = frame
        self.set_calls = []
        self.released = False

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        return True

    def get(self, prop):
        return 1.0

    def read(self):
        return True, self.frame.copy()

    def release(self):
        self.released = True


def _run_preview(tmp_path, monkeypatch, frame):
    camera = FakeCamera(frame)
    open_calls = {}

    def fake_open(camera_id, **kwargs):
        open_calls.update(camera_id=camera_id, **kwargs)
        return camera, frame.copy(), {
            "backend": "DirectShow", "requested_format": "MJPG",
            "width": 1600, "height": 1200, "fps": 30.0,
        }

    monkeypatch.setattr(dashboard_server, "open_usb_camera", fake_open)
    controller = ScannerDashboard(tmp_path)
    worker = threading.Thread(
        target=controller._idle_camera_preview_loop,
        args=(1, "DirectShow", "Arducam OV2311"),
        daemon=True,
    )
    worker.start()
    assert controller._idle_preview_ready.wait(timeout=3.0)
    controller._idle_preview_stop.set()
    worker.join(timeout=3.0)
    return controller, worker, camera, open_calls


def test_idle_preview_uses_selected_backend_and_reports_brightness(tmp_path, monkeypatch):
    frame = np.full((1200, 1600, 3), 64, dtype=np.uint8)
    controller, worker, camera, open_calls = _run_preview(tmp_path, monkeypatch, frame)

    assert not worker.is_alive()
    assert open_calls["camera_id"] == 1
    assert open_calls["backend_name"] == "DirectShow"
    assert open_calls["exposure"] is None
    assert (dashboard_server.cv2.CAP_PROP_AUTO_EXPOSURE, 1.0) in camera.set_calls
    assert controller._idle_preview_info["brightness_mean"] == 64.0
    assert controller._idle_preview_info["brightness_p95"] == 64.0
    assert controller._latest_jpeg.startswith(b"\xff\xd8")
    assert camera.released


def test_idle_preview_rejects_all_zero_virtual_camera_frame(tmp_path, monkeypatch):
    frame = np.zeros((1200, 1600, 3), dtype=np.uint8)
    controller, worker, camera, open_calls = _run_preview(tmp_path, monkeypatch, frame)

    assert not worker.is_alive()
    assert camera.released
    assert controller._idle_preview_error is not None
    assert "all-black frame" in controller._idle_preview_error
    assert "Arducam" in controller._idle_preview_error
    assert controller._latest_jpeg is None
