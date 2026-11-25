import math

import pytest
from fastapi.testclient import TestClient

from ioscan.dashboard.server import create_app


def test_dashboard_home_and_idle_status(tmp_path):
    client = TestClient(create_app(tmp_path))
    home = client.get("/")
    assert home.status_code == 200
    assert "IOSCAN" in home.text
    assert "Scan Console" in home.text
    assert "Ready check" in home.text and "Library" in home.text and "Calibration" in home.text
    assert "btn-save-geometry" in home.text and "btn-cal-record" in home.text
    assert "app.js?v=" in home.text                       # cache-busted front-end script

    status = client.get("/api/status")
    assert status.status_code == 200
    payload = status.json()
    assert payload["phase"] == "idle"
    assert payload["model_asset"] is None


def test_dashboard_does_not_fake_camera_or_model(tmp_path):
    client = TestClient(create_app(tmp_path))
    assert client.get("/api/camera/preview.jpg").status_code == 204
    assert client.get("/api/model/segmented").status_code == 404
    assert client.get("/api/model/tooth/14").status_code == 404
    assert client.post("/api/segmentation/start", json={"jaw": "lower"}).status_code == 409


def test_dashboard_can_load_its_own_pipeline_script(tmp_path):
    from ioscan.dashboard.server import ScannerDashboard

    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    source = (
        "from pathlib import Path\n"
        "def run_pipeline(*args, **kwargs):\n"
        "    return {'ok': True}\n"
    )
    (project / "scripts" / "run_pipeline.py").write_text(source, encoding="utf-8")
    module = ScannerDashboard(project)._load_pipeline_script()
    assert module.run_pipeline()["ok"] is True


def test_camera_device_endpoint_returns_device_names_and_backend(tmp_path, monkeypatch):
    from ioscan.dashboard import server as dashboard_server

    expected = [{
        "camera_id": 1, "backend": "DirectShow", "name": "Arducam OV2311",
        "path": None, "vid": 0x2BDF, "pid": 0x0210,
        "label": "Arducam OV2311 — DirectShow (index 1)",
    }]
    monkeypatch.setattr(dashboard_server, "enumerate_usb_cameras", lambda: expected)
    response = TestClient(create_app(tmp_path)).get("/api/camera/devices")
    assert response.status_code == 200
    assert response.json()["devices"] == expected



def test_dashboard_hardware_geometry_defaults_and_save(tmp_path):
    client = TestClient(create_app(tmp_path))

    defaults = client.get("/api/hardware-geometry")
    assert defaults.status_code == 200
    assert defaults.json()["baseline_mm"] == 18.0
    assert defaults.json()["laser_tilt_deg"] == 10.0
    assert defaults.json()["working_distance_mm"] == pytest.approx(18.0 / math.tan(math.radians(10.0)), abs=0.001)

    payload = {
        "baseline_mm": 14.0,
        "laser_tilt_deg": 35.0,
        "fan_half_angle_deg": 22.5,
        "exposure": -6,
        "speed_mm_s": 4.2,
        "depth_min_mm": 8.0,
        "depth_max_mm": 38.0,
    }
    saved = client.post("/api/hardware-geometry", json=payload)
    assert saved.status_code == 200
    assert saved.json()["working_distance_mm"] == pytest.approx(14.0 / math.tan(math.radians(35.0)), abs=0.001)

    reloaded = client.get("/api/hardware-geometry")
    assert reloaded.status_code == 200
    assert reloaded.json()["baseline_mm"] == 14.0
    assert reloaded.json()["depth_max_mm"] == 38.0


def test_dashboard_rejects_invalid_depth_range(tmp_path):
    client = TestClient(create_app(tmp_path))
    payload = {
        "baseline_mm": 12.0,
        "laser_tilt_deg": 30.0,
        "fan_half_angle_deg": 25.0,
        "exposure": -7,
        "speed_mm_s": 5.0,
        "depth_min_mm": 45.0,
        "depth_max_mm": 10.0,
    }
    response = client.post("/api/hardware-geometry", json=payload)
    assert response.status_code == 422
    assert "smaller than maximum" in response.json()["detail"]


def test_start_capture_creates_session_directory_before_saving_roi(tmp_path, monkeypatch):
    from ioscan.dashboard.server import CaptureRequest, ScannerDashboard

    project = tmp_path / "project"
    controller = ScannerDashboard(project)
    monkeypatch.setattr(controller, "_stop_idle_camera_preview", lambda: True)
    # Keep the test focused on session setup; do not start real camera hardware.
    monkeypatch.setattr(controller, "_capture_worker", lambda *args, **kwargs: None)

    result = controller.start_capture(CaptureRequest(
        auto_segment=False, frames=20, session_name="roi_session_test"
    ))

    session_dir = project / "data" / "sessions" / "roi_session_test"
    assert result["started"] is True
    assert session_dir.is_dir()
    assert (session_dir / "roi.json").is_file()
    assert (project / "data" / "results" / "roi_session_test").is_dir()
    controller._worker.join(timeout=2.0)


def test_calibration_status_and_marker_sheet_endpoints(tmp_path):
    client = TestClient(create_app(tmp_path))
    cal = client.get("/api/calibration").json()
    assert cal["calibrated"] is False and cal["board"]["cols"] == 16 and cal["board"]["marker_mm"] == 6.0
    assert client.get("/api/marker-sheet.pdf").status_code == 404          # no docs folder in an empty project
    assert client.post("/api/calibration/start", json={}).status_code == 404   # nothing recorded yet
    assert client.post("/api/results/open", json={"name": "nope"}).status_code == 404
    assert client.post("/api/results/open", json={"name": "../etc"}).status_code == 422
    (tmp_path / "data" / "calibration").mkdir(parents=True)
    (tmp_path / "data" / "calibration" / "calibration.json").write_text(
        '{"camera": {"width": 1600, "height": 1200, "fx": 1010.0, "fy": 1010.0, "cx": 800, "cy": 600, '
        '"dist_coeffs": [-0.09, 0, 0, 0, 0]}, "laser": {"normal": [0, -0.98, -0.19], "d": 18.0}, '
        '"metadata": {"quality": "good", "sheet_residual_mm": 0.08, "camera_rms_px": 0.5}}', encoding="utf-8")
    cal = client.get("/api/calibration").json()
    assert cal["calibrated"] is True and cal["focal_px"] == 1010.0 and cal["quality"] == "good"


def test_library_lists_scans_calibration_recordings_and_arches_and_opens_results(tmp_path):
    import json as _json
    from ioscan.dashboard.server import ScannerDashboard

    c = ScannerDashboard(tmp_path)
    for name in ("scan_a", "calib_b"):
        d = tmp_path / "data" / "sessions" / name
        d.mkdir(parents=True)
        (d / "frames.csv").write_text("idx,t_ns,laser_on,file\n0,0,0,frames/0.jpg\n", encoding="utf-8")
    res = tmp_path / "data" / "results" / "scan_a"
    res.mkdir(parents=True)
    (res / "reconstructed_mesh.ply").write_bytes(b"ply")
    (res / "pipeline_report.json").write_text(_json.dumps({
        "verdict": {"status": "check", "title": "Usable, with limits", "reasons": ["x"]},
        "translation": {"used": "markers"}, "mesh_triangles": 12, "mesh_vertices": 9, "warnings": ["w"]}), encoding="utf-8")
    arch = tmp_path / "data" / "results" / "arch_1"
    arch.mkdir()
    (arch / "reconstructed_mesh.ply").write_bytes(b"ply")
    items = {x["name"]: x for x in c.list_sessions()}
    assert items["scan_a"]["kind"] == "scan" and items["scan_a"]["verdict"] == "check" and items["scan_a"]["motion"] == "markers"
    assert items["calib_b"]["kind"] == "calibration" and items["calib_b"]["has_mesh"] is False
    assert items["arch_1"]["kind"] == "arch" and items["arch_1"]["frames"] is None
    assert c.open_result("scan_a")["opened"] is True
    snap = c.snapshot()
    assert snap["phase"] == "complete" and snap["model_asset"] == "mesh" and snap["mesh_faces"] == 12
    assert snap["verdict"]["status"] == "check" and snap["translation_source"] == "markers"
    assert c.results_dir == res


def test_ready_check_reports_markers_light_and_line(tmp_path):
    import cv2
    import numpy as np
    from ioscan.dashboard.server import ScannerDashboard
    from ioscan.pipeline.markers import MarkerBoard, render_board_image

    c = ScannerDashboard(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "roi.json").write_text(
        '{"version": 1, "width": 1600, "height": 1200, "enabled": true, '
        '"points": [[0.32, 0.3], [0.68, 0.3], [0.68, 0.7], [0.32, 0.7]]}', encoding="utf-8")
    tex, _m = render_board_image(MarkerBoard(), 10.0, 4.0)
    off = np.full((1200, 1600), 40, np.uint8)
    patch = cv2.resize(tex, (700, 620), interpolation=cv2.INTER_AREA)
    off[300:920, 450:1150] = (20 + patch.astype(np.float32) * 0.3).astype(np.uint8)
    on = off.copy()
    on[598:603, 500:1100] = np.clip(on[598:603, 500:1100].astype(int) + 90, 0, 255).astype(np.uint8)
    c._idle_ready_check(off, on)
    rc = c.snapshot()["ready_check"]
    items = {i["key"]: i for i in rc["items"]}
    assert items["markers"]["state"] == "ok" and int(items["markers"]["value"].split()[0]) >= 20
    assert items["line"]["state"] == "ok" and items["calibration"]["state"] == "warn"
    assert rc["state"] == "warn" and "Calibrate" in rc["advice"]             # only the calibration is missing
    c._ready_pub_at = 0.0
    c._ready_markers["at"] = 0.0
    bright = np.full((1200, 1600), 160, np.uint8)
    c._idle_ready_check(bright, bright)
    rc = c.snapshot()["ready_check"]
    items = {i["key"]: i for i in rc["items"]}
    assert rc["state"] == "bad" and items["light"]["state"] == "bad" and items["markers"]["state"] == "warn"
    assert items["line"]["state"] == "idle" and "light" in rc["advice"].lower()
