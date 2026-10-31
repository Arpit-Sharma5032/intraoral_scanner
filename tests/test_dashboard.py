from fastapi.testclient import TestClient

from ioscan.dashboard.server import create_app


def test_dashboard_home_and_idle_status(tmp_path):
    client = TestClient(create_app(tmp_path))
    home = client.get("/")
    assert home.status_code == 200
    assert "IOSCAN" in home.text
    assert "Scan Console" in home.text

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
