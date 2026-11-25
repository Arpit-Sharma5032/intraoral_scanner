import numpy as np

from ioscan.calibration.models import CameraIntrinsics, LaserPlane


def test_laser_plane_normalization_scales_offset_together():
    plane = LaserPlane(normal=np.array([0.0, -2.0, -1.2]), d=24.0)
    expected_norm = np.sqrt(2.0**2 + 1.2**2)
    assert np.isclose(np.linalg.norm(plane.normal), 1.0)
    assert np.isclose(plane.d, 24.0 / expected_norm)
    p = np.array([0.0, 0.0, 20.0])
    assert np.isclose(plane.signed_distance(p), 0.0, atol=1e-9)


def test_nominal_laser_plane_matches_simulator_geometry():
    baseline = 12.0
    angle = 30.963756532
    plane = LaserPlane.from_baseline_and_angle(baseline, angle, axis="y")
    point, valid = plane.intersect_rays(np.array([0.0, 0.0, 1.0]))
    assert valid
    assert np.isclose(point[2], baseline / np.tan(np.deg2rad(angle)), atol=1e-8)
    assert plane.normal[1] < 0 and plane.normal[2] < 0


def test_distorted_camera_project_unproject_round_trip():
    cam = CameraIntrinsics(
        width=1600,
        height=1200,
        fx=1200.0,
        fy=1190.0,
        cx=800.0,
        cy=600.0,
        dist_coeffs=(-0.12, 0.03, 0.001, -0.002, 0.0),
    )
    uv = np.array([[150.0, 200.0], [800.0, 600.0], [1450.0, 1100.0]])
    rays = cam.unproject(uv)
    reconstructed_uv = cam.project(rays)
    assert np.allclose(reconstructed_uv, uv, atol=1e-5)


def test_capture_config_drives_nominal_laser_geometry():
    from ioscan.capture.logger import CaptureConfig, _build_calibration

    cfg = CaptureConfig(laser_baseline_mm=15.0, laser_tilt_deg=40.0)
    _camera, laser, nominal = _build_calibration(cfg, None, None)
    point, valid = laser.intersect_rays(np.array([0.0, 0.0, 1.0]))
    assert nominal is True
    assert valid
    assert np.isclose(point[2], 15.0 / np.tan(np.deg2rad(40.0)), atol=1e-8)


def test_nominal_camera_model_uses_28mm_lens_config_not_old_8mm_hardcode():
    from ioscan.capture.logger import CaptureConfig, _build_calibration

    camera, _laser, nominal = _build_calibration(CaptureConfig(), None, None)
    assert nominal is True
    assert camera.width == 1600 and camera.height == 1200
    assert np.isclose(camera.fx, 2.8 / 0.003, atol=0.1)
    assert np.isclose(camera.fy, 2.8 / 0.003, atol=0.1)

def test_updated_nominal_geometry_intersects_at_about_102mm():
    plane = LaserPlane.from_baseline_and_angle(18.0, 10.0)
    point, valid = plane.intersect_rays(np.array([0.0, 0.0, 1.0]))
    assert valid
    assert np.isclose(point[2], 18.0 / np.tan(np.deg2rad(10.0)), atol=1e-8)
