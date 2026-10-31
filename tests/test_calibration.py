from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from ioscan.calibration.calibrate import (
    fit_laser_plane_from_points,
    intersect_rays_with_board_plane,
    save_calibration_bundle,
)
from ioscan.calibration.models import CameraIntrinsics, LaserPlane


def test_fit_laser_plane_svd_recovers_exact_plane():
    gt_plane = LaserPlane.from_baseline_and_angle(baseline_mm=12.0, angle_deg=30.963756532)
    rng = np.random.default_rng(123)

    # Generate 500 points on the plane: n_x * x + n_y * y + n_z * z + d = 0
    x = rng.uniform(-8.0, 8.0, size=500)
    z = rng.uniform(15.0, 30.0, size=500)
    y = -(gt_plane.normal[0] * x + gt_plane.normal[2] * z + gt_plane.d) / gt_plane.normal[1]
    pts = np.column_stack([x, y, z])

    fitted, rms_mm = fit_laser_plane_from_points(pts)
    assert rms_mm < 1e-9
    assert np.allclose(fitted.normal, gt_plane.normal, atol=1e-6)
    assert np.isclose(fitted.d, gt_plane.d, atol=1e-6)


def test_ray_board_intersection_and_bundle_roundtrip(tmp_path: Path):
    cam = CameraIntrinsics(width=1600, height=1300, fx=2666.7, fy=2666.7, cx=800.0, cy=650.0)
    R_cb, _ = cv2.Rodrigues(np.array([0.1, -0.15, 0.0], dtype=np.float64))
    t_cb = np.array([0.0, 0.0, 22.0], dtype=np.float64)

    uv = np.array([[800.0, 650.0], [600.0, 500.0], [1000.0, 750.0]], dtype=np.float64)
    rays = cam.unproject(uv)
    pts_cam, valid = intersect_rays_with_board_plane(rays, R_cb, t_cb)

    assert np.all(valid)
    # Check that all intersected points lie on the board plane: n_board . (P - t_cb) == 0
    n_board = R_cb[:, 2]
    residuals = (pts_cam - t_cb) @ n_board
    assert np.allclose(residuals, 0.0, atol=1e-7)

    laser = LaserPlane.from_baseline_and_angle(12.0, 30.96)
    cal_file = save_calibration_bundle(tmp_path / "calibration.json", cam, laser)
    assert cal_file.exists()