from __future__ import annotations
import numpy as np
import pytest

from ioscan.calibration.models import CameraIntrinsics, LaserPlane
from ioscan.pipeline.ridge import RidgeConfig
from ioscan.pipeline.triangulate import triangulate_frame


def _render_synthetic_flat_plane(
    cam: CameraIntrinsics,
    laser: LaserPlane,
    z_depth_mm: float = 22.0,
    line_sigma_px: float = 1.4,
    peak_intensity: float = 1200.0,
    ambient: float = 80.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Render an exact sub-pixel Gaussian laser line for a flat plane at z = z_depth_mm."""
    xs = np.linspace(-8.0, 8.0, 3200)
    nx, ny, nz = laser.n  # Fixed attribute name from .normal to .n
    ys = -(nx * xs + nz * z_depth_mm + laser.d) / ny
    zs = np.full_like(xs, z_depth_mm)

    pts_cam = np.column_stack([xs, ys, zs])
    uv = cam.project(pts_cam)

    in_bounds = (uv[:, 0] >= 0) & (uv[:, 0] < cam.width) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height)
    uv = uv[in_bounds]

    order = np.argsort(uv[:, 0])
    u_sorted, v_sorted = uv[order, 0], uv[order, 1]

    cols = np.arange(cam.width, dtype=np.float64)
    valid_cols = (cols >= u_sorted[0]) & (cols <= u_sorted[-1])
    u_int = np.where(valid_cols)[0]
    v_center = np.interp(cols[valid_cols], u_sorted, v_sorted)

    off_frame = np.full((cam.height, cam.width), ambient, dtype=np.uint16)
    on_frame = off_frame.astype(np.float64).copy()

    rows = np.arange(cam.height, dtype=np.float64)[:, None]
    gauss = peak_intensity * np.exp(-0.5 * ((rows - v_center[None, :]) / line_sigma_px) ** 2)
    on_frame[:, u_int] += gauss

    return np.clip(np.round(on_frame), 0, 65535).astype(np.uint16), off_frame


@pytest.mark.parametrize("method,tol_mm", [("cog", 0.005), ("gaussian3", 0.005), ("steger", 0.015)])
def test_ridge_methods_recover_flat_plane_depth(method: str, tol_mm: float):
    cam = CameraIntrinsics(width=1600, height=1200, fx=2666.7, fy=2666.7, cx=800.0, cy=600.0)
    
    angle_rad = np.deg2rad(30.963756532)
    n_vec = [0.0, float(np.sin(angle_rad)), float(np.cos(angle_rad))]
    laser = LaserPlane.from_dict({"n": n_vec, "d": -20.0})
    
    true_z = 22.0
    on_frame, off_frame = _render_synthetic_flat_plane(cam, laser, z_depth_mm=true_z)

    pts_cam, _, _ = triangulate_frame(
        on_frame, off_frame, cam, laser, ridge_cfg=RidgeConfig(method=method)
    )
    assert pts_cam.shape[0] > 800
    rms_mm = float(np.sqrt(np.mean((pts_cam[:, 2] - true_z) ** 2)))
    assert rms_mm < tol_mm

def test_triangulate_accepts_color_jpeg_decodes_as_bgr():
    """JPEG decoders may return three-channel BGR for a monochrome sensor."""
    import cv2

    cam = CameraIntrinsics(width=1600, height=1200, fx=2666.7, fy=2666.7, cx=800.0, cy=600.0)
    angle_rad = np.deg2rad(30.963756532)
    laser = LaserPlane.from_dict({"n": [0.0, float(np.sin(angle_rad)), float(np.cos(angle_rad))], "d": -20.0})
    on_gray, off_gray = _render_synthetic_flat_plane(cam, laser, z_depth_mm=22.0)
    on_bgr = cv2.cvtColor(on_gray, cv2.COLOR_GRAY2BGR)
    off_bgr = cv2.cvtColor(off_gray, cv2.COLOR_GRAY2BGR)

    gray_pts, gray_uv, _ = triangulate_frame(on_gray, off_gray, cam, laser, z_range=(10, 45))
    bgr_pts, bgr_uv, _ = triangulate_frame(on_bgr, off_bgr, cam, laser, z_range=(10, 45))
    assert bgr_pts.shape == gray_pts.shape
    assert np.allclose(bgr_pts, gray_pts)
    assert np.allclose(bgr_uv, gray_uv)


def test_laser_plane_depth_bounds_detects_depth_mismatch():
    from ioscan.pipeline.triangulate import laser_plane_depth_bounds

    cam = CameraIntrinsics(width=1600, height=1200, fx=2666.7, fy=2666.7, cx=800.0, cy=600.0)
    laser = LaserPlane.from_baseline_and_angle(baseline_mm=25.0, angle_deg=6.0)
    bounds = laser_plane_depth_bounds(cam, laser)
    assert bounds is not None
    assert bounds[0] > 45.0
