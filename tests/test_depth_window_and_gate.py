"""Regression tests for the 'broken point cloud despite >100k points' failure."""
import numpy as np
import pytest

from ioscan.calibration.models import CameraIntrinsics, LaserPlane
from ioscan.pipeline.pose import PoseConfig, PoseTracker
from ioscan.pipeline.quality import assess_scan_geometry
from ioscan.pipeline.triangulate import resolve_depth_window, roi_depth_coverage, triangulate_frame

from ._synth_session import USER_ROI, make_session

CAM = CameraIntrinsics(1600, 1200, 933.3, 933.3, 800.0, 600.0)
LASER = LaserPlane.from_baseline_and_angle(18.0, 10.0, "y")


def test_saved_roi_is_incompatible_with_default_depth_window():
    cov = roi_depth_coverage(CAM, LASER, USER_ROI, (75.0, 125.0))
    assert cov["fraction_in_range"] < 0.01
    assert cov["columns_reachable"] < 0.15 * cov["columns_total"]
    lo, hi = cov["suggested_z_range_mm"]
    assert lo < 25.0 and hi > 70.0


def test_resolve_depth_window_adjusts_or_raises():
    z, info, msg = resolve_depth_window(CAM, LASER, USER_ROI, (75.0, 125.0))
    assert info["auto_adjusted"] and msg and z[0] < 25.0
    with pytest.raises(RuntimeError, match="--z-min"):
        resolve_depth_window(CAM, LASER, USER_ROI, (75.0, 125.0), auto_adjust=False)
    # A window that matches the ROI is left untouched; no ROI keeps the request as-is.
    z2, _, msg2 = resolve_depth_window(CAM, LASER, USER_ROI, z)
    assert msg2 is None and z2 == z
    assert resolve_depth_window(CAM, LASER, None, (75.0, 125.0))[0] == (75.0, 125.0)


def test_roi_derived_window_recovers_the_surface(tmp_path):
    from ioscan.session.loader import load_frame_records, read_grayscale_frame
    from ._synth_session import surface_height
    make_session(tmp_path, n_frames=4, roi_points=USER_ROI)
    rec = load_frame_records(tmp_path)
    on, off = read_grayscale_frame(rec[1]["path"]), read_grayscale_frame(rec[0]["path"])
    old, _, _ = triangulate_frame(on, off, CAM, LASER, z_range=(75.0, 125.0), roi_polygon=USER_ROI)
    z, _, _ = resolve_depth_window(CAM, LASER, USER_ROI, (75.0, 125.0))
    new, _, _ = triangulate_frame(on, off, CAM, LASER, z_range=z, roi_polygon=USER_ROI)
    assert len(old) < 20 and len(new) > 400
    # Camera is at y=0.02*5 mm for frame 1; surface heights must match the ground truth
    yc = 5.0 * 0.02
    assert np.median(np.abs(new[:, 2] - surface_height(new[:, 0], new[:, 1] + yc))) < 0.05


def _grid_mesh(nx, ny, sx, sy):
    xs, ys = np.meshgrid(np.linspace(0, sx, nx), np.linspace(0, sy, ny))
    v = np.c_[xs.ravel(), ys.ravel(), np.zeros(nx * ny)]
    idx = np.arange(nx * ny).reshape(ny, nx)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    return v, np.r_[np.c_[a, b, c], np.c_[b, d, c]]


def test_gate_accepts_arch_sized_mesh_and_rejects_smeared_ribbon():
    v, f = _grid_mesh(80, 60, 60.0, 45.0)
    assert assess_scan_geometry(v, f, path_mm=90.0)["ok"]
    v, f = _grid_mesh(300, 30, 205.0, 15.0)          # 205 mm ribbon: >35 mm, passed the old gate
    res = assess_scan_geometry(v, f, path_mm=467.0)
    assert not res["ok"] and any("exceeds any dental arch" in r for r in res["reasons"])


def test_gate_rejects_fragmented_mesh():
    v1, f1 = _grid_mesh(60, 60, 50.0, 40.0)
    v2, f2 = _grid_mesh(60, 60, 50.0, 40.0)
    v2 = v2 + [0, 0, 5]
    V, F = np.vstack([v1, v2]), np.vstack([f1, f2 + len(v1)])
    assert assess_scan_geometry(V, F)["metrics"]["component_count"] == 2
    v3, f3 = _grid_mesh(70, 70, 50.0, 40.0)
    small = [_grid_mesh(10, 10, 2.0, 2.0) for _ in range(12)]
    V = np.vstack([v3] + [s[0] + [0, 0, 3 * (k + 1)] for k, s in enumerate(small)])
    off = len(v3); F = [f3]
    for s in small:
        F.append(s[1] + off); off += len(s[0])
    res = assess_scan_geometry(V, np.vstack(F))
    assert res["metrics"]["component_count"] == 13


def test_imu_to_camera_extrinsic_is_applied_and_validated():
    from scipy.spatial.transform import Rotation
    Rz = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    q = np.array([1.0, 0.0, 0.0, 0.0])
    base = PoseTracker(PoseConfig(), initial_timestamp_s=0.0)
    rot = PoseTracker(PoseConfig(imu_to_camera=tuple(map(tuple, Rz))), initial_timestamp_s=0.0)
    T0, T1 = base.update(1, 1.0, q), rot.update(1, 1.0, q)
    assert np.allclose(T0[:3, 3], [0, 5, 0]) and np.allclose(T1[:3, 3], [-5, 0, 0])
    with pytest.raises(ValueError):
        PoseTracker(PoseConfig(imu_to_camera=((2, 0, 0), (0, 1, 0), (0, 0, 1))))
