"""Validate the simulator math against a flat surface with an analytic answer."""
import numpy as np
import pytest

pytest.importorskip("open3d")
import trimesh

from ioscan.calibration.models import CameraIntrinsics
from ioscan.sim.scanner import LaserSpec, RenderParams, VirtualScanner
from ioscan.utils.config import load_yaml, repo_root

DEPTH = 18.0


def _setup():
    cam = CameraIntrinsics.from_config(load_yaml(repo_root() / "configs" / "camera.yaml"))
    laser = LaserSpec(baseline_mm=12.0, tilt_deg=float(np.degrees(np.arctan(0.6))), fan_half_angle_deg=25.0)
    plane_mesh = trimesh.creation.box(extents=(200.0, 200.0, 2.0))
    plane_mesh.apply_translation((0, 0, -1.0))  # top face at z = 0
    scanner = VirtualScanner(plane_mesh, cam, laser, RenderParams())
    # camera DEPTH mm above the plane, looking down (-Z); camera y along world +Y
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = (-1, 0, 0), (0, 1, 0), (0, 0, -1), (0, 0, DEPTH)
    return cam, laser, scanner, T


def test_gt_points_lie_on_plane_and_at_depth():
    cam, laser, scanner, T = _setup()
    _, prof = scanner.render_frame(T, laser_on=True)
    P = prof["P_c"]
    assert len(P) > 1000
    plane = laser.plane
    assert np.max(np.abs(P @ plane.n + plane.d)) < 1e-3
    assert np.max(np.abs(P[:, 2] - DEPTH)) < 1e-3
    # GT pixels agree with the camera projection
    assert np.max(np.abs(cam.project(P) - prof["uv"])) < 1e-6


def test_line_triangulates_back_to_depth():
    """CoG ridge per column -> ray/plane intersection must recover the plane depth."""
    cam, laser, scanner, T = _setup()
    on, _ = scanner.render_frame(T, laser_on=True)
    off, _ = scanner.render_frame(T, laser_on=False)
    diff = np.maximum(0.0, on.astype(np.float64) - off.astype(np.float64))

    cols = np.arange(60, cam.width - 60, 10)
    v_peak = []
    for u in cols:
        col = diff[:, u]
        v0 = int(np.argmax(col))
        sl = slice(max(0, v0 - 8), v0 + 9)
        v = np.arange(col.size)[sl]
        v_peak.append(np.sum(v * col[sl]) / np.sum(col[sl]))
    uv = np.column_stack([cols, v_peak])
    P = laser.plane.intersect(cam.unproject(uv))
    err = np.abs(P[:, 2] - DEPTH)
    assert err.max() < 0.01, f"max depth error {err.max():.4f} mm"


def test_off_frame_is_flat_ambient():
    cam, laser, scanner, T = _setup()
    off, prof = scanner.render_frame(T, laser_on=False)
    assert prof is None
    assert off.min() == off.max() == 40


def test_wall_parallel_to_laser_renders_as_solid_line():
    """Regression: surfaces nearly parallel to the laser rays get few fan samples and used to
    render as dotted lines. A bump plus a step must give only a few connected line fragments."""
    import cv2

    cam, laser, _, T = _setup()
    base = trimesh.creation.box(extents=(200.0, 200.0, 2.0))
    base.apply_translation((0, 0, -1.0))
    sph = trimesh.creation.icosphere(subdivisions=4, radius=3.0)
    sph.apply_translation((-2.5, 0.0, 2.0))
    step = trimesh.creation.box(extents=(4.0, 12.0, 4.0))
    step.apply_translation((3.0, 0.0, 2.0))
    scanner = VirtualScanner(trimesh.util.concatenate([base, sph, step]), cam, laser, RenderParams())
    T = T.copy()
    T[2, 3] = 20.0
    on, _ = scanner.render_frame(T, laser_on=True)
    lit = (on.astype(np.int32) - 40 > 20).astype(np.uint8)
    n_labels, _ = cv2.connectedComponents(lit, connectivity=8)
    assert n_labels - 1 <= 4, f"line is fragmented into {n_labels - 1} pieces"
