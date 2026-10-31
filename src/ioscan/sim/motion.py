"""Scan trajectory over a dental arch and the ideal IMU stream.

World frame = canonical mesh frame: arch lies in the XY plane, +Z points in the occlusal
direction (from gingiva toward the cusps), origin at the tooth-vertex centroid.

v1 trajectory: a single occlusal pass. The camera looks straight down (-Z) from a fixed height
above the cusp tips and follows the arch centreline. The camera y axis points along the
direction of travel, so the laser line (along camera x) is perpendicular to the motion.
Buccal/lingual walls are not seen from above; multi-pass trajectories come later.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp


def canonicalize(
    mesh: trimesh.Trimesh, labels: np.ndarray | None
) -> tuple[trimesh.Trimesh, np.ndarray, np.ndarray]:
    """Rotate/translate the mesh so the arch lies in XY with +Z occlusal.

    Returns (mesh_canonical, R, mu) with canonical = (original - mu) @ R.
    """
    V = np.asarray(mesh.vertices, dtype=np.float64)
    has_labels = labels is not None and len(labels) == len(V) and np.any(labels > 0)
    teeth = V[labels > 0] if has_labels else V
    mu = teeth.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov((teeth - mu).T))  # ascending variance
    x = evecs[:, 2]
    z = evecs[:, 0]
    if has_labels and np.any(labels == 0):
        # occlusal direction points from the gingiva toward the teeth
        if (teeth.mean(axis=0) - V[labels == 0].mean(axis=0)) @ z < 0:
            z = -z
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])  # right-handed: x cross y = z
    out = trimesh.Trimesh(vertices=(V - mu) @ R, faces=mesh.faces, process=False)
    return out, R, mu


def arch_centerline(xy: np.ndarray, n_bins: int = 120, poly_deg: int = 6) -> np.ndarray:
    """Centreline of a horseshoe-shaped point set in 2D, ordered along the arch (n_bins, 2).

    Angular bins around the centroid; median radius per bin; the cut is placed in the largest
    angular gap (the open end of the horseshoe). The radius profile is then fitted with a
    low-order polynomial: real arches are smooth, while the raw median jumps at tooth contacts,
    which would make the simulated hand (and its heading) swing unrealistically.
    """
    c = xy.mean(axis=0)
    d = xy - c
    phi = np.arctan2(d[:, 1], d[:, 0])
    r = np.hypot(d[:, 0], d[:, 1])

    s = np.sort(phi)
    gaps = np.diff(np.r_[s, s[0] + 2.0 * np.pi])
    k = int(np.argmax(gaps))
    cut = s[k] + gaps[k] / 2.0
    phi_s = np.mod(phi - cut, 2.0 * np.pi)

    edges = np.linspace(phi_s.min(), phi_s.max(), n_bins + 1)
    bin_idx = np.clip(np.digitize(phi_s, edges[1:-1]), 0, n_bins - 1)
    r_med = np.full(n_bins, np.nan)
    for i in range(n_bins):
        m = bin_idx == i
        if np.any(m):
            r_med[i] = np.median(r[m])
    centers = 0.5 * (edges[:-1] + edges[1:])
    good = np.isfinite(r_med)
    r_med = np.interp(centers, centers[good], r_med[good])
    x = (centers - centers.mean()) / (0.5 * np.ptp(centers))
    r_med = np.polyval(np.polyfit(x, r_med, poly_deg), x)

    ang = centers + cut
    return c + r_med[:, None] * np.column_stack([np.cos(ang), np.sin(ang)])


def make_trajectory(
    mesh: trimesh.Trimesh,
    labels: np.ndarray | None,
    working_distance_mm: float,
    speed_mm_s: float,
    fps: float,
    tremor_mm: float = 0.0,
    tremor_hz: float = 8.0,
    seed: int = 0,
    max_frames: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Camera-to-world poses T_wc (N,4,4), one per frame, plus an info dict."""
    V = np.asarray(mesh.vertices)
    has_labels = labels is not None and len(labels) == len(V) and np.any(labels > 0)
    teeth = V[labels > 0] if has_labels else V

    center = arch_centerline(teeth[:, :2])
    seg = np.linalg.norm(np.diff(center, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    spline = CubicSpline(s, center, axis=0)

    ds = speed_mm_s / fps
    n = int(np.floor(s[-1] / ds)) + 1
    if max_frames is not None:
        n = min(n, int(max_frames))
    sk = np.arange(n) * ds
    pos_xy = spline(sk)
    tan = spline(sk, 1)
    tan /= np.linalg.norm(tan, axis=1, keepdims=True)

    z_cam = float(np.percentile(teeth[:, 2], 95)) + working_distance_mm
    pos = np.column_stack([pos_xy, np.full(n, z_cam)])

    if tremor_mm > 0.0:
        rng = np.random.default_rng(seed)
        tt = np.arange(n) / fps
        phase = rng.uniform(0.0, 2.0 * np.pi, size=3)
        pos = pos + tremor_mm * np.sin(2.0 * np.pi * tremor_hz * tt[:, None] + phase[None, :])

    z_c = np.tile(np.array([0.0, 0.0, -1.0]), (n, 1))
    y_c = np.column_stack([tan[:, 0], tan[:, 1], np.zeros(n)])
    x_c = np.cross(y_c, z_c)  # y cross z = x for a right-handed (x right, y down, z forward) frame

    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, 0], T[:, :3, 1], T[:, :3, 2], T[:, :3, 3] = x_c, y_c, z_c, pos
    info = {
        "centerline": center,
        "path_len_mm": float(s[-1]),
        "z_cam": z_cam,
        "frames": n,
        "duration_s": n / fps,
    }
    return T, info


def ideal_imu(t_ns: np.ndarray, R_wc: np.ndarray, rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Noise-free IMU quaternions (w,x,y,z) at rate_hz, slerped between true frame poses."""
    t_ns = np.asarray(t_ns, dtype=np.int64).reshape(-1)
    rotations = np.asarray(R_wc, dtype=np.float64).reshape(-1, 3, 3)
    if len(t_ns) != len(rotations) or len(t_ns) == 0:
        raise ValueError("IMU generation requires matching, non-empty timestamps and rotations")
    if not np.isfinite(rate_hz) or rate_hz <= 0.0:
        raise ValueError("IMU sample rate must be finite and positive")
    t_s = t_ns.astype(np.float64) * 1e-9
    if len(t_s) == 1:
        q = Rotation.from_matrix(rotations).as_quat()[0]
        return t_ns.copy(), np.array([[q[3], q[0], q[1], q[2]]], dtype=np.float64)

    step = 1.0 / rate_hz
    t_imu_s = np.arange(t_s[0], t_s[-1], step, dtype=np.float64)
    if t_imu_s.size == 0:
        t_imu_s = np.array([t_s[0], t_s[-1]], dtype=np.float64)
    elif t_imu_s[-1] < t_s[-1] - 1e-12:
        # Include the last camera frame so the final orientation does not need clamping.
        t_imu_s = np.r_[t_imu_s, t_s[-1]]
    slerp = Slerp(t_s, Rotation.from_matrix(rotations))
    q_xyzw = slerp(t_imu_s).as_quat()
    q_wxyz = np.column_stack([q_xyzw[:, 3], q_xyzw[:, :3]])
    return np.rint(t_imu_s * 1e9).astype(np.int64), q_wxyz
