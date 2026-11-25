"""Synthetic full dental arch scanned as several overlapping straight passes (no Open3D needed).

Each pass is what a sweep reconstruction returns: surface points in the pass's own camera frame,
with the errors a real pass has (small wand tilt, a few % optical-flow scale error along the scan
axis, depth noise). Used to validate multi-pass stitching against ground truth.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

BASE_Z = 84.0
# mesiodistal tooth lengths from the midline to the last molar (mm), one side
TOOTH_LEN = [5.4, 5.9, 6.9, 7.0, 7.1, 10.8, 10.3]
TOOTH_W = [5.8, 6.0, 7.0, 7.6, 8.0, 10.2, 9.8]
TOOTH_H = [6.5, 6.8, 7.5, 7.2, 7.0, 6.5, 6.2]


def arch_centerline(n: int = 4000) -> np.ndarray:
    """Parabolic horseshoe, incisors at +Y, opening toward -Y (mm)."""
    total = 2 * sum(TOOTH_LEN)
    x = np.linspace(-28, 28, 20000)
    y = 40.0 - 0.06 * x ** 2
    seg = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0], np.cumsum(seg)])
    s -= s[-1] / 2.0
    keep = np.abs(s) <= total / 2 + 3
    xs, ys, ss = x[keep], y[keep], s[keep]
    idx = np.linspace(0, len(xs) - 1, n).astype(int)
    return np.column_stack([xs[idx], ys[idx], ss[idx]])


def arch_height(xy: np.ndarray, line: np.ndarray | None = None) -> np.ndarray:
    line = arch_centerline() if line is None else line
    tree = cKDTree(line[:, :2])
    d, i = tree.query(xy)
    s = line[i, 2]
    # signed offset: outside (buccal) positive
    nxt = np.clip(i + 1, 0, len(line) - 1)
    prv = np.clip(i - 1, 0, len(line) - 1)
    tang = line[nxt, :2] - line[prv, :2]
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12
    rel = xy - line[i, :2]
    nrm = np.column_stack([tang[:, 1], -tang[:, 0]])
    off = np.einsum("ij,ij->i", rel, nrm)
    h = 2.0 * np.exp(-(off / 7.0) ** 2)                      # gingiva ridge
    centers, lens, ws, hs = [], [], [], []
    acc = 0.0
    for L, W, H in zip(TOOTH_LEN, TOOTH_W, TOOTH_H):
        for side in (1, -1):
            centers.append(side * (acc + L / 2)); lens.append(L); ws.append(W); hs.append(H)
        acc += L
    for c, L, W, H in zip(centers, lens, ws, hs):
        u = (s - c) / (L / 2 * 0.92)
        v = off / (W / 2)
        r = (np.abs(u) ** 4 + np.abs(v) ** 4) ** 0.25
        crown = H * np.clip(1 - r ** 6, 0, 1) ** 0.3
        cusp = 0.9 * (np.exp(-((u - 0.45) ** 2 + (v - 0.45) ** 2) / 0.08) + np.exp(-((u + 0.45) ** 2 + (v - 0.45) ** 2) / 0.08)
                      + np.exp(-((u - 0.45) ** 2 + (v + 0.45) ** 2) / 0.08) + np.exp(-((u + 0.45) ** 2 + (v + 0.45) ** 2) / 0.08))
        fossa = 0.7 * np.exp(-(v / 0.25) ** 2) * (r < 0.8)
        h = np.maximum(h, np.where(r < 1, crown + cusp - fossa, 0))
    return h


def make_passes(n_passes: int = 6, seed: int = 0, length: float = 30.0, width: float = 30.0,
                scale_err: float = 0.02, tilt_deg: float = 3.0, noise: float = 0.03, step: float = 0.15):
    """Return (list of (points_cam (N,3), R_wc, t_wc)), ground-truth world points."""
    rng = np.random.default_rng(seed)
    line = arch_centerline()
    total = line[-1, 2] - line[0, 2]
    centers = np.linspace(line[0, 2] + length * 0.4, line[-1, 2] - length * 0.4, n_passes)
    passes = []
    for k, sc in enumerate(centers):
        i = int(np.argmin(np.abs(line[:, 2] - sc)))
        j = min(i + 20, len(line) - 1); i0 = max(i - 20, 0)
        tang = line[j, :2] - line[i0, :2]; tang /= np.linalg.norm(tang)
        yaw = np.arctan2(tang[1], tang[0]) - np.pi / 2          # camera +Y along the tangent
        center = line[i, :2]
        # footprint grid in the pass frame
        u, v = np.meshgrid(np.arange(-width / 2, width / 2, step), np.arange(-length / 2, length / 2, step))
        Rz2 = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        xy = np.column_stack([u.ravel(), v.ravel()]) @ Rz2.T + center
        hw = arch_height(xy, line)
        Pw = np.column_stack([xy, BASE_Z - hw])                  # world: camera looks along +Z
        tilt = Rotation.from_euler("xy", rng.normal(0, tilt_deg, 2), degrees=True).as_matrix()
        R = Rotation.from_euler("z", yaw).as_matrix() @ tilt
        t = np.array([center[0], center[1], 0.0]) + rng.normal(0, 1.0, 3)
        Pc = (Pw - t) @ R                                         # R^T (Pw - t)
        Pc[:, 1] *= 1.0 + rng.uniform(-scale_err, scale_err)      # flow scale error along the scan
        Pc[:, 2] += rng.normal(0, noise, len(Pc))
        passes.append((Pc, R, t))
    gx, gy = np.meshgrid(np.arange(-40, 40, 0.2), np.arange(-15, 50, 0.2))
    gxy = np.column_stack([gx.ravel(), gy.ravel()])
    gh = arch_height(gxy, line)
    gt = np.column_stack([gxy, BASE_Z - gh])
    return passes, gt, line
