"""Virtual laser-line scanner (simulator v1: perfect optics, no noise).

Physical model
- The laser is a point source (apex) emitting a planar fan of rays. We cast the fan against the
  mesh with Open3D: each ray's first hit is a point the laser actually illuminates, so laser
  shadowing is handled for free.
- A lit point is only imaged if the camera can see it: we cast a ray from the camera centre to
  the point and reject it if something is in front (camera occlusion).
- Visible points are projected with the camera model and rendered as a Gaussian line whose
  centre is exact to sub-pixel precision. Intensity falls with the incidence angle.

Frames: camera frame is OpenCV (x right, y down, z forward), mm. World frame is the mesh frame.
T_wc is the camera-to-world transform.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d
import trimesh

from ioscan.calibration.models import CameraIntrinsics, LaserPlane


@dataclass
class LaserSpec:
    """Line laser: apex offset below the camera (+y), fan lies in the plane spanned by x and the
    tilted central direction."""

    baseline_mm: float = 12.0
    tilt_deg: float = 30.96
    fan_half_angle_deg: float = 25.0

    @property
    def apex(self) -> np.ndarray:
        return np.array([0.0, self.baseline_mm, 0.0])

    @property
    def axis(self) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0])

    @property
    def central_dir(self) -> np.ndarray:
        t = np.radians(self.tilt_deg)
        return np.array([0.0, -np.sin(t), np.cos(t)])

    @property
    def plane(self) -> LaserPlane:
        n = np.cross(self.axis, self.central_dir)
        return LaserPlane(n=n, d=-float(n @ self.apex))

    def fan_directions(self, n_rays: int) -> np.ndarray:
        a = np.radians(np.linspace(-self.fan_half_angle_deg, self.fan_half_angle_deg, n_rays))
        return np.cos(a)[:, None] * self.central_dir[None, :] + np.sin(a)[:, None] * self.axis[None, :]

    @classmethod
    def from_config(cls, cfg: dict) -> "LaserSpec":
        return cls(
            baseline_mm=float(cfg["baseline_mm"]),
            tilt_deg=float(cfg["tilt_deg"]),
            fan_half_angle_deg=float(cfg["fan_half_angle_deg"]),
        )


@dataclass
class RenderParams:
    fan_rays: int = 5000
    laser_peak_dn: float = 600.0
    ambient_dn: float = 40.0
    max_dn: int = 1023
    max_gap_mm: float = 0.15
    occlusion_tol_mm: float = 0.02
    line_sigma_px: float = 1.5


def splat_gaussian(uv: np.ndarray, coeff: np.ndarray, shape: tuple[int, int], sigma: float) -> np.ndarray:
    """Accumulate coeff_i * exp(-d^2 / 2 sigma^2) around sub-pixel positions uv (pixel centres at
    integer coordinates)."""
    H, W = shape
    r = int(np.ceil(4.0 * sigma))
    off = np.arange(-r, r + 1)
    cu = np.rint(uv[:, 0]).astype(np.int64)
    cv = np.rint(uv[:, 1]).astype(np.int64)
    px = cu[:, None, None] + off[None, None, :]  # (M,1,K)
    py = cv[:, None, None] + off[None, :, None]  # (M,K,1)
    d2 = (px - uv[:, 0, None, None]) ** 2 + (py - uv[:, 1, None, None]) ** 2  # (M,K,K)
    val = coeff[:, None, None] * np.exp(-d2 / (2.0 * sigma**2))
    inside = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    flat = py * W + px
    img = np.bincount(flat[inside], weights=val[inside], minlength=H * W)
    return img.reshape(H, W)


class VirtualScanner:
    def __init__(
        self,
        mesh: trimesh.Trimesh,
        camera: CameraIntrinsics,
        laser: LaserSpec,
        params: RenderParams | None = None,
    ) -> None:
        self.camera = camera
        self.laser = laser
        self.p = params or RenderParams()
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(
            o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
            o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32)),
        )
        self._dirs_c = laser.fan_directions(self.p.fan_rays)  # (N,3) camera frame

    def _quantize(self, img: np.ndarray) -> np.ndarray:
        return np.clip(np.rint(img), 0, self.p.max_dn).astype(np.uint16)

    def _cast(self, origins: np.ndarray, dirs: np.ndarray) -> dict[str, np.ndarray]:
        rays = np.hstack([origins, dirs]).astype(np.float32)
        ans = self.scene.cast_rays(o3d.core.Tensor(rays))
        return {k: v.numpy() for k, v in ans.items()}

    def render_frame(self, T_wc: np.ndarray, laser_on: bool) -> tuple[np.ndarray, dict | None]:
        """Render one frame. Returns (uint16 image, ground-truth profile or None).

        The profile dict has uv (M,2) pixel positions and P_c (M,3) camera-frame points of the
        true visible laser line, both in fan order.
        """
        cam, p = self.camera, self.p
        H, W = cam.height, cam.width
        img = np.full((H, W), p.ambient_dn, dtype=np.float64)
        if not laser_on:
            return self._quantize(img), None

        R, t = T_wc[:3, :3], T_wc[:3, 3]
        N = len(self._dirs_c)

        # 1) laser fan against the mesh
        apex_w = R @ self.laser.apex + t
        dirs_w = self._dirs_c @ R.T
        ans = self._cast(np.broadcast_to(apex_w, (N, 3)), dirs_w)
        t_hit = ans["t_hit"]
        hit = np.isfinite(t_hit)
        P_w = np.full((N, 3), np.nan)
        P_w[hit] = apex_w + t_hit[hit, None] * dirs_w[hit]

        # 2) camera visibility of the lit points
        v = P_w - t
        dist = np.linalg.norm(v, axis=1)
        dirs_cam = np.tile(np.array([0.0, 0.0, 1.0]), (N, 1))
        dirs_cam[hit] = v[hit] / dist[hit, None]
        t2 = self._cast(np.broadcast_to(t, (N, 3)), dirs_cam)["t_hit"]
        with np.errstate(invalid="ignore"):
            visible = hit & (t2 >= dist - p.occlusion_tol_mm)
        P_c = (P_w - t) @ R  # R^T (P - t) in row form
        with np.errstate(invalid="ignore"):
            valid = visible & (P_c[:, 2] > 1e-3)

        idx = np.flatnonzero(valid)
        if idx.size == 0:
            return self._quantize(img), {"uv": np.zeros((0, 2)), "P_c": np.zeros((0, 3))}

        uv = np.full((N, 2), np.nan)
        uv[idx] = cam.project(P_c[idx])

        # 3) incidence-angle intensity
        normals = ans["primitive_normals"].astype(np.float64)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        cos_inc = np.abs(np.einsum("ij,ij->i", normals, dirs_w))
        amp = p.laser_peak_dn * (0.25 + 0.75 * cos_inc)

        # 4) render the line as continuous segments. Neighbouring fan samples that are close in
        #    3D are linked; each link is re-sampled finer than the line width so that surfaces
        #    nearly parallel to the laser (sparse samples) still render as a solid line, and
        #    intensity does not depend on sample density. Isolated samples get a small weight.
        with np.errstate(invalid="ignore"):
            gap3 = np.linalg.norm(P_c[1:] - P_c[:-1], axis=1)
            gap_px = np.linalg.norm(uv[1:] - uv[:-1], axis=1)
            link = valid[:-1] & valid[1:] & (gap3 < p.max_gap_mm)
        segs = np.flatnonzero(link)
        has_link = np.zeros(N, dtype=bool)
        has_link[:-1] |= link
        has_link[1:] |= link
        iso = np.flatnonzero(valid & ~has_link)

        pts = [uv[iso]]
        wts = [np.full(len(iso), 0.5)]
        amps = [amp[iso]]
        if segs.size:
            n_sub = np.maximum(1, np.ceil(gap_px[segs] / 0.5)).astype(np.int64)
            seg_id = np.repeat(segs, n_sub)
            first = np.cumsum(n_sub) - n_sub
            k = np.arange(int(n_sub.sum())) - np.repeat(first, n_sub)
            f = (k + 0.5) / np.repeat(n_sub, n_sub)  # midpoints of the sub-intervals
            pts.append(uv[seg_id] * (1.0 - f[:, None]) + uv[seg_id + 1] * f[:, None])
            amps.append(amp[seg_id] * (1.0 - f) + amp[seg_id + 1] * f)
            wts.append(np.repeat(gap_px[segs] / n_sub, n_sub))
        pts_a, w_a, amp_a = np.vstack(pts), np.concatenate(wts), np.concatenate(amps)

        sigma = p.line_sigma_px
        margin = 4.0 * sigma + 1.0
        u, vv = uv[:, 0], uv[:, 1]
        near = (
            (pts_a[:, 0] > -margin) & (pts_a[:, 0] < W + margin)
            & (pts_a[:, 1] > -margin) & (pts_a[:, 1] < H + margin)
        )
        if near.any():
            coeff = w_a[near] * amp_a[near] / (np.sqrt(2.0 * np.pi) * sigma)
            img += splat_gaussian(pts_a[near], coeff, (H, W), sigma)

        with np.errstate(invalid="ignore"):
            inside = valid & (u >= 0) & (u <= W - 1) & (vv >= 0) & (vv <= H - 1)
        gt_idx = np.flatnonzero(inside)
        profile = {"uv": uv[gt_idx], "P_c": P_c[gt_idx]}
        return self._quantize(img), profile
