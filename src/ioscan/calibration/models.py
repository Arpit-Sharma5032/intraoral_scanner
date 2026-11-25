from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
import cv2


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    pixel_size_um: float = 3.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Camera width and height must be positive")
        values = np.asarray([self.fx, self.fy, self.cx, self.cy, self.pixel_size_um], dtype=np.float64)
        if not np.isfinite(values).all() or self.fx <= 0.0 or self.fy <= 0.0 or self.pixel_size_um <= 0.0:
            raise ValueError("Camera intrinsics and pixel size must be finite; fx, fy and pixel size must be positive")
        if len(self.dist_coeffs) not in (4, 5):
            raise ValueError("dist_coeffs must contain 4 or 5 OpenCV radial/tangential coefficients")
        if not np.isfinite(np.asarray(self.dist_coeffs, dtype=np.float64)).all():
            raise ValueError("Camera distortion coefficients must be finite")

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def unproject(self, uv: np.ndarray) -> np.ndarray:
        """Convert (N, 2) pixel coordinates [u, v] to unit ray direction vectors (N, 3) in camera frame."""
        uv = np.asarray(uv, dtype=np.float64)
        single = uv.ndim == 1
        if single:
            uv = uv[None, :]
        if any(abs(c) > 1e-12 for c in self.dist_coeffs):
            xy = cv2.undistortPoints(
                uv.reshape(-1, 1, 2), self.K, np.asarray(self.dist_coeffs, dtype=np.float64)
            ).reshape(-1, 2)
            x, y = xy[:, 0], xy[:, 1]
        else:
            x = (uv[:, 0] - self.cx) / self.fx
            y = (uv[:, 1] - self.cy) / self.fy
        z = np.ones_like(x)
        rays = np.stack([x, y, z], axis=-1)
        norms = np.linalg.norm(rays, axis=-1, keepdims=True)
        rays = rays / np.clip(norms, 1e-12, None)
        return rays[0] if single else rays

    def project(self, pts_cam: np.ndarray) -> np.ndarray:
        """Project (N, 3) 3D points in camera frame to (N, 2) pixel coordinates [u, v]."""
        pts = np.asarray(pts_cam, dtype=np.float64)
        single = pts.ndim == 1
        if single:
            pts = pts[None, :]
        if any(abs(c) > 1e-12 for c in self.dist_coeffs):
            uv, _ = cv2.projectPoints(
                pts.reshape(-1, 1, 3),
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                self.K,
                np.asarray(self.dist_coeffs, dtype=np.float64),
            )
            uv = uv.reshape(-1, 2)
        else:
            z_safe = np.where(np.abs(pts[:, 2]) < 1e-9, 1e-9, pts[:, 2])
            u = self.fx * (pts[:, 0] / z_safe) + self.cx
            v = self.fy * (pts[:, 1] / z_safe) + self.cy
            uv = np.stack([u, v], axis=-1)
        return uv[0] if single else uv

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": int(self.width),
            "height": int(self.height),
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "dist_coeffs": list(self.dist_coeffs),
            "pixel_size_um": float(self.pixel_size_um),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CameraIntrinsics":
        dc = d.get("dist_coeffs", d.get("distortion", [0.0, 0.0, 0.0, 0.0, 0.0]))
        if isinstance(dc, dict):
            dc = [dc.get("k1", 0.0), dc.get("k2", 0.0), dist_get(dc, "p1"), dist_get(dc, "p2"), dist_get(dc, "k3")]
        return cls(
            width=int(d["width"]),
            height=int(d["height"]),
            fx=float(d["fx"]),
            fy=float(d["fy"]),
            cx=float(d["cx"]),
            cy=float(d["cy"]),
            dist_coeffs=tuple(float(x) for x in dc[:5]),
            pixel_size_um=float(d.get("pixel_size_um", 3.0)),
        )

    @classmethod
    def from_config(cls, cfg_or_path: dict[str, Any] | str | Path) -> "CameraIntrinsics":
        if isinstance(cfg_or_path, dict):
            raw = cfg_or_path
        else:
            raw = yaml.safe_load(Path(cfg_or_path).read_text(encoding="utf-8"))
        cam = raw.get("camera", raw)
        intr = cam.get("intrinsics", cam)
        sensor = cam.get("sensor", {})
        dist = cam.get("distortion", intr.get("dist_coeffs", [0.0] * 5))
        if isinstance(dist, dict):
            dist = [dist.get("k1", 0.0), dist.get("k2", 0.0), dist.get("p1", 0.0), dist.get("p2", 0.0), dist.get("k3", 0.0)]
        return cls(
            width=int(sensor.get("width_px", intr.get("width", cam.get("width", 1600)))),
            height=int(sensor.get("height_px", intr.get("height", cam.get("height", 1200)))),
            fx=float(intr["fx"]),
            fy=float(intr["fy"]),
            cx=float(intr["cx"]),
            cy=float(intr["cy"]),
            dist_coeffs=tuple(float(x) for x in dist[:5]),
            pixel_size_um=float(sensor.get("pixel_size_um", cam.get("pixel_size_um", 3.0))),
        )


def dist_get(d: dict[str, Any], k: str) -> float:
    return float(d.get(k, 0.0))


@dataclass(frozen=True)
class LaserPlane:
    """
    Laser sheet plane in camera frame:
        normal . P + d = 0, with ||normal|| = 1.
    """
    normal: np.ndarray
    d: float

    def __init__(self, normal: np.ndarray | None = None, d: float = 0.0, *, n: np.ndarray | None = None):
        vec = normal if normal is not None else n
        if vec is None:
            raise ValueError("Specify 'normal' (or 'n') for LaserPlane.")
        arr = np.asarray(vec, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(arr))
        if not np.isfinite(arr).all() or not np.isfinite(float(d)) or not np.isfinite(norm) or norm < 1e-12:
            raise ValueError("LaserPlane normal and offset must be finite, and the normal must be non-zero")
        # Normalize the complete plane equation exactly once. Scaling only the normal (or
        # normalizing it twice) changes the represented physical plane.
        object.__setattr__(self, "normal", arr / norm)
        object.__setattr__(self, "d", float(d) / norm)

    @property
    def n(self) -> np.ndarray:
        return self.normal

    @property
    def plane(self) -> "LaserPlane":
        """Self-reference so both `laser.intersect(...)` and `laser.plane.intersect(...)` work."""
        return self

    @classmethod
    def from_baseline_and_angle(
        cls,
        baseline_mm: float = 18.0,
        angle_deg: float = 10.0,
        axis: str = "y",
    ) -> "LaserPlane":
        if not np.isfinite(baseline_mm) or baseline_mm <= 0.0:
            raise ValueError("Laser baseline must be finite and positive")
        if not np.isfinite(angle_deg) or not 0.0 < float(angle_deg) < 90.0:
            raise ValueError("Laser tilt angle must be between 0 and 90 degrees")
        theta = np.deg2rad(float(angle_deg))
        if axis == "y":
            # Match the simulator geometry: apex=(0,+baseline,0), central ray=(0,-sin,cos).
            normal = np.array([0.0, -np.cos(theta), -np.sin(theta)], dtype=np.float64)
        elif axis == "x":
            normal = np.array([-np.cos(theta), 0.0, -np.sin(theta)], dtype=np.float64)
        else:
            raise ValueError(f"Unsupported baseline axis: {axis}")
        d = float(baseline_mm * np.cos(theta))
        return cls(normal=normal, d=d)

    def intersect_rays(self, rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rays = np.asarray(rays, dtype=np.float64)
        single = rays.ndim == 1
        if single:
            rays = rays[None, :]
        denom = rays @ self.normal
        valid = np.abs(denom) > 1e-9
        t = np.zeros(rays.shape[0], dtype=np.float64)
        t[valid] = -self.d / denom[valid]
        valid = valid & (t > 0.0)
        pts = rays * t[:, None]
        if single:
            return pts[0], bool(valid[0])
        return pts, valid

    def intersect(self, rays: np.ndarray) -> np.ndarray:
        """Intersect camera rays (N, 3) with the laser plane and return 3D points (N, 3)."""
        pts, _ = self.intersect_rays(rays)
        return pts

    def intersect_ray(self, rays: tuple[np.ndarray, np.ndarray] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(rays, tuple):
            origins, dirs = rays
            denom = np.dot(dirs, self.normal)
            valid = np.abs(denom) > 1e-9
            t = np.zeros(len(dirs), dtype=np.float64)
            t[valid] = -(np.dot(origins[valid], self.normal) + self.d) / denom[valid]
            valid = valid & (t > 0.0)
            pts = origins + t[:, None] * dirs
            return pts, valid
        return self.intersect_rays(rays)

    def z_at_ray(self, ray: np.ndarray) -> float:
        pts, valid = self.intersect_rays(ray)
        if np.ndim(valid) == 0:
            return float(pts[2]) if valid else float("nan")
        out = np.full(pts.shape[0], np.nan, dtype=np.float64)
        out[valid] = pts[valid, 2]
        return out

    def signed_distance(self, pts_cam: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_cam, dtype=np.float64)
        return (pts @ self.normal) + self.d

    def perturbed(self, angle_deg: float = 0.0, offset_mm: float = 0.0, axis: str = "x") -> "LaserPlane":
        n = self.normal.copy()
        rad = np.deg2rad(angle_deg)
        if axis == "x":
            c, s = np.cos(rad), np.sin(rad)
            R = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)
        elif axis == "y":
            c, s = np.cos(rad), np.sin(rad)
            R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
        else:
            raise ValueError(f"Unsupported rotation axis: {axis}")

        n_rot = R @ n
        n_rot /= np.linalg.norm(n_rot)
        p0 = -self.d * n + offset_mm * n_rot
        d_new = -float(np.dot(n_rot, p0))
        return LaserPlane(normal=n_rot, d=d_new)

    def to_dict(self) -> dict[str, Any]:
        return {
            "normal": [float(x) for x in self.normal],
            "n": [float(x) for x in self.normal],
            "d": float(self.d),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LaserPlane":
        vec = d.get("normal", d.get("n"))
        if vec is not None and "d" in d:
            return cls(normal=np.asarray(vec, dtype=np.float64), d=float(d["d"]))
        return cls.from_baseline_and_angle(
            baseline_mm=float(d.get("baseline_mm", 18.0)),
            angle_deg=float(d.get("angle_deg", 10.0)),
            axis=str(d.get("axis", "y")),
        )

    @classmethod
    def from_config(cls, cfg_or_path: dict[str, Any] | str | Path) -> "LaserPlane":
        if isinstance(cfg_or_path, dict):
            raw = cfg_or_path
        else:
            raw = yaml.safe_load(Path(cfg_or_path).read_text(encoding="utf-8"))
        las = raw.get("laser", raw.get("laser_plane", raw))
        if "normal" in las or "n" in las:
            return cls.from_dict(las)
        geom = las.get("geometry", las)
        return cls.from_baseline_and_angle(
            baseline_mm=float(geom.get("baseline_mm", 18.0)),
            angle_deg=float(geom.get("angle_deg", geom.get("tilt_deg", geom.get("convergence_angle_deg", 10.0)))),
            axis=str(geom.get("baseline_axis", geom.get("axis", "y"))),
        )