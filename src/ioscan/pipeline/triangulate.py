from __future__ import annotations

import numpy as np

from ioscan.calibration.models import CameraIntrinsics, LaserPlane
from ioscan.pipeline.ridge import RidgeConfig, extract_ridge_subpixel
from ioscan.pipeline.subtraction import subtract_ambient


def triangulate_frame(
    on_frame: np.ndarray,
    off_frame: np.ndarray | None,
    camera: CameraIntrinsics,
    laser: LaserPlane,
    ridge_cfg: RidgeConfig = RidgeConfig(),
    noise_floor: float = 15.0,
    z_range: tuple[float, float] = (10.0, 45.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subtract ambient light, extract a sub-pixel laser ridge, and triangulate it.

    Returns camera-frame points in millimetres, their image coordinates, and peak weights.
    Lens distortion is handled by ``CameraIntrinsics.unproject`` before ray/plane intersection.
    """
    if on_frame.ndim != 2:
        raise ValueError(f"Expected a single-channel ON image, got shape {on_frame.shape}")
    if off_frame is not None and off_frame.shape != on_frame.shape:
        raise ValueError("ON and OFF images must have identical shapes")
    if on_frame.shape != (camera.height, camera.width):
        raise ValueError(
            f"Frame shape {on_frame.shape} does not match calibrated camera "
            f"({camera.height}, {camera.width})"
        )

    diff = subtract_ambient(on_frame, off_frame, noise_floor=noise_floor)
    u, v, weights = extract_ridge_subpixel(diff, cfg=ridge_cfg)
    if u.size == 0:
        return np.empty((0, 3)), np.empty((0, 2)), np.empty(0)

    uv = np.column_stack([u, v])
    rays = camera.unproject(uv)
    points, valid = laser.intersect_rays(rays)
    z_min, z_max = map(float, z_range)
    if z_min >= z_max:
        raise ValueError("z_range must be an increasing (min_mm, max_mm) pair")
    valid &= np.isfinite(points).all(axis=1)
    valid &= (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    return points[valid], uv[valid], weights[valid]
