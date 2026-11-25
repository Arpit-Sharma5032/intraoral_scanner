from __future__ import annotations

import cv2
import numpy as np

from ioscan.calibration.models import CameraIntrinsics, LaserPlane
from ioscan.pipeline.ridge import RidgeConfig, extract_ridge_subpixel
from ioscan.pipeline.subtraction import subtract_ambient
from ioscan.utils.roi import rasterize_roi


def _as_grayscale(frame: np.ndarray, label: str) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim == 2:
        return np.ascontiguousarray(frame)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    if frame.ndim == 3 and frame.shape[2] == 1:
        return np.ascontiguousarray(frame[:, :, 0])
    raise ValueError(f"Expected a grayscale or BGR/BGRA {label} image, got shape {frame.shape}")


_ROI_CACHE: dict = {}


def _roi_mask_and_box(shape: tuple[int, int], roi_polygon, pad: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Rasterised ROI mask and its padded bounding box (cached: the ROI is fixed during a session)."""
    key = (tuple(shape), tuple(tuple(map(float, p)) for p in roi_polygon), int(pad))
    hit = _ROI_CACHE.get(key)
    if hit is not None:
        return hit
    mask = rasterize_roi(shape, roi_polygon) > 0
    if mask.any():
        ys, xs = np.nonzero(mask)
        box = (max(int(ys.min()) - pad, 0), min(int(ys.max()) + 1 + pad, shape[0]),
               max(int(xs.min()) - 1, 0), min(int(xs.max()) + 2, shape[1]))
    else:
        box = (0, shape[0], 0, shape[1])
    if len(_ROI_CACHE) > 16:
        _ROI_CACHE.clear()
    _ROI_CACHE[key] = (mask, box)
    return mask, box


def laser_plane_depth_bounds(
    camera: CameraIntrinsics, laser: LaserPlane, samples: int = 41
) -> tuple[float, float] | None:
    """Estimate positive camera-Z intersections of the laser plane over the sensor."""
    us = np.linspace(0.0, camera.width - 1.0, samples)
    vs = np.linspace(0.0, camera.height - 1.0, samples)
    uv = np.asarray([(u, v) for v in vs for u in us], dtype=np.float64)
    pts, valid = laser.intersect_rays(camera.unproject(uv))
    valid = np.asarray(valid, dtype=bool) & np.isfinite(pts).all(axis=1) & (pts[:, 2] > 0.0)
    if not np.any(valid):
        return None
    z = pts[valid, 2]
    return float(np.min(z)), float(np.max(z))


def roi_depth_coverage(
    camera: CameraIntrinsics,
    laser: LaserPlane,
    roi_polygon: list[list[float]] | np.ndarray | None,
    z_range: tuple[float, float],
    stride: int = 4,
) -> dict:
    """How much of the (ROI-masked) sensor can produce points inside ``z_range``.

    The laser line can only ever appear on image pixels, and each pixel maps to exactly one
    depth on the calibrated laser plane. If the operator-drawn ROI maps to depths that lie
    outside ``z_range``, the depth gate silently discards almost every ridge point.
    """
    h, w = camera.height, camera.width
    if roi_polygon is not None:
        mask = rasterize_roi((h, w), roi_polygon) > 0
    else:
        mask = np.ones((h, w), dtype=bool)
    vs, us = np.nonzero(mask[::stride, ::stride])
    vs, us = vs * stride, us * stride
    if us.size == 0:
        return {"roi_pixels": 0, "fraction_in_range": 0.0, "columns_total": 0,
                "columns_reachable": 0, "depth_p02_mm": None, "depth_p98_mm": None,
                "suggested_z_range_mm": None}
    uv = np.column_stack([us, vs]).astype(np.float64)
    pts, valid = laser.intersect_rays(camera.unproject(uv))
    valid = np.asarray(valid, dtype=bool) & np.isfinite(pts).all(axis=1) & (pts[:, 2] > 0.0)
    z = pts[:, 2]
    z_min, z_max = float(z_range[0]), float(z_range[1])
    in_range = valid & (z >= z_min) & (z <= z_max)
    # Columns were sampled every `stride` pixels; report full-resolution column counts.
    full_cols = int(mask.any(axis=0).sum())
    cols_total = full_cols
    cols_ok = min(full_cols, int(np.unique(us[in_range]).size) * stride)
    out = {
        "roi_pixels": int(us.size),
        "fraction_in_range": float(in_range.sum() / us.size),
        "columns_total": cols_total,
        "columns_reachable": cols_ok,
        "depth_p02_mm": None,
        "depth_p98_mm": None,
        "suggested_z_range_mm": None,
    }
    if valid.any():
        lo, hi = np.percentile(z[valid], [2.0, 98.0])
        out["depth_p02_mm"], out["depth_p98_mm"] = float(lo), float(hi)
        margin = 0.10 * (hi - lo) + 1.0
        out["suggested_z_range_mm"] = (
            float(np.floor(max(lo - margin, 0.5) * 2.0) / 2.0),
            float(np.ceil((hi + margin) * 2.0) / 2.0),
        )
    return out


def resolve_depth_window(
    camera: CameraIntrinsics,
    laser: LaserPlane,
    roi_polygon: list[list[float]] | np.ndarray | None,
    z_range: tuple[float, float],
    *,
    min_pixel_fraction: float = 0.25,
    min_column_fraction: float = 0.50,
    auto_adjust: bool = True,
) -> tuple[tuple[float, float], dict, str | None]:
    """Return ``(z_range_to_use, coverage_info, warning_or_None)``.

    With an enabled ROI, the depth window must overlap the depths that the ROI actually maps to
    under the stored laser plane. When it does not, the window is replaced (``auto_adjust``) by
    the ROI-derived range, or a RuntimeError is raised (``auto_adjust=False``). The ROI is the
    operator's statement of where the laser line can be seen; it is not widened here.
    """
    z_range = (float(z_range[0]), float(z_range[1]))
    if roi_polygon is None:
        return z_range, {}, None
    info = roi_depth_coverage(camera, laser, roi_polygon, z_range)
    cols = max(info["columns_total"], 1)
    bad = (
        info["fraction_in_range"] < min_pixel_fraction
        or info["columns_reachable"] / cols < min_column_fraction
    )
    if not bad:
        return z_range, info, None
    sug = info["suggested_z_range_mm"]
    msg = (
        f"Depth window Z={z_range[0]:.1f}..{z_range[1]:.1f} mm only covers "
        f"{100 * info['fraction_in_range']:.2f}% of the ROI pixels ({info['columns_reachable']} of "
        f"{info['columns_total']} ROI columns can yield a point). Under the stored laser plane the ROI "
        f"maps to Z={info['depth_p02_mm']:.1f}..{info['depth_p98_mm']:.1f} mm."
    )
    if sug is None:
        raise RuntimeError(msg + " The laser plane does not intersect the ROI in front of the camera.")
    if not auto_adjust:
        raise RuntimeError(msg + f" Use --z-min {sug[0]} --z-max {sug[1]} or fix the calibration/ROI.")
    info = dict(info, requested_z_range_mm=list(z_range), auto_adjusted=True)
    return sug, info, msg + f" Using the ROI-derived window Z={sug[0]:.1f}..{sug[1]:.1f} mm instead."


def _adaptive_line_image(diff: np.ndarray, inside: np.ndarray | None, cfg: RidgeConfig) -> tuple[np.ndarray, RidgeConfig]:
    """Noise-adaptive laser image.

    A fixed threshold (25 DN) only suits one lighting condition: under bright ambient light the
    line on bright teeth is only 10-20 DN above the (noisier) background and was discarded, leaving
    just the flat base. Here the signed ON-OFF image is smoothed along the line (a matched filter,
    which also averages JPEG and shot noise), its noise is measured robustly, and the threshold is
    set a fixed number of noise sigmas above zero, never above the configured ``min_prominence``.
    """
    from dataclasses import replace
    sm = cv2.GaussianBlur(diff, (0, 0), sigmaX=2.5, sigmaY=1.0)
    ref = sm[inside] if inside is not None and inside.any() else sm.ravel()
    ref = ref[::7]
    med = float(np.median(ref))
    sigma = 1.4826 * float(np.median(np.abs(ref - med)))
    thr = float(np.clip(cfg.adaptive_sigmas * sigma, cfg.adaptive_min, max(cfg.min_prominence, cfg.adaptive_min)))
    sm = sm - med
    sm[sm < 0.5 * thr] = 0.0
    return sm, replace(cfg, min_prominence=thr)


def triangulate_frame(
    on_frame: np.ndarray,
    off_frame: np.ndarray | None,
    camera: CameraIntrinsics,
    laser: LaserPlane,
    ridge_cfg: RidgeConfig = RidgeConfig(),
    noise_floor: float = 15.0,
    z_range: tuple[float, float] = (10.0, 45.0),
    roi_polygon: list[list[float]] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subtract ambient light, extract a sub-pixel laser ridge, and triangulate it.

    Returns camera-frame points in millimetres, their image coordinates, and peak weights.
    Lens distortion is handled by ``CameraIntrinsics.unproject`` before ray/plane intersection.
    """
    # JPEG decoders commonly return BGR (H,W,3), even for monochrome sensors.
    # Normalize both members of the laser pair here as a final defensive boundary.
    on_frame = _as_grayscale(on_frame, "ON")
    if off_frame is not None:
        off_frame = _as_grayscale(off_frame, "OFF")
    if off_frame is not None and off_frame.shape != on_frame.shape:
        raise ValueError("ON and OFF images must have identical shapes")
    if on_frame.shape != (camera.height, camera.width):
        raise ValueError(
            f"Frame shape {on_frame.shape} does not match calibrated camera "
            f"({camera.height}, {camera.width})"
        )

    # Restrict laser-ridge extraction to the operator-drawn camera ROI. Only the ROI's bounding box
    # is processed: full-frame float temporaries (~100 MB per frame in the ridge search) starved the
    # live capture of CPU and memory. Pixels outside the ROI polygon are zeroed after subtraction.
    v0 = u0 = 0
    adaptive = bool(getattr(ridge_cfg, "adaptive_floor", False)) and off_frame is not None
    floor = -1.0e9 if adaptive else noise_floor
    inside = None
    if roi_polygon is not None:
        pad = max(int(ridge_cfg.half_window), 2) + 2
        roi_mask, (y0, y1, x0, x1) = _roi_mask_and_box(on_frame.shape, roi_polygon, pad)
        diff = subtract_ambient(on_frame[y0:y1, x0:x1],
                                None if off_frame is None else off_frame[y0:y1, x0:x1],
                                noise_floor=floor)
        inside = roi_mask[y0:y1, x0:x1]
        v0, u0 = y0, x0
    else:
        diff = subtract_ambient(on_frame, off_frame, noise_floor=floor)
    if adaptive:
        diff, ridge_cfg = _adaptive_line_image(diff, inside, ridge_cfg)
    if inside is not None:
        diff[~inside] = 0.0
    u, v, weights = extract_ridge_subpixel(diff, cfg=ridge_cfg)
    u = u + u0
    v = v + v0
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
