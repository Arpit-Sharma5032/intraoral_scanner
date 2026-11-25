from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass
class RidgeConfig:
    method: str = "cog"           # "cog" | "gaussian3" | "steger"
    min_prominence: float = 30.0  # minimum peak intensity above noise floor
    saturation_limit: float = 65000.0  # reject saturated columns (use ~254 for 8-bit)
    half_window: int = 4          # half-window in rows around peak for CoG
    max_secondary_ratio: float = 0.45  # reject column if a secondary peak > ratio * main peak
    secondary_guard_px: int = 6   # distance outside main peak to check for multipath/reflections
    # Widen the guard band to the measured width of each column's main peak. With a fixed 6 px
    # guard, the shoulder of a broad or haloed laser line was counted as a "secondary peak" and
    # most columns of a real capture were rejected.
    adaptive_guard: bool = True
    steger_sigma: float = 1.2     # Gaussian smoothing sigma for Steger / Hessian
    # Adapt the detection threshold to the noise of each ON-OFF image (see triangulate_frame):
    # min_prominence then acts as the upper limit of the threshold instead of a fixed value.
    adaptive_floor: bool = False
    adaptive_sigmas: float = 6.0
    adaptive_min: float = 7.0


def extract_ridge_subpixel(
    diff_img: np.ndarray,
    cfg: RidgeConfig = RidgeConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract sub-pixel vertical row coordinates v(u) for each horizontal column u.

    Returns
    -------
    u_cols : (M,) float64 array of valid column indices
    v_rows : (M,) float64 array of sub-pixel row centers
    weights : (M,) float64 array of peak intensities
    """
    H, W = diff_img.shape
    img = diff_img.astype(np.float64)

    # 1. Peak finding per column
    peak_rows = np.argmax(img, axis=0)
    cols = np.arange(W)
    peak_vals = img[peak_rows, cols]

    # Basic validity: above prominence, not saturated, not touching image top/bottom border
    margin = max(cfg.half_window, 2)
    valid = (
        (peak_vals >= cfg.min_prominence)
        & (peak_vals < cfg.saturation_limit)
        & (peak_rows >= margin)
        & (peak_rows < H - margin)
    )

    valid_cols = cols[valid]
    if valid_cols.size == 0:
        return np.empty(0), np.empty(0), np.empty(0)

    p_rows = peak_rows[valid]
    p_vals = peak_vals[valid]

    # 2. Secondary peak rejection (multipath / enamel specular reflection check)
    if cfg.max_secondary_ratio < 1.0:
        r_idx = np.arange(H)[:, None]
        col_slice = img[:, valid_cols]
        dist = np.abs(r_idx - p_rows[None, :])
        guard = np.full(valid_cols.size, float(cfg.secondary_guard_px))
        if cfg.adaptive_guard:
            near = dist <= 24
            width = np.sum((col_slice >= 0.5 * p_vals[None, :]) & near, axis=0)
            guard = np.maximum(guard, 1.5 * width + 2.0)
        outside_mask = dist > guard[None, :]
        secondary_max = np.max(np.where(outside_mask, col_slice, 0.0), axis=0)
        keep = secondary_max <= (cfg.max_secondary_ratio * p_vals)
        valid_cols = valid_cols[keep]
        p_rows = p_rows[keep]
        p_vals = p_vals[keep]

    if valid_cols.size == 0:
        return np.empty(0), np.empty(0), np.empty(0)

    # 3. Sub-pixel refinement
    method = cfg.method.lower()
    if method == "cog":
        # Centre of gravity in a window that is re-centred on the estimate. A window fixed on the
        # brightest pixel cuts the line's tails unevenly whenever the true centre lies between two
        # rows, which pulls every estimate towards a whole row ("pixel locking"). On a moving model
        # that bias shows up as regular ripples across the surface. Re-centring the window (with
        # fractional weights at its ends) and removing the local baseline makes it unbiased.
        hw = cfg.half_window
        offsets = np.arange(-hw - 1, hw + 2)[:, None]
        win_rows = np.clip(p_rows[None, :] + offsets, 0, H - 1)
        win_vals = img[win_rows, valid_cols[None, :]]
        base = np.minimum(win_vals[0], win_vals[-1])
        win_vals = np.clip(win_vals - np.maximum(base, 0.0)[None, :], 0.0, None)
        rel = offsets.astype(np.float64)                     # row offsets from the brightest pixel
        c = np.zeros(valid_cols.size)
        good = np.ones(valid_cols.size, dtype=bool)
        for _ in range(3):
            # weight 1 inside [c - hw, c + hw], linear fall-off over the last half pixel at each end
            wgt = np.clip(hw + 0.5 - np.abs(rel - c[None, :]), 0.0, 1.0)
            wv = wgt * win_vals
            denom = wv.sum(axis=0)
            good = denom > 1e-9
            c = np.where(good, (rel * wv).sum(axis=0) / np.maximum(denom, 1e-9), c)
            c = np.clip(c, -1.0, 1.0)
        v_sub = p_rows[good].astype(np.float64) + c[good]
        return valid_cols[good].astype(np.float64), v_sub, p_vals[good]

    elif method == "gaussian3":
        ym1 = np.maximum(img[p_rows - 1, valid_cols], 1.0)
        y0 = np.maximum(img[p_rows, valid_cols], 1.0)
        yp1 = np.maximum(img[p_rows + 1, valid_cols], 1.0)
        ln_ym1, ln_y0, ln_yp1 = np.log(ym1), np.log(y0), np.log(yp1)
        denom = 2.0 * (ln_ym1 - 2.0 * ln_y0 + ln_yp1)
        good = np.abs(denom) > 1e-9
        delta = (ln_ym1[good] - ln_yp1[good]) / denom[good]
        delta = np.clip(delta, -0.5, 0.5)
        v_sub = p_rows[good].astype(np.float64) + delta
        return valid_cols[good].astype(np.float64), v_sub, p_vals[good]

    elif method == "steger":
        # 1D directional Taylor expansion along the column normal (since line is predominantly horizontal)
        smoothed = gaussian_filter(img, sigma=cfg.steger_sigma)
        s_rows = np.argmax(smoothed[:, valid_cols], axis=0)
        s_rows = np.clip(s_rows, 1, H - 2)
        ry = 0.5 * (smoothed[s_rows + 1, valid_cols] - smoothed[s_rows - 1, valid_cols])
        ryy = smoothed[s_rows + 1, valid_cols] - 2.0 * smoothed[s_rows, valid_cols] + smoothed[s_rows - 1, valid_cols]
        good = ryy < -1e-6
        t = -ry[good] / ryy[good]
        in_pixel = np.abs(t) <= 0.6
        good_idx = np.where(good)[0][in_pixel]
        v_sub = s_rows[good_idx].astype(np.float64) + t[in_pixel]
        return valid_cols[good_idx].astype(np.float64), v_sub, p_vals[good_idx]

    else:
        raise ValueError(f"Unknown ridge extraction method: {cfg.method}")