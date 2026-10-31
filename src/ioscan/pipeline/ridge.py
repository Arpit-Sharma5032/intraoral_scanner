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
    steger_sigma: float = 1.2     # Gaussian smoothing sigma for Steger / Hessian


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
        keep = np.ones(valid_cols.size, dtype=bool)
        r_idx = np.arange(H)[:, None]
        col_slice = img[:, valid_cols]
        outside_mask = np.abs(r_idx - p_rows[None, :]) > cfg.secondary_guard_px
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
        hw = cfg.half_window
        offsets = np.arange(-hw, hw + 1)[:, None]
        win_rows = p_rows[None, :] + offsets
        win_vals = img[win_rows, valid_cols[None, :]]
        denom = np.sum(win_vals, axis=0)
        good = denom > 1e-9
        v_sub = p_rows[good].astype(np.float64) + np.sum(offsets * win_vals[:, good], axis=0) / denom[good]
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