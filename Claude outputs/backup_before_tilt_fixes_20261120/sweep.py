"""Linear-sweep reconstruction for a single-line laser scanner.

Why this exists
---------------
A line laser yields one 1-D profile per frame. The IMU only measures orientation, so the old
``imu_hybrid`` path placed every profile along a curve built from IMU orientation and an
*assumed* constant speed. During a hand-held pass the scanner turns tens of degrees and the
speed is never constant, so the profiles were laid on a made-up trajectory and the result was
a smeared ribbon. No meshing parameter can repair that.

This module reconstructs what a line scanner can actually measure reliably: a 2.5-D surface
swept by translating the scanner across the object (a straight pass over a quadrant, a jig, a
slider or a motorised stage). It:

1. cleans each profile (spikes, reflections and very short fragments are removed),
2. keeps only the time window in which the laser line is actually on the surface,
3. measures translation from the laser-OFF frames that are already captured for ambient
   subtraction (image shift x measured depth / focal length), falling back to a known scan
   length or a constant speed,
4. fuses all profiles into a height map (robust per-cell median), fills small holes,
5. triangulates the height map into one continuous surface mesh.

Coordinates are millimetres in the *scan frame*: the camera frame of the first kept profile
(OpenCV axes: x right along the laser line, y down along the scan axis, z forward = depth).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import struct

import cv2
import numpy as np
from scipy import ndimage


# --------------------------------------------------------------------------- configuration
@dataclass
class SweepConfig:
    cell_mm: float = 0.15                 # height-map grid spacing
    translation: str = "auto"             # auto | markers | planar | flow | length | speed
    speed_mm_s: float = 5.0               # used by "speed" (and as the last fallback of "auto")
    scan_length_mm: float | None = None   # used by "length" (and by "auto" when flow fails)
    scan_direction: float = 1.0           # +1: camera +Y, -1: camera -Y (speed/length modes)
    auto_trim: bool = True                # keep only the window where the line is on the surface
    min_profile_points: int = 15          # a profile with fewer cleaned points counts as "empty"
    trim_window_s: float = 1.0            # smoothing window for the activity detector
    trim_min_active_fraction: float = 0.5
    profile_spike_mm: float = 0.8         # max deviation from the running median along the line
    profile_jump_mm: float = 0.6          # depth step that splits a profile into runs
    profile_gap_px: float = 3.0           # column gap that splits a profile into runs
    min_run_px: int = 8                   # runs shorter than this many columns are dropped
    cell_outlier_mm: float = 0.6          # point rejected if this far from its cell median
    despeckle_mm: float = 1.0             # cell rejected if this far from its 5x5 neighbourhood
    fill_holes_mm: float = 0.6            # holes up to this radius are interpolated
    smooth_sigma_cells: float = 0.8       # light Gaussian smoothing of the height map
    max_edge_jump_mm: float = 1.2         # triangles spanning a larger depth step are not built
    min_island_cells: int = 60            # isolated valid patches smaller than this are dropped
    min_component_fraction: float = 0.05  # mesh pieces smaller than this x largest are dropped
    # Optical-flow translation from the OFF frames
    flow_downsample: int = 2
    flow_min_response: float = 0.08
    flow_max_step_px: float = 60.0
    flow_min_valid_fraction: float = 0.6
    # Planar tracking (rotation + translation of the model in the image, from the OFF frames)
    planar_downsample: int = 4
    planar_min_cc: float = 0.5
    planar_min_valid_fraction: float = 0.6
    planar_rekey_deg: float = 6.0
    planar_rekey_px: float = 12.0          # in downsampled pixels
    # Marker plate (printed ArUco sheet under the model): absolute 6-DOF pose per OFF frame
    marker_min_valid_fraction: float = 0.4  # "auto" falls back to image tracking below this
    marker_min_height_mm: float = 1.0       # points closer than this to the sheet are the sheet itself
    marker_max_height_mm: float = 45.0      # nothing on the plate is taller than this


@dataclass
class Profile:
    index: int
    t: float
    pts: np.ndarray          # (N,3) camera-frame mm
    u: np.ndarray            # (N,) image column


@dataclass
class FlowSample:
    t: float                 # timestamp of the second OFF frame
    du: float                # image shift (px, full resolution) of frame k relative to k-1
    dv: float
    response: float
    valid: bool


@dataclass
class SweepResult:
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    cloud: np.ndarray
    report: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- profile cleaning
def clean_profile(pts: np.ndarray, u: np.ndarray, cfg: SweepConfig) -> tuple[np.ndarray, np.ndarray]:
    """Remove spikes, reflections and tiny fragments from one laser profile.

    The ridge extractor picks one peak per column. Reflections, enamel glints and stray light
    produce isolated columns whose depth disagrees with their neighbours; those become the
    needle-like streaks seen in the old clouds.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    if len(pts) < cfg.min_run_px:
        return pts[:0], u[:0]
    order = np.argsort(u)
    pts, u = pts[order], u[order]
    z = pts[:, 2]
    # running median along the line (index space; fine because columns are nearly contiguous)
    k = 9 if len(z) >= 9 else (len(z) // 2) * 2 + 1
    zmed = ndimage.median_filter(z, size=k, mode="nearest")
    keep = np.abs(z - zmed) <= cfg.profile_spike_mm
    pts, u, z = pts[keep], u[keep], z[keep]
    if len(pts) < cfg.min_run_px:
        return pts[:0], u[:0]
    # split into runs at gaps / depth steps and drop short runs
    brk = (np.diff(u) > cfg.profile_gap_px) | (np.abs(np.diff(z)) > cfg.profile_jump_mm)
    run_id = np.concatenate([[0], np.cumsum(brk)])
    counts = np.bincount(run_id)
    good = counts[run_id] >= cfg.min_run_px
    return pts[good], u[good]


# --------------------------------------------------------------------------- activity window
def active_window(times: np.ndarray, counts: np.ndarray, cfg: SweepConfig) -> tuple[np.ndarray, dict]:
    """Boolean mask of the profiles inside the longest window where the line is on the surface."""
    times = np.asarray(times, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    n = len(times)
    info = {"auto_trim": bool(cfg.auto_trim)}
    if n == 0:
        return np.zeros(0, dtype=bool), info
    good = counts >= cfg.min_profile_points
    if not cfg.auto_trim or n < 5:
        info.update(t_start_s=float(times[0]), t_end_s=float(times[-1]), kept_profiles=int(n))
        return np.ones(n, dtype=bool), info
    rate = (n - 1) / max(times[-1] - times[0], 1e-6)
    win = max(3, int(round(cfg.trim_window_s * rate)) | 1)
    frac = ndimage.uniform_filter1d(good.astype(np.float64), size=win, mode="nearest")
    active = frac >= cfg.trim_min_active_fraction
    lab, nlab = ndimage.label(active)
    if nlab == 0:
        info.update(t_start_s=None, t_end_s=None, kept_profiles=0)
        return np.zeros(n, dtype=bool), info
    # choose the segment holding the most good profiles
    scores = ndimage.sum(good, lab, index=np.arange(1, nlab + 1))
    best = int(np.argmax(scores)) + 1
    idx = np.flatnonzero(lab == best)
    i0, i1 = int(idx[0]), int(idx[-1])
    # trim the empty ends of the chosen segment
    g = np.flatnonzero(good[i0:i1 + 1])
    if g.size:
        i0, i1 = i0 + int(g[0]), i0 + int(g[-1])
    mask = np.zeros(n, dtype=bool)
    mask[i0:i1 + 1] = True
    info.update(
        t_start_s=round(float(times[i0] - times[0]), 3),
        t_end_s=round(float(times[i1] - times[0]), 3),
        duration_s=round(float(times[i1] - times[i0]), 3),
        kept_profiles=int(mask.sum()),
        segments_found=int(nlab),
        dropped_profiles=int(n - mask.sum()),
    )
    return mask, info


# --------------------------------------------------------------------------- optical flow
class OffFrameFlow:
    """Incremental image-shift estimator on consecutive laser-OFF frames.

    The ROI (optical opening) is used so that static chassis pixels, which move with the camera,
    cannot pull the estimate to zero. A band-pass filter removes vignetting and fixed illumination.
    Each frame is registered against a *keyframe* (not just the previous frame) with phase
    correlation for the coarse shift and ECC for sub-pixel refinement; phase correlation alone
    locks to whole pixels and that error accumulates over hundreds of steps.
    """

    def __init__(self, shape: tuple[int, int], roi_mask: np.ndarray | None, cfg: SweepConfig):
        self.cfg = cfg
        h, w = shape
        if roi_mask is not None and np.any(roi_mask):
            ys, xs = np.nonzero(roi_mask)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        else:
            y0, y1, x0, x1 = 0, h, 0, w
        self.rect = (int(y0), int(y1), int(x0), int(x1))
        ds = max(1, int(cfg.flow_downsample))
        self.ds = ds
        hh, ww = (y1 - y0) // ds, (x1 - x0) // ds
        self.size = (int(ww), int(hh))
        self.window = cv2.createHanningWindow(self.size, cv2.CV_32F) if min(hh, ww) >= 16 else None
        self._key: np.ndarray | None = None
        self._key_cum = np.zeros(2)       # cumulative shift (px, downsampled) at the keyframe
        self._cum = np.zeros(2)           # cumulative shift of the latest frame
        self._last_step = np.zeros(2)
        self._rekey = 0.08 * min(hh, ww)  # switch keyframe before overlap (and ECC score) drops
        self.samples: list[FlowSample] = []

    def _prep(self, frame: np.ndarray) -> np.ndarray:
        y0, y1, x0, x1 = self.rect
        crop = frame[y0:y1, x0:x1].astype(np.float32)
        if self.ds > 1:
            crop = cv2.resize(crop, self.size, interpolation=cv2.INTER_AREA)
        crop = cv2.GaussianBlur(crop, (0, 0), 1.5) - cv2.GaussianBlur(crop, (0, 0), 12.0)
        return crop

    def _register(self, key: np.ndarray, cur: np.ndarray, guess: np.ndarray) -> tuple[np.ndarray, float, bool]:
        (dx, dy), resp = cv2.phaseCorrelate(key, cur, self.window)
        coarse = np.array([dx, dy])
        # trust the motion-model guess when phase correlation is ambiguous
        init = coarse if (resp >= self.cfg.flow_min_response or not np.isfinite(guess).all()) else guess
        warp = np.array([[1, 0, init[0]], [0, 1, init[1]]], dtype=np.float32)
        try:
            cc, warp = cv2.findTransformECC(
                key, cur, warp, cv2.MOTION_TRANSLATION,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-4), None, 1)
            shift = np.array([warp[0, 2], warp[1, 2]], dtype=np.float64)
            ok = bool(cc >= 0.3) and np.isfinite(shift).all()
            quality = float(cc)
        except cv2.error:
            shift, quality, ok = coarse, float(resp), bool(resp >= self.cfg.flow_min_response)
        return shift, quality, ok

    def add(self, frame: np.ndarray, t: float) -> None:
        if self.window is None:
            return
        cur = self._prep(frame)
        if self._key is None:
            self._key = cur
            return
        if float(cur.std()) <= 0.5:     # no texture (dark frame / lens covered)
            self.samples.append(FlowSample(float(t), 0.0, 0.0, 0.0, False))
            self._key, self._key_cum = cur, self._cum.copy()
            return
        guess = (self._cum - self._key_cum) + self._last_step
        rel, quality, ok = self._register(self._key, cur, guess)
        step = (self._key_cum + rel) - self._cum
        ok = ok and float(np.hypot(*step)) * self.ds <= self.cfg.flow_max_step_px
        if ok:
            self._cum = self._key_cum + rel
            self._last_step = step
            du, dv = float(step[0]) * self.ds, float(step[1]) * self.ds
        else:
            du = dv = 0.0
        self.samples.append(FlowSample(float(t), du, dv, quality, bool(ok)))
        if not ok or float(np.hypot(*rel)) > self._rekey:
            self._key, self._key_cum = cur, self._cum.copy()


def build_motion_mask(off_frames_small: list[np.ndarray]) -> tuple[np.ndarray | None, dict]:
    """Mask of image pixels that see the moving model, from a sample of downsampled OFF frames.

    Excludes everything fixed to the camera: the housing outside the window, the window frame,
    shadow edges and streaks on the window glass. Those stay sharp in the temporal median, while
    the moving model blurs out, so their edges are found from the median image's gradient.
    """
    if len(off_frames_small) < 8:
        return None, {"reason": "too few OFF frames"}
    sm = np.stack([f.astype(np.float32) for f in off_frames_small])
    # Undo auto-exposure / room-light drift first: a global brightness change makes every static
    # pixel "vary" and, with a lit (textured) background, the whole scene would count as moving.
    g = sm.reshape(len(sm), -1).mean(axis=1)
    sm *= (float(np.median(g)) / np.maximum(g, 1.0))[:, None, None]
    med = np.median(sm, axis=0)
    st = sm.std(axis=0)
    # Per-pixel noise of this camera/scene: most of the frame is static, so a low percentile of
    # the temporal std is the sensor + JPEG noise level.
    noise = float(np.percentile(st, 25))
    gb = cv2.GaussianBlur(med, (0, 0), 1.0)
    g = cv2.magnitude(cv2.Sobel(gb, cv2.CV_32F, 1, 0), cv2.Sobel(gb, cv2.CV_32F, 0, 1))
    static_edge = cv2.dilate((g > np.percentile(g, 92)).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    bright = cv2.erode((np.percentile(sm, 90, axis=0) > 30).astype(np.uint8), np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bright)
    if n < 2:
        return None, {"reason": "no lit window region"}
    win = lab == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))
    # "moving" relative to this scene: dim or low-contrast models change by only a few grey levels
    st_thr = float(np.clip(max(0.25 * float(np.median(st[win])), 3.0 * noise), 1.5, 6.0))
    moving = cv2.morphologyEx((st > st_thr).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    mask = win & moving & ~static_edge
    frac = float(mask.mean())
    info = {"mask_fraction": round(frac, 4), "window_fraction": round(float(win.mean()), 4),
            "motion_threshold": round(st_thr, 2), "noise_level": round(noise, 2)}
    if mask.sum() < 400:
        info["reason"] = "the model did not move enough to separate it from the housing"
        return None, info
    return mask.astype(np.uint8), info


@dataclass
class PlanarSample:
    t: float
    A: np.ndarray            # 3x3: reference-frame (downsampled) image coords -> this frame
    cc: float
    valid: bool


class PlanarTracker:
    """Rotation + translation of the model in the image (ECC, Euclidean), keyframe based."""

    def __init__(self, mask_small: np.ndarray, cfg: SweepConfig):
        self.cfg = cfg
        self.mask = mask_small
        self.ds = int(cfg.planar_downsample)
        self.size = (mask_small.shape[1], mask_small.shape[0])
        self._key = None
        self._key_A = np.eye(3)
        self._A = np.eye(3)
        self._step = np.eye(3)
        self.samples: list[PlanarSample] = []
        # Coarse-to-fine ECC: a tight single-level criterion (100 iterations, eps 1e-5) rarely converged
        # and cost ~160 ms per frame; half-resolution first, then a short refinement, is ~10x faster.
        self._crit_coarse = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-3)
        self._crit_fine = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1e-4)
        self._crit_fallback = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
        self.size_small = (max(16, self.size[0] // 2), max(16, self.size[1] // 2))
        self.mask_small = cv2.resize(mask_small, self.size_small, interpolation=cv2.INTER_NEAREST)
        self._key_small = None
        self.reference_frame: np.ndarray | None = None

    def _prep(self, frame: np.ndarray) -> np.ndarray:
        small = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA).astype(np.float32)
        return cv2.GaussianBlur(small, (0, 0), 1.2)

    def _ecc(self, cur: np.ndarray, cur_small: np.ndarray, W: np.ndarray) -> tuple[float, np.ndarray]:
        Wc = W.copy()
        Wc[:, 2] *= 0.5
        try:
            _, Wc = cv2.findTransformECC(self._key_small, cur_small, Wc, cv2.MOTION_EUCLIDEAN,
                                         self._crit_coarse, self.mask_small, 3)
            Wf = Wc.copy()
            Wf[:, 2] *= 2.0
            return cv2.findTransformECC(self._key, cur, Wf, cv2.MOTION_EUCLIDEAN, self._crit_fine, self.mask, 5)
        except cv2.error:
            pass
        try:
            return cv2.findTransformECC(self._key, cur, W, cv2.MOTION_EUCLIDEAN, self._crit_fallback, self.mask, 5)
        except cv2.error:
            return 0.0, W

    def add(self, frame: np.ndarray, t: float) -> None:
        cur = self._prep(frame)
        cur_small = cv2.resize(cur, self.size_small, interpolation=cv2.INTER_AREA)
        if self._key is None:
            self._key, self._key_small = cur, cur_small
            self.reference_frame = frame.copy()   # the image that defines the model frame
            self.samples.append(PlanarSample(float(t), np.eye(3), 1.0, True))
            return
        guess = self._step @ self._A @ np.linalg.inv(self._key_A)
        W = guess[:2].astype(np.float32)
        cc, W = self._ecc(cur, cur_small, W)
        ok = cc >= self.cfg.planar_min_cc
        if ok:
            newA = np.vstack([W, [0.0, 0.0, 1.0]]) @ self._key_A
            stp = newA @ np.linalg.inv(self._A)
            # a hand-moved model cannot jump between two frames 40 ms apart
            if abs(np.degrees(np.arctan2(stp[1, 0], stp[0, 0]))) > 8.0 or np.hypot(stp[0, 2], stp[1, 2]) > 25.0:
                ok = False
        if ok:
            self._step = newA @ np.linalg.inv(self._A)
            self._A = newA
        self.samples.append(PlanarSample(float(t), self._A.copy(), float(cc), bool(ok)))
        rel = np.linalg.inv(self._key_A) @ self._A
        if (not ok or abs(np.degrees(np.arctan2(rel[1, 0], rel[0, 0]))) > self.cfg.planar_rekey_deg
                or np.hypot(rel[0, 2], rel[1, 2]) > self.cfg.planar_rekey_px):
            self._key, self._key_small, self._key_A = cur, cur_small, self._A.copy()


def planar_to_model_frame(samples: list[PlanarSample], times: np.ndarray, depth_t: np.ndarray,
                          depth_z: np.ndarray, camera_fx: float, camera_fy: float, cx: float, cy: float,
                          ds: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Per-profile model rotation (rad) and metric translation (mm) about the optical axis."""
    st = np.array([s.t for s in samples])
    ang = np.unwrap(np.array([np.arctan2(s.A[1, 0], s.A[0, 0]) for s in samples]))
    c = np.array([cx, cy])
    d = []
    for s in samples:  # translation about the principal point, full-resolution pixels
        R = s.A[:2, :2]
        d.append(R @ c + ds * s.A[:2, 2] - c)
    d = np.array(d)
    th = np.interp(times, st, ang)
    dx = np.interp(times, st, d[:, 0])
    dy = np.interp(times, st, d[:, 1])
    z = np.interp(times, depth_t, depth_z) if depth_t.size else np.full(len(times), 80.0)
    D = np.column_stack([dx * z / camera_fx, dy * z / camera_fy])
    ok = np.array([s.valid for s in samples])
    # fastest turn over any 1 s window: a hand-turned model does not spin 45 deg in a second
    max_rate, at = 0.0, None
    if len(st) > 3:
        j = np.searchsorted(st, st + 1.0)
        j = np.clip(j, 0, len(st) - 1)
        rate = np.abs(np.degrees(ang[j] - ang))
        k = int(np.argmax(rate))
        max_rate, at = float(rate[k]), float(st[k] - st[0])
    info = {"samples": len(samples), "valid_fraction": round(float(ok.mean()), 4),
            "max_turn_deg_per_s": round(max_rate, 1), "max_turn_at_s": None if at is None else round(at, 1),
            "median_cc": round(float(np.median([s.cc for s in samples])), 4),
            "rotation_range_deg": round(float(np.degrees(np.ptp(th))), 2) if len(th) else 0.0,
            "translation_range_mm": [round(float(np.ptp(D[:, 0])), 2), round(float(np.ptp(D[:, 1])), 2)] if len(D) else [0, 0]}
    return th, D, info


def flow_to_translation(
    samples: list[FlowSample],
    depth_t: np.ndarray,
    depth_z: np.ndarray,
    fx: float,
    fy: float,
    cfg: SweepConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Integrate image shifts into camera translation (mm) using the measured profile depth.

    A camera translation (Tx, Ty) moves a scene point at depth Z by (-fx Tx / Z, -fy Ty / Z)
    pixels. Returns (times, cumulative translation (M,2), info).
    """
    info = {"samples": len(samples)}
    if not samples:
        info["valid_fraction"] = 0.0
        return np.zeros(0), np.zeros((0, 2)), info
    t = np.array([s.t for s in samples])
    du = np.array([s.du for s in samples])
    dv = np.array([s.dv for s in samples])
    ok = np.array([s.valid for s in samples])
    info["valid_fraction"] = round(float(ok.mean()), 4)
    info["median_response"] = round(float(np.median([s.response for s in samples])), 4)
    if ok.sum() < 3:
        return t, np.zeros((len(t), 2)), info
    # reject outlier steps (e.g. a glint moving through the window) against a running median
    idx = np.flatnonzero(ok)
    for comp in (du, dv):
        med = ndimage.median_filter(comp[idx], size=7, mode="nearest")
        mad = np.median(np.abs(comp[idx] - med)) + 0.5
        bad = np.abs(comp[idx] - med) > 6.0 * mad
        ok[idx[bad]] = False
    info["valid_fraction_after_outliers"] = round(float(ok.mean()), 4)
    # fill invalid steps by interpolating neighbouring valid velocities
    idx = np.flatnonzero(ok)
    if idx.size >= 2:
        du = np.interp(np.arange(len(t)), idx, du[idx])
        dv = np.interp(np.arange(len(t)), idx, dv[idx])
    if depth_t.size:
        z = np.interp(t, depth_t, depth_z)
    else:
        z = np.full(len(t), 80.0)
    tx = -du * z / fx
    ty = -dv * z / fy
    cum = np.column_stack([np.cumsum(tx), np.cumsum(ty)])
    info["path_mm"] = round(float(np.hypot(tx, ty).sum()), 3)
    info["net_mm"] = [round(float(cum[-1, 0]), 3), round(float(cum[-1, 1]), 3)]
    return t, cum, info


# --------------------------------------------------------------------------- height map
def _cell_median(cell_id: np.ndarray, z: np.ndarray, n_cells: int) -> np.ndarray:
    order = np.lexsort((z, cell_id))
    cs, zs = cell_id[order], z[order]
    starts = np.flatnonzero(np.concatenate([[True], cs[1:] != cs[:-1]]))
    ends = np.concatenate([starts[1:], [len(cs)]])
    med = np.full(n_cells, np.nan)
    mid = (starts + ends - 1) // 2
    mid2 = (starts + ends) // 2
    med[cs[starts]] = 0.5 * (zs[mid] + zs[mid2])
    return med


def _normalized_blur(Z: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    w = valid.astype(np.float64)
    num = ndimage.gaussian_filter(np.where(valid, Z, 0.0), sigma, mode="constant")
    den = ndimage.gaussian_filter(w, sigma, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def build_heightmap(points: np.ndarray, cfg: SweepConfig) -> tuple[np.ndarray, np.ndarray, tuple[float, float], dict]:
    """Fuse scan-frame points into a depth map Z[iy, ix]; returns (Z, valid, origin_xy, info)."""
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    c = float(cfg.cell_mm)
    lo = np.floor(P[:, :2].min(axis=0) / c) * c - 2 * c
    hi = P[:, :2].max(axis=0) + 2 * c
    nx, ny = (np.ceil((hi - lo) / c).astype(int) + 1)
    ix = np.clip(((P[:, 0] - lo[0]) / c).astype(int), 0, nx - 1)
    iy = np.clip(((P[:, 1] - lo[1]) / c).astype(int), 0, ny - 1)
    cid = iy * nx + ix
    n = nx * ny
    med = _cell_median(cid, P[:, 2], n)
    keep = np.abs(P[:, 2] - med[cid]) <= cfg.cell_outlier_mm
    cnt = np.bincount(cid[keep], minlength=n).astype(np.float64)
    zsum = np.bincount(cid[keep], weights=P[keep, 2], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = (zsum / cnt).reshape(ny, nx)
    valid = (cnt > 0).reshape(ny, nx)
    info = {"grid": [int(ny), int(nx)], "cell_mm": c, "points_in": int(len(P)),
            "points_rejected_cell_outlier": int((~keep).sum()), "cells_measured": int(valid.sum())}
    # despeckle against the local median of measured cells
    if not valid.any():
        raise RuntimeError("No height-map cell received a laser point.")
    Zb = _normalized_blur(Z, valid, 2.0)
    Zf = np.where(valid, Z, np.where(np.isfinite(Zb), Zb, np.nanmedian(Z[valid])))
    loc = ndimage.median_filter(Zf, size=5, mode="nearest")
    speck = valid & (np.abs(Z - loc) > cfg.despeckle_mm)
    valid &= ~speck
    info["cells_despeckled"] = int(speck.sum())
    # drop tiny isolated islands. Profiles can be spaced wider than a cell, leaving empty rows
    # between measured rows, so islands are labelled on a mask bridged across the fill radius.
    bridge = max(1, int(np.ceil(cfg.fill_holes_mm / c)))
    bridged = ndimage.binary_dilation(valid, structure=np.ones((3, 3)), iterations=bridge)
    lab, nl = ndimage.label(bridged, structure=np.ones((3, 3)))
    if nl:
        sizes = ndimage.sum(valid, lab, index=np.arange(1, nl + 1))
        small = np.isin(lab, np.flatnonzero(sizes < cfg.min_island_cells) + 1) & valid
        valid &= ~small
        info["islands_removed"] = int((sizes < cfg.min_island_cells).sum())
    # fill small holes only (cells close to measured cells)
    r = cfg.fill_holes_mm / c
    if r > 0 and valid.any():
        dist = ndimage.distance_transform_edt(~valid)
        # only fill cells enclosed by data: a hole must be closed by a morphological closing
        closed = ndimage.binary_closing(valid, structure=np.ones((3, 3)), iterations=max(1, int(np.ceil(r))))
        fill = (~valid) & (dist <= r) & closed
        Zfill = _normalized_blur(Z, valid, max(r / 2.0, 1.0))
        Z = np.where(fill, Zfill, Z)
        valid = valid | (fill & np.isfinite(Zfill))
        info["cells_filled"] = int(fill.sum())
    if cfg.smooth_sigma_cells > 0 and valid.any():
        Zs = _normalized_blur(Z, valid, cfg.smooth_sigma_cells)
        Z = np.where(valid, Zs, np.nan)
    Z = np.where(valid, Z, np.nan)
    info["cells_valid"] = int(valid.sum())
    info["area_mm2"] = round(float(valid.sum() * c * c), 2)
    return Z, valid, (float(lo[0]), float(lo[1])), info


def heightmap_to_mesh(Z: np.ndarray, valid: np.ndarray, origin: tuple[float, float], cfg: SweepConfig):
    """Triangulate a depth map. Returns (vertices, faces, normals)."""
    ny, nx = Z.shape
    c = cfg.cell_mm
    vid = -np.ones((ny, nx), dtype=np.int64)
    vid[valid] = np.arange(int(valid.sum()))
    yy, xx = np.nonzero(valid)
    V = np.column_stack([origin[0] + (xx + 0.5) * c, origin[1] + (yy + 0.5) * c, Z[yy, xx]])
    a, b = vid[:-1, :-1], vid[:-1, 1:]
    cc, d = vid[1:, :-1], vid[1:, 1:]
    za, zb, zc, zd = Z[:-1, :-1], Z[:-1, 1:], Z[1:, :-1], Z[1:, 1:]
    J = cfg.max_edge_jump_mm

    def tri(i, j, k, zi, zj, zk):
        ok = (i >= 0) & (j >= 0) & (k >= 0)
        with np.errstate(invalid="ignore"):
            ok &= (np.abs(zi - zj) <= J) & (np.abs(zj - zk) <= J) & (np.abs(zi - zk) <= J)
        return np.column_stack([i[ok], j[ok], k[ok]])

    # pick the diagonal with the smaller depth difference in each quad
    with np.errstate(invalid="ignore"):
        use_ad = np.nan_to_num(np.abs(za - zd), nan=1e9) <= np.nan_to_num(np.abs(zb - zc), nan=1e9)
    F = []
    m = use_ad
    F.append(tri(a[m], cc[m], d[m], za[m], zc[m], zd[m]))
    F.append(tri(a[m], d[m], b[m], za[m], zd[m], zb[m]))
    m = ~use_ad
    F.append(tri(a[m], cc[m], b[m], za[m], zc[m], zb[m]))
    F.append(tri(b[m], cc[m], d[m], zb[m], zc[m], zd[m]))
    faces = np.vstack(F) if F else np.zeros((0, 3), dtype=np.int64)
    faces, V = _keep_large_components(faces, V, cfg.min_component_fraction)
    normals = vertex_normals(V, faces)
    # Orient towards the camera (-Z in the scan frame)
    if len(normals) and np.mean(normals[:, 2]) > 0:
        faces = faces[:, [0, 2, 1]]
        normals = -normals
    return V, faces, normals


def _keep_large_components(faces: np.ndarray, V: np.ndarray, min_fraction: float):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    if len(faces) == 0:
        return faces, V[:0]
    n = len(V)
    r = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    cidx = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    g = coo_matrix((np.ones(r.size, dtype=np.int8), (r, cidx)), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    flab = lab[faces[:, 0]]
    counts = np.bincount(flab)
    keep_lab = np.flatnonzero(counts >= max(1, min_fraction * counts.max()))
    fk = faces[np.isin(flab, keep_lab)]
    used = np.unique(fk)
    remap = -np.ones(n, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return remap[fk], V[used]


def vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    N = np.zeros_like(V)
    if len(F) == 0:
        return N
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    L = np.linalg.norm(N, axis=1, keepdims=True)
    return N / np.maximum(L, 1e-12)


# --------------------------------------------------------------------------- reconstruction
def reconstruct_sweep(
    profiles: list[Profile],
    cfg: SweepConfig,
    *,
    camera_fx: float,
    camera_fy: float,
    flow_samples: list[FlowSample] | None = None,
    planar_samples: list[PlanarSample] | None = None,
    camera_cx: float = 800.0,
    camera_cy: float = 600.0,
) -> SweepResult:
    """Clean, trim, place and fuse laser profiles into one height-map surface."""
    report: dict = {"profiles_in": len(profiles)}
    if not profiles:
        raise RuntimeError("No laser profiles were triangulated.")
    cleaned: list[Profile] = []
    raw_pts = cleaned_pts = 0
    for p in profiles:
        raw_pts += len(p.pts)
        q, u = clean_profile(p.pts, p.u, cfg)
        cleaned_pts += len(q)
        cleaned.append(Profile(p.index, p.t, q, u))
    report["profile_cleaning"] = {
        "points_in": int(raw_pts), "points_kept": int(cleaned_pts),
        "fraction_kept": round(cleaned_pts / max(raw_pts, 1), 4),
    }
    times = np.array([p.t for p in cleaned])
    counts = np.array([len(p.pts) for p in cleaned])
    planar_ok = False
    planar_rejected = None
    if planar_samples and cfg.translation in ("auto", "planar"):
        pv = np.mean([s.valid for s in planar_samples])
        planar_ok = pv >= cfg.planar_min_valid_fraction
        if not planar_ok:
            planar_rejected = {"valid_fraction": round(float(pv), 4),
                               "reason": "too few ambient frames tracked; fell back"}
        if cfg.translation == "planar" and not planar_ok:
            raise RuntimeError(
                f"Planar tracking failed: only {100 * pv:.0f}% of OFF frames were tracked reliably.")
    if planar_ok:
        # With full motion known, every profile that saw the surface is usable: the line may leave
        # the teeth briefly while the model is turned, so do not trim to one window.
        good = counts >= cfg.min_profile_points
        mask = good
        trim = {"auto_trim": False, "mode": "planar: all profiles with a line",
                "kept_profiles": int(good.sum()), "dropped_profiles": int((~good).sum()),
                "t_start_s": round(float(times[good][0] - times[0]), 3) if good.any() else None,
                "t_end_s": round(float(times[good][-1] - times[0]), 3) if good.any() else None,
                "duration_s": round(float(times[good][-1] - times[good][0]), 3) if good.any() else 0.0}
    else:
        mask, trim = active_window(times, counts, cfg)
    report["active_window"] = trim
    kept = [p for p, m in zip(cleaned, mask) if m and len(p.pts)]
    if not kept:
        raise RuntimeError(
            "No time window with a continuous laser line on the surface was found. "
            "Check exposure/laser power and that the object sits inside the depth window."
        )
    t0, t1 = kept[0].t, kept[-1].t
    duration = max(t1 - t0, 1e-6)
    kept_counts = np.array([len(p.pts) for p in kept])
    report["kept_profiles"] = len(kept)
    report["mean_points_per_kept_profile"] = round(float(kept_counts.mean()), 2)

    # per-profile median depth (scales optical-flow pixels into millimetres)
    depth_t = np.array([p.t for p in kept])
    depth_z = np.array([float(np.median(p.pts[:, 2])) for p in kept])
    if len(depth_z) >= 5:
        depth_z = ndimage.median_filter(depth_z, size=min(25, len(depth_z) // 2 * 2 + 1), mode="nearest")

    mode = cfg.translation
    tr_info: dict = {"requested": mode}
    if planar_rejected:
        tr_info["planar_rejected"] = planar_rejected
    pos_x = np.zeros(len(kept))
    pos_y = np.zeros(len(kept))
    used = None
    theta = None
    if planar_ok:
        theta, D, pinfo = planar_to_model_frame(planar_samples, depth_t, depth_t, depth_z, camera_fx, camera_fy,
                                                camera_cx, camera_cy, cfg.planar_downsample)
        tr_info["planar"] = pinfo
        used = "planar"
    elif mode in ("auto", "flow") and flow_samples:
        ft, cum, finfo = flow_to_translation(flow_samples, depth_t, depth_z, camera_fx, camera_fy, cfg)
        tr_info["flow"] = finfo
        if ft.size >= 3 and finfo.get("valid_fraction", 0.0) >= cfg.flow_min_valid_fraction:
            px = np.interp(depth_t, ft, cum[:, 0])
            py = np.interp(depth_t, ft, cum[:, 1])
            pos_x, pos_y = px - px[0], py - py[0]
            used = "flow"
        elif mode == "flow":
            raise RuntimeError(
                f"Optical-flow translation failed: only {100 * finfo.get('valid_fraction', 0):.0f}% of "
                "OFF-frame pairs had a reliable image shift (needs visible texture through the ROI). "
                "Use translation 'length' with the known pass length, or 'speed'."
            )
    elif mode == "flow":
        raise RuntimeError("Optical-flow translation needs laser-OFF frames; none were recorded.")
    if used is None and mode in ("auto", "length") and cfg.scan_length_mm:
        speed = float(cfg.scan_length_mm) / duration
        pos_y = cfg.scan_direction * speed * (depth_t - t0)
        used = "length"
        tr_info["speed_mm_s"] = round(speed, 4)
    elif used is None and mode == "length":
        raise RuntimeError("Translation 'length' needs the scan length in mm.")
    if used is None:
        pos_y = cfg.scan_direction * float(cfg.speed_mm_s) * (depth_t - t0)
        used = "speed"
        tr_info["speed_mm_s"] = float(cfg.speed_mm_s)
    tr_info["used"] = used
    if used == "planar":
        # model-frame placement (tracker reference frame): X0 = R(theta)^T (X - D)
        world = []
        for p, th, dd in zip(kept, theta, D):
            ct, sn = np.cos(th), np.sin(th)
            xy = p.pts[:, :2] - dd
            x0 = ct * xy[:, 0] + sn * xy[:, 1]
            y0 = -sn * xy[:, 0] + ct * xy[:, 1]
            world.append(np.column_stack([x0, y0, p.pts[:, 2]]))
        steps = np.hypot(np.diff(D[:, 0]), np.diff(D[:, 1]))
        tr_info["path_mm"] = round(float(steps.sum()), 3)
        tr_info["median_profile_spacing_mm"] = round(float(np.median(steps)), 4) if steps.size else 0.0
        tr_info["extent_along_scan_mm"] = round(float(np.ptp(D[:, 1])), 3)
        tr_info["extent_across_scan_mm"] = round(float(np.ptp(D[:, 0])), 3)
        tr_info["rotation_range_deg"] = round(float(np.degrees(np.ptp(theta))), 2)
        report["translation"] = tr_info
        P, report["surface_cleaning"] = keep_main_surface(np.vstack(world))
        Z, valid, origin, hm = build_heightmap(P, cfg)
        report["heightmap"] = hm
        V, F, N = heightmap_to_mesh(Z, valid, origin, cfg)
        report["mesh"] = {"vertices": int(len(V)), "faces": int(len(F))}
        return SweepResult(V, F, N, _thin(P, cfg.cell_mm * 0.5), report)
    steps = np.hypot(np.diff(pos_x), np.diff(pos_y))
    tr_info["path_mm"] = round(float(steps.sum()), 3)
    tr_info["median_profile_spacing_mm"] = round(float(np.median(steps)), 4) if steps.size else 0.0
    tr_info["extent_along_scan_mm"] = round(float(np.ptp(pos_y)), 3)
    tr_info["extent_across_scan_mm"] = round(float(np.ptp(pos_x)), 3)
    report["translation"] = tr_info

    world = [p.pts + np.array([px, py, 0.0]) for p, px, py in zip(kept, pos_x, pos_y)]
    P, report["surface_cleaning"] = keep_main_surface(np.vstack(world))
    Z, valid, origin, hm = build_heightmap(P, cfg)
    report["heightmap"] = hm
    V, F, N = heightmap_to_mesh(Z, valid, origin, cfg)
    report["mesh"] = {"vertices": int(len(V)), "faces": int(len(F))}
    # point cloud: inlier points thinned to ~cell spacing
    cloud = _thin(P, cfg.cell_mm * 0.5)
    return SweepResult(V, F, N, cloud, report)


def reconstruct_marker_plate(
    profiles: list[Profile],
    cfg: SweepConfig,
    pose_sm: dict,
    *,
    mirrored: bool = False,
    refine_poses=None,
) -> SweepResult:
    """Place every profile in the marker-plate frame using the plate pose measured in each OFF frame.

    No assumption is made about the motion: the plate may slide, turn, tilt, lift, stop, reverse or
    be scanned in several passes. Profiles recorded while the plate pose is unknown are skipped
    rather than guessed. Plate frame: Z = 0 on the sheet, teeth at negative Z (towards the camera).
    """
    from ioscan.pipeline.markers import poses_at, to_plate_frame

    report: dict = {"profiles_in": len(profiles)}
    if not profiles:
        raise RuntimeError("No laser profiles were triangulated.")
    cleaned: list[Profile] = []
    raw_pts = cleaned_pts = 0
    for p in profiles:
        raw_pts += len(p.pts)
        q, u = clean_profile(p.pts, p.u, cfg)
        cleaned_pts += len(q)
        cleaned.append(Profile(p.index, p.t, q, u))
    report["profile_cleaning"] = {
        "points_in": int(raw_pts), "points_kept": int(cleaned_pts),
        "fraction_kept": round(cleaned_pts / max(raw_pts, 1), 4),
    }
    times = np.array([p.t for p in cleaned])
    counts = np.array([len(p.pts) for p in cleaned])
    R, tv, ok = poses_at(times, pose_sm)

    def _sheet_offset() -> float | None:
        zs = [to_plate_frame(cleaned[i].pts, R[i], tv[i])[:, 2] for i in np.flatnonzero(ok & (counts >= 8))[::3]]
        zs = np.concatenate(zs) if zs else np.empty(0)
        zs = zs[np.abs(zs) < 3.0]
        return float(np.median(zs)) if len(zs) >= 200 else None

    # Markers and laser are two independent measurements of where the sheet is. Before the laser is
    # used to steady the plate pose, their disagreement is the check of the calibration.
    marker_laser_offset = _sheet_offset()
    report["marker_laser_offset_mm"] = None if marker_laser_offset is None else round(marker_laser_offset, 3)
    if refine_poses is not None and marker_laser_offset is not None and abs(marker_laser_offset) <= 0.8:
        # Laser points that fell on the bare sheet are precise measurements of the plate plane.
        # Feed them back into the pose estimate (twice: the first pass sharpens which points they are).
        for _ in range(2):
            st, sx = [], []
            for i in np.flatnonzero(ok & (counts >= 8)):
                q = cleaned[i].pts
                z = to_plate_frame(q, R[i], tv[i])[:, 2]
                on = np.flatnonzero(np.abs(z - np.median(z[np.abs(z) < 3.0]) if (np.abs(z) < 3.0).sum() >= 6 else np.inf) < 0.8)
                if len(on) >= 6:
                    on = on[np.linspace(0, len(on) - 1, min(len(on), 10)).astype(int)]
                    st.append(np.full(len(on), cleaned[i].t))
                    sx.append(q[on])
            if len(st) < 10:
                break
            pose_sm = refine_poses((np.concatenate(st), np.vstack(sx)))
            R, tv, ok = poses_at(times, pose_sm)
        report["pose_refinement"] = {"plate_plane": pose_sm.get("plane_mode"),
                                     "laser_assisted_frames": int(pose_sm.get("laser_plane_fits", 0))}
    has_line = counts >= cfg.min_profile_points
    use = has_line & ok
    report["active_window"] = {
        "auto_trim": False, "mode": "markers: every profile with a line and a plate pose",
        "kept_profiles": int(use.sum()), "dropped_profiles": int((~has_line).sum()),
        "profiles_without_pose": int((has_line & ~ok).sum()),
        "t_start_s": round(float(times[use][0] - times[0]), 3) if use.any() else None,
        "t_end_s": round(float(times[use][-1] - times[0]), 3) if use.any() else None,
        "duration_s": round(float(times[use][-1] - times[use][0]), 3) if use.any() else 0.0,
    }
    if not use.any():
        raise RuntimeError(
            "No laser profile was recorded while the marker plate was visible. Keep at least two markers "
            "in view of the camera for the whole scan and make sure the laser line crosses the model.")
    idx = np.flatnonzero(use)
    world = [to_plate_frame(cleaned[i].pts, R[i], tv[i]) for i in idx]
    # Passes recorded at different times disagree by up to ~1 mm in height where they overlap (the
    # plate pose carries small errors that change as other markers come into view). Neighbouring
    # profiles share the error, so it is solved as one height correction per short time segment
    # that makes all overlaps agree.
    world, report["pass_alignment"] = align_passes(world, times[idx])
    # A few profiles get a wrong plate pose (a marker misread or motion blur in the OFF frame next to
    # them). Every spot of the model is seen by many profiles, so such a profile disagrees with the
    # others over its whole length and shows up as a straight ridge or groove. Drop it.
    keep_p, report["profile_outliers"] = reject_outlier_profiles(world)
    idx = idx[keep_p]
    world = [w for w, k in zip(world, keep_p) if k]
    report["kept_profiles"] = int(len(idx))
    report["mean_points_per_kept_profile"] = round(float(counts[idx].mean()), 2)
    P = np.vstack(world)
    pid_all = np.repeat(np.arange(len(world)), [len(w) for w in world])

    # Laser points that fell on the sheet itself must lie on Z = 0: a free check of the calibration.
    near = np.abs(P[:, 2]) < 2.5
    plate_check = {"points_on_sheet": int(near.sum())}
    if near.sum() >= 200:
        zs = P[near, 2]
        plate_check["offset_mm"] = round(float(np.median(zs)), 3)
        plate_check["spread_mm"] = round(float(1.4826 * np.median(np.abs(zs - np.median(zs)))), 3)
    if marker_laser_offset is not None:
        plate_check["offset_mm"] = round(marker_laser_offset, 3)      # markers vs laser, before any refinement
    report["plate_check"] = plate_check
    off = float(np.median(P[near, 2])) if near.sum() >= 200 else 0.0
    above = (P[:, 2] < off - cfg.marker_min_height_mm) & (P[:, 2] > -cfg.marker_max_height_mm)
    report["plate_points_removed"] = int((~above).sum())
    P = P[above]
    # How well overlapping profiles agree on the model itself: the direct measure of a steady plate
    # pose and a consistent calibration. (Laser hits on the paper are noisier: there the line crosses
    # black and white marker cells.)
    report["model_consistency"] = profile_consistency(P, pid_all[above])
    if len(P) < 200:
        raise RuntimeError(
            "The laser line only hit the marker sheet, not the model. Position the model so the line "
            "crosses the teeth (watch the Ready check) and scan again.")
    if mirrored:
        P = P * np.array([-1.0, 1.0, 1.0])       # undo the mirror in the optical path: true handedness

    cam_pos = -np.einsum("nji,nj->ni", R[idx], tv[idx])           # camera centre in the plate frame
    steps = np.linalg.norm(np.diff(cam_pos[:, :2], axis=0), axis=1)
    yaw = np.degrees(np.unwrap(np.arctan2(R[idx][:, 1, 0], R[idx][:, 0, 0])))
    tilt = np.degrees(np.arccos(np.clip(R[idx][:, 2, 2], -1.0, 1.0)))
    report["translation"] = {
        "requested": cfg.translation, "used": "markers",
        "path_mm": round(float(steps.sum()), 3),
        "median_profile_spacing_mm": round(float(np.median(steps)), 4) if steps.size else 0.0,
        "extent_along_scan_mm": round(float(np.ptp(cam_pos[:, 1])), 3),
        "extent_across_scan_mm": round(float(np.ptp(cam_pos[:, 0])), 3),
        "rotation_range_deg": round(float(np.ptp(yaw)), 2),
        "tilt_max_deg": round(float(tilt.max()), 2),
    }
    P, report["surface_cleaning"] = keep_main_surface(P, on_plate=True)
    Z, valid, origin, hm = build_heightmap(P, cfg)
    report["heightmap"] = hm
    V, F, N = heightmap_to_mesh(Z, valid, origin, cfg)
    report["mesh"] = {"vertices": int(len(V)), "faces": int(len(F))}
    return SweepResult(V, F, N, _thin(P, cfg.cell_mm * 0.5), report)


def _thin(P: np.ndarray, voxel: float) -> np.ndarray:
    if len(P) == 0:
        return P
    key = np.floor(P / voxel).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return P[np.sort(idx)]


def _cell_groups(P: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort points into XY cells: (order, start of each cell in that order, points per cell)."""
    k = np.floor(P[:, :2] / cell).astype(np.int64)
    k -= k.min(axis=0)
    key = k[:, 1] * (int(k[:, 0].max()) + 1) + k[:, 0]
    order = np.argsort(key, kind="stable")
    _u, start, cnt = np.unique(key[order], return_index=True, return_counts=True)
    return order, start, cnt


def _cell_median_residual(P: np.ndarray, cell: float, min_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Height of each point minus the median height of its XY cell; mask of points in busy cells."""
    order, start, cnt = _cell_groups(P, cell)
    z = P[order, 2]
    med = np.empty(len(P))
    busy = np.zeros(len(P), dtype=bool)
    for a, c in zip(start, cnt):
        sl = order[a:a + c]
        med[sl] = np.median(z[a:a + c])
        busy[sl] = c >= min_count
    return P[:, 2] - med, busy


def align_passes(world: list[np.ndarray], t: np.ndarray, seg_s: float = 2.0, cell: float = 0.4,
                 lam_offset: float = 0.02, lam_tilt: float = 3.0, iters: int = 4) -> tuple[list[np.ndarray], dict]:
    """Make overlapping passes agree: z += a(t) + b(t) x + c(t) y, one (a, b, c) per time segment.

    Inputs are plate-frame profiles and their times. Every XY cell seen in two or more time segments
    gives one equation per pair of segments; the corrections are solved by robust (Huber) least
    squares, pulled towards zero (tilts strongly), and interpolated in time between segment centres.
    Only points on the model (above the sheet) are used to fit. Validated on a real scan by fitting
    on half of the surface and checking the other half: the mismatch between passes fell from
    0.41 mm to 0.17 mm.
    """
    t = np.asarray(t, dtype=np.float64)
    info: dict = {"applied": False}
    if len(world) < 30 or float(np.ptp(t)) < 2 * seg_s:
        info["reason"] = "single short pass"
        return world, info
    t0 = float(t.min())
    nseg = int(np.floor((t.max() - t0) / seg_s)) + 1
    centers = t0 + (np.arange(nseg) + 0.5) * seg_s
    P = np.vstack(world)
    pt = np.repeat(t, [len(w) for w in world])
    m = P[:, 2] < -1.5
    Q, qt = P[m], pt[m]
    if len(Q) < 2000:
        info["reason"] = "too few model points"
        return world, info
    sg = np.minimum(((qt - t0) / seg_s).astype(np.int64), nseg - 1)
    k = np.floor(Q[:, :2] / cell).astype(np.int64)
    k -= k.min(axis=0)
    key = (k[:, 1] * (int(k[:, 0].max()) + 1) + k[:, 0]) * (nseg + 1) + sg
    o = np.argsort(key, kind="stable")
    u, st, cn = np.unique(key[o], return_index=True, return_counts=True)
    good = cn >= 3
    z, x, y = Q[o, 2], Q[o, 0], Q[o, 1]
    med = np.array([np.median(z[a:a + c]) for a, c in zip(st[good], cn[good])])
    mx = np.array([x[a:a + c].mean() for a, c in zip(st[good], cn[good])])
    my = np.array([y[a:a + c].mean() for a, c in zip(st[good], cn[good])])
    cid = u[good] // (nseg + 1)
    sid = (u[good] % (nseg + 1)).astype(np.int64)
    o2 = np.argsort(cid, kind="stable")
    cid, sid, med, mx, my = cid[o2], sid[o2], med[o2], mx[o2], my[o2]
    _uc, st2, cn2 = np.unique(cid, return_index=True, return_counts=True)
    I, J = [], []
    for a, c in zip(st2, cn2):
        if c >= 2:
            ii, jj = np.triu_indices(int(c), 1)
            I.append(a + ii)
            J.append(a + jj)
    if not I or sum(len(i) for i in I) < 200:
        info["reason"] = "passes hardly overlap"
        return world, info
    I, J = np.concatenate(I), np.concatenate(J)
    xc, yc = float(np.median(mx)), float(np.median(my))
    X = 0.5 * (mx[I] + mx[J]) - xc
    Y = 0.5 * (my[I] + my[J]) - yc
    X, Y = X / 20.0, Y / 20.0                       # lever arm in units of 20 mm
    n = 3 * nseg
    rows = np.arange(len(I))
    A = np.zeros((len(I), n))
    si, sj = sid[I], sid[J]
    A[rows, 3 * si] += 1.0
    A[rows, 3 * si + 1] += X
    A[rows, 3 * si + 2] += Y
    A[rows, 3 * sj] -= 1.0
    A[rows, 3 * sj + 1] -= X
    A[rows, 3 * sj + 2] -= Y
    b = -(med[I] - med[J])
    reg = np.diag(np.tile([lam_offset, lam_tilt, lam_tilt], nseg) * len(I) / nseg) + 1e-6 * np.eye(n)
    w = np.ones(len(I))
    for _ in range(iters):
        Aw = A * w[:, None]
        prm = np.linalg.solve(Aw.T @ A + reg, Aw.T @ b)
        r = A @ prm - b
        s = 1.4826 * float(np.median(np.abs(r))) + 1e-6
        w = np.clip(1.5 * s / np.maximum(np.abs(r), 1e-9), 0.0, 1.0)
    before = float(np.median(np.abs(b)))
    after = float(np.median(np.abs(A @ prm - b)))
    if after > 0.9 * before:
        info.update(reason="no gain", mismatch_before_mm=round(before, 3), mismatch_after_mm=round(after, 3))
        return world, info
    prm = prm.reshape(nseg, 3)
    out = []
    for wp, tp in zip(world, t):
        a0, b0, c0 = (float(np.interp(tp, centers, prm[:, j])) for j in range(3))
        q = wp.copy()
        q[:, 2] += a0 + b0 * (wp[:, 0] - xc) / 20.0 + c0 * (wp[:, 1] - yc) / 20.0
        out.append(q)
    info.update(applied=True, segments=nseg, overlap_pairs=int(len(I)),
                mismatch_before_mm=round(before, 3), mismatch_after_mm=round(after, 3),
                max_offset_mm=round(float(np.abs(prm[:, 0]).max()), 3),
                max_tilt_mm_per_20mm=round(float(np.abs(prm[:, 1:]).max()), 3))
    return out, info


def reject_outlier_profiles(world: list[np.ndarray], cell: float = 0.3, max_offset_mm: float = 0.35,
                            max_bad_fraction: float = 0.3, min_shared: int = 20) -> tuple[np.ndarray, dict]:
    """Profiles (plate frame) that disagree with the consensus of the others along their whole length.

    A profile is dropped when its median height offset from the cell medians exceeds ``max_offset_mm``
    or more than ``max_bad_fraction`` of its points are over 1 mm off. Only points on the model
    (above the sheet) in cells seen by several profiles take part.
    """
    n = len(world)
    keep = np.ones(n, dtype=bool)
    info = {"dropped": 0, "checked": 0, "max_offset_mm": max_offset_mm}
    if n < 30:
        return keep, info
    P = np.vstack(world)
    pid = np.repeat(np.arange(n), [len(w) for w in world])
    on_model = P[:, 2] < -1.0
    if on_model.sum() < 1000:
        return keep, info
    P, pid = P[on_model], pid[on_model]
    r, busy = _cell_median_residual(P, cell, 5)
    r, pid = r[busy], pid[busy]
    order = np.argsort(pid, kind="stable")
    r, pid = r[order], pid[order]
    ids, start, cnt = np.unique(pid, return_index=True, return_counts=True)
    offs = []
    for i, a, c in zip(ids, start, cnt):
        if c < min_shared:
            continue
        rr = r[a:a + c]
        off = float(np.median(rr))
        offs.append(off)
        if abs(off) > max_offset_mm or float((np.abs(rr) > 1.0).mean()) > max_bad_fraction:
            keep[i] = False
    info["checked"] = len(offs)
    info["dropped"] = int((~keep).sum())
    if offs:
        o = np.asarray(offs)
        info["offset_spread_mm"] = round(float(1.4826 * np.median(np.abs(o - np.median(o)))), 3)
    if keep.sum() < 0.7 * n:            # disagreement everywhere is not a few bad poses: keep all
        info["dropped"] = 0
        info["skipped"] = "too many profiles disagree"
        keep[:] = True
    return keep, info


def profile_consistency(P: np.ndarray, pid: np.ndarray, cell: float = 0.3) -> dict:
    """Robust spread of heights within XY cells seen by >= 3 different profiles (model points)."""
    if len(P) < 1000:
        return {"cells": 0}
    order, start, cnt = _cell_groups(P, cell)
    z, p = P[order, 2], pid[order]
    spreads = []
    for a, c in zip(start, cnt):
        if c < 6 or len(np.unique(p[a:a + c])) < 3:
            continue
        zz = z[a:a + c]
        spreads.append(1.4826 * float(np.median(np.abs(zz - np.median(zz)))))
    if not spreads:
        return {"cells": 0}
    return {"cells": len(spreads), "spread_mm": round(float(np.median(spreads)), 3),
            "spread_p90_mm": round(float(np.percentile(spreads, 90)), 3)}


def keep_main_surface(P: np.ndarray, voxel: float = 0.5, deeper_mm: float = 3.0,
                      on_plate: bool = False) -> tuple[np.ndarray, dict]:
    """Drop disconnected surfaces lying well behind the teeth (table/floor seen past the model edge).

    Such a strip sits at the same place relative to the camera in every pass, so it would pull
    the passes onto each other instead of onto the teeth.

    ``on_plate=True`` (marker plate): the sheet, and with it the table, has already been removed by
    height, so every surface left is part of the model or something standing on the plate. Gums,
    the floor of the arch and the model base lie well below the crown tops and look like "a second
    population behind a gap" from above; here they must be kept. Only tiny specks are removed.
    """
    if len(P) < 100:
        return P, {"dropped_points": 0}
    info_cut = {}
    if on_plate:
        deeper_mm = float("inf")
    # 1) depth gap: the occlusal surface fills the first few mm behind the crown tops; a table or
    #    floor seen past the model edge forms a second population behind an empty gap.
    z = P[:, 2]
    ref = float(np.percentile(z, 3))
    edges = np.arange(-1.0, 25.0, 0.5)
    h, _ = np.histogram(z - ref, bins=edges)
    frac = h / len(P)
    peak = int(np.argmax(frac))
    for i in range(peak + 1, (len(frac) - 1) if not on_plate else 0):
        # a real void: at least 1 mm (two bins) almost empty, with a separate population behind it
        # The population behind must be a minority: when the base/gum holds most points (it is the
        # main peak), nothing behind the teeth is removed.
        if (frac[i] < 0.005 and frac[i + 1] < 0.005 and frac[i] + frac[i + 1] < 0.008
                and 0.01 < frac[i:].sum() < 0.40 and edges[i] > 3.0):
            cut = ref + edges[i] + 0.5
            behind = z > cut
            info_cut = {"depth_cut_mm_behind_tops": round(float(edges[i] + 0.5), 2),
                        "dropped_behind_gap": int(behind.sum())}
            P = P[~behind]
            break
    key = np.floor(P / voxel).astype(np.int64)
    key -= key.min(axis=0)
    dims = key.max(axis=0) + 1
    occ = np.zeros(dims, dtype=bool)
    occ[key[:, 0], key[:, 1], key[:, 2]] = True
    lab, n = ndimage.label(ndimage.binary_dilation(occ, iterations=1), structure=np.ones((3, 3, 3)))
    plab = lab[key[:, 0], key[:, 1], key[:, 2]]
    if n <= 1:
        return P, dict(info_cut, dropped_points=int(info_cut.get("dropped_behind_gap", 0)), components=int(n))
    counts = np.bincount(plab, minlength=n + 1)
    counts[0] = 0
    main = int(np.argmax(counts))
    z_main = float(np.median(P[plab == main, 2]))
    keep = np.zeros(len(P), dtype=bool)
    dropped = []
    for c in range(1, n + 1):
        if counts[c] == 0:
            continue
        m = plab == c
        zc = float(np.median(P[m, 2]))
        tiny = counts[c] < 0.002 * len(P)
        behind = zc - z_main > deeper_mm                 # +z is away from the camera
        if c == main or not (tiny or behind):
            keep |= m
        else:
            dropped.append({"points": int(counts[c]), "depth_behind_mm": round(zc - z_main, 2)})
    return P[keep], dict(info_cut, dropped_points=int((~keep).sum()) + int(info_cut.get("dropped_behind_gap", 0)),
                         components=int(n), dropped_surfaces=[d for d in dropped if d["points"] > 200][:6])



# --------------------------------------------------------------------------- PLY / STL writers
def write_ply_points(path: str | Path, pts: np.ndarray, normals: np.ndarray | None = None) -> None:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    hdr = ["ply", "format binary_little_endian 1.0", f"element vertex {len(pts)}",
           "property float x", "property float y", "property float z"]
    data = pts
    if normals is not None:
        hdr += ["property float nx", "property float ny", "property float nz"]
        data = np.hstack([pts, np.asarray(normals, dtype=np.float32).reshape(-1, 3)])
    hdr.append("end_header")
    path = Path(path)
    tmp = path.with_suffix(".tmp.ply")
    with open(tmp, "wb") as f:
        f.write(("\n".join(hdr) + "\n").encode("ascii"))
        f.write(np.ascontiguousarray(data, dtype="<f4").tobytes())
    tmp.replace(path)


def write_ply_mesh(path: str | Path, V: np.ndarray, F: np.ndarray, N: np.ndarray | None = None) -> None:
    V = np.asarray(V, dtype="<f4").reshape(-1, 3)
    F = np.asarray(F, dtype="<i4").reshape(-1, 3)
    hdr = ["ply", "format binary_little_endian 1.0", f"element vertex {len(V)}",
           "property float x", "property float y", "property float z"]
    vdata = V
    if N is not None:
        hdr += ["property float nx", "property float ny", "property float nz"]
        vdata = np.hstack([V, np.asarray(N, dtype="<f4").reshape(-1, 3)])
    hdr += [f"element face {len(F)}", "property list uchar int vertex_indices", "end_header"]
    fdt = np.dtype([("n", "u1"), ("i", "<i4", 3)])
    fr = np.empty(len(F), dtype=fdt)
    fr["n"] = 3
    fr["i"] = F
    path = Path(path)
    tmp = path.with_suffix(".tmp.ply")
    with open(tmp, "wb") as f:
        f.write(("\n".join(hdr) + "\n").encode("ascii"))
        f.write(np.ascontiguousarray(vdata, dtype="<f4").tobytes())
        f.write(fr.tobytes())
    tmp.replace(path)


def write_stl(path: str | Path, V: np.ndarray, F: np.ndarray) -> None:
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(F), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"] = n
    rec["v"] = tri
    path = Path(path)
    with open(path, "wb") as f:
        f.write(b"IOSCAN linear-sweep surface (mm)".ljust(80, b" "))
        f.write(struct.pack("<I", len(F)))
        f.write(rec.tobytes())
