"""Stitch several straight-pass (sweep) reconstructions into one full-arch surface.

Each sweep pass is an accurate 2.5-D patch in its own frame (the camera frame at the start of
the pass). A full arch is scanned as 4-6 overlapping straight passes; this module aligns them
from the tooth geometry in the overlaps and fuses them into one surface.

Alignment
---------
1. Coarse: every patch is rasterised to a height map seen along its viewing axis. For yaw angles
   around the full circle, a masked normalised cross-correlation (FFT) finds the best in-plane
   shift against the growing map. Height offset is irrelevant to NCC, so different scanning
   distances do not matter.
2. Fine: trimmed point-to-plane ICP (6 DoF) against the map.
3. Global: every patch is re-aligned to the union of all others a few times, so the error is
   spread around the arch instead of accumulating at the last pass.

IMU use: the IMU's rotation between passes is a cross-check and a tie-breaker between
similar-scoring coarse candidates. It never sets a position, so gyro drift and the unknown
IMU-to-camera mounting cannot add error.

The fused arch is written in the project's canonical dental frame (``sim.motion.canonicalize``):
arch in XY, teeth toward +Z, centred, X along the widest axis, incisors toward +Y.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from ioscan.pipeline.sweep import (
    SweepConfig, build_heightmap, heightmap_to_mesh, keep_main_surface, vertex_normals, write_ply_mesh,
    write_ply_points, write_stl,
)
from ioscan.utils.plyio import read_ply


@dataclass
class StitchConfig:
    coarse_res_mm: float = 0.3
    yaw_step_deg: float = 3.0
    min_overlap_frac: float = 0.12        # of the smaller footprint, for a coarse match to count
    icp_iters: int = 40
    icp_max_dist_mm: float = 1.5
    icp_final_dist_mm: float = 0.4
    icp_samples: int = 25000
    global_rounds: int = 3
    map_voxel_mm: float = 0.2
    imu_tiebreak_ncc: float = 0.08        # candidates within this NCC of the best are "ties"
    # The IMU only sees the wand turn. If the operator turns the model between passes it reports a
    # misleading angle, so it is a cross-check by default, not a tie-breaker.
    use_imu_tiebreak: bool = False
    imu_warn_deg: float = 20.0
    weak_overlap_frac: float = 0.15
    weak_rmse_mm: float = 0.25
    # Keep teeth plus this much gum around them (None: keep the whole model base). Intraoral scans
    # used to train MeshSegNet contain teeth and a gingival band, not a large flat base.
    trim_to_teeth_mm: float | None = 5.0
    tooth_relief_mm: float = 1.5
    fuse: SweepConfig = field(default_factory=lambda: SweepConfig(fill_holes_mm=0.9, min_island_cells=200,
                                                                  max_edge_jump_mm=1.5))


@dataclass
class Patch:
    name: str
    points: np.ndarray                    # (N,3) in the patch's own scan frame
    quat_wxyz: np.ndarray | None = None   # mean IMU orientation over the pass
    info: dict = field(default_factory=dict)
    ambient: dict | None = None           # reference ambient image defining the model frame (planar passes)
    views: np.ndarray | None = None       # (N,3) unit vectors from each point towards the camera (plate scans)


# ------------------------------------------------------------------ small geometry helpers
def _T(R=None, t=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def apply(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    return P @ T[:3, :3].T + T[:3, 3]


def rot_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def voxel_thin(P: np.ndarray, voxel: float) -> np.ndarray:
    if len(P) == 0:
        return P
    key = np.floor(P / voxel).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return P[np.sort(idx)]


def estimate_normals(P: np.ndarray, tree: cKDTree, k: int = 12) -> np.ndarray:
    _, idx = tree.query(P, k=min(k, len(P)))
    nb = P[idx] - P[idx].mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb)
    _, vecs = np.linalg.eigh(cov)
    return vecs[:, :, 0]


# ------------------------------------------------------------------ patch cleaning
# ------------------------------------------------------------------ loading
def mean_quaternion(q_wxyz: np.ndarray) -> np.ndarray | None:
    q = np.asarray(q_wxyz, dtype=np.float64)
    if len(q) == 0:
        return None
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    q[(q @ q[0]) < 0] *= -1.0
    m = q.mean(axis=0)
    return m / np.linalg.norm(m)


def load_patch(results_dir: str | Path, session_dir: str | Path | None = None) -> Patch:
    """Load a sweep result (fused_cloud.ply + pipeline_report.json) as a stitch patch."""
    rd = Path(results_dir)
    cloud = rd / "fused_cloud.ply"
    if not cloud.is_file():
        raise FileNotFoundError(f"{cloud} not found: reconstruct the pass in sweep mode first")
    P_raw, _ = read_ply(cloud)
    report = {}
    rp = rd / "pipeline_report.json"
    if rp.is_file():
        report = json.loads(rp.read_text(encoding="utf-8"))
    on_plate = (report.get("translation") or {}).get("used") == "markers"
    views = _read_ply_normals(cloud)
    if views is not None and len(views) == len(P_raw):
        PU, clean = keep_main_surface(np.hstack([P_raw, views]), on_plate=on_plate)
        P, views = PU[:, :3], PU[:, 3:]
    else:
        views = None
        # on the plate the sheet is already gone: gums and base below the crowns are part of the model
        P, clean = keep_main_surface(P_raw, on_plate=on_plate)
    if report and report.get("pose_mode") not in (None, "sweep"):
        raise ValueError(f"{rd.name}: only sweep-mode passes can be stitched (got {report.get('pose_mode')})")
    sd = Path(session_dir) if session_dir else None
    if sd is None and report.get("session"):
        cand = Path(report["session"])
        if not cand.is_dir():  # reports store absolute paths; fall back to the project layout
            cand = rd.parents[1] / "sessions" / rd.name
        sd = cand if cand.is_dir() else None
    quat = None
    if sd is not None and (sd / "imu.csv").is_file() and (sd / "frames.csv").is_file():
        try:
            imu = np.atleast_2d(np.loadtxt(sd / "imu.csv", delimiter=",", skiprows=1))
            t0_ns = float(np.loadtxt(sd / "frames.csv", delimiter=",", skiprows=1, usecols=1, max_rows=1))
            win = report.get("active_window") or {}
            a = t0_ns + 1e9 * float(win.get("t_start_s") or 0.0)
            b = t0_ns + 1e9 * float(win.get("t_end_s") or 1e9)
            sel = (imu[:, 0] >= a) & (imu[:, 0] <= b)
            if sel.sum() >= 3:
                quat = mean_quaternion(imu[sel, 1:5])
        except (OSError, ValueError, IndexError):
            quat = None
    ambient = None
    if (rd / "ambient_reference.png").is_file() and (rd / "ambient_reference.json").is_file():
        img = cv2.imread(str(rd / "ambient_reference.png"), cv2.IMREAD_GRAYSCALE)
        msk = cv2.imread(str(rd / "ambient_reference_mask.png"), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            ambient = {"image": img, "mask": msk, **json.loads((rd / "ambient_reference.json").read_text(encoding="utf-8"))}
    tr_rep = report.get("translation") or {}
    return Patch(rd.name, P, quat, ambient=ambient, views=views, info={"results_dir": str(rd), "session_dir": str(sd) if sd else None,
                                    "points": int(len(P)), "cleaning": clean,
                                    "motion_source": tr_rep.get("used"),
                                    "marker_board": (tr_rep.get("markers") or {}).get("board"),
                                    "imu_rotation_span_deg": (report.get("quality") or {}).get("imu_rotation_span_deg")})


# ------------------------------------------------------------------ ambient-image registration
def _read_ply_normals(path: str | Path) -> np.ndarray | None:
    """nx, ny, nz of a binary point PLY written by write_ply_points (None if absent)."""
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                return None
            line = line.decode("ascii", "replace").strip()
            header.append(line)
            if line == "end_header":
                break
        body = f.read()
    if "format binary_little_endian 1.0" not in header:
        return None
    props, n = [], 0
    in_vertex = False
    for line in header:
        parts = line.split()
        if parts[:2] == ["element", "vertex"]:
            n, in_vertex = int(parts[2]), True
        elif parts and parts[0] == "element":
            in_vertex = False
        elif in_vertex and parts[:1] == ["property"]:
            if parts[1] != "float":
                return None
            props.append(parts[2])
    if not {"nx", "ny", "nz"} <= set(props) or len(body) < n * 4 * len(props):
        return None
    data = np.frombuffer(body[:n * 4 * len(props)], dtype="<f4").reshape(n, len(props))
    return data[:, [props.index("nx"), props.index("ny"), props.index("nz")]].astype(np.float64)


def _window_mask(img: np.ndarray) -> np.ndarray:
    """Lit window opening of an ambient image, eroded so the (static) window frame is excluded."""
    b = (cv2.GaussianBlur(img, (0, 0), 3) > np.percentile(img, 60)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b)
    if n < 2:
        return np.full(img.shape, 255, np.uint8)
    win = (lab == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))).astype(np.uint8)
    return cv2.erode(win, np.ones((15, 15), np.uint8)) * 255


def ambient_align(prev: Patch, cur: Patch, min_inliers: int = 25) -> tuple[np.ndarray | None, dict]:
    """Model-frame transform cur -> prev from the passes' ambient reference images.

    The camera sees the whole model through the window, so two passes can be registered from tooth
    outlines, gaps and the model's edge across the arch, not just the small laser overlap.
    """
    a, b = prev.ambient, cur.ambient
    if not a or not b:
        return None, {"reason": "no ambient reference"}
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    ia, ib = clahe.apply(a["image"]), clahe.apply(b["image"])
    ma, mb = _window_mask(a["image"]), _window_mask(b["image"])
    sift = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.015)
    ka, da = sift.detectAndCompute(ia, ma)
    kb, db = sift.detectAndCompute(ib, mb)
    if da is None or db is None or len(ka) < min_inliers or len(kb) < min_inliers:
        return None, {"reason": "too few image features", "features": [len(ka or []), len(kb or [])]}
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(db, da, k=2)
    good = [m for m, n in (x for x in matches if len(x) == 2) if m.distance < 0.8 * n.distance]
    if len(good) < min_inliers:
        return None, {"reason": "too few feature matches", "matches": len(good)}
    pb = np.float32([kb[m.queryIdx].pt for m in good])
    pa = np.float32([ka[m.trainIdx].pt for m in good])
    M, inl = cv2.estimateAffinePartial2D(pb, pa, method=cv2.RANSAC, ransacReprojThreshold=2.5,
                                         maxIters=5000, confidence=0.999)
    if M is None:
        return None, {"reason": "no consistent image transform"}
    n_in = int(inl.sum())
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    info = {"matches": len(good), "inliers": n_in, "inlier_ratio": round(n_in / len(good), 3), "scale": round(scale, 4)}
    if n_in < min_inliers or not 0.85 <= scale <= 1.15:
        info["reason"] = "image match not reliable"
        return None, info
    k = 1.0 / float(a.get("scale", 0.5))                  # stored image -> full resolution
    R2 = M[:, :2] / scale
    c = np.array([a["cx"], a["cy"]])
    d = scale * R2 @ c + k * M[:, 2] - c                    # full-res translation about the principal point
    z = float(a.get("tracking_depth_mm") or np.median(prev.points[:, 2]))
    D = np.array([d[0] * z / a["fx"], d[1] * z / a["fy"]])
    T = np.eye(4)
    T[:2, :2] = R2
    T[:2, 3] = D
    moved = apply(T, cur.points)
    tree = cKDTree(prev.points[:, :2])
    dist, idx = tree.query(moved[:, :2], distance_upper_bound=0.6)
    ok = np.isfinite(dist)
    if ok.sum() > 50:
        T[2, 3] = float(np.median(prev.points[idx[ok], 2] - moved[ok, 2]))
    info["rotation_deg"] = round(float(np.degrees(np.arctan2(R2[1, 0], R2[0, 0]))), 2)
    info["translation_mm"] = [round(float(D[0]), 2), round(float(D[1]), 2)]
    return T, info


# ------------------------------------------------------------------ coarse registration
def rasterize(P: np.ndarray, res: float, origin: np.ndarray | None = None, shape=None):
    """Height map (height = -z, i.e. towards the camera) on an XY grid. Returns (H, M, origin)."""
    xy = P[:, :2]
    if origin is None:
        origin = np.floor(xy.min(axis=0) / res) * res - 2 * res
    if shape is None:
        ext = xy.max(axis=0) - origin
        shape = (int(np.ceil(ext[1] / res)) + 3, int(np.ceil(ext[0] / res)) + 3)
    ix = np.floor((xy[:, 0] - origin[0]) / res).astype(int)
    iy = np.floor((xy[:, 1] - origin[1]) / res).astype(int)
    ok = (ix >= 0) & (iy >= 0) & (ix < shape[1]) & (iy < shape[0])
    cnt = np.zeros(shape)
    hs = np.zeros(shape)
    np.add.at(cnt, (iy[ok], ix[ok]), 1.0)
    np.add.at(hs, (iy[ok], ix[ok]), -P[ok, 2])
    M = cnt > 0
    H = np.where(M, hs / np.maximum(cnt, 1), 0.0)
    # close pinholes so masks are solid footprints
    Mc = ndimage.binary_closing(M, iterations=2) & ndimage.binary_fill_holes(M)
    if (Mc & ~M).any():
        w = ndimage.gaussian_filter(M.astype(float), 1.5)
        v = ndimage.gaussian_filter(H, 1.5)
        H = np.where(M, H, np.where(Mc, v / np.maximum(w, 1e-9), 0.0))
    return H, Mc, origin


def highpass(H: np.ndarray, M: np.ndarray, sigma_cells: float) -> np.ndarray:
    """Remove the smooth gum/base slope so correlation is driven by tooth relief."""
    w = ndimage.gaussian_filter(M.astype(float), sigma_cells)
    v = ndimage.gaussian_filter(np.where(M, H, 0.0), sigma_cells)
    return np.where(M, H - v / np.maximum(w, 1e-9), 0.0)


class _FixedSpectra:
    """FFTs of the fixed raster's terms for masked NCC (Padfield 2010)."""

    def __init__(self, F: np.ndarray, Mf: np.ndarray, pad_shape):
        self.shape = pad_shape
        Ff = F * Mf
        self.Mf = np.fft.rfft2(Mf.astype(float), pad_shape)
        self.F1 = np.fft.rfft2(Ff, pad_shape)
        self.F2 = np.fft.rfft2(Ff * F, pad_shape)
        self.area = float(Mf.sum())


def masked_ncc(fs: _FixedSpectra, G: np.ndarray, Mg: np.ndarray, min_overlap: float):
    """NCC for every shift of the moving raster G over the fixed one. Returns (ncc, overlap)."""
    Gr, Mr = G[::-1, ::-1] * Mg[::-1, ::-1], Mg[::-1, ::-1].astype(float)
    s = fs.shape
    Mg_ = np.fft.rfft2(Mr, s)
    G1 = np.fft.rfft2(Gr, s)
    G2 = np.fft.rfft2(Gr * G[::-1, ::-1], s)
    ov = np.fft.irfft2(fs.Mf * Mg_, s)
    sF = np.fft.irfft2(fs.F1 * Mg_, s)
    sG = np.fft.irfft2(fs.Mf * G1, s)
    sF2 = np.fft.irfft2(fs.F2 * Mg_, s)
    sG2 = np.fft.irfft2(fs.Mf * G2, s)
    sFG = np.fft.irfft2(fs.F1 * G1, s)
    ov = np.round(ov)
    with np.errstate(invalid="ignore", divide="ignore"):
        num = sFG - sF * sG / ov
        den = np.sqrt(np.clip(sF2 - sF ** 2 / ov, 0, None) * np.clip(sG2 - sG ** 2 / ov, 0, None))
        ncc = num / den
    ncc[(ov < min_overlap) | ~np.isfinite(ncc) | (den < 1e-6)] = -1.0
    return ncc, ov


def _peaks(ncc: np.ndarray, ov: np.ndarray, k: int, sep: int) -> list[tuple[int, int]]:
    """Top-k local maxima of the overlap-weighted NCC, at least ``sep`` cells apart."""
    score = np.where(ncc > 0, ncc * np.sqrt(np.maximum(ov, 0)), -1.0)
    mx = ndimage.maximum_filter(score, size=2 * sep + 1)
    ys, xs = np.nonzero((score == mx) & (score > 0))
    order = np.argsort(-score[ys, xs])[:k]
    return [(int(ys[i]), int(xs[i])) for i in order]


def coarse_candidates(map_pts: np.ndarray, pts: np.ndarray, cfg: StitchConfig, n_keep: int = 8) -> list[dict]:
    """Yaw + XY + Z hypotheses for ``pts`` on ``map_pts`` from high-passed height-map correlation."""
    res = cfg.coarse_res_mm
    sig = 2.0 / res
    F, Mf, fo = rasterize(map_pts, res)
    Fh = highpass(F, Mf, sig)
    c = pts.mean(axis=0)
    rel = pts - c
    probe = rasterize(rel, res)[1]
    max_dim = int(np.ceil(np.hypot(*probe.shape))) + 4
    pad = (F.shape[0] + max_dim, F.shape[1] + max_dim)
    fs = _FixedSpectra(Fh, Mf, pad)
    raw = []
    for yaw in np.arange(0.0, 360.0, cfg.yaw_step_deg):
        Rz = Rotation.from_euler("z", yaw, degrees=True).as_matrix()
        G, Mg, go = rasterize(rel @ Rz.T, res)
        Gh = highpass(G, Mg, sig)
        ncc, ov = masked_ncc(fs, Gh, Mg, cfg.min_overlap_frac * min(Mg.sum(), fs.area))
        for iy, ix in _peaks(ncc, ov, 3, int(3.0 / res)):
            oy, ox = iy - (G.shape[0] - 1), ix - (G.shape[1] - 1)
            t_xy = fo + np.array([ox, oy]) * res - go
            raw.append({"ncc": float(ncc[iy, ix]), "overlap_cells": float(ov[iy, ix]), "yaw_deg": float(yaw),
                        "t_xy": t_xy, "score": float(ncc[iy, ix]) * np.sqrt(float(ov[iy, ix]))})
    raw.sort(key=lambda d: -d["score"])
    kept: list[dict] = []
    for cd in raw:
        dup = any(min(abs(cd["yaw_deg"] - k["yaw_deg"]), 360 - abs(cd["yaw_deg"] - k["yaw_deg"])) < 8.0
                  and np.linalg.norm(cd["t_xy"] - k["t_xy"]) < 3.0 for k in kept)
        if not dup:
            kept.append(cd)
        if len(kept) >= n_keep:
            break
    tree2 = cKDTree(map_pts[:, :2])
    for cd in kept:
        Rz = Rotation.from_euler("z", cd["yaw_deg"], degrees=True).as_matrix()
        Q = rel @ Rz.T
        Q[:, :2] += cd["t_xy"]
        d, idx = tree2.query(Q[:, :2], distance_upper_bound=res * 2)
        ok = np.isfinite(d)
        dz = float(np.median(map_pts[idx[ok], 2] - Q[ok, 2])) if ok.sum() > 50 else 0.0
        cd["T"] = _T(Rz, np.array([cd["t_xy"][0], cd["t_xy"][1], dz]) - Rz @ c)
    return kept


def relief_agreement(map_pts: np.ndarray, moved_pts: np.ndarray, res: float = 0.3) -> tuple[float, int]:
    """NCC of tooth relief (high-passed height) where the two clouds overlap, and the overlap size.

    Flat gum or base fits anywhere, so contact area alone cannot judge an alignment; the relief
    of cusps and grooves only correlates at the right place.
    """
    lo = np.minimum(map_pts[:, :2].min(axis=0), moved_pts[:, :2].min(axis=0))
    hi = np.maximum(map_pts[:, :2].max(axis=0), moved_pts[:, :2].max(axis=0))
    origin = np.floor(lo / res) * res - 2 * res
    shape = (int(np.ceil((hi[1] - origin[1]) / res)) + 3, int(np.ceil((hi[0] - origin[0]) / res)) + 3)
    A, Ma, _ = rasterize(map_pts, res, origin, shape)
    B, Mb, _ = rasterize(moved_pts, res, origin, shape)
    sig = 2.0 / res
    Ah, Bh = highpass(A, Ma, sig), highpass(B, Mb, sig)
    m = ndimage.binary_erosion(Ma & Mb, iterations=2)
    n = int(m.sum())
    if n < 50:
        return -1.0, n
    a, b = Ah[m] - Ah[m].mean(), Bh[m] - Bh[m].mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return (float((a * b).sum() / den) if den > 1e-9 else -1.0), n


def coarse_align(map_pts: np.ndarray, pts: np.ndarray, cfg: StitchConfig, prev_R: np.ndarray | None = None,
                 imu_angle: float | None = None, map_tree: cKDTree | None = None,
                 map_normals: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Pick the hypothesis that a quick ICP confirms best (most points in contact, low residual)."""
    cands = coarse_candidates(map_pts, pts, cfg)
    if not cands:
        raise RuntimeError("No overlap found between this pass and the passes before it.")
    tree = map_tree or cKDTree(map_pts)
    N = map_normals if map_normals is not None else estimate_normals(map_pts, tree)
    quick = StitchConfig(icp_iters=18, icp_samples=6000, icp_max_dist_mm=1.5, icp_final_dist_mm=0.4)
    for cd in cands:
        T, ii = icp_point_to_plane(pts, map_pts, cd["T"], quick, tree, N)
        cd["T_icp"], cd["icp"] = T, ii
        ncc, n = relief_agreement(map_pts, apply(T, pts), cfg.coarse_res_mm)
        cd["relief_ncc"] = ncc
        # relief agreement decides; overlap size only breaks near-ties
        cd["support"] = max(ncc, 0.0) * (n ** 0.15)
    cands.sort(key=lambda d: -d["support"])
    best = cands[0]
    reason = "best ICP support"
    if cfg.use_imu_tiebreak and imu_angle is not None and prev_R is not None and len(cands) > 1:
        ties = [cd for cd in cands[1:] if cd["support"] >= 0.9 * best["support"]]
        if ties:
            pool = [best] + ties
            pick = min(pool, key=lambda cd: abs(rot_angle_deg(prev_R.T @ cd["T_icp"][:3, :3]) - imu_angle))
            if pick is not best:
                best, reason = pick, "IMU tie-break"
    info = {"ncc": round(best["ncc"], 4), "yaw_deg": round(best["yaw_deg"] % 360, 2),
            "overlap_cells": int(best["overlap_cells"]), "support": round(best["support"], 4),
            "relief_ncc": round(best["relief_ncc"], 4),
            "selection": reason, "candidates": len(cands),
            "runner_up_support": round(cands[1]["support"], 4) if len(cands) > 1 else None}
    return best["T_icp"], info


# ------------------------------------------------------------------ ICP
def icp_point_to_plane(src: np.ndarray, dst: np.ndarray, T0: np.ndarray, cfg: StitchConfig,
                       dst_tree: cKDTree | None = None, dst_normals: np.ndarray | None = None):
    rng = np.random.default_rng(0)
    if len(src) > cfg.icp_samples:
        src = src[rng.choice(len(src), cfg.icp_samples, replace=False)]
    tree = dst_tree or cKDTree(dst)
    N = dst_normals if dst_normals is not None else estimate_normals(dst, tree)
    T = T0.copy()
    rmse, frac = np.inf, 0.0
    for it in range(cfg.icp_iters):
        md = cfg.icp_max_dist_mm + (cfg.icp_final_dist_mm - cfg.icp_max_dist_mm) * min(1.0, it / (0.6 * cfg.icp_iters))
        P = apply(T, src)
        d, idx = tree.query(P, distance_upper_bound=md)
        ok = np.isfinite(d)
        if ok.sum() < 100:
            break
        med = np.median(d[ok])
        ok &= d <= max(3.0 * med, 0.05)
        p, q, n = P[ok], dst[idx[ok]], N[idx[ok]]
        r = np.einsum("ij,ij->i", q - p, n)
        A = np.hstack([np.cross(p, n), n])
        x, *_ = np.linalg.lstsq(A, r, rcond=None)
        dR = Rotation.from_rotvec(x[:3]).as_matrix()
        T = _T(dR, x[3:]) @ T
        rmse = float(np.sqrt(np.mean(r ** 2)))
        frac = float(ok.mean())
        if np.linalg.norm(x[:3]) < 1e-6 and np.linalg.norm(x[3:]) < 1e-5:
            break
    P = apply(T, src)
    d, _ = tree.query(P, distance_upper_bound=cfg.icp_final_dist_mm)
    close = np.isfinite(d)
    return T, {"rmse_mm": round(float(np.sqrt(np.mean(d[close] ** 2))) if close.any() else None or 0.0, 4),
               "overlap_frac": round(float(close.mean()), 4), "pairs_frac": round(frac, 4)}


# ------------------------------------------------------------------ orchestration
def _view_dir(T: np.ndarray) -> np.ndarray:
    return T[:3, :3] @ np.array([0.0, 0.0, 1.0])


def stitch_patches(patches: list[Patch], cfg: StitchConfig | None = None, progress=None) -> tuple[list[np.ndarray], dict]:
    """Align patches (in scan order) into the frame of the first one. Returns (transforms, report)."""
    cfg = cfg or StitchConfig()
    if len(patches) < 2:
        raise ValueError("Stitching needs at least two passes.")
    Ts = [np.eye(4)]
    links = []
    for i in range(1, len(patches)):
        if progress:
            progress(f"Aligning pass {i + 1} of {len(patches)}", 100.0 * i / (len(patches) + cfg.global_rounds))
        p = patches[i]
        imu_angle = None
        if p.quat_wxyz is not None and patches[i - 1].quat_wxyz is not None:
            qa, qb = patches[i - 1].quat_wxyz, p.quat_wxyz
            ra = Rotation.from_quat([qa[1], qa[2], qa[3], qa[0]])
            rb = Rotation.from_quat([qb[1], qb[2], qb[3], qb[0]])
            imu_angle = float(np.degrees((ra.inv() * rb).magnitude()))
        # Match against the previous pass only: the protocol guarantees that overlap, and it avoids
        # snapping onto look-alike teeth elsewhere (left and right molars are near mirror images).
        ref = voxel_thin(apply(Ts[i - 1], patches[i - 1].points), cfg.map_voxel_mm)
        tree = cKDTree(ref)
        normals = estimate_normals(ref, tree)
        T_img, ainfo = ambient_align(patches[i - 1], p)
        if T_img is not None:
            # the whole-arch image match is the starting point; ICP refines it on the laser overlap
            T0 = Ts[i - 1] @ T_img
            quick = StitchConfig(icp_iters=25, icp_samples=8000, icp_max_dist_mm=1.0, icp_final_dist_mm=0.4)
            T0, _ = icp_point_to_plane(p.points, ref, T0, quick, tree, normals)
            ncc, _n = relief_agreement(ref, apply(T0, p.points), cfg.coarse_res_mm)
            cinfo = {"selection": f"ambient image match ({ainfo['inliers']} inliers)", "ambient": ainfo,
                     "relief_ncc": round(ncc, 4), "ncc": None, "runner_up_support": None}
        else:
            T0, cinfo = coarse_align(ref, p.points, cfg, prev_R=Ts[i - 1][:3, :3], imu_angle=imu_angle,
                                     map_tree=tree, map_normals=normals)
            cinfo["ambient"] = ainfo
        T, iinfo = icp_point_to_plane(p.points, ref, T0, cfg, tree, normals)
        reg_angle = rot_angle_deg(Ts[i - 1][:3, :3].T @ T[:3, :3])
        link = {"pass": p.name, "coarse": cinfo, "icp": iinfo,
                "rotation_vs_previous_deg": round(reg_angle, 2),
                "imu_rotation_vs_previous_deg": None if imu_angle is None else round(imu_angle, 2)}
        Ts.append(T)
        links.append(link)
    # global refinement: each pass against the union of the others (pass 0 fixed)
    for rnd in range(cfg.global_rounds):
        if progress:
            progress(f"Global refinement {rnd + 1}/{cfg.global_rounds}",
                     100.0 * (len(patches) + rnd) / (len(patches) + cfg.global_rounds))
        for i in range(1, len(patches)):
            others = voxel_thin(np.vstack([apply(Ts[j], patches[j].points) for j in range(len(patches)) if j != i]),
                                cfg.map_voxel_mm)
            tree = cKDTree(others)
            Ts[i], iinfo = icp_point_to_plane(patches[i].points, others, Ts[i], cfg, tree,
                                              estimate_normals(others, tree))
            links[i - 1]["icp_global"] = iinfo
    warnings = []
    unreliable = 0
    for i, lk in enumerate(links, start=1):
        g = lk.get("icp_global", lk["icp"])
        co = lk["coarse"]
        sup, run = co.get("support"), co.get("runner_up_support")
        if sup and run and run >= 0.95 * sup:
            unreliable += 1
            lk["reliable"] = False
            warnings.append(
                f"Pass {lk['pass']}: AMBIGUOUS joint: another placement fits almost as well "
                f"({run:.3f} vs {sup:.3f}). The overlap with the previous pass is too small or too uniform; "
                "rescan with two or more shared teeth, or scan the arch as one continuous session.")
        elif g["overlap_frac"] > 0.85:
            unreliable += 1
            lk["reliable"] = False
            warnings.append(
                f"Pass {lk['pass']} lies almost entirely on top of the previous pass "
                f"({100 * g['overlap_frac']:.0f}% overlap). Either it rescanned the same teeth or the match failed.")
        else:
            lk["reliable"] = True
        if g["overlap_frac"] < cfg.weak_overlap_frac or (g["rmse_mm"] or 0) > cfg.weak_rmse_mm:
            warnings.append(
                f"Pass {lk['pass']} is weakly connected (overlap {100 * g['overlap_frac']:.0f}%, "
                f"RMS {g['rmse_mm']:.3f} mm). Rescan it with one or two teeth of overlap with its neighbours.")
        if lk["imu_rotation_vs_previous_deg"] is not None:
            reg = rot_angle_deg(Ts[i - 1][:3, :3].T @ Ts[i][:3, :3])
            lk["rotation_vs_previous_deg"] = round(reg, 2)
            if abs(reg - lk["imu_rotation_vs_previous_deg"]) > cfg.imu_warn_deg:
                warnings.append(
                    f"Pass {lk['pass']}: geometry says the view turned {reg:.0f} deg from the previous pass, "
                    f"the IMU says the wand turned {lk['imu_rotation_vs_previous_deg']:.0f} deg. Fine if the model "
                    "was rotated between passes; otherwise check this joint in the viewer.")
    return Ts, {"passes": [p.name for p in patches], "links": links, "warnings": warnings,
                "unreliable_joints": unreliable}


def _canonical_frame(V: np.ndarray, view: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Rotation/centre so the arch lies in XY, teeth toward +Z, incisors toward +Y."""
    z = -view / np.linalg.norm(view)                    # towards the camera = occlusal
    a = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x0 = np.cross(a, z); x0 /= np.linalg.norm(x0)
    y0 = np.cross(z, x0)
    B = np.column_stack([x0, y0, z])
    mu = V.mean(axis=0)
    W = (V - mu) @ B
    evals, evecs = np.linalg.eigh(np.cov(W[:, :2].T))
    ex = evecs[:, 1]
    phi = np.arctan2(ex[1], ex[0])
    Rz = Rotation.from_euler("z", -phi).as_matrix()
    W2 = W @ Rz.T
    # incisors (single closed end of the U) toward +Y: the end with the narrower x-spread
    y = W2[:, 1]
    lo, hi = np.percentile(y, [12, 88])
    spread_lo = np.ptp(W2[y <= lo, 0]) if (y <= lo).sum() > 10 else 0
    spread_hi = np.ptp(W2[y >= hi, 0]) if (y >= hi).sum() > 10 else 0
    flip = spread_hi > spread_lo
    if flip:
        Rz = Rotation.from_euler("z", np.pi).as_matrix() @ Rz
    R = Rz @ B.T                                         # canonical = R @ (v - mu)
    return R, mu, {"incisor_end_flipped": bool(flip), "spread_ends_mm": [round(float(spread_lo), 2), round(float(spread_hi), 2)]}


def fuse_patches(patches: list[Patch], Ts: list[np.ndarray], cfg: StitchConfig | None = None):
    """Fuse aligned patches into one height-map surface; returns (V, F, N, cloud, info) in canonical frame."""
    cfg = cfg or StitchConfig()
    views = np.array([_view_dir(T) for T in Ts])
    view = views.mean(axis=0)
    view /= np.linalg.norm(view)
    if all(p.views is not None for p in patches) and all(p.info.get("motion_source") == "markers" for p in patches):
        # Plate scans keep each point's viewing direction: fuse them as a volume, so the walls seen in
        # tilted scans and the tops seen from above end up in one surface.
        P = np.vstack([apply(T, p.points) for T, p in zip(Ts, patches)])
        U = np.vstack([p.views @ T[:3, :3].T for T, p in zip(Ts, patches)])
        tilt = np.degrees(np.arccos(np.clip(-U[:, 2], -1.0, 1.0)))
        if float(np.percentile(tilt, 90)) > 20.0:
            from ioscan.pipeline.volume import fuse_views
            V, F, N, vinfo = fuse_views(P, U)
            look = np.array([0.0, 0.0, 1.0])             # plate frame: the camera looks along +Z (teeth at -Z)
            cloud = voxel_thin(P, 0.1)
            R, mu, cinfo = _canonical_frame(V, look)
            Vc, Nc, cloud_c = (V - mu) @ R.T, N @ R.T, (cloud - mu) @ R.T
            info = {"heightmap": {"area_mm2": vinfo["surface_mm2"], "mode": "volume"}, "volume": vinfo,
                    "view_tilt_spread_deg": round(float(np.percentile(tilt, 95) - np.percentile(tilt, 5)), 2),
                    "canonical": cinfo, "canonical_R": R.tolist(), "canonical_mu": mu.tolist()}
            return Vc, F, Nc, cloud_c, info
    # fusion frame: z along the mean viewing direction
    z = view
    a = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(a, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    Rv = np.column_stack([x, y, z]).T                   # common -> fusion
    P = np.vstack([apply(T, p.points) for T, p in zip(Ts, patches)]) @ Rv.T
    Z, valid, origin, hm = build_heightmap(P, cfg.fuse)
    if cfg.trim_to_teeth_mm is not None and valid.any():
        c = cfg.fuse.cell_mm
        H = np.where(valid, -Z, np.nan)
        Hf = np.where(valid, H, np.nanmin(H))
        size = max(3, int(round(10.0 / c)) | 1)               # base level: opening wider than a tooth
        base = ndimage.grey_opening(Hf, size=(size, size))
        teeth = valid & ((Hf - base) > cfg.tooth_relief_mm)
        lab, nl = ndimage.label(teeth)
        if nl:
            sizes = ndimage.sum(teeth, lab, index=np.arange(1, nl + 1))
            teeth = np.isin(lab, np.flatnonzero(sizes >= (2.0 / c) ** 2) + 1)   # >= ~2x2 mm blobs
        if teeth.sum() > 0.02 * valid.sum():
            r = int(round(cfg.trim_to_teeth_mm / c))
            keep = ndimage.distance_transform_edt(~teeth) <= r
            hm["trimmed_base_cells"] = int((valid & ~keep).sum())
            valid = valid & keep
            Z = np.where(valid, Z, np.nan)
    V, F, N = heightmap_to_mesh(Z, valid, origin, cfg.fuse)
    # back to the common frame, then canonical
    V = V @ Rv
    N = N @ Rv
    cloud = voxel_thin(P @ Rv, 0.1)
    R, mu, cinfo = _canonical_frame(V, view)
    Vc = (V - mu) @ R.T
    Nc = N @ R.T
    cloud_c = (cloud - mu) @ R.T
    if len(Nc) and Nc[:, 2].mean() < 0:                  # normals outward = towards +Z (occlusal)
        F = F[:, [0, 2, 1]]
        Nc = -Nc
    info = {"heightmap": hm, "view_tilt_spread_deg": round(float(np.degrees(np.arccos(np.clip(views @ view, -1, 1))).max()), 2),
            "canonical": cinfo, "canonical_R": R.tolist(), "canonical_mu": mu.tolist()}
    return Vc, F, Nc, cloud_c, info


def _plate_merge_check(patches: list[Patch], res: float = 0.3) -> list[str]:
    """Where two marker-plate scans overlap they must agree; a height step means the model moved."""
    out = []
    allp = np.vstack([p.points[:, :2] for p in patches])
    origin = np.floor(allp.min(axis=0) / res) * res
    shape = tuple((np.ceil((allp.max(axis=0) - origin) / res).astype(int) + 2)[::-1])
    maps = [rasterize(p.points, res, origin, shape) for p in patches]
    for i in range(1, len(patches)):
        Hi, Mi = maps[i][0], maps[i][1]
        best = None
        for j in range(i):
            both = Mi & maps[j][1]
            if both.sum() >= 150:
                d = np.abs(Hi[both] - maps[j][0][both])
                med = float(np.median(d))
                best = med if best is None else min(best, med)
        if best is not None and best > 0.6:
            out.append(f"Scan {patches[i].name} disagrees with the earlier scans by {best:.1f} mm where they overlap: "
                       "the model probably moved on the plate between scans. Fix it down and rescan.")
    return out


def stitch_to_files(results_dirs: list[str | Path], out_dir: str | Path, cfg: StitchConfig | None = None,
                    progress=None) -> dict:
    """Load sweep results, align, fuse and write reconstructed_mesh.ply/.stl, fused_cloud.ply and a report."""
    from ioscan.pipeline.quality import assess_scan_geometry
    cfg = cfg or StitchConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    patches = [load_patch(r) for r in results_dirs]
    sources = [p.info.get("motion_source") for p in patches]
    merged_on_plate = all(src == "markers" for src in sources)
    if merged_on_plate:
        # Every pass was measured in the frame of the same printed marker plate, so the passes are
        # already aligned: nothing is estimated here. (Valid as long as the model was not moved on
        # the plate between the passes.)
        if progress:
            progress("Passes share the marker-plate frame: merging without stitching", 50.0)
        Ts = [np.eye(4) for _ in patches]
        rep = {"passes": [p.name for p in patches], "links": [], "unreliable_joints": 0,
               "warnings": [f"Merged {len(patches)} marker-plate scans directly in the plate frame (no stitching). "
                            "This assumes the model stayed fixed on the plate between the scans."]}
        rep["warnings"] += _plate_merge_check(patches)
    else:
        if any(src == "markers" for src in sources):
            raise ValueError(
                "Some of the selected scans used the marker plate and some did not. Merge marker-plate scans "
                "with each other, or stitch scans made without the plate with each other.")
        Ts, rep = stitch_patches(patches, cfg, progress=progress)
    if progress:
        progress("Fusing passes into one surface", 92.0)
    V, F, N, cloud, finfo = fuse_patches(patches, Ts, cfg)
    write_ply_mesh(out / "reconstructed_mesh.ply", V, F, N)
    write_stl(out / "reconstructed_mesh.stl", V, F)
    write_ply_points(out / "fused_cloud.ply", cloud)
    geometry = assess_scan_geometry(V, F)
    extent = (V.max(axis=0) - V.min(axis=0)).tolist() if len(V) else [0, 0, 0]
    warnings = list(rep["warnings"])
    if finfo["view_tilt_spread_deg"] > 35 and "volume" not in finfo:
        warnings.append(f"Passes were viewed from directions up to {finfo['view_tilt_spread_deg']:.0f} deg apart; "
                        "steep side views do not fit one occlusal height map and may leave holes.")
    for r in geometry["reasons"]:
        warnings.append(f"Mesh-quality gate: {r}.")
    report = {
        "pose_mode": "stitched_sweeps",
        "merged_on_marker_plate": bool(merged_on_plate),
        "passes": rep["passes"],
        "links": rep["links"],
        "unreliable_joints": rep.get("unreliable_joints", 0),
        "transforms_to_first_pass": [T.tolist() for T in Ts],
        "patch_info": [p.info for p in patches],
        "fusion": finfo,
        "mesh_vertices": int(len(V)), "mesh_triangles": int(len(F)), "fused_cloud_points": int(len(cloud)),
        "quality": {"mesh_extent_mm": [round(float(e), 3) for e in extent], "geometry_gate": geometry,
                    "segmentation_ready_by_basic_geometry_gate": bool(geometry["ok"])},
        "frame": "canonical dental frame: arch in XY, teeth toward +Z, incisors toward +Y (mm)",
        "warnings": warnings,
        "paths": {"mesh_ply": str(out / "reconstructed_mesh.ply"), "mesh_stl": str(out / "reconstructed_mesh.stl"),
                  "cloud_ply": str(out / "fused_cloud.ply")},
    }
    (out / "pipeline_report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    return report
