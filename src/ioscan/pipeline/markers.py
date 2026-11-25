"""Marker plate: absolute 6-DOF pose of the model from printed ArUco markers.

A single laser line measures one profile per frame, so the motion between profiles has to come
from somewhere else. Tracking the plain white model in the ambient frames only recovers in-plane
motion and drifts. A printed marker sheet under the model gives the full pose (position, turn,
tilt, height) of the plate in every laser-OFF frame, in millimetres, without drift: every profile
is placed directly in the plate frame, however the plate was moved.

Plate frame: origin at the sheet centre, +X to the right and +Y down when reading the sheet,
+Z into the sheet (away from the camera). Teeth therefore have negative Z, the same "smaller Z is
closer to the camera" convention the rest of the sweep pipeline uses.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path

import cv2
import numpy as np

DEFAULT_BOARD_FILE = Path(__file__).resolve().parents[3] / "configs" / "marker_board.json"


# --------------------------------------------------------------------------- board layout
@dataclass(frozen=True)
class MarkerBoard:
    cols: int = 16
    rows: int = 14
    marker_mm: float = 6.0
    pitch_mm: float = 7.5
    scale: float = 1.0                    # measured print scale (measured bar length / 100 mm)
    dictionary: str = "DICT_4X4_250"

    @property
    def count(self) -> int:
        return self.cols * self.rows

    @property
    def size_mm(self) -> tuple[float, float]:
        return (((self.cols - 1) * self.pitch_mm + self.marker_mm) * self.scale,
                ((self.rows - 1) * self.pitch_mm + self.marker_mm) * self.scale)

    def marker_origin_mm(self, marker_id: int) -> tuple[float, float]:
        """Top-left corner of a marker, unscaled sheet millimetres from the grid's top-left."""
        r, c = divmod(int(marker_id), self.cols)
        return c * self.pitch_mm, r * self.pitch_mm

    def object_points(self) -> np.ndarray:
        """(count, 4, 3) corner coordinates (TL, TR, BR, BL) in the plate frame, millimetres."""
        w = (self.cols - 1) * self.pitch_mm + self.marker_mm
        h = (self.rows - 1) * self.pitch_mm + self.marker_mm
        ids = np.arange(self.count)
        r, c = np.divmod(ids, self.cols)
        x0 = c * self.pitch_mm - w / 2.0
        y0 = r * self.pitch_mm - h / 2.0
        m = self.marker_mm
        pts = np.zeros((self.count, 4, 3))
        pts[:, 0, :2] = np.column_stack([x0, y0])
        pts[:, 1, :2] = np.column_stack([x0 + m, y0])
        pts[:, 2, :2] = np.column_stack([x0 + m, y0 + m])
        pts[:, 3, :2] = np.column_stack([x0, y0 + m])
        return pts * float(self.scale)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "MarkerBoard":
        d = d or {}
        keys = ("cols", "rows", "marker_mm", "pitch_mm", "scale", "dictionary")
        return cls(**{k: d[k] for k in keys if k in d})

    @classmethod
    def load(cls, path: str | Path | None = None) -> "MarkerBoard":
        p = Path(path) if path else DEFAULT_BOARD_FILE
        try:
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            return cls()

    def save(self, path: str | Path | None = None) -> Path:
        p = Path(path) if path else DEFAULT_BOARD_FILE
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p


def aruco_dictionary(name: str):
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def marker_bits(board: MarkerBoard, marker_id: int) -> np.ndarray:
    """(n, n) array of the marker's cells including its black border: 1 = white, 0 = black."""
    d = aruco_dictionary(board.dictionary)
    n = int(d.markerSize) + 2
    img = cv2.aruco.generateImageMarker(d, int(marker_id), n)
    return (img > 127).astype(np.uint8)


def render_board_image(board: MarkerBoard, px_per_mm: float = 20.0, margin_mm: float = 4.0) -> tuple[np.ndarray, float]:
    """Raster of the marker grid (white background). Returns (image, margin_mm) for tests/previews."""
    w = (board.cols - 1) * board.pitch_mm + board.marker_mm + 2 * margin_mm
    h = (board.rows - 1) * board.pitch_mm + board.marker_mm + 2 * margin_mm
    img = np.full((int(round(h * px_per_mm)), int(round(w * px_per_mm))), 255, np.uint8)
    side = int(round(board.marker_mm * px_per_mm))
    for i in range(board.count):
        x0, y0 = board.marker_origin_mm(i)
        bits = marker_bits(board, i) * 255
        tile = cv2.resize(bits, (side, side), interpolation=cv2.INTER_NEAREST)
        px, py = int(round((x0 + margin_mm) * px_per_mm)), int(round((y0 + margin_mm) * px_per_mm))
        img[py:py + side, px:px + side] = tile
    return img, margin_mm


# --------------------------------------------------------------------------- detection
class MarkerDetector:
    """Detects the board's markers in a grayscale frame.

    ``mirrored=True`` handles an optical path with a mirror: the frame is flipped for detection and
    the pose is solved against the mirror image of the board, which is again a rigid frame.
    """

    def __init__(self, board: MarkerBoard, mirrored: bool = False):
        self.board = board
        self.mirrored = bool(mirrored)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 4
        params.adaptiveThreshWinSizeMin = 9
        params.adaptiveThreshWinSizeMax = 33
        params.adaptiveThreshWinSizeStep = 12
        params.minMarkerPerimeterRate = 0.02
        params.maxMarkerPerimeterRate = 1.2
        params.polygonalApproxAccuracyRate = 0.06
        params.errorCorrectionRate = 0.5
        self._det = cv2.aruco.ArucoDetector(aruco_dictionary(board.dictionary), params)
        obj = board.object_points()
        if self.mirrored:
            obj = obj * np.array([-1.0, 1.0, 1.0])
        self.object_points = obj

    def detect(self, gray: np.ndarray, box: tuple[int, int, int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (ids (M,), corners (M, 4, 2)) in full-frame pixel coordinates."""
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        x0 = y0 = 0
        img = gray
        if box is not None:
            x0, y0, x1, y1 = box
            img = gray[y0:y1, x0:x1]
        w = img.shape[1]
        if self.mirrored:
            img = cv2.flip(img, 1)
        corners, ids, _rej = self._det.detectMarkers(img)
        if ids is None or len(ids) == 0:
            return np.empty(0, np.int64), np.empty((0, 4, 2))
        ids = ids.ravel().astype(np.int64)
        c = np.asarray(corners, dtype=np.float64).reshape(-1, 4, 2)
        ok = ids < self.board.count
        ids, c = ids[ok], c[ok]
        if self.mirrored:
            c[:, :, 0] = (w - 1) - c[:, :, 0]
        c[:, :, 0] += x0
        c[:, :, 1] += y0
        # a marker seen twice (rare) is ambiguous: drop both
        uniq, cnt = np.unique(ids, return_counts=True)
        keep = np.isin(ids, uniq[cnt == 1])
        return ids[keep], c[keep]


def probe_orientation(frames: list[np.ndarray], board: MarkerBoard) -> dict:
    """Decide whether markers are visible and whether the optical path is mirrored."""
    out = {}
    for mirrored in (False, True):
        det = MarkerDetector(board, mirrored)
        counts = [len(det.detect(f)[0]) for f in frames]
        out[mirrored] = np.asarray(counts)
    best = bool(out[True].sum() > out[False].sum())
    counts = out[best]
    return {"mirrored": best, "frames": int(len(frames)),
            "frames_with_2plus": int((counts >= 2).sum()),
            "fraction_with_2plus": round(float((counts >= 2).mean()), 4) if len(counts) else 0.0,
            "median_markers": float(np.median(counts)) if len(counts) else 0.0}


# --------------------------------------------------------------------------- pose
def _rms_px(obj: np.ndarray, img_n: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, f: float) -> tuple[float, np.ndarray]:
    R, _ = cv2.Rodrigues(rvec)
    p = obj @ R.T + tvec.reshape(1, 3)
    proj = p[:, :2] / np.maximum(p[:, 2:3], 1e-9)
    err = np.linalg.norm(proj - img_n, axis=1) * f
    return float(np.sqrt(np.mean(err ** 2))), err


def _ippe(obj: np.ndarray, img_n: np.ndarray, f: float, prev_R: np.ndarray | None):
    """Best of the two planar-pose candidates: (rvec, tvec, rms_px, per-point error px) or None."""
    eye = np.eye(3)
    try:
        _n, rvecs, tvecs, _e = cv2.solvePnPGeneric(obj, img_n, eye, None, flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return None
    cands = []
    for rv, tv in zip(rvecs, tvecs):
        try:
            rv, tv = cv2.solvePnPRefineLM(obj, img_n, eye, None, rv, tv)
        except cv2.error:
            pass
        if float(tv.ravel()[2]) <= 0:
            continue
        rms, err = _rms_px(obj, img_n, rv, tv, f)
        cands.append((rms, rv, tv, err))
    if not cands:
        return None
    cands.sort(key=lambda c: c[0])
    pick = cands[0]
    if prev_R is not None and len(cands) > 1 and cands[1][0] < 1.6 * cands[0][0] + 0.15:
        def ang(c):
            R, _ = cv2.Rodrigues(c[1])
            return float(np.arccos(np.clip((np.trace(R @ prev_R.T) - 1.0) / 2.0, -1.0, 1.0)))
        pick = min(cands[:2], key=ang)
    rms, rv, tv, err = pick
    return rv, tv, rms, err


def solve_board_pose(obj: np.ndarray, img_px: np.ndarray, K: np.ndarray, dist: np.ndarray | None,
                     prev_R: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float, np.ndarray] | None:
    """Pose of the plate from marker corners (4 per marker, marker by marker).

    Returns (R, t, rms_px, inlier mask over points) or None.

    * Wrongly decoded markers are removed first with a RANSAC homography (plate -> image).
    * A planar target has two candidate poses (tilt +a / -a); both are refined and the one with the
      lower reprojection error wins, unless they are nearly tied, in which case the one closer to
      the previous frame is taken.
    * Defocus rounds the corners of every black square towards its centre. The marker centres stay
      correct, so the detected squares are rescaled about their centres to the size the pose
      predicts, and the pose is solved once more.
    """
    obj = np.asarray(obj, dtype=np.float64).reshape(-1, 3)
    img_px = np.asarray(img_px, dtype=np.float64).reshape(-1, 2)
    if len(obj) < 8:
        return None
    f = float(K[0, 0])
    dist = np.zeros(5) if dist is None else np.asarray(dist, dtype=np.float64)
    img_n = cv2.undistortPoints(img_px.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    keep = np.ones(len(obj), dtype=bool)
    if len(obj) >= 12:
        H, inl = cv2.findHomography(obj[:, :2], img_n, cv2.RANSAC, 4.0 / f)
        if H is not None and inl is not None:
            m_ok = inl.ravel().astype(bool).reshape(-1, 4).all(axis=1)
            if m_ok.sum() >= 2:
                keep = np.repeat(m_ok, 4)
    sol = _ippe(obj[keep], img_n[keep], f, prev_R)
    if sol is None:
        return None
    rv, tv, rms, err = sol
    # reject whole markers that still disagree
    m_err = err.reshape(-1, 4).max(axis=1)
    bad = m_err > max(3.0, 4.0 * float(np.median(m_err)))
    if bad.any() and (~bad).sum() >= 2:
        idx = np.flatnonzero(keep)
        keep[idx.reshape(-1, 4)[bad].ravel()] = False
        sol = _ippe(obj[keep], img_n[keep], f, prev_R)
        if sol is None:
            return None
        rv, tv, rms, err = sol
    # corner-rounding (blur) correction
    if keep.sum() >= 12:
        R, _ = cv2.Rodrigues(rv)
        p = obj[keep] @ R.T + tv.reshape(1, 3)
        proj = (p[:, :2] / p[:, 2:3]).reshape(-1, 4, 2)
        det = img_n[keep].reshape(-1, 4, 2)
        size_p = np.linalg.norm(proj - proj.mean(axis=1, keepdims=True), axis=2).mean(axis=1)
        size_d = np.linalg.norm(det - det.mean(axis=1, keepdims=True), axis=2).mean(axis=1)
        s = float(np.median(size_p / np.maximum(size_d, 1e-9)))
        if 0.85 < s < 1.15 and abs(s - 1.0) > 0.003:
            c = det.mean(axis=1, keepdims=True)
            fixed = (c + (det - c) * s).reshape(-1, 2)
            sol2 = _ippe(obj[keep], fixed, f, prev_R)
            if sol2 is not None and sol2[2] < rms:
                rv, tv, rms, err = sol2
    R, _ = cv2.Rodrigues(rv)
    return R, tv.reshape(3).astype(np.float64), float(rms), keep


@dataclass
class PoseSample:
    t: float
    R: np.ndarray            # plate -> camera
    tvec: np.ndarray
    n_markers: int
    rms_px: float
    valid: bool
    obj: np.ndarray | None = None      # (N,3) plate points of the markers used
    img_n: np.ndarray | None = None    # (N,2) their undistorted, normalised image points


class MarkerTracker:
    """Per-OFF-frame plate pose. ``add`` is called in time order (tracker thread).

    Marker detection (the expensive part) runs on a small thread pool; poses are solved strictly in
    frame order because each one uses the previous pose to resolve the planar ambiguity.
    Call ``flush()`` after the last frame.
    """

    def __init__(self, board: MarkerBoard, K: np.ndarray, dist, mirrored: bool = False,
                 min_markers: int = 2, max_rms_px: float = 2.0, workers: int | None = None):
        import os
        import threading
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor
        self.board = board
        self.mirrored = bool(mirrored)
        self.det = MarkerDetector(board, mirrored)
        self.K = np.asarray(K, dtype=np.float64)
        self.dist = None if dist is None else np.asarray(dist, dtype=np.float64)
        self.min_markers = int(min_markers)
        self.max_rms_px = float(max_rms_px)
        self.samples: list[PoseSample] = []
        self._prev_R: np.ndarray | None = None
        self._box: tuple[int, int, int, int] | None = None
        self._miss = 0
        self.reference_frame: np.ndarray | None = None
        n = workers if workers is not None else max(1, min(4, (os.cpu_count() or 2) - 1))
        self._pool = ThreadPoolExecutor(max_workers=n, thread_name_prefix="ioscan-markers") if n > 1 else None
        self._pending: deque = deque()
        self._max_pending = 3 * n
        self._local = threading.local()

    def _search_box(self, shape, corners: np.ndarray) -> tuple[int, int, int, int]:
        h, w = shape[:2]
        lo = corners.reshape(-1, 2).min(axis=0)
        hi = corners.reshape(-1, 2).max(axis=0)
        pad = 0.35 * float(max(hi - lo)) + 120.0
        return (int(max(0, lo[0] - pad)), int(max(0, lo[1] - pad)),
                int(min(w, hi[0] + pad)), int(min(h, hi[1] + pad)))

    def _detect(self, frame: np.ndarray, box) -> tuple[np.ndarray, np.ndarray]:
        det = getattr(self._local, "det", None)
        if det is None:                                     # one detector per worker thread
            det = self._local.det = MarkerDetector(self.board, self.mirrored)
        ids, corners = det.detect(frame, box)
        if len(ids) < self.min_markers and box is not None:
            ids, corners = det.detect(frame, None)          # lost in the crop: search everywhere
        return ids, corners

    def add(self, frame: np.ndarray, t: float) -> None:
        if self._pool is None:
            self._consume(frame, float(t), self._detect(frame, self._box))
            return
        self._pending.append((frame, float(t), self._pool.submit(self._detect, frame, self._box)))
        while self._pending and (self._pending[0][2].done() or len(self._pending) > self._max_pending):
            f, tt, fut = self._pending.popleft()
            self._consume(f, tt, fut.result())

    def flush(self) -> None:
        while self._pending:
            f, tt, fut = self._pending.popleft()
            self._consume(f, tt, fut.result())
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def _consume(self, frame: np.ndarray, t: float, found: tuple[np.ndarray, np.ndarray]) -> None:
        ids, corners = found
        sample = PoseSample(float(t), np.eye(3), np.zeros(3), int(len(ids)), float("inf"), False)
        if len(ids) >= self.min_markers:
            obj = self.det.object_points[ids].reshape(-1, 3)
            sol = solve_board_pose(obj, corners.reshape(-1, 2), self.K, self.dist, self._prev_R)
            if sol is not None:
                R, tv, rms, keep = sol
                n_used = int(keep.sum() // 4)
                ok = rms <= self.max_rms_px and n_used >= self.min_markers
                sample = PoseSample(float(t), R, tv, n_used, rms, bool(ok))
                if ok:
                    pts = corners.reshape(-1, 1, 2)[keep]
                    dist = np.zeros(5) if self.dist is None else self.dist
                    sample.obj = obj[keep]
                    sample.img_n = cv2.undistortPoints(pts, self.K, dist).reshape(-1, 2)
                    self._prev_R = R
                    if self.reference_frame is None:
                        self.reference_frame = frame.copy()
        if sample.valid:
            # The window through which the plate is seen does not move, so later frames are searched
            # in the union of where markers have been found (much cheaper than the whole frame).
            b = self._search_box(frame.shape, corners)
            self._box = b if self._box is None else (min(b[0], self._box[0]), min(b[1], self._box[1]),
                                                     max(b[2], self._box[2]), max(b[3], self._box[3]))
            self._miss = 0
        else:
            self._miss += 1
            if self._miss >= 15:
                self._box = None
        self.samples.append(sample)


def _rotvec(R: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(R, dtype=np.float64))[0].ravel()


def _rotmat(v: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(v, dtype=np.float64).reshape(3, 1))[0]


def _local_linear(t: np.ndarray, Y: np.ndarray, w: np.ndarray, seg: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Weighted local linear fit of each column of Y at every sample time (never across segments)."""
    out = Y.copy()
    for i in range(len(t)):
        sg = float(sigma[i])
        lo = np.searchsorted(t, t[i] - 2.5 * sg)
        hi = np.searchsorted(t, t[i] + 2.5 * sg, side="right")
        j = np.arange(lo, hi)
        j = j[seg[j] == seg[i]]
        if len(j) < 3:
            continue
        dt = t[j] - t[i]
        ww = np.exp(-0.5 * (dt / sg) ** 2) * w[j]
        A = np.column_stack([np.ones_like(dt), dt])
        Aw = A * ww[:, None]
        M = A.T @ Aw
        if abs(np.linalg.det(M)) < 1e-12 * max(M[0, 0] * M[1, 1], 1e-12):
            out[i] = (ww @ Y[j]) / max(ww.sum(), 1e-12)
            continue
        out[i] = np.linalg.solve(M, Aw.T @ Y[j])[0]
    return out


def _solve_in_plane(obj: np.ndarray, img_n: np.ndarray, R0: np.ndarray, t0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best turn about the plate normal and shift within the plate plane, the plane itself fixed."""
    x, y = R0[:, 0], R0[:, 1]
    p = np.zeros(3)                                        # theta, a, b

    def resid(q):
        c, sn = np.cos(q[0]), np.sin(q[0])
        R = R0 @ np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
        P = obj @ R.T + (t0 + q[1] * x + q[2] * y)
        return (P[:, :2] / np.maximum(P[:, 2:3], 1e-9) - img_n).ravel(), R

    for _ in range(4):
        r0, _R = resid(p)
        J = np.empty((len(r0), 3))
        for k, h in enumerate((1e-5, 1e-3, 1e-3)):
            q = p.copy()
            q[k] += h
            J[:, k] = (resid(q)[0] - r0) / h
        try:
            step = np.linalg.lstsq(J, -r0, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        p = p + step
        if np.abs(step).max() < 1e-7:
            break
    _r, R = resid(p)
    return R, t0 + p[1] * x + p[2] * y


def _expand_corners(img_n: np.ndarray, delta: float) -> np.ndarray:
    """Move every marker corner outwards along its diagonal by ``delta`` (normalised image units)."""
    q = img_n.reshape(-1, 4, 2)
    c = q.mean(axis=1, keepdims=True)
    v = q - c
    return (q + delta * v / np.maximum(np.linalg.norm(v, axis=2, keepdims=True), 1e-12)).reshape(-1, 2)


def debias_marker_corners(samples: list[PoseSample]) -> float:
    """Undo the corner rounding of an out-of-focus image; returns the correction (normalised units).

    Blur rounds the corners of each black square towards the square's centre, by the same amount
    everywhere in the image. Marker centres (their spacing) stay right while the markers look
    smaller, which biases the tilt and distance of the plate, most when only a few markers are in
    view. The one rounding amount that makes spacing and size agree best over the whole recording
    is found, the corners are moved back, and every pose is re-solved. Modifies ``samples`` in place.
    """
    val = [s for s in samples if s.valid and s.obj is not None and s.img_n is not None and len(s.obj) >= 8]
    if len(val) < 8:
        return 0.0
    probe = sorted(val, key=lambda s: -len(s.obj))[:60]
    eye = np.eye(3)

    def cost(delta: float) -> float:
        tot = 0.0
        for smp in probe:
            pts = _expand_corners(smp.img_n, delta)
            try:
                rv, tv = cv2.solvePnPRefineLM(smp.obj, pts, eye, None, _rotvec(smp.R).reshape(3, 1), smp.tvec.reshape(3, 1).copy())
            except cv2.error:
                return float("inf")
            tot += _rms_px(smp.obj, pts, rv, tv, 1.0)[0] ** 2
        return tot

    grid = np.linspace(0.0, 4.0e-3, 17)
    costs = np.array([cost(d) for d in grid])
    k = int(np.argmin(costs))
    best = float(grid[k])
    if 0 < k < len(grid) - 1:                               # parabolic refinement between grid points
        a, b, c = costs[k - 1], costs[k], costs[k + 1]
        den = a - 2 * b + c
        if den > 0:
            best += 0.5 * (a - c) / den * float(grid[1] - grid[0])
    if best <= 1e-5 or not np.isfinite(costs[k]) or costs[k] > 0.97 * costs[0]:
        return 0.0                                          # sharp image: nothing to correct
    for smp in val:
        pts = _expand_corners(smp.img_n, best)
        try:
            rv, tv = cv2.solvePnPRefineLM(smp.obj, pts, eye, None, _rotvec(smp.R).reshape(3, 1), smp.tvec.reshape(3, 1).copy())
        except cv2.error:
            continue
        if float(tv.ravel()[2]) > 0:
            smp.R, smp.tvec, smp.img_n = _rotmat(rv), tv.reshape(3).astype(np.float64), pts
    return best


def _fit_plane_track(t: np.ndarray, normal: np.ndarray, dist: np.ndarray, strength: np.ndarray, seg: np.ndarray,
                     sigma: np.ndarray, sheet: tuple[np.ndarray, np.ndarray] | None) -> tuple[np.ndarray, np.ndarray, int]:
    """Plate plane (unit normal, distance) at every sample time.

    Local linear model in time, n(dt) = n0 + n1 dt and d(dt) = d0 + d1 dt, fitted to
    * the per-frame marker planes (weak when few markers are in view), and
    * laser points that lie on the bare sheet next to the model (camera-frame millimetres): each
      must satisfy n.X = d. These are accurate to a few hundredths of a millimetre and fix the
      plate's height and its tilt along the laser line, which few small markers cannot.
    """
    LEVER = 30.0                                          # mm: balances angle and distance residuals
    n_out = normal.copy()
    d_out = dist.copy()
    used = 0
    st = sX = None
    if sheet is not None and len(sheet[0]) >= 20:
        order = np.argsort(sheet[0])
        st, sX = np.asarray(sheet[0])[order], np.asarray(sheet[1], dtype=np.float64)[order]
    for i in range(len(t)):
        sg = float(sigma[i])
        lo = np.searchsorted(t, t[i] - 2.5 * sg)
        hi = np.searchsorted(t, t[i] + 2.5 * sg, side="right")
        j = np.arange(lo, hi)
        j = j[seg[j] == seg[i]]
        if len(j) < 3:
            continue
        dt = t[j] - t[i]
        w = np.sqrt(np.exp(-0.5 * (dt / sg) ** 2) * strength[j])
        rows, rhs = [], []
        for c in range(3):                                 # normal components
            A = np.zeros((len(j), 8))
            A[:, c] = w
            A[:, 4 + c] = w * dt
            rows.append(A)
            rhs.append(w * normal[j, c])
        A = np.zeros((len(j), 8))                          # distance
        A[:, 3] = w / LEVER
        A[:, 7] = w * dt / LEVER
        rows.append(A)
        rhs.append(w * dist[j] / LEVER)
        if st is not None:
            a = np.searchsorted(st, t[j[0]] - 0.03)
            b = np.searchsorted(st, t[j[-1]] + 0.03, side="right")
            if b - a >= 6:
                k = np.arange(a, b)
                if len(k) > 400:
                    k = k[np.linspace(0, len(k) - 1, 400).astype(int)]
                dk = st[k] - t[i]
                X = sX[k]
                # trust the laser in proportion to how weak the markers are around this frame
                lam = np.sqrt(np.exp(-0.5 * (dk / sg) ** 2) * 40.0) / LEVER
                A = np.zeros((len(k), 8))
                A[:, 0:3] = X * lam[:, None]
                A[:, 3] = -lam
                A[:, 4:7] = X * (lam * dk)[:, None]
                A[:, 7] = -lam * dk
                rows.append(A)
                rhs.append(np.zeros(len(k)))
                used += 1
        try:
            coef = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        nrm = float(np.linalg.norm(coef[:3]))
        if nrm > 0.5:
            n_out[i] = coef[:3] / nrm
            d_out[i] = coef[3] / nrm
    return n_out, d_out, used


def smooth_poses(samples: list[PoseSample], window_s: float = 0.16, max_gap_s: float = 0.25,
                 sheet: tuple[np.ndarray, np.ndarray] | None = None) -> dict:
    """Turn the per-frame marker poses into a smooth plate trajectory.

    Returns arrays over the accepted samples: t (N,), R (N,3,3), tvec (N,3).

    Where the plate lies *in* its plane (turn and shift) is measured well even by a single marker.
    Its tilt and height are not: with only a few small markers in view they are noisy. A hand-moved
    plate changes tilt and height slowly, so

    1. poses whose plane disagrees with their neighbours in time are dropped,
    2. the plate plane (normal + distance) is fitted over time, trusting frames with many markers
       and using a longer time window around frames with few; ``sheet`` = (times, camera-frame
       laser points that fell on the bare sheet) ties the plane to the laser measurements, and a
       plate that stayed on a flat support gets one fixed plane for the whole scan,
    3. each frame's turn and shift are re-solved from its own markers with that plane fixed,
    4. a short local-linear fit removes the remaining frame-to-frame jitter (it does not lag a
       steady motion, and samples across a tracking gap are never mixed).
    """
    if samples and not getattr(samples[0], "_debiased", False):
        debias_marker_corners(samples)                     # once per recording (smooth_poses may be re-run)
        samples[0]._debiased = True
    val = [s for s in samples if s.valid]
    empty = {"t": np.empty(0), "R": np.empty((0, 3, 3)), "tvec": np.empty((0, 3)), "rejected": 0}
    if not val:
        return empty
    t = np.array([s.t for s in val])
    R = np.stack([s.R for s in val])
    tv = np.stack([s.tvec for s in val])
    nm = np.array([s.n_markers for s in val], dtype=np.float64)
    normal = R[:, :, 2]
    dist = np.einsum("ni,ni->n", normal, tv)
    keep = np.ones(len(val), dtype=bool)
    for i in range(len(val)):
        lo = np.searchsorted(t, t[i] - 0.25)
        hi = np.searchsorted(t, t[i] + 0.25, side="right")
        j = np.arange(lo, hi)
        j = j[j != i]
        if len(j) < 4:
            continue
        wj = np.minimum(nm[j], 12.0)
        order = np.argsort(dist[j])
        cw = np.cumsum(wj[order])
        med_d = float(dist[j][order][np.searchsorted(cw, 0.5 * cw[-1])])
        mn = (normal[j] * wj[:, None]).sum(axis=0)
        mn /= max(np.linalg.norm(mn), 1e-9)
        ang = np.degrees(np.arccos(np.clip(float(normal[i] @ mn), -1.0, 1.0)))
        limit = 4.0 if nm[i] >= 6 else 8.0                 # weak frames are corrected below, not dropped
        if ang > limit or abs(dist[i] - med_d) > (1.5 if nm[i] >= 6 else 4.0):
            keep[i] = False
    rejected = int((~keep).sum())
    val = [s for s, k in zip(val, keep) if k]
    t, R, tv, nm = t[keep], R[keep], tv[keep], nm[keep]
    n = len(t)
    if n == 0:
        return dict(empty, rejected=rejected)
    seg = np.concatenate([[0], np.cumsum(np.diff(t) > max_gap_s)])

    # 2. plate plane over time
    normal = R[:, :, 2]
    dist = np.einsum("ni,ni->n", normal, tv)
    strength = np.minimum(nm, 12.0) ** 2
    sigma = np.where(nm >= 8, 0.08, 0.30)
    n_s, d_s, laser_fits = _fit_plane_track(t, normal, dist, strength, seg, sigma, sheet)
    # A plate that slides on a flat support (the box floor, a wedge) keeps one plane relative to the
    # camera. If the well-determined frames agree on it, use that single plane for every frame.
    plane_mode = "free"
    strong = nm >= 6
    sel, tol_deg, tol_mm, q, label = strong, 1.0, 0.5, 90, "fixed support"
    if strong.sum() < 30:
        # Too few well-determined frames to tell. With only a few markers in view the per-frame tilt
        # is unreliable anyway, so a single plane is assumed unless the frames clearly contradict it.
        sel, tol_deg, tol_mm, q, label = np.ones(n, dtype=bool), 3.0, 1.2, 70, "fixed support (assumed, few markers)"
    if sel.sum() >= 10:
        mean_n = (n_s[sel] * strength[sel, None]).sum(axis=0)
        mean_n /= max(np.linalg.norm(mean_n), 1e-9)
        ang = np.degrees(np.arccos(np.clip(n_s[sel] @ mean_n, -1.0, 1.0)))
        mean_d = float(np.average(d_s[sel], weights=strength[sel]))
        if np.percentile(ang, q) < tol_deg and np.percentile(np.abs(d_s[sel] - mean_d), q) < tol_mm:
            n_s = np.tile(mean_n, (n, 1))
            d_s = np.full(n, mean_d)
            plane_mode = label

    # 3. in-plane pose of every frame on that plane
    for i in range(n):
        z = n_s[i]
        x = R[i][:, 0] - (R[i][:, 0] @ z) * z
        x /= max(np.linalg.norm(x), 1e-9)
        R0 = np.column_stack([x, np.cross(z, x), z])
        t0 = tv[i] + (d_s[i] - float(z @ tv[i])) * z
        if val[i].obj is not None and val[i].img_n is not None and len(val[i].obj) >= 4:
            R[i], tv[i] = _solve_in_plane(val[i].obj, val[i].img_n, R0, t0)
        else:
            R[i], tv[i] = R0, t0

    # 4. remove frame-to-frame jitter
    R_out = R.copy()
    t_out = tv.copy()
    sig = max(window_s / 2.0, 1e-3)
    for i in range(n):
        lo = np.searchsorted(t, t[i] - 2.5 * sig)
        hi = np.searchsorted(t, t[i] + 2.5 * sig, side="right")
        j = np.arange(lo, hi)
        j = j[seg[j] == seg[i]]
        if len(j) < 3:
            continue
        dt = t[j] - t[i]
        w = np.exp(-0.5 * (dt / sig) ** 2) * np.minimum(nm[j], 8.0)
        # rotations relative to sample i are small, so their rotation vectors can be fitted linearly
        rel = np.array([_rotvec(R[k] @ R[i].T) for k in j])
        Y = np.column_stack([rel, tv[j]])
        A = np.column_stack([np.ones_like(dt), dt])
        Aw = A * w[:, None]
        try:
            coef = np.linalg.solve(A.T @ Aw, Aw.T @ Y)
        except np.linalg.LinAlgError:
            continue
        R_out[i] = _rotmat(coef[0, :3]) @ R[i]
        t_out[i] = coef[0, 3:]
    return {"t": t, "R": R_out, "tvec": t_out, "rejected": rejected, "plane_mode": plane_mode,
            "laser_plane_fits": int(laser_fits)}


def poses_at(times: np.ndarray, sm: dict, max_gap_s: float = 0.12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate plate poses at ``times``. Returns (R (M,3,3), tvec (M,3), ok (M,))."""
    times = np.asarray(times, dtype=np.float64)
    t = sm["t"]
    M = len(times)
    R = np.tile(np.eye(3), (M, 1, 1))
    tv = np.zeros((M, 3))
    ok = np.zeros(M, dtype=bool)
    if len(t) == 0:
        return R, tv, ok
    hi = np.clip(np.searchsorted(t, times), 0, len(t) - 1)
    lo = np.clip(hi - 1, 0, len(t) - 1)
    for k in range(M):
        a, b = lo[k], hi[k]
        ta, tb, tk = t[a], t[b], times[k]
        if a != b and ta <= tk <= tb and (tb - ta) <= max_gap_s:
            al = (tk - ta) / max(tb - ta, 1e-9)
            rel = _rotvec(sm["R"][b] @ sm["R"][a].T)
            R[k] = _rotmat(al * rel) @ sm["R"][a]
            tv[k] = (1 - al) * sm["tvec"][a] + al * sm["tvec"][b]
            ok[k] = True
        else:
            j = a if abs(tk - ta) <= abs(tk - tb) else b
            if abs(tk - t[j]) <= max_gap_s / 2.0:
                R[k], tv[k], ok[k] = sm["R"][j], sm["tvec"][j], True
    return R, tv, ok


def to_plate_frame(pts_cam: np.ndarray, R: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Camera-frame points -> plate frame: X_plate = R^T (X_cam - t)."""
    return (np.asarray(pts_cam, dtype=np.float64) - tvec.reshape(1, 3)) @ R


def tracking_summary(samples: list[PoseSample], sm: dict) -> dict:
    n = len(samples)
    val = np.array([s.valid for s in samples]) if n else np.zeros(0, bool)
    nm = np.array([s.n_markers for s in samples]) if n else np.zeros(0)
    rms = np.array([s.rms_px for s in samples if s.valid])
    info = {"samples": int(n), "valid_fraction": round(float(val.mean()), 4) if n else 0.0,
            "median_markers": float(np.median(nm)) if n else 0.0,
            "median_rms_px": round(float(np.median(rms)), 3) if len(rms) else None,
            "pose_outliers_dropped": int(sm.get("rejected", 0)),
            "plate_plane": sm.get("plane_mode", "free"),
            "laser_assisted": bool(sm.get("laser_plane_fits", 0))}
    if len(sm["t"]) >= 2:
        R, tv = sm["R"], sm["tvec"]
        tilt = np.degrees(np.arccos(np.clip(R[:, 2, 2], -1.0, 1.0)))
        yaw = np.degrees(np.unwrap(np.arctan2(R[:, 1, 0], R[:, 0, 0])))
        pos = -(np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), tv))      # camera centre in the plate frame
        steps = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1)
        info.update({
            "rotation_range_deg": round(float(np.ptp(yaw)), 2),
            "tilt_range_deg": [round(float(tilt.min()), 2), round(float(tilt.max()), 2)],
            "plate_distance_mm": [round(float(tv[:, 2].min()), 2), round(float(tv[:, 2].max()), 2)],
            "path_mm": round(float(steps.sum()), 3),
            "extent_mm": [round(float(np.ptp(pos[:, 0])), 2), round(float(np.ptp(pos[:, 1])), 2)],
        })
    return info
