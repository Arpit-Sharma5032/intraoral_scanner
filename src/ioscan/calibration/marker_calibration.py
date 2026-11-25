"""Camera + laser-plane calibration from one recording of the bare marker sheet.

The same printed sheet that carries the model is used as the calibration target: record ~30 s
while the flat sheet (no model) is moved, lifted and tilted under the scanner. From that recording:

1. the camera's focal length and lens distortion are fitted to the marker corners of the
   laser-OFF frames (many views at different tilts and heights);
2. in every laser-ON frame the laser line drawn on the sheet is intersected with the sheet's
   plane (known from the markers), which gives 3D points *on the laser plane*; a robust plane fit
   through all of them is the laser plane;
3. the result is checked on itself: triangulating the same line with the fitted plane must put
   the points back on the sheet (reported as the sheet residual, in mm).
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np

from ioscan.calibration.models import CameraIntrinsics, LaserPlane
from ioscan.pipeline.markers import (
    MarkerBoard, MarkerDetector, MarkerTracker, poses_at, probe_orientation, smooth_poses, solve_board_pose,
)
from ioscan.pipeline.ridge import RidgeConfig
from ioscan.pipeline.triangulate import triangulate_frame
from ioscan.session.laser_phase import verify_laser_flags
from ioscan.session.loader import load_calibration, load_frame_records, read_grayscale_frame
from ioscan.utils.roi import enabled_roi_points, load_roi_config


class CalibrationError(RuntimeError):
    """The recording cannot give a trustworthy calibration; the message says what to change."""


def _tilt_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(abs(R[2, 2]), -1.0, 1.0))))


def calibrate_camera_from_views(views: list[tuple[np.ndarray, np.ndarray]], size: tuple[int, int],
                                guess: CameraIntrinsics) -> tuple[CameraIntrinsics, float, dict]:
    """Fit focal length + radial distortion to marker-corner views [(object (N,3), image (N,2))]."""
    w, h = size
    obj = [v[0].astype(np.float32).reshape(-1, 1, 3) for v in views]
    img = [v[1].astype(np.float32).reshape(-1, 1, 2) for v in views]
    K0 = np.array([[guess.fx, 0, w / 2.0], [0, guess.fx, h / 2.0], [0, 0, 1.0]])
    base = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST
            | cv2.CALIB_FIX_K3)
    # Stage 1: one focal length, principal point at the image centre, one distortion term.
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj, img, (w, h), K0.copy(), np.zeros(5),
        flags=base | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_K2)
    info = {"stage": "focal+k1", "rms_px": float(rms)}
    # Stage 2: also solve for the principal point. It is only kept when the data pin it down (small
    # standard deviation, which needs tilted views) and it stays near the image centre.
    allp = np.vstack([v[1] for v in views])
    info["coverage"] = [round(float(np.ptp(allp[:, 0]) / w), 3), round(float(np.ptp(allp[:, 1]) / h), 3)]
    try:
        rms2, K2, dist2, rv2, tv2, std_in, _se, _pv = cv2.calibrateCameraExtended(
            obj, img, (w, h), K.copy(), dist.copy(), flags=base | cv2.CALIB_FIX_K2)
        std_in = np.asarray(std_in).ravel()
        shift = max(abs(K2[0, 2] - w / 2.0) / w, abs(K2[1, 2] - h / 2.0) / h)
        info["principal_point_std_px"] = [round(float(std_in[2]), 2), round(float(std_in[3]), 2)]
        if rms2 <= rms and shift < 0.10 and max(std_in[2], std_in[3]) < 4.0:
            rms, K, dist, rvecs, tvecs = rms2, K2, dist2, rv2, tv2
            info["stage"] = "focal+k1+principal point"
    except cv2.error:
        pass
    dc = [float(x) for x in np.asarray(dist).ravel()[:5]]
    dc += [0.0] * (5 - len(dc))
    cam = CameraIntrinsics(width=int(w), height=int(h), fx=float(K[0, 0]), fy=float(K[1, 1]),
                           cx=float(K[0, 2]), cy=float(K[1, 2]), dist_coeffs=tuple(dc),
                           pixel_size_um=guess.pixel_size_um)
    tilts = [_tilt_deg(cv2.Rodrigues(rv)[0]) for rv in rvecs]
    dists = [float(np.asarray(tv).ravel()[2]) for tv in tvecs]
    info.update(rms_px=float(rms), views=len(views), tilt_max_deg=round(float(max(tilts)), 1),
                tilt_p75_deg=round(float(np.percentile(tilts, 75)), 1),
                distance_range_mm=[round(min(dists), 1), round(max(dists), 1)])
    return cam, float(rms), info


def fit_plane_ransac(P: np.ndarray, thr_mm: float = 0.25, iters: int = 400, seed: int = 0) -> tuple[np.ndarray, float, np.ndarray]:
    """Robust plane n.P + d = 0 (unit n). Returns (n, d, inlier mask)."""
    rng = np.random.default_rng(seed)
    P = np.asarray(P, dtype=np.float64)
    best = None
    for _ in range(iters):
        a = P[rng.choice(len(P), 3, replace=False)]
        nrm = np.cross(a[1] - a[0], a[2] - a[0])
        L = np.linalg.norm(nrm)
        if L < 1e-6:
            continue
        nrm /= L
        inl = np.abs((P - a[0]) @ nrm) < thr_mm
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 50:
        raise CalibrationError("The laser line could not be found on the sheet in enough frames.")
    for _ in range(3):
        Q = P[best]
        c = Q.mean(axis=0)
        _u, _s, vh = np.linalg.svd(Q - c, full_matrices=False)
        nrm = vh[-1]
        d = -float(nrm @ c)
        best = np.abs(P @ nrm + d) < thr_mm
    return nrm, d, best


def calibrate_from_marker_session(session_dir: str | Path, *, board: MarkerBoard | None = None,
                                  max_views: int = 70, progress=None) -> dict:
    """Run the full calibration on a recorded session. Returns a dict with camera, laser and stats."""
    session_dir = Path(session_dir)
    board = board or MarkerBoard.load()

    def tell(pct: float, msg: str) -> None:
        if progress is not None:
            progress({"stage": "calibrating", "progress": float(pct), "message": msg})

    guess, laser_guess, _nominal = load_calibration(session_dir)
    records = load_frame_records(session_dir)
    roi_polygon = enabled_roi_points(load_roi_config(session_dir / "roi.json"))
    box = None
    if roi_polygon is not None:
        rp = np.asarray(roi_polygon, dtype=np.float64)
        box = (float(rp[:, 0].min()), float(rp[:, 1].min()), float(rp[:, 0].max()), float(rp[:, 1].max()))
    tell(2, "Checking laser on/off frames")
    records, flag_info = verify_laser_flags(records, roi_box=box)
    off = [r for r in records if not r["laser_on"]]
    on = [r for r in records if r["laser_on"]]
    if len(off) < 40 or len(on) < 40:
        raise CalibrationError("The recording is too short. Record about 30 seconds of the bare marker sheet.")

    # ---- 1. markers in the OFF frames -------------------------------------------------------
    tell(6, "Finding the markers")
    probe = probe_orientation([read_grayscale_frame(r["path"]) for r in off[::max(1, len(off) // 16)][:20]], board)
    if probe["frames_with_2plus"] == 0:
        raise CalibrationError(
            "No markers were found. Put the printed marker sheet (without the model) under the scanner, "
            "make sure it is lit and in focus, and record again.")
    mirrored = bool(probe["mirrored"])
    det = MarkerDetector(board, mirrored)
    step = max(1, len(off) // 260)
    cand = []
    sub = off[::step]
    for k, r in enumerate(sub):
        ids, corners = det.detect(read_grayscale_frame(r["path"]))
        if len(ids) >= 8:
            cand.append((float(r["timestamp_s"]), det.object_points[ids].reshape(-1, 3), corners.reshape(-1, 2)))
        if k % 20 == 0:
            tell(6 + 34.0 * k / max(len(sub), 1), "Finding the markers")
    if len(cand) < 15:
        raise CalibrationError(
            f"Only {len(cand)} frames showed 8 or more markers. Use the bare sheet (no model on it), keep it "
            "inside the camera window and avoid shadows and glare.")
    # drop views with a wrongly decoded marker, using a first pose with the starting intrinsics
    views = []
    for _t, o, c in cand:
        sol = solve_board_pose(o, c, guess.K, None)
        if sol is None:
            continue
        keep = sol[3]
        if keep.sum() >= 32:
            views.append((o[keep], c[keep]))
    if len(views) > max_views:
        views = [views[i] for i in np.linspace(0, len(views) - 1, max_views).astype(int)]
    if len(views) < 12:
        raise CalibrationError("Too few clear views of the marker sheet. Record again with the sheet flat and well lit.")

    # ---- 2. camera -------------------------------------------------------------------------
    tell(42, "Fitting the camera")
    camera, rms, cam_info = calibrate_camera_from_views(views, (guess.width, guess.height), guess)
    if cam_info["tilt_p75_deg"] < 8.0:
        raise CalibrationError(
            f"The sheet stayed almost flat to the camera (largest tilt {cam_info['tilt_max_deg']:.0f} deg). The focal "
            "length can only be measured if the sheet is also TILTED: lift one edge by 1-2 cm, in several "
            "directions, and also raise and lower the whole sheet. Record again.")
    if rms > 1.5:
        raise CalibrationError(
            f"The camera fit is poor ({rms:.2f} px). Check that the sheet is glued flat on a stiff card, printed "
            "at 100 % and sharp in the camera image, then record again.")

    # ---- 3. sheet pose in every OFF frame, with the fitted camera ----------------------------
    tell(48, "Tracking the sheet")
    tracker = MarkerTracker(board, camera.K, camera.dist_coeffs, mirrored=mirrored, min_markers=4)
    off_frames_t = []
    for k, r in enumerate(off):
        tracker.add(read_grayscale_frame(r["path"]), float(r["timestamp_s"]))
        off_frames_t.append(float(r["timestamp_s"]))
        if k % 40 == 0:
            tell(48 + 22.0 * k / len(off), "Tracking the sheet")
    tracker.flush()
    sm = smooth_poses(tracker.samples)
    on_t = np.array([float(r["timestamp_s"]) for r in on])
    R_on, t_on, ok_on = poses_at(on_t, sm)

    # ---- 4. laser line on the sheet -> points on the laser plane ------------------------------
    tell(70, "Measuring the laser plane")
    obj_all = det.object_points.reshape(-1, 3)
    lo, hi = obj_all[:, :2].min(axis=0), obj_all[:, :2].max(axis=0)
    ridge = RidgeConfig(saturation_limit=254.0, max_secondary_ratio=0.6, min_prominence=25.0, adaptive_floor=True)
    off_by_index = {int(r["index"]): r for r in off}
    pts_plane, rays_keep, pose_keep = [], [], []
    # A plane that sees every pixel: the ridge pixels are intersected with the SHEET, not this plane.
    dummy = LaserPlane(normal=np.array([0.0, 0.0, 1.0]), d=-100.0)
    idx_on = np.flatnonzero(ok_on)
    if len(idx_on) > 500:
        idx_on = idx_on[np.linspace(0, len(idx_on) - 1, 500).astype(int)]
    for k, i in enumerate(idx_on):
        r = on[i]
        prev, nxt = off_by_index.get(int(r["index"]) - 1), off_by_index.get(int(r["index"]) + 1)
        refs = [read_grayscale_frame(q["path"]) for q in (prev, nxt) if q is not None]
        if not refs:
            continue
        ref = refs[0] if len(refs) == 1 else cv2.addWeighted(refs[0], 0.5, refs[1], 0.5, 0.0)
        frame = read_grayscale_frame(r["path"])
        _pts, uv, _w = triangulate_frame(frame, ref, camera, dummy, ridge_cfg=ridge,
                                         z_range=(1.0, 1.0e6), roi_polygon=roi_polygon)
        if len(uv) < 20:
            continue
        rays = camera.unproject(uv)
        R, t = R_on[i], t_on[i]
        nb = R[:, 2]
        denom = rays @ nb
        good = np.abs(denom) > 1e-6
        lam = np.zeros(len(rays))
        lam[good] = float(nb @ t) / denom[good]
        P = rays * lam[:, None]
        Pb = (P - t) @ R                                   # sheet coordinates: keep only points on the print
        good &= (lam > 0) & (Pb[:, 0] > lo[0]) & (Pb[:, 0] < hi[0]) & (Pb[:, 1] > lo[1]) & (Pb[:, 1] < hi[1])
        if good.sum() < 20:
            continue
        sel = np.flatnonzero(good)[::max(1, int(good.sum()) // 120)]
        pts_plane.append(P[sel])
        rays_keep.append(rays[sel])
        pose_keep.append(np.full(len(sel), i))
        if k % 25 == 0:
            tell(70 + 24.0 * k / max(len(idx_on), 1), "Measuring the laser plane")
    if len(pts_plane) < 30:
        raise CalibrationError(
            "The laser line was found on the sheet in too few frames. Make sure the laser is on, the line falls on "
            "the sheet inside the ROI, and the room is not too bright; then record again.")
    P = np.vstack(pts_plane)
    rays_all = np.vstack(rays_keep)
    pose_idx = np.concatenate(pose_keep)
    nrm, d, inl = fit_plane_ransac(P)
    Q = P[inl]
    sv = np.linalg.svd(Q - Q.mean(axis=0), compute_uv=False) / np.sqrt(len(Q))
    if sv[1] < 1.2:
        raise CalibrationError(
            "All laser lines fell at nearly the same height, so the laser plane is not determined "
            f"(spread {sv[1]:.1f} mm, need more than 1.2). Raise and lower the sheet by 1-2 cm and tilt it "
            "while recording.")
    if d < 0:
        nrm, d = -nrm, -d
    laser = LaserPlane(normal=nrm, d=float(d))

    # ---- 5. self-check: triangulate with the fitted plane, compare with the sheet ----------------
    pts_tri, valid = laser.intersect_rays(rays_all[inl])
    Rk, tk = R_on[pose_idx[inl]], t_on[pose_idx[inl]]
    z_sheet = np.einsum("nj,nj->n", pts_tri - tk, Rk[:, :, 2])
    z_sheet = z_sheet[valid]
    resid = float(np.sqrt(np.mean(z_sheet ** 2))) if len(z_sheet) else float("nan")
    tell(97, "Checking the result")

    ang = float(np.degrees(np.arccos(np.clip(abs(float(nrm @ np.asarray(laser_guess.normal))), -1.0, 1.0))))
    stats = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "session": str(session_dir),
        "method": "marker sheet",
        "marker_board": board.to_dict(),
        "mirrored_optics": mirrored,
        "camera_rms_px": round(rms, 3),
        "camera_fit": cam_info,
        "focal_change_percent": round(100.0 * (camera.fx / guess.fx - 1.0), 2),
        "laser_frames_used": int(len(pts_plane)),
        "laser_points": int(inl.sum()),
        "laser_inlier_fraction": round(float(inl.mean()), 3),
        "laser_plane_rms_mm": round(float(np.sqrt(np.mean((Q @ nrm + d) ** 2))), 4),
        "laser_depth_spread_mm": round(float(sv[1]), 2),
        "laser_plane_change_deg": round(ang, 2),
        "sheet_residual_mm": round(resid, 4),
        "laser_flags": flag_info,
        "markers_tracked_fraction": round(float(np.mean([s.valid for s in tracker.samples])), 3),
    }
    quality = "good"
    notes = []
    if resid > 0.15 or stats["laser_plane_rms_mm"] > 0.15:
        quality = "check"
        notes.append("Residuals are larger than expected: check that the sheet is flat on a stiff card.")
    if cam_info["tilt_max_deg"] < 15.0:
        quality = "check"
        notes.append("Tilt the sheet more next time (15-25 deg) for a firmer focal length.")
    stats["quality"] = quality
    stats["notes"] = notes
    return {"camera": camera, "laser": laser, "stats": stats}


def save_marker_calibration(result: dict, out_json: str | Path) -> Path:
    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"camera": result["camera"].to_dict(), "laser": result["laser"].to_dict(),
               "metadata": result["stats"]}
    out.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return out
