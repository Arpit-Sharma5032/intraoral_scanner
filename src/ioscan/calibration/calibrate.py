from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ioscan.calibration.models import CameraIntrinsics, LaserPlane
from ioscan.pipeline.ridge import RidgeConfig, extract_ridge_subpixel


def make_object_points(pattern_size: tuple[int, int], square_size_mm: float) -> np.ndarray:
    """Create (N, 3) 3D checkerboard corner coordinates in the board's local frame (Z = 0)."""
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), dtype=np.float64)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid * float(square_size_mm)
    return objp


def calibrate_camera_checkerboard(
    images: list[np.ndarray],
    pattern_size: tuple[int, int] = (9, 6),
    square_size_mm: float = 2.0,
) -> tuple[CameraIntrinsics, float]:
    """
    Calibrate CameraIntrinsics from a list of grayscale checkerboard images.
    Returns (CameraIntrinsics, reprojection_rms_px).
    """
    if not images:
        raise ValueError("No images provided for camera calibration.")

    cols, rows = pattern_size
    objp = make_object_points(pattern_size, square_size_mm).astype(np.float32)

    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    h, w = images[0].shape[:2]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
        if found:
            corners_sub = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners_sub)

    if len(objpoints) < 3:
        raise ValueError(f"Need at least 3 valid checkerboard detections, found {len(objpoints)}.")

    rms, K, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
    dc = dist.ravel().tolist()
    while len(dc) < 5:
        dc.append(0.0)

    intrinsics = CameraIntrinsics(
        width=int(w),
        height=int(h),
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
        dist_coeffs=tuple(float(x) for x in dc[:5]),
    )
    return intrinsics, float(rms)


def intersect_rays_with_board_plane(
    rays_cam: np.ndarray,
    R_cb: np.ndarray,
    t_cb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Intersect unit camera rays (N, 3) with a 3D checkerboard plane defined by (R_cb, t_cb).
    Plane equation: n_board . (P - t_cb) = 0  =>  t_ray = (n_board . t_cb) / (n_board . ray).
    """
    rays = np.asarray(rays_cam, dtype=np.float64)
    n_board = R_cb[:, 2].astype(np.float64)
    n_board /= np.linalg.norm(n_board)
    num = float(np.dot(n_board, t_cb.reshape(3)))

    denom = rays @ n_board
    valid = np.abs(denom) > 1e-7
    t_ray = np.zeros(rays.shape[0], dtype=np.float64)
    t_ray[valid] = num / denom[valid]
    valid = valid & (t_ray > 0.0)
    pts_cam = rays * t_ray[:, None]
    return pts_cam, valid


def extract_laser_points_on_checkerboard(
    frame_on: np.ndarray,
    frame_off: np.ndarray,
    camera: CameraIntrinsics,
    pattern_size: tuple[int, int] = (9, 6),
    square_size_mm: float = 2.0,
    ridge_cfg: RidgeConfig = RidgeConfig(method="cog"),
) -> np.ndarray:
    """
    Given a paired (Laser-ON, Laser-OFF) view of a checkerboard:
      1. Detect checkerboard corners on frame_off and estimate 3D board pose (R_cb, t_cb) via solvePnP.
      2. Subtract frame_off from frame_on and extract sub-pixel laser ridge coordinates (u, v)
         restricted to the convex hull of the detected checkerboard corners.
      3. Intersect rays with the 3D checkerboard plane.
    """
    gray_off = cv2.cvtColor(frame_off, cv2.COLOR_BGR2GRAY) if frame_off.ndim == 3 else frame_off
    gray_on = cv2.cvtColor(frame_on, cv2.COLOR_BGR2GRAY) if frame_on.ndim == 3 else frame_on

    cols, rows = pattern_size
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray_off, (cols, rows), flags)
    if not found:
        return np.empty((0, 3), dtype=np.float64)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
    corners_sub = cv2.cornerSubPix(gray_off, corners, (5, 5), (-1, -1), criteria)

    objp = make_object_points(pattern_size, square_size_mm)
    dist = np.array(camera.dist_coeffs, dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(objp, corners_sub, camera.K, dist)
    if not ok:
        return np.empty((0, 3), dtype=np.float64)

    R_cb, _ = cv2.Rodrigues(rvec)
    t_cb = tvec.reshape(3)

    hull = cv2.convexHull(corners_sub.astype(np.int32))
    board_mask = np.zeros_like(gray_off, dtype=np.uint8)
    cv2.fillConvexPoly(board_mask, hull, 255)

    # Ambient subtraction: clip negative values to 0
    diff = np.maximum(0.0, gray_on.astype(np.float32) - gray_off.astype(np.float32)).astype(np.uint8)
    diff_masked = np.where(board_mask > 0, diff, 0)

    uv, _ = extract_ridge_subpixel(diff_masked, cfg=ridge_cfg)
    if uv.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)

    rays = camera.unproject(uv)
    pts_cam, valid = intersect_rays_with_board_plane(rays, R_cb, t_cb)
    return pts_cam[valid]


def fit_laser_plane_from_points(pts_3d_cam: np.ndarray) -> tuple[LaserPlane, float]:
    """Fit a 3D plane (normal . P + d = 0) to (N, 3) points via SVD."""
    pts = np.asarray(pts_3d_cam, dtype=np.float64)
    if pts.shape[0] < 10:
        raise ValueError(f"Need at least 10 3D points to fit laser plane, got {pts.shape[0]}.")

    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1, :].copy()
    normal /= np.linalg.norm(normal)

    d = -float(np.dot(normal, centroid))
    if d < 0.0:
        normal = -normal
        d = -d

    residuals = (pts @ normal) + d
    rms_mm = float(np.sqrt(np.mean(residuals ** 2)))
    return LaserPlane(normal=normal, d=d), rms_mm


def save_calibration_bundle(
    out_json: str | Path,
    camera: CameraIntrinsics,
    laser: LaserPlane,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """Save calibrated CameraIntrinsics and LaserPlane to calibration.json."""
    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "camera": camera.to_dict(),
        "laser": laser.to_dict(),
    }
    if extra_meta:
        payload["metadata"] = extra_meta
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out