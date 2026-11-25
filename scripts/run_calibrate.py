from __future__ import annotations

import argparse
from pathlib import Path
import sys
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_pipeline import _load_frame_records

from ioscan.calibration.calibrate import (
    calibrate_camera_checkerboard,
    extract_laser_points_on_checkerboard,
    fit_laser_plane_from_points,
    intersect_rays_with_board_plane,
    save_calibration_bundle,
)
from ioscan.calibration.models import CameraIntrinsics, LaserPlane


def run_synthetic_selftest() -> None:
    """Verify SVD laser-plane fitting and ray-board intersection on synthetic multi-pose data."""
    print("=== Running Calibration Engine Synthetic Self-Test ===")
    cam = CameraIntrinsics(width=1600, height=1200, fx=2666.7, fy=2666.7, cx=800.0, cy=600.0)
    gt_laser = LaserPlane.from_baseline_and_angle(baseline_mm=12.0, angle_deg=30.963756532)

    rng = np.random.default_rng(42)
    all_pts = []

    # Simulate 8 tilted checkerboard poses between Z = 15 mm and Z = 30 mm
    for z_depth in np.linspace(15.0, 30.0, 8):
        tilt_x = rng.uniform(-0.25, 0.25)
        tilt_y = rng.uniform(-0.25, 0.25)
        R_cb, _ = cv2.Rodrigues(np.array([tilt_x, tilt_y, 0.0], dtype=np.float64))
        t_cb = np.array([0.0, 0.0, float(z_depth)], dtype=np.float64)

        # Sample 250 columns across the sensor
        u_cols = np.linspace(200.0, 1400.0, 250)
        # Find exact pixel v where the laser plane intersects the tilted board plane
        # On the intersection line, both (n_laser . P + d_laser = 0) and (n_board . (P - t_cb) = 0) hold
        n_b = R_cb[:, 2]
        d_b = -float(np.dot(n_b, t_cb))

        # For each u, ray_unnorm = [(u - cx)/fx, (v - cy)/fy, 1].
        # Equal depth t on both planes: -d_laser / (n_l . r) = -d_b / (n_b . r)
        x_n = (u_cols - cam.cx) / cam.fx
        nl_x, nl_y, nl_z = gt_laser.normal
        nb_x, nb_y, nb_z = n_b

        # Solve linear equation for y_n = (v - cy)/fy:
        # d_laser * (nb_x * x_n + nb_y * y_n + nb_z) = d_b * (nl_x * x_n + nl_y * y_n + nl_z)
        denom = gt_laser.d * nb_y - d_b * nl_y
        num = d_b * (nl_x * x_n + nl_z) - gt_laser.d * (nb_x * x_n + nb_z)
        y_n = num / denom
        v_rows = y_n * cam.fy + cam.cy

        # Add realistic sub-pixel sensor noise (0.08 px std)
        v_noisy = v_rows + rng.normal(0.0, 0.08, size=v_rows.shape)
        uv = np.column_stack([u_cols, v_noisy])

        rays = cam.unproject(uv)
        pts_cam, valid = intersect_rays_with_board_plane(rays, R_cb, t_cb)
        all_pts.append(pts_cam[valid])

    pts_stack = np.vstack(all_pts)
    fit_laser, rms_mm = fit_laser_plane_from_points(pts_stack)

    cos_ang = np.clip(np.abs(np.dot(gt_laser.normal, fit_laser.normal)), -1.0, 1.0)
    ang_err_deg = float(np.rad2deg(np.arccos(cos_ang)))
    d_err_um = abs(fit_laser.d - gt_laser.d) * 1000.0

    print(f"  Sampled 3D points  : {pts_stack.shape[0]} across 8 board poses")
    print(f"  GT Normal / d      : {np.round(gt_laser.normal, 6)} | d = {gt_laser.d:.5f} mm")
    print(f"  Fitted Normal / d  : {np.round(fit_laser.normal, 6)} | d = {fit_laser.d:.5f} mm")
    print(f"  Plane Fit RMS      : {rms_mm * 1000.0:.2f} µm")
    print(f"  Normal Angle Error : {ang_err_deg:.5f}°")
    print(f"  Offset |Δd| Error  : {d_err_um:.2f} µm")


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate camera intrinsics and 3D laser plane from checkerboard session.")
    ap.add_argument("--session", type=Path, default=None, help="Calibration session directory with frames/")
    ap.add_argument("--cols", type=int, default=9, help="Interior checkerboard columns")
    ap.add_argument("--rows", type=int, default=6, help="Interior checkerboard rows")
    ap.add_argument("--square-mm", type=float, default=2.0, help="Checkerboard square size in mm")
    ap.add_argument("--out", type=Path, default=Path("data/calibration/calibration.json"), help="Output JSON path")
    ap.add_argument("--self-test", action="store_true", help="Run synthetic multi-pose SVD calibration self-test")
    args = ap.parse_args()

    if args.self_test or args.session is None:
        run_synthetic_selftest()
        return

    records = _load_frame_records(args.session)
    off_images = []
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    last_off = None

    for rec in records:
        img = cv2.imread(str(rec["path"]), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if not rec["laser_on"]:
            off_images.append(img)
            last_off = img
        elif last_off is not None:
            pairs.append((img, last_off))

    pattern = (args.cols, args.rows)
    # Never silently substitute nominal intrinsics here. A generated calibration.json
    # must represent measured checkerboard data, not the project's example geometry.
    camera, cam_rms = calibrate_camera_checkerboard(
        off_images, pattern_size=pattern, square_size_mm=args.square_mm
    )
    print(
        f"Calibrated camera from checkerboard images (RMS = {cam_rms:.3f} px): "
        f"fx={camera.fx:.2f}, fy={camera.fy:.2f}, cx={camera.cx:.2f}, cy={camera.cy:.2f}"
    )

    laser_pts = []
    for on_img, off_img in pairs:
        pts = extract_laser_points_on_checkerboard(
            on_img, off_img, camera, pattern_size=pattern, square_size_mm=args.square_mm
        )
        if pts.shape[0] > 0:
            laser_pts.append(pts)

    if not laser_pts:
        raise RuntimeError("No valid checkerboard + laser stripe pairs detected in session.")

    all_laser_pts = np.vstack(laser_pts)
    laser_plane, plane_rms_mm = fit_laser_plane_from_points(all_laser_pts)
    saved = save_calibration_bundle(
        args.out,
        camera,
        laser_plane,
        extra_meta={
            "calibration_type": "measured_checkerboard_and_laser_plane",
            "checkerboard_inner_corners": [int(args.cols), int(args.rows)],
            "checkerboard_square_size_mm": float(args.square_mm),
            "camera_rms_px": float(cam_rms),
            "laser_plane_rms_um": float(plane_rms_mm * 1000.0),
            "laser_calibration_points": int(len(all_laser_pts)),
            "source_session": str(args.session),
        },
    )
    print(f"Fitted Laser Plane from {len(all_laser_pts)} points (RMS = {plane_rms_mm * 1000.0:.2f} µm)")
    print(f"Saved calibration bundle to: {saved}")


if __name__ == "__main__":
    main()