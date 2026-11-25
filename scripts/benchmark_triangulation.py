from __future__ import annotations

# Prefer this checkout's src/ package over any stale ioscan installation in site-packages.
import sys as _sys
from pathlib import Path as _Path
_project_src = _Path(__file__).resolve().parents[1] / "src"
if _project_src.is_dir():
    _sys.path.insert(0, str(_project_src))
import argparse
from pathlib import Path
import sys
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_pipeline import _load_calibration, _load_frame_records
from ioscan.session.loader import read_grayscale_frame

from ioscan.pipeline.ridge import RidgeConfig
from ioscan.pipeline.triangulate import triangulate_frame


def _load_gt_profiles_npz(session_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load (frame_idx, u_px, z_cam_mm) directly from gt/profiles.npz."""
    npz_path = session_dir / "gt" / "profiles.npz"
    data = np.load(npz_path)
    keys = list(data.files)

    f_key = next((k for k in keys if "frame" in k.lower() or "idx" in k.lower()), keys[0])
    pts_key = next((k for k in keys if data[k].ndim == 2 and data[k].shape[1] == 3), None)
    u_key = next((k for k in keys if k not in (f_key, pts_key) and ("u" in k.lower() or "px" in k.lower() or "uv" in k.lower())), None)

    gt_fidx = data[f_key].astype(int)
    gt_z = data[pts_key][:, 2] if pts_key else data[keys[-1]][:, 2]

    if u_key:
        u_arr = data[u_key]
        gt_u = u_arr[:, 0] if u_arr.ndim == 2 else u_arr
    else:
        cam, _ = _load_calibration(session_dir)
        pts_cam = data[pts_key]
        gt_u = cam.project(pts_cam)[:, 0]

    return gt_fidx, gt_u, gt_z


def evaluate_session(session_dir: Path, method: str, cal_error_deg: float = 0.0) -> dict[str, float]:
    camera, laser_obj = _load_calibration(session_dir)
    records = _load_frame_records(session_dir)
    gt_fidx, gt_u_all, gt_z_all = _load_gt_profiles_npz(session_dir)

    laser_used = laser_obj.perturbed(angle_deg=cal_error_deg) if cal_error_deg != 0.0 else laser_obj
    cfg = RidgeConfig(method=method)

    z_errors_um = []
    last_off = None

    for rec in records:
        frame = read_grayscale_frame(rec["path"])
        if not rec["laser_on"]:
            last_off = frame
            continue

        pts_cam, uv, _ = triangulate_frame(
            frame, last_off, camera, laser_used, ridge_cfg=cfg, z_range=(10.0, 45.0)
        )
        if pts_cam.shape[0] == 0:
            continue

        mask = gt_fidx == rec["index"]
        if not np.any(mask):
            continue

        u_gt = gt_u_all[mask]
        z_gt = gt_z_all[mask]
        order = np.argsort(u_gt)
        u_gt, z_gt = u_gt[order], z_gt[order]

        in_span = (uv[:, 0] >= u_gt[0]) & (uv[:, 0] <= u_gt[-1])
        if not np.any(in_span):
            continue

        z_gt_interp = np.interp(uv[in_span, 0], u_gt, z_gt)
        err_um = (pts_cam[in_span, 2] - z_gt_interp) * 1000.0
        z_errors_um.append(err_um)

    all_err = np.concatenate(z_errors_um) if z_errors_um else np.zeros(1)
    return {
        "n_points": int(all_err.size),
        "bias_um": float(np.mean(all_err)),
        "rms_um": float(np.sqrt(np.mean(all_err ** 2))),
        "p95_um": float(np.percentile(np.abs(all_err), 95)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark ridge extraction and triangulation accuracy.")
    ap.add_argument("session", type=Path, help="Path to simulated session folder")
    args = ap.parse_args()

    print(f"=== Triangulation Benchmark on {args.session.name} ===")
    print(f"{'Method':<12} | {'Cal Err':<8} | {'Points':>8} | {'Bias (µm)':>10} | {'RMS (µm)':>10} | {'P95 (µm)':>10}")
    print("-" * 72)

    for method in ["cog", "gaussian3", "steger"]:
        stats = evaluate_session(args.session, method=method, cal_error_deg=0.0)
        print(
            f"{method:<12} | {'0.00°':<8} | {stats['n_points']:>8d} | "
            f"{stats['bias_um']:>+10.2f} | {stats['rms_um']:>10.2f} | {stats['p95_um']:>10.2f}"
        )

    print("\n--- Calibration Sensitivity (+0.10° Plane Angle Error) ---")
    for method in ["cog", "gaussian3"]:
        stats = evaluate_session(args.session, method=method, cal_error_deg=0.10)
        print(
            f"{method:<12} | {'+0.10°':<8} | {stats['n_points']:>8d} | "
            f"{stats['bias_um']:>+10.2f} | {stats['rms_um']:>10.2f} | {stats['p95_um']:>10.2f}"
        )


if __name__ == "__main__":
    main()