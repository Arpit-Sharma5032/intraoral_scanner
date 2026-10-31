from __future__ import annotations

# Prefer this checkout's src/ package over any stale ioscan installation in site-packages.
import sys as _sys
from pathlib import Path as _Path
_project_src = _Path(__file__).resolve().parents[1] / "src"
if _project_src.is_dir():
    _sys.path.insert(0, str(_project_src))

import argparse
import json
from pathlib import Path
import warnings
import os

import cv2
import numpy as np
from tqdm import tqdm

from ioscan.calibration.models import CameraIntrinsics, LaserPlane
from ioscan.pipeline.pose import PoseConfig, PoseTracker
from ioscan.pipeline.ridge import RidgeConfig
from ioscan.pipeline.triangulate import triangulate_frame
from ioscan.session.loader import (
    load_calibration,
    load_frame_records,
    load_gt_poses,
    session_source,
)


def _load_calibration(session_dir: Path) -> tuple[CameraIntrinsics, LaserPlane]:
    """Camera + laser plane stored with the session (raises if none; no silent nominal values)."""
    cam, laser, _ = load_calibration(session_dir)
    return cam, laser


def _load_frame_records(session_dir: Path) -> list[dict]:
    return load_frame_records(session_dir)


def run_pipeline(
    session_dir: Path,
    out_dir: Path,
    pose_mode: str = "imu_hybrid",
    ridge_method: str = "cog",
    mesh_method: str = "bpa",
    voxel_size_mm: float = 0.08,
    speed_mm_s: float = 5.0,
    world_velocity_mm_s: tuple[float, float, float] | None = None,
    init_from_gt: bool = False,
    z_range: tuple[float, float] = (10.0, 45.0),
    segment: bool = False,
    seg_cfg=None,
    progress_callback=None,
) -> dict:
    session_dir, out_dir = Path(session_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if segment:
        # Fail fast on missing PyTorch, an unknown jaw, or a missing/mismatched checkpoint
        # rather than reconstructing the scan and failing only at the final segmentation stage.
        from ioscan.pipeline.segment import (
            SegmentConfig, _import_torch, resolve_segmentation_model,
        )
        seg_cfg = seg_cfg or SegmentConfig()
        _import_torch()
        resolve_segmentation_model(seg_cfg, session_hint=session_dir)

    try:
        from ioscan.pipeline.fusion import FusionConfig, ScanFuser
        from ioscan.pipeline.mesh import MeshConfig, export_scan_outputs, reconstruct_mesh
    except ImportError as exc:
        if exc.name == "open3d":
            raise RuntimeError(
                "The reconstruction pipeline requires Open3D. Install the project dependencies "
                "with: python -m pip install -e ."
            ) from exc
        raise

    camera, laser, nominal_cal = load_calibration(session_dir)
    records = load_frame_records(session_dir)
    source = session_source(session_dir)
    gt_poses = load_gt_poses(session_dir) if (pose_mode == "gt" or init_from_gt) else {}

    run_warnings: list[str] = []
    if pose_mode == "submap_icp":
        warnings.warn("pose mode 'submap_icp' is a deprecated alias of 'imu_hybrid'", DeprecationWarning)
        pose_mode = "imu_hybrid"
    if nominal_cal:
        run_warnings.append("Calibration is NOMINAL (not measured). Do not use for metrology.")
    if pose_mode == "imu_hybrid":
        run_warnings.append(
            f"Translation is assumed: constant {speed_mm_s} mm/s along the camera scan axis; "
            "the IMU only supplies orientation."
        )
    if source == "sim":
        run_warnings.append("Simulated session: the IMU stream is noise-free and equals ground truth.")
    if init_from_gt:
        run_warnings.append("Start position was taken from simulator ground truth (--init-from-gt).")

    init_pos = (0.0, 0.0, 0.0)
    if init_from_gt:
        if 0 not in gt_poses:
            raise ValueError("--init-from-gt needs gt/poses.npz (simulator sessions only)")
        init_pos = tuple(gt_poses[0][:3, 3].tolist())

    if pose_mode == "imu_const_vel" and world_velocity_mm_s is None:
        raise ValueError("pose mode 'imu_const_vel' needs --world-velocity VX VY VZ (mm/s)")
    pose_cfg = PoseConfig(
        mode=pose_mode,
        nominal_speed_mm_s=speed_mm_s,
        world_velocity_mm_s=tuple(world_velocity_mm_s) if world_velocity_mm_s else (0.0, 0.0, 0.0),
        initial_position_mm=init_pos,
    )
    tracker = PoseTracker(pose_cfg, initial_timestamp_s=records[0]["timestamp_s"])
    ridge_cfg = RidgeConfig(method=ridge_method)
    ridge_cfg_8bit = RidgeConfig(method=ridge_method, saturation_limit=254.0)  # hardware frames are 8-bit
    fuser = ScanFuser(camera, FusionConfig(voxel_size_mm=voxel_size_mm, use_tsdf=(mesh_method == "tsdf")))
    mesh_cfg = MeshConfig(method=mesh_method)

    last_off: np.ndarray | None = None
    n_on_processed = 0
    total_points = 0
    unpaired_on = 0

    preview_path = out_dir / "reconstruction_preview.ply"
    preview_every = max(5, len(records) // 24)
    for rec_pos, rec in enumerate(tqdm(records, desc=f"Pipeline ({pose_mode})")):
        frame = cv2.imread(str(rec["path"]), cv2.IMREAD_UNCHANGED)
        if frame is None:
            raise IOError(f"Cannot read frame {rec['path']}")
        if not rec["laser_on"]:
            last_off = frame
            continue
        if last_off is None:
            unpaired_on += 1

        pts_cam, uv, _ = triangulate_frame(frame, last_off, camera, laser, ridge_cfg=ridge_cfg_8bit if frame.dtype == np.uint8 else ridge_cfg,
                                            z_range=z_range)
        T_wc = tracker.update(
            frame_index=rec["index"],
            timestamp_s=rec["timestamp_s"],
            quat_wxyz=rec["quat_wxyz"],  # None raises in imu modes: no made-up orientation
            pts_cam=pts_cam,
            gt_T_wc=gt_poses.get(rec["index"]),
        )
        if pts_cam.shape[0] > 0:
            fuser.integrate(pts_cam, uv, T_wc)
            n_on_processed += 1
            total_points += pts_cam.shape[0]

        if progress_callback is not None:
            pct = 82.0 * (rec_pos + 1) / max(len(records), 1)
            preview_written = False
            if n_on_processed and ((rec_pos + 1) % preview_every == 0 or rec_pos + 1 == len(records)):
                preview_pcd = fuser.get_preview_point_cloud(voxel_downsample_mm=max(voxel_size_mm * 2.0, 0.16))
                if len(preview_pcd.points) > 0:
                    import open3d as o3d
                    preview_tmp = out_dir / "reconstruction_preview.tmp.ply"
                    if o3d.io.write_point_cloud(str(preview_tmp), preview_pcd, write_ascii=False):
                        os.replace(preview_tmp, preview_path)
                        preview_written = True
            progress_callback({
                "stage": "reconstructing", "progress": round(pct, 2),
                "frames_done": rec_pos + 1, "frames_total": len(records),
                "laser_on_frames": n_on_processed, "triangulated_points": total_points,
                "preview_path": str(preview_path) if preview_written else None,
            })

    if progress_callback is not None:
        progress_callback({"stage": "meshing", "progress": 86.0, "message": "Building the surface mesh"})
    if unpaired_on:
        run_warnings.append(f"{unpaired_on} ON frame(s) had no preceding OFF frame (no ambient subtraction).")
    if n_on_processed == 0:
        raise RuntimeError("No laser points were triangulated; check calibration, z-range and frames.")

    mesh, pcd = reconstruct_mesh(fuser, cfg=mesh_cfg)
    paths = export_scan_outputs(out_dir, mesh, pcd)
    if progress_callback is not None:
        progress_callback({"stage": "mesh_ready", "progress": 100.0, "message": "Surface reconstruction complete", "mesh_path": str(paths["mesh_ply"])})
    if len(mesh.triangles) == 0:
        run_warnings.append("Mesh is empty (too few points); no mesh files were written.")

    all_fidxs = sorted(tracker.poses.keys())
    all_T = np.array([tracker.poses[f] for f in all_fidxs], dtype=np.float64) if all_fidxs else np.empty((0, 4, 4))
    poses_npz_path = out_dir / "estimated_poses.npz"
    np.savez_compressed(poses_npz_path, frame_indices=np.array(all_fidxs), T_wc=all_T)
    path_mm = float(np.linalg.norm(np.diff(all_T[:, :3, 3], axis=0), axis=1).sum()) if len(all_T) > 1 else 0.0
    if path_mm < 30.0:
        run_warnings.append(f"Trajectory is only {path_mm:.1f} mm long: a local patch, not a full arch.")
    run_warnings.append("A single laser pass observes the occlusal surface only; walls are not captured.")

    seg_report = None
    if segment:
        if len(mesh.triangles) > 20:
            from ioscan.pipeline.segment import SegmentConfig, segment_mesh_and_cloud  # needs the [seg] extra
            seg_report = segment_mesh_and_cloud(
                paths["mesh_ply"], out_dir, cfg=seg_cfg or SegmentConfig(),
                cloud_path=paths["cloud_ply"], session_hint=session_dir,
            )
        else:
            run_warnings.append("--segment requested but the mesh has <= 20 triangles; segmentation skipped.")

    report_path = out_dir / "pipeline_report.json"
    report_path.write_text(json.dumps({
        "session": str(session_dir),
        "session_source": source,
        "pose_mode": tracker.cfg.mode,
        "nominal_speed_mm_s": speed_mm_s,
        "ridge_method": ridge_method,
        "mesh_method": mesh_method,
        "camera": camera.to_dict(),
        "laser_plane": laser.to_dict(),
        "calibration_is_nominal": nominal_cal,
        "frames_total": len(records),
        "laser_on_frames_with_points": n_on_processed,
        "raw_triangulated_points": total_points,
        "fused_cloud_points": len(pcd.points),
        "mesh_vertices": len(mesh.vertices),
        "mesh_triangles": len(mesh.triangles),
        "estimated_trajectory_length_mm": path_mm,
        "warnings": run_warnings,
        "paths": {k: str(v) for k, v in paths.items()},
        "segmentation": seg_report,
    }, indent=2), encoding="utf-8")

    return {
        "on_frames_with_points": n_on_processed,
        "raw_triangulated_points": total_points,
        "fused_cloud_points": len(pcd.points),
        "mesh_vertices": len(mesh.vertices),
        "mesh_triangles": len(mesh.triangles),
        "path_mm": path_mm,
        "paths": paths,
        "segmentation": seg_report,
        "warnings": run_warnings,
        "report_json": report_path,
        "poses_npz": poses_npz_path,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the 3D intraoral reconstruction pipeline.")
    ap.add_argument("session", type=Path, help="Input session directory")
    ap.add_argument("--out", type=Path, default=Path("data/results/latest_scan"))
    ap.add_argument("--pose-mode", choices=["imu_hybrid", "imu_const_vel", "gt", "submap_icp"], default="imu_hybrid")
    ap.add_argument("--speed-mm-s", type=float, default=5.0, help="Assumed scan speed for imu_hybrid")
    ap.add_argument("--world-velocity", type=float, nargs=3, metavar=("VX", "VY", "VZ"), default=None,
                    help="World velocity (mm/s) for imu_const_vel")
    ap.add_argument("--init-from-gt", action="store_true",
                    help="Simulator sessions only: start at the ground-truth position (for --no-icp evaluation)")
    ap.add_argument("--ridge", choices=["cog", "gaussian3", "steger"], default="cog")
    ap.add_argument("--mesh", choices=["bpa", "poisson", "tsdf"], default="bpa")
    ap.add_argument("--voxel-mm", type=float, default=0.08)
    ap.add_argument("--z-min", type=float, default=10.0, help="Minimum accepted depth (mm)")
    ap.add_argument("--z-max", type=float, default=45.0, help="Maximum accepted depth (mm)")
    ap.add_argument("--segment", action="store_true", help="Run MeshSegNet segmentation (needs the [seg] extra)")
    ap.add_argument("--jaw", choices=["auto", "lower", "upper"], default="auto")
    ap.add_argument("--models-dir", type=Path, default=None, help="MeshSegNet checkpoints (or $IOSCAN_MESHSEGNET_MODELS)")
    ap.add_argument("--ckpt", type=Path, default=None, help="Explicit MeshSegNet checkpoint")
    ap.add_argument("--repo-dir", type=Path, default=None,
                    help="Optional reference MeshSegNet checkout containing meshsegnet.py")
    ap.add_argument("--target-cells", type=int, default=10000,
                    help="Maximum triangle count passed to MeshSegNet (default: 10000)")
    ap.add_argument("--seg-device", type=str, default=None,
                    help="MeshSegNet device, e.g. cpu or cuda:0 (default: auto with CPU fallback)")
    args = ap.parse_args()

    seg_cfg = None
    if args.segment:
        from ioscan.pipeline.segment import SegmentConfig
        kw = dict(jaw=args.jaw, ckpt_path=args.ckpt, repo_dir=args.repo_dir,
                  target_cells=args.target_cells, device=args.seg_device)
        if args.models_dir is not None:
            kw["models_dir"] = args.models_dir
        seg_cfg = SegmentConfig(**kw)

    stats = run_pipeline(
        session_dir=args.session, out_dir=args.out, pose_mode=args.pose_mode, ridge_method=args.ridge,
        mesh_method=args.mesh, voxel_size_mm=args.voxel_mm, speed_mm_s=args.speed_mm_s,
        world_velocity_mm_s=tuple(args.world_velocity) if args.world_velocity else None,
        init_from_gt=args.init_from_gt, z_range=(args.z_min, args.z_max),
        segment=args.segment, seg_cfg=seg_cfg,
    )

    print("\n=== 3D Reconstruction Complete ===")
    print(f"  ON frames processed : {stats['on_frames_with_points']}")
    print(f"  Raw 3D points       : {stats['raw_triangulated_points']}")
    print(f"  Fused cloud points  : {stats['fused_cloud_points']}")
    print(f"  Mesh vertices       : {stats['mesh_vertices']}")
    print(f"  Mesh triangles      : {stats['mesh_triangles']}")
    print(f"  Estimated path      : {stats['path_mm']:.2f} mm")
    print(f"  Saved Point Cloud   : {stats['paths']['cloud_ply']}")
    print(f"  Saved 3D Mesh (STL) : {stats['paths']['mesh_stl']}")
    print(f"  Saved pose/report   : {stats['poses_npz']} / {stats['report_json']}")
    for w in stats["warnings"]:
        print(f"  WARNING: {w}")

    seg = stats.get("segmentation")
    if seg:
        print("\n=== MeshSegNet Segmentation Summary ===")
        print(f"  Model Checkpoint    : {seg['checkpoint']} (jaw: {seg['jaw']})")
        print(f"  Predicted Gingiva   : {seg['gingiva_pct']:.1f}% | Teeth: {seg['teeth_pct']:.1f}%")
        print(f"  Segmented Full Mesh : {seg['full_colored_ply']}")
        print(f"  Per-tooth meshes    : {len(seg['per_tooth_meshes'])} class(es)")
        print(f"  Label data          : {seg['segmentation_labels_npz']}")
        print(f"  Inference device    : {seg['device']}")
        print(f"  Clean Teeth Mesh    : {seg['clean_teeth_ply'] or 'NOT WRITTEN (no teeth predicted)'}")
        if seg["segmented_cloud_ply"]:
            print(f"  Segmented Cloud     : {seg['segmented_cloud_ply']}")
        for w in seg["warnings"]:
            print(f"  WARNING: {w}")


if __name__ == "__main__":
    main()
