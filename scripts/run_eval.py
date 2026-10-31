from __future__ import annotations

# Prefer this checkout's src/ package over any stale ioscan installation in site-packages.
import sys as _sys
from pathlib import Path as _Path
_project_src = _Path(__file__).resolve().parents[1] / "src"
if _project_src.is_dir():
    _sys.path.insert(0, str(_project_src))
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
import trimesh

from ioscan.eval.metrics import compute_cloud_to_mesh_metrics, save_eval_report
from ioscan.pipeline.pose import quat_wxyz_to_rotmat


def _extract_gt_world_points(session_dir: Path) -> np.ndarray:
    """
    Extract exact ground-truth 3D world points from session_dir/gt/profiles.npz
    and session_dir/gt/poses.npz regardless of internal key naming.
    """
    prof_path = session_dir / "gt" / "profiles.npz"
    pose_path = session_dir / "gt" / "poses.npz"

    prof_data = np.load(prof_path)
    p_keys = list(prof_data.files)

    # 1. Check if world points are already stored directly in profiles.npz
    for k in p_keys:
        if "world" in k.lower() or "pts_w" in k.lower():
            arr = prof_data[k]
            if arr.ndim == 2 and arr.shape[1] == 3 and len(arr) > 0:
                return arr.astype(np.float64)

    # 2. Otherwise, extract camera-frame points and frame indices
    f_key = next((k for k in p_keys if prof_data[k].ndim == 1), p_keys[0])
    pts_key = next((k for k in p_keys if prof_data[k].ndim == 2 and prof_data[k].shape[1] == 3), None)
    if pts_key is None:
        raise KeyError(f"Could not find (N,3) points array in {prof_path}. Keys: {p_keys}")

    fidxs = prof_data[f_key].astype(int)
    pts_cam_all = prof_data[pts_key].astype(np.float64)

    # 3. Load camera-to-world poses from poses.npz
    pose_data = np.load(pose_path)
    q_keys = list(pose_data.files)

    # Case A: (N, 4, 4) homogeneous matrices T_wc
    T_key = next((k for k in q_keys if pose_data[k].ndim == 3 and pose_data[k].shape[1:] == (4, 4)), None)
    # Case B: (N, 3, 3) R_wc and (N, 3) t_wc
    R_key = next((k for k in q_keys if pose_data[k].ndim == 3 and pose_data[k].shape[1:] == (3, 3)), None)
    t_key = next((k for k in q_keys if pose_data[k].ndim == 2 and pose_data[k].shape[1] == 3), None)
    # Case C: (N, 4) quaternion + (N, 3) translation
    quat_key = next((k for k in q_keys if pose_data[k].ndim == 2 and pose_data[k].shape[1] == 4), None)

    T_all: dict[int, np.ndarray] = {}
    if T_key is not None:
        arr = pose_data[T_key].astype(np.float64)
        for i in range(len(arr)):
            T_all[i] = arr[i]
    elif R_key is not None and t_key is not None:
        R_arr = pose_data[R_key].astype(np.float64)
        t_arr = pose_data[t_key].astype(np.float64)
        for i in range(min(len(R_arr), len(t_arr))):
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R_arr[i]
            T[:3, 3] = t_arr[i]
            T_all[i] = T
    elif quat_key is not None and t_key is not None:
        q_arr = pose_data[quat_key].astype(np.float64)
        t_arr = pose_data[t_key].astype(np.float64)
        for i in range(min(len(q_arr), len(t_arr))):
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = quat_wxyz_to_rotmat(q_arr[i])
            T[:3, 3] = t_arr[i]
            T_all[i] = T
    else:
        raise KeyError(
            f"Unrecognized pose format in {pose_path}. Keys & shapes: "
            f"{[(k, pose_data[k].shape) for k in q_keys]}"
        )

    world_pts = []
    for fidx in np.unique(fidxs):
        if fidx in T_all:
            T = T_all[fidx]
            pts_c = pts_cam_all[fidxs == fidx]
            pts_w = (pts_c @ T[:3, :3].T) + T[:3, 3]
            world_pts.append(pts_w)

    if not world_pts:
        raise RuntimeError("No matching frame indices between profiles.npz and poses.npz")
    return np.vstack(world_pts)


def _build_gt_mesh_from_session(session_dir: Path) -> o3d.geometry.TriangleMesh:
    """Build a dense ground-truth reference surface mesh in world coordinates."""
    all_w = _extract_gt_world_points(session_dir)

    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(all_w)
    gt_pcd = gt_pcd.voxel_down_sample(voxel_size=0.05)
    gt_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    # Orient toward where the camera actually was (not an assumed +Z world direction).
    pose_data = np.load(session_dir / "gt" / "poses.npz")
    cam_center = pose_data["T_wc"][:, :3, 3].mean(axis=0) if "T_wc" in pose_data.files else gt_pcd.get_center()
    gt_pcd.orient_normals_towards_camera_location(cam_center)

    radii = [0.10, 0.20, 0.40]
    gt_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        gt_pcd, o3d.utility.DoubleVector(radii)
    )
    gt_mesh.compute_vertex_normals()
    return gt_mesh


def _align_scan_to_roi_mesh(
    scan_pcd: o3d.geometry.PointCloud,
    ref_mesh: o3d.geometry.TriangleMesh,
    use_icp: bool = True,
) -> o3d.geometry.PointCloud:
    """Align scan_pcd to the single-tooth reference ROI mesh."""
    aligned = o3d.geometry.PointCloud(scan_pcd)
    if not use_icp:
        # A true no-ICP evaluation compares the original world coordinates. Do not apply a
        # centroid translation here; that would hide the very pose drift this mode measures.
        return aligned

    ref_pcd = ref_mesh.sample_points_uniformly(number_of_points=30000)
    ref_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))

    d_init = float(np.median(np.asarray(aligned.compute_point_cloud_distance(ref_pcd))))
    if d_init > 1.0:
        shift = ref_pcd.get_center() - aligned.get_center()
        aligned.translate(shift)

    if use_icp:
        for max_corr in (2.0, 0.5, 0.15):
            reg = o3d.pipelines.registration.registration_icp(
                aligned,
                ref_pcd,
                max_correspondence_distance=max_corr,
                init=np.eye(4),
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
            )
            aligned.transform(reg.transformation)

    return aligned


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate reconstructed scan against ground-truth surface.")
    ap.add_argument("result_dir", type=Path, help="Directory with fused_cloud.ply")
    ap.add_argument(
        "--session",
        type=Path,
        default=Path("data/sessions/FUYJPUPX_lower"),
        help="Session directory containing gt/profiles.npz and gt/poses.npz",
    )
    ap.add_argument("--gt-mesh", type=Path, default=None, help="Reference mesh (default: <session>/gt/reference_mesh.ply when present)")
    ap.add_argument("--no-icp", action="store_true", help="Disable rigid ICP alignment")
    args = ap.parse_args()

    cloud_path = args.result_dir / "fused_cloud.ply"
    if not cloud_path.exists():
        raise FileNotFoundError(f"Missing reconstructed point cloud at {cloud_path}")

    scan_pcd = o3d.io.read_point_cloud(str(cloud_path))

    gt_mesh_path = args.gt_mesh
    if gt_mesh_path is None:
        default_ref = args.session / "gt" / "reference_mesh.ply"
        if default_ref.exists():
            gt_mesh_path = default_ref
    if gt_mesh_path is not None:
        if not gt_mesh_path.exists():
            raise FileNotFoundError(f"Ground-truth mesh not found: {gt_mesh_path}")
        t_mesh = trimesh.load(str(gt_mesh_path), process=False)
        if isinstance(t_mesh, trimesh.Scene):
            t_mesh = t_mesh.dump(concatenate=True)
        ref_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(t_mesh.vertices, dtype=np.float64)),
            o3d.utility.Vector3iVector(np.asarray(t_mesh.faces, dtype=np.int32)),
        )
        ref_mesh.compute_vertex_normals()
        print(f"Reference surface: {gt_mesh_path}")
    else:
        print(
            "WARNING: no reference mesh found; building the reference from the session's own laser "
            "samples. This is circular (it cannot reveal accuracy errors). Pass --gt-mesh."
        )
        ref_mesh = _build_gt_mesh_from_session(args.session)

    aligned_pcd = _align_scan_to_roi_mesh(scan_pcd, ref_mesh, use_icp=(not args.no_icp))
    metrics, _ = compute_cloud_to_mesh_metrics(aligned_pcd, ref_mesh)
    report_path = save_eval_report(metrics, args.result_dir / "eval_report.json")

    print(f"=== Cloud-to-Mesh Evaluation ({args.result_dir.name}) ===")
    print(f"  Evaluated Points       : {metrics.n_scan_points}")
    print(f"  Trueness Mean (µm)     : {metrics.trueness_mean_um:.2f} µm")
    print(f"  Trueness RMS (µm)      : {metrics.trueness_rms_um:.2f} µm")
    print(f"  Trueness Median (µm)   : {metrics.trueness_median_um:.2f} µm")
    print(f"  Trueness P95 (µm)      : {metrics.trueness_p95_um:.2f} µm")
    print(f"  Completeness (<100 µm) : {metrics.completeness_100um_pct:.1f}%")
    print(f"  Saved JSON Report      : {report_path}")


if __name__ == "__main__":
    main()