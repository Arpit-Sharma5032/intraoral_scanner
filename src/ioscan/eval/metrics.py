from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np
import open3d as o3d


@dataclass
class ScanMetrics:
    n_scan_points: int
    trueness_mean_um: float
    trueness_rms_um: float
    trueness_median_um: float
    trueness_p95_um: float
    trueness_max_um: float
    completeness_100um_pct: float  # % of reference ROI within 100 um of scan


def compute_cloud_to_mesh_metrics(
    scan_pcd: o3d.geometry.PointCloud,
    ref_mesh: o3d.geometry.TriangleMesh,
    roi_margin_mm: float = 0.5,
) -> tuple[ScanMetrics, np.ndarray]:
    """
    Compute distances from scan_pcd to ref_mesh surface (using Open3D RaycastingScene)
    and completeness over the scanned bounding box ROI.
    """
    scan_pts = np.asarray(scan_pcd.points, dtype=np.float32)
    if scan_pts.shape[0] == 0:
        raise ValueError("Scan point cloud is empty.")

    # Use Open3D tensor RaycastingScene for exact point-to-triangle-surface distance
    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(ref_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(t_mesh)

    unsigned_dist_mm = scene.compute_distance(o3d.core.Tensor(scan_pts)).numpy()
    dist_um = unsigned_dist_mm.astype(np.float64) * 1000.0

    # Completeness within the scanned bounding box (so partial arch scans are judged fairly)
    bbox = scan_pcd.get_axis_aligned_bounding_box()
    min_b = bbox.get_min_bound() - roi_margin_mm
    max_b = bbox.get_max_bound() + roi_margin_mm
    roi_box = o3d.geometry.AxisAlignedBoundingBox(min_b, max_b)
    ref_roi = ref_mesh.crop(roi_box)

    if len(ref_roi.vertices) > 10:
        ref_pcd = o3d.geometry.PointCloud()
        ref_pcd.points = ref_roi.vertices
        d_ref_to_scan = np.asarray(ref_pcd.compute_point_cloud_distance(scan_pcd)) * 1000.0
        comp_100 = float(np.mean(d_ref_to_scan <= 100.0) * 100.0)
    else:
        comp_100 = 0.0

    metrics = ScanMetrics(
        n_scan_points=int(dist_um.size),
        trueness_mean_um=float(np.mean(dist_um)),
        trueness_rms_um=float(np.sqrt(np.mean(dist_um ** 2))),
        trueness_median_um=float(np.median(dist_um)),
        trueness_p95_um=float(np.percentile(dist_um, 95)),
        trueness_max_um=float(np.max(dist_um)),
        completeness_100um_pct=comp_100,
    )
    return metrics, dist_um


def save_eval_report(metrics: ScanMetrics, out_path: str | Path) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")
    return p