from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from ioscan.pipeline.fusion import ScanFuser


@dataclass
class MeshConfig:
    method: str = "bpa"  # bpa | poisson | tsdf
    poisson_depth: int = 9
    density_trim_quantile: float = 0.12
    taubin_iterations: int = 3
    remove_outliers: bool = True
    keep_largest_cluster: bool = False
    min_component_triangles: int = 80
    min_component_fraction: float = 0.01


def _adaptive_bpa_radii(pcd: o3d.geometry.PointCloud) -> list[float]:
    """Choose BPA ball radii from observed point spacing (all distances are millimetres)."""
    distances = np.asarray(pcd.compute_nearest_neighbor_distance(), dtype=np.float64)
    distances = distances[np.isfinite(distances) & (distances > 1e-6)]
    if distances.size == 0:
        return [0.12, 0.22, 0.36]
    spacing = float(np.median(distances))
    # Small, medium and bridging radii. The upper cap avoids stitching across large gaps.
    radii = [
        float(np.clip(1.5 * spacing, 0.08, 0.18)),
        float(np.clip(2.5 * spacing, 0.12, 0.26)),
        float(np.clip(3.5 * spacing, 0.16, 0.34)),
    ]
    return sorted(set(radii))


def _clean_mesh(mesh: o3d.geometry.TriangleMesh, cfg: MeshConfig) -> o3d.geometry.TriangleMesh:
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return mesh

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    if len(mesh.triangles) > 0:
        clusters, counts, _ = mesh.cluster_connected_triangles()
        counts = np.asarray(counts, dtype=np.int64)
        if counts.size:
            if cfg.keep_largest_cluster:
                keep_ids = {int(np.argmax(counts))}
            else:
                threshold = max(
                    int(cfg.min_component_triangles),
                    int(np.ceil(float(counts.max()) * cfg.min_component_fraction)),
                )
                keep_ids = set(np.flatnonzero(counts >= threshold).astype(int).tolist())
                if not keep_ids:
                    keep_ids = {int(np.argmax(counts))}
            remove = [i for i, cluster_id in enumerate(clusters) if int(cluster_id) not in keep_ids]
            if remove:
                mesh.remove_triangles_by_index(remove)
                mesh.remove_unreferenced_vertices()

    if cfg.taubin_iterations > 0 and len(mesh.triangles) > 100:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=cfg.taubin_iterations)
    if len(mesh.vertices) > 0:
        mesh.compute_vertex_normals()
    return mesh


def reconstruct_mesh(
    fuser: ScanFuser,
    cfg: MeshConfig | None = None,
) -> tuple[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud]:
    """Reconstruct a mesh and return it with the fused millimetre-space point cloud."""
    cfg = cfg or MeshConfig()
    pcd = fuser.get_point_cloud(voxel_downsample_mm=fuser.cfg.voxel_size_mm)
    if len(pcd.points) < 50:
        return o3d.geometry.TriangleMesh(), pcd

    if cfg.remove_outliers and len(pcd.points) > 100:
        cleaned, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        # Avoid accidentally discarding most of a thin line-scan surface.
        if len(cleaned.points) >= 0.70 * len(pcd.points):
            pcd = cleaned

    method = cfg.method.lower()
    if method == "tsdf" and fuser.tsdf is not None:
        mesh = fuser.tsdf.extract_triangle_mesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * 1000.0)
        if len(mesh.vertices) > 0:
            mesh.compute_vertex_normals()
            return mesh, pcd
        method = "bpa"

    if method == "bpa":
        radii = _adaptive_bpa_radii(pcd)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
    elif method == "poisson":
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=cfg.poisson_depth
        )
        density = np.asarray(densities)
        if density.size and cfg.density_trim_quantile > 0:
            threshold = np.quantile(density, cfg.density_trim_quantile)
            mesh.remove_vertices_by_mask(density < threshold)
        mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())
    else:
        raise ValueError(f"Unknown mesh method: {cfg.method!r}")

    mesh = _clean_mesh(mesh, cfg)
    return mesh, pcd


def export_scan_outputs(
    out_dir: str | Path,
    mesh: o3d.geometry.TriangleMesh,
    pcd: o3d.geometry.PointCloud,
) -> dict[str, Path]:
    """Write point cloud and mesh in PLY/STL formats; all coordinates are millimetres."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cloud_path = out / "fused_cloud.ply"
    mesh_ply_path = out / "reconstructed_mesh.ply"
    mesh_stl_path = out / "reconstructed_mesh.stl"

    if len(pcd.points) > 0:
        if not o3d.io.write_point_cloud(str(cloud_path), pcd):
            raise IOError(f"Failed to write point cloud: {cloud_path}")
    elif cloud_path.exists():
        # Avoid leaving a previous run's point cloud masquerading as the current result.
        cloud_path.unlink()

    if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        mesh.compute_vertex_normals()
        if not o3d.io.write_triangle_mesh(str(mesh_ply_path), mesh):
            raise IOError(f"Failed to write mesh: {mesh_ply_path}")
        if not o3d.io.write_triangle_mesh(str(mesh_stl_path), mesh):
            raise IOError(f"Failed to write mesh: {mesh_stl_path}")
    else:
        # Empty meshes must remove stale meshes from an earlier successful run.
        for stale_path in (mesh_ply_path, mesh_stl_path):
            if stale_path.exists():
                stale_path.unlink()

    return {"cloud_ply": cloud_path, "mesh_ply": mesh_ply_path, "mesh_stl": mesh_stl_path}
