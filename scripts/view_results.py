from __future__ import annotations

# Prefer this checkout's src/ package over any stale ioscan installation in site-packages.
import sys as _sys
from pathlib import Path as _Path
_project_src = _Path(__file__).resolve().parents[1] / "src"
if _project_src.is_dir():
    _sys.path.insert(0, str(_project_src))
import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="View reconstructed or segmented 3D dental mesh and point cloud.")
    ap.add_argument("result_dir", type=Path, help="Directory containing scan outputs")
    ap.add_argument("--cloud-only", action="store_true", help="Show only the fused point cloud")
    ap.add_argument("--segmented", action="store_true", help="Show MeshSegNet colored full mesh")
    ap.add_argument("--clean-teeth", action="store_true", help="Show MeshSegNet gingiva-removed teeth mesh")
    args = ap.parse_args()

    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "The 3D viewer requires Open3D. Install project dependencies with: "
            "python -m pip install -e ."
        ) from exc

    if args.segmented and args.clean_teeth:
        ap.error("choose only one of --segmented or --clean-teeth")

    geoms = []
    if args.segmented:
        p = args.result_dir / "segmented_full_colored.ply"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run pipeline with --segment first.")
        mesh = o3d.io.read_triangle_mesh(str(p))
        mesh.compute_vertex_normals()
        geoms.append(mesh)
    elif args.clean_teeth:
        p = args.result_dir / "segmented_clean_teeth.ply"
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Run the pipeline with --segment (note: no file is written when no teeth are predicted).")
        mesh = o3d.io.read_triangle_mesh(str(p))
        mesh.compute_vertex_normals()
        geoms.append(mesh)
    elif args.cloud_only:
        seg_c = args.result_dir / "segmented_cloud.ply"
        cloud_path = seg_c if seg_c.exists() else (args.result_dir / "fused_cloud.ply")
        if not cloud_path.exists():
            raise FileNotFoundError(f"Missing point cloud at {cloud_path}")
        pcd = o3d.io.read_point_cloud(str(cloud_path))
        if not pcd.has_colors():
            pcd.paint_uniform_color([0.1, 0.7, 0.3])
        geoms.append(pcd)
    else:
        mesh_path = args.result_dir / "reconstructed_mesh.ply"
        if not mesh_path.exists():
            mesh_path = args.result_dir / "reconstructed_mesh.stl"
        if not mesh_path.exists():
            raise FileNotFoundError(f"No mesh found in {args.result_dir}")
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.85, 0.82, 0.75])
        geoms.append(mesh)

    o3d.visualization.draw_geometries(
        geoms,
        window_name=f"Intraoral Scan - {args.result_dir.name}",
        width=1280,
        height=800,
        mesh_show_back_face=True,
    )


if __name__ == "__main__":
    main()