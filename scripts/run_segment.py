from __future__ import annotations

# Prefer this checkout's src/ package over any stale ioscan installation in site-packages.
import sys as _sys
from pathlib import Path as _Path
_project_src = _Path(__file__).resolve().parents[1] / "src"
if _project_src.is_dir():
    _sys.path.insert(0, str(_project_src))
import argparse
from pathlib import Path

from ioscan.pipeline.segment import SegmentConfig, segment_mesh_and_cloud


def main() -> None:
    ap = argparse.ArgumentParser(description="Run MeshSegNet 3D tooth segmentation on a reconstructed scan.")
    ap.add_argument("input_path", type=Path,
                    help="Pipeline result directory (with reconstructed_mesh.ply) or a .ply/.obj/.stl file")
    ap.add_argument("--out", type=Path, default=None, help="Output directory (default: input directory)")
    ap.add_argument("--session", type=Path, default=None,
                    help="Session directory; its meta.json supplies the jaw when --jaw auto")
    ap.add_argument("--jaw", choices=["auto", "lower", "upper"], default="auto")
    ap.add_argument("--models-dir", type=Path, default=None,
                    help="Directory with MeshSegNet checkpoints (default: $IOSCAN_MESHSEGNET_MODELS)")
    ap.add_argument("--ckpt", type=Path, default=None, help="Explicit checkpoint file (must exist)")
    ap.add_argument("--repo-dir", type=Path, default=None,
                    help="Optional MeshSegNet repo containing meshsegnet.py (default: bundled compatible model)")
    ap.add_argument("--target-cells", type=int, default=10000,
                    help="Maximum triangle count passed to MeshSegNet (default: 10000)")
    ap.add_argument("--device", type=str, default=None,
                    help="Inference device, e.g. cpu or cuda:0 (default: auto with CPU fallback)")
    args = ap.parse_args()

    if args.input_path.is_dir():
        mesh_path = args.input_path / "reconstructed_mesh.ply"
        if not mesh_path.exists():
            mesh_path = args.input_path / "reconstructed_mesh.stl"
        cloud_path = args.input_path / "fused_cloud.ply"
        out_dir = args.out or args.input_path
    else:
        mesh_path = args.input_path
        cloud_path = None
        out_dir = args.out or args.input_path.parent
    if not mesh_path.exists():
        raise FileNotFoundError(f"Could not find mesh at {mesh_path}")

    kwargs = dict(jaw=args.jaw, ckpt_path=args.ckpt, target_cells=args.target_cells,
                  repo_dir=args.repo_dir, device=args.device)
    if args.models_dir is not None:
        kwargs["models_dir"] = args.models_dir
    res = segment_mesh_and_cloud(mesh_path, out_dir, cfg=SegmentConfig(**kwargs),
                                 cloud_path=cloud_path, session_hint=args.session or "")

    print("\n=== MeshSegNet 3D Tooth Segmentation Complete ===")
    print(f"  Checkpoint used    : {res['checkpoint']} (jaw: {res['jaw']})")
    print(f"  Mesh cells         : {res['original_cells']} -> {res['decimated_cells']} decimated")
    print(f"  Predicted Gingiva  : {res['gingiva_pct']:.1f}%   Teeth: {res['teeth_pct']:.1f}% "
          f"(classes {res['detected_classes']})")
    print(f"  Colored Full Mesh  : {res['full_colored_ply']}")
    print(f"  Clean Teeth Mesh   : {res['clean_teeth_ply'] or 'NOT WRITTEN (no teeth predicted)'}")
    print(f"  Per-tooth meshes   : {len(res['per_tooth_meshes'])} class(es) in {out_dir / 'segmented_teeth_by_class'}")
    print(f"  Label data         : {res['segmentation_labels_npz']}")
    print(f"  Inference device   : {res['device']}")
    if res["segmented_cloud_ply"]:
        print(f"  Segmented Cloud    : {res['segmented_cloud_ply']}")
    for w in res["warnings"]:
        print(f"  WARNING: {w}")


if __name__ == "__main__":
    main()
