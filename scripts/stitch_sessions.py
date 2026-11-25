"""Stitch several straight-pass (sweep) reconstructions into one full-arch mesh.

    python scripts/stitch_sessions.py scan_A scan_B scan_C scan_D scan_E --out data/results/arch_1 [--segment --jaw lower]

Give the passes IN SCAN ORDER around the arch; each must overlap the previous one by one or two
teeth. Arguments may be session names (results are read from data/results/<name>) or results
folders. Each pass must have been reconstructed in sweep mode first (the default pipeline).
Outputs (canonical dental frame: arch in XY, teeth toward +Z, incisors toward +Y, mm):
reconstructed_mesh.ply/.stl, fused_cloud.ply, pipeline_report.json.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_src = _Path(__file__).resolve().parents[1] / "src"
if _src.is_dir():
    _sys.path.insert(0, str(_src))

import argparse
from datetime import datetime
from pathlib import Path

from ioscan.pipeline.stitch import StitchConfig, stitch_to_files

ROOT = Path(__file__).resolve().parents[1]


def resolve_pass(arg: str) -> Path:
    p = Path(arg)
    if (p / "fused_cloud.ply").is_file():
        return p
    q = ROOT / "data" / "results" / arg
    if (q / "fused_cloud.ply").is_file():
        return q
    raise SystemExit(f"Pass not found or not reconstructed: {arg} (looked in {p} and {q})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("passes", nargs="+", help="sessions or results folders, in scan order")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--keep-base", action="store_true", help="keep the whole model base (default: teeth + 5 mm gum)")
    ap.add_argument("--segment", action="store_true", help="run MeshSegNet on the stitched arch")
    ap.add_argument("--jaw", choices=["auto", "lower", "upper"], default="auto")
    ap.add_argument("--models-dir", type=Path, default=None)
    args = ap.parse_args()
    if len(args.passes) < 2:
        raise SystemExit("Give at least two passes.")
    dirs = [resolve_pass(a) for a in args.passes]
    out = args.out or ROOT / "data" / "results" / datetime.now().strftime("arch_%Y%m%d_%H%M%S")
    cfg = StitchConfig()
    if args.keep_base:
        cfg.trim_to_teeth_mm = None
    rep = stitch_to_files(dirs, out, cfg, progress=lambda m, p: print(f"  [{p:5.1f}%] {m}", flush=True))
    print("\n=== Stitched arch ===")
    for lk in rep["links"]:
        g = lk.get("icp_global", lk["icp"])
        imu = lk["imu_rotation_vs_previous_deg"]
        print(f"  {lk['pass']:<28} relief match {lk['coarse']['relief_ncc']:.3f}  overlap {100 * g['overlap_frac']:4.0f}%  "
              f"RMS {g['rmse_mm']:.3f} mm  turn {lk['rotation_vs_previous_deg']:6.1f} deg"
              + (f" (IMU {imu:.1f})" if imu is not None else ""))
    q = rep["quality"]
    print(f"  Mesh: {rep['mesh_vertices']:,} vertices, {rep['mesh_triangles']:,} faces, extent "
          f"{' x '.join(f'{e:.1f}' for e in q['mesh_extent_mm'])} mm")
    print(f"  Output: {out}")
    for w in rep["warnings"]:
        print(f"  WARNING: {w}")
    if args.segment:
        if not q["segmentation_ready_by_basic_geometry_gate"]:
            print("  Segmentation skipped: the mesh-quality gate failed (see warnings).")
            return
        from ioscan.pipeline.segment import SegmentConfig, segment_mesh_and_cloud
        kw = {"jaw": args.jaw}
        if args.models_dir:
            kw["models_dir"] = args.models_dir
        seg = segment_mesh_and_cloud(out / "reconstructed_mesh.ply", out, cfg=SegmentConfig(**kw),
                                     cloud_path=out / "fused_cloud.ply", session_hint=dirs[0])
        print(f"  MeshSegNet: teeth {seg['teeth_pct']:.1f}%, {len(seg['per_tooth_meshes'])} tooth classes -> {seg['full_colored_ply']}")


if __name__ == "__main__":
    main()
