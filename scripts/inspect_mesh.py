"""Inspect a Teeth3DS case: print stats, optionally open a 3D viewer.

Examples:
    python scripts/inspect_mesh.py                    # first lower case
    python scripts/inspect_mesh.py --jaw upper
    python scripts/inspect_mesh.py --case FUYJPUPX --view
"""

# Prefer this checkout's src/ package over any stale ioscan installation in site-packages.
import sys as _sys
from pathlib import Path as _Path
_project_src = _Path(__file__).resolve().parents[1] / "src"
if _project_src.is_dir():
    _sys.path.insert(0, str(_project_src))
import argparse

from ioscan.sim.teeth3ds import describe, list_cases, load_labels, load_mesh


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="Dataset root (default: $IOSCAN_TEETH3DS)")
    ap.add_argument("--jaw", default="lower", choices=["lower", "upper"])
    ap.add_argument("--case", default=None, help="Case ID (default: first found)")
    ap.add_argument("--view", action="store_true", help="Open an Open3D viewer")
    args = ap.parse_args()

    cases = list_cases(args.root, args.jaw)
    print(f"{len(cases)} {args.jaw} cases found")
    if not cases:
        return

    case = next((c for c in cases if c.case_id == args.case), None) if args.case else cases[0]
    if case is None:
        raise SystemExit(f"Case {args.case!r} not found")

    mesh = load_mesh(case.obj_path)
    print(f"Case: {case.case_id}  ({case.obj_path})")
    for k, v in describe(mesh).items():
        print(f"  {k}: {v}")

    if case.json_path:
        labels = load_labels(case.json_path)
        for k, v in labels.items():
            print(f"  {k}: {len(v)} entries, {len(set(v.tolist()))} unique")

    if args.view:
        import open3d as o3d

        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        m.triangles = o3d.utility.Vector3iVector(mesh.faces)
        m.compute_vertex_normals()
        o3d.visualization.draw_geometries([m], window_name=case.case_id)


if __name__ == "__main__":
    main()