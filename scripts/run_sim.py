"""Generate a simulated scan session from a Teeth3DS case.

Examples:
    python scripts/run_sim.py --root "C:\\path\\to\\Teeth3DS" --max-frames 100 --overwrite
    python scripts/run_sim.py --root "C:\\path\\to\\Teeth3DS" --case FUYJPUPX --overwrite
"""

# Prefer this checkout's src/ package over any stale ioscan installation in site-packages.
import sys as _sys
from pathlib import Path as _Path
_project_src = _Path(__file__).resolve().parents[1] / "src"
if _project_src.is_dir():
    _sys.path.insert(0, str(_project_src))
import argparse
from pathlib import Path

from ioscan.sim.generate import generate_session
from ioscan.utils.config import repo_root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="Teeth3DS root (default: sim.yaml / $IOSCAN_TEETH3DS)")
    ap.add_argument("--jaw", default=None, choices=["lower", "upper"])
    ap.add_argument("--case", default=None, help="Case ID (default: from sim.yaml)")
    ap.add_argument("--out", default=None, help="Output session dir (default: data/sessions/<case>_<jaw>)")
    ap.add_argument("--max-frames", type=int, default=None, help="Cap frame count for quick tests")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    # Resolve the output directory only after we know the case, via a temporary default
    out = Path(args.out) if args.out else None
    if out is None:
        from ioscan.utils.config import load_yaml

        sim = load_yaml(repo_root() / "configs" / "sim.yaml")
        case = args.case or sim["teeth3ds"].get("case_id") or "first"
        jaw = args.jaw or sim["teeth3ds"]["jaw"]
        out = repo_root() / "data" / "sessions" / f"{case}_{jaw}"

    info = generate_session(
        out, root=args.root, case_id=args.case, jaw=args.jaw,
        max_frames=args.max_frames, overwrite=args.overwrite,
    )
    print(f"\nSession written to {info['out_dir']}")
    print(f"  case {info['case']} ({info['jaw']})")
    print(f"  path length {info['path_len_mm']:.1f} mm, camera height z = {info['z_cam']:.1f} mm")
    print(f"  {info['frames']} frames ({info['duration_s']:.1f} s), {info['gt_points']} GT line samples")
    print(f"  check the path: {Path(info['out_dir']) / 'gt' / 'trajectory.png'}")


if __name__ == "__main__":
    main()
