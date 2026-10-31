"""Sanity-check a simulated session: numeric stats plus a preview image.

    python scripts/check_session.py data/sessions/FUYJPUPX_lower

Writes <session>/gt/check.png with:
  1. top view: reference mesh (grey) and the true laser-line points seen so far (red)
  2. side view (x-z): the red points must sit on the UPPER (occlusal) surface of the mesh
  3. the ON-minus-OFF image of the frame with the longest line
"""

# Prefer this checkout's src/ package over any stale ioscan installation in site-packages.
import sys as _sys
from pathlib import Path as _Path
_project_src = _Path(__file__).resolve().parents[1] / "src"
if _project_src.is_dir():
    _sys.path.insert(0, str(_project_src))
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh

from ioscan.session.format import SessionReader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    args = ap.parse_args()
    root = Path(args.session)

    r = SessionReader(root)
    prof = r.gt_profiles()
    _, T = r.gt_poses()
    counts = np.bincount(prof["frame"], minlength=len(r))
    on = np.flatnonzero(r.laser_on)

    print(f"{len(r)} frames, {len(on)} laser-ON")
    print(f"ON frames with a visible line: {np.mean(counts[on] > 0):.0%}")
    z = prof["P_c"][:, 2]
    p1, p50, p99 = np.percentile(z, [1, 50, 99])
    print(f"depth z (mm), 1/50/99 percentile: {p1:.1f} / {p50:.1f} / {p99:.1f}")
    print("  expect: 1% >= ~14, 99% <= ~34, median a bit above the working distance (20 mm)")

    # true line points in world coordinates
    Rw, tw = T[prof["frame"], :3, :3], T[prof["frame"], :3, 3]
    P_w = np.einsum("nij,nj->ni", Rw, prof["P_c"]) + tw
    mesh = trimesh.load(str(root / "gt" / "reference_mesh.ply"), process=False)
    V = np.asarray(mesh.vertices)[::8]
    Pw = P_w[:: max(1, len(P_w) // 40000)]

    # which side of the mesh is being scanned? fraction of points in the top 40% of the local height
    print(f"scanned points: z in [{Pw[:, 2].min():.1f}, {Pw[:, 2].max():.1f}] mm; "
          f"mesh vertices: z in [{V[:, 2].min():.1f}, {V[:, 2].max():.1f}] mm")

    best = int(np.argmax(counts))
    pair = next((p for p in r.on_off_pairs() if p[0] == best), None)
    img = None
    if pair:
        img = r.read_frame(pair[0]).astype(np.float32) - r.read_frame(pair[1]).astype(np.float32)

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    ax[0].scatter(V[:, 0], V[:, 1], s=1, c="lightgray")
    ax[0].scatter(Pw[:, 0], Pw[:, 1], s=1, c="r")
    ax[0].set_aspect("equal")
    ax[0].set_title("top view (x-y)")
    ax[1].scatter(V[:, 0], V[:, 2], s=1, c="lightgray")
    ax[1].scatter(Pw[:, 0], Pw[:, 2], s=1, c="r")
    ax[1].set_aspect("equal")
    ax[1].set_title("side view (x-z): red should hug the TOP of the grey")
    if img is not None:
        ax[2].imshow(np.clip(img, 0, None), cmap="gray")
        ax[2].set_title(f"ON - OFF, frame {best}")
    fig.tight_layout()
    out = root / "gt" / "check.png"
    fig.savefig(out, dpi=110)
    print(f"preview written to {out}")


if __name__ == "__main__":
    main()