"""Create the printable marker sheet (A4 PDF) used as the model plate and as the calibration target.

    python scripts/make_marker_sheet.py            -> docs/marker_sheet_A4.pdf

Print at 100 % ("Actual size", NOT "Fit to page"), then measure the 100 mm bar with a ruler.
The layout comes from ioscan.pipeline.markers.MarkerBoard, which is what the software expects.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from ioscan.pipeline.markers import MarkerBoard, marker_bits

A4_W, A4_H = 210.0, 297.0


def make_sheet(out: Path, board: MarkerBoard) -> Path:
    gw = (board.cols - 1) * board.pitch_mm + board.marker_mm
    gh = (board.rows - 1) * board.pitch_mm + board.marker_mm
    gx, gy = (A4_W - gw) / 2.0, 62.0
    fig = plt.figure(figsize=(A4_W / 25.4, A4_H / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, A4_W)
    ax.set_ylim(A4_H, 0)
    ax.axis("off")
    black, white = [], []
    for i in range(board.count):
        x0, y0 = board.marker_origin_mm(i)
        x0, y0 = x0 + gx, y0 + gy
        bits = marker_bits(board, i)
        n = bits.shape[0]
        cell = board.marker_mm / n
        black.append(Rectangle((x0, y0), board.marker_mm, board.marker_mm))
        for r in range(n):
            for c in range(n):
                if bits[r, c]:
                    # slight overlap between neighbouring white cells avoids hairlines in some PDF viewers
                    white.append(Rectangle((x0 + c * cell - 0.01, y0 + r * cell - 0.01), cell + 0.02, cell + 0.02))
    ax.add_collection(PatchCollection(black, facecolor="black", edgecolor="none"))
    ax.add_collection(PatchCollection(white, facecolor="white", edgecolor="none"))

    m = 8.0                                                    # white margin kept around the markers
    ax.add_patch(Rectangle((gx - m, gy - m), gw + 2 * m, gh + 2 * m, fill=False, edgecolor="0.55", linewidth=0.5,
                           linestyle=(0, (4, 3))))
    ax.text(gx - m, gy - m - 2.0, "cut along the dashed line", fontsize=7, color="0.4", va="bottom")

    ax.text(A4_W / 2, 18, "IOSCAN marker sheet", fontsize=20, ha="center", va="center", weight="bold")
    ax.text(A4_W / 2, 28, 'Print at 100 % ("Actual size") on matt paper. Do not use "Fit to page".',
            fontsize=10.5, ha="center", va="center")
    ax.text(A4_W / 2, 35, "Print two copies: one stays bare for calibration, one carries the model.",
            fontsize=10.5, ha="center", va="center")
    ax.text(A4_W / 2, 42, "Glue each copy flat on a stiff card. Keep it clean, flat and free of tape or glare.",
            fontsize=10.5, ha="center", va="center")

    by = gy + gh + m + 22.0
    bx = (A4_W - 100.0) / 2.0
    ax.plot([bx, bx + 100.0], [by, by], color="black", linewidth=0.6)
    for k in range(11):
        h = 4.0 if k % 5 == 0 else 2.2
        ax.plot([bx + 10.0 * k] * 2, [by - h, by + h], color="black", linewidth=0.6)
    ax.text(bx, by + 8, "0", fontsize=8, ha="center")
    ax.text(bx + 50, by + 8, "50", fontsize=8, ha="center")
    ax.text(bx + 100, by + 8, "100 mm", fontsize=8, ha="center")
    ax.text(A4_W / 2, by + 17, "Check with a ruler: this bar must be 100 mm long (end tick to end tick).",
            fontsize=10.5, ha="center", va="center")
    ax.text(A4_W / 2, by + 24, "If it is not, enter the measured length in the dashboard when you calibrate.",
            fontsize=10.5, ha="center", va="center")

    ax.text(A4_W / 2, A4_H - 12,
            f"{board.cols} x {board.rows} markers  |  {board.dictionary}  |  marker {board.marker_mm:g} mm  |  "
            f"pitch {board.pitch_mm:g} mm  |  grid {gw:g} x {gh:g} mm",
            fontsize=7.5, ha="center", color="0.35")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "docs" / "marker_sheet_A4.pdf")
    args = ap.parse_args()
    print(make_sheet(args.out, MarkerBoard()))
