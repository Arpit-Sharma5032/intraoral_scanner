"""Explain why a saved session yields few / smeared laser points.

    python scripts/diagnose_session.py data/sessions/scan_YYYYMMDD_HHMMSS [--z-min 75 --z-max 125]

Reports (all from the saved frames, nothing is re-captured):
  * where the laser line really is in the image (rows) and what depth the stored laser plane gives it,
  * how many ridge points each gate removes (prominence, secondary-peak, ROI, depth window),
  * how much the IMU orientation rotates during the pass and what path the assumed speed implies,
  * whether the laser-OFF frames carry enough texture to measure translation (sweep mode, optical flow),
  * which time window has a continuous laser line (what sweep auto-trim will keep).
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_src = _Path(__file__).resolve().parents[1] / "src"
if _src.is_dir():
    _sys.path.insert(0, str(_src))

import argparse
import numpy as np
from scipy.spatial.transform import Rotation

from ioscan.pipeline.ridge import RidgeConfig, extract_ridge_subpixel
from ioscan.pipeline.sweep import OffFrameFlow, SweepConfig
from ioscan.pipeline.subtraction import subtract_ambient
from ioscan.pipeline.triangulate import roi_depth_coverage
from ioscan.session.loader import load_calibration, load_frame_records, read_grayscale_frame
from ioscan.utils.roi import enabled_roi_points, load_roi_config, rasterize_roi


def _pct(a, q):
    return np.percentile(a, q) if len(a) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=_Path)
    ap.add_argument("--z-min", type=float, default=75.0)
    ap.add_argument("--z-max", type=float, default=125.0)
    ap.add_argument("--speed-mm-s", type=float, default=5.0)
    ap.add_argument("--samples", type=int, default=60, help="laser-ON frames to inspect")
    args = ap.parse_args()

    camera, laser, nominal = load_calibration(args.session)
    records = load_frame_records(args.session)
    roi = enabled_roi_points(load_roi_config(args.session / "roi.json"))
    z_range = (args.z_min, args.z_max)
    print(f"Session: {args.session}\nCalibration: {'NOMINAL (unmeasured)' if nominal else 'measured'};  "
          f"ROI: {'enabled' if roi else 'disabled'};  depth gate: {z_range[0]}..{z_range[1]} mm\n")

    # ---- 1. ROI vs depth window ------------------------------------------------------------
    cov = roi_depth_coverage(camera, laser, roi, z_range)
    print("[1] ROI vs depth window (stored laser plane)")
    if cov["depth_p02_mm"] is not None:
        print(f"    ROI pixels map to Z = {cov['depth_p02_mm']:.1f} .. {cov['depth_p98_mm']:.1f} mm")
    print(f"    pixels inside the depth window: {100 * cov['fraction_in_range']:.2f}%   "
          f"columns that can yield a point: {cov['columns_reachable']} / {cov['columns_total']}")
    if cov["fraction_in_range"] < 0.25:
        print(f"    >>> PROBLEM: the window discards almost the whole ROI. Suggested window: {cov['suggested_z_range_mm']}")

    # ---- 2. ridge statistics on real frames ------------------------------------------------
    on_idx = [i for i, r in enumerate(records) if r["laser_on"]]
    pick = np.linspace(0, len(on_idx) - 1, min(args.samples, len(on_idx))).astype(int)
    mask = rasterize_roi((camera.height, camera.width), roi) > 0 if roi else None
    rows_all, z_all, peak_all = [], [], []
    n_cols_roi = int(mask.any(0).sum()) if mask is not None else camera.width
    stats = dict(ridge=[], ridge_nosec=[], gated=[], sat=[])
    for k in pick:
        i = on_idx[k]
        off = next((records[j] for j in range(i - 1, max(i - 4, -1), -1) if not records[j]["laser_on"]), None)
        on = read_grayscale_frame(records[i]["path"])
        offf = read_grayscale_frame(off["path"]) if off is not None else None
        diff = subtract_ambient(on, offf)
        if mask is not None:
            diff[~mask] = 0.0
        sat = 254.0 if on.dtype == np.uint8 else 65000.0
        cfg = RidgeConfig(saturation_limit=sat, max_secondary_ratio=0.6, min_prominence=25.0)
        u, v, w = extract_ridge_subpixel(diff, cfg)
        u2, v2, _ = extract_ridge_subpixel(diff, RidgeConfig(saturation_limit=sat, max_secondary_ratio=1.0))
        stats["ridge"].append(len(u)); stats["ridge_nosec"].append(len(u2))
        stats["sat"].append(float((diff.max(axis=0) >= sat).sum()))
        if len(u):
            pts, valid = laser.intersect_rays(camera.unproject(np.column_stack([u, v])))
            z = pts[:, 2]
            rows_all.append(v); z_all.append(z); peak_all.append(w)
            stats["gated"].append(int(((z >= z_range[0]) & (z <= z_range[1]) & valid).sum()))
        else:
            stats["gated"].append(0)
    print(f"\n[2] Ridge extraction on {len(pick)} sampled laser-ON frames (ROI columns: {n_cols_roi})")
    m = lambda k: float(np.mean(stats[k])) if stats[k] else float("nan")
    print(f"    ridge points/profile with the pipeline gates (ratio 0.6)  : {m('ridge'):.1f}")
    print(f"    ... if the secondary-peak rejection were off               : {m('ridge_nosec'):.1f}")
    print(f"    ... of those, inside the depth window                      : {m('gated'):.1f}")
    if rows_all:
        rows, zs, pk = np.concatenate(rows_all), np.concatenate(z_all), np.concatenate(peak_all)
        print(f"    laser line image rows  p5/p50/p95 = {_pct(rows,5):.0f} / {_pct(rows,50):.0f} / {_pct(rows,95):.0f}")
        print(f"    depth on stored plane  p5/p50/p95 = {_pct(zs,5):.1f} / {_pct(zs,50):.1f} / {_pct(zs,95):.1f} mm")
        print(f"    peak intensity         p5/p50/p95 = {_pct(pk,5):.0f} / {_pct(pk,50):.0f} / {_pct(pk,95):.0f} DN; "
              f"saturated columns/frame: {m('sat'):.1f}")
        out = ((zs < z_range[0]) | (zs > z_range[1])).mean()
        if out > 0.5:
            print(f"    >>> PROBLEM: {100*out:.0f}% of detected line points lie outside the depth window "
                  f"(real line depth {_pct(zs,2):.1f}..{_pct(zs,98):.1f} mm under the stored plane).")
    else:
        print("    >>> PROBLEM: no ridge found at all (exposure, filter, ROI or laser-ON flag pairing).")
    if m("ridge") and m("ridge_nosec") > 1.5 * m("ridge"):
        print("    >>> The secondary-peak gate is rejecting >33% of columns: broad/halo line or reflections.")

    # ---- 3. pose / translation -------------------------------------------------------------
    print("\n[3] Orientation and assumed translation")
    t = np.array([r["timestamp_s"] for r in records])
    qs = [r["quat_wxyz"] for r in records]
    dur = t[-1] - t[0]
    on_t = t[on_idx]
    print(f"    duration {dur:.1f} s;  assumed {args.speed_mm_s:g} mm/s  ->  path {args.speed_mm_s * dur:.0f} mm")
    if all(q is not None for q in qs):
        R = Rotation.from_quat(np.array([[q[1], q[2], q[3], q[0]] for q in qs]))
        rel = (R[0].inv() * R).magnitude() * 180 / np.pi
        step = (R[:-1].inv() * R[1:]).magnitude() * 180 / np.pi / np.maximum(np.diff(t), 1e-6)
        print(f"    IMU rotation away from the start pose: max {rel.max():.1f} deg; "
              f"angular speed p50/p95 = {np.percentile(step,50):.1f} / {np.percentile(step,95):.1f} deg/s")
        lever = float(np.median(np.concatenate(z_all))) if z_all else 0.5 * (z_range[0] + z_range[1])
        print(f"    a 1 deg orientation error moves a profile {np.tan(np.deg2rad(1.0)) * lever:.2f} mm at {lever:.0f} mm range")
    print("    A dental pass is ~50-100 mm long. If that is the real travel, the true average speed was "
          f"~{50/dur:.2f}-{100/dur:.2f} mm/s, not {args.speed_mm_s:g}; rerun with --scan-length-mm <real travel>.")

    # ---- 4. on-surface window ----------------------------------------------------------------
    print("\n[4] Where the laser line is on the surface (sampled)")
    if stats["gated"]:
        t_s = t[np.array([on_idx[k] for k in pick])] - t[0]
        good = np.array(stats["ridge"]) >= 15
        if good.any():
            print(f"    profiles with >=15 points: {100 * good.mean():.0f}% of samples; first {t_s[good][0]:.1f} s, "
                  f"last {t_s[good][-1]:.1f} s of {dur:.1f} s")
            print("    timeline (# = line found, . = none): " + "".join("#" if g else "." for g in good))

    # ---- 5. optical-flow translation ---------------------------------------------------------
    print("\n[5] Translation measurable from laser-OFF frames (sweep mode, optical flow)")
    off_idx = [i for i, r in enumerate(records) if not r["laser_on"]]
    if len(off_idx) < 3:
        print("    no OFF frames recorded")
    else:
        # a contiguous run of OFF frames from the middle of the capture
        mid = len(off_idx) // 2
        run = off_idx[max(0, mid - 60): mid + 60]
        flow = OffFrameFlow((camera.height, camera.width), mask, SweepConfig())
        crops = []
        for i in run:
            img = read_grayscale_frame(records[i]["path"])
            flow.add(img, records[i]["timestamp_s"])
            crops.append(float(img[mask].std()) if mask is not None else float(img.std()))
        ok = np.array([smp.valid for smp in flow.samples]) if flow.samples else np.zeros(0, bool)
        resp = np.array([smp.response for smp in flow.samples]) if flow.samples else np.zeros(0)
        lever = float(np.median(np.concatenate(z_all))) if z_all else 80.0
        steps = np.array([np.hypot(smp.du, smp.dv) for smp in flow.samples]) if flow.samples else np.zeros(0)
        print(f"    OFF-frame texture inside ROI (std): p50 {np.median(crops):.1f} DN")
        print(f"    reliable frame pairs: {100 * ok.mean():.0f}%  (median match score {np.median(resp):.2f}); "
              f"median image motion {np.median(steps):.2f} px/pair = {np.median(steps) * lever / camera.fx * 25:.2f} mm/s")
        if ok.mean() >= 0.6:
            print("    OK: sweep mode will measure translation from these frames (translation 'auto').")
        else:
            print("    >>> Too little texture/ambient light through the ROI for optical flow. Add diffuse ambient light "
                  "(not the laser), or set the real pass length (translation 'length').")


if __name__ == "__main__":
    main()
