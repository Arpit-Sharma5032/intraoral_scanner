"""Plain-language verdict for a finished scan: is the result usable, and if not, what to change.

The pipeline always produces *some* surface. A demo must not present a bad one as a result, so the
measurable symptoms of each known failure are turned into one of three states:

* ``good``   - nothing suspicious was measured
* ``check``  - usable, but with a stated weakness
* ``rescan`` - the surface should not be trusted; the first reason says what to change
"""
from __future__ import annotations

import numpy as np


def scan_verdict(report: dict, vertices: np.ndarray | None = None) -> dict:
    bad: list[str] = []
    warn: list[str] = []
    tr = report.get("translation") or {}
    used = tr.get("used")
    q = report.get("quality") or {}
    pts = float(q.get("mean_clean_points_per_kept_profile") or 0.0)
    kept = int(q.get("kept_profiles") or 0)
    hm = report.get("heightmap") or {}
    area = float(hm.get("area_mm2") or 0.0)
    V = np.asarray(vertices, dtype=np.float64).reshape(-1, 3) if vertices is not None else np.zeros((0, 3))

    # --- surface -----------------------------------------------------------------------------
    relief = 0.0
    if len(V) >= 50:
        relief = float(np.percentile(V[:, 2], 98) - np.percentile(V[:, 2], 2))
    plate_off = abs(float((report.get("plate_check") or {}).get("offset_mm") or 0.0))
    uncalibrated = used == "markers" and (plate_off > 0.8 or (report.get("calibration_is_nominal") and plate_off > 0.4))
    if uncalibrated:
        bad.append(f"The scanner is not calibrated: the laser places the marker sheet {plate_off:.1f} mm away from "
                   "where the markers say it is, so heights and shapes are wrong. Run Calibrate with the bare "
                   "sheet, then press Reprocess on this scan (no need to scan again).")
    elif int(report.get("mesh_triangles") or 0) == 0 or area < 15.0:
        bad.append("Almost no surface was captured. The laser line must cross the teeth for the whole pass: "
                   "use the Ready check before recording.")
    elif relief < 1.2:
        bad.append(f"The surface is flat ({relief:.1f} mm of relief): the laser line ran over the base or the "
                   "table, not over the teeth. Reposition the model so the line crosses the tooth row.")
    if kept and kept < 60:
        warn.append(f"Only {kept} usable laser profiles; scan for longer or more slowly.")

    # --- laser line ---------------------------------------------------------------------------
    if 0 < pts < 20:
        bad.append(f"The laser line is too weak on the model ({pts:.0f} points per profile). Dim the room light "
                   "or shade the box, and check the laser focus and power.")
    elif 0 < pts < 60:
        warn.append(f"The laser line is weak ({pts:.0f} points per profile); expect holes. Less ambient light "
                    "or a matte spray on the teeth gives a stronger line.")

    # --- motion -------------------------------------------------------------------------------
    if used == "markers":
        mk = tr.get("markers") or {}
        vf = float(mk.get("valid_fraction") or 0.0)
        if vf < 0.6:
            bad.append(f"The marker plate was visible in only {100 * vf:.0f}% of the frames. Keep at least two "
                       "markers in the camera view for the whole scan.")
        elif vf < 0.9:
            warn.append(f"Markers were lost in {100 * (1 - vf):.0f}% of the frames; those parts are missing.")
        rms = mk.get("median_rms_px")
        pc = report.get("plate_check") or {}
        spread = float(pc.get("spread_mm") or 0.0)
        few = float(mk.get("median_markers") or 99) < 4
        if spread > 0.45:
            bad.append(f"The plate position was unsteady (the laser sees the sheet wobble by {spread:.1f} mm). "
                       + ("Only a few markers were in view: leave more of the printed sheet visible and slide the "
                          "plate flat on the floor of the box." if few else
                          "Check that the sheet is flat on a stiff card, in focus, and the scanner is calibrated."))
        elif few and str(mk.get("plate_plane", "")).startswith("free"):
            (bad if spread > 0.2 else warn).append(
                f"Only {mk.get('median_markers', 0):.0f} markers were in view per frame while the plate was "
                "tilted or lifted, so heights are uncertain. Slide the plate flat on the floor of the box, "
                "or leave more of the sheet visible.")
        elif spread > 0.2 or (rms is not None and rms > 1.2):
            warn.append("Marker tracking was looser than usual. Check that the sheet is flat, in focus and evenly lit.")
        if not uncalibrated and abs(float(pc.get("offset_mm") or 0.0)) > 0.4:
            warn.append(f"Calibration is off: the laser places the sheet {pc['offset_mm']:+.2f} mm from its true "
                        "plane. Run Calibrate with the bare sheet.")
    elif used == "planar":
        pl = tr.get("planar") or {}
        mm = pl.get("motion_mask") or {}
        if float(pl.get("valid_fraction") or 0.0) < 0.8 or float(pl.get("max_turn_deg_per_s") or 0.0) > 45.0:
            bad.append("Image tracking of the model lost or mis-followed it. Use the marker plate: it does not "
                       "depend on how the model looks.")
        else:
            warn.append("Motion was estimated from the model's appearance, which cannot see tilt and can be wrong "
                        "without notice. Use the marker plate for a dependable result.")
        if float(mm.get("mask_fraction") or 0.1) > 0.4 or float(mm.get("mask_fraction") or 0.1) < 0.04:
            warn.append("The tracker had too little or too much moving image to follow.")
    elif used == "flow":
        warn.append("Motion was estimated from image shift inside the laser window (straight passes only). "
                    "Use the marker plate for a dependable result.")
    elif used in ("speed", "length"):
        warn.append("Motion was assumed, not measured, so proportions along the pass are approximate. "
                    "Use the marker plate.")

    # --- calibration and capture ----------------------------------------------------------------
    if report.get("calibration_is_nominal") and not uncalibrated:
        warn.append("Camera and laser geometry are nominal (not calibrated): sizes are approximate.")
    lf = report.get("laser_flags") or {}
    if lf.get("action") == "unverified" or lf.get("reason"):
        warn.append("Laser on/off frames could not be told apart reliably (line too faint).")

    status = "rescan" if bad else ("check" if warn else "good")
    title = {"good": "Good scan", "check": "Usable, with limits", "rescan": "Rescan needed"}[status]
    return {"status": status, "title": title, "reasons": bad + warn,
            "metrics": {"points_per_profile": round(pts, 1), "area_mm2": round(area, 1),
                        "relief_mm": round(relief, 2), "motion_source": used}}
