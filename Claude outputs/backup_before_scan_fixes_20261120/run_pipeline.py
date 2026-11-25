from __future__ import annotations

# Prefer this checkout's src/ package over any stale ioscan installation in site-packages.
import sys as _sys
from pathlib import Path as _Path
_project_src = _Path(__file__).resolve().parents[1] / "src"
if _project_src.is_dir():
    _sys.path.insert(0, str(_project_src))

import argparse
import os
import threading
import json
from pathlib import Path
import warnings
import os
import shutil

import cv2
import numpy as np
try:
    from tqdm import tqdm
except ImportError:  # progress bars are optional
    def tqdm(iterable, **_kwargs):
        return iterable

from ioscan.calibration.models import CameraIntrinsics, LaserPlane
from ioscan.pipeline.pose import PoseConfig, PoseTracker
from ioscan.pipeline.ridge import RidgeConfig
from ioscan.pipeline.quality import assess_scan_geometry
from ioscan.pipeline.triangulate import (
    triangulate_frame, laser_plane_depth_bounds, resolve_depth_window, roi_depth_coverage,
)
from ioscan.pipeline.sweep import (
    FlowSample, OffFrameFlow, PlanarTracker, Profile, SweepConfig, build_motion_mask, flow_to_translation,
    reconstruct_marker_plate, reconstruct_sweep,
    write_ply_mesh, write_ply_points, write_stl,
)
from ioscan.pipeline.markers import MarkerBoard, MarkerTracker, probe_orientation, smooth_poses, tracking_summary
from ioscan.pipeline.verdict import scan_verdict
from ioscan.utils.roi import rasterize_roi
from ioscan.session.laser_phase import verify_laser_flags
from ioscan.session.loader import (
    load_calibration,
    load_frame_records,
    load_gt_poses,
    session_source,
    read_grayscale_frame,
)
from ioscan.utils.roi import enabled_roi_points, load_roi_config




def _clear_stale_scan_outputs(out_dir: Path) -> None:
    """Remove artifacts from an earlier run before writing a new scan result.

    Without this, a failed/partial reconstruction can leave old previews or
    tooth-labelled meshes next to the new report and appear to be current.
    """
    stale_files = (
        "reconstruction_preview.ply", "reconstruction_preview.tmp.ply",
        "estimated_poses.npz", "pipeline_report.json",
        "segmented_full_colored.ply", "segmented_clean_teeth.ply",
        "segmented_clean_teeth.stl", "segmented_cloud.ply",
        "segmentation_labels.npz",
    )
    for name in stale_files:
        path = out_dir / name
        if path.is_file():
            path.unlink()
    per_tooth = out_dir / "segmented_teeth_by_class"
    if per_tooth.is_dir():
        shutil.rmtree(per_tooth)


def _load_calibration(session_dir: Path) -> tuple[CameraIntrinsics, LaserPlane]:
    """Camera + laser plane stored with the session (raises if none; no silent nominal values)."""
    cam, laser, _ = load_calibration(session_dir)
    return cam, laser


def _load_frame_records(session_dir: Path) -> list[dict]:
    return load_frame_records(session_dir)


def _io_workers() -> int:
    return max(2, min(8, (os.cpu_count() or 2)))


def _prefetch_frames(records: list[dict], lookahead: int = 32):
    """Yield (record, decoded frame) in order while worker threads decode ahead (cv2 releases the GIL)."""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=_io_workers(), thread_name_prefix="ioscan-decode") as pool:
        pending: deque = deque()
        it = iter(records)
        for rec in it:
            pending.append((rec, pool.submit(read_grayscale_frame, rec["path"])))
            if len(pending) >= lookahead:
                break
        while pending:
            rec, fut = pending.popleft()
            nxt = next(it, None)
            if nxt is not None:
                pending.append((nxt, pool.submit(read_grayscale_frame, nxt["path"])))
            yield rec, fut.result()


class _TrackerThread:
    """Runs OFF-frame motion tracking in order on its own thread, alongside laser extraction."""

    def __init__(self, trackers: list):
        import queue as _q
        self.trackers = [t for t in trackers if t is not None]
        self.q: "_q.Queue" = _q.Queue(maxsize=64)
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, name="ioscan-tracker", daemon=True)
        if self.trackers:
            self.thread.start()

    def _run(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                try:
                    for tr in self.trackers:
                        if hasattr(tr, "flush"):
                            tr.flush()
                except BaseException as exc:
                    self.error = self.error or exc
                return
            if self.error is not None:
                continue
            frame, t = item
            try:
                for tr in self.trackers:
                    tr.add(frame, t)
            except BaseException as exc:  # surfaced after the loop
                self.error = exc

    def add(self, frame, t: float) -> None:
        if self.trackers:
            self.q.put((frame, t))

    def finish(self) -> None:
        if self.trackers:
            self.q.put(None)
            self.thread.join()
            if self.error is not None:
                raise self.error


def _imu_rotation_span_deg(records: list[dict], t0: float, t1: float) -> float | None:
    """Largest IMU rotation away from the orientation at t0 within [t0, t1] (degrees)."""
    qs = [r["quat_wxyz"] for r in records if r["quat_wxyz"] is not None and t0 <= r["timestamp_s"] <= t1]
    if len(qs) < 2:
        return None
    q = np.asarray(qs, dtype=np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    dots = np.clip(np.abs(q @ q[0]), 0.0, 1.0)
    return float(np.degrees(2.0 * np.arccos(dots)).max())


def _run_sweep(*, session_dir, out_dir, records, camera, laser, nominal_cal, source, session_meta,
               roi_config, roi_polygon, z_range, depth_window_info, run_warnings, ridge_method,
               speed_mm_s, scan_length_mm, translation, auto_trim, ridge_secondary_ratio,
               ridge_min_prominence, sweep_cfg, segment, seg_cfg, progress_callback,
               measured_capture_fps, requested_camera_fps, frame_intervals, on_records,
               laser_profile_rate_hz, actual_camera_fourcc, laser_flag_info=None,
               marker_board=None, adaptive_line=True) -> dict:
    """Linear-sweep reconstruction: measured (or known) translation, no IMU rotation, height-map mesh."""
    cfg = sweep_cfg or SweepConfig()
    cfg.translation = translation
    cfg.speed_mm_s = float(speed_mm_s)
    cfg.scan_length_mm = float(scan_length_mm) if scan_length_mm else None
    cfg.auto_trim = bool(auto_trim)

    ridge_kw = dict(method=ridge_method, max_secondary_ratio=float(ridge_secondary_ratio),
                    min_prominence=float(ridge_min_prominence), adaptive_floor=bool(adaptive_line))
    ridge_cfg = RidgeConfig(**ridge_kw)
    ridge_cfg_8bit = RidgeConfig(saturation_limit=254.0, **ridge_kw)
    roi_mask = rasterize_roi((camera.height, camera.width), roi_polygon) > 0 if roi_polygon is not None else None
    planar = None
    planar_mask_info: dict = {}
    off_all = [r for r in records if not r["laser_on"]]
    # Marker plate first: printed markers under the model give the full plate pose in every OFF
    # frame (no drift, tilt included), so nothing has to be inferred from the plain white model.
    marker = None
    marker_probe: dict = {}
    board = marker_board or MarkerBoard.load()
    if translation in ("auto", "markers") and off_all:
        if progress_callback is not None:
            progress_callback({"stage": "reconstructing", "progress": 0.5,
                               "message": "Looking for the marker plate"})
        probe_recs = off_all[::max(1, len(off_all) // 24)][:30]
        marker_probe = probe_orientation([im for _r, im in _prefetch_frames(probe_recs)], board)
        if translation == "markers" and marker_probe["frames_with_2plus"] == 0:
            raise RuntimeError(
                "Marker tracking was requested but no markers of the printed sheet were found in the camera "
                "frames. Check that the sheet is under the model, in view of the camera, lit and in focus.")
        if translation == "markers" or marker_probe["fraction_with_2plus"] >= 0.5:
            marker = MarkerTracker(board, camera.K, camera.dist_coeffs, mirrored=marker_probe["mirrored"])
    if marker is None and translation in ("auto", "planar"):
        # The camera sees the whole model through the window in the OFF frames, so its full in-plane
        # motion (rotation + translation) can be tracked, not just a shift inside the laser ROI.
        if progress_callback is not None:
            progress_callback({"stage": "reconstructing", "progress": 1.0,
                               "message": "Finding the moving model in the ambient frames"})
        off_recs = [r for r in records if not r["laser_on"]]
        step = max(1, len(off_recs) // 120)
        ds = int(cfg.planar_downsample)
        small = [cv2.resize(im, (im.shape[1] // ds, im.shape[0] // ds), interpolation=cv2.INTER_AREA)
                 for _r, im in _prefetch_frames(off_recs[::step])]
        pmask, planar_mask_info = build_motion_mask(small)
        if pmask is not None:
            planar = PlanarTracker(pmask, cfg)
    # The ROI-shift estimator is only a fallback now: running it beside full motion tracking doubled the work.
    flow = (OffFrameFlow((camera.height, camera.width), roi_mask, cfg)
            if translation == "flow" or (translation == "auto" and planar is None and marker is None) else None)

    profiles: list[Profile] = []
    last_off = None
    unpaired_on = 0
    total_points = 0
    usable = 0
    preview_path = out_dir / "reconstruction_preview.ply"
    preview_every = max(5, len(records) // 24)
    t_first = float(records[0]["timestamp_s"])
    tracker = _TrackerThread([flow, planar, marker])
    # Ambient reference of a laser frame = mean of the OFF frame before and the OFF frame after it.
    # For a model that moves steadily this predicts the ambient image *at* the laser frame, so moving
    # edges (tooth outlines, printed markers) cancel instead of showing up as false laser peaks.
    pair_gap_s = 1.7 * (float(np.median(frame_intervals)) if len(frame_intervals) else 0.02)
    state = {"unpaired": 0, "usable": 0, "points": 0}

    def _triangulate_pending(item, off_after, t_after) -> None:
        rec_on, frame_on, off_before, t_before = item
        t_on = float(rec_on["timestamp_s"])
        refs = [(o, abs(t_on - t)) for o, t in ((off_before, t_before), (off_after, t_after)) if o is not None]
        near = [o for o, gap in refs if gap <= pair_gap_s and o.shape == frame_on.shape and o.dtype == frame_on.dtype]
        if len(near) == 2 and frame_on.dtype == np.uint8:
            ref = cv2.addWeighted(near[0], 0.5, near[1], 0.5, 0.0)
        elif near:
            ref = near[0]
        elif refs:
            ref = min(refs, key=lambda r: r[1])[0]
        else:
            ref = None
            state["unpaired"] += 1
        pts_cam, uv, _ = triangulate_frame(
            frame_on, ref, camera, laser,
            ridge_cfg=ridge_cfg_8bit if frame_on.dtype == np.uint8 else ridge_cfg,
            z_range=z_range, roi_polygon=roi_polygon,
        )
        u = uv[:, 0] if len(uv) else np.empty(0)
        profiles.append(Profile(int(rec_on["index"]), t_on, pts_cam, u))
        if len(pts_cam):
            state["usable"] += 1
            state["points"] += int(len(pts_cam))

    pending = None
    last_off_t = 0.0
    for rec_pos, (rec, frame) in enumerate(tqdm(_prefetch_frames(records), total=len(records), desc="Pipeline (sweep)")):
        t_rec = float(rec["timestamp_s"])
        if not rec["laser_on"]:
            if pending is not None:
                _triangulate_pending(pending, frame, t_rec)
                pending = None
            last_off, last_off_t = frame, t_rec
            tracker.add(frame, t_rec)
        else:
            if pending is not None:                       # two laser frames in a row (dropped OFF frame)
                _triangulate_pending(pending, None, 0.0)
            pending = (rec, frame, last_off, last_off_t)
        if rec_pos + 1 == len(records) and pending is not None:
            _triangulate_pending(pending, None, 0.0)
            pending = None
        usable, total_points = state["usable"], state["points"]
        if progress_callback is not None:
            pct = 80.0 * (rec_pos + 1) / max(len(records), 1)
            preview_written = False
            if usable and ((rec_pos + 1) % preview_every == 0 or rec_pos + 1 == len(records)):
                preview_written = _write_sweep_preview(preview_path, profiles, flow, cfg, camera, t_first)
            progress_callback({
                "stage": "reconstructing", "progress": round(pct, 2),
                "frames_done": rec_pos + 1, "frames_total": len(records),
                "laser_on_frames": usable, "triangulated_points": total_points,
                "preview_path": str(preview_path) if preview_written else None,
            })
    unpaired_on = state["unpaired"]
    usable, total_points = state["usable"], state["points"]
    if progress_callback is not None:
        progress_callback({"stage": "reconstructing", "progress": 82.0, "message": "Finishing motion tracking"})
    tracker.finish()
    marker_sm = None
    marker_info: dict = {}
    if marker is not None:
        marker_sm = smooth_poses(marker.samples)
        marker_info = dict(tracking_summary(marker.samples, marker_sm), probe=marker_probe,
                           mirrored=bool(marker_probe.get("mirrored")), board=board.to_dict())
        if translation == "auto" and marker_info["valid_fraction"] < cfg.marker_min_valid_fraction:
            # Markers were seen at the start but lost for most of the scan: use image tracking instead.
            if progress_callback is not None:
                progress_callback({"stage": "reconstructing", "progress": 83.0,
                                   "message": "Markers lost for most of the scan; tracking the model image instead"})
            marker_sm = None
            ds = int(cfg.planar_downsample)
            small = [cv2.resize(im, (im.shape[1] // ds, im.shape[0] // ds), interpolation=cv2.INTER_AREA)
                     for _r, im in _prefetch_frames(off_all[::max(1, len(off_all) // 120)])]
            pmask, planar_mask_info = build_motion_mask(small)
            if pmask is not None:
                planar = PlanarTracker(pmask, cfg)
                for rec, frame in _prefetch_frames(off_all):
                    planar.add(frame, float(rec["timestamp_s"]))
            else:
                flow = OffFrameFlow((camera.height, camera.width), roi_mask, cfg)
                for rec, frame in _prefetch_frames(off_all):
                    flow.add(frame, float(rec["timestamp_s"]))
    if (planar is not None and flow is None and translation == "auto" and planar.samples
            and np.mean([smp.valid for smp in planar.samples]) < cfg.planar_min_valid_fraction):
        # Full motion tracking lost the model: fall back to the ROI-shift estimator in a second pass
        # (only in this case, so the normal path does not pay for both).
        if progress_callback is not None:
            progress_callback({"stage": "reconstructing", "progress": 84.0,
                               "message": "Motion tracking unreliable; measuring shift inside the ROI instead"})
        flow = OffFrameFlow((camera.height, camera.width), roi_mask, cfg)
        for rec, frame in _prefetch_frames([r for r in records if not r["laser_on"]]):
            flow.add(frame, float(rec["timestamp_s"]))
    if usable == 0:
        raise RuntimeError(
            "No laser points were triangulated. Check exposure/laser power, the ROI and that the "
            "object lies inside the depth window."
        )
    if unpaired_on:
        run_warnings.append(f"{unpaired_on} ON frame(s) had no preceding OFF frame (no ambient subtraction).")
    if progress_callback is not None:
        progress_callback({"stage": "meshing", "progress": 86.0, "message": "Fusing profiles into a height map"})

    if marker_sm is not None:
        final_sm = {}

        def _refine(sheet):
            final_sm.clear()
            final_sm.update(smooth_poses(marker.samples, sheet=sheet))
            return final_sm

        result = reconstruct_marker_plate(profiles, cfg, marker_sm, mirrored=bool(marker_probe.get("mirrored")),
                                          refine_poses=_refine)
        if final_sm:
            marker_info.update(tracking_summary(marker.samples, final_sm))
        result.report["translation"]["markers"] = marker_info
    else:
        result = reconstruct_sweep(
            profiles, cfg, camera_fx=camera.fx, camera_fy=camera.fy,
            flow_samples=flow.samples if flow is not None else None,
            planar_samples=planar.samples if planar is not None else None,
            camera_cx=camera.cx, camera_cy=camera.cy,
        )
        if marker_info:
            result.report["translation"]["markers_rejected"] = marker_info
    if planar_mask_info:
        rep_tr = result.report.setdefault("translation", {})
        rep_tr.setdefault("planar", {}).update({"motion_mask": planar_mask_info})
    if planar is not None and planar.reference_frame is not None and result.report["translation"].get("used") == "planar":
        # Ambient image that defines this pass's model frame: lets multi-pass stitching align passes
        # from the whole visible arch instead of only the small laser overlaps.
        ref = planar.reference_frame
        half = cv2.resize(ref, (ref.shape[1] // 2, ref.shape[0] // 2), interpolation=cv2.INTER_AREA)
        m = cv2.resize(planar.mask * 255, (half.shape[1], half.shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out_dir / "ambient_reference.png"), half)
        cv2.imwrite(str(out_dir / "ambient_reference_mask.png"), m)
        kept_z = [float(np.median(p.pts[:, 2])) for p in profiles if len(p.pts)]
        (out_dir / "ambient_reference.json").write_text(json.dumps({
            "scale": 0.5, "fx": camera.fx, "fy": camera.fy, "cx": camera.cx, "cy": camera.cy,
            "tracking_depth_mm": float(np.median(kept_z)) if kept_z else None,
        }, indent=2), encoding="utf-8")
    rep = result.report
    tr = rep["translation"]
    win = rep["active_window"]

    cloud_path = out_dir / "fused_cloud.ply"
    mesh_ply = out_dir / "reconstructed_mesh.ply"
    mesh_stl = out_dir / "reconstructed_mesh.stl"
    write_ply_points(cloud_path, result.cloud)
    if len(result.faces):
        write_ply_mesh(mesh_ply, result.vertices, result.faces, result.normals)
        write_stl(mesh_stl, result.vertices, result.faces)
    else:
        for stale in (mesh_ply, mesh_stl):
            if stale.exists():
                stale.unlink()
        run_warnings.append("Mesh is empty (too few points); no mesh files were written.")
    paths = {"cloud_ply": cloud_path, "mesh_ply": mesh_ply, "mesh_stl": mesh_stl}
    if progress_callback is not None:
        progress_callback({"stage": "mesh_ready", "progress": 100.0, "message": "Surface reconstruction complete",
                           "mesh_path": str(mesh_ply)})

    # translation-only poses of the kept profiles (scan frame = camera frame of the first kept profile)
    kept_t = np.array([p.t for p in profiles])
    t0 = t_first + (win.get("t_start_s") or 0.0)
    t1 = t_first + (win.get("t_end_s") if win.get("t_end_s") is not None else kept_t[-1] - t_first)
    poses_npz_path = out_dir / "estimated_poses.npz"
    np.savez_compressed(poses_npz_path, frame_indices=np.array([p.index for p in profiles]),
                        timestamps_s=kept_t, translation_source=np.array(tr["used"]))

    rot_span = _imu_rotation_span_deg(records, t0, t1)
    path_mm = float(tr.get("path_mm", 0.0))
    V, F = result.vertices, result.faces
    extent = (V.max(axis=0) - V.min(axis=0)).tolist() if len(V) else [0.0, 0.0, 0.0]
    geometry = assess_scan_geometry(V, F, path_mm=path_mm)
    cleaned_mean = rep.get("mean_points_per_kept_profile", 0.0)

    if tr["used"] == "markers":
        mk = tr.get("markers", {})
        no_pose = win.get("profiles_without_pose", 0)
        run_warnings.append(
            f"Model motion measured from the marker plate: {100 * mk.get('valid_fraction', 0):.0f}% of frames, "
            f"{mk.get('median_markers', 0):.0f} markers per frame, {tr.get('rotation_range_deg', 0):.0f} deg turn, "
            f"tilt up to {tr.get('tilt_max_deg', 0):.0f} deg.")
        if mk.get("valid_fraction", 1.0) < 0.9:
            run_warnings.append(
                f"MARKERS LOST in {100 * (1 - mk['valid_fraction']):.0f}% of the frames: {no_pose} laser profiles "
                "recorded there were skipped (not guessed), which leaves gaps. Keep at least two markers in the "
                "camera view for the whole scan and keep fingers off the sheet.")
        pc = rep.get("plate_check", {})
        if abs(pc.get("offset_mm", 0.0)) > 0.4:
            run_warnings.append(
                f"CALIBRATION CHECK: laser points on the marker sheet lie {pc['offset_mm']:+.2f} mm off the sheet "
                "plane (should be 0). Heights are biased by about that much; run Calibrate with the bare sheet.")
        if mk.get("mirrored"):
            run_warnings.append("The camera sees the scene through a mirror; the model was un-mirrored automatically.")
    elif tr["used"] == "planar":
        pl = tr.get("planar", {})
        mm = pl.get("motion_mask", {})
        if mm.get("mask_fraction", 1.0) < 0.04:
            run_warnings.append(
                f"TRACKING RISK: the model is visible in only {100 * mm['mask_fraction']:.1f}% of the image "
                "(dark room / narrow lit area). Turn the room light on so the whole model is visible through the "
                "window (background ~40-80 DN on the brightness panel) and keep fingers out of the view.")
        elif mm.get("mask_fraction", 0.0) > 0.4:
            run_warnings.append(
                f"TRACKING RISK: {100 * mm['mask_fraction']:.0f}% of the image was treated as moving, so the tracker may "
                "have followed the static housing/background instead of the model (motion would then look near zero). "
                "Keep the background still and evenly lit, and avoid light flicker or exposure changes during the scan.")
        if pl.get("valid_fraction", 1.0) < 0.97:
            run_warnings.append(
                f"TRACKING RISK: {100 * (1 - pl['valid_fraction']):.0f}% of ambient frames could not be tracked; "
                "parts of the surface may be misplaced.")
        if pl.get("max_turn_deg_per_s", 0.0) > 45.0:
            run_warnings.append(
                f"TRACKING SLIPPED: the tracked model turned {pl['max_turn_deg_per_s']:.0f} deg within one second "
                f"around t={pl.get('max_turn_at_s')} s, faster than a hand-turned model. The surface after that point is "
                "probably misplaced; rescan with better lighting and a slower turn.")
        run_warnings.append(
            f"Model motion tracked from the ambient frames (rotation + translation): {pl.get('rotation_range_deg', 0):.0f} deg "
            f"turn, {path_mm:.1f} mm path, {100 * pl.get('valid_fraction', 0):.0f}% of frames tracked.")
    elif tr["used"] == "flow":
        run_warnings.append(
            f"Translation measured from laser-OFF frames (optical flow): {path_mm:.1f} mm path, "
            f"{100 * tr['flow'].get('valid_fraction', 0):.0f}% of frame pairs reliable."
        )
    elif tr["used"] == "length":
        run_warnings.append(
            f"Translation from the known scan length: {cfg.scan_length_mm:.1f} mm over the active "
            f"{win.get('duration_s', 0):.1f} s = {tr['speed_mm_s']:.2f} mm/s (assumes a steady pass)."
        )
    else:
        flow_note = ""
        if "flow" in tr:
            flow_note = (f" Optical flow was unusable ({100 * tr['flow'].get('valid_fraction', 0):.0f}% reliable "
                         "frame pairs: too little texture visible through the ROI).")
        run_warnings.append(
            f"Translation is ASSUMED: constant {cfg.speed_mm_s:.2f} mm/s along the scan axis.{flow_note} "
            "Set the real pass length (Scan length) for correct proportions."
        )
    if win.get("dropped_profiles"):
        run_warnings.append(
            f"Auto-trim kept {win['t_start_s']:.1f}-{win['t_end_s']:.1f} s of the capture "
            f"({win['kept_profiles']} profiles); {win['dropped_profiles']} profiles outside that window "
            "had no continuous laser line and were ignored."
        )
    if rot_span is not None and rot_span > 15.0:
        run_warnings.append(
            f"The scanner rotated up to {rot_span:.0f} deg during the kept window. Sweep mode assumes a "
            "straight pass without turning; keep the wand orientation fixed and translate it."
        )
    if cleaned_mean < 100.0:
        run_warnings.append(
            f"Only {cleaned_mean:.0f} clean laser points per profile in the kept window; raise exposure or "
            "laser power until the line peak is ~150-220 DN without saturating."
        )
    if nominal_cal:
        pass  # already reported by the caller
    if geometry["metrics"]["max_extent_mm"] < 35.0:
        run_warnings.append(
            f"Surface spans {geometry['metrics']['max_extent_mm']:.1f} mm: a single-pass patch. "
            "Automatic MeshSegNet needs a full-arch scan and is skipped for sweep patches."
        )
    if tr["used"] == "markers":
        run_warnings.append(
            "A top view cannot see steep tooth walls. Scan again with the plate tilted on a wedge and combine the "
            "scans with Build arch in the Library (the model must not move on the plate in between).")
    else:
        run_warnings.append("A single laser pass observes the occlusal surface only; walls are not captured.")

    # MeshSegNet was trained on full-arch scans; a single sweep is a partial patch.
    segmentation_ready = False
    seg_report = None
    if segment:
        run_warnings.append(
            "--segment skipped: sweep mode produces a single-pass partial surface, and MeshSegNet expects "
            "a full arch. Run it manually from the dashboard if you still want to try."
        )

    quality_report = {
        "measured_capture_fps": round(measured_capture_fps, 3),
        "median_frame_interval_ms": round(float(np.median(frame_intervals) * 1000.0), 3) if frame_intervals.size else None,
        "laser_on_profile_count": len(on_records),
        "laser_profile_rate_hz": round(laser_profile_rate_hz, 3),
        "usable_profiles": usable,
        "kept_profiles": rep.get("kept_profiles"),
        "mean_points_per_usable_profile": round(total_points / max(usable, 1), 2),
        "mean_clean_points_per_kept_profile": cleaned_mean,
        "median_profile_spacing_mm": tr.get("median_profile_spacing_mm"),
        "mesh_extent_mm": [round(float(v), 4) for v in extent],
        "segmentation_ready_by_basic_geometry_gate": segmentation_ready,
        "geometry_gate": geometry,
        "depth_window": depth_window_info,
        "imu_rotation_span_deg": None if rot_span is None else round(rot_span, 2),
    }
    report = {
        "session": str(session_dir),
        "session_source": source,
        "pose_mode": "sweep",
        "laser_flags": laser_flag_info,
        "translation": tr,
        "active_window": win,
        "profile_cleaning": rep.get("profile_cleaning"),
        "plate_check": rep.get("plate_check"),
        "heightmap": rep.get("heightmap"),
        "ridge_method": ridge_method,
        "ridge": {"max_secondary_ratio": float(ridge_secondary_ratio), "min_prominence": float(ridge_min_prominence),
                  "adaptive_guard": True},
        "mesh_method": "heightmap",
        "depth_range_mm": [float(z_range[0]), float(z_range[1])],
        "depth_window": depth_window_info,
        "speed_mm_s_used": tr.get("speed_mm_s"),
        "camera": camera.to_dict(),
        "laser_plane": laser.to_dict(),
        "calibration_is_nominal": nominal_cal,
        "roi_enabled": bool(roi_config["enabled"]),
        "roi_vertices": len(roi_config["points"]),
        "frames_total": len(records),
        "laser_on_frames_with_points": usable,
        "raw_triangulated_points": total_points,
        "fused_cloud_points": int(len(result.cloud)),
        "mesh_vertices": int(len(V)),
        "mesh_triangles": int(len(F)),
        "estimated_trajectory_length_mm": path_mm,
        "quality": quality_report,
        "camera_actual_fourcc": actual_camera_fourcc,
        "warnings": run_warnings,
        "paths": {k: str(v) for k, v in paths.items()},
        "segmentation": seg_report,
    }
    report["verdict"] = scan_verdict(report, result.vertices)
    report_path = out_dir / "pipeline_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    return {
        "verdict": report["verdict"],
        "on_frames_with_points": usable,
        "raw_triangulated_points": total_points,
        "fused_cloud_points": int(len(result.cloud)),
        "mesh_vertices": int(len(V)),
        "mesh_triangles": int(len(F)),
        "path_mm": path_mm,
        "quality": quality_report,
        "paths": paths,
        "segmentation": seg_report,
        "warnings": run_warnings,
        "report_json": report_path,
        "poses_npz": poses_npz_path,
        "translation_source": tr["used"],
        "pose_mode": "sweep",
    }


def _write_sweep_preview(path: Path, profiles: list, flow, cfg: SweepConfig, camera, t_first: float) -> bool:
    """Provisional live cloud while frames are processed (flow if reliable so far, else speed)."""
    pts = [p for p in profiles if len(p.pts)]
    if not pts:
        return False
    stride = max(1, len(pts) // 400)
    pts = pts[::stride]
    t = np.array([p.t for p in pts])
    off = np.zeros((len(pts), 2))
    used_flow = False
    if flow is not None and len(flow.samples) >= 5:
        dz = np.array([float(np.median(p.pts[:, 2])) for p in pts])
        ft, cum, info = flow_to_translation(flow.samples, t, dz, camera.fx, camera.fy, cfg)
        if ft.size >= 3 and info.get("valid_fraction", 0) >= cfg.flow_min_valid_fraction:
            off = np.column_stack([np.interp(t, ft, cum[:, 0]), np.interp(t, ft, cum[:, 1])])
            used_flow = True
    if not used_flow:
        off[:, 1] = cfg.scan_direction * cfg.speed_mm_s * (t - t_first)
    cloud = np.vstack([p.pts[::2] + np.array([o[0], o[1], 0.0]) for p, o in zip(pts, off)])
    try:
        write_ply_points(path, cloud)
        return True
    except OSError:
        return False


def run_pipeline(
    session_dir: Path,
    out_dir: Path,
    pose_mode: str = "sweep",
    ridge_method: str = "cog",
    mesh_method: str = "bpa",
    voxel_size_mm: float = 0.08,
    speed_mm_s: float = 5.0,
    world_velocity_mm_s: tuple[float, float, float] | None = None,
    init_from_gt: bool = False,
    z_range: tuple[float, float] = (75.0, 125.0),
    segment: bool = False,
    seg_cfg=None,
    progress_callback=None,
    scan_length_mm: float | None = None,
    auto_depth_window: bool = True,
    imu_to_camera: np.ndarray | None = None,
    translation: str = "auto",
    auto_trim: bool = True,
    auto_depth_from_roi: bool = True,
    ridge_secondary_ratio: float = 0.6,
    ridge_min_prominence: float = 25.0,
    sweep_cfg: SweepConfig | None = None,
    verify_flags: bool = True,
    calibration_file: str | Path | None = None,
    marker_board: MarkerBoard | None = None,
    adaptive_line: bool = True,
) -> dict:
    session_dir, out_dir = Path(session_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if pose_mode != "sweep":
        try:
            from ioscan.pipeline.fusion import FusionConfig, ScanFuser
            from ioscan.pipeline.mesh import MeshConfig, export_scan_outputs, reconstruct_mesh
        except ImportError as exc:
            if exc.name == "open3d":
                raise RuntimeError(
                    "The IMU reconstruction modes require Open3D. Install the project dependencies "
                    "with: python -m pip install -e .  (the default 'sweep' mode does not need it)"
                ) from exc
            raise

    camera, laser, nominal_cal = load_calibration(session_dir)
    calibration_override = None
    if calibration_file:
        # Reprocess with a calibration measured after the session was recorded (same optics).
        cal = json.loads(Path(calibration_file).read_text(encoding="utf-8"))
        cam_new = CameraIntrinsics.from_dict(cal["camera"])
        if (cam_new.width, cam_new.height) != (camera.width, camera.height):
            raise ValueError(
                f"Calibration {calibration_file} is for {cam_new.width}x{cam_new.height} images, but this "
                f"session has {camera.width}x{camera.height}.")
        camera, laser, nominal_cal = cam_new, LaserPlane.from_dict(cal["laser"]), False
        calibration_override = str(calibration_file)
    # Fail early with an actionable optical-geometry diagnostic. A plane that
    # intersects the camera rays only outside z_range can never yield points.
    plane_z_bounds = laser_plane_depth_bounds(camera, laser)
    if plane_z_bounds is not None and (plane_z_bounds[1] < z_range[0] or plane_z_bounds[0] > z_range[1]):
        raise RuntimeError(
            "Laser geometry does not intersect the selected reconstruction depth range. "
            f"The stored camera/laser model predicts positive intersections around "
            f"Z={plane_z_bounds[0]:.1f}..{plane_z_bounds[1]:.1f} mm across the image, "
            f"but the pipeline is filtering to Z={z_range[0]:.1f}..{z_range[1]:.1f} mm. "
            "Use measured camera/laser calibration and set the depth range to the actual "
            "tooth working distance; do not change these values just to force points."
        )
    records = load_frame_records(session_dir)
    laser_flag_info: dict = {"action": "not_checked"}
    if verify_flags:
        if progress_callback is not None:
            progress_callback({"stage": "reconstructing", "progress": 0.5,
                               "message": "Verifying laser ON/OFF frames from the images"})
        flag_box = None
        try:
            _roi_pts = enabled_roi_points(load_roi_config(session_dir / "roi.json"))
            if _roi_pts is not None:
                _rp = np.asarray(_roi_pts, dtype=np.float64)
                flag_box = (float(_rp[:, 0].min()), float(_rp[:, 1].min()), float(_rp[:, 0].max()), float(_rp[:, 1].max()))
        except Exception:
            flag_box = None
        records, laser_flag_info = verify_laser_flags(records, roi_box=flag_box)
    # Only clear outputs after required dependencies and session inputs validate.
    _clear_stale_scan_outputs(out_dir)
    source = session_source(session_dir)
    meta_path = session_dir / "meta.json"
    session_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    camera_runtime = session_meta.get("camera_runtime") or {}
    requested_camera_fps = float(session_meta.get("requested_fps", session_meta.get("fps", 50.0)) or 50.0)
    actual_camera_fourcc = camera_runtime.get("actual_fourcc_text")
    timestamps = np.asarray([float(r["timestamp_s"]) for r in records], dtype=np.float64)
    frame_intervals = np.diff(timestamps)
    measured_capture_fps = (
        float((len(timestamps) - 1) / (timestamps[-1] - timestamps[0]))
        if len(timestamps) > 1 and timestamps[-1] > timestamps[0] else 0.0
    )
    on_records = [r for r in records if bool(r["laser_on"])]
    on_times = np.asarray([float(r["timestamp_s"]) for r in on_records], dtype=np.float64)
    on_intervals = np.diff(on_times)
    laser_profile_rate_hz = (
        float((len(on_times) - 1) / (on_times[-1] - on_times[0]))
        if len(on_times) > 1 and on_times[-1] > on_times[0] else 0.0
    )
    # The dashboard snapshots the operator-drawn ROI into each session. Older
    # sessions without roi.json intentionally retain full-frame processing.
    roi_config = load_roi_config(session_dir / "roi.json")
    roi_polygon = enabled_roi_points(roi_config)
    gt_poses = load_gt_poses(session_dir) if (pose_mode == "gt" or init_from_gt) else {}

    run_warnings: list[str] = []
    if calibration_override:
        run_warnings.append("Processed with the current calibration file, not the values stored when the scan was recorded.")
    if laser_flag_info.get("action") in ("inverted", "replaced"):
        run_warnings.append(
            f"Laser ON/OFF flags were wrong in the recording ({laser_flag_info['action']}, "
            f"{laser_flag_info.get('frames_changed', 0)} frames, image agreement "
            f"{100 * laser_flag_info.get('agreement', 0):.0f}%). They were corrected from the images; "
            "the capture is pairing camera frames and ESP32 strobe packets one frame apart."
        )
    elif laser_flag_info.get("reason"):
        run_warnings.append(f"Laser ON/OFF flags could not be verified from the images: {laser_flag_info['reason']}.")
    # The ROI decides which pixels can hold the laser line; each pixel has exactly one depth on the
    # laser plane. A depth window that does not overlap the ROI's depths discards ~all points.
    requested_z_range = (float(z_range[0]), float(z_range[1]))
    depth_window_info: dict = {}
    if auto_depth_from_roi and roi_polygon is not None:
        # Every ROI pixel maps to exactly one depth on the laser plane, so the ROI already bounds
        # the depths that can be observed. A narrower manual window only clips real surface.
        cov = roi_depth_coverage(camera, laser, roi_polygon, requested_z_range)
        if cov.get("suggested_z_range_mm"):
            z_range = tuple(cov["suggested_z_range_mm"])
            depth_window_info = dict(cov, mode="auto_from_roi", requested_z_range_mm=list(requested_z_range))
    if not depth_window_info:
        z_range, depth_window_info, depth_window_msg = resolve_depth_window(
            camera, laser, roi_polygon, z_range, auto_adjust=auto_depth_window,
        )
        depth_window_info = dict(depth_window_info, mode="manual")
        if depth_window_msg:
            run_warnings.append(depth_window_msg)
    if scan_length_mm is not None and pose_mode != "sweep":
        if len(on_times) < 2 or on_times[-1] <= on_times[0]:
            raise ValueError("--scan-length-mm needs at least two laser-ON frames with increasing timestamps")
        duration_s = float(on_times[-1] - on_times[0])
        speed_mm_s = float(scan_length_mm) / duration_s
        run_warnings.append(
            f"Scan speed set from the known travel: {scan_length_mm:.1f} mm over {duration_s:.1f} s = "
            f"{speed_mm_s:.3f} mm/s (assumes constant speed over the whole capture, no idle time)."
        )
    if pose_mode == "submap_icp":
        warnings.warn("pose mode 'submap_icp' is a deprecated alias of 'imu_hybrid'", DeprecationWarning)
        pose_mode = "imu_hybrid"
    if nominal_cal:
        run_warnings.append("Calibration is NOMINAL (not measured). Do not use for metrology.")
    if actual_camera_fourcc and str(actual_camera_fourcc).upper() != "MJPG":
        run_warnings.append(
            f"Camera negotiated {actual_camera_fourcc}, not MJPG. The OV2311 1600x1200 YUY2 mode is a low-rate mode; "
            "reconfigure the UVC stream to MJPG before scanning."
        )
    serial_warnings = session_meta.get("serial_warnings") or []
    if serial_warnings:
        run_warnings.append(
            f"Firmware recorded {len(serial_warnings)} serial warning(s); first warning: {serial_warnings[0]}"
        )
    if roi_polygon is not None:
        run_warnings.append(
            f"Scan area (ROI) with {len(roi_polygon)} corners was applied: laser points are taken only inside it."
        )
    else:
        run_warnings.append(
            "Camera ROI mask is DISABLED. Chassis/interior pixels are not excluded; draw and enable "
            "the optical-opening ROI in the dashboard Setup tab before rescanning."
        )
    if pose_mode == "sweep":
        return _run_sweep(
            session_dir=session_dir, out_dir=out_dir, records=records, camera=camera, laser=laser,
            nominal_cal=nominal_cal, source=source, session_meta=session_meta, roi_config=roi_config,
            roi_polygon=roi_polygon, z_range=z_range, depth_window_info=depth_window_info,
            run_warnings=run_warnings, ridge_method=ridge_method, speed_mm_s=speed_mm_s,
            scan_length_mm=scan_length_mm, translation=translation, auto_trim=auto_trim,
            ridge_secondary_ratio=ridge_secondary_ratio, ridge_min_prominence=ridge_min_prominence,
            sweep_cfg=sweep_cfg, segment=segment, seg_cfg=seg_cfg, progress_callback=progress_callback,
            measured_capture_fps=measured_capture_fps, requested_camera_fps=requested_camera_fps,
            frame_intervals=frame_intervals, on_records=on_records,
            laser_profile_rate_hz=laser_profile_rate_hz, actual_camera_fourcc=actual_camera_fourcc,
            laser_flag_info=laser_flag_info, marker_board=marker_board, adaptive_line=adaptive_line,
        )
    if pose_mode == "imu_hybrid":
        run_warnings.append(
            f"Translation is assumed: constant {speed_mm_s} mm/s along the camera scan axis; "
            "the IMU only supplies orientation."
        )
    if source == "sim":
        run_warnings.append("Simulated session: the IMU stream is noise-free and equals ground truth.")
    if init_from_gt:
        run_warnings.append("Start position was taken from simulator ground truth (--init-from-gt).")

    init_pos = (0.0, 0.0, 0.0)
    if init_from_gt:
        if 0 not in gt_poses:
            raise ValueError("--init-from-gt needs gt/poses.npz (simulator sessions only)")
        init_pos = tuple(gt_poses[0][:3, 3].tolist())

    if pose_mode == "imu_const_vel" and world_velocity_mm_s is None:
        raise ValueError("pose mode 'imu_const_vel' needs --world-velocity VX VY VZ (mm/s)")
    pose_cfg = PoseConfig(
        mode=pose_mode,
        nominal_speed_mm_s=speed_mm_s,
        world_velocity_mm_s=tuple(world_velocity_mm_s) if world_velocity_mm_s else (0.0, 0.0, 0.0),
        initial_position_mm=init_pos,
        imu_to_camera=None if imu_to_camera is None else tuple(map(tuple, np.asarray(imu_to_camera, dtype=float).reshape(3, 3))),
    )
    tracker = PoseTracker(pose_cfg, initial_timestamp_s=records[0]["timestamp_s"])
    ridge_cfg = RidgeConfig(method=ridge_method, max_secondary_ratio=ridge_secondary_ratio,
                            min_prominence=ridge_min_prominence)
    ridge_cfg_8bit = RidgeConfig(method=ridge_method, saturation_limit=254.0,  # hardware frames are 8-bit
                                 max_secondary_ratio=ridge_secondary_ratio, min_prominence=ridge_min_prominence)
    fuser = ScanFuser(camera, FusionConfig(voxel_size_mm=voxel_size_mm, use_tsdf=(mesh_method == "tsdf")))
    mesh_cfg = MeshConfig(method=mesh_method)

    last_off: np.ndarray | None = None
    n_on_processed = 0
    total_points = 0
    unpaired_on = 0

    preview_path = out_dir / "reconstruction_preview.ply"
    preview_every = max(5, len(records) // 24)
    for rec_pos, rec in enumerate(tqdm(records, desc=f"Pipeline ({pose_mode})")):
        frame = read_grayscale_frame(rec["path"])
        if not rec["laser_on"]:
            last_off = frame
            continue
        if last_off is None:
            unpaired_on += 1

        pts_cam, uv, _ = triangulate_frame(
            frame, last_off, camera, laser,
            ridge_cfg=ridge_cfg_8bit if frame.dtype == np.uint8 else ridge_cfg,
            z_range=z_range, roi_polygon=roi_polygon,
        )
        T_wc = tracker.update(
            frame_index=rec["index"],
            timestamp_s=rec["timestamp_s"],
            quat_wxyz=rec["quat_wxyz"],  # None raises in imu modes: no made-up orientation
            pts_cam=pts_cam,
            gt_T_wc=gt_poses.get(rec["index"]),
        )
        if pts_cam.shape[0] > 0:
            fuser.integrate(pts_cam, uv, T_wc)
            n_on_processed += 1
            total_points += pts_cam.shape[0]

        if progress_callback is not None:
            pct = 82.0 * (rec_pos + 1) / max(len(records), 1)
            preview_written = False
            if n_on_processed and ((rec_pos + 1) % preview_every == 0 or rec_pos + 1 == len(records)):
                preview_pcd = fuser.get_preview_point_cloud(voxel_downsample_mm=max(voxel_size_mm * 2.0, 0.16))
                if len(preview_pcd.points) > 0:
                    import open3d as o3d
                    preview_tmp = out_dir / "reconstruction_preview.tmp.ply"
                    if o3d.io.write_point_cloud(str(preview_tmp), preview_pcd, write_ascii=False):
                        os.replace(preview_tmp, preview_path)
                        preview_written = True
            progress_callback({
                "stage": "reconstructing", "progress": round(pct, 2),
                "frames_done": rec_pos + 1, "frames_total": len(records),
                "laser_on_frames": n_on_processed, "triangulated_points": total_points,
                "preview_path": str(preview_path) if preview_written else None,
            })

    if progress_callback is not None:
        progress_callback({"stage": "meshing", "progress": 86.0, "message": "Building the surface mesh"})
    if unpaired_on:
        run_warnings.append(f"{unpaired_on} ON frame(s) had no preceding OFF frame (no ambient subtraction).")
    if n_on_processed == 0:
        raise RuntimeError("No laser points were triangulated; check calibration, z-range and frames.")

    mesh, pcd = reconstruct_mesh(fuser, cfg=mesh_cfg)
    paths = export_scan_outputs(out_dir, mesh, pcd)
    if progress_callback is not None:
        progress_callback({"stage": "mesh_ready", "progress": 100.0, "message": "Surface reconstruction complete", "mesh_path": str(paths["mesh_ply"])})
    if len(mesh.triangles) == 0:
        run_warnings.append("Mesh is empty (too few points); no mesh files were written.")

    all_fidxs = sorted(tracker.poses.keys())
    all_T = np.array([tracker.poses[f] for f in all_fidxs], dtype=np.float64) if all_fidxs else np.empty((0, 4, 4))
    poses_npz_path = out_dir / "estimated_poses.npz"
    np.savez_compressed(poses_npz_path, frame_indices=np.array(all_fidxs), T_wc=all_T)
    path_mm = float(np.linalg.norm(np.diff(all_T[:, :3, 3], axis=0), axis=1).sum()) if len(all_T) > 1 else 0.0
    if path_mm < 30.0:
        run_warnings.append(f"Trajectory is only {path_mm:.1f} mm long: a local patch, not a full arch.")
    run_warnings.append("A single laser pass observes the occlusal surface only; walls are not captured.")

    mesh_vertices_np = np.asarray(mesh.vertices, dtype=np.float64)
    mesh_extent = (
        (mesh_vertices_np.max(axis=0) - mesh_vertices_np.min(axis=0)).tolist()
        if mesh_vertices_np.size else [0.0, 0.0, 0.0]
    )
    pose_on = [tracker.poses[r["index"]] for r in on_records if r["index"] in tracker.poses]
    pose_positions = np.asarray([T[:3, 3] for T in pose_on], dtype=np.float64)
    profile_steps = np.linalg.norm(np.diff(pose_positions, axis=0), axis=1) if len(pose_positions) > 1 else np.asarray([])
    median_profile_spacing_mm = float(np.median(profile_steps)) if profile_steps.size else 0.0
    mean_points_per_profile = float(total_points / n_on_processed) if n_on_processed else 0.0

    # Engineering guard: size, compactness and connectivity of the mesh and plausibility of the path.
    # (An extent >= 35 mm alone accepted a 205 mm smeared ribbon.)
    geometry = assess_scan_geometry(mesh_vertices_np, np.asarray(mesh.triangles), path_mm=path_mm)
    segmentation_ready = bool(geometry["ok"]) and len(mesh.triangles) > 0
    mesh_extent_max_mm = float(max(mesh_extent)) if mesh_extent else 0.0
    for reason in geometry["reasons"]:
        run_warnings.append(f"Mesh-quality gate failed: {reason}.")
    if path_mm > 150.0:
        run_warnings.append(
            f"The assumed speed ({speed_mm_s} mm/s) produced a {path_mm:.0f} mm path. If the real pass "
            "length L (mm) is known, rerun with --scan-length-mm L; otherwise use a stage/encoder."
        )
    if measured_capture_fps < 0.85 * requested_camera_fps:
        run_warnings.append(
            f"Measured capture rate is {measured_capture_fps:.2f} FPS; the requested "
            f"{requested_camera_fps:.1f} FPS acquisition target was not met."
        )
    if mean_points_per_profile < 100.0:
        run_warnings.append(
            f"Only {mean_points_per_profile:.1f} laser points per usable profile on average; "
            "inspect exposure, optical filter, ROI and ridge extraction before meshing."
        )
    if median_profile_spacing_mm > 0.34:
        run_warnings.append(
            f"Median assumed profile spacing is {median_profile_spacing_mm:.3f} mm, exceeding "
            "the current BPA maximum radius of 0.34 mm; adjacent profiles cannot be reliably bridged."
        )

    seg_report = None
    if segment:
        if segmentation_ready:
            from ioscan.pipeline.segment import SegmentConfig, segment_mesh_and_cloud  # needs the [seg] extra
            seg_report = segment_mesh_and_cloud(
                paths["mesh_ply"], out_dir, cfg=seg_cfg or SegmentConfig(),
                cloud_path=paths["cloud_ply"], session_hint=session_dir,
            )
        else:
            run_warnings.append(
                "--segment requested but the scan-quality gate failed; segmentation was skipped. "
                "Recapture a dense, overlapping dental surface first."
            )

    quality_report = {
        "measured_capture_fps": round(measured_capture_fps, 3),
        "median_frame_interval_ms": round(float(np.median(frame_intervals) * 1000.0), 3) if frame_intervals.size else None,
        "laser_on_profile_count": len(on_records),
        "laser_profile_rate_hz": round(laser_profile_rate_hz, 3),
        "usable_profiles": n_on_processed,
        "mean_points_per_usable_profile": round(mean_points_per_profile, 2),
        "median_profile_spacing_mm": round(median_profile_spacing_mm, 4),
        "mesh_extent_mm": [round(float(v), 4) for v in mesh_extent],
        "segmentation_ready_by_basic_geometry_gate": bool(segmentation_ready),
        "geometry_gate": geometry,
        "depth_window": depth_window_info,
        "quality_gate_thresholds": {
            "min_mesh_faces": 5000,
            "min_max_extent_mm": 35.0,
            "max_extent_mm": 110.0,
            "requested_capture_fps": requested_camera_fps,
            "min_capture_fps_fraction_of_requested": 0.85,
            "recommended_min_points_per_profile": 100,
            "bpa_max_radius_mm": 0.34,
        },
    }

    report_path = out_dir / "pipeline_report.json"
    report_path.write_text(json.dumps({
        "session": str(session_dir),
        "session_source": source,
        "pose_mode": tracker.cfg.mode,
        "nominal_speed_mm_s": speed_mm_s,
        "ridge_method": ridge_method,
        "mesh_method": mesh_method,
        "depth_range_mm": [float(z_range[0]), float(z_range[1])],
        "depth_window": depth_window_info,
        "speed_mm_s_used": float(speed_mm_s),
        "camera": camera.to_dict(),
        "laser_plane": laser.to_dict(),
        "calibration_is_nominal": nominal_cal,
        "roi_enabled": bool(roi_config["enabled"]),
        "roi_vertices": len(roi_config["points"]),
        "frames_total": len(records),
        "laser_on_frames_with_points": n_on_processed,
        "raw_triangulated_points": total_points,
        "fused_cloud_points": len(pcd.points),
        "mesh_vertices": len(mesh.vertices),
        "mesh_triangles": len(mesh.triangles),
        "estimated_trajectory_length_mm": path_mm,
        "quality": quality_report,
        "camera_actual_fourcc": actual_camera_fourcc,
        "warnings": run_warnings,
        "paths": {k: str(v) for k, v in paths.items()},
        "segmentation": seg_report,
    }, indent=2), encoding="utf-8")

    return {
        "on_frames_with_points": n_on_processed,
        "raw_triangulated_points": total_points,
        "fused_cloud_points": len(pcd.points),
        "mesh_vertices": len(mesh.vertices),
        "mesh_triangles": len(mesh.triangles),
        "path_mm": path_mm,
        "quality": quality_report,
        "paths": paths,
        "segmentation": seg_report,
        "warnings": run_warnings,
        "report_json": report_path,
        "poses_npz": poses_npz_path,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the 3D intraoral reconstruction pipeline.")
    ap.add_argument("session", type=Path, help="Input session directory")
    ap.add_argument("--out", type=Path, default=Path("data/results/latest_scan"))
    ap.add_argument("--pose-mode", choices=["sweep", "imu_hybrid", "imu_const_vel", "gt", "submap_icp"], default="sweep",
                    help="sweep (default): straight pass, measured/known translation, height-map surface. "
                         "imu_*: legacy IMU-orientation modes (experimental, need Open3D)")
    ap.add_argument("--translation", choices=["auto", "markers", "planar", "flow", "length", "speed"], default="auto",
                    help="sweep mode: auto = marker plate if its markers are visible, else image tracking of the "
                         "model, else --scan-length-mm, else --speed-mm-s")
    ap.add_argument("--calibration", type=Path, default=None,
                    help="Process with this calibration.json instead of the calibration stored in the session")
    ap.add_argument("--fixed-line-threshold", action="store_true",
                    help="Use --min-prominence as a fixed threshold instead of adapting it to the image noise")
    ap.add_argument("--no-auto-trim", action="store_true",
                    help="sweep mode: keep the whole capture instead of the window where the line is on the surface")
    ap.add_argument("--manual-depth-window", action="store_true",
                    help="Use --z-min/--z-max as given instead of deriving the window from the ROI")
    ap.add_argument("--secondary-ratio", type=float, default=0.6,
                    help="Reject a column if another peak exceeds this fraction of the main peak (1.0 = off)")
    ap.add_argument("--trust-laser-flags", action="store_true",
                    help="Use the recorded laser ON/OFF flags as-is (default: verify them from the images)")
    ap.add_argument("--min-prominence", type=float, default=25.0,
                    help="Minimum laser peak above ambient (DN) for a column to count")
    ap.add_argument("--speed-mm-s", type=float, default=5.0, help="Assumed scan speed for imu_hybrid")
    ap.add_argument("--world-velocity", type=float, nargs=3, metavar=("VX", "VY", "VZ"), default=None,
                    help="World velocity (mm/s) for imu_const_vel")
    ap.add_argument("--init-from-gt", action="store_true",
                    help="Simulator sessions only: start at the ground-truth position (for --no-icp evaluation)")
    ap.add_argument("--ridge", choices=["cog", "gaussian3", "steger"], default="cog")
    ap.add_argument("--mesh", choices=["bpa", "poisson", "tsdf"], default="bpa")
    ap.add_argument("--voxel-mm", type=float, default=0.08)
    ap.add_argument("--z-min", type=float, default=75.0, help="Minimum accepted depth (mm)")
    ap.add_argument("--z-max", type=float, default=125.0, help="Maximum accepted depth (mm)")
    ap.add_argument("--strict-depth-window", action="store_true",
                    help="Fail instead of replacing a depth window that does not overlap the ROI's depths")
    ap.add_argument("--scan-length-mm", type=float, default=None,
                    help="Known physical travel of the pass (mm); sets the assumed speed = length / duration")
    ap.add_argument("--imu-to-camera-deg", type=float, nargs=3, metavar=("RX", "RY", "RZ"), default=None,
                    help="Optional IMU->camera mounting rotation (extrinsic xyz Euler degrees)")
    ap.add_argument("--segment", action="store_true", help="Run MeshSegNet segmentation (needs the [seg] extra)")
    ap.add_argument("--jaw", choices=["auto", "lower", "upper"], default="auto")
    ap.add_argument("--models-dir", type=Path, default=None, help="MeshSegNet checkpoints (or $IOSCAN_MESHSEGNET_MODELS)")
    ap.add_argument("--ckpt", type=Path, default=None, help="Explicit MeshSegNet checkpoint")
    ap.add_argument("--repo-dir", type=Path, default=None,
                    help="Optional reference MeshSegNet checkout containing meshsegnet.py")
    ap.add_argument("--target-cells", type=int, default=10000,
                    help="Maximum triangle count passed to MeshSegNet (default: 10000)")
    ap.add_argument("--seg-device", type=str, default=None,
                    help="MeshSegNet device, e.g. cpu or cuda:0 (default: auto with CPU fallback)")
    args = ap.parse_args()

    seg_cfg = None
    if args.segment:
        from ioscan.pipeline.segment import SegmentConfig
        kw = dict(jaw=args.jaw, ckpt_path=args.ckpt, repo_dir=args.repo_dir,
                  target_cells=args.target_cells, device=args.seg_device)
        if args.models_dir is not None:
            kw["models_dir"] = args.models_dir
        seg_cfg = SegmentConfig(**kw)

    stats = run_pipeline(
        session_dir=args.session, out_dir=args.out, pose_mode=args.pose_mode, ridge_method=args.ridge,
        mesh_method=args.mesh, voxel_size_mm=args.voxel_mm, speed_mm_s=args.speed_mm_s,
        world_velocity_mm_s=tuple(args.world_velocity) if args.world_velocity else None,
        init_from_gt=args.init_from_gt, z_range=(args.z_min, args.z_max),
        segment=args.segment, seg_cfg=seg_cfg,
        scan_length_mm=args.scan_length_mm, auto_depth_window=not args.strict_depth_window,
        translation=args.translation, auto_trim=not args.no_auto_trim,
        calibration_file=args.calibration, adaptive_line=not args.fixed_line_threshold,
        verify_flags=not args.trust_laser_flags,
        auto_depth_from_roi=not args.manual_depth_window,
        ridge_secondary_ratio=args.secondary_ratio, ridge_min_prominence=args.min_prominence,
        imu_to_camera=(None if args.imu_to_camera_deg is None else
                       __import__("scipy.spatial.transform", fromlist=["Rotation"]).Rotation.from_euler(
                           "xyz", args.imu_to_camera_deg, degrees=True).as_matrix()),
    )

    print("\n=== 3D Reconstruction Complete ===")
    print(f"  ON frames processed : {stats['on_frames_with_points']}")
    print(f"  Raw 3D points       : {stats['raw_triangulated_points']}")
    print(f"  Fused cloud points  : {stats['fused_cloud_points']}")
    print(f"  Mesh vertices       : {stats['mesh_vertices']}")
    print(f"  Mesh triangles      : {stats['mesh_triangles']}")
    print(f"  Estimated path      : {stats['path_mm']:.2f} mm"
          + (f"  (translation: {stats['translation_source']})" if stats.get('translation_source') else ""))
    print(f"  Saved Point Cloud   : {stats['paths']['cloud_ply']}")
    print(f"  Saved 3D Mesh (STL) : {stats['paths']['mesh_stl']}")
    print(f"  Saved pose/report   : {stats['poses_npz']} / {stats['report_json']}")
    for w in stats["warnings"]:
        print(f"  WARNING: {w}")

    seg = stats.get("segmentation")
    if seg:
        print("\n=== MeshSegNet Segmentation Summary ===")
        print(f"  Model Checkpoint    : {seg['checkpoint']} (jaw: {seg['jaw']})")
        print(f"  Predicted Gingiva   : {seg['gingiva_pct']:.1f}% | Teeth: {seg['teeth_pct']:.1f}%")
        print(f"  Segmented Full Mesh : {seg['full_colored_ply']}")
        print(f"  Per-tooth meshes    : {len(seg['per_tooth_meshes'])} class(es)")
        print(f"  Label data          : {seg['segmentation_labels_npz']}")
        print(f"  Inference device    : {seg['device']}")
        print(f"  Clean Teeth Mesh    : {seg['clean_teeth_ply'] or 'NOT WRITTEN (no teeth predicted)'}")
        if seg["segmented_cloud_ply"]:
            print(f"  Segmented Cloud     : {seg['segmented_cloud_ply']}")
        for w in seg["warnings"]:
            print(f"  WARNING: {w}")


if __name__ == "__main__":
    main()
