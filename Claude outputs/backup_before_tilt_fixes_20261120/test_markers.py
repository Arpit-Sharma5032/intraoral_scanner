"""Marker plate: pose from printed markers, free-motion reconstruction, calibration and merging."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from ioscan.calibration.models import CameraIntrinsics, LaserPlane
from ioscan.pipeline.markers import (
    MarkerBoard, MarkerDetector, MarkerTracker, marker_bits, poses_at, render_board_image, smooth_poses,
    solve_board_pose, to_plate_frame,
)
from ioscan.pipeline.verdict import scan_verdict

ROOT = Path(__file__).resolve().parents[1]
K = np.array([[933.3, 0, 800.0], [0, 933.3, 600.0], [0, 0, 1.0]])


def _load_pipeline():
    spec = importlib.util.spec_from_file_location("ioscan_run_pipeline_markers", ROOT / "scripts" / "run_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _view(board, R, t, *, blur=1.5, mirrored=False, seed=0, ppm=16.0):
    rng = np.random.default_rng(seed)
    tex, margin = render_board_image(board, ppm, 4.0)
    w = (board.cols - 1) * board.pitch_mm + board.marker_mm
    h = (board.rows - 1) * board.pitch_mm + board.marker_mm
    S = np.array([[1 / ppm, 0, -(w / 2 + margin)], [0, 1 / ppm, -(h / 2 + margin)], [0, 0, 1.0]])
    H = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ S
    img = cv2.warpPerspective(tex, H, (1600, 1200), flags=cv2.INTER_AREA, borderValue=120).astype(np.float32)
    img = 25 + 65 * img / 255.0
    img = cv2.GaussianBlur(img, (0, 0), blur) + rng.normal(0, 2.0, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.flip(img, 1) if mirrored else img


def _pose(yaw, tx, ty):
    return cv2.Rodrigues(np.array([tx, ty, 0.0]))[0] @ cv2.Rodrigues(np.array([0.0, 0.0, yaw]))[0]


def test_board_layout_is_consistent_with_the_printed_sheet():
    b = MarkerBoard()
    obj = b.object_points()
    assert obj.shape == (b.count, 4, 3)
    assert np.allclose(obj[:, :, 2], 0.0)
    assert np.allclose(np.linalg.norm(obj[:, 1] - obj[:, 0], axis=1), b.marker_mm)       # TL -> TR
    assert np.allclose(obj[1, 0, 0] - obj[0, 0, 0], b.pitch_mm)                           # next column
    assert np.allclose(obj.reshape(-1, 3).mean(axis=0), 0.0, atol=1e-9)                   # centred
    assert marker_bits(b, 0).shape == (6, 6) and marker_bits(b, 0)[0].sum() == 0          # black border
    scaled = MarkerBoard(scale=1.02).object_points()
    assert np.allclose(scaled, obj * 1.02)


@pytest.mark.parametrize("mirrored", [False, True])
def test_plate_pose_from_markers_is_accurate(mirrored):
    b = MarkerBoard()
    rng = np.random.default_rng(3)
    errs = []
    for i in range(6):
        R = _pose(rng.uniform(-3, 3), rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2))
        t = np.array([rng.uniform(-15, 15), rng.uniform(-5, 20), rng.uniform(82, 96)])
        tr = MarkerTracker(b, K, None, mirrored=mirrored)
        tr.add(_view(b, R, t, mirrored=mirrored, seed=i), 0.0)
        tr.flush()
        smp = tr.samples[-1]
        assert smp.valid and smp.n_markers >= 20
        # a probe point 15 mm above the sheet must land where the true pose puts it
        probe_cam = R @ np.array([10.0, -5.0, -15.0]) + t
        if mirrored:   # the tracker works in the mirrored camera/plate pair
            probe_cam = probe_cam * np.array([-1.0, 1.0, 1.0]) + np.array([2 * (799.5 - 800.0) / 933.3 * probe_cam[2], 0, 0])
            expect = np.array([-10.0, -5.0, -15.0])
        else:
            expect = np.array([10.0, -5.0, -15.0])
        errs.append(np.linalg.norm(to_plate_frame(probe_cam[None], smp.R, smp.tvec)[0] - expect))
    assert np.median(errs) < 0.15 and max(errs) < 0.5


def test_pose_rejects_a_wrongly_decoded_marker():
    b = MarkerBoard()
    R, t = _pose(0.7, 0.1, -0.05), np.array([3.0, 6.0, 90.0])
    det = MarkerDetector(b)
    ids, corners = det.detect(_view(b, R, t))
    obj = det.object_points[ids].copy()
    obj[2] = det.object_points[(ids[2] + 37) % b.count]            # one marker with the wrong identity
    sol = solve_board_pose(obj.reshape(-1, 3), corners.reshape(-1, 2), K, None)
    assert sol is not None
    Rm, tm, rms, keep = sol
    assert not keep.reshape(-1, 4)[2].any() and keep.sum() >= 4 * (len(ids) - 2)
    assert rms < 1.0 and np.linalg.norm(tm - t) < 0.4


def test_pose_interpolation_and_gaps():
    b = MarkerBoard()
    tr = MarkerTracker(b, K, None, workers=3)             # detection on a pool, poses still in frame order
    for i in range(6):
        if i == 3:
            tr.add(np.full((1200, 1600), 60, np.uint8), i * 0.04)   # markers hidden in one frame
            continue
        tr.add(_view(b, _pose(0.1 * i, 0.0, 0.0), np.array([0.5 * i, 0.0, 90.0]), seed=i), i * 0.04)
    tr.flush()
    assert [s.valid for s in tr.samples] == [True, True, True, False, True, True]
    assert [round(s.t, 2) for s in tr.samples] == [0.0, 0.04, 0.08, 0.12, 0.16, 0.2]
    sm = smooth_poses(tr.samples)
    R, tv, ok = poses_at(np.array([0.02, 0.12, 0.5]), sm)
    assert ok.tolist() == [True, True, False]                        # bridged one frame, not half a second
    assert tv[0, 0] == pytest.approx(0.25, abs=0.08)
    assert tv[1, 0] == pytest.approx(1.5, abs=0.08)


def test_marker_plate_scan_reconstructs_the_arch_under_free_motion(tmp_path):
    """Slide + ~70 deg turn + tilt + lift, mirrored optics: the half arch must match ground truth."""
    from tests._plyread import read_ply_mesh
    from tests._synth_marker import make_marker_session

    info = make_marker_session(tmp_path / "s", duration_s=5.0, span=(-0.9, 0.0), mirrored=True, seed=2)
    pipe = _load_pipeline()
    stats = pipe.run_pipeline(tmp_path / "s", tmp_path / "out")          # defaults: auto -> marker plate
    rep = json.loads((tmp_path / "out" / "pipeline_report.json").read_text())
    assert stats["translation_source"] == "markers"
    mk = rep["translation"]["markers"]
    assert mk["mirrored"] is True and mk["valid_fraction"] > 0.97 and mk["median_rms_px"] < 1.0
    assert rep["laser_flags"]["action"] == "kept"                 # moving markers must not flip laser flags
    assert rep["translation"]["rotation_range_deg"] > 50 and rep["translation"]["tilt_max_deg"] > 3
    assert abs(rep["plate_check"]["offset_mm"]) < 0.15
    V, F = read_ply_mesh(tmp_path / "out" / "reconstructed_mesh.ply")
    err = (-V[:, 2]) - info["model"].height(V[:, 0], V[:, 1])
    assert abs(np.median(err)) < 0.12
    assert 1.4826 * np.median(np.abs(err - np.median(err))) < 0.15
    assert np.ptp(V[:, 1]) > 40 and V[:, 0].mean() < -5            # true-handed: this half lies at -X
    assert rep["verdict"]["status"] == "good"


def test_few_visible_markers_are_enough_when_the_plate_slides_on_a_flat_floor(tmp_path):
    """A wide model hides most of the sheet (about 4 markers in view), the image is soft and dim."""
    from tests._plyread import read_ply_mesh
    from tests._synth_marker import make_marker_session

    info = make_marker_session(tmp_path / "s", duration_s=5.0, span=(-0.9, -0.3), blur=2.3, occlude_mm=13.0,
                               ambient=(14.0, 55.0), tilt_deg=0.0, lift_mm=0.0, seed=12)
    pipe = _load_pipeline()
    pipe.run_pipeline(tmp_path / "s", tmp_path / "out")
    rep = json.loads((tmp_path / "out" / "pipeline_report.json").read_text())
    mk = rep["translation"]["markers"]
    assert mk["median_markers"] <= 6
    assert mk["plate_plane"].startswith("fixed support") and mk["laser_assisted"] is True
    assert rep["plate_check"]["spread_mm"] < 0.1
    V, _F = read_ply_mesh(tmp_path / "out" / "reconstructed_mesh.ply")
    err = (-V[:, 2]) - info["model"].height(V[:, 0], V[:, 1])
    assert abs(np.median(err)) < 0.2 and 1.4826 * np.median(np.abs(err - np.median(err))) < 0.2
    assert rep["verdict"]["status"] in ("good", "check")


def test_calibration_from_the_marker_sheet_and_merge_of_two_scans(tmp_path):
    from tests._plyread import read_ply_mesh
    from tests._synth_marker import FlatSheet, PlateModel, make_marker_session
    from ioscan.calibration.marker_calibration import calibrate_from_marker_session, save_marker_calibration
    from ioscan.pipeline.stitch import StitchConfig, stitch_to_files

    true_cam = CameraIntrinsics(1600, 1200, 1012.0, 1012.0, 818.0, 585.0, (-0.09, 0.0, 0.0, 0.0, 0.0))
    true_las = LaserPlane.from_baseline_and_angle(18.1, 10.2, "y")
    nom_cam = CameraIntrinsics(1600, 1200, 933.3, 933.3, 800.0, 600.0)
    nom_las = LaserPlane.from_baseline_and_angle(19.0, 11.0, "y")
    kw = dict(camera=true_cam, laser=true_las, stored_camera=nom_cam, stored_laser=nom_las)
    make_marker_session(tmp_path / "cal", duration_s=9.0, tilt_deg=17.0, lift_mm=8.0, model=FlatSheet(),
                        plate_z=82.0, seed=5, **kw)
    res = calibrate_from_marker_session(tmp_path / "cal")
    cam, las, st = res["camera"], res["laser"], res["stats"]
    assert cam.fx == pytest.approx(1012.0, rel=0.01)
    assert cam.cx == pytest.approx(818.0, abs=6) and cam.cy == pytest.approx(585.0, abs=6)
    assert cam.dist_coeffs[0] == pytest.approx(-0.09, abs=0.02)
    assert np.degrees(np.arccos(abs(float(las.normal @ true_las.normal)))) < 0.3
    assert st["sheet_residual_mm"] < 0.2 and st["camera_rms_px"] < 1.0
    cal_file = save_marker_calibration(res, tmp_path / "calibration.json")

    model = PlateModel()
    pipe = _load_pipeline()
    make_marker_session(tmp_path / "a", duration_s=5.0, span=(-0.92, 0.06), seed=6, model=model, **kw)
    make_marker_session(tmp_path / "b", duration_s=5.0, span=(-0.06, 0.92), seed=7, model=model, **kw)
    # without the calibration the laser and the markers disagree, and the verdict says so
    bad = pipe.run_pipeline(tmp_path / "a", tmp_path / "a_nominal")
    assert bad["verdict"]["status"] == "rescan" and "not calibrated" in bad["verdict"]["reasons"][0]
    for name in ("a", "b"):
        ok = pipe.run_pipeline(tmp_path / name, tmp_path / "results" / name, calibration_file=cal_file)
        assert ok["translation_source"] == "markers" and ok["verdict"]["status"] in ("good", "check")
        V, _F = read_ply_mesh(tmp_path / "results" / name / "reconstructed_mesh.ply")
        err = (-V[:, 2]) - model.height(V[:, 0], V[:, 1])
        assert abs(np.median(err)) < 0.3 and 1.4826 * np.median(np.abs(err - np.median(err))) < 0.3

    cfg = StitchConfig()
    cfg.trim_to_teeth_mm = None
    rep = stitch_to_files([tmp_path / "results" / "a", tmp_path / "results" / "b"], tmp_path / "arch", cfg)
    assert rep["merged_on_marker_plate"] is True and rep["links"] == [] and rep["unreliable_joints"] == 0
    assert not any("disagrees" in w for w in rep["warnings"])
    Vc, _F = read_ply_mesh(tmp_path / "arch" / "reconstructed_mesh.ply")
    R, mu = np.array(rep["fusion"]["canonical_R"]), np.array(rep["fusion"]["canonical_mu"])
    Vp = Vc @ R + mu                                              # back to the plate frame
    err = (-Vp[:, 2]) - model.height(Vp[:, 0], Vp[:, 1])
    assert np.ptp(Vp[:, 0]) > 55                                   # both sides of the U
    assert abs(np.median(err)) < 0.3 and np.percentile(np.abs(err), 75) < 0.6


def test_marker_and_markerless_scans_cannot_be_mixed(tmp_path):
    from ioscan.pipeline.stitch import stitch_to_files
    from ioscan.pipeline.sweep import write_ply_points

    rng = np.random.default_rng(0)
    for name, used in (("m", "markers"), ("p", "planar")):
        d = tmp_path / name
        d.mkdir()
        P = np.column_stack([rng.uniform(0, 20, 4000), rng.uniform(0, 10, 4000), rng.uniform(-8, -6, 4000)])
        write_ply_points(d / "fused_cloud.ply", P)
        (d / "pipeline_report.json").write_text(json.dumps({"pose_mode": "sweep", "translation": {"used": used}}))
    with pytest.raises(ValueError, match="marker plate"):
        stitch_to_files([tmp_path / "m", tmp_path / "p"], tmp_path / "out")


def test_verdict_names_the_cause():
    flat = np.column_stack([np.random.default_rng(0).uniform(0, 20, (500, 2)), np.full(500, 70.0)])
    base = {"mesh_triangles": 5000, "heightmap": {"area_mm2": 200.0},
            "quality": {"mean_clean_points_per_kept_profile": 150.0, "kept_profiles": 400}}
    v = scan_verdict(dict(base, translation={"used": "markers", "markers": {"valid_fraction": 0.99, "median_rms_px": 0.4}}), flat)
    assert v["status"] == "rescan" and "flat" in v["reasons"][0]
    relief = flat.copy()
    relief[:, 2] += np.linspace(0, 6, 500)
    v = scan_verdict(dict(base, translation={"used": "markers", "markers": {"valid_fraction": 0.99, "median_rms_px": 0.4}}), relief)
    assert v["status"] == "good" and v["reasons"] == []
    v = scan_verdict(dict(base, translation={"used": "markers", "markers": {"valid_fraction": 0.4}}), relief)
    assert v["status"] == "rescan" and "marker plate was visible" in v["reasons"][0]
    v = scan_verdict(dict(base, translation={"used": "planar", "planar": {"valid_fraction": 1.0}}), relief)
    assert v["status"] == "check" and "marker plate" in v["reasons"][0]
    weak = dict(base, quality={"mean_clean_points_per_kept_profile": 12.0, "kept_profiles": 400},
                translation={"used": "markers", "markers": {"valid_fraction": 0.99}})
    assert "too weak" in scan_verdict(weak, relief)["reasons"][0]


def test_adaptive_line_threshold_finds_a_faint_line_on_a_bright_noisy_background():
    from ioscan.pipeline.ridge import RidgeConfig
    from ioscan.pipeline.triangulate import triangulate_frame

    rng = np.random.default_rng(1)
    cam = CameraIntrinsics(1600, 1200, 933.3, 933.3, 800.0, 600.0)
    laser = LaserPlane.from_baseline_and_angle(19.0, 11.0, "y")
    roi = [[0.41, 0.47], [0.59, 0.47], [0.59, 0.66], [0.41, 0.66]]
    off = np.clip(150 + rng.normal(0, 3.0, (1200, 1600)), 0, 255)
    on = np.clip(150 + rng.normal(0, 3.0, (1200, 1600)), 0, 255)
    rows = np.arange(1200)[:, None]
    on[:, 680:920] += 14.0 * np.exp(-((rows - 650.0) ** 2) / (2 * 2.0 ** 2))       # 14 DN line on 150 DN
    on, off = on.astype(np.uint8), off.astype(np.uint8)
    kw = dict(z_range=(40.0, 130.0), roi_polygon=roi)
    fixed = triangulate_frame(on, off, cam, laser, ridge_cfg=RidgeConfig(saturation_limit=254.0, min_prominence=25.0), **kw)
    adapt = triangulate_frame(on, off, cam, laser, ridge_cfg=RidgeConfig(saturation_limit=254.0, min_prominence=25.0,
                                                                         adaptive_floor=True), **kw)
    assert len(fixed[0]) < 10
    uv = adapt[1]
    on_line = (uv[:, 0] > 680) & (uv[:, 0] < 920)
    assert on_line.sum() > 200 and np.abs(uv[on_line, 1] - 650.0).max() < 1.5
    assert (~on_line).sum() < 5                                                     # no line invented in noise
    # and nothing is detected in a pair without a laser line
    other = np.clip(150 + rng.normal(0, 3.0, (1200, 1600)), 0, 255).astype(np.uint8)
    none = triangulate_frame(other, off, cam, laser,
                             ridge_cfg=RidgeConfig(saturation_limit=254.0, min_prominence=25.0, adaptive_floor=True), **kw)
    assert len(none[0]) < 5


def test_laser_flags_decode_alternation_through_weak_and_dropped_frames(tmp_path):
    from ioscan.session.laser_phase import verify_laser_flags

    rng = np.random.default_rng(0)
    recs, t, state = [], 0.0, False
    for i in range(80):
        img = np.clip(60 + rng.normal(0, 2, (300, 400)), 0, 255)
        if state and not (30 <= i < 40):                      # line off the surface for ten frames
            img[150:153, 100:300] += 70
        p = tmp_path / f"{i:03d}.png"
        cv2.imwrite(str(p), img.astype(np.uint8))
        recs.append({"index": i, "timestamp_s": t, "laser_on": (i % 2 == 0), "path": p})   # recorded flags
        step = 2 if i == 50 else 1                           # one dropped camera frame after frame 50
        t += 0.02 * step
        state = not state if step == 1 else state            # the laser kept toggling during the drop
    out, info = verify_laser_flags(recs)
    truth = []
    s = False
    for i in range(80):
        truth.append(s)
        s = (not s) if i != 50 else s
    assert [r["laser_on"] for r in out] == truth
    assert info["action"] in ("replaced", "inverted") and info["dropped_frame_gaps"] == 1
