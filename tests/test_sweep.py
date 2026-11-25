"""Linear-sweep reconstruction: profile cleaning, trimming, optical-flow translation, height-map mesh."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from ioscan.pipeline.ridge import RidgeConfig, extract_ridge_subpixel
from ioscan.pipeline.sweep import (
    OffFrameFlow, Profile, SweepConfig, active_window, build_heightmap, clean_profile,
    flow_to_translation, heightmap_to_mesh, reconstruct_sweep,
)
from ioscan.pipeline.quality import component_face_counts

ROOT = Path(__file__).resolve().parents[1]


def _load_pipeline():
    spec = importlib.util.spec_from_file_location("ioscan_run_pipeline_sweep", ROOT / "scripts" / "run_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_clean_profile_removes_spikes_and_fragments():
    u = np.arange(200, dtype=float)
    z = 80.0 + 0.01 * u
    z[50] += 6.0                      # single-column reflection spike
    z[120:124] -= 8.0                 # 4-column fragment at another depth
    pts = np.column_stack([u * 0.08, np.zeros_like(u), z])
    q, uk = clean_profile(pts, u, SweepConfig())
    assert 50 not in uk
    assert not np.any((uk >= 120) & (uk < 124))
    assert len(q) >= 190


def test_active_window_keeps_on_surface_span():
    t = np.arange(300) / 25.0
    counts = np.zeros(300)
    counts[60:240] = 150
    counts[10:14] = 40               # brief false start
    mask, info = active_window(t, counts, SweepConfig())
    assert mask[60] and mask[239] and not mask[30] and not mask[260]
    assert info["kept_profiles"] == 180


def test_offframe_flow_recovers_known_translation():
    rng = np.random.default_rng(1)
    tex = cv2.GaussianBlur(rng.uniform(20, 110, (1400, 1400)).astype(np.float32), (0, 0), 4)
    roi = np.zeros((600, 800), dtype=bool)
    roi[150:450, 200:600] = True
    cfg = SweepConfig()
    flow = OffFrameFlow((600, 800), roi, cfg)
    step_px = 1.3                     # scene content moves down (+v) by 1.3 px per OFF frame
    n = 40
    for k in range(n):
        M = np.float32([[1, 0, 0], [0, 1, step_px * k]])
        img = cv2.warpAffine(tex, M, (800, 600), flags=cv2.INTER_LINEAR) + rng.normal(0, 2, (600, 800)).astype(np.float32)
        flow.add(img, k * 0.04)
    dv = np.array([s.dv for s in flow.samples])
    assert np.mean([s.valid for s in flow.samples]) > 0.95
    # sub-pixel accurate and drift-free (phase correlation alone pixel-locks to ~1 px per step)
    assert dv.sum() == pytest.approx((n - 1) * step_px, rel=0.02)
    t, cum, info = flow_to_translation(flow.samples, np.array([0.0, 2.0]), np.array([80.0, 80.0]),
                                       933.3, 933.3, cfg)
    # content moving +v means the camera moved -Y: 39 x 1.3 px x 80 mm / 933.3 px = 4.35 mm
    assert cum[-1, 1] == pytest.approx(-(n - 1) * step_px * 80.0 / 933.3, rel=0.03)
    assert abs(cum[-1, 0]) < 0.15


def test_heightmap_bridges_profiles_spaced_wider_than_a_cell():
    """Regression: profiles 0.3 mm apart left empty rows and every row was deleted as an 'island'."""
    cfg = SweepConfig()
    xs = np.linspace(-8, 8, 200)
    pts = []
    for y in np.arange(0, 20, 0.3):
        z = 80.0 - 2.0 * np.exp(-((xs / 3) ** 2 + ((y - 10) / 3) ** 2))
        pts.append(np.column_stack([xs, np.full_like(xs, y), z]))
    P = np.vstack(pts)
    Z, valid, origin, info = build_heightmap(P, cfg)
    assert info["islands_removed"] == 0
    V, F, N = heightmap_to_mesh(Z, valid, origin, cfg)
    counts = component_face_counts(F)
    assert counts.size == 1
    true = 80.0 - 2.0 * np.exp(-((V[:, 0] / 3) ** 2 + ((V[:, 1] - 10) / 3) ** 2))
    assert np.median(np.abs(V[:, 2] - true)) < 0.03
    assert np.ptp(V[:, 1]) > 19.0
    assert N[:, 2].mean() < 0          # normals face the camera


def test_reconstruct_sweep_length_mode_scales_scan_axis():
    cfg = SweepConfig(translation="length", scan_length_mm=30.0)
    xs = np.linspace(-6, 6, 150)
    profiles = []
    for i, t in enumerate(np.arange(0, 6.0, 0.04)):
        z = 80.0 + 0.5 * np.sin(xs)
        profiles.append(Profile(i, float(t), np.column_stack([xs, np.zeros_like(xs), z]), np.arange(150.0)))
    res = reconstruct_sweep(profiles, cfg, camera_fx=933.3, camera_fy=933.3)
    assert res.report["translation"]["used"] == "length"
    assert np.ptp(res.vertices[:, 1]) == pytest.approx(30.0, abs=0.6)


def test_ridge_adaptive_guard_keeps_broad_line_but_rejects_real_reflection():
    H, W = 200, 60
    rows = np.arange(H)[:, None]
    img = 120.0 * np.exp(-((rows - 100) ** 2) / (2 * 6.0 ** 2)) * np.ones((1, W))   # broad / haloed line
    img[30:36, 40:50] += 110.0                                                      # separate glint
    fixed = RidgeConfig(adaptive_guard=False, max_secondary_ratio=0.45)
    adaptive = RidgeConfig(adaptive_guard=True, max_secondary_ratio=0.45)
    u_fixed, _, _ = extract_ridge_subpixel(img, fixed)
    u_adapt, v_adapt, _ = extract_ridge_subpixel(img, adaptive)
    assert len(u_fixed) < 10                 # the shoulder of the broad line was rejected
    assert len(u_adapt) == W - 10            # all clean columns kept ...
    assert not np.any((u_adapt >= 40) & (u_adapt < 50))   # ... glint columns still rejected
    assert np.allclose(v_adapt, 100.0, atol=0.2)


def test_sweep_pipeline_end_to_end_measures_hand_motion(tmp_path):
    from tests._synth_dental import make_dental_session

    info = make_dental_session(tmp_path / "s", t_idle=0.4, t_scan=4.0, mean_speed=4.0, seed=3)
    pipe = _load_pipeline()
    stats = pipe.run_pipeline(tmp_path / "s", tmp_path / "out")      # defaults: sweep + auto translation
    rep = json.loads((tmp_path / "out" / "pipeline_report.json").read_text())
    assert stats["translation_source"] == "flow"
    assert rep["active_window"]["t_start_s"] == pytest.approx(0.4, abs=0.1)
    true_travel = float(info["ty"].max() - info["ty"].min())
    assert rep["translation"]["extent_along_scan_mm"] == pytest.approx(true_travel, rel=0.08)
    assert stats["mesh_triangles"] > 20000
    from tests._plyread import read_ply_mesh
    V, F = read_ply_mesh(tmp_path / "out" / "reconstructed_mesh.ply")
    counts = component_face_counts(F)
    assert counts[0] / counts.sum() > 0.95          # one continuous surface
    assert (tmp_path / "out" / "reconstructed_mesh.stl").stat().st_size > 84


def test_auto_translation_falls_back_when_flow_is_unreliable():
    from ioscan.pipeline.sweep import FlowSample
    xs = np.linspace(-6, 6, 150)
    profiles = [Profile(i, i * 0.04, np.column_stack([xs, np.zeros_like(xs), np.full_like(xs, 80.0)]),
                        np.arange(150.0)) for i in range(100)]
    bad_flow = [FlowSample(i * 0.04 + 0.02, 0.0, 0.0, 0.0, False) for i in range(100)]
    res = reconstruct_sweep(profiles, SweepConfig(translation="auto", scan_length_mm=20.0),
                            camera_fx=933.3, camera_fy=933.3, flow_samples=bad_flow)
    assert res.report["translation"]["used"] == "length"
    res = reconstruct_sweep(profiles, SweepConfig(translation="auto", speed_mm_s=3.0),
                            camera_fx=933.3, camera_fy=933.3, flow_samples=bad_flow)
    assert res.report["translation"]["used"] == "speed"
    with pytest.raises(RuntimeError, match="Optical-flow translation failed"):
        reconstruct_sweep(profiles, SweepConfig(translation="flow"), camera_fx=933.3, camera_fy=933.3,
                          flow_samples=bad_flow)


def test_swapped_laser_flags_are_detected_and_corrected(tmp_path):
    """Hardware session scan_20261114_205403 recorded every ON/OFF flag one frame out of phase."""
    from tests._synth_dental import make_dental_session
    from ioscan.session.laser_phase import verify_laser_flags, laser_pair_polarity
    from ioscan.session.loader import load_frame_records, read_grayscale_frame

    make_dental_session(tmp_path / "s", t_idle=0.2, t_scan=1.2, seed=5)
    recs = load_frame_records(tmp_path / "s")
    kept, info = verify_laser_flags(recs)
    assert info["action"] == "kept"
    swapped = [dict(r, laser_on=not r["laser_on"]) for r in recs]
    fixed, info = verify_laser_flags(swapped)
    assert info["action"] == "inverted"
    assert [r["laser_on"] for r in fixed] == [r["laser_on"] for r in recs]
    on = read_grayscale_frame(recs[21]["path"])
    off = read_grayscale_frame(recs[20]["path"])
    assert laser_pair_polarity(on, off) > 20 and laser_pair_polarity(off, on) < -20


def test_planar_tracker_follows_model_rotation_behind_static_window():
    """Hardware: the scanner is fixed and the model is turned/slid under it; the window frame and
    streaks on the glass are static and must not pull the tracker toward 'no motion'."""
    from ioscan.pipeline.sweep import PlanarTracker, build_motion_mask
    rng = np.random.default_rng(4)
    # stone-model-like texture: blobs a few mm across (several pixels at tracking resolution)
    tex = cv2.GaussianBlur(rng.uniform(0, 255, (900, 900)).astype(np.float32), (0, 0), 6)
    tex = cv2.normalize(tex, None, 40, 220, cv2.NORM_MINMAX)
    H, W = 600, 800
    yy, xx = np.mgrid[0:H, 0:W]
    window = (xx > 150) & (xx < 650) & (yy > 100) & (yy < 520)

    def frame(theta_deg, tx, ty):
        M = cv2.getRotationMatrix2D((450, 450), theta_deg, 1.0)
        M[:, 2] += [tx - 50, ty - 150]
        img = cv2.warpAffine(tex, M, (W, H))
        img = np.where(window, img, 15.0)
        img[(np.abs(xx - 150) < 4) | (np.abs(xx - 650) < 4)] = 230.0      # static window frame
        img[(np.abs(yy - 300) < 2) & (xx > 300) & (xx < 500)] = 250.0     # streak on the glass
        return img.astype(np.uint8)

    n = 60
    path = [(0.5 * k, 1.2 * k, -0.8 * k) for k in range(n)]               # 29.5 deg, slides 70 px
    frames = [frame(*p) for p in path]
    small = [cv2.resize(f, (W // 4, H // 4), interpolation=cv2.INTER_AREA) for f in frames[::3]]
    mask, info = build_motion_mask(small)
    assert mask is not None
    cfg = SweepConfig()
    tr = PlanarTracker(mask, cfg)
    for k, f in enumerate(frames):
        tr.add(f, k * 0.04)
    assert np.mean([s.valid for s in tr.samples]) > 0.95
    A = tr.samples[-1].A
    ang = np.degrees(np.arctan2(A[1, 0], A[0, 0]))
    # OpenCV's angle is counter-clockwise in image coordinates; ECC's warp maps the first frame onto the last
    assert abs(abs(ang) - 0.5 * (n - 1)) < 1.0
