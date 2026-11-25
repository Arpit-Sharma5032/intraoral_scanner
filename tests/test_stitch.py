"""Multi-pass stitching: coarse correlation, relief agreement, full synthetic arch vs ground truth."""
from __future__ import annotations

import json

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from ioscan.pipeline.quality import component_face_counts
from ioscan.pipeline.stitch import (
    Patch, StitchConfig, _FixedSpectra, _T, apply, fuse_patches, masked_ncc, relief_agreement,
    rot_angle_deg, stitch_patches, stitch_to_files,
)
from ioscan.pipeline.sweep import write_ply_points
from tests._synth_arch import make_passes


def test_masked_ncc_recovers_known_shift():
    rng = np.random.default_rng(0)
    F = np.zeros((80, 90))
    F[10:70, 10:80] = rng.normal(size=(60, 70))
    Mf = np.zeros_like(F, dtype=bool); Mf[10:70, 10:80] = True
    G = F[30:60, 40:75].copy(); Mg = np.ones_like(G, dtype=bool)
    fs = _FixedSpectra(F, Mf, (F.shape[0] + G.shape[0], F.shape[1] + G.shape[1]))
    ncc, ov = masked_ncc(fs, G, Mg, 100)
    iy, ix = np.unravel_index(np.argmax(ncc), ncc.shape)
    assert (iy - (G.shape[0] - 1), ix - (G.shape[1] - 1)) == (30, 40)
    assert ncc[iy, ix] > 0.99


def test_relief_agreement_rejects_flat_base_fits():
    passes, _, _ = make_passes(n_passes=4, seed=1, step=0.25)
    (P0, R0, t0), (P1, R1, t1) = passes[0], passes[1]
    T_true = _T(R0.T @ R1, R0.T @ (t1 - t0))
    good, n_good = relief_agreement(P0, apply(T_true, P1))
    flipped = _T(Rotation.from_euler("z", 180, degrees=True).as_matrix()) @ T_true
    bad, _ = relief_agreement(P0, apply(flipped, P1))
    assert good > 0.9 and good - bad > 0.2


def test_stitched_synthetic_arch_matches_ground_truth(tmp_path):
    passes, gt, _ = make_passes(n_passes=5, seed=3, step=0.25)
    # IMU orientations (only the rotation between passes is used, as a cross-check)
    patches = []
    for k, (P, R, t) in enumerate(passes):
        q = Rotation.from_matrix(R).as_quat()
        patches.append(Patch(f"p{k}", P, np.array([q[3], q[0], q[1], q[2]])))
    Ts, rep = stitch_patches(patches, StitchConfig())
    R0, t0 = passes[0][1], passes[0][2]
    for k, (P, R, t) in enumerate(passes):
        assert rot_angle_deg((R0.T @ R).T @ Ts[k][:3, :3]) < 1.5
    assert not [w for w in rep["warnings"] if "IMU says" in w]
    V, F, N, cloud, info = fuse_patches(patches, Ts)
    counts = component_face_counts(F)
    assert counts[0] / counts.sum() > 0.97
    Rc, mu = np.array(info["canonical_R"]), np.array(info["canonical_mu"])
    d, _ = cKDTree(gt).query(((V @ Rc + mu) @ R0.T) + t0)
    assert np.median(d) < 0.25
    # canonical frame: teeth toward +Z, incisors (narrow end of the U) toward +Y
    assert N[:, 2].mean() > 0
    top, bottom = V[V[:, 1] > np.percentile(V[:, 1], 88)], V[V[:, 1] < np.percentile(V[:, 1], 12)]
    assert np.ptp(top[:, 0]) < np.ptp(bottom[:, 0])


def test_stitch_to_files_writes_pipeline_compatible_outputs(tmp_path):
    passes, _, _ = make_passes(n_passes=3, seed=2, step=0.3)
    dirs = []
    for k, (P, _, _) in enumerate(passes):
        d = tmp_path / f"pass{k}"
        d.mkdir()
        write_ply_points(d / "fused_cloud.ply", P)
        (d / "pipeline_report.json").write_text(json.dumps({"pose_mode": "sweep"}))
        dirs.append(d)
    rep = stitch_to_files(dirs, tmp_path / "arch")
    for f in ("reconstructed_mesh.ply", "reconstructed_mesh.stl", "fused_cloud.ply", "pipeline_report.json"):
        assert (tmp_path / "arch" / f).is_file()
    assert rep["mesh_triangles"] > 10000 and len(rep["links"]) == 2
