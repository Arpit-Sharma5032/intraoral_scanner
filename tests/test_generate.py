"""End-to-end: fake Teeth3DS dataset -> simulated session -> read back and check vs GT."""
import numpy as np
import pytest

pytest.importorskip("open3d")

from ioscan.session.format import SessionReader
from ioscan.sim.generate import generate_session


def test_generate_and_read_back(fake_dataset, tmp_path):
    out = tmp_path / "sess"
    info = generate_session(out, root=fake_dataset, jaw="lower", case_id="FAKE0001",
                            max_frames=60, progress=False)
    assert info["frames"] == 60 and info["gt_points"] > 10_000

    r = SessionReader(out)
    assert len(r) == 60
    assert r.laser_on.sum() == 30
    assert (out / "gt" / "trajectory.png").exists()
    assert (out / "gt" / "reference_mesh.ply").exists()

    # the OFF frame is flat ambient; the ON frame with the longest line shows it clearly.
    # (Frames at the very start may see no line yet: the camera begins beyond the arch end.)
    pairs = r.on_off_pairs()
    assert len(pairs) == 30
    prof = r.gt_profiles()
    counts = np.bincount(prof["frame"], minlength=len(r))
    best_on, best_off = max(pairs, key=lambda pr: counts[pr[0]])
    on, off = r.read_frame(best_on), r.read_frame(best_off)
    assert off.min() == off.max()
    assert (on.astype(int) - off.astype(int)).max() > 100

    # GT samples lie on the stored laser plane and have positive depth
    plane = r.laser_plane
    assert np.max(np.abs(prof["P_c"] @ plane.n + plane.d)) < 1e-3
    assert prof["P_c"][:, 2].min() > 5.0

    # IMU: unit quaternions, increasing timestamps
    t, q = r.imu()
    assert np.all(np.diff(t) > 0)
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-6)
