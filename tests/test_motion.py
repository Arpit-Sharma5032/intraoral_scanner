import numpy as np

from ioscan.sim.motion import arch_centerline, canonicalize, make_trajectory
from tests.conftest import make_fake_arch


def test_centerline_follows_horseshoe():
    rng = np.random.default_rng(1)
    a = np.radians(rng.uniform(-100, 100, 6000))
    r = 25.0 + rng.uniform(-3, 3, a.size)
    xy = np.column_stack([r * np.cos(a), r * np.sin(a)])
    c = arch_centerline(xy)
    err = np.abs(np.hypot(c[:, 0], c[:, 1]) - 25.0)
    k = len(err) // 10
    assert err[k:-k].max() < 1.0  # interior: median noise from ~50 pts/bin, well inside the +-6 mm line coverage
    assert err.max() < 2.5        # the extreme end bins are biased by the cut face; harmless


def test_canonicalize_puts_arch_flat_and_occlusal_up():
    mesh, lab = make_fake_arch()
    mc, R, _ = canonicalize(mesh, lab)
    assert np.isclose(np.linalg.det(R), 1.0)
    ext = mc.extents
    assert ext[2] < ext[1] < ext[0]  # z thinnest
    V = np.asarray(mc.vertices)
    assert V[lab > 0, 2].mean() > V[lab == 0, 2].mean()  # teeth above gingiva


def test_trajectory_poses_are_rigid_and_look_down():
    mesh, lab = make_fake_arch()
    mc, _, _ = canonicalize(mesh, lab)
    T, info = make_trajectory(mc, lab, working_distance_mm=20.0, speed_mm_s=5.0, fps=50.0)
    Rm = T[:, :3, :3]
    assert np.allclose(np.einsum("nij,nkj->nik", Rm, Rm), np.eye(3), atol=1e-9)
    assert np.allclose(np.linalg.det(Rm), 1.0)
    assert np.allclose(T[:, :3, 2], [0, 0, -1])  # optical axis straight down
    step = np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1)
    assert abs(step.mean() - 5.0 / 50.0) < 0.01  # ~0.1 mm per frame


def test_ideal_imu_handles_single_frame_and_includes_last_timestamp():
    from ioscan.sim.motion import ideal_imu

    R = np.eye(3)[None, :, :]
    t, q = ideal_imu(np.array([123_000_000]), R, rate_hz=100.0)
    assert np.array_equal(t, [123_000_000])
    assert np.allclose(q, [[1.0, 0.0, 0.0, 0.0]])

    R2 = np.repeat(np.eye(3)[None, :, :], 3, axis=0)
    t2, q2 = ideal_imu(np.array([0, 20_000_000, 40_000_000]), R2, rate_hz=100.0)
    assert t2[0] == 0 and t2[-1] == 40_000_000
    assert q2.shape[0] == len(t2)
