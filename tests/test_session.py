import numpy as np
import pytest

from ioscan.calibration.models import CameraIntrinsics, LaserPlane
from ioscan.session.format import SessionReader, SessionWriter


def _cam():
    return CameraIntrinsics(width=64, height=48, fx=100.0, fy=100.0, cx=32.0, cy=24.0)


def test_roundtrip(tmp_path):
    cam, plane = _cam(), LaserPlane(n=np.array([0.0, -0.8, -0.6]), d=7.5)
    rng = np.random.default_rng(0)
    imgs = [rng.integers(0, 1024, (48, 64)).astype(np.uint16) for _ in range(4)]
    with SessionWriter(tmp_path / "s", cam, plane, fps=50.0) as w:
        for i, im in enumerate(imgs):
            w.add_frame(im, t_ns=i * 20_000_000, laser_on=bool(i % 2))
        w.set_imu(np.array([0, 10_000_000]), np.array([[1.0, 0, 0, 0], [1.0, 0, 0, 0]]))
        w.set_gt_poses(np.arange(4) * 20_000_000, np.tile(np.eye(4), (4, 1, 1)))
        w.set_gt_profiles(np.array([1, 1]), np.zeros((2, 2)), np.ones((2, 3)))

    r = SessionReader(tmp_path / "s")
    assert len(r) == 4
    assert np.array_equal(r.laser_on, [False, True, False, True])
    assert np.array_equal(r.read_frame(2), imgs[2])
    assert r.on_off_pairs() == [(1, 0), (3, 2)]
    assert r.camera == cam
    assert np.allclose(r.laser_plane.n, plane.n / np.linalg.norm(plane.n))
    t, q = r.imu()
    assert t.dtype == np.int64 and q.shape == (2, 4)
    assert r.gt_poses()[1].shape == (4, 4, 4)
    assert r.gt_profiles()["P_c"].shape == (2, 3)


def test_rejects_wrong_dtype(tmp_path):
    with SessionWriter(tmp_path / "s", _cam(), LaserPlane(np.array([0, 1.0, 0]), 1.0), 50.0) as w:
        with pytest.raises(ValueError):
            w.add_frame(np.zeros((48, 64), dtype=np.uint8), 0, False)


def test_refuses_overwrite(tmp_path):
    (tmp_path / "s").mkdir()
    with pytest.raises(FileExistsError):
        SessionWriter(tmp_path / "s", _cam(), LaserPlane(np.array([0, 1.0, 0]), 1.0), 50.0)


def test_read_grayscale_frame_converts_color_jpeg(tmp_path):
    import cv2
    from ioscan.session.loader import read_grayscale_frame

    gray = np.full((24, 32), 127, dtype=np.uint8)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    path = tmp_path / "color_decoded.jpg"
    assert cv2.imwrite(str(path), bgr)

    loaded = read_grayscale_frame(path)
    assert loaded.shape == (24, 32)
    assert loaded.ndim == 2
    assert loaded.dtype == np.uint8
