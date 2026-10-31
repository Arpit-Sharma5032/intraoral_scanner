"""Session loading: IMU interpolation, both frames.csv layouts, no invented orientation."""
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from ioscan.pipeline.pose import PoseConfig, PoseTracker
from ioscan.session.loader import interpolate_imu, load_calibration, load_frame_records, load_gt_poses

SESSION = Path(__file__).resolve().parents[1] / "data" / "sessions" / "FUYJPUPX_lower"


def _wxyz(rot: Rotation) -> np.ndarray:
    x, y, z, w = rot.as_quat()
    return np.array([w, x, y, z])


def test_interpolate_imu_midpoint_and_clamp():
    q0, q1 = Rotation.identity(), Rotation.from_euler("z", 90, degrees=True)
    q = np.array([_wxyz(q0), _wxyz(q1)])
    out = interpolate_imu(np.array([-1.0, 0.5, 2.0]), np.array([0.0, 1.0]), q)
    mid = Rotation.from_quat([out[1][1], out[1][2], out[1][3], out[1][0]])
    assert mid.magnitude() == pytest.approx(np.radians(45), abs=1e-6)
    assert np.allclose(out[0], q[0]) and np.allclose(out[2], q[1])


def test_shared_format_uses_t_ns_and_imu_csv(tmp_path):
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames.csv").write_text("idx,t_ns,laser_on,file\n0,0,0,frames/a.png\n1,25000000,1,frames/b.png\n")
    (tmp_path / "imu.csv").write_text("t_ns,qw,qx,qy,qz\n0,1,0,0,0\n50000000,0.7071068,0,0,0.7071068\n")
    rec = load_frame_records(tmp_path)
    assert rec[1]["timestamp_s"] == pytest.approx(0.025)          # t_ns honoured, not idx*0.02
    assert rec[1]["laser_on"] and rec[1]["path"] == tmp_path / "frames/b.png"
    r = Rotation.from_quat([rec[1]["quat_wxyz"][1], rec[1]["quat_wxyz"][2], rec[1]["quat_wxyz"][3], rec[1]["quat_wxyz"][0]])
    assert r.magnitude() == pytest.approx(np.radians(45), abs=1e-3)


def test_missing_imu_gives_none_and_tracker_refuses(tmp_path):
    (tmp_path / "frames.csv").write_text("idx,t_ns,laser_on,file\n0,0,1,frames/a.png\n")
    rec = load_frame_records(tmp_path)
    assert rec[0]["quat_wxyz"] is None
    with pytest.raises(ValueError):
        PoseTracker(PoseConfig(mode="imu_hybrid")).update(0, 0.0, rec[0]["quat_wxyz"])


def test_legacy_layout(tmp_path):
    (tmp_path / "frames.csv").write_text(
        "index,timestamp_s,laser_on,qw,qx,qy,qz,filename\n3,0.5,1,1,0,0,0,000003.png\n")
    r = load_frame_records(tmp_path)[0]
    assert r["index"] == 3 and r["timestamp_s"] == 0.5 and r["path"] == tmp_path / "frames" / "000003.png"


def test_unknown_layout_rejected(tmp_path):
    (tmp_path / "frames.csv").write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        load_frame_records(tmp_path)


def test_no_calibration_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_calibration(tmp_path)


@pytest.mark.skipif(not SESSION.exists(), reason="sample session not present")
def test_sample_session_pose_matches_ground_truth():
    rec = load_frame_records(SESSION)
    gt = load_gt_poses(SESSION)
    tr = PoseTracker(PoseConfig(mode="imu_hybrid", initial_position_mm=tuple(gt[0][:3, 3])),
                     initial_timestamp_s=rec[0]["timestamp_s"])
    err = [np.linalg.norm(tr.update(r["index"], r["timestamp_s"], r["quat_wxyz"])[:3, 3] - gt[r["index"]][:3, 3])
           for r in rec if r["laser_on"]]
    assert max(err) < 0.01  # mm
