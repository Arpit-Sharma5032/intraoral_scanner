import numpy as np
from scipy.spatial.transform import Rotation

from ioscan.pipeline.pose import PoseConfig, PoseTracker


def _camera_rotation_for_tangent(theta: float) -> np.ndarray:
    """Camera +Y follows a circular path tangent; camera +Z looks down."""
    y_axis = np.array([np.cos(theta), np.sin(theta), 0.0])
    z_axis = np.array([0.0, 0.0, -1.0])
    x_axis = np.cross(y_axis, z_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def _wxyz_from_rotation(R: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(R).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def test_imu_hybrid_integrates_translation_along_curved_camera_axis():
    speed = 5.0
    radius = 30.0
    tracker = PoseTracker(
        PoseConfig(mode="imu_hybrid", nominal_speed_mm_s=speed),
        initial_timestamp_s=0.0,
    )

    errors = []
    for frame_index in range(1, 200, 2):
        t = frame_index / 50.0
        theta = speed * t / radius
        R = _camera_rotation_for_tangent(theta)
        T = tracker.update(frame_index, t, _wxyz_from_rotation(R))
        expected = np.array(
            [radius * np.sin(theta), radius * (1.0 - np.cos(theta)), 0.0]
        )
        errors.append(np.linalg.norm(T[:3, 3] - expected))

    assert max(errors) < 0.06
    assert np.linalg.norm(T[:3, 3]) > 15.0


def test_imu_const_vel_remains_world_fixed():
    tracker = PoseTracker(
        PoseConfig(mode="imu_const_vel", world_velocity_mm_s=(0.0, 5.0, 0.0)),
        initial_timestamp_s=0.0,
    )
    q = np.array([1.0, 0.0, 0.0, 0.0])
    tracker.update(1, 0.02, q)
    T = tracker.update(3, 0.06, q)
    assert np.allclose(T[:3, 3], [0.0, 0.3, 0.0])
