from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.spatial.transform import Rotation


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert a [qw, qx, qy, qz] quaternion to a rotation matrix."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("IMU quaternion must be finite and non-zero")
    w, x, y, z = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Pack camera-to-world rotation and translation into a 4x4 transform."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def transform_points(pts_cam: np.ndarray, T_wc: np.ndarray) -> np.ndarray:
    """Transform points from camera coordinates to world coordinates (millimetres)."""
    pts = np.asarray(pts_cam, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)
    return (pts @ T_wc[:3, :3].T) + T_wc[:3, 3]


@dataclass
class PoseConfig:
    """Pose prior for a line scanner.

    The scanner's IMU reports camera-to-world orientation. Translation is not observable
    from a gyroscope-only stream, so the hybrid mode integrates a nominal scan speed along
    the configured camera-frame scan axis. For this project the scan axis is camera +Y.
    """

    mode: str = "imu_hybrid"  # imu_hybrid | imu_const_vel | gt | submap_icp (legacy alias)
    nominal_speed_mm_s: float = 5.0
    scan_axis_camera: tuple[float, float, float] = (0.0, 1.0, 0.0)
    world_velocity_mm_s: tuple[float, float, float] = (0.0, 5.0, 0.0)
    initial_position_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)


class PoseTracker:
    """Estimate camera-to-world poses from IMU orientation and a translation prior.

    Important: a single laser profile is a 1-D curve, not a full 3-D scan. Rigid ICP between
    consecutive profiles (or a profile and a growing map) is under-constrained and can pull
    adjacent profiles onto one another. The old implementation did exactly that, producing
    folds/ribbons. The hybrid mode therefore keeps the measured IMU rotation and advances along
    the camera-frame scan axis using a configurable speed prior. Use an encoder/visual odometry
    source for measured translation when available.
    """

    def __init__(
        self,
        cfg: PoseConfig | None = None,
        *,
        initial_timestamp_s: float | None = None,
    ) -> None:
        self.cfg = cfg or PoseConfig()
        if self.cfg.mode == "submap_icp":
            warnings.warn(
                "pose mode 'submap_icp' is deprecated: rigid ICP is not reliable for a single "
                "laser profile. Using the constrained imu_hybrid mode instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.cfg.mode = "imu_hybrid"

        if self.cfg.mode not in {"imu_hybrid", "imu_const_vel", "gt"}:
            raise ValueError(f"Unsupported pose mode: {self.cfg.mode}")

        axis = np.asarray(self.cfg.scan_axis_camera, dtype=np.float64).reshape(3)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-12:
            raise ValueError("scan_axis_camera must be a non-zero vector")
        self._scan_axis_camera = axis / axis_norm
        self._world_velocity = np.asarray(self.cfg.world_velocity_mm_s, dtype=np.float64).reshape(3)
        self._position = np.asarray(self.cfg.initial_position_mm, dtype=np.float64).reshape(3)
        self._last_time_s = None if initial_timestamp_s is None else float(initial_timestamp_s)
        self._last_rotation: np.ndarray | None = None
        self.poses: dict[int, np.ndarray] = {}

    def update(
        self,
        frame_index: int,
        timestamp_s: float,
        quat_wxyz: np.ndarray | None,
        pts_cam: np.ndarray | None = None,
        gt_T_wc: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the estimated camera-to-world pose for one laser-ON frame."""
        if self.cfg.mode == "gt":
            if gt_T_wc is None:
                raise ValueError(
                    "pose mode 'gt' requires ground-truth poses; use --pose-mode imu_hybrid "
                    "for a real capture"
                )
            T = np.asarray(gt_T_wc, dtype=np.float64).reshape(4, 4).copy()
            self.poses[int(frame_index)] = T
            self._position = T[:3, 3].copy()
            self._last_time_s = float(timestamp_s)
            return T

        if quat_wxyz is None:
            raise ValueError(
                "No IMU quaternion is available. Check the session's imu.csv or frame metadata."
            )

        timestamp_s = float(timestamp_s)
        if not np.isfinite(timestamp_s):
            raise ValueError("Frame timestamp must be finite")
        R_wc = quat_wxyz_to_rotmat(quat_wxyz)

        if self._last_time_s is not None:
            dt = timestamp_s - self._last_time_s
            if dt < -1e-9:
                raise ValueError("Frame timestamps must be non-decreasing")
            dt = max(dt, 0.0)
            if self.cfg.mode == "imu_const_vel":
                delta = self._world_velocity * dt
            else:
                direction_now = R_wc @ self._scan_axis_camera
                if self._last_rotation is None:
                    # No orientation was recorded at the initial timestamp (typically an OFF
                    # frame), so use the first available IMU orientation for this short step.
                    direction_avg = direction_now
                else:
                    direction_prev = self._last_rotation @ self._scan_axis_camera
                    # Trapezoidal integration is substantially less biased than advancing along
                    # only the end-of-interval orientation while following an arch.
                    direction_avg = 0.5 * (direction_prev + direction_now)
                delta = direction_avg * self.cfg.nominal_speed_mm_s * dt
            self._position = self._position + delta
        self._last_time_s = timestamp_s
        self._last_rotation = R_wc.copy()

        T = make_transform(R_wc, self._position.copy())
        self.poses[int(frame_index)] = T
        return T
