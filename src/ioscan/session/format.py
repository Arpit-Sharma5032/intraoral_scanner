"""On-disk session format, shared by the simulator and the real capture logger.

Layout of one session directory:
    meta.json        format version, source ("sim" | "hardware"), camera, laser plane, fps, config
    frames.csv       idx, t_ns, laser_on, file
    frames/          000000.png ...  single-channel PNG (sim: uint16/10-bit; hardware: uint8/8-bit)
    imu.csv          t_ns, qw, qx, qy, qz     (unit quaternion, orientation of the camera)
    gt/              simulator only (never read by the pipeline itself, only by evaluation)
        poses.npz        t_ns (N,), T_wc (N,4,4)  camera-to-world transforms
        profiles.npz     frame (M,), uv (M,2), P_c (M,3)  true laser-line samples of ON frames
        reference_mesh.ply   the mesh the simulator scanned (world frame)

Notes
- Timestamps are integer nanoseconds from a common clock for frames and IMU.
- IMU orientation is only meaningful up to a fixed offset (the real BNO085 world frame is not
  the mesh frame), so consumers should use relative rotations R_i^T R_j.
- Frames alternate laser OFF / ON. ON frames are paired with an adjacent OFF frame.
"""
from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ioscan.calibration.models import CameraIntrinsics, LaserPlane

FORMAT_VERSION = 1


class SessionWriter:
    def __init__(
        self,
        root: str | Path,
        camera: CameraIntrinsics,
        laser_plane: LaserPlane,
        fps: float,
        source: str = "sim",
        config: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> None:
        self.root = Path(root)
        if self.root.exists():
            if not overwrite:
                raise FileExistsError(f"{self.root} exists (use overwrite=True)")
            shutil.rmtree(self.root)
        (self.root / "frames").mkdir(parents=True)
        (self.root / "gt").mkdir()
        self.camera = camera
        self.laser_plane = laser_plane
        self.fps = float(fps)
        self.source = source
        self.config = config or {}
        self._rows: list[tuple[int, int, int, str]] = []
        self._imu: tuple[np.ndarray, np.ndarray] | None = None
        self._poses: tuple[np.ndarray, np.ndarray] | None = None
        self._profiles: dict[str, np.ndarray] | None = None
        self._mesh: Any = None

    # ------------------------------------------------------------------ adding data
    def add_frame(self, image: np.ndarray, t_ns: int, laser_on: bool) -> int:
        if image.dtype != np.uint16 or image.shape != (self.camera.height, self.camera.width):
            raise ValueError(
                f"Frame must be uint16 {(self.camera.height, self.camera.width)}, "
                f"got {image.dtype} {image.shape}"
            )
        idx = len(self._rows)
        rel = f"frames/{idx:06d}.png"
        if not cv2.imwrite(str(self.root / rel), image):
            raise IOError(f"Failed to write {rel}")
        self._rows.append((idx, int(t_ns), int(bool(laser_on)), rel))
        return idx

    def set_imu(self, t_ns: np.ndarray, quat_wxyz: np.ndarray) -> None:
        t = np.asarray(t_ns, dtype=np.int64).reshape(-1)
        q = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1, 4)
        if len(t) != len(q):
            raise ValueError("IMU timestamps and quaternions must have the same length")
        norms = np.linalg.norm(q, axis=1)
        if np.any(~np.isfinite(norms)) or np.any(norms < 1e-12):
            raise ValueError("IMU quaternions must be finite and non-zero")
        self._imu = (t, q / norms[:, None])

    def set_gt_poses(self, t_ns: np.ndarray, T_wc: np.ndarray) -> None:
        self._poses = (np.asarray(t_ns, dtype=np.int64), np.asarray(T_wc, dtype=np.float64))

    def set_gt_profiles(self, frame: np.ndarray, uv: np.ndarray, P_c: np.ndarray) -> None:
        self._profiles = {
            "frame": np.asarray(frame, dtype=np.int32),
            "uv": np.asarray(uv, dtype=np.float64),
            "P_c": np.asarray(P_c, dtype=np.float64),
        }

    def set_reference_mesh(self, mesh: Any) -> None:
        self._mesh = mesh

    # ------------------------------------------------------------------ finalize
    def close(self) -> None:
        with open(self.root / "frames.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "t_ns", "laser_on", "file"])
            w.writerows(self._rows)

        if self._imu is not None:
            t, q = self._imu
            np.savetxt(
                self.root / "imu.csv",
                np.column_stack([t, q]),
                delimiter=",",
                header="t_ns,qw,qx,qy,qz",
                comments="",
                fmt=["%d", "%.9f", "%.9f", "%.9f", "%.9f"],
            )
        if self._poses is not None:
            np.savez_compressed(self.root / "gt" / "poses.npz", t_ns=self._poses[0], T_wc=self._poses[1])
        if self._profiles is not None:
            np.savez_compressed(self.root / "gt" / "profiles.npz", **self._profiles)
        if self._mesh is not None:
            self._mesh.export(str(self.root / "gt" / "reference_mesh.ply"))

        meta = {
            "format_version": FORMAT_VERSION,
            "source": self.source,
            "created": datetime.now(timezone.utc).isoformat(),
            "fps": self.fps,
            "num_frames": len(self._rows),
            "image": {"width": self.camera.width, "height": self.camera.height, "dtype": "uint16", "bit_depth": 10},
            "camera": self.camera.to_dict(),
            "laser_plane": self.laser_plane.to_dict(),
            "config": self.config,
        }
        with open(self.root / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    def __enter__(self) -> "SessionWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:  # do not finalize a half-written session
            self.close()


class SessionReader:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        with open(self.root / "meta.json", "r") as f:
            self.meta: dict[str, Any] = json.load(f)
        if self.meta["format_version"] != FORMAT_VERSION:
            raise ValueError(f"Unsupported session format version {self.meta['format_version']}")
        rows = []
        with open(self.root / "frames.csv", "r", newline="") as f:
            for r in csv.DictReader(f):
                rows.append((int(r["idx"]), int(r["t_ns"]), int(r["laser_on"]), r["file"]))
        self._files = [r[3] for r in rows]
        self.t_ns = np.array([r[1] for r in rows], dtype=np.int64)
        self.laser_on = np.array([bool(r[2]) for r in rows])

    def __len__(self) -> int:
        return len(self._files)

    @property
    def camera(self) -> CameraIntrinsics:
        return CameraIntrinsics.from_dict(self.meta["camera"])

    @property
    def laser_plane(self) -> LaserPlane:
        return LaserPlane.from_dict(self.meta["laser_plane"])

    @property
    def fps(self) -> float:
        return float(self.meta["fps"])

    def read_frame(self, idx: int) -> np.ndarray:
        img = cv2.imread(str(self.root / self._files[idx]), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Cannot read frame {idx}")
        return img

    def on_off_pairs(self) -> list[tuple[int, int]]:
        """(on_idx, off_idx) pairs: each ON frame with the adjacent OFF frame (previous first)."""
        pairs = []
        n = len(self)
        for i in range(n):
            if not self.laser_on[i]:
                continue
            if i > 0 and not self.laser_on[i - 1]:
                pairs.append((i, i - 1))
            elif i + 1 < n and not self.laser_on[i + 1]:
                pairs.append((i, i + 1))
        return pairs

    def imu(self) -> tuple[np.ndarray, np.ndarray]:
        data = np.atleast_2d(np.loadtxt(self.root / "imu.csv", delimiter=",", skiprows=1))
        if data.shape[1] < 5:
            raise ValueError(f"Expected t_ns,qw,qx,qy,qz in {self.root / 'imu.csv'}")
        return data[:, 0].astype(np.int64), data[:, 1:5]

    def gt_poses(self) -> tuple[np.ndarray, np.ndarray]:
        d = np.load(self.root / "gt" / "poses.npz")
        return d["t_ns"], d["T_wc"]

    def gt_profiles(self) -> dict[str, np.ndarray]:
        d = np.load(self.root / "gt" / "profiles.npz")
        return {k: d[k] for k in d.files}
