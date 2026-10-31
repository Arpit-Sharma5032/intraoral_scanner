"""Session loading for the reconstruction pipeline (sim and hardware sessions).

Two on-disk layouts are understood:

* shared format (simulator and the capture logger): ``meta.json``, ``frames.csv`` with
  ``idx,t_ns,laser_on,file`` and ``imu.csv`` with ``t_ns,qw,qx,qy,qz``;
* legacy hardware layout: ``frames.csv`` with ``index,timestamp_s,laser_on,qw,qx,qy,qz,filename``.

Nothing is invented: a missing calibration or missing IMU orientation raises an error instead of
falling back to nominal values or to a made-up quaternion.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from ioscan.calibration.models import CameraIntrinsics, LaserPlane


def load_calibration(session_dir: str | Path) -> tuple[CameraIntrinsics, LaserPlane, bool]:
    """Return (camera, laser_plane, is_nominal). Raises FileNotFoundError if none is stored."""
    root = Path(session_dir)
    meta_path = root / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if "camera" in meta and "laser_plane" in meta:
            return (
                CameraIntrinsics.from_dict(meta["camera"]),
                LaserPlane.from_dict(meta["laser_plane"]),
                bool(meta.get("calibration_is_nominal", False)),
            )
    cal_file = root / "calibration.json"
    if cal_file.exists():
        data = json.loads(cal_file.read_text(encoding="utf-8"))
        cam = CameraIntrinsics.from_dict(data.get("camera", data))
        laser = LaserPlane.from_dict(data.get("laser", data.get("laser_plane", {})))
        return cam, laser, bool(data.get("is_nominal", False))
    raise FileNotFoundError(
        f"No calibration found in {root} (need meta.json with camera/laser_plane or calibration.json)."
    )


def interpolate_imu(t_query_s: np.ndarray, t_imu_s: np.ndarray, q_wxyz: np.ndarray) -> np.ndarray:
    """Slerp IMU quaternions (WXYZ) to the query times; queries outside the IMU span are clamped."""
    t_imu_s = np.asarray(t_imu_s, dtype=np.float64)
    q = np.asarray(q_wxyz, dtype=np.float64).reshape(-1, 4)
    order = np.argsort(t_imu_s, kind="stable")
    t_imu_s, q = t_imu_s[order], q[order]
    keep = np.concatenate([[True], np.diff(t_imu_s) > 0])  # Slerp needs strictly increasing times
    t_imu_s, q = t_imu_s[keep], q[keep]
    if len(t_imu_s) == 0:
        raise ValueError("IMU stream is empty")
    xyzw = np.column_stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]])
    tq = np.asarray(t_query_s, dtype=np.float64)
    if len(t_imu_s) == 1:
        out = np.repeat(xyzw, len(tq), axis=0)
    else:
        slerp = Slerp(t_imu_s, Rotation.from_quat(xyzw))
        out = slerp(np.clip(tq, t_imu_s[0], t_imu_s[-1])).as_quat()
    return np.column_stack([out[:, 3], out[:, 0], out[:, 1], out[:, 2]])


def _truthy(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def load_frame_records(session_dir: str | Path) -> list[dict]:
    """Per-frame records: index, timestamp_s, laser_on, path, quat_wxyz (or None)."""
    root = Path(session_dir)
    csv_path = root / "frames.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{csv_path} has no rows")

    cols = set(rows[0].keys())
    records: list[dict] = []
    if {"idx", "t_ns", "laser_on", "file"} <= cols:
        for r in rows:
            records.append({
                "index": int(r["idx"]),
                "timestamp_s": int(r["t_ns"]) * 1e-9,
                "laser_on": _truthy(r["laser_on"]),
                "path": root / r["file"],
                "quat_wxyz": None,
            })
        imu_path = root / "imu.csv"
        if imu_path.exists():
            data = np.atleast_2d(np.loadtxt(imu_path, delimiter=",", skiprows=1))
            if data.shape[1] >= 5 and len(data) > 0:
                t_f = np.array([rec["timestamp_s"] for rec in records])
                q = interpolate_imu(t_f, data[:, 0] * 1e-9, data[:, 1:5])
                for rec, qi in zip(records, q):
                    rec["quat_wxyz"] = qi
    elif {"index", "timestamp_s", "laser_on", "filename"} <= cols:
        has_q = all(k in cols for k in ("qw", "qx", "qy", "qz"))
        for r in rows:
            q = None
            if has_q:
                q = np.array([float(r["qw"]), float(r["qx"]), float(r["qy"]), float(r["qz"])])
            records.append({
                "index": int(r["index"]),
                "timestamp_s": float(r["timestamp_s"]),
                "laser_on": _truthy(r["laser_on"]),
                "path": root / "frames" / Path(r["filename"]).name,
                "quat_wxyz": q,
            })
    else:
        raise ValueError(
            f"Unrecognised frames.csv columns {sorted(cols)}; expected 'idx,t_ns,laser_on,file' "
            "or 'index,timestamp_s,laser_on,qw,qx,qy,qz,filename'."
        )

    ts = np.array([r["timestamp_s"] for r in records])
    if np.any(np.diff(ts) < 0):
        warnings.warn("Frame timestamps are not monotonic; sorting by time.", stacklevel=2)
        records.sort(key=lambda r: r["timestamp_s"])
    return records


def load_gt_poses(session_dir: str | Path) -> dict[int, np.ndarray]:
    """Simulator ground-truth camera-to-world poses {frame_index: 4x4}; empty if unavailable."""
    p = Path(session_dir) / "gt" / "poses.npz"
    if not p.exists():
        return {}
    d = np.load(p)
    if "T_wc" not in d.files:
        return {}
    T = np.asarray(d["T_wc"], dtype=np.float64)
    return {i: T[i] for i in range(len(T))}


def session_source(session_dir: str | Path) -> str:
    meta = Path(session_dir) / "meta.json"
    if meta.exists():
        return str(json.loads(meta.read_text(encoding="utf-8")).get("source", "unknown"))
    return "unknown"
