"""Synthetic hardware-format session of a dental quadrant scanned by hand (no Open3D needed).

Matches the real IOSCAN geometry (19 mm baseline, 11 deg tilt, ~80 mm working distance, the
operator ROI) and reproduces the failure modes seen on hardware:
- hand speed is NOT constant (2-7 mm/s) and the hand drifts sideways,
- the laser is off the surface for a while at the start and end of the capture,
- stray reflections add secondary bright blobs to some laser-ON frames,
- OFF frames carry ambient texture, so translation can be measured from them.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ioscan.calibration.models import CameraIntrinsics, LaserPlane

FPS = 50.0
BASE_Z = 84.0           # gingiva / model base depth (mm from the camera)
TEX_Z = 80.0            # depth at which the ambient texture is rendered (median surface depth)

# Operator ROI from the hardware session (normalised corners)
HW_ROI = [[0.4203236149587233, 0.45672709672766615], [0.6905314014343438, 0.46597919369356705],
          [0.6882217916126875, 0.6263493058074341], [0.4203236149587233, 0.6325176449605583]]

# tooth centres along the scan axis (y, mm), buccolingual half width, mesiodistal half length
TEETH = [(6.0, 4.6, 3.6, 7.0), (14.0, 4.8, 3.8, 7.2), (23.5, 5.4, 5.0, 7.6), (34.5, 5.6, 5.3, 7.4)]


def dental_height(x, y):
    """Height above the base (mm) of a lower posterior quadrant: 2 premolars + 2 molars."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    h = 1.2 * np.exp(-(x / 9.0) ** 2)                         # alveolar ridge / gingiva
    for cy, hw, hl, top in TEETH:
        r = np.sqrt((x / hw) ** 4 + ((y - cy) / hl) ** 4) ** 0.5  # rounded-box crown footprint
        crown = top * np.clip(1.0 - r ** 6, 0.0, 1.0) ** 0.25
        # cusps: two buccal + two lingual bumps, a central fossa groove
        cusp = 0.9 * (np.exp(-(((x - 0.45 * hw) / 1.6) ** 2 + ((y - cy - 0.4 * hl) / 1.6) ** 2))
                      + np.exp(-(((x - 0.45 * hw) / 1.6) ** 2 + ((y - cy + 0.4 * hl) / 1.6) ** 2))
                      + np.exp(-(((x + 0.45 * hw) / 1.6) ** 2 + ((y - cy - 0.4 * hl) / 1.6) ** 2))
                      + np.exp(-(((x + 0.45 * hw) / 1.6) ** 2 + ((y - cy + 0.4 * hl) / 1.6) ** 2)))
        fossa = 0.8 * np.exp(-(x / 1.0) ** 2) * (r < 0.8)
        h = np.maximum(h, np.where(r < 1.0, crown + cusp - fossa, 0.0))
    return h


def surface_z(x, y):
    return BASE_Z - dental_height(x, y)


def hand_motion(t: np.ndarray, t_idle: float, t_scan: float, mean_speed: float):
    """Camera translation (tx, ty) in mm: idle, a varying-speed pass, idle."""
    t = np.asarray(t, dtype=np.float64)
    ts = np.clip(t - t_idle, 0.0, t_scan)
    # speed v = mean * (1 + 0.55 sin) -> integrate analytically
    w = 2 * np.pi / 3.7
    ty = mean_speed * (ts + 0.55 * (1 - np.cos(w * ts)) / w) - 4.0
    tx = 1.6 * np.sin(2 * np.pi * ts / 6.5)
    return tx, ty


def make_dental_session(root: Path, *, t_idle: float = 0.8, t_scan: float = 9.0, mean_speed: float = 4.6,
                        roi_points=HW_ROI, reflections: bool = True, seed: int = 0,
                        jpeg: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    root = Path(root)
    (root / "frames").mkdir(parents=True, exist_ok=True)
    cam = CameraIntrinsics(1600, 1200, 933.3, 933.3, 800.0, 600.0)
    laser = LaserPlane.from_baseline_and_angle(19.0, 11.0, "y")
    ny, nz, d = float(laser.normal[1]), float(laser.normal[2]), float(laser.d)
    n_frames = int(round((2 * t_idle + t_scan) * FPS))

    tex = cv2.GaussianBlur(rng.uniform(20, 110, (2600, 2600)).astype(np.float32), (0, 0), 5)
    uu, vv = np.meshgrid(np.arange(1600, dtype=np.float32), np.arange(1200, dtype=np.float32))
    xs = np.linspace(-30, 30, 5000)
    t_all = np.arange(n_frames) / FPS
    TX, TY = hand_motion(t_all, t_idle, t_scan, mean_speed)
    roi_px = np.asarray(roi_points) * [1600, 1200] if roi_points else None

    rows = []
    ext = ".jpg" if jpeg else ".png"
    for i in range(n_frames):
        t = t_all[i]
        tx, ty = TX[i], TY[i]
        laser_on = (i % 2 == 1)
        mx = (((uu - cam.cx) / cam.fx * TEX_Z + tx) * 10.0 + 1300).astype(np.float32)
        my = (((vv - cam.cy) / cam.fy * TEX_Z + ty) * 10.0 + 600).astype(np.float32)
        img = cv2.remap(tex, mx, my, cv2.INTER_LINEAR).astype(np.float32)
        on_surface = t_idle <= t <= t_idle + t_scan
        if laser_on and on_surface:
            y = np.full_like(xs, ty)
            for _ in range(12):  # fixed point of plane/surface intersection (world coords)
                zz = surface_z(xs + tx, y)
                y = ty + (-nz * zz - d) / ny
            zz = surface_z(xs + tx, y)
            u = cam.fx * xs / zz + cam.cx
            v = cam.fy * (y - ty) / zz + cam.cy
            ok = (u >= 0) & (u < 1599) & (v > 8) & (v < 1190)
            u, v = u[ok], v[ok]
            order = np.argsort(u)
            cols = np.arange(int(np.ceil(u.min())), int(np.floor(u.max())) + 1)
            vc = np.interp(cols, u[order], v[order])
            amp = 90.0 + 40.0 * rng.random()
            for dr in range(-8, 9):
                r = np.rint(vc).astype(int) + dr
                good = (r >= 0) & (r < 1200)
                img[r[good], cols[good]] += amp * np.exp(-((r[good] - vc[good]) ** 2) / (2 * 1.8 ** 2))
            if reflections and roi_px is not None and rng.random() < 0.3:
                for _ in range(rng.integers(1, 4)):
                    cx_ = rng.uniform(roi_px[:, 0].min(), roi_px[:, 0].max())
                    cy_ = rng.uniform(roi_px[:, 1].min(), roi_px[:, 1].max())
                    rad = rng.uniform(3, 9)
                    img += rng.uniform(60, 140) * np.exp(-((uu - cx_) ** 2 + (vv - cy_) ** 2) / (2 * rad ** 2))
        img += rng.normal(0, 2.0, img.shape)
        fn = f"frames/{i:05d}{ext}"
        cv2.imwrite(str(root / fn), np.clip(img, 0, 255).astype(np.uint8))
        rows.append((i, int(round(t * 1e9)), int(laser_on), fn))

    with open(root / "frames.csv", "w", newline="") as f:
        f.write("idx,t_ns,laser_on,file\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]}\n")
    # IMU with a slow wobble (the sweep reconstruction must not depend on it)
    with open(root / "imu.csv", "w") as f:
        f.write("t_ns,qw,qx,qy,qz\n")
        for i in range(n_frames + 2):
            a = np.radians(4.0) * np.sin(i / FPS)
            f.write(f"{int(i * 1e9 / FPS)},{np.cos(a / 2)},{np.sin(a / 2)},0,0\n")
    (root / "meta.json").write_text(json.dumps({
        "source": "hardware", "camera": cam.to_dict(), "laser_plane": laser.to_dict(),
        "calibration_is_nominal": True, "requested_fps": FPS, "fps": FPS, "jaw": "lower",
    }))
    roi = {"version": 1, "width": 1600, "height": 1200, "enabled": roi_points is not None,
           "points": roi_points or []}
    (root / "roi.json").write_text(json.dumps(roi))
    return {"camera": cam, "laser": laser, "tx": TX, "ty": TY, "t": t_all,
            "t_idle": t_idle, "t_scan": t_scan, "mean_speed": mean_speed}
