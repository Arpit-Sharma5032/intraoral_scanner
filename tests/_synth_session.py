"""Tiny synthetic hardware-format session with a known surface and known motion (no Open3D needed)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ioscan.calibration.models import CameraIntrinsics, LaserPlane

FPS = 50.0
BASE_Z = 45.0


def surface_height(x, y):
    """Smooth bumpy surface, camera-frame-at-start coordinates (mm)."""
    return BASE_Z + 2.5 * np.sin(x / 5.0) + 1.5 * np.cos(y / 4.0)


def make_session(root: Path, n_frames: int = 160, speed_mm_s: float = 5.0, roi_points=None, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    root = Path(root)
    (root / "frames").mkdir(parents=True, exist_ok=True)
    cam = CameraIntrinsics(1600, 1200, 933.3, 933.3, 800.0, 600.0)
    laser = LaserPlane.from_baseline_and_angle(18.0, 10.0, "y")
    ny, nz, d = float(laser.normal[1]), float(laser.normal[2]), float(laser.d)

    tex = cv2.GaussianBlur(rng.uniform(30, 120, (2400, 2400)).astype(np.float32), (0, 0), 6)
    uu, vv = np.meshgrid(np.arange(1600, dtype=np.float32), np.arange(1200, dtype=np.float32))
    xs = np.linspace(-40, 40, 4000)

    rows = []
    for i in range(n_frames):
        t = i / FPS
        yc = speed_mm_s * t
        laser_on = (i % 2 == 1)
        mx = ((uu - cam.cx) / cam.fx * BASE_Z * 10.0 + 1200).astype(np.float32)
        my = (((vv - cam.cy) / cam.fy * BASE_Z + yc) * 10.0 + 400).astype(np.float32)
        img = cv2.remap(tex, mx, my, cv2.INTER_LINEAR).astype(np.float32)
        if laser_on:
            y = np.full_like(xs, yc)
            for _ in range(8):  # fixed point of the plane/surface intersection
                h = surface_height(xs, y)
                y = yc + (-nz * h - d) / ny
            h = surface_height(xs, y)
            u = cam.fx * xs / h + cam.cx
            v = cam.fy * (y - yc) / h + cam.cy
            ok = (u >= 0) & (u < 1599) & (v > 8) & (v < 1190)
            u, v = u[ok], v[ok]
            order = np.argsort(u)
            cols = np.arange(int(np.ceil(u.min())), int(np.floor(u.max())) + 1)
            vc = np.interp(cols, u[order], v[order])
            for dr in range(-7, 8):
                r = np.rint(vc).astype(int) + dr
                good = (r >= 0) & (r < 1200)
                img[r[good], cols[good]] += 150.0 * np.exp(-((r[good] - vc[good]) ** 2) / (2 * 1.5 ** 2))
        img += rng.normal(0, 2.0, img.shape)
        fn = f"frames/{i:05d}.png"
        cv2.imwrite(str(root / fn), np.clip(img, 0, 255).astype(np.uint8))
        rows.append((i, int(round(t * 1e9)), int(laser_on), fn))

    with open(root / "frames.csv", "w", newline="") as f:
        f.write("idx,t_ns,laser_on,file\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]}\n")
    with open(root / "imu.csv", "w") as f:
        f.write("t_ns,qw,qx,qy,qz\n")
        for i in range(n_frames + 2):
            f.write(f"{int(i * 1e9 / FPS)},1,0,0,0\n")
    (root / "meta.json").write_text(json.dumps({
        "source": "hardware", "camera": cam.to_dict(), "laser_plane": laser.to_dict(),
        "calibration_is_nominal": True, "requested_fps": FPS, "fps": FPS,
    }))
    roi = {"version": 1, "width": 1600, "height": 1200, "enabled": roi_points is not None,
           "points": roi_points or []}
    (root / "roi.json").write_text(json.dumps(roi))
    return {"camera": cam, "laser": laser, "speed": speed_mm_s}


# The ROI actually saved with the failing hardware session (normalised corners).
USER_ROI = [[0.3856815820157724, 0.977929931686035], [0.3256352506225313, 0.5955089041565672],
            [0.7575058574985893, 0.5430803154675301], [0.6974598785023305, 0.9594257377542332]]
