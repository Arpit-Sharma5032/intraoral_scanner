"""Synthetic hardware-format sessions recorded on the printed marker plate (no Open3D needed).

A full arch model stands on the marker sheet and is moved freely under a fixed camera + laser:
it slides, turns through ~180 deg, tilts and lifts, as a hand-held plate does. OFF frames show the
sheet (hidden where the model stands) through the scanner's window; ON frames add the laser line
where the laser plane meets the model. Ground truth is the height field in the plate frame.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ioscan.calibration.models import CameraIntrinsics, LaserPlane
from ioscan.pipeline.markers import MarkerBoard, render_board_image

from ._synth_arch import arch_centerline, arch_height

FPS = 50.0
PPM = 16.0                                   # board texture resolution (px / mm)
WINDOW = (530, 480, 1060, 1200)              # the optical opening: x0, y0, x1, y1 (pixels)
ROI = [[0.41, 0.47], [0.59, 0.47], [0.59, 0.66], [0.41, 0.66]]
BASE_H = 6.0                                 # height of the model base above the sheet (mm)
ARCH_OFF = 18.0                              # arch-frame Y that sits on the sheet centre
GRID_MM = 0.1


class PlateModel:
    """Arch on a solid base, as a height raster above the sheet (plate frame, mm)."""

    def __init__(self):
        self.line = arch_centerline()
        self.line_xy = self.line[:, :2] * [1.0, -1.0] + [0.0, ARCH_OFF]
        self.x0, self.y0 = -45.0, -40.0
        nx, ny = int(90 / GRID_MM), int(80 / GRID_MM)
        gx = self.x0 + (np.arange(nx) + 0.5) * GRID_MM
        gy = self.y0 + (np.arange(ny) + 0.5) * GRID_MM
        X, Y = np.meshgrid(gx, gy)
        # arch_height works in its own frame (incisors at +Y); map plate -> arch coordinates
        ax = X.ravel()
        ay = ARCH_OFF - Y.ravel()
        teeth = arch_height(np.column_stack([ax, ay]), self.line).reshape(ny, nx)
        # solid base: everything within 9 mm outside the arch line and all of the inside of the U
        from scipy.spatial import cKDTree
        d, i = cKDTree(self.line[:, :2]).query(np.column_stack([ax, ay]))
        inside_u = (ay < 40.0 - 0.06 * ax ** 2) & (ay > self.line[:, 1].min() - 4.0) & (np.abs(ax) < 30)
        base = ((d < 9.0) | inside_u).reshape(ny, nx)
        self.h = np.where(base, BASE_H + teeth, 0.0).astype(np.float32)
        self.h = cv2.GaussianBlur(self.h, (0, 0), 1.0)
        self.footprint = base

    def height(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        from scipy.ndimage import map_coordinates
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        cx = (x - self.x0) / GRID_MM - 0.5
        cy = (y - self.y0) / GRID_MM - 0.5
        return map_coordinates(self.h, [cy.ravel(), cx.ravel()], order=1, mode="constant", cval=0.0).reshape(x.shape)


class FlatSheet(PlateModel):
    """The bare marker sheet (calibration recording): nothing stands on it."""

    def __init__(self):
        self.line = arch_centerline()
        self.line_xy = self.line[:, :2] * [1.0, -1.0] + [0.0, ARCH_OFF]
        self.x0, self.y0 = -45.0, -40.0
        self.h = np.zeros((int(80 / GRID_MM), int(90 / GRID_MM)), np.float32)
        self.footprint = np.zeros(self.h.shape, bool)


def _pose(yaw: float, tilt_x: float, tilt_y: float) -> np.ndarray:
    return cv2.Rodrigues(np.array([tilt_x, tilt_y, 0.0]))[0] @ cv2.Rodrigues(np.array([0.0, 0.0, yaw]))[0]


def make_marker_session(root: Path, *, duration_s: float = 16.0, span: tuple[float, float] = (-0.92, 0.92),
                        tilt_deg: float = 4.0, lift_mm: float = 2.0, base_tilt_deg: tuple[float, float] = (0.0, 0.0),
                        mirrored: bool = False, blur: float = 1.4, ambient: tuple[float, float] = (22.0, 95.0),
                        seed: int = 0, jpeg: bool = True, board: MarkerBoard | None = None,
                        camera: CameraIntrinsics | None = None, laser: LaserPlane | None = None,
                        stored_camera: CameraIntrinsics | None = None, stored_laser: LaserPlane | None = None,
                        model: PlateModel | None = None, plate_z: float = 90.0, occlude_mm: float = 3.0) -> dict:
    """Write a session. ``camera``/``laser`` are the true optics; ``stored_*`` go into meta.json."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    (root / "frames").mkdir(parents=True, exist_ok=True)
    board = board or MarkerBoard()
    cam = camera or CameraIntrinsics(1600, 1200, 933.3, 933.3, 800.0, 600.0)
    laser = laser or LaserPlane.from_baseline_and_angle(19.0, 11.0, "y")
    model = model or PlateModel()
    K = cam.K
    n = np.asarray(laser.normal, dtype=np.float64)
    d = float(laser.d)

    tex, margin = render_board_image(board, PPM, 4.0)
    bw = (board.cols - 1) * board.pitch_mm + board.marker_mm
    bh = (board.rows - 1) * board.pitch_mm + board.marker_mm
    # paint the model over the sheet (footprint dilated for the parallax of a tall object)
    fp = cv2.dilate(model.footprint.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=int(occlude_mm / GRID_MM))
    ys, xs = np.nonzero(fp)
    px = ((model.x0 + (xs + 0.5) * GRID_MM) / board.scale + bw / 2 + margin) * PPM
    py = ((model.y0 + (ys + 0.5) * GRID_MM) / board.scale + bh / 2 + margin) * PPM
    ok = (px >= 0) & (px < tex.shape[1] - 1) & (py >= 0) & (py < tex.shape[0] - 1)
    shade = np.clip(170 + 4.0 * model.h[ys, xs], 0, 235).astype(np.uint8)
    tex = tex.copy()
    for dx in (0, 1):
        for dy in (0, 1):
            tex[(py[ok] + dy).astype(int), (px[ok] + dx).astype(int)] = shade[ok]
    S = np.array([[board.scale / PPM, 0, -(bw / 2 + margin) * board.scale],
                  [0, board.scale / PPM, -(bh / 2 + margin) * board.scale], [0, 0, 1.0]])

    line = model.line_xy                                     # arch centre line in the plate frame
    n_frames = int(round(duration_s * FPS))
    t_all = np.arange(n_frames) / FPS
    # progress along the arch with a varying hand speed
    prog = t_all / duration_s
    prog = prog + 0.04 * np.sin(2 * np.pi * prog * 3.0)
    sidx = np.clip(((span[0] + (span[1] - span[0]) * prog) * 0.5 + 0.5) * (len(line) - 1), 1, len(line) - 2)
    roi_px = np.asarray(ROI) * [1600, 1200]
    u_cols = np.arange(int(roi_px[:, 0].min()) + 2, int(roi_px[:, 0].max()) - 2, dtype=np.float64)
    v_s = np.arange(roi_px[:, 1].min() + 3, roi_px[:, 1].max() - 3, 0.5)
    UU, VV = np.meshgrid(u_cols, v_s)
    rays = np.stack([(UU - cam.cx) / cam.fx, (VV - cam.cy) / cam.fy, np.ones_like(UU)], axis=-1)
    lam = -d / (rays @ n)
    P_cam_grid = rays * lam[..., None]                       # laser-plane point seen by every ROI pixel
    x0w, y0w, x1w, y1w = WINDOW
    wmask = np.zeros((1200, 1600), np.float32)
    wmask[y0w:y1w, x0w:x1w] = 1.0
    wmask = cv2.GaussianBlur(wmask, (0, 0), 3)

    dist_map = None
    if any(abs(c) > 1e-12 for c in cam.dist_coeffs):
        # the scene is rendered with an ideal pinhole, then warped into the real (distorted) image
        gu, gv = np.meshgrid(np.arange(1600, dtype=np.float32), np.arange(1200, dtype=np.float32))
        und = cv2.undistortPoints(np.stack([gu.ravel(), gv.ravel()], axis=1).reshape(-1, 1, 2), K,
                                  np.asarray(cam.dist_coeffs, dtype=np.float64), P=K).reshape(1200, 1600, 2)
        dist_map = (und[..., 0].astype(np.float32), und[..., 1].astype(np.float32))

    rows, truth = [], []
    ext = ".jpg" if jpeg else ".png"
    for i in range(n_frames):
        t = t_all[i]
        k = sidx[i]
        k0 = int(np.floor(k))
        c = line[k0] + (k - k0) * (line[k0 + 1] - line[k0])
        tang = line[min(k0 + 8, len(line) - 1)] - line[max(k0 - 8, 0)]
        # turn the plate so the tooth row runs along the camera Y axis (across the laser line)
        yaw = np.pi / 2 - np.arctan2(tang[1], tang[0]) + np.radians(6.0) * np.sin(2 * np.pi * t / 5.3)
        tx_ = np.radians(base_tilt_deg[0]) + np.radians(tilt_deg) * np.sin(2 * np.pi * t / 4.1)
        ty_ = np.radians(base_tilt_deg[1]) + np.radians(tilt_deg) * np.cos(2 * np.pi * t / 6.7)
        R = _pose(yaw, tx_, ty_)
        z_pl = plate_z + lift_mm * np.sin(2 * np.pi * t / 3.3)
        h_c = float(model.height(np.array([c[0]]), np.array([c[1]]))[0])
        z_tooth = z_pl - h_c
        target = np.array([1.5 * np.sin(2 * np.pi * t / 7.9), (-d - n[2] * z_tooth) / n[1], z_tooth])
        tvec = target - R @ np.array([c[0], c[1], -h_c])
        truth.append((t, R.copy(), tvec.copy()))

        H = K @ np.column_stack([R[:, 0], R[:, 1], tvec]) @ S
        img = cv2.warpPerspective(tex, H, (1600, 1200), flags=cv2.INTER_AREA, borderValue=120).astype(np.float32)
        img = ambient[0] + (ambient[1] - ambient[0]) * img / 255.0
        img = img * wmask + 14.0 * (1 - wmask)
        laser_on = (i % 2 == 1)
        if laser_on:
            Pp = (P_cam_grid.reshape(-1, 3) - tvec) @ R      # plate frame
            g = (-Pp[:, 2]) - model.height(Pp[:, 0], Pp[:, 1])
            g = g.reshape(UU.shape)
            # nearest-to-camera surface crossing in each column (rows run far -> near)
            below = g < 0
            cross = below[:-1] & ~below[1:]
            has = cross.any(axis=0)
            last = cross.shape[0] - 1 - np.argmax(cross[::-1], axis=0)
            cols = np.flatnonzero(has)
            r0 = last[cols]
            g0, g1 = g[r0, cols], g[r0 + 1, cols]
            vc = v_s[r0] + 0.5 * g0 / (g0 - g1)
            amp = 80.0 + 30.0 * rng.random()
            ci = u_cols[cols].astype(int)
            for dr in range(-7, 8):
                r = np.rint(vc).astype(int) + dr
                good = (r >= 0) & (r < 1200)
                img[r[good], ci[good]] += amp * np.exp(-((r[good] - vc[good]) ** 2) / (2 * 1.7 ** 2))
        if dist_map is not None:
            img = cv2.remap(img, dist_map[0], dist_map[1], cv2.INTER_LINEAR, borderValue=14.0)
        if blur > 0:
            img = cv2.GaussianBlur(img, (0, 0), blur)
        img += rng.normal(0, 2.0, img.shape)
        out = np.clip(img, 0, 255).astype(np.uint8)
        if mirrored:
            out = cv2.flip(out, 1)
        fn = f"frames/{i:05d}{ext}"
        cv2.imwrite(str(root / fn), out, [cv2.IMWRITE_JPEG_QUALITY, 92] if jpeg else [])
        rows.append((i, int(round(t * 1e9)), int(laser_on), fn))

    with open(root / "frames.csv", "w", newline="") as f:
        f.write("idx,t_ns,laser_on,file\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]}\n")
    with open(root / "imu.csv", "w") as f:
        f.write("t_ns,qw,qx,qy,qz\n")
        for i in range(n_frames + 2):
            f.write(f"{int(i * 1e9 / FPS)},1,0,0,0\n")
    sc, sl = stored_camera or cam, stored_laser or laser
    (root / "meta.json").write_text(json.dumps({
        "source": "hardware", "camera": sc.to_dict(), "laser_plane": sl.to_dict(),
        "calibration_is_nominal": stored_camera is not None, "requested_fps": FPS, "fps": FPS, "jaw": "lower",
    }))
    (root / "roi.json").write_text(json.dumps(
        {"version": 1, "width": 1600, "height": 1200, "enabled": True, "points": ROI}))
    return {"camera": cam, "laser": laser, "board": board, "model": model, "truth": truth}
