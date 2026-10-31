"""Generate a simulated scan session from a Teeth3DS case."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from ioscan.calibration.models import CameraIntrinsics
from ioscan.session.format import SessionWriter
from ioscan.sim.motion import canonicalize, ideal_imu, make_trajectory
from ioscan.sim.scanner import LaserSpec, RenderParams, VirtualScanner
from ioscan.sim.teeth3ds import list_cases, load_labels, load_mesh
from ioscan.utils.config import load_yaml, repo_root


def plot_trajectory(mesh, T_wc: np.ndarray, info: dict, path: Path) -> None:
    """Top-down sanity plot: mesh vertices, arch centreline, camera positions and heading."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    V = np.asarray(mesh.vertices)[::15]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(V[:, 0], V[:, 1], s=1, c="lightgray")
    c = info["centerline"]
    ax.plot(c[:, 0], c[:, 1], "b-", lw=1, label="arch centreline")
    ax.plot(T_wc[:, 0, 3], T_wc[:, 1, 3], "r.", ms=1, label="camera path")
    step = max(1, len(T_wc) // 25)
    ax.quiver(
        T_wc[::step, 0, 3], T_wc[::step, 1, 3], T_wc[::step, 0, 1], T_wc[::step, 1, 1],
        color="r", scale=30, width=0.004,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Trajectory (top view, arrows = direction of travel)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def generate_session(
    out_dir: str | Path,
    root: str | Path | None = None,
    case_id: str | None = None,
    jaw: str | None = None,
    max_frames: int | None = None,
    overwrite: bool = False,
    progress: bool = True,
) -> dict:
    cfg_dir = repo_root() / "configs"
    cam_cfg = load_yaml(cfg_dir / "camera.yaml")
    laser_cfg = load_yaml(cfg_dir / "laser.yaml")
    sim_cfg = load_yaml(cfg_dir / "sim.yaml")

    root = root or sim_cfg["teeth3ds"].get("root")
    jaw = jaw or sim_cfg["teeth3ds"]["jaw"]
    case_id = case_id or sim_cfg["teeth3ds"].get("case_id")
    cases = list_cases(root, jaw)
    if not cases:
        raise FileNotFoundError(f"No {jaw} cases found")
    case = next((c for c in cases if c.case_id == case_id), None) if case_id else cases[0]
    if case is None:
        raise FileNotFoundError(f"Case {case_id!r} not found among {len(cases)} {jaw} cases")

    mesh = load_mesh(case.obj_path, center=False)
    labels = load_labels(case.json_path).get("labels") if case.json_path else None
    mesh_c, _, _ = canonicalize(mesh, labels)

    scan = sim_cfg["scan"]
    fps = float(cam_cfg["fps"])
    T_wc, info = make_trajectory(
        mesh_c,
        labels,
        working_distance_mm=float(scan["working_distance_mm"]),
        speed_mm_s=float(scan["speed_mm_s"]),
        fps=fps,
        tremor_mm=float(scan["tremor_mm"]),
        tremor_hz=float(scan["tremor_hz"]),
        seed=int(sim_cfg["seed"]),
        max_frames=max_frames if max_frames is not None else scan.get("max_frames"),
    )
    n = len(T_wc)

    camera = CameraIntrinsics.from_config(cam_cfg)
    laser = LaserSpec.from_config(laser_cfg)
    params = RenderParams(line_sigma_px=float(laser_cfg["line_sigma_px"]), **sim_cfg["render"])
    scanner = VirtualScanner(mesh_c, camera, laser, params)

    t_ns = np.rint(np.arange(n) * 1e9 / fps).astype(np.int64)
    gt_frame, gt_uv, gt_P = [], [], []

    meta_cfg = {
        "camera": cam_cfg, "laser": laser_cfg, "sim": sim_cfg,
        "case": {"id": case.case_id, "jaw": case.jaw},
    }
    meta_cfg["sim"]["scan"] = dict(meta_cfg["sim"]["scan"])
    meta_cfg["sim"]["scan"]["generated_frames"] = int(n)
    meta_cfg["sim"]["scan"]["recorded_path_length_mm"] = float(
        np.linalg.norm(np.diff(T_wc[:, :3, 3], axis=0), axis=1).sum()
    ) if n > 1 else 0.0
    meta_cfg["sim"]["scan"]["full_centerline_length_mm"] = float(info["path_len_mm"])
    with SessionWriter(
        out_dir, camera, laser.plane, fps, source="sim", config=meta_cfg, overwrite=overwrite
    ) as w:
        for k in tqdm(range(n), desc="rendering", disable=not progress):
            laser_on = bool(k % 2 == 1)  # OFF first, so every ON frame has a preceding OFF frame
            img, prof = scanner.render_frame(T_wc[k], laser_on)
            w.add_frame(img, int(t_ns[k]), laser_on)
            if prof is not None and len(prof["uv"]):
                gt_frame.append(np.full(len(prof["uv"]), k, dtype=np.int32))
                gt_uv.append(prof["uv"])
                gt_P.append(prof["P_c"])

        imu_t, imu_q = ideal_imu(t_ns, T_wc[:, :3, :3], float(sim_cfg["imu"]["rate_hz"]))
        w.set_imu(imu_t, imu_q)
        w.set_gt_poses(t_ns, T_wc)
        if gt_frame:
            w.set_gt_profiles(np.concatenate(gt_frame), np.vstack(gt_uv), np.vstack(gt_P))
        w.set_reference_mesh(mesh_c)

    plot_trajectory(mesh_c, T_wc, info, Path(out_dir) / "gt" / "trajectory.png")
    info.update(case=case.case_id, jaw=case.jaw, out_dir=str(out_dir),
                gt_points=int(sum(len(a) for a in gt_frame)))
    return info
