"""MeshSegNet tooth segmentation for reconstructed dental triangle meshes.

The inference preprocessing in this module follows the public MeshSegNet reference
implementation (Tai-Hsien/MeshSegNet): 15 channels are ordered as 9 normalized triangle
vertex coordinates, 3 normalized cell-centre coordinates, and 3 normalized face-normal
components. The network predicts 15 classes (gingiva + 14 tooth IDs).

MeshSegNet labels only geometry that exists in the input mesh. It cannot reconstruct missing
teeth or unseen tooth walls from a partial scan.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any
import zipfile

import numpy as np
from scipy.spatial import cKDTree

try:
    import open3d as o3d
except ImportError:  # permit model/config/feature tests without the optional geometry runtime
    o3d = None


def _require_open3d():
    """Load Open3D only when mesh I/O or geometry operations are requested."""
    global o3d
    if o3d is None:
        try:
            import open3d as _open3d
        except ImportError as exc:
            raise RuntimeError(
                "Mesh I/O and full segmentation need Open3D. Install project dependencies with "
                "python -m pip install -e ."
            ) from exc
        o3d = _open3d
    return o3d

PALETTE = np.array([
    [160, 160, 160],  # 0 = gingiva
    [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
    [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212],
    [0, 128, 128], [220, 190, 255], [170, 110, 40], [128, 0, 0],
], dtype=np.uint8)

# The supplied user checkpoints are named .zip; those archives may contain a .tar/.pth file,
# or may themselves be torch.save zip containers. Both cases are supported.
CKPT_SUFFIXES = (".tar", ".pth", ".pt", ".ckpt", ".zip")
INNER_CKPT_SUFFIXES = (".tar", ".pth", ".pt", ".ckpt")
PARTIAL_SCAN_EXTENT_MM = 35.0
MIN_SEGMENTATION_INPUT_FACES = 5000
MIN_SEGMENTATION_EXTENT_MM = 35.0


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Segmentation needs PyTorch. Install it with: python -m pip install -e '.[seg]'"
        ) from exc
    return torch


def default_models_dir() -> Path | None:
    env = os.environ.get("IOSCAN_MESHSEGNET_MODELS")
    return Path(env).expanduser() if env else None


@dataclass
class SegmentConfig:
    models_dir: Path | None = field(default_factory=default_models_dir)
    jaw: str = "auto"                 # "lower" | "upper" | "auto" (read from session meta.json)
    ckpt_path: Path | None = None     # explicit checkpoint/archive; must exist if given
    target_cells: int = 10000
    num_classes: int = 15
    device: str | None = None         # None -> cuda if available else cpu
    repo_dir: Path | None = None      # optional reference MeshSegNet checkout

    def __post_init__(self) -> None:
        if not 20 <= self.target_cells <= 10000:
            raise ValueError("target_cells must be between 20 and 10000 (MeshSegNet uses dense N x N adjacency)")
        if self.num_classes != 15:
            raise ValueError("The supplied MeshSegNet models require exactly 15 classes")
        if self.jaw not in ("auto", "lower", "upper"):
            raise ValueError("jaw must be 'auto', 'lower', or 'upper'")


def infer_jaw(session_dir: str | Path | None) -> str | None:
    """Read the jaw from metadata, falling back to a jaw suffix in the session name."""
    if not session_dir:
        return None
    session_path = Path(session_dir)
    meta = session_path / "meta.json"
    if meta.is_file():
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        config = data.get("config") or {}
        candidates = [
            (config.get("case") or {}).get("jaw"),
            (config.get("sim") or {}).get("teeth3ds", {}).get("jaw"),
            config.get("jaw"),
            data.get("jaw"),
        ]
        for jaw in candidates:
            if isinstance(jaw, str) and jaw.lower() in ("lower", "upper"):
                return jaw.lower()

    # Some imported/older sessions have no jaw metadata. Dataset session names
    # may encode it explicitly, for example FUYJPUPX_lower. Metadata always wins.
    return infer_jaw_from_name(session_path.name)


_LOWER_RE = re.compile(r"(?<![a-z])(lower|mandib[a-z]*|man)(?![a-z])", re.IGNORECASE)
_UPPER_RE = re.compile(r"(?<![a-z])(upper|maxil[a-z]*|max)(?![a-z])", re.IGNORECASE)


def infer_jaw_from_name(path: str | Path) -> str | None:
    """Infer the jaw from a known checkpoint naming convention."""
    name = Path(path).stem.lower()
    lower = bool(_LOWER_RE.search(name))
    upper = bool(_UPPER_RE.search(name))
    if lower == upper:  # neither or conflicting token
        return None
    return "lower" if lower else "upper"


def resolve_checkpoint(cfg: SegmentConfig, jaw: str | None = None) -> Path:
    """Resolve a lower/upper checkpoint, accepting both raw files and .zip archives."""
    if cfg.ckpt_path is not None:
        p = Path(cfg.ckpt_path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p

    if cfg.models_dir is None:
        raise FileNotFoundError(
            "No MeshSegNet checkpoint configured. Pass --ckpt, pass --models-dir, "
            "or set IOSCAN_MESHSEGNET_MODELS to the folder containing the Man/Max model files."
        )
    models_dir = Path(cfg.models_dir).expanduser()
    if not models_dir.is_dir():
        raise FileNotFoundError(f"MeshSegNet models directory not found: {models_dir}")

    selected_jaw = (jaw or cfg.jaw).lower()
    if selected_jaw not in ("lower", "upper"):
        raise ValueError(
            "Cannot determine the jaw (session meta.json is missing or incomplete). "
            "Pass --jaw lower|upper or an explicit --ckpt."
        )
    pattern = _LOWER_RE if selected_jaw == "lower" else _UPPER_RE
    matches = sorted(
        f for f in models_dir.iterdir()
        if f.is_file() and f.suffix.lower() in CKPT_SUFFIXES and pattern.search(f.stem)
    )
    if not matches:
        token = "Man/lower/mandible" if selected_jaw == "lower" else "Max/upper/maxilla"
        raise FileNotFoundError(
            f"No {selected_jaw}-jaw MeshSegNet checkpoint found in {models_dir}. "
            f"Expected a filename containing {token} and one of {CKPT_SUFFIXES}; "
            "for example MeshSegNet_Man_15_classes_72samples_lr1e-2_best.zip."
        )
    if len(matches) > 1:
        best = [p for p in matches if "best" in p.stem.lower()]
        if len(best) == 1:
            return best[0]
        raise RuntimeError(
            f"Found multiple {selected_jaw}-jaw checkpoints in {models_dir}: "
            f"{', '.join(p.name for p in matches)}. Pass --ckpt to choose one explicitly."
        )
    return matches[0]


def resolve_segmentation_model(
    cfg: SegmentConfig, session_hint: str | Path | None = None
) -> tuple[Path, str]:
    """Resolve and validate the model path/jaw before expensive scan reconstruction starts."""
    jaw = cfg.jaw.lower()
    if jaw == "auto":
        jaw = infer_jaw(session_hint) or "auto"
    checkpoint = resolve_checkpoint(cfg, jaw=jaw)
    checkpoint_jaw = infer_jaw_from_name(checkpoint)
    if jaw in ("lower", "upper") and checkpoint_jaw and checkpoint_jaw != jaw:
        raise ValueError(
            f"Jaw mismatch: session/--jaw selects '{jaw}', but checkpoint filename "
            f"'{checkpoint.name}' identifies the '{checkpoint_jaw}' model. Choose the matching "
            "checkpoint or pass the intended --jaw."
        )
    if jaw == "auto":
        jaw = checkpoint_jaw or "auto"
    return checkpoint, jaw


# ----------------------------------------------------------------------------- mesh helpers
def _load_triangle_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    _require_open3d()
    if not path.is_file():
        raise FileNotFoundError(f"Input mesh not found: {path}")
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.is_empty():
        raise RuntimeError(f"Could not read a triangle mesh from: {path}")
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) == 0:
        raise RuntimeError(f"Cannot run MeshSegNet on an empty triangle mesh: {path}")
    vertices = np.asarray(mesh.vertices)
    if not np.isfinite(vertices).all():
        raise ValueError(f"Input mesh has NaN or infinite vertex coordinates: {path}")
    return mesh


def _cell_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = points[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    return (n / np.maximum(norm, 1e-12)).astype(np.float32)


def _mesh_point_center(points: np.ndarray) -> np.ndarray:
    """Point-weighted mesh centre, matching vedo's ``mesh.center_of_mass()``.

    Vedo delegates this calculation to VTK's vtkCenterOfMass, which weights each mesh
    vertex equally unless scalar weights are explicitly enabled. It is not the area-weighted
    centroid of triangle surfaces, so using an area-weighted centre here changes the trained
    model's coordinate normalization.
    """
    return points.mean(axis=0)


def build_features(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Build the exact 15-channel MeshSegNet feature layout.

    Feature columns are [three vertices (9), normalized barycenter (3), normalized
    face normal (3)]. This order follows the reference implementation's
    ``np.column_stack((cells, barycenters, normals))`` and is significant to the trained weights.
    """
    points = np.asarray(points, dtype=np.float64).copy()
    faces = np.asarray(faces, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3); got {points.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (M, 3); got {faces.shape}")
    if len(points) == 0 or len(faces) == 0:
        raise ValueError("Cannot build MeshSegNet features for an empty mesh")
    if faces.min() < 0 or faces.max() >= len(points):
        raise ValueError("Triangle face indices refer outside the vertex array")
    if not np.isfinite(points).all():
        raise ValueError("Mesh vertices contain NaN or infinite coordinates")

    center = _mesh_point_center(points)
    points -= center
    n_cells = len(faces)
    tri = points[faces]
    cells = tri.reshape(n_cells, 9).astype(np.float32)
    normals = _cell_normals(points, faces)
    barycenters = tri.mean(axis=1).astype(np.float32)

    maxs, mins = points.max(axis=0), points.min(axis=0)
    means, stds = points.mean(axis=0), points.std(axis=0)
    nmeans, nstds = normals.mean(axis=0), normals.std(axis=0)
    stds = np.where(stds < 1e-6, 1.0, stds)
    spans = np.where((maxs - mins) < 1e-6, 1.0, maxs - mins)
    nstds = np.where(nstds < 1e-6, 1.0, nstds)

    for i in range(3):
        for offset in (0, 3, 6):
            cells[:, i + offset] = (cells[:, i + offset] - means[i]) / stds[i]
        barycenters[:, i] = (barycenters[:, i] - mins[i]) / spans[i]
        normals[:, i] = (normals[:, i] - nmeans[i]) / nstds[i]

    # Important: barycentres are in channels 9:12; adjacency matrices use these values.
    features = np.column_stack((cells, barycenters, normals)).astype(np.float32)
    if not np.isfinite(features).all():
        raise RuntimeError("MeshSegNet feature normalization produced non-finite values")
    return features


def _normalized_adjacency(coords: np.ndarray, radius: float) -> np.ndarray:
    """Build dense row-normalized adjacency without materializing an N x N distance matrix."""
    n_cells = len(coords)
    tree = cKDTree(coords)
    neighbors = tree.query_ball_point(coords, r=np.nextafter(float(radius), 0.0), workers=-1)
    adjacency = np.zeros((n_cells, n_cells), dtype=np.float32)
    for row, cols in enumerate(neighbors):
        if not cols:
            adjacency[row, row] = 1.0
        else:
            adjacency[row, np.asarray(cols, dtype=np.int64)] = 1.0 / len(cols)
    return adjacency


def prepare_decimated_inputs(
    mesh: o3d.geometry.TriangleMesh,
    target_cells: int = 10000,
    device: str = "cpu",
):
    """Decimate and build reference-compatible features and S1/S2 adjacency matrices.

    Adjacency is assembled on CPU to avoid the temporary N x N distance tensor on GPU. The
    final dense matrices still follow the original model API and are moved to the selected device.
    """
    torch = _import_torch()
    _require_open3d()
    if not 20 <= target_cells <= 10000:
        raise ValueError("target_cells must be between 20 and 10000")
    work = o3d.geometry.TriangleMesh(mesh)
    if len(work.triangles) > target_cells:
        work = work.simplify_quadric_decimation(target_number_of_triangles=int(target_cells))
        work.remove_degenerate_triangles()
        work.remove_unreferenced_vertices()
    if len(work.triangles) < 20:
        raise RuntimeError(
            f"Mesh has only {len(work.triangles)} usable triangles after cleanup/decimation; "
            "at least 20 are required for segmentation."
        )

    points = np.asarray(work.vertices, dtype=np.float64)
    faces = np.asarray(work.triangles, dtype=np.int64)
    X = build_features(points, faces)
    X_t = torch.from_numpy(X.T.copy()).unsqueeze(0).to(device=device, dtype=torch.float32)

    # Move each dense matrix separately so CPU RAM does not temporarily hold duplicate copies
    # of both N x N tensors while CUDA inference is selected.
    s1 = torch.from_numpy(_normalized_adjacency(X[:, 9:12], 0.1)).unsqueeze(0)
    S1 = s1.to(device=device, dtype=torch.float32)
    del s1
    s2 = torch.from_numpy(_normalized_adjacency(X[:, 9:12], 0.2)).unsqueeze(0)
    S2 = s2.to(device=device, dtype=torch.float32)
    del s2
    return work, X_t, S1, S2


def _torch_load_checkpoint(torch, path: Path):
    """Load a torch checkpoint or a zip containing one, without unsafe pickle fallback."""
    def load(source):
        try:
            return torch.load(source, map_location="cpu", weights_only=True)
        except TypeError as exc:  # old torch without the safe weights_only argument
            raise RuntimeError(
                "This project requires a PyTorch version that supports safe weights_only loading. "
                "Upgrade with: python -m pip install --upgrade torch"
            ) from exc

    try:
        return load(str(path)), None
    except Exception as direct_error:
        if not zipfile.is_zipfile(path):
            raise RuntimeError(
                f"Could not load MeshSegNet checkpoint '{path}'. It must be a PyTorch state-dict "
                "checkpoint or a .zip archive containing one."
            ) from direct_error

    with zipfile.ZipFile(path, "r") as archive:
        members = [
            info for info in archive.infolist()
            if not info.is_dir()
            and PurePosixPath(info.filename).suffix.lower() in INNER_CKPT_SUFFIXES
        ]
        if not members:
            raise RuntimeError(
                f"'{path.name}' is a ZIP file, but it contains no .tar/.pth/.pt/.ckpt checkpoint. "
                "Re-create the archive with the original MeshSegNet checkpoint inside it."
            )
        if len(members) > 1:
            best = [m for m in members if "best" in PurePosixPath(m.filename).stem.lower()]
            if len(best) == 1:
                members = best
            else:
                raise RuntimeError(
                    f"'{path.name}' contains multiple checkpoints: "
                    f"{', '.join(m.filename for m in members)}. Keep one checkpoint in the ZIP "
                    "or pass the checkpoint file directly with --ckpt."
                )
        member = members[0]
        if member.file_size > 2 * 1024**3:
            raise RuntimeError(f"Checkpoint member is unexpectedly large: {member.file_size} bytes")
        try:
            ckpt = load(io.BytesIO(archive.read(member)))
        except Exception as exc:
            raise RuntimeError(
                f"Could not load '{member.filename}' from '{path.name}' as a safe PyTorch "
                "checkpoint. Re-save the model weights as a standard torch state_dict checkpoint."
            ) from exc
        return ckpt, member.filename


def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    """Extract common training-checkpoint wrappers and normalize DataParallel prefixes."""
    state = checkpoint
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            candidate = state.get(key)
            if isinstance(candidate, dict) and candidate:
                state = candidate
                break
    if not isinstance(state, dict) or not state:
        raise RuntimeError("Checkpoint does not contain a non-empty model state dictionary")
    if all(isinstance(k, str) and k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    return state


def _load_network(cfg: SegmentConfig, ckpt_path: Path, device: str):
    torch = _import_torch()
    from ioscan.pipeline.meshsegnet_arch import load_meshsegnet_class

    net_cls = load_meshsegnet_class(cfg.repo_dir)
    net = net_cls(num_classes=cfg.num_classes, num_channels=15).to(device)
    checkpoint, inner_name = _torch_load_checkpoint(torch, ckpt_path)
    state = _extract_state_dict(checkpoint)
    try:
        net.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint '{ckpt_path.name}' is not compatible with the MeshSegNet architecture. "
            "The 15-class Man/Max models from Tai-Hsien/MeshSegNet should load strictly. "
            "Check that the correct checkpoint was selected and that --repo-dir points to the "
            "matching reference repository if you trained a modified model.\n"
            f"Original PyTorch error: {exc}"
        ) from exc
    net.eval()
    return net, inner_name


def _predict_decimated(net, X_t, S1, S2, num_classes: int):
    torch = _import_torch()
    with torch.inference_mode():
        output = net(X_t, S1, S2)
    if output.ndim != 3:
        raise RuntimeError(f"MeshSegNet output must have 3 dimensions; received {tuple(output.shape)}")
    if output.shape[-1] == num_classes:
        probs = output
    elif output.shape[1] == num_classes:
        probs = output.transpose(1, 2)
    else:
        raise RuntimeError(
            f"MeshSegNet returned {tuple(output.shape)}; expected (1, N, {num_classes}) probabilities"
        )
    if probs.shape[0] != 1 or probs.shape[1] != X_t.shape[2]:
        raise RuntimeError(
            f"MeshSegNet returned {tuple(probs.shape)} probabilities for {X_t.shape[2]} cells"
        )
    if not torch.isfinite(probs).all():
        raise RuntimeError("MeshSegNet returned NaN or infinite class probabilities")
    row_sums = probs.sum(dim=-1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3, rtol=1e-3):
        raise RuntimeError("MeshSegNet output is not a normalized class-probability tensor")
    confidence, labels = probs.max(dim=-1)
    return labels[0].to("cpu").numpy().astype(np.int32), confidence[0].to("cpu").numpy().astype(np.float32)


def _to_o3d_colors(rgb: np.ndarray) -> o3d.utility.Vector3dVector:
    _require_open3d()
    return o3d.utility.Vector3dVector(np.asarray(rgb, dtype=np.float64) / 255.0)


def _unshared_mesh(
    points: np.ndarray,
    faces: np.ndarray,
    face_rgb: np.ndarray,
) -> o3d.geometry.TriangleMesh:
    """Mesh with unshared vertices so every face keeps its own label color."""
    _require_open3d()
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(points[faces].reshape(-1, 3))
    m.triangles = o3d.utility.Vector3iVector(np.arange(3 * len(faces), dtype=np.int32).reshape(-1, 3))
    m.vertex_colors = _to_o3d_colors(np.repeat(face_rgb, 3, axis=0))
    return m


def _write_mesh(path: Path, mesh: o3d.geometry.TriangleMesh) -> None:
    _require_open3d()
    if len(mesh.triangles) == 0 or not o3d.io.write_triangle_mesh(str(path), mesh):
        raise IOError(f"Failed to write segmented mesh: {path}")


def _write_tooth_classes(
    out: Path,
    points: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Export every predicted tooth class as a standalone coloured PLY and geometry STL."""
    class_dir = out / "segmented_teeth_by_class"
    class_dir.mkdir(parents=True, exist_ok=True)
    for stale in class_dir.glob("tooth_class_*.*"):
        if stale.is_file():
            stale.unlink()

    exports: dict[str, dict[str, Any]] = {}
    for class_id in sorted(int(c) for c in np.unique(labels) if int(c) > 0):
        mask = labels == class_id
        if not mask.any():
            continue
        tooth_mesh = _unshared_mesh(points, faces[mask], PALETTE[class_id:class_id + 1].repeat(int(mask.sum()), axis=0))
        ply_path = class_dir / f"tooth_class_{class_id:02d}.ply"
        stl_path = class_dir / f"tooth_class_{class_id:02d}.stl"
        _write_mesh(ply_path, tooth_mesh)
        tooth_mesh.merge_close_vertices(1e-6)
        tooth_mesh.remove_degenerate_triangles()
        tooth_mesh.remove_unreferenced_vertices()
        tooth_mesh.compute_vertex_normals()
        _write_mesh(stl_path, tooth_mesh)
        exports[str(class_id)] = {
            "faces": int(mask.sum()),
            "ply": str(ply_path),
            "stl": str(stl_path),
        }
    return exports


def segment_mesh_and_cloud(
    mesh_path: str | Path,
    out_dir: str | Path,
    cfg: SegmentConfig | None = None,
    cloud_path: str | Path | None = None,
    session_hint: str | Path = "",
    progress_callback=None,
) -> dict[str, Any]:
    """Run MeshSegNet and write a fully labelled version of every input mesh face.

    The function also writes a combined teeth-only mesh, standalone meshes for each predicted
    tooth class, a coloured point cloud if supplied, and the face labels/confidences in NPZ form.
    """
    def report(stage: str, progress: float, message: str) -> None:
        if progress_callback is not None:
            progress_callback({"stage": stage, "progress": float(progress), "message": message})

    report("segmentation", 3.0, "Checking scan geometry before loading the segmentation model")
    _require_open3d()
    cfg = cfg or SegmentConfig()
    mesh_path = Path(mesh_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Prevent outputs from a previous scan from being mistaken for this attempt.
    for name in ("segmented_full_colored.ply", "segmented_clean_teeth.ply", "segmented_clean_teeth.stl",
                 "segmented_cloud.ply", "segmentation_labels.npz"):
        stale = out / name
        if stale.is_file():
            stale.unlink()
    stale_tooth_dir = out / "segmented_teeth_by_class"
    if stale_tooth_dir.is_dir():
        import shutil
        shutil.rmtree(stale_tooth_dir)

    mesh = _load_triangle_mesh(mesh_path)
    points = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int64)

    if len(faces) < MIN_SEGMENTATION_INPUT_FACES:
        raise RuntimeError(
            f"Mesh has only {len(faces):,} triangles; at least {MIN_SEGMENTATION_INPUT_FACES:,} "
            "are required by the IOSCAN scan-quality gate before MeshSegNet. This is a project "
            "quality guard, not a claim about the model's intrinsic minimum. Recapture with dense, "
            "overlapping profiles; segmentation cannot reconstruct missing surface geometry."
        )

    notes: list[str] = []
    extent = points.max(axis=0) - points.min(axis=0)
    if float(extent.max()) < MIN_SEGMENTATION_EXTENT_MM:
        raise RuntimeError(
            f"Mesh spans only {float(extent.max()):.2f} mm; at least "
            f"{MIN_SEGMENTATION_EXTENT_MM:.0f} mm is required by the IOSCAN dental-arch quality gate. "
            "This is a project guard, not a model-intrinsic limit. A small patch is not a complete "
            "arch; recapture the scan before attempting tooth segmentation."
        )
    from ioscan.pipeline.quality import MAX_EXTENT_MM
    if float(extent.max()) > MAX_EXTENT_MM:
        raise RuntimeError(
            f"Mesh spans {float(extent.max()):.1f} mm, larger than any dental arch (> {MAX_EXTENT_MM:.0f} mm). "
            "This is the signature of a smeared reconstruction (wrong scan speed or orientation path), "
            "not a bigger jaw; MeshSegNet labels on it would be meaningless. Fix the pose/translation "
            "and recapture."
        )
    notes.append(
        "The input mesh is not automatically aligned to a canonical dental coordinate frame. "
        "Predictions can be orientation-sensitive."
    )

    # Avoid importing PyTorch/loading checkpoints until the mesh has passed quality checks.
    torch = _import_torch()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_path, jaw = resolve_segmentation_model(cfg, session_hint=session_hint)
    report("segmentation", 12.0, f"Selected {jaw} jaw checkpoint")
    if jaw == "auto":
        notes.append("Jaw could not be inferred from session metadata or checkpoint filename.")

    # OOM on a CUDA device is recoverable: redo preprocessing and inference on CPU.
    def infer_on(selected_device: str):
        report("segmentation", 28.0, "Preparing mesh features and adjacency matrices")
        decimated_mesh, X_t, S1, S2 = prepare_decimated_inputs(mesh, cfg.target_cells, selected_device)
        net, inner_checkpoint = _load_network(cfg, ckpt_path, selected_device)
        report("segmentation", 48.0, f"Running MeshSegNet inference on {selected_device}")
        labels, confidence = _predict_decimated(net, X_t, S1, S2, cfg.num_classes)
        return decimated_mesh, labels, confidence, inner_checkpoint

    try:
        decimated, preds_dec, confidence_dec, inner_checkpoint = infer_on(device)
    except RuntimeError as exc:
        is_cuda_oom = device.startswith("cuda") and (
            isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", RuntimeError))
            or "out of memory" in str(exc).lower()
        )
        if not is_cuda_oom:
            raise
        notes.append(f"CUDA ran out of memory; inference was retried on CPU. Original error: {exc}")
        torch.cuda.empty_cache()
        device = "cpu"
        decimated, preds_dec, confidence_dec, inner_checkpoint = infer_on(device)

    report("segmentation", 78.0, "Projecting tooth labels onto the full-resolution mesh")
    dec_pts = np.asarray(decimated.vertices, dtype=np.float64)
    dec_faces = np.asarray(decimated.triangles, dtype=np.int64)
    dec_centers = dec_pts[dec_faces].mean(axis=1)
    tree = cKDTree(dec_centers)
    full_centers = points[faces].mean(axis=1)
    _, nearest = tree.query(full_centers, workers=-1)
    full_pred = preds_dec[nearest].astype(np.int32)
    full_confidence = confidence_dec[nearest].astype(np.float32)

    gingiva_pct = float(100.0 * np.mean(full_pred == 0))
    tooth_mask = full_pred > 0
    teeth_found = bool(tooth_mask.any())
    if not teeth_found:
        notes.append("No tooth cells were predicted; no teeth-only meshes were written.")
    cell_colors = PALETTE[np.clip(full_pred, 0, len(PALETTE) - 1)]

    report("segmentation", 88.0, "Exporting labelled scan and individual tooth meshes")
    full_colored_ply = out / "segmented_full_colored.ply"
    colored_mesh = _unshared_mesh(points, faces, cell_colors)
    _write_mesh(full_colored_ply, colored_mesh)

    clean_teeth_ply = out / "segmented_clean_teeth.ply"
    clean_teeth_stl = out / "segmented_clean_teeth.stl"
    for stale in (clean_teeth_ply, clean_teeth_stl):
        if stale.exists():
            stale.unlink()
    if teeth_found:
        teeth_mesh = _unshared_mesh(points, faces[tooth_mask], cell_colors[tooth_mask])
        _write_mesh(clean_teeth_ply, teeth_mesh)
        teeth_mesh.merge_close_vertices(1e-6)
        teeth_mesh.remove_degenerate_triangles()
        teeth_mesh.remove_unreferenced_vertices()
        teeth_mesh.compute_vertex_normals()
        _write_mesh(clean_teeth_stl, teeth_mesh)

    # Always clear stale per-tooth files, including when the current scan has no tooth labels.
    per_tooth_meshes = _write_tooth_classes(out, points, faces, full_pred)

    seg_cloud_ply = out / "segmented_cloud.ply"
    if seg_cloud_ply.exists():
        seg_cloud_ply.unlink()
    if cloud_path is not None and Path(cloud_path).is_file():
        pcd = o3d.io.read_point_cloud(str(cloud_path))
        cloud_pts = np.asarray(pcd.points, dtype=np.float64)
        if len(cloud_pts) > 0:
            _, cloud_nn = tree.query(cloud_pts, workers=-1)
            pcd.colors = _to_o3d_colors(PALETTE[np.clip(preds_dec[cloud_nn], 0, len(PALETTE) - 1)])
            if not o3d.io.write_point_cloud(str(seg_cloud_ply), pcd):
                raise IOError(f"Failed to write segmented point cloud: {seg_cloud_ply}")

    labels_path = out / "segmentation_labels.npz"
    np.savez_compressed(
        labels_path,
        face_labels=full_pred,
        face_confidence=full_confidence,
        decimated_face_labels=preds_dec,
        checkpoint=str(ckpt_path),
        jaw=jaw,
    )
    class_counts = {str(int(c)): int(np.sum(full_pred == c)) for c in np.unique(full_pred)}

    report("segmentation", 100.0, "Tooth segmentation complete")
    return {
        "checkpoint": str(ckpt_path),
        "checkpoint_member": inner_checkpoint,
        "jaw": jaw,
        "device": device,
        "original_cells": int(len(faces)),
        "decimated_cells": int(len(dec_faces)),
        "gingiva_pct": round(gingiva_pct, 2),
        "teeth_pct": round(100.0 - gingiva_pct, 2),
        "detected_classes": [int(c) for c in np.unique(full_pred)],
        "class_face_counts": class_counts,
        "teeth_found": teeth_found,
        "warnings": notes,
        "full_colored_ply": str(full_colored_ply),
        "clean_teeth_ply": str(clean_teeth_ply) if teeth_found else None,
        "clean_teeth_stl": str(clean_teeth_stl) if clean_teeth_stl.exists() else None,
        "per_tooth_meshes": per_tooth_meshes,
        "segmentation_labels_npz": str(labels_path),
        "segmented_cloud_ply": str(seg_cloud_ply) if seg_cloud_ply.exists() else None,
    }
