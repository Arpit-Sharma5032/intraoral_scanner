"""Teeth3DS mesh access. Data stays outside the repo.

Expected layout (as on disk):
    <root>/<jaw>/<CASE_ID>/<CASE_ID>_<jaw>.obj
    <root>/<jaw>/<CASE_ID>/<CASE_ID>_<jaw>.json   (optional per-vertex labels)
where jaw is "lower" or "upper". Meshes are in millimetres.

Set the root once (Windows, then open a NEW terminal):
    setx IOSCAN_TEETH3DS "C:\\path\\to\\Teeth3DS"
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

ENV_VAR = "IOSCAN_TEETH3DS"
JAWS = ("lower", "upper")


@dataclass(frozen=True)
class Case:
    case_id: str
    jaw: str
    obj_path: Path
    json_path: Path | None


def resolve_root(root: str | Path | None = None) -> Path:
    """Dataset root from the argument or the IOSCAN_TEETH3DS env var."""
    r = root or os.environ.get(ENV_VAR)
    if not r:
        raise FileNotFoundError(
            f"Teeth3DS root not set. Pass root=... or set the {ENV_VAR} environment variable."
        )
    p = Path(r).expanduser()
    if not p.is_dir():
        raise FileNotFoundError(f"Teeth3DS root does not exist: {p}")
    return p


def list_cases(root: str | Path | None = None, jaw: str = "lower") -> list[Case]:
    """All cases for one jaw, sorted by ID."""
    if jaw not in JAWS:
        raise ValueError(f"jaw must be one of {JAWS}, got {jaw!r}")
    base = resolve_root(root) / jaw
    if not base.is_dir():
        raise FileNotFoundError(f"No '{jaw}' folder under {base.parent}")
    cases: list[Case] = []
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        obj = d / f"{d.name}_{jaw}.obj"
        if obj.is_file():
            js = d / f"{d.name}_{jaw}.json"
            cases.append(Case(d.name, jaw, obj, js if js.is_file() else None))
    return cases


def load_mesh(path: str | Path, center: bool = True) -> trimesh.Trimesh:
    """Load an .obj as a single Trimesh (no vertex merging or repair).

    center=True translates the mesh so its bounding-box centre is the origin.
    """
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"Could not load a triangle mesh from {path}")
    if center:
        mesh = mesh.copy()
        mesh.apply_translation(-mesh.bounds.mean(axis=0))
    return mesh


def load_labels(json_path: str | Path) -> dict[str, np.ndarray]:
    """Per-vertex tooth labels / instances if present (used later for per-tooth metrics)."""
    with open(json_path, "r") as f:
        data = json.load(f)
    out: dict[str, np.ndarray] = {}
    for key in ("labels", "instances"):
        if key in data:
            out[key] = np.asarray(data[key], dtype=np.int32)
    return out


def describe(mesh: trimesh.Trimesh) -> dict:
    """Quick sanity stats for a mesh (mm)."""
    extents = mesh.extents
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "extents_mm": [round(float(e), 2) for e in extents],
        "area_mm2": round(float(mesh.area), 1),
        "watertight": bool(mesh.is_watertight),
        "mean_edge_mm": round(float(mesh.edges_unique_length.mean()), 3),
    }