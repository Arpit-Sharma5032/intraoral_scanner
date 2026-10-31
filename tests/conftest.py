"""Shared fixtures: a fake Teeth3DS-style dataset so tests need no real data."""
import json

import numpy as np
import pytest
import trimesh


def make_fake_arch(seed: int = 0) -> tuple[trimesh.Trimesh, np.ndarray]:
    """Horseshoe of 14 tooth-like blocks on a gingiva ribbon. Returns (mesh, per-vertex labels).

    Deliberately rotated and offset so that canonicalize() has real work to do.
    """
    rng = np.random.default_rng(seed)
    parts, labels = [], []
    angles = np.radians(np.linspace(-80, 80, 14))
    R_arch = 25.0
    for i, a in enumerate(angles):
        h = 8.0 + rng.uniform(-1.5, 1.5)
        box = trimesh.creation.box(extents=(7.0, 8.0, h))  # tangential, radial, height
        box.apply_translation((0, 0, h / 2))
        rot = trimesh.transformations.rotation_matrix(a, (0, 0, 1))
        box.apply_transform(rot)
        box.apply_translation((R_arch * np.sin(a), R_arch * np.cos(a) - R_arch, 0))
        parts.append(box)
        labels.append(np.full(len(box.vertices), i + 1))
    # gingiva ribbon under the teeth (z in [-4, 0])
    for a in np.radians(np.linspace(-85, 85, 40)):
        g = trimesh.creation.box(extents=(5.0, 13.0, 4.0))
        g.apply_translation((0, 0, -2.0))
        g.apply_transform(trimesh.transformations.rotation_matrix(a, (0, 0, 1)))
        g.apply_translation((R_arch * np.sin(a), R_arch * np.cos(a) - R_arch, 0))
        parts.append(g)
        labels.append(np.zeros(len(g.vertices), dtype=int))
    mesh = trimesh.util.concatenate(parts)
    lab = np.concatenate(labels).astype(np.int32)
    # arbitrary rigid motion so the arch is NOT already canonical
    M = trimesh.transformations.random_rotation_matrix(rng.random(3))
    M[:3, 3] = (10.0, -20.0, 5.0)
    mesh.apply_transform(M)
    return mesh, lab


@pytest.fixture()
def fake_dataset(tmp_path):
    mesh, lab = make_fake_arch()
    case_dir = tmp_path / "lower" / "FAKE0001"
    case_dir.mkdir(parents=True)
    mesh.export(str(case_dir / "FAKE0001_lower.obj"))
    with open(case_dir / "FAKE0001_lower.json", "w") as f:
        json.dump({"labels": lab.tolist(), "instances": lab.tolist()}, f)
    return tmp_path
