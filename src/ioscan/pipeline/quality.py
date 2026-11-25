"""Geometric plausibility checks for a reconstructed dental surface.

The previous gate only required ``extent >= 35 mm`` and ``faces >= 5000``. A smeared ribbon swept
along a wrong pose path passes both (a real failing scan had a 100 x 113 x 205 mm "arch" and was
still sent to MeshSegNet). A dental scan must also be *compact* and *connected*.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

MIN_FACES = 5000
MIN_EXTENT_MM = 35.0
# A full adult arch is ~60 mm wide / ~50 mm deep; allow generous margin for a typodont base.
MAX_EXTENT_MM = 110.0
MIN_LARGEST_COMPONENT_FRACTION = 0.60
MAX_PLAUSIBLE_PATH_MM = 150.0


def component_face_counts(triangles: np.ndarray) -> np.ndarray:
    """Face count of each vertex-connected component (descending)."""
    tri = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    if tri.size == 0:
        return np.zeros(0, dtype=np.int64)
    n = int(tri.max()) + 1
    rows = np.concatenate([tri[:, 0], tri[:, 1], tri[:, 2]])
    cols = np.concatenate([tri[:, 1], tri[:, 2], tri[:, 0]])
    graph = coo_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    face_labels = labels[tri[:, 0]]
    return np.sort(np.bincount(face_labels))[::-1]


def assess_scan_geometry(
    vertices: np.ndarray,
    triangles: np.ndarray,
    path_mm: float | None = None,
) -> dict:
    """Return ``{"ok": bool, "reasons": [str], "metrics": {...}}`` for a reconstructed mesh."""
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    tri = np.asarray(triangles).reshape(-1, 3)
    reasons: list[str] = []
    extent = (v.max(axis=0) - v.min(axis=0)) if len(v) else np.zeros(3)
    max_extent = float(extent.max())
    counts = component_face_counts(tri)
    largest_frac = float(counts[0] / counts.sum()) if counts.size else 0.0
    # Extent of the dominant component (stray far-away fragments should not decide this).
    metrics = {
        "faces": int(len(tri)),
        "extent_mm": [round(float(e), 3) for e in extent],
        "max_extent_mm": round(max_extent, 3),
        "component_count": int(counts.size),
        "largest_component_face_fraction": round(largest_frac, 4),
        "path_mm": None if path_mm is None else round(float(path_mm), 2),
    }
    if len(tri) < MIN_FACES:
        reasons.append(f"only {len(tri):,} faces (need >= {MIN_FACES:,})")
    if max_extent < MIN_EXTENT_MM:
        reasons.append(f"extent {max_extent:.1f} mm is below a dental arch (>= {MIN_EXTENT_MM:.0f} mm)")
    if max_extent > MAX_EXTENT_MM:
        reasons.append(
            f"extent {max_extent:.1f} mm exceeds any dental arch (<= {MAX_EXTENT_MM:.0f} mm): "
            "the pose path is smearing the surface (wrong translation/orientation), not a bigger object"
        )
    if counts.size and largest_frac < MIN_LARGEST_COMPONENT_FRACTION:
        reasons.append(
            f"largest connected piece holds only {100 * largest_frac:.0f}% of the faces "
            f"({counts.size} fragments)"
        )
    if path_mm is not None and path_mm > MAX_PLAUSIBLE_PATH_MM:
        reasons.append(
            f"assumed scan path is {path_mm:.0f} mm; a dental pass is well under {MAX_PLAUSIBLE_PATH_MM:.0f} mm, "
            "so the assumed scan speed is wrong"
        )
    return {"ok": not reasons, "reasons": reasons, "metrics": metrics}
