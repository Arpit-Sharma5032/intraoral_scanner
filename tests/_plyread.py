"""Minimal binary PLY reader for tests (no Open3D)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def read_ply_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break
        body = f.read()
    nv = nf = 0
    props: list[tuple[str, str]] = []
    current = None
    for line in header:
        parts = line.split()
        if parts[0] == "element":
            current = parts[1]
            if current == "vertex":
                nv = int(parts[2])
            elif current == "face":
                nf = int(parts[2])
        elif parts[0] == "property" and current == "vertex":
            props.append((parts[2], parts[1]))
    tmap = {"float": "<f4", "double": "<f8"}
    vdt = np.dtype([(n, tmap[t]) for n, t in props])
    v = np.frombuffer(body, dtype=vdt, count=nv)
    V = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    fdt = np.dtype([("n", "u1"), ("i", "<i4", 3)])
    F = np.frombuffer(body[nv * vdt.itemsize:], dtype=fdt, count=nf)["i"].astype(np.int64)
    return V, F
