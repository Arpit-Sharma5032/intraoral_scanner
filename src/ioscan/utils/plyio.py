"""Minimal binary/ASCII PLY reader (vertices + triangle faces), independent of Open3D."""
from __future__ import annotations

from pathlib import Path

import numpy as np

_TYPES = {"char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1", "short": "<i2", "ushort": "<u2",
          "int16": "<i2", "uint16": "<u2", "int": "<i4", "uint": "<u4", "int32": "<i4", "uint32": "<u4",
          "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8"}


def read_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (vertices (N,3) float64, faces (M,3) int64 or None)."""
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: truncated PLY header")
            line = line.decode("ascii", "replace").strip()
            header.append(line)
            if line == "end_header":
                break
        body = f.read()
    fmt = "binary_little_endian"
    elements: list[list] = []
    for line in header:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            elements.append([parts[1], int(parts[2]), []])
        elif parts[0] == "property" and elements:
            elements[-1][2].append(parts[1:])
    if fmt == "ascii":
        return _read_ascii(body.decode("ascii", "replace"), elements)
    if fmt != "binary_little_endian":
        raise ValueError(f"{path}: unsupported PLY format {fmt}")
    V = None
    F = None
    off = 0
    for name, count, props in elements:
        if any(p[0] == "list" for p in props):
            if name != "face" or len(props) != 1:
                raise ValueError(f"{path}: unsupported list element {name}")
            _, ct, it, _ = props[0]
            cdt, idt = np.dtype(_TYPES[ct]), np.dtype(_TYPES[it])
            # fast path: all triangles
            rec = np.dtype([("n", cdt), ("i", idt, 3)])
            arr = np.frombuffer(body, dtype=rec, count=count, offset=off)
            if np.all(arr["n"] == 3):
                F = arr["i"].astype(np.int64)
                off += count * rec.itemsize
            else:
                raise ValueError(f"{path}: non-triangle faces are not supported")
        else:
            dt = np.dtype([(p[1], _TYPES[p[0]]) for p in props])
            arr = np.frombuffer(body, dtype=dt, count=count, offset=off)
            off += count * dt.itemsize
            if name == "vertex":
                V = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float64)
    if V is None:
        raise ValueError(f"{path}: no vertex element")
    return V, F


def _read_ascii(text: str, elements) -> tuple[np.ndarray, np.ndarray | None]:
    lines = iter(text.split("\n"))
    V = F = None
    for name, count, props in elements:
        rows = [next(lines).split() for _ in range(count)]
        if name == "vertex":
            names = [p[1] for p in props]
            a = np.array(rows, dtype=np.float64)
            V = a[:, [names.index("x"), names.index("y"), names.index("z")]]
        elif name == "face":
            F = np.array([[int(v) for v in r[1:4]] for r in rows], dtype=np.int64)
    return V, F
