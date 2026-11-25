"""Volumetric fusion of laser profiles for scans that see walls (tilted plate) and for merged scans.

A height map stores one height per (x, y) and so cannot hold the outer and inner walls of the teeth:
a scan made with the plate tilted to look at those walls collapses them into stripes and holes.
Here every measured point instead updates a truncated signed distance field (TSDF) along its own
viewing ray, the fields of all profiles are averaged, and the zero level is extracted as a mesh
(naive surface nets), smoothed lightly and cleaned of small fragments.

Only numpy and scipy are used, so it runs wherever the rest of the pipeline runs.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def tsdf_fuse(P: np.ndarray, U: np.ndarray, voxel: float = 0.35, trunc: float = 1.2,
              pad: float = 1.5, chunk: int = 150000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse surface points P (N,3) seen along unit vectors U (N,3, from the point towards the camera).

    Each point writes signed distances s in [-trunc, +trunc] along its ray (positive in front of the
    surface, i.e. outside the model), splatted trilinearly into the eight neighbouring voxels so
    that neighbouring profiles a few tenths of a millimetre apart still meet. Returns
    (sdf, weight, origin) with sdf/weight shaped (nx, ny, nz).
    """
    P = np.asarray(P, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)
    lo = P.min(axis=0) - trunc - pad
    hi = P.max(axis=0) + trunc + pad
    dims = (np.ceil((hi - lo) / voxel).astype(np.int64) + 2)
    nv = int(np.prod(dims))
    s = np.arange(-trunc, trunc + 1e-9, 0.7 * voxel)
    wt_s = np.where(s < 0, np.clip(1.0 + s / trunc, 0.2, 1.0), 1.0)       # trust less behind the surface
    sdf_sum = np.zeros(nv, dtype=np.float64)
    w_sum = np.zeros(nv, dtype=np.float64)
    for a in range(0, len(P), chunk):
        X = P[a:a + chunk, None, :] + s[None, :, None] * U[a:a + chunk, None, :]
        g = (X - lo) / voxel
        base = np.floor(g).astype(np.int64)
        fr = g - base
        val = np.broadcast_to(s[None, :], base.shape[:2])
        w0 = np.broadcast_to(wt_s[None, :], base.shape[:2])
        for dx in (0, 1):
            wx = fr[..., 0] if dx else 1.0 - fr[..., 0]
            for dy in (0, 1):
                wy = fr[..., 1] if dy else 1.0 - fr[..., 1]
                for dz in (0, 1):
                    wz = fr[..., 2] if dz else 1.0 - fr[..., 2]
                    w = (w0 * wx * wy * wz).ravel()
                    lin = (((base[..., 0] + dx) * dims[1] + base[..., 1] + dy) * dims[2] + base[..., 2] + dz).ravel()
                    sdf_sum += np.bincount(lin, weights=val.ravel() * w, minlength=nv)
                    w_sum += np.bincount(lin, weights=w, minlength=nv)
    sdf = np.where(w_sum > 0, sdf_sum / np.maximum(w_sum, 1e-12), trunc).astype(np.float32)
    return sdf.reshape(dims), w_sum.astype(np.float32).reshape(dims), lo


def surface_nets(sdf: np.ndarray, weight: np.ndarray, origin: np.ndarray, voxel: float,
                 min_weight: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Zero level of the field as a triangle mesh (naive surface nets), outward-facing triangles.

    One vertex per voxel cube whose corners change sign (at the mean of the edge crossings), one quad
    per grid edge that changes sign. Only voxels with enough weight take part, so the surface stops
    where nothing was measured instead of closing over unseen space.
    """
    nx, ny, nz = sdf.shape
    valid = weight >= min_weight
    inside = sdf < 0
    corners = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]

    def sub(arr, a, b, c):
        return arr[a:nx - 1 + a, b:ny - 1 + b, c:nz - 1 + c]

    allvalid = np.ones((nx - 1, ny - 1, nz - 1), dtype=bool)
    anyin = np.zeros_like(allvalid)
    allin = np.ones_like(allvalid)
    for a, b, c in corners:
        allvalid &= sub(valid, a, b, c)
        ins = sub(inside, a, b, c)
        anyin |= ins
        allin &= ins
    ci, cj, ck = np.nonzero(allvalid & anyin & ~allin)
    if len(ci) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    vid = -np.ones((nx - 1, ny - 1, nz - 1), dtype=np.int64)
    vid[ci, cj, ck] = np.arange(len(ci))
    edges = [((0, 0, 0), (1, 0, 0)), ((0, 1, 0), (1, 1, 0)), ((0, 0, 1), (1, 0, 1)), ((0, 1, 1), (1, 1, 1)),
             ((0, 0, 0), (0, 1, 0)), ((1, 0, 0), (1, 1, 0)), ((0, 0, 1), (0, 1, 1)), ((1, 0, 1), (1, 1, 1)),
             ((0, 0, 0), (0, 0, 1)), ((1, 0, 0), (1, 0, 1)), ((0, 1, 0), (0, 1, 1)), ((1, 1, 0), (1, 1, 1))]
    acc = np.zeros((len(ci), 3))
    cnt = np.zeros(len(ci))
    for p, q in edges:
        s0 = sdf[ci + p[0], cj + p[1], ck + p[2]].astype(np.float64)
        s1 = sdf[ci + q[0], cj + q[1], ck + q[2]].astype(np.float64)
        m = (s0 < 0) != (s1 < 0)
        t = np.where(m, s0 / np.where(m, s0 - s1, 1.0), 0.0)
        pos = np.stack([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]), p[2] + t * (q[2] - p[2])], axis=1)
        acc[m] += pos[m]
        cnt[m] += 1
    V = (np.stack([ci, cj, ck], axis=1) + acc / np.maximum(cnt, 1)[:, None]) * voxel + origin
    faces = []
    for ax in range(3):
        o = [0, 0, 0]
        o[ax] = 1
        a0 = inside[:nx - o[0], :ny - o[1], :nz - o[2]]
        a1 = inside[o[0]:, o[1]:, o[2]:]
        both = valid[:nx - o[0], :ny - o[1], :nz - o[2]] & valid[o[0]:, o[1]:, o[2]:]
        ei, ej, ek = np.nonzero((a0 != a1) & both)
        flip = a0[ei, ej, ek]
        u_, v_ = [x for x in range(3) if x != ax]
        quad = []
        for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
            idx = [ei.copy(), ej.copy(), ek.copy()]
            idx[u_] -= du
            idx[v_] -= dv
            ok = ((idx[0] >= 0) & (idx[1] >= 0) & (idx[2] >= 0)
                  & (idx[0] < nx - 1) & (idx[1] < ny - 1) & (idx[2] < nz - 1))
            qv = np.full(len(ei), -1, dtype=np.int64)
            qv[ok] = vid[idx[0][ok], idx[1][ok], idx[2][ok]]
            quad.append(qv)
        Q = np.stack(quad, axis=1)
        good = (Q >= 0).all(axis=1)
        Q, fl = Q[good], flip[good]
        if ax == 1:
            fl = ~fl
        Q = np.where(fl[:, None], Q, Q[:, ::-1])          # outward (towards positive distance)
        faces.append(np.vstack([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]]))
    F = np.vstack(faces)
    return V, F


def _adjacency(nv: int, F: np.ndarray) -> sparse.csr_matrix:
    i = np.concatenate([F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]])
    j = np.concatenate([F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]])
    A = sparse.csr_matrix((np.ones(len(i)), (i, j)), shape=(nv, nv))
    A.data[:] = 1.0
    return A


def taubin_smooth(V: np.ndarray, F: np.ndarray, iterations: int = 6, lam: float = 0.5, mu: float = -0.53) -> np.ndarray:
    """Low-pass the surface without shrinking it (Taubin): removes voxel steps and profile ripples."""
    if len(F) == 0:
        return V
    A = _adjacency(len(V), F)
    deg = np.asarray(A.sum(axis=1)).ravel()
    Dinv = sparse.diags(1.0 / np.maximum(deg, 1.0))
    W = Dinv @ A
    V = V.copy()
    for _ in range(iterations):
        V = V + lam * (W @ V - V)
        V = V + mu * (W @ V - V)
    return V


def drop_small_pieces(V: np.ndarray, F: np.ndarray, min_fraction: float = 0.01) -> tuple[np.ndarray, np.ndarray, dict]:
    """Remove disconnected fragments smaller than ``min_fraction`` of the faces (fins from stray profiles)."""
    if len(F) == 0:
        return V, F, {"pieces": 0, "dropped_faces": 0}
    n, lab = connected_components(_adjacency(len(V), F), directed=False)
    flab = lab[F[:, 0]]
    sizes = np.bincount(flab, minlength=n)
    keep_lab = sizes >= max(min_fraction * len(F), 50)
    keep_lab[np.argmax(sizes)] = True
    keepf = keep_lab[flab]
    F2 = F[keepf]
    used = np.zeros(len(V), dtype=bool)
    used[F2.ravel()] = True
    remap = -np.ones(len(V), dtype=np.int64)
    remap[used] = np.arange(used.sum())
    return V[used], remap[F2], {"pieces": int(n), "kept_pieces": int(keep_lab.sum()),
                                "dropped_faces": int((~keepf).sum())}


def vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    N = np.zeros_like(V)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)


def surface_area(V: np.ndarray, F: np.ndarray) -> float:
    if len(F) == 0:
        return 0.0
    return float(0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1).sum())


def sample_field(field: np.ndarray, origin: np.ndarray, voxel: float, X: np.ndarray) -> np.ndarray:
    """Trilinear sample of a voxel field at points X (outside the grid: nan)."""
    g = (np.asarray(X, dtype=np.float64) - origin) / voxel
    b = np.floor(g).astype(np.int64)
    f = g - b
    dims = np.array(field.shape)
    ok = np.all((b >= 0) & (b < dims - 1), axis=1)
    out = np.full(len(X), np.nan)
    b, f = b[ok], f[ok]
    acc = np.zeros(len(b))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                acc += w * field[b[:, 0] + dx, b[:, 1] + dy, b[:, 2] + dz]
    out[ok] = acc
    return out


def fuse_views(P: np.ndarray, U: np.ndarray, voxel: float = 0.4, trunc: float = 1.5,
               min_weight: float = 0.3, smooth_iterations: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Points + viewing directions -> smoothed, cleaned, outward-facing mesh (V, F, N, info).

    ``info["point_to_surface_mm"]`` is the median distance of the measured points from the fused
    surface: how well the overlapping passes agree (the 3-D counterpart of the height-map check).
    """
    sdf, w, lo = tsdf_fuse(P, U, voxel=voxel, trunc=trunc)
    V, F = surface_nets(sdf, w, lo, voxel, min_weight=min_weight)
    V, F, pieces = drop_small_pieces(V, F)
    V = taubin_smooth(V, F, iterations=smooth_iterations)
    N = vertex_normals(V, F)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(P), size=min(len(P), 50000), replace=False) if len(P) else np.zeros(0, dtype=np.int64)
    d = np.abs(sample_field(sdf, lo, voxel, P[pick]))
    d = d[np.isfinite(d) & (d < trunc * 0.999)]
    info = {"voxel_mm": voxel, "truncation_mm": trunc, "grid": list(map(int, sdf.shape)),
            "vertices": int(len(V)), "faces": int(len(F)), "surface_mm2": round(surface_area(V, F), 1),
            "point_to_surface_mm": round(float(np.median(d)), 3) if len(d) else None, **pieces}
    return V, F, N, info
