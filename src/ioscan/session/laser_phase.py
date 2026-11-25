"""Verify the laser ON/OFF flag of every frame from the images themselves.

The recorded ``laser_on`` flag comes from pairing ESP32 strobe packets with camera frames in
order. A constant one-frame offset between the two streams (for example a buffered frame at the
start of the FFmpeg stream) swaps every flag: the pipeline then subtracts the laser frame from the
ambient frame and finds almost nothing. Session scan_20261114_205403 had exactly this.

A laser frame differs from both neighbours by a thin, bright, positive-only structure (the line on
the object and its reflections in the housing). The evidence of each frame is combined with the
fact that the hardware alternates laser and ambient frames, so a faint line, a line that misses
the surface for a while, or moving printed markers do not flip individual frames.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _read_small(path: str | Path) -> np.ndarray | None:
    p = str(path)
    img = cv2.imread(p, cv2.IMREAD_REDUCED_GRAYSCALE_4) if p.lower().endswith((".jpg", ".jpeg")) else None
    if img is None:
        full = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if full is None:
            return None
        if full.ndim == 3:
            full = cv2.cvtColor(full, cv2.COLOR_BGR2GRAY)
        if full.dtype != np.uint8:
            full = (full.astype(np.float32) * (255.0 / max(float(full.max()), 1.0))).astype(np.uint8)
        img = cv2.resize(full, (full.shape[1] // 4, full.shape[0] // 4), interpolation=cv2.INTER_AREA)
    return img.astype(np.int16)


def laser_pair_polarity(on: np.ndarray, off: np.ndarray, q: float = 99.9) -> float:
    """Positive when ``on`` holds the laser, negative when ``off`` does (DN, high percentile)."""
    a = np.asarray(on)[::4, ::4].astype(np.int16)
    b = np.asarray(off)[::4, ::4].astype(np.int16)
    return float(np.percentile(np.clip(a - b, 0, None), q) - np.percentile(np.clip(b - a, 0, None), q))


def _decode_alternation(e: np.ndarray, steps: np.ndarray, stay_cost: float = 2.5) -> np.ndarray:
    """Most likely ON(1)/OFF(0) sequence for evidence ``e`` in [-1, 1] (positive = laser).

    The firmware toggles the laser on every camera strobe, so consecutive saved frames alternate;
    only a dropped camera frame (``steps`` = 2 frame intervals) makes two neighbours equal. A
    Viterbi pass uses that structure: single frames with weak or misleading evidence (line off the
    surface, a moving marker edge) are decided by their neighbours instead of being flipped.
    """
    n = len(e)
    cost = np.stack([e, -e], axis=1)            # cost[i, state]: state 0 = OFF, 1 = ON
    back = np.zeros((n, 2), dtype=np.int8)
    acc = cost[0].copy()
    for i in range(1, n):
        same_expected = (int(steps[i]) % 2 == 0)
        new = np.empty(2)
        for st in (0, 1):
            c_same = acc[st] + (0.0 if same_expected else stay_cost)
            c_flip = acc[1 - st] + (stay_cost if same_expected else 0.0)
            if c_same < c_flip:
                new[st], back[i, st] = c_same, st
            else:
                new[st], back[i, st] = c_flip, 1 - st
        acc = new + cost[i]
    out = np.zeros(n, dtype=bool)
    st = int(np.argmin(acc))
    for i in range(n - 1, -1, -1):
        out[i] = bool(st)
        st = int(back[i, st])
    return out


def verify_laser_flags(records: list[dict], *, progress=None,
                       roi_box: tuple[float, float, float, float] | None = None) -> tuple[list[dict], dict]:
    """Return records with image-verified ``laser_on`` flags and a report of what was done.

    ``roi_box`` (x0, y0, x1, y1, normalised) limits the comparison to where the laser line can be.
    """
    n = len(records)
    info: dict = {"frames": n, "action": "kept"}
    if n < 6:
        info["reason"] = "too few frames"
        return records, info
    import os
    from concurrent.futures import ThreadPoolExecutor
    workers = max(2, min(8, os.cpu_count() or 2))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ioscan-flags") as pool:
        imgs: list[np.ndarray | None] = list(pool.map(_read_small, [r["path"] for r in records], chunksize=16))
    if progress is not None:
        progress(n, n)
    if roi_box is not None:
        cropped = []
        for im in imgs:
            if im is None:
                cropped.append(None)
                continue
            h, w = im.shape[:2]
            x0, y0 = int(max(0, roi_box[0] * w - 2)), int(max(0, roi_box[1] * h - 2))
            x1, y1 = int(min(w, roi_box[2] * w + 2)), int(min(h, roi_box[3] * h + 2))
            cropped.append(im[y0:y1, x0:x1] if (x1 - x0 >= 8 and y1 - y0 >= 8) else im)
        imgs = cropped
    # Signed laser evidence of every frame against the MEAN of its two neighbours. For a scene that
    # moves steadily the mean of the neighbours predicts the frame itself, so moving edges (teeth,
    # printed markers) largely cancel, while the laser line remains: positive in a laser frame,
    # negative in an ambient frame.
    score = np.full(n, np.nan)
    for i in range(n):
        a = imgs[i]
        if a is None:
            continue
        nb = [imgs[j] for j in (i - 1, i + 1) if 0 <= j < n and imgs[j] is not None and imgs[j].shape == a.shape]
        if not nb:
            continue
        ref = nb[0].astype(np.float32) if len(nb) == 1 else 0.5 * (nb[0].astype(np.float32) + nb[1])
        d = a.astype(np.float32) - ref
        score[i] = float(np.percentile(np.clip(d, 0, None), 99.8) - np.percentile(np.clip(-d, 0, None), 99.8))
    ok = np.isfinite(score)
    if ok.sum() < 6:
        info["reason"] = "frames unreadable"
        return records, info
    amp = float(np.percentile(np.abs(score[ok]), 85))
    info.update(score_p15=round(float(np.percentile(score[ok], 15)), 2),
                score_p85=round(float(np.percentile(score[ok], 85)), 2), amplitude=round(amp, 2))
    if amp < 5.0:
        info["reason"] = "laser/ambient frames not separable from the images (line too weak?)"
        return records, info
    e = np.where(ok, np.clip(score / amp, -1.0, 1.0), 0.0)
    t = np.array([float(r.get("timestamp_s", i)) for i, r in enumerate(records)])
    dt = np.diff(t)
    med = float(np.median(dt)) if len(dt) else 1.0
    steps = np.ones(n, dtype=np.int64)
    if med > 0:
        steps[1:] = np.clip(np.rint(dt / med), 1, 50).astype(np.int64)
    decoded = _decode_alternation(e, steps)
    support = float(np.mean(np.where(decoded, e, -e)[ok]))
    recorded = np.array([bool(r["laser_on"]) for r in records])
    agree = float((decoded == recorded).mean())
    info.update(agreement=round(agree, 4), support=round(support, 3),
                detected_on_fraction=round(float(decoded.mean()), 4),
                dropped_frame_gaps=int((steps > 1).sum()))
    if support < 0.25:
        info["reason"] = "laser/ambient frames not separable from the images (line too weak?)"
        return records, info
    if agree >= 0.999:
        return records, info
    out = [dict(r, laser_on=bool(d)) for r, d in zip(records, decoded)]
    info["action"] = "inverted" if agree <= 0.02 else "replaced"
    info["frames_changed"] = int((decoded != recorded).sum())
    return out, info
