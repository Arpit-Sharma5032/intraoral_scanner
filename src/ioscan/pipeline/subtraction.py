from __future__ import annotations
import numpy as np


def subtract_ambient(
    on_frame: np.ndarray,
    off_frame: np.ndarray | None = None,
    noise_floor: float = 15.0,
) -> np.ndarray:
    """Compute clean laser intensity image (ON - OFF) clamped at noise_floor."""
    diff = on_frame.astype(np.float32)
    if off_frame is not None:
        diff = diff - off_frame.astype(np.float32)
    diff[diff < noise_floor] = 0.0
    return diff


def classify_laser_frame(
    frame: np.ndarray,
    p999_threshold: float = 120.0,
    min_bright_pixels: int = 150,
) -> bool:
    """
    Fallback classifier when hardware trigger flags are unavailable.
    Uses P99.9 and bright pixel count rather than P99.5 (since laser covers ~0.4% of frame).
    """
    p999 = float(np.percentile(frame, 99.9))
    if p999 < p999_threshold:
        return False
    bright_count = int(np.count_nonzero(frame > (p999 * 0.5)))
    return bright_count >= min_bright_pixels