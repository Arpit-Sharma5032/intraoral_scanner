"""Persistent, resolution-independent camera region-of-interest masks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROI_VERSION = 1
DEFAULT_WIDTH = 1600
DEFAULT_HEIGHT = 1200
MAX_ROI_POINTS = 128


def default_roi_config(width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT) -> dict[str, Any]:
    return {
        "version": ROI_VERSION,
        "width": int(width),
        "height": int(height),
        "enabled": False,
        "points": [],
    }


def validate_roi_config(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize an ROI payload. Coordinates are normalized to [0, 1]."""
    if not isinstance(payload, dict):
        raise ValueError("ROI configuration must be a JSON object.")
    width = int(payload.get("width", DEFAULT_WIDTH))
    height = int(payload.get("height", DEFAULT_HEIGHT))
    if width < 1 or height < 1 or width > 20000 or height > 20000:
        raise ValueError("ROI image dimensions are invalid.")
    enabled = bool(payload.get("enabled", False))
    raw_points = payload.get("points", [])
    if not isinstance(raw_points, list):
        raise ValueError("ROI points must be a list of [x, y] coordinates.")
    if len(raw_points) > MAX_ROI_POINTS:
        raise ValueError(f"ROI can contain at most {MAX_ROI_POINTS} vertices.")
    points: list[list[float]] = []
    for point in raw_points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("Each ROI vertex must contain exactly two coordinates.")
        x, y = float(point[0]), float(point[1])
        if not np.isfinite([x, y]).all() or not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError("ROI coordinates must be finite values between 0 and 1.")
        points.append([x, y])
    if enabled:
        if len(points) < 3:
            raise ValueError("Draw at least three ROI vertices before enabling the mask.")
        arr = np.asarray(points, dtype=np.float64)
        area2 = abs(np.dot(arr[:, 0], np.roll(arr[:, 1], -1)) - np.dot(arr[:, 1], np.roll(arr[:, 0], -1)))
        if area2 < 1e-5:
            raise ValueError("ROI polygon is too small or its vertices are collinear.")
    return {"version": ROI_VERSION, "width": width, "height": height, "enabled": enabled, "points": points}


def load_roi_config(path: str | Path | None) -> dict[str, Any]:
    if path is None or not Path(path).is_file():
        return default_roi_config()
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return validate_roi_config(data)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid ROI configuration at {path}: {exc}") from exc


def save_roi_config(path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    config = validate_roi_config(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    tmp.replace(target)
    return config


def rasterize_roi(shape: tuple[int, int], points: list[list[float]] | np.ndarray) -> np.ndarray:
    """Return a uint8 mask (255 inside, 0 outside) for normalized polygon points."""
    height, width = int(shape[0]), int(shape[1])
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 3:
        raise ValueError("ROI polygon must have at least three [x, y] vertices.")
    if not np.isfinite(arr).all() or np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError("ROI polygon coordinates must be finite and normalized to [0, 1].")
    pixel_points = np.rint(arr * np.array([width - 1, height - 1])).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pixel_points], 255)
    return mask


def enabled_roi_points(config: dict[str, Any]) -> list[list[float]] | None:
    validated = validate_roi_config(config)
    return validated["points"] if validated["enabled"] else None
