from __future__ import annotations

import json

import numpy as np
import pytest

from ioscan.utils.roi import (
    default_roi_config,
    enabled_roi_points,
    load_roi_config,
    rasterize_roi,
    save_roi_config,
    validate_roi_config,
)
from ioscan.calibration.models import CameraIntrinsics, LaserPlane
from ioscan.pipeline.ridge import RidgeConfig
from ioscan.pipeline.triangulate import triangulate_frame
from tests.test_triangulation import _render_synthetic_flat_plane


def test_default_roi_is_disabled():
    cfg = default_roi_config()
    assert cfg["enabled"] is False
    assert enabled_roi_points(cfg) is None


def test_roi_config_round_trips(tmp_path):
    path = tmp_path / "configs" / "roi.json"
    payload = {
        "width": 1600, "height": 1200, "enabled": True,
        "points": [[0.2, 0.1], [0.8, 0.1], [0.8, 0.9], [0.2, 0.9]],
    }
    saved = save_roi_config(path, payload)
    loaded = load_roi_config(path)
    assert loaded == saved
    assert enabled_roi_points(loaded) == payload["points"]


def test_invalid_roi_is_rejected():
    with pytest.raises(ValueError, match="at least three"):
        validate_roi_config({"enabled": True, "points": [[0.1, 0.1], [0.9, 0.9]]})
    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_roi_config({"enabled": True, "points": [[-0.1, 0.1], [0.9, 0.1], [0.5, 0.9]]})


def test_rasterized_mask_keeps_only_polygon():
    points = [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]
    mask = rasterize_roi((120, 160), points)
    assert mask.shape == (120, 160)
    assert mask[60, 80] == 255
    assert mask[5, 5] == 0
    assert mask[60, 5] == 0


def test_triangulation_returns_only_points_inside_camera_roi():
    cam = CameraIntrinsics(width=1600, height=1200, fx=2666.7, fy=2666.7, cx=800.0, cy=600.0)
    angle = np.deg2rad(30.963756532)
    laser = LaserPlane.from_dict({"n": [0.0, float(np.sin(angle)), float(np.cos(angle))], "d": -20.0})
    on_frame, off_frame = _render_synthetic_flat_plane(cam, laser, z_depth_mm=22.0)
    roi = [[0.35, 0.0], [0.65, 0.0], [0.65, 1.0], [0.35, 1.0]]
    _, uv, _ = triangulate_frame(
        on_frame, off_frame, cam, laser,
        ridge_cfg=RidgeConfig(method="cog"), roi_polygon=roi,
    )
    assert len(uv) > 100
    assert np.all(uv[:, 0] >= 0.35 * (cam.width - 1) - 1)
    assert np.all(uv[:, 0] <= 0.65 * (cam.width - 1) + 1)
