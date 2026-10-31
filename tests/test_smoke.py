from ioscan import __version__
from ioscan.utils.config import load_yaml, repo_root


def test_version():
    assert __version__


def test_configs_load():
    cfg_dir = repo_root() / "configs"
    for name in ["camera", "laser", "sim", "pipeline"]:
        assert isinstance(load_yaml(cfg_dir / f"{name}.yaml"), dict)


def test_camera_focal_matches_lens():
    cam = load_yaml(repo_root() / "configs" / "camera.yaml")
    expected = cam["focal_length_mm"] / (cam["pixel_size_um"] / 1000.0)
    assert abs(cam["fx"] - expected) < 1.0
