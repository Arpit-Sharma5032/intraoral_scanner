"""Segmentation helpers that need no checkpoint."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ioscan.pipeline.meshsegnet_arch import BuiltinMeshSegNet
from ioscan.pipeline.segment import (
    SegmentConfig, _load_network, _torch_load_checkpoint, build_features,
    _normalized_adjacency, infer_jaw, infer_jaw_from_name, resolve_checkpoint,
    resolve_segmentation_model,
)


def test_checkpoint_resolution_never_guesses(tmp_path):
    (tmp_path / "MeshSegNet_Man_15_classes_72samples_lr1e-2_best.zip").write_bytes(b"x")
    (tmp_path / "MeshSegNet_Max_15_classes_72samples_lr1e-2_best.zip").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    cfg = SegmentConfig(models_dir=tmp_path)
    assert "_Man_" in resolve_checkpoint(cfg, "lower").name
    assert "_Max_" in resolve_checkpoint(cfg, "upper").name
    with pytest.raises(ValueError):
        resolve_checkpoint(cfg, "auto")          # unknown jaw must not silently become 'lower'
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(SegmentConfig(models_dir=tmp_path, ckpt_path=tmp_path / "typo.tar"), "lower")
    (tmp_path / "MeshSegNet_Man_15_classes_72samples_lr1e-2_best.zip").unlink()
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(cfg, "lower")         # no arbitrary candidates[0] fallback


def test_infer_jaw_from_meta(tmp_path):
    (tmp_path / "meta.json").write_text('{"config": {"case": {"jaw": "upper"}}}')
    assert infer_jaw(tmp_path) == "upper" and infer_jaw(tmp_path / "missing") is None
    (tmp_path / "meta.json").write_text('{"config": {"sim": {"teeth3ds": {"jaw": "lower"}}}}')
    assert infer_jaw(tmp_path) == "lower"
    assert infer_jaw_from_name("MeshSegNet_Man_15_classes_best.zip") == "lower"
    assert infer_jaw_from_name("MeshSegNet_Max_15_classes_best.zip") == "upper"


def test_infer_jaw_falls_back_to_session_directory_name(tmp_path):
    lower = tmp_path / "FUYJPUPX_lower"
    lower.mkdir()
    assert infer_jaw(lower) == "lower"
    upper = tmp_path / "ABCD1234_upper"
    upper.mkdir()
    assert infer_jaw(upper) == "upper"


def test_resolve_segmentation_model_uses_session_jaw_and_rejects_mismatch(tmp_path):
    (tmp_path / "meta.json").write_text('{"config": {"case": {"jaw": "lower"}}}')
    models = tmp_path / "models"
    models.mkdir()
    lower = models / "MeshSegNet_Man_15_classes_72samples_lr1e-2_best.zip"
    upper = models / "MeshSegNet_Max_15_classes_72samples_lr1e-2_best.zip"
    lower.write_bytes(b"placeholder")
    upper.write_bytes(b"placeholder")

    checkpoint, jaw = resolve_segmentation_model(
        SegmentConfig(models_dir=models), session_hint=tmp_path
    )
    assert checkpoint == lower and jaw == "lower"

    with pytest.raises(ValueError, match="Jaw mismatch"):
        resolve_segmentation_model(
            SegmentConfig(jaw="lower", ckpt_path=upper), session_hint=tmp_path
        )


def test_features_shape_and_ranges():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(60, 3)) * 10
    faces = rng.integers(0, 60, size=(100, 3))
    X = build_features(pts, faces)
    assert X.shape == (100, 15) and np.isfinite(X).all()
    assert X[:, 9:12].min() >= -1e-6 and X[:, 9:12].max() <= 1 + 1e-6   # barycenters in [0, 1]
    # Reference checkpoint expects barycentres in channels 9:12 and normals in 12:15.
    assert np.allclose(X[:, 12:15].mean(axis=0), 0.0, atol=1e-5)


def test_network_forward_and_layer_names():
    net = BuiltinMeshSegNet(num_classes=15, num_channels=15).eval()
    n = 40
    x = torch.randn(1, 15, n)
    a = torch.eye(n)[None]
    out = net(x, a, a)
    assert out.shape == (1, n, 15)
    assert torch.allclose(out.sum(-1), torch.ones(1, n), atol=1e-4)
    keys = set(net.state_dict())
    assert "output_conv.weight" in keys and "mlp3_conv1.weight" in keys and "fstn.conv1.weight" in keys
    assert "fstn.conv1.weight" in keys and "stn.conv1.weight" not in keys


def test_zip_checkpoint_container_loads_strictly(tmp_path):
    import io
    import zipfile

    reference = BuiltinMeshSegNet(num_classes=15, num_channels=15).eval()
    buffer = io.BytesIO()
    torch.save({"model_state_dict": reference.state_dict(), "epoch": 5}, buffer)
    archive_path = tmp_path / "MeshSegNet_Man_15_classes_72samples_lr1e-2_best.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("models/MeshSegNet_Man_best.tar", buffer.getvalue())

    checkpoint, member = _torch_load_checkpoint(torch, archive_path)
    assert member == "models/MeshSegNet_Man_best.tar"
    assert "model_state_dict" in checkpoint
    net, loaded_member = _load_network(
        SegmentConfig(jaw="lower", ckpt_path=archive_path), archive_path, "cpu"
    )
    assert loaded_member == member
    assert set(net.state_dict()) == set(reference.state_dict())


def test_adjacency_is_row_normalized_and_includes_self():
    coords = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=np.float32)
    adjacency = _normalized_adjacency(coords, 0.1)
    assert adjacency.shape == (3, 3)
    assert np.allclose(adjacency.sum(axis=1), 1.0)
    assert np.all(np.diag(adjacency) > 0.0)
    assert adjacency[0, 1] > 0.0 and adjacency[2, 0] == 0.0


def test_segment_config_rejects_oversized_dense_input():
    with pytest.raises(ValueError, match="between 20 and 10000"):
        SegmentConfig(target_cells=10001)
