from __future__ import annotations

from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from dluxshera.ml import PairPolicy, generate_split_registry, load_sample_catalog, write_split_registry
from dluxshera.datasets.schema import read_json
from dluxshera.ml.models import build_pairwise_correction_model, count_parameters
from dluxshera.ml.training import (
    default_s01_e00_config,
    default_s01_e01_config,
    resolve_device,
    train_pairwise_correction,
)
from tests.ml.test_catalog_splits_pairs import _write_prepared_fixture


def _patch_device_availability(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cuda: bool,
    mps: bool,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


def test_resolve_device_auto_prefers_cuda_then_mps_then_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_device_availability(monkeypatch, cuda=True, mps=True)
    assert str(resolve_device("auto")) == "cuda:0"

    _patch_device_availability(monkeypatch, cuda=False, mps=True)
    assert str(resolve_device("auto")) == "mps"

    _patch_device_availability(monkeypatch, cuda=False, mps=False)
    assert str(resolve_device("auto")) == "cpu"


def test_resolve_device_explicit_requests_do_not_silently_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_device_availability(monkeypatch, cuda=False, mps=False)
    assert str(resolve_device("cpu")) == "cpu"
    with pytest.raises(RuntimeError, match="CUDA.*not available"):
        resolve_device("cuda")
    with pytest.raises(RuntimeError, match="CUDA.*not available"):
        resolve_device("cuda:1")
    with pytest.raises(RuntimeError, match="MPS.*not available"):
        resolve_device("mps")


def test_default_training_configs_use_concise_s01_ids() -> None:
    e00 = default_s01_e00_config()
    assert (e00["study_id"], e00["experiment_id"], e00["run_id"]) == (
        "S01",
        "S01-E00",
        "S01-E00-R001",
    )
    assert e00["pair_policy"]["policy_id"] == "s01_e00_tiny_same_nuisance_v1"

    e01 = default_s01_e01_config()
    assert (e01["study_id"], e01["experiment_id"], e01["run_id"]) == (
        "S01",
        "S01-E01",
        "S01-E01-R001",
    )
    assert e01["pair_policy"]["policy_id"] == "s01_e01_clean_same_pair_grid_v1"


def test_shared_cnn_shapes_comparators_and_gradients() -> None:
    model = build_pairwise_correction_model(
        2,
        {
            "channels": [4, 8],
            "embedding_dim": 8,
            "encoder_hidden_dim": 16,
            "head_hidden_dim": 16,
            "adaptive_pool_shape": [2, 2],
            "normalization": "none",
            "comparator": "concat_diff",
        },
    )
    image_a = torch.randn(3, 1, 16, 16)
    image_b = torch.randn(3, 1, 16, 16)
    pred, h_a, h_b = model(image_a, image_b, return_embeddings=True)
    assert pred.shape == (3, 2)
    assert h_a.shape == h_b.shape == (3, 8)
    assert count_parameters(model) > 0
    loss = pred.pow(2).mean()
    loss.backward()
    assert any(param.grad is not None for param in model.encoder.parameters())

    diff_model = build_pairwise_correction_model(
        2,
        {
            "channels": [4, 8],
            "embedding_dim": 8,
            "encoder_hidden_dim": 16,
            "head_hidden_dim": 16,
            "adaptive_pool_shape": [2, 2],
            "normalization": "none",
            "comparator": "difference",
        },
    )
    assert diff_model(image_a, image_b).shape == (3, 2)
    assert diff_model.encoder is diff_model.encoder


def test_tiny_training_smoke_writes_artifacts_without_test_eval(tmp_path: Path) -> None:
    prepared = _write_prepared_fixture(tmp_path / "prepared")
    catalog = load_sample_catalog(prepared)
    registry = generate_split_registry(
        catalog,
        seed=7,
        science_fractions={"train": 1.0},
        nuisance_fractions={"train": 1.0},
    )
    split_path = tmp_path / "split.json"
    write_split_registry(split_path, registry)
    config = default_s01_e00_config()
    config["device"] = "cpu"
    config["training"]["epochs"] = 2
    config["training"]["pairs_per_epoch"] = 16
    config["training"]["batch_size"] = 4
    config["validation"]["pairs_per_slice"] = 4
    summary = train_pairwise_correction(
        config=config,
        prepared_root=prepared,
        split_registry_path=split_path,
        output_dir=tmp_path / "run",
    )
    run_dir = Path(summary["output_dir"])
    assert (run_dir / "checkpoint_best.pt").exists()
    assert (run_dir / "checkpoint_last.pt").exists()
    assert (run_dir / "history.csv").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "evaluation_predictions.npz").exists()
    assert not (run_dir / "test_predictions.npz").exists()
    manifest = read_json(run_dir / "run_manifest.json")
    assert manifest["study_id"] == "S01"
    assert manifest["experiment_id"] == "S01-E00"
    assert manifest["run_id"] == "S01-E00-R001"
    assert manifest["pair_policy"]["include_reverse"] is True
    assert manifest["training_pair_stream"] == {
        "pairs_per_epoch_ordered": 16,
        "include_reverse": True,
        "reverse_pair_augmentation": True,
        "base_pairs_per_epoch": 8,
    }
