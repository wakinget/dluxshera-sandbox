from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from dluxshera.ml import PairPolicy, generate_split_registry, load_sample_catalog, write_split_registry
from dluxshera.datasets.schema import read_json
import dluxshera.ml.training as training_module
from dluxshera.ml.models import build_pairwise_correction_model, count_parameters
from dluxshera.ml.training import (
    CHECKPOINT_SCHEMA_VERSION,
    default_s01_e00_config,
    default_s01_e01_config,
    resolve_device,
    train_pairwise_correction,
)
from tests.ml.test_catalog_splits_pairs import _write_prepared_fixture


def _fake_evaluate_losses(monkeypatch: pytest.MonkeyPatch, losses: list[float]) -> None:
    remaining = iter(losses)

    def fake_evaluate(model, loader, *, device, catalog, fisher_distance_bin_edges=None):
        loss = float(next(remaining))
        rmse = float(np.sqrt(loss))
        metrics = {
            "sample_count": 1,
            "fisher_overall_rmse": rmse,
            "fisher_per_dim_rmse": [rmse] * catalog.science_dim,
            "by_eval_slice": {},
            "by_pair_family": {},
            "by_distance_bin": {
                "bin_edges": [0.0, 1.0],
                "below_range_count": 0,
                "above_range_count": 0,
                "outside_range_count": 0,
                "bins": {"0-1": {"sample_count": 1, "fisher_overall_rmse": rmse}},
            },
        }
        predictions = {
            "pair_record_id": np.asarray(["pair"], dtype=str),
            "eval_slice": np.asarray(["slice"], dtype=str),
            "pair_family": np.asarray(["same_nuisance_different_science"], dtype=str),
            "fisher_distance_l2": np.asarray([0.5], dtype=np.float32),
            "y_pred_z": np.zeros((1, catalog.science_dim), dtype=np.float32),
            "y_true_z": np.zeros((1, catalog.science_dim), dtype=np.float32),
        }
        return metrics, predictions

    monkeypatch.setattr(training_module, "_evaluate", fake_evaluate)


def _tiny_training_inputs(tmp_path: Path) -> tuple[Path, Path, dict]:
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
    config["training"]["pairs_per_epoch"] = 16
    config["training"]["batch_size"] = 4
    config["training"]["num_workers"] = 0
    config["validation"]["pairs_per_slice"] = 4
    return prepared, split_path, config


def _history_epochs(path: Path) -> list[int]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [int(row["epoch"]) for row in csv.DictReader(handle)]


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
    assert e00["pair_policy"]["policy_id"] == "s01_clean_same_pair_grid_v1"

    e01 = default_s01_e01_config()
    assert (e01["study_id"], e01["experiment_id"], e01["run_id"]) == (
        "S01",
        "S01-E01",
        "S01-E01-R001",
    )
    assert e01["pair_policy"]["policy_id"] == "s01_clean_same_pair_grid_v1"
    assert e01["training"]["epochs"] == 100
    assert e01["training"]["early_stopping"]["enabled"] is True


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
    assert manifest["validation_manifest_identity"]["sha256"]
    assert manifest["runtime"]["resolved_device"] == "cpu"
    assert manifest["early_stopping"]["enabled"] is False
    assert manifest["training_pair_stream"] == {
        "pairs_per_epoch_ordered": 16,
        "include_reverse": True,
        "reverse_pair_augmentation": True,
        "base_pairs_per_epoch": 8,
    }


def test_training_checkpoint_loads_with_explicit_weights_only_true(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, split_path, config = _tiny_training_inputs(tmp_path)
    run_dir = tmp_path / "run"
    config["training"]["epochs"] = 1
    _fake_evaluate_losses(monkeypatch, [1.0])
    train_pairwise_correction(
        config=config,
        prepared_root=prepared,
        split_registry_path=split_path,
        output_dir=run_dir,
    )

    checkpoint_path = run_dir / "checkpoint_last.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert isinstance(checkpoint["runtime_provenance"]["torch_version"], str)
    assert isinstance(checkpoint["config"]["runtime"]["torch_version"], str)

    get_unsafe = getattr(torch.serialization, "get_unsafe_globals_in_checkpoint", None)
    if get_unsafe is not None:
        assert get_unsafe(checkpoint_path) == []


def test_same_run_resume_preserves_history_and_best_when_no_new_best(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, split_path, config = _tiny_training_inputs(tmp_path)
    run_dir = tmp_path / "run"
    config["training"]["epochs"] = 2
    _fake_evaluate_losses(monkeypatch, [1.0, 0.9])
    train_pairwise_correction(
        config=config,
        prepared_root=prepared,
        split_registry_path=split_path,
        output_dir=run_dir,
    )
    assert _history_epochs(run_dir / "history.csv") == [0, 1]
    metrics_before = read_json(run_dir / "metrics.json")
    best_identity = read_json(run_dir / "run_manifest.json")["validation_manifest_identity"]

    resume_config = dict(config)
    resume_config["training"] = dict(config["training"])
    resume_config["training"]["epochs"] = 3
    resume_config["resume_checkpoint"] = str(run_dir / "checkpoint_last.pt")
    _fake_evaluate_losses(monkeypatch, [1.1])
    summary = train_pairwise_correction(
        config=resume_config,
        prepared_root=prepared,
        split_registry_path=split_path,
        output_dir=run_dir,
    )

    assert _history_epochs(run_dir / "history.csv") == [0, 1, 2]
    assert summary["best_epoch"] == 1
    assert summary["best_validation_loss"] == pytest.approx(0.9)
    metrics_after = read_json(run_dir / "metrics.json")
    assert metrics_after["best_epoch"] == 1
    assert metrics_after["validation"] == metrics_before["validation"]
    assert (run_dir / "checkpoint_best.pt").exists()
    assert (run_dir / "evaluation_predictions.npz").exists()
    assert read_json(run_dir / "run_manifest.json")["validation_manifest_identity"] == best_identity


def test_same_run_resume_updates_best_when_new_best_occurs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, split_path, config = _tiny_training_inputs(tmp_path)
    run_dir = tmp_path / "run"
    config["training"]["epochs"] = 1
    _fake_evaluate_losses(monkeypatch, [1.0])
    train_pairwise_correction(
        config=config,
        prepared_root=prepared,
        split_registry_path=split_path,
        output_dir=run_dir,
    )

    resume_config = dict(config)
    resume_config["training"] = dict(config["training"])
    resume_config["training"]["epochs"] = 2
    resume_config["resume_checkpoint"] = str(run_dir / "checkpoint_last.pt")
    _fake_evaluate_losses(monkeypatch, [0.5])
    summary = train_pairwise_correction(
        config=resume_config,
        prepared_root=prepared,
        split_registry_path=split_path,
        output_dir=run_dir,
    )
    assert summary["best_epoch"] == 1
    assert summary["best_validation_loss"] == pytest.approx(0.5)
    assert _history_epochs(run_dir / "history.csv") == [0, 1]


def test_resume_rejects_changed_validation_manifest_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, split_path, config = _tiny_training_inputs(tmp_path)
    run_dir = tmp_path / "run"
    config["training"]["epochs"] = 1
    _fake_evaluate_losses(monkeypatch, [1.0])
    train_pairwise_correction(
        config=config,
        prepared_root=prepared,
        split_registry_path=split_path,
        output_dir=run_dir,
    )

    other_validation = run_dir / "other_validation_pairs"
    catalog = load_sample_catalog(prepared)
    registry = generate_split_registry(
        catalog,
        seed=7,
        science_fractions={"train": 1.0},
        nuisance_fractions={"train": 1.0},
    )
    from dluxshera.ml import generate_frozen_pair_manifest, write_pair_manifest

    write_pair_manifest(
        other_validation,
        generate_frozen_pair_manifest(
            catalog,
            registry,
            policy=PairPolicy.from_dict(config["pair_policy"]),
            split="validation",
            seed=999,
            pairs_per_slice=4,
            eval_slices=config["validation"]["eval_slices"],
        ),
    )
    resume_config = dict(config)
    resume_config["training"] = dict(config["training"])
    resume_config["training"]["epochs"] = 2
    resume_config["resume_checkpoint"] = str(run_dir / "checkpoint_last.pt")
    with pytest.raises(ValueError, match="scientific identity"):
        train_pairwise_correction(
            config=resume_config,
            prepared_root=prepared,
            split_registry_path=split_path,
            output_dir=run_dir,
            validation_manifest_path=other_validation,
        )
