from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ANALYSIS_DIR = Path(__file__).resolve().parents[2] / "work" / "experiments" / "ml" / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

from campaign_analysis import (  # noqa: E402
    compute_initializer_metrics,
    compute_prediction_geometry,
    load_campaign,
    time_to_thresholds,
)


PARAMETERS = ("source.contrast", "optics.primary.zernike_coeffs_nm[0]")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_run(
    root: Path,
    *,
    run_id: str = "S05-E01-R001",
    study_id: str = "S05",
    experiment_id: str = "S05-E01",
    seed: int = 11,
    complete: bool = True,
    scheduler_history: bool = False,
    source_commit: str = "abc123",
    validation_sha: str = "validation-sha",
) -> Path:
    run_dir = root / "nested" / experiment_id / run_id
    manifest = {
        "schema_version": "dluxshera_ml_run_manifest/1",
        "study_id": study_id,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "seed": seed,
        "git": {"source_commit": source_commit, "source_archive_id": "archive"},
        "prepared_dataset": {"artifact_id": "PREP", "prepared_dataset_hash": "prep-sha"},
        "split_registry": {"artifact_id": "SPLIT", "content_sha256": "split-sha"},
        "validation_manifest_identity": {"sha256": validation_sha},
        "test_evaluated": False,
        "model": {
            "comparator": "concat_diff",
            "channels": [4, 8],
            "embedding_dim": 16,
            "encoder_hidden_dim": 32,
            "head_hidden_dim": 32,
            "normalization": "batch",
            "adaptive_pool_shape": [2, 2],
            "parameter_count": 1234,
        },
        "training": {
            "optimizer": "adamw",
            "weight_decay": 1.0e-4,
            "learning_rate": 5.0e-4,
            "epochs": 3,
        },
        "early_stopping": {
            "epochs_completed": 3,
            "early_stopped": False,
            "reached_max_epochs": True,
        },
        "best_epoch": 1,
        "best_validation_loss": 2.5,
    }
    if scheduler_history:
        manifest["training"]["lr_scheduler"] = {"name": "reduce_on_plateau", "factor": 0.5}
        manifest["optimization"] = {
            "initial_learning_rate": 5.0e-4,
            "final_learning_rate": 2.5e-4,
            "lr_scheduler": {"name": "reduce_on_plateau", "factor": 0.5},
            "lr_reduction_count": 1,
            "epochs_completed": 3,
            "best_epoch": 1,
            "early_stopped": False,
            "reached_max_epochs": True,
        }
    config = {
        "study_id": study_id,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "seed": seed,
        "validation_artifact": "validation",
        "test_artifact": "test",
        "evaluate_test": False,
        "dataset": {"artifact_id": "PREP", "prepared_dataset_hash": "prep-sha"},
        "training": manifest["training"],
        "evaluation": {"fisher_distance_bin_edges": [0.0, 1.0, 2.0]},
        "model": manifest["model"],
    }
    y_true = np.asarray([[1.0, 2.0], [0.0, 2.0], [2.0, 0.0], [0.0, 0.0]])
    y_pred = np.asarray([[0.5, 2.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    residual = y_pred - y_true
    metrics = {
        "schema_version": "dluxshera_ml_metrics/2",
        "best_epoch": 1,
        "best_validation_loss": float(np.mean(residual**2)),
        "validation": {
            "fisher_overall_rmse": float(np.sqrt(np.mean(residual**2))),
            "fisher_per_parameter_rmse": {
                label: float(np.sqrt(np.mean(residual[:, idx] ** 2)))
                for idx, label in enumerate(PARAMETERS)
            },
            "physical_per_parameter_rmse": {
                "source.contrast": 0.1,
                "optics.primary.zernike_coeffs_nm[0]": 0.2,
            },
        },
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    if not complete:
        return run_dir
    _write_json(run_dir / "run_config_resolved.json", config)
    _write_json(run_dir / "metrics.json", metrics)
    if scheduler_history:
        history = pd.DataFrame(
            {
                "epoch": [0, 1, 2],
                "train_loss": [10.0, 8.0, 7.0],
                "validation_loss": [10000.0, 8100.0, 6400.0],
                "validation_overall_rmse": [100.0, 90.0, 80.0],
                "epoch_seconds": [1.0, 2.0, 3.0],
                "is_best": [True, True, True],
                "early_stopping_bad_epochs": [0, 0, 0],
                "learning_rate": [5.0e-4, 5.0e-4, 2.5e-4],
                "learning_rate_next": [5.0e-4, 2.5e-4, 2.5e-4],
                "lr_reduced": [False, True, False],
            }
        )
    else:
        history = pd.DataFrame(
            {
                "epoch": [0, 1, 2],
                "train_loss": [10.0, 8.0, 7.0],
                "validation_loss": [10000.0, 8100.0, 6400.0],
                "validation_overall_rmse": [100.0, 90.0, 80.0],
                "epoch_seconds": [1.0, 2.0, 3.0],
                "is_best": [True, True, True],
                "early_stopping_bad_epochs": [0, 0, 0],
            }
        )
    history.to_csv(run_dir / "history.csv", index=False)
    np.savez(
        run_dir / "evaluation_predictions.npz",
        pair_record_id=np.asarray(["a", "b", "c", "d"]),
        eval_slice=np.asarray(
            [
                "heldout_science_seen_nuisance",
                "heldout_science_heldout_nuisance",
                "heldout_science_seen_nuisance",
                "heldout_science_heldout_nuisance",
            ]
        ),
        pair_family=np.asarray(["same"] * 4),
        fisher_distance_l2=np.asarray([0.0, 0.999, 1.0, 2.0]),
        y_true_z=y_true,
        y_pred_z=y_pred,
    )
    return run_dir


def test_discovery_run_table_and_fixed_lr_history(tmp_path: Path) -> None:
    _write_run(tmp_path)
    (tmp_path / "unrelated.txt").write_text("ignore", encoding="utf-8")
    campaign = load_campaign([tmp_path])

    assert campaign.runs["run_id"].tolist() == ["S05-E01-R001"]
    row = campaign.runs.iloc[0]
    assert row["study_id"] == "S05"
    assert row["experiment_id"] == "S05-E01"
    assert row["seed"] == 11
    assert row["status"] == "complete"
    assert row["source_commit"] == "abc123"
    assert row["prepared_dataset_hash"] == "prep-sha"
    assert row["validation_manifest_sha256"] == "validation-sha"
    assert row["model_comparator"] == "concat_diff"
    assert row["optimizer"] == "adamw"
    assert row["initial_learning_rate"] == pytest.approx(5.0e-4)
    assert set(campaign.history["learning_rate_source"]) == {"config_fixed"}
    assert campaign.history["learning_rate"].tolist() == [5.0e-4, 5.0e-4, 5.0e-4]
    assert campaign.history["epoch"].tolist() == [0, 1, 2]
    assert campaign.history["epoch_number"].tolist() == [1, 2, 3]


def test_partial_run_and_scheduler_history(tmp_path: Path) -> None:
    _write_run(tmp_path / "partial", run_id="partial", complete=False)
    _write_run(tmp_path / "scheduled", run_id="scheduled", scheduler_history=True)

    campaign = load_campaign([tmp_path])

    assert dict(zip(campaign.runs["run_id"], campaign.runs["status"])) == {
        "partial": "incomplete",
        "scheduled": "complete",
    }
    scheduled = campaign.history[campaign.history["run_id"] == "scheduled"]
    assert set(scheduled["learning_rate_source"]) == {"history"}
    assert scheduled["learning_rate"].tolist() == [5.0e-4, 5.0e-4, 2.5e-4]
    assert scheduled["lr_reduced"].tolist() == [False, True, False]


def test_core_metrics_parameter_slices_bins_and_geometry(tmp_path: Path) -> None:
    _write_run(tmp_path)
    campaign = load_campaign([tmp_path])

    truth = np.asarray([[1.0, 2.0], [0.0, 2.0], [2.0, 0.0], [0.0, 0.0]])
    pred = np.asarray([[0.5, 2.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    expected = compute_initializer_metrics(truth, pred)
    row = campaign.runs.iloc[0]
    assert row["zero_baseline_fisher_rmse"] == pytest.approx(expected["baseline_rmse"])
    assert row["best_fisher_rmse"] == pytest.approx(expected["model_rmse"])
    assert row["mse_skill"] == pytest.approx(expected["mse_skill"])
    assert row["rmse_reduction"] == pytest.approx(expected["rmse_reduction"])

    parameter = campaign.parameters.set_index("parameter")
    assert parameter.loc["source.contrast", "fisher_mse_skill"] == pytest.approx(0.75)
    assert parameter.loc["optics.primary.zernike_coeffs_nm[0]", "parameter_family"] == "M1"
    assert parameter.loc["optics.primary.zernike_coeffs_nm[0]", "physical_unit"] == "nm"

    slices = campaign.slices.set_index("eval_slice")
    assert slices.loc["heldout_science_seen_nuisance", "sample_count"] == 2
    assert np.isfinite(slices.loc["heldout_science_seen_nuisance", "mse_skill"])

    bins = campaign.distance_bins.set_index("distance_bin")
    assert bins.loc["0-1", "sample_count"] == 2
    assert bins.loc["1-2", "sample_count"] == 2
    assert bins.loc["1-2", "distance_bin_includes_hi"]

    predictions = campaign.load_predictions("S05-E01-R001")
    assert predictions.geometry.shape[0] == 4
    geom = compute_prediction_geometry(np.asarray([[0.0, 0.0], [1.0, 0.0]]), np.asarray([[1.0, 0.0], [1.0, 0.0]]))
    assert np.isnan(geom["cosine_alignment"][0])
    assert np.isnan(geom["correction_norm_ratio"][0])
    assert np.isnan(geom["relative_residual_norm"][0])
    assert geom["cosine_alignment"][1] == pytest.approx(1.0)


def test_duplicate_resolution_and_conflict_detection(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    durable = tmp_path / "durable"
    _write_run(scratch, run_id="dup", complete=False)
    _write_run(durable, run_id="dup", complete=True)
    campaign = load_campaign([scratch, durable])
    assert campaign.runs.iloc[0]["status"] == "complete"
    assert "durable" in campaign.runs.iloc[0]["result_path"]

    conflict = tmp_path / "conflict"
    _write_run(conflict, run_id="dup", complete=True, source_commit="different")
    with pytest.raises(ValueError, match="Conflicting scientific identity"):
        load_campaign([durable, conflict])


def test_time_to_thresholds(tmp_path: Path) -> None:
    _write_run(tmp_path)
    campaign = load_campaign([tmp_path])
    thresholds = time_to_thresholds(campaign.history, [95.0, 75.0])
    reached = thresholds.set_index("threshold")
    assert reached.loc[95.0, "epoch"] == 1
    assert reached.loc[95.0, "epoch_number"] == 2
    assert reached.loc[95.0, "cumulative_seconds"] == pytest.approx(3.0)
    assert np.isnan(reached.loc[75.0, "epoch"])
    assert np.isnan(reached.loc[75.0, "epoch_number"])
