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
    add_physical_display_columns,
    classify_metric_deltas,
    compute_initializer_metrics,
    compute_prediction_geometry,
    dataset_family_summary_table,
    distance_bin_delta_table,
    effect_size_summary_table,
    experiment_summary_table,
    load_campaign,
    matched_seed_delta_table,
    residual_error_budget_table,
    time_to_thresholds,
    validation_cutoff_table,
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
    include_physical_metrics: bool = True,
    include_dataset_family: bool = False,
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
    validation_metrics = {
        "fisher_overall_rmse": float(np.sqrt(np.mean(residual**2))),
        "fisher_per_parameter_rmse": {
            label: float(np.sqrt(np.mean(residual[:, idx] ** 2)))
            for idx, label in enumerate(PARAMETERS)
        },
    }
    if include_physical_metrics:
        validation_metrics.update(
            {
                "physical_per_parameter_mae": {
                    "source.contrast": 0.05,
                    "optics.primary.zernike_coeffs_nm[0]": 0.1,
                    "source.separation_as": 0.0015,
                },
                "physical_per_parameter_rmse": {
                    "source.contrast": 0.1,
                    "optics.primary.zernike_coeffs_nm[0]": 0.2,
                    "source.separation_as": 0.002,
                },
            }
        )
    metrics = {
        "schema_version": "dluxshera_ml_metrics/2",
        "best_epoch": 1,
        "best_validation_loss": float(np.mean(residual**2)),
        "validation": validation_metrics,
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    if not complete:
        return run_dir
    _write_json(run_dir / "run_config_resolved.json", config)
    if include_dataset_family:
        validation_metrics["by_dataset_family"] = {
            "joint": {
                "sample_count": 2,
                "physical_per_parameter_rmse": {"source.separation_as": 0.003},
                "physical_per_parameter_mae": {"source.separation_as": 0.002},
                "fisher_per_parameter_rmse": {"source.separation_as": 1.0},
            },
            "radial": {
                "sample_count": 2,
                "physical_per_parameter_rmse": {"source.separation_as": 0.004},
                "physical_per_parameter_mae": {"source.separation_as": 0.0025},
                "fisher_per_parameter_rmse": {"source.separation_as": 2.0},
            },
        }
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
    if include_dataset_family:
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
            dataset_family=np.asarray(["joint", "joint", "radial", "radial"]),
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
    assert parameter.loc["source.contrast", "physical_mae"] == pytest.approx(0.05)
    assert parameter.loc["optics.primary.zernike_coeffs_nm[0]", "parameter_family"] == "M1"
    assert parameter.loc["optics.primary.zernike_coeffs_nm[0]", "physical_unit"] == "nm"
    assert row["separation_rmse_mas"] == pytest.approx(2.0)
    assert row["separation_mae_mas"] == pytest.approx(1.5)
    assert row["separation_physical_unit"] == "mas"

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


def test_physical_display_columns_convert_arcsec_to_mas() -> None:
    parameters = pd.DataFrame(
        {
            "parameter": ["source.separation_as", "optics.primary.zernike_coeffs_nm[0]"],
            "physical_rmse": [0.003, 2.0],
            "physical_mae": [0.002, 1.5],
            "physical_unit": ["arcsec", "nm"],
        }
    )

    converted = add_physical_display_columns(parameters)

    assert converted.loc[0, "physical_rmse_display"] == pytest.approx(3.0)
    assert converted.loc[0, "physical_mae_display"] == pytest.approx(2.0)
    assert converted.loc[0, "physical_display_unit"] == "mas"
    assert converted.loc[1, "physical_rmse_display"] == pytest.approx(2.0)
    assert converted.loc[1, "physical_display_unit"] == "nm"


def test_missing_physical_metrics_degrade_to_nan(tmp_path: Path) -> None:
    _write_run(tmp_path, include_physical_metrics=False)
    campaign = load_campaign([tmp_path])

    row = campaign.runs.iloc[0]
    assert np.isnan(row["separation_rmse_mas"])
    assert np.isnan(row["separation_mae_mas"])
    assert campaign.parameters["physical_rmse"].isna().all()
    assert campaign.parameters["physical_mae"].isna().all()


def test_experiment_summary_and_matched_seed_deltas() -> None:
    runs = pd.DataFrame(
        {
            "study_id": ["S", "S", "S", "S"],
            "experiment_id": ["E01", "E01", "E02", "E02"],
            "run_id": ["a1", "a2", "b1", "b2"],
            "seed": [1, 2, 1, 2],
            "status": ["complete", "complete", "complete", "partial"],
            "best_fisher_rmse": [10.0, 12.0, 9.0, 15.0],
            "separation_rmse_mas": [4.0, 6.0, 3.0, 7.0],
        }
    )

    summary = experiment_summary_table(runs, metrics=["best_fisher_rmse", "separation_rmse_mas"])
    indexed = summary.set_index("experiment_id")
    assert indexed.loc["E01", "run_count"] == 2
    assert indexed.loc["E02", "complete_count"] == 1
    assert indexed.loc["E01", "best_fisher_rmse_mean"] == pytest.approx(11.0)
    assert indexed.loc["E02", "separation_rmse_mas_mean"] == pytest.approx(5.0)

    deltas = matched_seed_delta_table(runs, "E01", "E02", metrics=["best_fisher_rmse", "separation_rmse_mas"])
    by_seed = deltas.set_index("seed")
    assert by_seed.loc[1, "best_fisher_rmse_delta"] == pytest.approx(-1.0)
    assert by_seed.loc[2, "separation_rmse_mas_delta"] == pytest.approx(1.0)


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

    cutoffs = validation_cutoff_table(campaign.history, [2, 4])
    cutoff_index = cutoffs.set_index("cutoff_epoch")
    assert cutoff_index.loc[2.0, "epoch_number"] == 2
    assert cutoff_index.loc[2.0, "metric_value"] == pytest.approx(90.0)
    assert cutoff_index.loc[4.0, "epoch_number"] == 3
    assert cutoff_index.loc[4.0, "best_metric_so_far"] == pytest.approx(80.0)


def test_classify_metric_deltas_respects_direction_and_tolerance() -> None:
    assert classify_metric_deltas([-1.0, -0.5], higher_is_better=False) == "all improved"
    assert classify_metric_deltas([1.0, 0.5], higher_is_better=False) == "all worsened"
    assert classify_metric_deltas([-1.0, 0.5], higher_is_better=False) == "mixed"
    assert classify_metric_deltas([1.0e-10, -1.0e-10], higher_is_better=False) == "effectively tied"
    assert classify_metric_deltas([0.1, 0.2], higher_is_better=True) == "all improved"
    assert classify_metric_deltas([-0.1, -0.2], higher_is_better=True) == "all worsened"


def test_effect_size_summary_uses_metric_directionality() -> None:
    runs = pd.DataFrame(
        {
            "experiment_id": ["E01", "E01", "E02", "E02"],
            "run_id": ["a1", "a2", "b1", "b2"],
            "seed": [1, 2, 1, 2],
            "best_fisher_rmse": [10.0, 12.0, 9.0, 11.0],
            "mse_skill": [0.5, 0.6, 0.55, 0.65],
        }
    )

    summary = effect_size_summary_table(
        runs,
        [("E01", "E02")],
        metrics=["best_fisher_rmse", "mse_skill"],
        higher_is_better={"best_fisher_rmse": False, "mse_skill": True},
    ).set_index("metric")

    assert summary.loc["best_fisher_rmse", "absolute_delta"] == pytest.approx(-1.0)
    assert summary.loc["best_fisher_rmse", "improvement_percent"] == pytest.approx(100.0 / 11.0)
    assert summary.loc["best_fisher_rmse", "matched_delta_classification"] == "all improved"
    assert summary.loc["mse_skill", "absolute_delta"] == pytest.approx(0.05)
    assert summary.loc["mse_skill", "matched_delta_classification"] == "all improved"


def test_distance_bin_delta_table() -> None:
    bins = pd.DataFrame(
        {
            "experiment_id": ["E01", "E01", "E02", "E02"],
            "distance_bin": ["0-1", "1-2", "0-1", "1-2"],
            "distance_bin_lo": [0.0, 1.0, 0.0, 1.0],
            "distance_bin_hi": [1.0, 2.0, 1.0, 2.0],
            "model_fisher_rmse": [10.0, 20.0, 8.0, 25.0],
            "mse_skill": [0.1, 0.2, 0.3, 0.1],
            "sample_count": [5, 10, 5, 10],
        }
    )

    deltas = distance_bin_delta_table(
        bins,
        [("E01", "E02")],
        metrics=["model_fisher_rmse", "mse_skill"],
        higher_is_better={"model_fisher_rmse": False, "mse_skill": True},
    )
    indexed = deltas.set_index(["distance_bin", "metric"])

    assert indexed.loc[("0-1", "model_fisher_rmse"), "delta"] == pytest.approx(-2.0)
    assert indexed.loc[("0-1", "model_fisher_rmse"), "improvement_delta"] == pytest.approx(2.0)
    assert indexed.loc[("1-2", "mse_skill"), "delta"] == pytest.approx(-0.1)
    assert indexed.loc[("1-2", "mse_skill"), "improvement_delta"] == pytest.approx(-0.1)


def test_residual_error_budget_normalizes_within_run() -> None:
    parameters = pd.DataFrame(
        {
            "study_id": ["S", "S"],
            "experiment_id": ["E", "E"],
            "run_id": ["r", "r"],
            "seed": [1, 1],
            "parameter_index": [0, 1],
            "parameter": ["p0", "p1"],
            "parameter_display": ["p0", "p1"],
            "fisher_model_rmse": [3.0, 4.0],
            "fisher_mse_skill": [0.1, 0.2],
        }
    )

    budget = residual_error_budget_table(parameters).set_index("parameter")

    assert budget.loc["p1", "residual_mse_proxy"] == pytest.approx(16.0)
    assert budget.loc["p0", "fraction_of_residual_budget"] == pytest.approx(9.0 / 25.0)
    assert budget["fraction_of_residual_budget"].sum() == pytest.approx(1.0)
    assert budget.iloc[-1]["cumulative_fraction"] == pytest.approx(1.0)


def test_dataset_family_summary_missing_and_present_metadata(tmp_path: Path) -> None:
    _write_run(tmp_path / "missing")
    missing = load_campaign([tmp_path / "missing"])
    assert dataset_family_summary_table(missing.run_records).empty

    _write_run(tmp_path / "present", include_dataset_family=True)
    present = load_campaign([tmp_path / "present"])
    summary = dataset_family_summary_table(present.run_records).set_index("dataset_family")

    assert set(summary.index) == {"joint", "radial"}
    assert summary.loc["joint", "sample_count"] == 2
    assert np.isfinite(summary.loc["joint", "mse_skill"])
    assert summary.loc["joint", "separation_rmse_mas"] == pytest.approx(3.0)
