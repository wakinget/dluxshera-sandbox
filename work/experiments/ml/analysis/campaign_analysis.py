from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "CampaignData",
    "RunRecord",
    "PredictionData",
    "compute_initializer_metrics",
    "compute_prediction_geometry",
    "evaluation_contract_summary",
    "load_campaign",
    "time_to_thresholds",
]

REQUIRED_COMPLETE_ARTIFACTS = (
    "run_manifest.json",
    "run_config_resolved.json",
    "history.csv",
    "metrics.json",
    "evaluation_predictions.npz",
)

EPS = 1.0e-12


@dataclass(frozen=True)
class RunRecord:
    """Describe one discovered local run artifact directory."""

    run_id: str
    result_path: Path
    artifact_state: str
    artifacts: dict[str, bool]
    manifest: dict[str, Any]
    config: dict[str, Any] | None
    metrics: dict[str, Any] | None
    site: str | None


@dataclass(frozen=True)
class PredictionData:
    """Hold prediction arrays and per-pair initializer geometry for one run."""

    run_id: str
    arrays: dict[str, np.ndarray]
    geometry: pd.DataFrame


@dataclass
class CampaignData:
    """Container for normalized ML campaign analysis tables.

    Use ``load_predictions`` when detailed pair-level diagnostics are needed.
    The campaign table build does not concatenate raw predictions across runs.
    """

    runs: pd.DataFrame
    history: pd.DataFrame
    parameters: pd.DataFrame
    slices: pd.DataFrame
    distance_bins: pd.DataFrame
    run_records: dict[str, RunRecord]
    sync_manifests: pd.DataFrame
    warnings: list[str]

    def load_predictions(self, run_id: str) -> PredictionData:
        """Load raw prediction arrays and geometry for a selected run."""

        if run_id not in self.run_records:
            raise KeyError(f"Unknown run_id {run_id!r}.")
        record = self.run_records[run_id]
        path = record.result_path / "evaluation_predictions.npz"
        if not path.exists():
            raise FileNotFoundError(f"No evaluation_predictions.npz for {run_id}.")
        arrays = _load_prediction_arrays(path)
        geometry = prediction_geometry_table(run_id, arrays)
        return PredictionData(run_id=run_id, arrays=arrays, geometry=geometry)


def load_campaign(
    result_roots: Sequence[str | Path],
    *,
    strict_duplicates: bool = True,
) -> CampaignData:
    """Discover result artifacts and build normalized campaign tables.

    Runs are discovered recursively by ``run_manifest.json``. Optional artifacts
    are inspected in the same directory and missing files produce partial rows
    rather than import failures.
    """

    roots = [Path(root) for root in result_roots]
    warnings_out: list[str] = []
    discovered = _discover_runs(roots)
    selected_records = _deduplicate_runs(discovered, strict=strict_duplicates)
    run_records = {record.run_id: record for record in selected_records}

    run_rows: list[dict[str, Any]] = []
    history_tables: list[pd.DataFrame] = []
    parameter_tables: list[pd.DataFrame] = []
    slice_tables: list[pd.DataFrame] = []
    distance_tables: list[pd.DataFrame] = []

    for record in selected_records:
        pred_metrics: dict[str, Any] | None = None
        arrays: dict[str, np.ndarray] | None = None
        if record.artifacts.get("evaluation_predictions.npz", False):
            arrays = _load_prediction_arrays(record.result_path / "evaluation_predictions.npz")
            pred_metrics = compute_initializer_metrics(arrays["y_true_z"], arrays["y_pred_z"])
            warnings_out.extend(_metric_compatibility_warnings(record, pred_metrics))

        run_rows.append(_run_row(record, pred_metrics, arrays))

        if record.artifacts.get("history.csv", False):
            history_tables.append(_history_table(record))
        if arrays is not None:
            parameter_tables.append(_parameter_table(record, arrays))
            slice_tables.append(_slice_table(record, arrays))
            distance_tables.append(_distance_bin_table(record, arrays))

    runs = pd.DataFrame(run_rows)
    if not runs.empty:
        runs = runs.sort_values(["study_id", "experiment_id", "run_id"], na_position="last")
    history = _concat_or_empty(history_tables)
    parameters = _concat_or_empty(parameter_tables)
    slices = _concat_or_empty(slice_tables)
    distance_bins = _concat_or_empty(distance_tables)
    sync_manifests = _load_sync_manifests(roots)

    return CampaignData(
        runs=runs.reset_index(drop=True),
        history=history.reset_index(drop=True),
        parameters=parameters.reset_index(drop=True),
        slices=slices.reset_index(drop=True),
        distance_bins=distance_bins.reset_index(drop=True),
        run_records=run_records,
        sync_manifests=sync_manifests.reset_index(drop=True),
        warnings=warnings_out,
    )


def compute_initializer_metrics(y_true_z: np.ndarray, y_pred_z: np.ndarray) -> dict[str, Any]:
    """Compute zero-correction baseline and learned-correction metrics."""

    truth = np.asarray(y_true_z, dtype=np.float64)
    pred = np.asarray(y_pred_z, dtype=np.float64)
    _validate_prediction_shapes(truth, pred)
    residual = pred - truth
    baseline_mse = float(np.mean(truth**2))
    model_mse = float(np.mean(residual**2))
    baseline_rmse = math.sqrt(baseline_mse)
    model_rmse = math.sqrt(model_mse)
    return {
        "sample_count": int(truth.shape[0]),
        "parameter_count": int(truth.shape[1]),
        "baseline_mse": baseline_mse,
        "model_mse": model_mse,
        "baseline_rmse": baseline_rmse,
        "model_rmse": model_rmse,
        "mse_skill": _safe_skill(baseline_mse, model_mse),
        "rmse_reduction": _safe_skill(baseline_rmse, model_rmse),
    }


def compute_prediction_geometry(y_true_z: np.ndarray, y_pred_z: np.ndarray) -> dict[str, np.ndarray]:
    """Compute per-pair Fisher-space correction geometry.

    Ratios and cosines are reported as ``NaN`` when the truth or prediction norm
    needed by the denominator is zero to numerical precision.
    """

    truth = np.asarray(y_true_z, dtype=np.float64)
    pred = np.asarray(y_pred_z, dtype=np.float64)
    _validate_prediction_shapes(truth, pred)
    residual = pred - truth
    truth_norm = np.linalg.norm(truth, axis=1)
    pred_norm = np.linalg.norm(pred, axis=1)
    residual_norm = np.linalg.norm(residual, axis=1)
    dot = np.sum(pred * truth, axis=1)

    cosine = np.full(truth_norm.shape, np.nan, dtype=np.float64)
    denom = pred_norm * truth_norm
    mask = denom > EPS
    cosine[mask] = dot[mask] / denom[mask]

    norm_ratio = np.full(truth_norm.shape, np.nan, dtype=np.float64)
    residual_ratio = np.full(truth_norm.shape, np.nan, dtype=np.float64)
    truth_mask = truth_norm > EPS
    norm_ratio[truth_mask] = pred_norm[truth_mask] / truth_norm[truth_mask]
    residual_ratio[truth_mask] = residual_norm[truth_mask] / truth_norm[truth_mask]
    return {
        "truth_norm": truth_norm,
        "pred_norm": pred_norm,
        "residual_norm": residual_norm,
        "cosine_alignment": cosine,
        "correction_norm_ratio": norm_ratio,
        "relative_residual_norm": residual_ratio,
    }


def prediction_geometry_table(run_id: str, arrays: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Return a pair-level geometry table for one run's prediction artifact."""

    geom = compute_prediction_geometry(arrays["y_true_z"], arrays["y_pred_z"])
    rows: dict[str, Any] = {"run_id": run_id, **geom}
    for name in ("pair_record_id", "eval_slice", "pair_family", "fisher_distance_l2"):
        if name in arrays:
            rows[name] = arrays[name]
    return pd.DataFrame(rows)


def time_to_thresholds(
    history: pd.DataFrame,
    thresholds: Sequence[float],
    *,
    metric_col: str = "validation_overall_rmse",
) -> pd.DataFrame:
    """Return first epoch and elapsed wall time reaching each RMSE threshold."""

    rows: list[dict[str, Any]] = []
    if history.empty:
        return pd.DataFrame(
            columns=[
                "run_id",
                "threshold",
                "epoch",
                "epoch_number",
                "cumulative_seconds",
                "cumulative_minutes",
            ]
        )
    for run_id, group in history.sort_values(["run_id", "epoch"]).groupby("run_id"):
        values = pd.to_numeric(group.get(metric_col), errors="coerce")
        for threshold in thresholds:
            hit = group.loc[values <= float(threshold)]
            if hit.empty:
                rows.append(
                    {
                        "run_id": run_id,
                        "threshold": float(threshold),
                        "epoch": np.nan,
                        "epoch_number": np.nan,
                        "cumulative_seconds": np.nan,
                        "cumulative_minutes": np.nan,
                    }
                )
            else:
                first = hit.iloc[0]
                rows.append(
                    {
                        "run_id": run_id,
                        "threshold": float(threshold),
                        "epoch": int(first["epoch"]),
                        "epoch_number": int(first["epoch_number"])
                        if not pd.isna(first.get("epoch_number"))
                        else int(first["epoch"]) + 1,
                        "cumulative_seconds": first.get("cumulative_seconds", np.nan),
                        "cumulative_minutes": first.get("cumulative_minutes", np.nan),
                    }
                )
    return pd.DataFrame(rows)


def evaluation_contract_summary(
    runs: pd.DataFrame,
    *,
    run_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Summarize evaluation/provenance identities for selected runs."""

    if runs.empty:
        return pd.DataFrame()
    subset = runs if run_ids is None else runs[runs["run_id"].isin(run_ids)]
    columns = [
        "run_id",
        "study_id",
        "experiment_id",
        "source_commit",
        "source_archive_id",
        "prepared_dataset_hash",
        "split_registry_content_sha256",
        "validation_manifest_sha256",
        "test_manifest_sha256",
        "test_evaluated",
    ]
    return subset[[col for col in columns if col in subset.columns]].copy()


def _discover_runs(result_roots: Sequence[Path]) -> list[RunRecord]:
    records: list[RunRecord] = []
    for root in result_roots:
        if not root.exists():
            continue
        for manifest_path in sorted(root.rglob("run_manifest.json")):
            run_dir = manifest_path.parent
            manifest = _read_json(manifest_path)
            config = _maybe_read_json(run_dir / "run_config_resolved.json")
            metrics = _maybe_read_json(run_dir / "metrics.json")
            artifacts = {name: (run_dir / name).exists() for name in REQUIRED_COMPLETE_ARTIFACTS}
            run_id = str(
                manifest.get("run_id")
                or (config or {}).get("run_id")
                or run_dir.name
            )
            records.append(
                RunRecord(
                    run_id=run_id,
                    result_path=run_dir,
                    artifact_state=_artifact_state(artifacts),
                    artifacts=artifacts,
                    manifest=manifest,
                    config=config,
                    metrics=metrics,
                    site=_infer_site(run_dir, result_roots),
                )
            )
    return records


def _deduplicate_runs(records: Sequence[RunRecord], *, strict: bool) -> list[RunRecord]:
    grouped: dict[str, list[RunRecord]] = {}
    for record in records:
        grouped.setdefault(record.run_id, []).append(record)

    selected: list[RunRecord] = []
    for run_id, group in grouped.items():
        identities = {_identity_tuple(record) for record in group}
        if len(identities) > 1:
            message = f"Conflicting scientific identity for duplicate run_id {run_id!r}."
            if strict:
                raise ValueError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        selected.append(sorted(group, key=_duplicate_rank, reverse=True)[0])
    return sorted(selected, key=lambda record: (str(record.manifest.get("study_id")), str(record.run_id)))


def _duplicate_rank(record: RunRecord) -> tuple[int, int, float, str]:
    state_rank = {"complete": 3, "partial": 2, "incomplete": 1}.get(record.artifact_state, 0)
    path_text = str(record.result_path).lower()
    durable_rank = 0 if ("/scratch/" in path_text or "scratch" in record.result_path.parts) else 1
    newest = max(
        ((record.result_path / name).stat().st_mtime for name, present in record.artifacts.items() if present),
        default=(record.result_path / "run_manifest.json").stat().st_mtime,
    )
    return state_rank, durable_rank, newest, str(record.result_path)


def _artifact_state(artifacts: Mapping[str, bool]) -> str:
    if all(artifacts.values()):
        return "complete"
    present = sum(bool(value) for value in artifacts.values())
    return "partial" if present > 1 else "incomplete"


def _run_row(
    record: RunRecord,
    pred_metrics: Mapping[str, Any] | None,
    arrays: Mapping[str, np.ndarray] | None,
) -> dict[str, Any]:
    manifest = record.manifest
    config = record.config or {}
    metrics = record.metrics or {}
    training = _mapping(config.get("training")) or _mapping(manifest.get("training"))
    model = _mapping(manifest.get("model")) or _mapping(config.get("model"))
    early = _mapping(manifest.get("early_stopping")) or _mapping(metrics.get("early_stopping"))
    optimization = _mapping(manifest.get("optimization")) or _mapping(metrics.get("optimization"))
    git = _mapping(manifest.get("git"))
    prepared = _mapping(manifest.get("prepared_dataset")) or _mapping(config.get("dataset"))
    split = _mapping(manifest.get("split_registry"))
    validation_identity = _mapping(manifest.get("validation_manifest_identity"))
    test_identity = _mapping(manifest.get("test_manifest_identity"))

    validation = _mapping(metrics.get("validation"))
    best_loss = _first_not_none(metrics.get("best_validation_loss"), manifest.get("best_validation_loss"))
    best_rmse = pred_metrics.get("model_rmse") if pred_metrics else validation.get("fisher_overall_rmse")
    lr_scheduler = _mapping(training.get("lr_scheduler")) or _mapping(optimization.get("lr_scheduler"))
    scheduler_name = lr_scheduler.get("name")

    baseline_rmse = None if pred_metrics is None else pred_metrics.get("baseline_rmse")
    model_rmse = None if pred_metrics is None else pred_metrics.get("model_rmse")
    heldout_slice = _slice_metrics_for(arrays, "heldout_science_heldout_nuisance")
    seen_slice = _slice_metrics_for(arrays, "heldout_science_seen_nuisance")
    return {
        "site": record.site,
        "study_id": _first_not_none(manifest.get("study_id"), config.get("study_id")),
        "experiment_id": _first_not_none(manifest.get("experiment_id"), config.get("experiment_id")),
        "run_id": record.run_id,
        "seed": _first_not_none(config.get("seed"), manifest.get("seed")),
        "status": record.artifact_state,
        "artifact_state": record.artifact_state,
        "result_path": str(record.result_path),
        "source_commit": _first_not_none(git.get("source_commit"), git.get("commit")),
        "source_archive_id": git.get("source_archive_id"),
        "prepared_dataset_hash": prepared.get("prepared_dataset_hash"),
        "split_registry_artifact_id": split.get("artifact_id"),
        "split_registry_content_sha256": split.get("content_sha256"),
        "validation_artifact": config.get("validation_artifact"),
        "validation_manifest_sha256": validation_identity.get("sha256"),
        "validation_manifest_identity": validation_identity or None,
        "test_artifact": config.get("test_artifact"),
        "test_manifest_sha256": test_identity.get("sha256"),
        "test_evaluated": bool(manifest.get("test_evaluated", False)),
        "model_comparator": model.get("comparator"),
        "channels": tuple(model.get("channels", ())) if model.get("channels") is not None else None,
        "embedding_dim": model.get("embedding_dim"),
        "encoder_hidden_dim": model.get("encoder_hidden_dim"),
        "head_hidden_dim": model.get("head_hidden_dim"),
        "normalization": model.get("normalization"),
        "adaptive_pool_shape": tuple(model.get("adaptive_pool_shape", ()))
        if model.get("adaptive_pool_shape") is not None
        else None,
        "model_parameter_count": model.get("parameter_count"),
        "optimizer": training.get("optimizer"),
        "weight_decay": training.get("weight_decay"),
        "initial_learning_rate": _first_not_none(
            optimization.get("initial_learning_rate"),
            training.get("learning_rate"),
        ),
        "lr_scheduler_name": scheduler_name if scheduler_name is not None else "none",
        "max_epochs": training.get("epochs"),
        "epochs_completed": _first_not_none(optimization.get("epochs_completed"), early.get("epochs_completed")),
        "best_epoch": _first_not_none(optimization.get("best_epoch"), metrics.get("best_epoch"), manifest.get("best_epoch")),
        "early_stopped": _first_not_none(optimization.get("early_stopped"), early.get("early_stopped")),
        "reached_max_epochs": _first_not_none(optimization.get("reached_max_epochs"), early.get("reached_max_epochs")),
        "final_learning_rate": optimization.get("final_learning_rate"),
        "lr_reduction_count": optimization.get("lr_reduction_count"),
        "total_training_seconds": _history_total_seconds(record),
        "mean_epoch_seconds": _history_mean_epoch_seconds(record),
        "best_validation_loss": best_loss,
        "best_fisher_rmse": best_rmse,
        "zero_baseline_fisher_rmse": baseline_rmse,
        "learned_fisher_rmse": model_rmse,
        "rmse_reduction": None if pred_metrics is None else pred_metrics.get("rmse_reduction"),
        "mse_skill": None if pred_metrics is None else pred_metrics.get("mse_skill"),
        "heldout_nuisance_fisher_rmse": _first_not_none(
            heldout_slice.get("model_rmse"),
            _serialized_slice_value(validation, "heldout_science_heldout_nuisance", "fisher_overall_rmse"),
        ),
        "seen_nuisance_fisher_rmse": _first_not_none(
            seen_slice.get("model_rmse"),
            _serialized_slice_value(validation, "heldout_science_seen_nuisance", "fisher_overall_rmse"),
        ),
        "heldout_nuisance_mse_skill": heldout_slice.get("mse_skill"),
        "seen_nuisance_mse_skill": seen_slice.get("mse_skill"),
        "sample_count": None if pred_metrics is None else pred_metrics.get("sample_count"),
    }


def _history_table(record: RunRecord) -> pd.DataFrame:
    df = pd.read_csv(record.result_path / "history.csv")
    manifest = record.manifest
    config = record.config or {}
    training = _mapping(config.get("training")) or _mapping(manifest.get("training"))
    lr_scheduler = _mapping(training.get("lr_scheduler"))
    scheduler_name = str(lr_scheduler.get("name", "none"))
    fixed_lr = training.get("learning_rate")

    for col in ("train_loss", "validation_loss", "validation_overall_rmse", "epoch_seconds"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "epoch" in df.columns:
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce").astype("Int64")
        df["epoch_number"] = df["epoch"] + 1
    else:
        df["epoch"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
        df["epoch_number"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    if "validation_overall_rmse" not in df.columns and "validation_loss" in df.columns:
        df["validation_overall_rmse"] = np.sqrt(pd.to_numeric(df["validation_loss"], errors="coerce"))

    if "learning_rate" in df.columns:
        df["learning_rate"] = pd.to_numeric(df["learning_rate"], errors="coerce")
        if "learning_rate_next" in df.columns:
            df["learning_rate_next"] = pd.to_numeric(df["learning_rate_next"], errors="coerce")
        else:
            df["learning_rate_next"] = np.nan
        if "lr_reduced" in df.columns:
            df["lr_reduced"] = df["lr_reduced"].map(_as_bool)
        else:
            df["lr_reduced"] = False
        df["learning_rate_source"] = "history"
    elif scheduler_name in ("none", "", "None") and fixed_lr is not None:
        df["learning_rate"] = float(fixed_lr)
        df["learning_rate_next"] = float(fixed_lr)
        df["lr_reduced"] = False
        df["learning_rate_source"] = "config_fixed"
    elif fixed_lr is not None and not lr_scheduler:
        df["learning_rate"] = float(fixed_lr)
        df["learning_rate_next"] = float(fixed_lr)
        df["lr_reduced"] = False
        df["learning_rate_source"] = "config_fixed"
    else:
        df["learning_rate"] = np.nan
        df["learning_rate_next"] = np.nan
        df["lr_reduced"] = np.nan
        df["learning_rate_source"] = "unavailable"

    if "epoch_seconds" not in df.columns:
        df["epoch_seconds"] = np.nan
    df["cumulative_seconds"] = pd.to_numeric(df["epoch_seconds"], errors="coerce").cumsum()
    df["cumulative_minutes"] = df["cumulative_seconds"] / 60.0
    if "validation_overall_rmse" not in df.columns:
        df["validation_overall_rmse"] = np.nan
    df["best_validation_rmse_so_far"] = pd.to_numeric(df["validation_overall_rmse"], errors="coerce").cummin()
    if "is_best" in df.columns:
        df["is_best"] = df["is_best"].map(_as_bool)
    else:
        df["is_best"] = df["validation_overall_rmse"] == df["best_validation_rmse_so_far"]
    if "early_stopping_bad_epochs" not in df.columns:
        df["early_stopping_bad_epochs"] = np.nan

    for key in ("site", "study_id", "experiment_id", "run_id", "seed"):
        df.insert(
            0,
            key,
            record.site if key == "site" else _first_not_none(config.get(key), manifest.get(key), record.run_id if key == "run_id" else None),
        )
    return df


def _slice_metrics_for(
    arrays: Mapping[str, np.ndarray] | None,
    slice_name: str,
) -> dict[str, Any]:
    if arrays is None or "eval_slice" not in arrays:
        return {}
    labels = np.asarray(arrays["eval_slice"]).astype(str)
    mask = labels == slice_name
    if not np.any(mask):
        return {}
    return compute_initializer_metrics(arrays["y_true_z"][mask], arrays["y_pred_z"][mask])


def _parameter_table(record: RunRecord, arrays: Mapping[str, np.ndarray]) -> pd.DataFrame:
    truth = np.asarray(arrays["y_true_z"], dtype=np.float64)
    pred = np.asarray(arrays["y_pred_z"], dtype=np.float64)
    residual = pred - truth
    labels = _parameter_labels(record, truth.shape[1])
    manifest = record.manifest
    config = record.config or {}
    validation = _mapping((record.metrics or {}).get("validation"))
    physical_rmse = _mapping(validation.get("physical_per_parameter_rmse"))

    rows: list[dict[str, Any]] = []
    for idx, label in enumerate(labels):
        baseline_mse = float(np.mean(truth[:, idx] ** 2))
        model_mse = float(np.mean(residual[:, idx] ** 2))
        baseline_rmse = math.sqrt(baseline_mse)
        model_rmse = math.sqrt(model_mse)
        rows.append(
            {
                "site": record.site,
                "study_id": _first_not_none(config.get("study_id"), manifest.get("study_id")),
                "experiment_id": _first_not_none(config.get("experiment_id"), manifest.get("experiment_id")),
                "run_id": record.run_id,
                "seed": _first_not_none(config.get("seed"), manifest.get("seed")),
                "parameter_index": idx,
                "parameter": label,
                "parameter_display": _parameter_display(label),
                "parameter_family": _parameter_family(label),
                "fisher_model_rmse": model_rmse,
                "fisher_baseline_rmse": baseline_rmse,
                "fisher_rmse_reduction": _safe_skill(baseline_rmse, model_rmse),
                "fisher_mse_skill": _safe_skill(baseline_mse, model_mse),
                "physical_rmse": physical_rmse.get(label),
                "physical_unit": _parameter_unit(label),
            }
        )
    return pd.DataFrame(rows)


def _slice_table(record: RunRecord, arrays: Mapping[str, np.ndarray]) -> pd.DataFrame:
    if "eval_slice" not in arrays:
        return pd.DataFrame()
    labels = np.asarray(arrays["eval_slice"]).astype(str)
    rows = []
    for label in sorted(set(labels)):
        mask = labels == label
        rows.append(_group_metric_row(record, arrays, mask, {"eval_slice": label}))
    return pd.DataFrame(rows)


def _distance_bin_table(record: RunRecord, arrays: Mapping[str, np.ndarray]) -> pd.DataFrame:
    if "fisher_distance_l2" not in arrays:
        return pd.DataFrame()
    distances = np.asarray(arrays["fisher_distance_l2"], dtype=np.float64)
    edges = _distance_bin_edges(record)
    rows = []
    for idx, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if idx == len(edges) - 2:
            mask = (distances >= lo) & (distances <= hi)
        else:
            mask = (distances >= lo) & (distances < hi)
        row = _group_metric_row(
            record,
            arrays,
            mask,
            {
                "distance_bin": _distance_bin_label(lo, hi),
                "distance_bin_lo": float(lo),
                "distance_bin_hi": float(hi),
                "distance_bin_includes_hi": idx == len(edges) - 2,
            },
        )
        if np.any(mask):
            row["mean_fisher_distance_l2"] = float(np.mean(distances[mask]))
            row["median_fisher_distance_l2"] = float(np.median(distances[mask]))
        else:
            row["mean_fisher_distance_l2"] = np.nan
            row["median_fisher_distance_l2"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _group_metric_row(
    record: RunRecord,
    arrays: Mapping[str, np.ndarray],
    mask: np.ndarray,
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = record.manifest
    config = record.config or {}
    row: dict[str, Any] = {
        "site": record.site,
        "study_id": _first_not_none(config.get("study_id"), manifest.get("study_id")),
        "experiment_id": _first_not_none(config.get("experiment_id"), manifest.get("experiment_id")),
        "run_id": record.run_id,
        "seed": _first_not_none(config.get("seed"), manifest.get("seed")),
        **extra,
        "sample_count": int(np.sum(mask)),
    }
    if not np.any(mask):
        row.update(
            {
                "model_fisher_rmse": np.nan,
                "baseline_fisher_rmse": np.nan,
                "rmse_reduction": np.nan,
                "mse_skill": np.nan,
                "mean_cosine_alignment": np.nan,
                "mean_correction_norm_ratio": np.nan,
                "median_relative_residual_norm": np.nan,
            }
        )
        return row
    metrics = compute_initializer_metrics(arrays["y_true_z"][mask], arrays["y_pred_z"][mask])
    geom = compute_prediction_geometry(arrays["y_true_z"][mask], arrays["y_pred_z"][mask])
    row.update(
        {
            "model_fisher_rmse": metrics["model_rmse"],
            "baseline_fisher_rmse": metrics["baseline_rmse"],
            "rmse_reduction": metrics["rmse_reduction"],
            "mse_skill": metrics["mse_skill"],
            "mean_cosine_alignment": _nan_stat(np.nanmean, geom["cosine_alignment"]),
            "mean_correction_norm_ratio": _nan_stat(np.nanmean, geom["correction_norm_ratio"]),
            "median_relative_residual_norm": _nan_stat(np.nanmedian, geom["relative_residual_norm"]),
        }
    )
    return row


def _load_prediction_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    if "y_true_z" not in arrays or "y_pred_z" not in arrays:
        raise ValueError(f"{path} must contain y_true_z and y_pred_z arrays.")
    _validate_prediction_shapes(arrays["y_true_z"], arrays["y_pred_z"])
    return arrays


def _validate_prediction_shapes(truth: np.ndarray, pred: np.ndarray) -> None:
    if truth.shape != pred.shape:
        raise ValueError(f"Prediction shape {pred.shape} does not match truth shape {truth.shape}.")
    if truth.ndim != 2:
        raise ValueError("Predictions and truth must be 2D arrays.")


def _metric_compatibility_warnings(
    record: RunRecord,
    pred_metrics: Mapping[str, Any],
    *,
    relative_tolerance: float = 5.0e-6,
) -> list[str]:
    validation = _mapping((record.metrics or {}).get("validation"))
    serialized = validation.get("fisher_overall_rmse")
    if serialized is None:
        return []
    computed = float(pred_metrics["model_rmse"])
    if not np.isfinite(computed) or not np.isfinite(float(serialized)):
        return []
    denom = max(abs(float(serialized)), abs(computed), 1.0)
    if abs(float(serialized) - computed) / denom > relative_tolerance:
        return [
            "Computed prediction RMSE disagrees with serialized validation metric "
            f"for {record.run_id}: computed={computed:.8g}, serialized={float(serialized):.8g}."
        ]
    return []


def _identity_tuple(record: RunRecord) -> tuple[Any, ...]:
    manifest = record.manifest
    git = _mapping(manifest.get("git"))
    prepared = _mapping(manifest.get("prepared_dataset"))
    split = _mapping(manifest.get("split_registry"))
    validation = _mapping(manifest.get("validation_manifest_identity"))
    config = record.config or {}
    return (
        manifest.get("study_id"),
        manifest.get("experiment_id"),
        record.run_id,
        _first_not_none(config.get("seed"), manifest.get("seed")),
        _first_not_none(git.get("source_commit"), git.get("commit")),
        git.get("source_archive_id"),
        prepared.get("prepared_dataset_hash"),
        split.get("content_sha256"),
        validation.get("sha256"),
        bool(manifest.get("test_evaluated", False)),
    )


def _parameter_labels(record: RunRecord, dim: int) -> list[str]:
    validation = _mapping((record.metrics or {}).get("validation"))
    per_param = _mapping(validation.get("fisher_per_parameter_rmse"))
    if len(per_param) == dim:
        return list(per_param.keys())
    return [f"z[{idx}]" for idx in range(dim)]


def _parameter_family(label: str) -> str:
    if "primary.zernike" in label:
        return "M1"
    if "secondary.zernike" in label:
        return "M2"
    return "source/global"


def _parameter_display(label: str) -> str:
    if "primary.zernike_coeffs_nm[" in label:
        return f"M1 Z{_zernike_number(label)}"
    if "secondary.zernike_coeffs_nm[" in label:
        return f"M2 Z{_zernike_number(label)}"
    return label.replace("source.", "").replace("optics.", "")


def _zernike_number(label: str) -> int:
    idx = int(label.rsplit("[", 1)[1].split("]", 1)[0])
    return idx + 4


def _parameter_unit(label: str) -> str | None:
    if "zernike_coeffs_nm" in label:
        return "nm"
    if label.endswith("_as") or "separation_as" in label:
        return "arcsec"
    if "plate_scale_as_per_pix" in label:
        return "arcsec/pix"
    if "contrast" in label:
        return "dimensionless"
    if "log_flux" in label:
        return "log flux"
    return None


def _distance_bin_edges(record: RunRecord) -> list[float]:
    config_edges = _nested(record.config or {}, ("evaluation", "fisher_distance_bin_edges"))
    if config_edges is not None:
        return [float(value) for value in config_edges]
    metrics_edges = _nested(record.metrics or {}, ("validation", "by_distance_bin", "bin_edges"))
    if metrics_edges is not None:
        return [float(value) for value in metrics_edges]
    return [0.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 5000.0]


def _distance_bin_label(lo: float, hi: float) -> str:
    def fmt(value: float) -> str:
        return str(int(value)) if float(value).is_integer() else f"{value:g}"

    return f"{fmt(lo)}-{fmt(hi)}"


def _serialized_slice_value(validation: Mapping[str, Any], slice_name: str, key: str) -> Any:
    by_slice = _mapping(validation.get("by_eval_slice"))
    return _mapping(by_slice.get(slice_name)).get(key)


def _history_total_seconds(record: RunRecord) -> float | None:
    path = record.result_path / "history.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, usecols=lambda col: col == "epoch_seconds")
    if "epoch_seconds" not in df:
        return None
    return float(pd.to_numeric(df["epoch_seconds"], errors="coerce").sum())


def _history_mean_epoch_seconds(record: RunRecord) -> float | None:
    path = record.result_path / "history.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, usecols=lambda col: col == "epoch_seconds")
    if "epoch_seconds" not in df:
        return None
    values = pd.to_numeric(df["epoch_seconds"], errors="coerce")
    return None if values.dropna().empty else float(values.mean())


def _load_sync_manifests(roots: Sequence[Path]) -> pd.DataFrame:
    rows = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("sync_manifest.json")):
            payload = _read_json(path)
            rows.append({"manifest_path": str(path), **payload})
    return pd.DataFrame(rows)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _maybe_read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _read_json(path)


def _infer_site(run_dir: Path, roots: Sequence[Path]) -> str | None:
    parts = run_dir.parts
    for marker in ("ml", "hpc_imports"):
        if marker in parts:
            index = parts.index(marker)
            if len(parts) > index + 1 and parts[index + 1].lower() in {"ls6", "tacc_ls6"}:
                return "ls6"
    for part in parts:
        if part.lower() in {"ls6", "tacc_ls6"}:
            return "ls6"
    for root in roots:
        if root.name.lower() in {"ls6", "tacc_ls6"}:
            return "ls6"
    return None


def _nested(mapping: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _safe_skill(baseline: float, model: float) -> float:
    if not np.isfinite(baseline) or abs(float(baseline)) <= EPS:
        return np.nan
    return float(1.0 - float(model) / float(baseline))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _nan_stat(func: Any, values: np.ndarray) -> float:
    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(func(values))


def _concat_or_empty(tables: Sequence[pd.DataFrame]) -> pd.DataFrame:
    valid = [table for table in tables if not table.empty]
    return pd.concat(valid, ignore_index=True) if valid else pd.DataFrame()
