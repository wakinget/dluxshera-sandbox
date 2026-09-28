from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

__all__ = [
    "add_common_zero_baseline",
    "plot_distance_bin_metric_by_experiment",
    "plot_experiment_metric_summary",
    "plot_best_fisher_rmse_by_run",
    "plot_best_so_far_rmse_vs_epoch",
    "plot_cosine_alignment_by_distance_bin",
    "plot_learning_rate_vs_epoch",
    "plot_paired_seed_deltas",
    "plot_parameter_physical_rmse_by_experiment",
    "plot_mse_skill_by_distance_bin",
    "plot_mse_skill_by_run",
    "plot_parameter_skill_by_run",
    "plot_parameter_skill_heatmap",
    "plot_prediction_norms",
    "plot_relative_residual_by_distance_bin",
    "plot_relative_residual_vs_distance",
    "plot_rmse_reduction_by_run",
    "plot_runtime_by_run",
    "plot_seen_vs_heldout_nuisance",
    "plot_validation_rmse_vs_epoch",
    "plot_validation_rmse_vs_wall_minutes",
]

BASELINE_RTOL = 1.0e-6
BASELINE_ATOL = 1.0e-8


def add_common_zero_baseline(
    ax: plt.Axes,
    runs: pd.DataFrame,
    *,
    rtol: float = BASELINE_RTOL,
    atol: float = BASELINE_ATOL,
) -> bool:
    """Add a shared zero-baseline line only for comparable selected runs."""

    required = {"validation_manifest_sha256", "zero_baseline_fisher_rmse"}
    if runs.empty or not required.issubset(runs.columns):
        return False
    subset = runs.dropna(subset=["validation_manifest_sha256", "zero_baseline_fisher_rmse"])
    if subset.empty or subset["validation_manifest_sha256"].nunique() != 1:
        return False
    values = subset["zero_baseline_fisher_rmse"].astype(float).to_numpy()
    if values.size == 0 or not np.all(np.isfinite(values)):
        return False
    reference = float(values[0])
    if not np.allclose(values, reference, rtol=rtol, atol=atol):
        return False
    ax.axhline(
        reference,
        color="black",
        linewidth=1,
        linestyle="--",
        label="zero-correction baseline",
    )
    ax.legend(fontsize="small")
    return True


def plot_validation_rmse_vs_epoch(
    history: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
    study: str | None = None,
    experiment: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot validation Fisher RMSE against epoch."""

    return _line_by_run(
        _filter(history, study=study, experiment=experiment),
        x="epoch",
        y="validation_overall_rmse",
        ylabel="Validation Fisher RMSE",
        xlabel="Epoch",
        ax=ax,
    )


def plot_best_so_far_rmse_vs_epoch(
    history: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
    study: str | None = None,
    experiment: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot best-so-far validation Fisher RMSE against epoch."""

    return _line_by_run(
        _filter(history, study=study, experiment=experiment),
        x="epoch",
        y="best_validation_rmse_so_far",
        ylabel="Best-so-far validation Fisher RMSE",
        xlabel="Epoch",
        ax=ax,
    )


def plot_validation_rmse_vs_wall_minutes(
    history: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
    study: str | None = None,
    experiment: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot validation Fisher RMSE against cumulative training time."""

    return _line_by_run(
        _filter(history, study=study, experiment=experiment),
        x="cumulative_minutes",
        y="validation_overall_rmse",
        ylabel="Validation Fisher RMSE",
        xlabel="Cumulative minutes",
        ax=ax,
    )


def plot_learning_rate_vs_epoch(
    history: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
    study: str | None = None,
    experiment: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot recorded or unambiguous fixed learning rate against epoch."""

    fig, ax = _figure_ax(ax)
    df = _filter(history, study=study, experiment=experiment)
    df = df.dropna(subset=["learning_rate"]) if "learning_rate" in df else pd.DataFrame()
    _plot_empty_message(ax, df, "No learning-rate data.")
    if not df.empty:
        for run_id, group in df.sort_values(["run_id", "epoch"]).groupby("run_id"):
            ax.plot(group["epoch"], group["learning_rate"], label=str(run_id))
        ax.set_yscale("log")
        ax.legend(fontsize="small")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    return fig, ax


def plot_best_fisher_rmse_by_run(
    runs: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot learned validation Fisher RMSE by run."""

    return _bar_by_run(runs, "best_fisher_rmse", "Best validation Fisher RMSE", ax=ax)


def plot_mse_skill_by_run(
    runs: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot initializer MSE skill by run."""

    return _bar_by_run(runs, "mse_skill", "MSE skill", ax=ax)


def plot_rmse_reduction_by_run(
    runs: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot initializer RMSE reduction by run."""

    return _bar_by_run(runs, "rmse_reduction", "RMSE reduction", ax=ax)


def plot_runtime_by_run(
    runs: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot total training runtime by run."""

    df = runs.copy()
    if "total_training_seconds" in df:
        df["total_training_minutes"] = df["total_training_seconds"] / 60.0
    return _bar_by_run(df, "total_training_minutes", "Training runtime (minutes)", ax=ax)


def plot_experiment_metric_summary(
    runs: pd.DataFrame,
    metric: str,
    *,
    ax: plt.Axes | None = None,
    group_col: str = "experiment_id",
    seed_col: str = "seed",
    ylabel: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot experiment means with individual seed points."""

    fig, ax = _figure_ax(ax)
    df = runs.copy()
    if df.empty or group_col not in df or metric not in df:
        _plot_empty_message(ax, pd.DataFrame(), f"No {metric} data.")
    else:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        df = df.dropna(subset=[group_col, metric])
        _plot_empty_message(ax, df, f"No {metric} data.")
        if not df.empty:
            order = _ordered_categories(df, group_col)
            positions = np.arange(len(order))
            grouped = df.groupby(group_col, sort=False)[metric]
            means = grouped.mean().reindex(order)
            stds = grouped.std(ddof=1).reindex(order)
            ax.bar(positions, means, yerr=stds, alpha=0.35, capsize=4)
            for xpos, experiment in zip(positions, order):
                group = df[df[group_col].eq(experiment)].sort_values(seed_col if seed_col in df else metric)
                jitter = _jitter(len(group), width=0.18)
                ax.scatter(
                    np.full(len(group), xpos) + jitter,
                    group[metric],
                    s=32,
                    zorder=3,
                    label=str(experiment) if len(order) == 1 else None,
                )
            ax.set_xticks(positions, labels=[str(value) for value in order], rotation=30, ha="right")
    ax.set_xlabel(group_col.replace("_", " ").title())
    ax.set_ylabel(ylabel or metric.replace("_", " "))
    return fig, ax


def plot_paired_seed_deltas(
    deltas: pd.DataFrame,
    metric: str,
    *,
    ax: plt.Axes | None = None,
    seed_col: str = "seed",
    ylabel: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot matched-seed metric deltas from ``matched_seed_delta_table``."""

    fig, ax = _figure_ax(ax)
    column = f"{metric}_delta"
    df = deltas.copy()
    if df.empty or column not in df:
        _plot_empty_message(ax, pd.DataFrame(), f"No paired {metric} delta data.")
    else:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.dropna(subset=[column])
        _plot_empty_message(ax, df, f"No paired {metric} delta data.")
        if not df.empty:
            labels = df[seed_col].astype(str) if seed_col in df else df.index.astype(str)
            positions = np.arange(len(df))
            ax.bar(positions, df[column])
            ax.axhline(0.0, color="black", linewidth=1, alpha=0.6)
            ax.set_xticks(positions, labels=labels)
    ax.set_xlabel("Seed")
    ax.set_ylabel(ylabel or f"{metric.replace('_', ' ')} delta")
    return fig, ax


def plot_distance_bin_metric_by_experiment(
    distance_bins: pd.DataFrame,
    *,
    metric: str = "mse_skill",
    ax: plt.Axes | None = None,
    min_sample_count: int | None = None,
    ylabel: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot distance-bin metric means and seed scatter grouped by experiment."""

    fig, ax = _figure_ax(ax)
    df = distance_bins.copy()
    required = {"experiment_id", "distance_bin", "distance_bin_lo", metric}
    if df.empty or not required.issubset(df.columns):
        _plot_empty_message(ax, pd.DataFrame(), f"No distance-bin {metric} data.")
    else:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        if min_sample_count is not None and "sample_count" in df:
            df.loc[pd.to_numeric(df["sample_count"], errors="coerce") < min_sample_count, metric] = np.nan
        df = df.dropna(subset=["experiment_id", "distance_bin", "distance_bin_lo", metric])
        _plot_empty_message(ax, df, f"No distance-bin {metric} data.")
        if not df.empty:
            ordered_bins = (
                df.sort_values("distance_bin_lo")[["distance_bin", "distance_bin_lo"]]
                .drop_duplicates("distance_bin")["distance_bin"]
                .tolist()
            )
            x = np.arange(len(ordered_bins))
            for experiment, group in df.groupby("experiment_id", sort=True):
                summary = group.groupby("distance_bin", sort=False)[metric].agg(["mean", "std"]).reindex(ordered_bins)
                ax.plot(x, summary["mean"], marker="o", label=str(experiment))
                lower = (summary["mean"] - summary["std"]).to_numpy(dtype=float)
                upper = (summary["mean"] + summary["std"]).to_numpy(dtype=float)
                ax.fill_between(x, lower, upper, alpha=0.12)
            ax.set_xticks(x, labels=[str(value) for value in ordered_bins], rotation=35, ha="right")
            ax.legend(fontsize="small")
    ax.set_xlabel("Fisher-distance bin")
    ax.set_ylabel(ylabel or metric.replace("_", " "))
    return fig, ax


def plot_validation_metric_by_experiment(
    history: pd.DataFrame,
    *,
    metric: str = "validation_overall_rmse",
    x: str = "epoch",
    ax: plt.Axes | None = None,
    ylabel: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot validation history as experiment mean +/- seed scatter."""

    fig, ax = _figure_ax(ax)
    df = history.copy()
    required = {"experiment_id", x, metric}
    if df.empty or not required.issubset(df.columns):
        _plot_empty_message(ax, pd.DataFrame(), f"No validation {metric} data.")
    else:
        df[x] = pd.to_numeric(df[x], errors="coerce")
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        df = df.dropna(subset=["experiment_id", x, metric])
        _plot_empty_message(ax, df, f"No validation {metric} data.")
        if not df.empty:
            for experiment, group in df.groupby("experiment_id", sort=True):
                summary = group.groupby(x, sort=True)[metric].agg(["mean", "std"]).reset_index()
                ax.plot(summary[x], summary["mean"], label=str(experiment))
                lower = (summary["mean"] - summary["std"]).to_numpy(dtype=float)
                upper = (summary["mean"] + summary["std"]).to_numpy(dtype=float)
                ax.fill_between(summary[x].to_numpy(dtype=float), lower, upper, alpha=0.12)
            ax.legend(fontsize="small")
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(ylabel or metric.replace("_", " "))
    return fig, ax


def plot_parameter_physical_rmse_by_experiment(
    parameters: pd.DataFrame,
    *,
    parameter: str = "source.separation_as",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot one parameter's physical RMSE by experiment with seed points."""

    df = parameters.copy()
    if "parameter" in df:
        df = df[df["parameter"].eq(parameter)].copy()
    metric = "physical_rmse_display" if "physical_rmse_display" in df else "physical_rmse"
    unit = None
    if not df.empty:
        unit_col = "physical_display_unit" if "physical_display_unit" in df else "physical_unit"
        if unit_col in df:
            units = df[unit_col].dropna().astype(str).unique()
            unit = units[0] if len(units) == 1 else None
    ylabel = f"{parameter} RMSE" + (f" ({unit})" if unit else "")
    return plot_experiment_metric_summary(df, metric, ax=ax, ylabel=ylabel)


def plot_seen_vs_heldout_nuisance(
    slices: pd.DataFrame,
    *,
    metric: str = "model_fisher_rmse",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot seen- and held-out-nuisance slice metrics by run."""

    fig, ax = _figure_ax(ax)
    df = slices.copy()
    _plot_empty_message(ax, df, "No evaluation-slice data.")
    if not df.empty and metric in df:
        pivot = df.pivot_table(index="run_id", columns="eval_slice", values=metric, aggfunc="first")
        pivot.plot(kind="bar", ax=ax)
        ax.legend(fontsize="small", title="Slice")
    ax.set_xlabel("Run")
    ax.set_ylabel(metric.replace("_", " "))
    ax.tick_params(axis="x", rotation=45)
    return fig, ax


def plot_parameter_skill_by_run(
    parameters: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot per-parameter Fisher MSE skill with one line per run."""

    fig, ax = _figure_ax(ax)
    df = parameters.copy()
    _plot_empty_message(ax, df, "No parameter table data.")
    if not df.empty:
        for run_id, group in df.sort_values(["run_id", "parameter_index"]).groupby("run_id"):
            ax.plot(group["parameter_display"], group["fisher_mse_skill"], marker="o", label=str(run_id))
        ax.legend(fontsize="small")
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Fisher MSE skill")
    ax.tick_params(axis="x", rotation=90)
    return fig, ax


def plot_parameter_skill_heatmap(
    parameters: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a run by parameter Fisher MSE skill heatmap."""

    fig, ax = _figure_ax(ax)
    df = parameters.copy()
    _plot_empty_message(ax, df, "No parameter table data.")
    if not df.empty:
        ordered_params = (
            df.sort_values("parameter_index")[["parameter", "parameter_display"]]
            .drop_duplicates("parameter")
        )
        pivot = df.pivot_table(index="run_id", columns="parameter", values="fisher_mse_skill", aggfunc="first")
        pivot = pivot.reindex(columns=ordered_params["parameter"].tolist())
        image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", interpolation="nearest")
        ax.set_yticks(np.arange(len(pivot.index)), labels=[str(v) for v in pivot.index])
        ax.set_xticks(
            np.arange(len(ordered_params)),
            labels=ordered_params["parameter_display"].tolist(),
            rotation=90,
        )
        fig.colorbar(image, ax=ax, label="Fisher MSE skill")
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Run")
    return fig, ax


def plot_mse_skill_by_distance_bin(
    distance_bins: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot MSE skill by Fisher-distance bin."""

    return _distance_line(distance_bins, "mse_skill", "MSE skill", ax=ax)


def plot_relative_residual_by_distance_bin(
    distance_bins: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot median relative residual norm by Fisher-distance bin."""

    return _distance_line(
        distance_bins,
        "median_relative_residual_norm",
        "Median relative residual norm",
        ax=ax,
    )


def plot_cosine_alignment_by_distance_bin(
    distance_bins: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot mean correction cosine alignment by Fisher-distance bin."""

    return _distance_line(distance_bins, "mean_cosine_alignment", "Mean cosine alignment", ax=ax)


def plot_prediction_norms(
    geometry: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
    sample: int | None = 5000,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot predicted correction norm against true correction norm."""

    fig, ax = _figure_ax(ax)
    df = _sample(geometry, sample)
    _plot_empty_message(ax, df, "No prediction geometry data.")
    if not df.empty:
        ax.scatter(df["truth_norm"], df["pred_norm"], s=8, alpha=0.35)
        limit = float(np.nanmax([df["truth_norm"].max(), df["pred_norm"].max()]))
        ax.plot([0, limit], [0, limit], color="black", linewidth=1, alpha=0.5)
    ax.set_xlabel("True correction norm")
    ax.set_ylabel("Predicted correction norm")
    return fig, ax


def plot_relative_residual_vs_distance(
    geometry: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
    sample: int | None = 5000,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot relative residual norm against input Fisher distance."""

    fig, ax = _figure_ax(ax)
    df = _sample(geometry, sample)
    _plot_empty_message(ax, df, "No prediction geometry data.")
    if not df.empty and "fisher_distance_l2" in df:
        ax.scatter(df["fisher_distance_l2"], df["relative_residual_norm"], s=8, alpha=0.35)
        ax.axhline(1.0, color="black", linewidth=1, alpha=0.5)
    ax.set_xlabel("Input Fisher distance")
    ax.set_ylabel("Relative residual norm")
    return fig, ax


def _line_by_run(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    xlabel: str,
    ylabel: str,
    ax: plt.Axes | None,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = _figure_ax(ax)
    _plot_empty_message(ax, df, "No training-history data.")
    if not df.empty and x in df and y in df:
        for run_id, group in df.dropna(subset=[x, y]).sort_values(["run_id", x]).groupby("run_id"):
            ax.plot(group[x], group[y], label=str(run_id))
        if ax.lines:
            ax.legend(fontsize="small")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def _bar_by_run(
    runs: pd.DataFrame,
    column: str,
    ylabel: str,
    *,
    ax: plt.Axes | None,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = _figure_ax(ax)
    df = runs.copy()
    _plot_empty_message(ax, df, "No run table data.")
    if not df.empty and column in df:
        df = df.dropna(subset=[column]).sort_values(column)
        ax.bar(df["run_id"].astype(str), df[column])
    ax.set_xlabel("Run")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    return fig, ax


def _distance_line(
    distance_bins: pd.DataFrame,
    column: str,
    ylabel: str,
    *,
    ax: plt.Axes | None,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = _figure_ax(ax)
    df = distance_bins.copy()
    _plot_empty_message(ax, df, "No Fisher-distance table data.")
    if not df.empty and column in df:
        for run_id, group in df.sort_values(["run_id", "distance_bin_lo"]).groupby("run_id"):
            ax.plot(group["distance_bin"], group[column], marker="o", label=str(run_id))
        ax.legend(fontsize="small")
    ax.set_xlabel("Fisher-distance bin")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    return fig, ax


def _filter(df: pd.DataFrame, *, study: str | None, experiment: str | None) -> pd.DataFrame:
    out = df.copy()
    if study is not None and "study_id" in out:
        out = out[out["study_id"] == study]
    if experiment is not None and "experiment_id" in out:
        out = out[out["experiment_id"] == experiment]
    return out


def _figure_ax(ax: plt.Axes | None) -> tuple[plt.Figure, plt.Axes]:
    if ax is not None:
        return ax.figure, ax
    fig, new_ax = plt.subplots(figsize=(8, 4.5))
    return fig, new_ax


def _plot_empty_message(ax: plt.Axes, df: pd.DataFrame, message: str) -> None:
    if df.empty:
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)


def _sample(df: pd.DataFrame, sample: int | None) -> pd.DataFrame:
    if sample is None or len(df) <= sample:
        return df
    return df.sample(n=sample, random_state=0)


def _ordered_categories(df: pd.DataFrame, column: str) -> list[object]:
    return sorted(df[column].dropna().unique().tolist(), key=lambda value: str(value))


def _jitter(size: int, *, width: float) -> np.ndarray:
    if size <= 1:
        return np.zeros(size)
    return np.linspace(-width, width, size)
