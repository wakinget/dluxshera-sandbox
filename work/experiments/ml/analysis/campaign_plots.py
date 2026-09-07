from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

__all__ = [
    "add_common_zero_baseline",
    "plot_best_fisher_rmse_by_run",
    "plot_best_so_far_rmse_vs_epoch",
    "plot_cosine_alignment_by_distance_bin",
    "plot_learning_rate_vs_epoch",
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
