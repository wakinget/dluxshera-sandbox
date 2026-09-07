from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

ANALYSIS_DIR = Path(__file__).resolve().parents[2] / "work" / "experiments" / "ml" / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

from campaign_plots import (  # noqa: E402
    add_common_zero_baseline,
    plot_best_fisher_rmse_by_run,
    plot_learning_rate_vs_epoch,
    plot_mse_skill_by_distance_bin,
    plot_parameter_skill_heatmap,
    plot_prediction_norms,
    plot_seen_vs_heldout_nuisance,
    plot_validation_rmse_vs_epoch,
)


def test_plot_helpers_return_figures_and_axes() -> None:
    runs = pd.DataFrame(
        {
            "run_id": ["a", "b"],
            "best_fisher_rmse": [2.0, 1.0],
            "mse_skill": [0.2, 0.3],
            "rmse_reduction": [0.1, 0.2],
            "total_training_seconds": [60.0, 120.0],
        }
    )
    history = pd.DataFrame(
        {
            "run_id": ["a", "a", "b", "b"],
            "study_id": ["S05"] * 4,
            "experiment_id": ["E"] * 4,
            "epoch": [0, 1, 0, 1],
            "validation_overall_rmse": [3.0, 2.0, 4.0, 1.0],
            "best_validation_rmse_so_far": [3.0, 2.0, 4.0, 1.0],
            "cumulative_minutes": [1.0, 2.0, 1.0, 2.0],
            "learning_rate": [1.0e-3, 1.0e-3, 5.0e-4, 2.5e-4],
        }
    )
    slices = pd.DataFrame(
        {
            "run_id": ["a", "a"],
            "eval_slice": ["heldout_science_seen_nuisance", "heldout_science_heldout_nuisance"],
            "model_fisher_rmse": [1.0, 2.0],
            "mse_skill": [0.5, 0.25],
        }
    )
    parameters = pd.DataFrame(
        {
            "run_id": ["a", "a", "b", "b"],
            "parameter": ["p0", "p1", "p0", "p1"],
            "parameter_display": ["p0", "p1", "p0", "p1"],
            "parameter_index": [0, 1, 0, 1],
            "fisher_mse_skill": [0.1, 0.2, 0.3, 0.4],
        }
    )
    bins = pd.DataFrame(
        {
            "run_id": ["a", "a"],
            "distance_bin": ["0-1", "1-2"],
            "distance_bin_lo": [0.0, 1.0],
            "mse_skill": [0.1, 0.2],
            "sample_count": [10, 10],
        }
    )
    geometry = pd.DataFrame(
        {
            "truth_norm": [1.0, 2.0],
            "pred_norm": [1.1, 1.9],
            "relative_residual_norm": [0.1, 0.2],
            "fisher_distance_l2": [1.0, 2.0],
        }
    )

    for func, arg in (
        (plot_validation_rmse_vs_epoch, history),
        (plot_learning_rate_vs_epoch, history),
        (plot_best_fisher_rmse_by_run, runs),
        (plot_seen_vs_heldout_nuisance, slices),
        (plot_parameter_skill_heatmap, parameters),
        (plot_mse_skill_by_distance_bin, bins),
        (plot_prediction_norms, geometry),
    ):
        fig, ax = func(arg)
        assert fig is ax.figure
        plt.close(fig)


def test_plot_helpers_handle_empty_tables() -> None:
    fig, ax = plot_best_fisher_rmse_by_run(pd.DataFrame())
    assert fig is ax.figure
    plt.close(fig)


def test_common_zero_baseline_added_for_compatible_runs() -> None:
    runs = pd.DataFrame(
        {
            "run_id": ["a", "b"],
            "validation_manifest_sha256": ["same", "same"],
            "zero_baseline_fisher_rmse": [250.0, 250.0 + 1.0e-8],
        }
    )
    fig, ax = plt.subplots()

    added = add_common_zero_baseline(ax, runs)

    assert added is True
    assert len(ax.lines) == 1
    assert ax.lines[0].get_ydata()[0] == pytest.approx(250.0)
    plt.close(fig)


def test_common_zero_baseline_omitted_for_incompatible_runs() -> None:
    incompatible_manifest = pd.DataFrame(
        {
            "run_id": ["a", "b"],
            "validation_manifest_sha256": ["one", "two"],
            "zero_baseline_fisher_rmse": [250.0, 250.0],
        }
    )
    incompatible_baseline = pd.DataFrame(
        {
            "run_id": ["a", "b"],
            "validation_manifest_sha256": ["same", "same"],
            "zero_baseline_fisher_rmse": [250.0, 251.0],
        }
    )
    fig, axes = plt.subplots(1, 2)

    assert add_common_zero_baseline(axes[0], incompatible_manifest) is False
    assert add_common_zero_baseline(axes[1], incompatible_baseline) is False
    assert len(axes[0].lines) == 0
    assert len(axes[1].lines) == 0
    plt.close(fig)
