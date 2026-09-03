from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dluxshera.ml import load_sample_catalog
from dluxshera.ml.training import _distance_binned_metrics, validate_fisher_distance_bin_edges
from dluxshera.ml.training import DEFAULT_S01_FISHER_DISTANCE_BIN_EDGES
from tests.ml.test_catalog_splits_pairs import _write_prepared_fixture


def _catalog(tmp_path: Path):
    return load_sample_catalog(_write_prepared_fixture(tmp_path / "prepared"))


def test_default_s01_bin_edges_are_production_scale() -> None:
    assert DEFAULT_S01_FISHER_DISTANCE_BIN_EDGES == (
        0.0,
        100.0,
        250.0,
        500.0,
        1000.0,
        2000.0,
        5000.0,
    )


def test_custom_bins_boundaries_final_upper_edge_and_empty_bins(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    y_true = np.zeros((5, catalog.science_dim), dtype=np.float64)
    y_pred = np.ones_like(y_true)
    distances = np.asarray([0.0, 99.999, 100.0, 499.0, 500.0], dtype=np.float64)
    metrics = _distance_binned_metrics(
        y_pred,
        y_true,
        distances,
        catalog=catalog,
        bin_edges=[0.0, 100.0, 250.0, 500.0],
    )
    assert metrics["bin_edges"] == [0.0, 100.0, 250.0, 500.0]
    assert metrics["bins"]["0-100"]["sample_count"] == 2
    assert metrics["bins"]["100-250"]["sample_count"] == 1
    assert metrics["bins"]["250-500"]["sample_count"] == 2
    assert metrics["outside_range_count"] == 0
    assert set(metrics) == {
        "bin_edges",
        "below_range_count",
        "above_range_count",
        "outside_range_count",
        "bins",
    }


def test_bins_report_samples_outside_configured_range(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    y_true = np.zeros((3, catalog.science_dim), dtype=np.float64)
    y_pred = np.zeros_like(y_true)
    metrics = _distance_binned_metrics(
        y_pred,
        y_true,
        np.asarray([-0.1, 0.5, 3.1]),
        catalog=catalog,
        bin_edges=[0.0, 1.0, 3.0],
    )
    assert metrics["below_range_count"] == 1
    assert metrics["above_range_count"] == 1
    assert metrics["outside_range_count"] == 2
    assert metrics["bins"]["1-3"]["sample_count"] == 0


def test_invalid_bins_fail() -> None:
    with pytest.raises(ValueError, match="at least two"):
        validate_fisher_distance_bin_edges([0.0])
    with pytest.raises(ValueError, match="finite"):
        validate_fisher_distance_bin_edges([0.0, float("inf")])
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_fisher_distance_bin_edges([0.0, 2.0, 2.0])


def test_regression_metrics_are_preserved_inside_each_nonempty_bin(tmp_path: Path) -> None:
    catalog = _catalog(tmp_path)
    y_true = np.zeros((2, catalog.science_dim), dtype=np.float64)
    y_pred = np.asarray([[1.0, 1.0], [3.0, 3.0]], dtype=np.float64)
    metrics = _distance_binned_metrics(
        y_pred,
        y_true,
        np.asarray([0.5, 1.5]),
        catalog=catalog,
        bin_edges=[0.0, 1.0, 2.0],
    )
    assert metrics["bins"]["0-1"]["fisher_overall_rmse"] == pytest.approx(1.0)
    assert metrics["bins"]["1-2"]["fisher_overall_rmse"] == pytest.approx(3.0)
    assert all(
        isinstance(value, dict) and "sample_count" in value for value in metrics["bins"].values()
    )
