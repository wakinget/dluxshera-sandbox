from __future__ import annotations

import math

import pytest

from dluxshera.utils.iterative_campaigns import (
    apply_physical_reference_update,
    posterior_float,
    posterior_label,
    posterior_offsets_from_rows,
    separation_update_diagnostics,
    vector_update_diagnostics,
)


def test_posterior_label_selection_order() -> None:
    assert posterior_label({"theta_label": "a", "parameter": "b"}) == "a"
    assert posterior_label({"parameter": "b"}) == "b"
    assert posterior_label({"label": "c"}) == "c"
    assert posterior_label({"name": "d"}) == "d"
    assert posterior_label({}) == ""


def test_posterior_float_uses_first_parseable_candidate() -> None:
    row = {"a": "", "b": "not-a-float", "c": "1.25", "d": "2.0"}
    assert posterior_float(row, ("a", "b", "c", "d")) == pytest.approx(1.25)
    assert math.isnan(posterior_float(row, ("missing", "a", "b")))


def test_apply_physical_reference_update_gain_and_bad_rows() -> None:
    truth = {"x": 10.0, "y": -2.0}
    current = {"x": 4.0, "y": 1.0}
    posterior = {
        "x": {"posterior_mean": "12.0"},
        "y": {"posterior_mean": "nan"},
        "z": {"posterior_mean": "100.0"},
    }

    full = apply_physical_reference_update(
        current_offsets=current,
        posterior_rows_by_label=posterior,
        truth_by_label=truth,
        update_gain=1.0,
    )
    damped = apply_physical_reference_update(
        current_offsets=current,
        posterior_rows_by_label=posterior,
        truth_by_label=truth,
        update_gain=0.5,
    )

    assert full["x"] == pytest.approx(2.0)
    assert damped["x"] == pytest.approx(3.0)
    assert full["y"] == pytest.approx(1.0)
    assert "z" not in full
    with pytest.raises(ValueError, match="update_gain"):
        apply_physical_reference_update(
            current_offsets=current,
            posterior_rows_by_label=posterior,
            truth_by_label=truth,
            update_gain=float("nan"),
        )


def test_posterior_offsets_from_rows_reports_missing_and_nonfinite() -> None:
    offsets, status = posterior_offsets_from_rows(
        labels=("x", "y", "z", "w"),
        posterior_rows_by_label={
            "x": {"posterior_mean": "12.0"},
            "y": {"posterior_mean": "bad"},
            "z": {"posterior_mean": "3.0"},
        },
        truth_by_label={"x": 10.0, "y": 0.0},
        fallback_offsets={"y": 4.0, "z": 5.0, "w": 6.0},
    )
    assert offsets["x"] == pytest.approx(2.0)
    assert offsets["y"] == pytest.approx(4.0)
    assert offsets["z"] == pytest.approx(5.0)
    assert offsets["w"] == pytest.approx(6.0)
    assert status == {
        "x": "ok",
        "y": "nonfinite_posterior_mean",
        "z": "missing_truth",
        "w": "missing_posterior_row",
    }


def test_vector_update_diagnostics_distinguishes_posterior_and_applied() -> None:
    row = vector_update_diagnostics(
        labels=("source.separation_as", "z"),
        current_offsets={"source.separation_as": 4.0, "z": 0.0},
        posterior_offsets={"source.separation_as": 0.0, "z": 0.0},
        next_offsets={"source.separation_as": 2.0, "z": 0.0},
        previous_residual_norm=5.0,
        previous_next_reference_norm=3.0,
    )
    assert row["posterior_update_norm"] == pytest.approx(4.0)
    assert row["applied_update_norm"] == pytest.approx(2.0)
    assert row["posterior_vector_gain"] == pytest.approx(1.0)
    assert row["applied_vector_gain"] == pytest.approx(0.5)
    assert row["next_reference_error_norm"] == pytest.approx(2.0)
    assert row["next_reference_error_norm_over_bias_norm"] == pytest.approx(0.5)
    assert row["residual_norm_decreased_from_previous_window"] is True
    assert row["next_reference_residual_decreased_from_previous_window"] is True
    assert row["update_norm"] == pytest.approx(row["posterior_update_norm"])


def test_separation_update_diagnostics_microas_and_direction() -> None:
    row = separation_update_diagnostics(
        current_offsets={"source.separation_as": 4.0e-6},
        posterior_offsets={"source.separation_as": 0.0},
        next_offsets={"source.separation_as": 2.0e-6},
    )
    assert row["separation_reference_error_before_microas"] == pytest.approx(4.0)
    assert row["separation_posterior_error_after_microas"] == pytest.approx(0.0)
    assert row["separation_next_reference_error_microas"] == pytest.approx(2.0)
    assert row["separation_posterior_update_microas"] == pytest.approx(-4.0)
    assert row["separation_applied_update_microas"] == pytest.approx(-2.0)
    assert row["separation_update_sign_toward_truth"] is True
    assert row["separation_next_reference_improved"] is True
