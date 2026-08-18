from __future__ import annotations

import numpy as np
import pytest

from dluxshera.utils.trajectory_filters import (
    apply_trajectory_filter,
    parse_trajectory_filter_config,
)


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _amp_at(time_s: np.ndarray, values: np.ndarray, freq_hz: float) -> float:
    basis_sin = np.sin(2.0 * np.pi * freq_hz * time_s)
    basis_cos = np.cos(2.0 * np.pi * freq_hz * time_s)
    return float(
        2.0
        * np.hypot(np.dot(values, basis_sin), np.dot(values, basis_cos))
        / values.size
    )


def test_cutoff_period_maps_to_frequency():
    spec = parse_trajectory_filter_config(
        {"enabled": True, "kind": "high_pass", "cutoff_period_s": 15.0}
    )

    assert np.isclose(spec.cutoff_hz, 1.0 / 15.0)


def test_low_pass_bessel_preserves_slow_and_suppresses_fast():
    time_s = np.arange(0.0, 120.0, 0.1)
    slow_hz = 0.02
    fast_hz = 1.0
    values = np.sin(2.0 * np.pi * slow_hz * time_s) + 0.5 * np.sin(2.0 * np.pi * fast_hz * time_s)
    spec = parse_trajectory_filter_config(
        {"enabled": True, "kind": "low_pass", "order": 4, "cutoff_hz": 0.1}
    )

    filtered, provenance = apply_trajectory_filter(time_s, values, spec)

    assert _amp_at(time_s, filtered, slow_hz) > 0.85
    assert _amp_at(time_s, filtered, fast_hz) < 0.08
    assert provenance["method"] == "bessel"
    assert "input_rms_by_column" in provenance


def test_high_pass_bessel_suppresses_slow_and_preserves_fast():
    time_s = np.arange(0.0, 120.0, 0.1)
    slow_hz = 0.02
    fast_hz = 1.0
    values = np.sin(2.0 * np.pi * slow_hz * time_s) + 0.5 * np.sin(2.0 * np.pi * fast_hz * time_s)
    spec = parse_trajectory_filter_config(
        {"enabled": True, "kind": "high_pass", "order": 4, "cutoff_hz": 0.1}
    )

    filtered, provenance = apply_trajectory_filter(time_s, values, spec)

    assert _amp_at(time_s, filtered, slow_hz) < 0.15
    assert _amp_at(time_s, filtered, fast_hz) > 0.35
    assert provenance["removed_rms_by_column"]["value"] > 0.4


def test_band_pass_preserves_intermediate_sinusoid():
    time_s = np.arange(0.0, 120.0, 0.05)
    low_hz = 0.03
    mid_hz = 0.5
    high_hz = 2.0
    values = (
        np.sin(2.0 * np.pi * low_hz * time_s)
        + 0.7 * np.sin(2.0 * np.pi * mid_hz * time_s)
        + 0.5 * np.sin(2.0 * np.pi * high_hz * time_s)
    )
    spec = parse_trajectory_filter_config(
        {
            "enabled": True,
            "kind": "band_pass",
            "order": 4,
            "low_cutoff_hz": 0.15,
            "high_cutoff_hz": 1.0,
        }
    )

    filtered, _ = apply_trajectory_filter(time_s, values, spec)

    assert _amp_at(time_s, filtered, mid_hz) > 0.45
    assert _amp_at(time_s, filtered, low_hz) < 0.2
    assert _amp_at(time_s, filtered, high_hz) < 0.15


def test_zero_phase_filter_does_not_introduce_measurable_lag():
    time_s = np.arange(0.0, 120.0, 0.1)
    values = np.sin(2.0 * np.pi * 0.5 * time_s)
    spec = parse_trajectory_filter_config(
        {"enabled": True, "kind": "low_pass", "order": 4, "cutoff_hz": 1.5, "zero_phase": True}
    )

    filtered, _ = apply_trajectory_filter(time_s, values, spec)
    corr = np.correlate(filtered - filtered.mean(), values - values.mean(), mode="full")
    lag_samples = int(np.argmax(corr) - (values.size - 1))

    assert abs(lag_samples) <= 1


def test_invalid_cutoff_above_nyquist_fails_clearly():
    time_s = np.arange(0.0, 10.0, 0.1)
    spec = parse_trajectory_filter_config(
        {"enabled": True, "kind": "low_pass", "cutoff_hz": 6.0}
    )

    with pytest.raises(ValueError, match="below Nyquist"):
        apply_trajectory_filter(time_s, np.ones_like(time_s), spec)


def test_nonuniform_time_grid_fails_clearly():
    time_s = np.asarray([0.0, 0.1, 0.21, 0.3, 0.4, 0.5])
    spec = parse_trajectory_filter_config(
        {"enabled": True, "kind": "low_pass", "cutoff_hz": 1.0}
    )

    with pytest.raises(ValueError, match="uniform sampling"):
        apply_trajectory_filter(time_s, np.ones_like(time_s), spec)


def test_too_short_zero_phase_series_fails_clearly():
    time_s = np.arange(0.0, 1.0, 0.1)
    spec = parse_trajectory_filter_config(
        {"enabled": True, "kind": "high_pass", "order": 4, "cutoff_hz": 0.5}
    )

    with pytest.raises(ValueError, match="zero-phase padding"):
        apply_trajectory_filter(time_s, np.ones_like(time_s), spec)


def test_none_filter_returns_values_unchanged_with_provenance():
    time_s = np.arange(0.0, 10.0, 0.1)
    values = np.column_stack([np.sin(time_s), np.cos(time_s)])
    spec = parse_trajectory_filter_config({"enabled": False, "kind": "none", "columns": ["x", "y"]})

    filtered, provenance = apply_trajectory_filter(time_s, values, spec)

    assert np.allclose(filtered, values)
    assert provenance["enabled"] is False
    assert provenance["columns_filtered"] == []
    assert _rms(np.asarray(list(provenance["removed_rms_by_column"].values()))) == 0.0
