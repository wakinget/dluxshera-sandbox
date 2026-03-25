from __future__ import annotations

import numpy as np
import pytest

from dluxshera.utils.obs_subblock_trace import REQUIRED_TRACE_COLUMNS
from dluxshera.utils.obs_subblock_trace_builders import (
    build_obs_subblock_trace_plan,
    generate_obs_subblock_trace_rows,
)


def _base_cfg() -> dict:
    return {
        "n_frames": 3,
        "dt_s": 0.5,
        "keys": {
            "source.x_position_as": {"mode": "explicit", "values": [0.0, 0.1, 0.2]},
            "source.y_position_as": {"mode": "explicit", "values": [0.0, -0.1, -0.2]},
            "source.position_angle_deg": {
                "mode": "explicit",
                "values": [90.0, 90.2, 90.4],
            },
        },
    }


def test_explicit_mode_requires_values_length_matching_n_frames():
    cfg = _base_cfg()
    cfg["keys"]["source.x_position_as"] = {"mode": "explicit", "values": [0.0, 0.1]}

    with pytest.raises(ValueError, match="length must equal n_frames"):
        build_obs_subblock_trace_plan(cfg)


def test_linear_drift_values_follow_time_axis():
    cfg = _base_cfg()
    cfg["keys"]["source.x_position_as"] = {
        "mode": "linear_drift",
        "start": 1.0,
        "rate_per_s": 2.0,
    }

    plan = build_obs_subblock_trace_plan(cfg)
    rows = generate_obs_subblock_trace_rows(plan)
    values = np.asarray([row["source.x_position_as"] for row in rows], dtype=float)

    assert np.allclose(values, np.asarray([1.0, 2.0, 3.0], dtype=float))


def test_random_walk_is_reproducible_under_fixed_seed():
    cfg = _base_cfg()
    cfg["seed"] = 123
    cfg["keys"]["source.y_position_as"] = {
        "mode": "random_walk",
        "start": 0.5,
        "sigma_step": 0.25,
    }

    plan_a = build_obs_subblock_trace_plan(cfg)
    rows_a = generate_obs_subblock_trace_rows(plan_a)

    plan_b = build_obs_subblock_trace_plan(cfg)
    rows_b = generate_obs_subblock_trace_rows(plan_b)

    series_a = [row["source.y_position_as"] for row in rows_a]
    series_b = [row["source.y_position_as"] for row in rows_b]
    assert np.allclose(series_a, series_b)


def test_iid_jitter_is_reproducible_under_fixed_seed():
    cfg = _base_cfg()
    cfg["seed"] = 77
    cfg["keys"]["source.position_angle_deg"] = {
        "mode": "iid_jitter",
        "center": 90.0,
        "sigma": 0.1,
    }

    plan_a = build_obs_subblock_trace_plan(cfg)
    rows_a = generate_obs_subblock_trace_rows(plan_a)

    plan_b = build_obs_subblock_trace_plan(cfg)
    rows_b = generate_obs_subblock_trace_rows(plan_b)

    series_a = [row["source.position_angle_deg"] for row in rows_a]
    series_b = [row["source.position_angle_deg"] for row in rows_b]
    assert np.allclose(series_a, series_b)


def test_mixed_mode_generation_produces_canonical_schema():
    cfg = {
        "n_frames": 4,
        "dt_s": 0.1,
        "seed": 5,
        "keys": {
            "source.x_position_as": {"mode": "explicit", "values": [0.0, 0.1, 0.2, 0.3]},
            "source.y_position_as": {
                "mode": "linear_drift",
                "start": 0.0,
                "rate_per_s": -0.5,
            },
            "source.position_angle_deg": {
                "mode": "random_walk",
                "start": 90.0,
                "sigma_step": 0.05,
            },
        },
    }

    plan = build_obs_subblock_trace_plan(cfg)
    rows = generate_obs_subblock_trace_rows(plan)

    assert len(rows) == 4
    assert list(rows[0].keys()) == list(REQUIRED_TRACE_COLUMNS)
    assert [row["frame_index"] for row in rows] == [0, 1, 2, 3]
    assert np.allclose([row["time_s"] for row in rows], [0.0, 0.1, 0.2, 0.3])
