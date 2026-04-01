from __future__ import annotations

import numpy as np
import pytest

from dluxshera.utils.obs_subblock_trace_builders import (
    build_obs_subblock_trace_plan,
    generate_obs_subblock_trace_rows,
    resolve_obs_subblock_trace_anchors,
)


def _three_key_cfg() -> dict:
    return {
        "n_frames": 3,
        "dt_s": 0.5,
        "varying_keys": [
            "source.x_position_as",
            "source.y_position_as",
            "source.position_angle_deg",
        ],
        "plan": {
            "source.x_position_as": {
                "base": 0.0,
                "effects": [{"kind": "constant_offset", "offset": 0.0}],
            },
            "source.y_position_as": {
                "base": 0.0,
                "effects": [{"kind": "constant_offset", "offset": 0.0}],
            },
            "source.position_angle_deg": {
                "base": 90.0,
                "effects": [{"kind": "constant_offset", "offset": 0.0}],
            },
        },
    }


def test_missing_nominal_anchor_fails_when_base_omitted():
    cfg = _three_key_cfg()
    cfg["plan"]["source.x_position_as"]["base"] = None
    plan = build_obs_subblock_trace_plan(cfg)

    with pytest.raises(ValueError, match="base is required"):
        resolve_obs_subblock_trace_anchors(plan)


def test_explicit_base_takes_precedence_over_nominal_anchor():
    cfg = _three_key_cfg()
    cfg["plan"]["source.x_position_as"] = {
        "base": 2.0,
        "effects": [{"kind": "constant_offset", "offset": 0.0}],
    }
    plan = build_obs_subblock_trace_plan(cfg)
    anchors = resolve_obs_subblock_trace_anchors(
        plan,
        nominal_anchors={
            "source.x_position_as": 100.0,
            "source.y_position_as": -1.0,
            "source.position_angle_deg": 5.0,
        },
    )
    rows = generate_obs_subblock_trace_rows(plan, anchors=anchors)
    assert np.allclose([row["source.x_position_as"] for row in rows], [2.0, 2.0, 2.0])


def test_omitted_base_uses_nominal_anchor():
    cfg = _three_key_cfg()
    cfg["plan"]["source.x_position_as"] = {
        "effects": [{"kind": "constant_offset", "offset": 0.25}],
    }
    plan = build_obs_subblock_trace_plan(cfg)
    anchors = resolve_obs_subblock_trace_anchors(
        plan,
        nominal_anchors={
            "source.x_position_as": 1.0,
            "source.y_position_as": 0.0,
            "source.position_angle_deg": 90.0,
        },
    )
    rows = generate_obs_subblock_trace_rows(plan, anchors=anchors)
    assert np.allclose([row["source.x_position_as"] for row in rows], [1.25, 1.25, 1.25])


def test_additive_effects_sum_for_single_key():
    cfg = _three_key_cfg()
    cfg["plan"]["source.x_position_as"] = {
        "base": 1.0,
        "effects": [
            {"kind": "constant_offset", "offset": 0.5},
            {"kind": "linear_drift", "start": 0.0, "rate_per_s": 2.0},
        ],
    }
    plan = build_obs_subblock_trace_plan(cfg)
    anchors = resolve_obs_subblock_trace_anchors(plan)
    rows = generate_obs_subblock_trace_rows(plan, anchors=anchors)
    values = np.asarray([row["source.x_position_as"] for row in rows], dtype=float)
    assert np.allclose(values, np.asarray([1.5, 2.5, 3.5], dtype=float))


def test_linear_plus_iid_jitter_is_reproducible():
    cfg = _three_key_cfg()
    cfg["seed"] = 17
    cfg["plan"]["source.position_angle_deg"] = {
        "base": 90.0,
        "effects": [
            {"kind": "linear_drift", "start": 0.0, "rate_per_s": 0.4},
            {"kind": "iid_jitter", "center": 0.0, "sigma": 0.1},
        ],
    }
    plan_a = build_obs_subblock_trace_plan(cfg)
    plan_b = build_obs_subblock_trace_plan(cfg)

    rows_a = generate_obs_subblock_trace_rows(
        plan_a,
        anchors=resolve_obs_subblock_trace_anchors(plan_a),
    )
    rows_b = generate_obs_subblock_trace_rows(
        plan_b,
        anchors=resolve_obs_subblock_trace_anchors(plan_b),
    )
    assert np.allclose(
        [row["source.position_angle_deg"] for row in rows_a],
        [row["source.position_angle_deg"] for row in rows_b],
    )


def test_random_walk_reproducible_under_fixed_seed():
    cfg = _three_key_cfg()
    cfg["seed"] = 23
    cfg["plan"]["source.y_position_as"] = {
        "base": 0.0,
        "effects": [{"kind": "random_walk", "start": 1.0, "sigma_step": 0.2}],
    }
    plan_a = build_obs_subblock_trace_plan(cfg)
    plan_b = build_obs_subblock_trace_plan(cfg)

    rows_a = generate_obs_subblock_trace_rows(
        plan_a,
        anchors=resolve_obs_subblock_trace_anchors(plan_a),
    )
    rows_b = generate_obs_subblock_trace_rows(
        plan_b,
        anchors=resolve_obs_subblock_trace_anchors(plan_b),
    )
    assert np.allclose(
        [row["source.y_position_as"] for row in rows_a],
        [row["source.y_position_as"] for row in rows_b],
    )


def test_mixed_mode_generation_supports_generalized_keys():
    cfg = {
        "n_frames": 4,
        "dt_s": 0.1,
        "seed": 5,
        "varying_keys": [
            "source.x_position_as",
            "optics.plate_scale_as_per_pix",
            "optics.primary.zernike_coeffs_nm[2]",
        ],
        "plan": {
            "source.x_position_as": {
                "base": 0.0,
                "effects": [{"kind": "linear_drift", "start": 0.0, "rate_per_s": 0.5}],
            },
            "optics.plate_scale_as_per_pix": {
                "base": 0.11,
                "effects": [{"kind": "iid_jitter", "center": 0.0, "sigma": 0.0}],
            },
            "optics.primary.zernike_coeffs_nm[2]": {
                "base": 0.0,
                "effects": [
                    {"kind": "random_walk", "start": 0.0, "sigma_step": 0.05},
                ],
            },
        },
    }

    plan = build_obs_subblock_trace_plan(cfg)
    rows = generate_obs_subblock_trace_rows(
        plan,
        anchors=resolve_obs_subblock_trace_anchors(plan),
    )

    assert len(rows) == 4
    assert list(rows[0].keys()) == [
        "frame_index",
        "time_s",
        "source.x_position_as",
        "optics.plate_scale_as_per_pix",
        "optics.primary.zernike_coeffs_nm[2]",
    ]
    assert [row["frame_index"] for row in rows] == [0, 1, 2, 3]
    assert np.allclose([row["time_s"] for row in rows], [0.0, 0.1, 0.2, 0.3])


def test_old_trace_plan_schema_is_rejected():
    cfg = {
        "n_frames": 3,
        "dt_s": 0.5,
        "trace_plan": {
            "source.x_position_as": {
                "base": 1.0,
                "effects": [{"kind": "constant_offset", "offset": 0.0}],
            },
        }
    }
    with pytest.raises(ValueError, match="experiment.trace must define 'plan'"):
        build_obs_subblock_trace_plan(cfg)
