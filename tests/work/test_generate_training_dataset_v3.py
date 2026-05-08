from __future__ import annotations

import pytest

from work.experiments.generate_training_dataset_v3 import (
    ScalarParameter,
    SweepConfig,
    _build_nuisance_draws,
    _build_pair_grid_plan,
    _build_sparse_mixture_plan,
    _build_pair_grid_levels,
    _resolve_sweep_for_label,
)


def _param(label: str, base_key: str | None = None, sigma: float = 0.1) -> ScalarParameter:
    sweep = SweepConfig(min_sigma=1.0, max_sigma=3.0, n_magnitudes=3, spacing="log")
    return ScalarParameter(
        label=label,
        base_key=base_key or label,
        component_index=None,
        nominal_value=0.0,
        parameter_sigma=sigma,
        sweep_source_key=base_key or label,
        sweep_config=sweep,
        min_abs_delta=sweep.min_sigma * sigma,
        max_abs_delta=sweep.max_sigma * sigma,
        display_label=label,
        group=label.split(".", 1)[0],
    )


def test_resolve_v2_style_sweep_for_scalarized_label() -> None:
    default = SweepConfig(min_sigma=1.0, max_sigma=10.0, n_magnitudes=5, spacing="log")
    zernike = SweepConfig(min_sigma=2.0, max_sigma=20.0, n_magnitudes=7, spacing="log")
    resolved = _resolve_sweep_for_label(
        "optics.primary.zernike_coeffs_nm[3]",
        base_key="optics.primary.zernike_coeffs_nm",
        per_parameter_cfg={"optics.primary.zernike_coeffs_nm": zernike},
        default_cfg=default,
    )
    assert resolved == zernike


def test_symmetric_pair_grid_levels_include_zero() -> None:
    levels = _build_pair_grid_levels(
        _param("a", sigma=2.0),
        pair_cfg={"level_mode": "symmetric_grid_from_sweeps", "grid_size": 3, "include_zero": True},
    )
    assert [level["sigma"] for level in levels] == pytest.approx([-3.0, 0.0, 3.0])
    assert [level["delta"] for level in levels] == pytest.approx([-6.0, 0.0, 6.0])


def test_pair_plan_count_for_tiny_parameter_space_nominal_nuisance_only() -> None:
    params = [_param("a"), _param("b"), _param("c")]
    rows = _build_pair_grid_plan(
        parameters=params,
        pair_cfg={"enabled": True, "level_mode": "symmetric_grid_from_sweeps", "grid_size": 3},
        nuisance_cfg={"enabled": True, "include_nominal": True, "n_random": 0, "keys": []},
        seed=0,
    )
    assert len(rows) == 3 * 9
    assert {row["pair_id"] for row in rows} == {"pair_000_001", "pair_000_002", "pair_001_002"}


def test_nuisance_collision_policy_skips_controlled_axis() -> None:
    params = [
        _param("source.x_position_as"),
        _param("source.y_position_as"),
        _param("source.contrast"),
    ]
    rows = _build_pair_grid_plan(
        parameters=params,
        pair_cfg={"enabled": True, "level_mode": "symmetric_grid_from_sweeps", "grid_size": 3},
        nuisance_cfg={
            "enabled": True,
            "include_nominal": False,
            "n_random": 1,
            "keys": ["source.x_position_as", "source.y_position_as"],
            "collision_policy": "skip_if_key_is_controlled_axis",
        },
        seed=123,
    )
    controlled_x_row = next(row for row in rows if row["pair_label_i"] == "source.x_position_as")
    assert "source.x_position_as" not in controlled_x_row["registration_nuisance_values"]
    assert "source.x_position_as" in controlled_x_row["skipped_nuisance_keys"]


def test_sparse_mixture_plan_is_deterministic() -> None:
    params = [_param("a"), _param("b"), _param("c")]
    cfg = {
        "enabled": True,
        "n_samples": 5,
        "active_count_probs": {1: 0.25, 2: 0.50, 3: 0.25},
        "amplitude_sampling": {"signed": True},
        "nuisance": {"enabled": False},
    }
    nuisance_cfg = {"enabled": False}
    first = _build_sparse_mixture_plan(parameters=params, sparse_cfg=cfg, nuisance_cfg=nuisance_cfg, seed=9)
    second = _build_sparse_mixture_plan(parameters=params, sparse_cfg=cfg, nuisance_cfg=nuisance_cfg, seed=9)
    assert [row["active_labels"] for row in first] == [row["active_labels"] for row in second]
    assert [row["theta_delta"] for row in first] == [row["theta_delta"] for row in second]


def test_build_nuisance_draws_includes_nominal_id_zero() -> None:
    draws = _build_nuisance_draws(
        parameters_by_label={"source.x_position_as": _param("source.x_position_as")},
        nuisance_cfg={"enabled": True, "include_nominal": True, "n_random": 0, "keys": ["source.x_position_as"]},
        seed=0,
    )
    assert draws == [
        {
            "nuisance_id": 0,
            "values": {"source.x_position_as": 0.0},
            "sigma_values": {"source.x_position_as": 0.0},
            "sample_role_suffix": "nominal_registration",
        }
    ]
