from __future__ import annotations

import pytest

from work.experiments.generate_training_dataset_v2 import (
    SweepConfig,
    _build_sigma_summary,
    _normalize_sweep_configs,
    compute_preview_counts,
    compute_expected_sample_counts,
    generate_mirrored_sigma_offsets,
)


def test_generate_mirrored_sigma_offsets_log_spacing_and_order() -> None:
    offsets = generate_mirrored_sigma_offsets(
        min_sigma=0.1,
        max_sigma=10.0,
        n_magnitudes=3,
        spacing="log",
    )
    assert offsets == pytest.approx([-10.0, -1.0, -0.1, 0.1, 1.0, 10.0])
    assert 0.0 not in offsets


def test_normalize_sweep_configs_supports_per_parameter_overrides() -> None:
    default_cfg = SweepConfig(min_sigma=0.1, max_sigma=10.0, n_magnitudes=8, spacing="log")
    out = _normalize_sweep_configs(
        infer_keys=["a", "b"],
        default_cfg=default_cfg,
        overrides={
            "a": {"min_sigma": 0.01, "max_sigma": 5.0, "n_magnitudes": 6},
        },
    )
    assert out["a"] == SweepConfig(min_sigma=0.01, max_sigma=5.0, n_magnitudes=6, spacing="log")
    assert out["b"] == default_cfg


def test_normalize_sweep_configs_rejects_conflicting_aliases() -> None:
    default_cfg = SweepConfig(min_sigma=0.1, max_sigma=10.0, n_magnitudes=8, spacing="log")
    with pytest.raises(ValueError, match="differ"):
        _normalize_sweep_configs(
            sweep_keys=["a"],
            infer_keys=["b"],
            default_cfg=default_cfg,
            overrides={},
        )


def test_sigma_to_delta_conversion_semantics() -> None:
    parameter_sigma = 2.5
    sigma_offset = -3.2
    delta_value = sigma_offset * parameter_sigma
    assert delta_value == pytest.approx(-8.0)


def test_build_sigma_summary_has_expected_counts_and_ranges() -> None:
    summary = _build_sigma_summary(
        parameter_name="binary.separation_as",
        nominal_value=1.2,
        parameter_sigma=0.5,
        sweep_cfg=SweepConfig(min_sigma=0.2, max_sigma=5.0, n_magnitudes=4, spacing="log"),
    )
    assert summary["total_nonzero_samples"] == 8
    assert summary["min_abs_delta"] == pytest.approx(0.1)
    assert summary["max_abs_delta"] == pytest.approx(2.5)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"min_sigma": 0.0, "max_sigma": 10.0, "n_magnitudes": 2, "spacing": "log"}, "min_sigma"),
        ({"min_sigma": 0.1, "max_sigma": 0.0, "n_magnitudes": 2, "spacing": "log"}, "max_sigma"),
        ({"min_sigma": 1.0, "max_sigma": 1.0, "n_magnitudes": 2, "spacing": "log"}, "<"),
        ({"min_sigma": 0.1, "max_sigma": 10.0, "n_magnitudes": 0, "spacing": "log"}, ">="),
    ],
)
def test_generate_mirrored_sigma_offsets_validation(kwargs: dict[str, float | int | str], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        generate_mirrored_sigma_offsets(**kwargs)


def test_nominal_sample_count_is_one_and_not_duplicated() -> None:
    counts = compute_expected_sample_counts(n_swept_components=5, n_magnitudes=3)
    assert counts == {"nominal": 1, "perturbed": 30, "total": 31}


def test_compute_preview_counts_honors_per_parameter_n_magnitudes() -> None:
    counts = compute_preview_counts(
        per_parameter_cfg={
            "a": SweepConfig(n_magnitudes=2),
            "z": SweepConfig(n_magnitudes=4),
        },
        scalar_keys=["a"],
        zernike_component_counts={"z": 3},
    )
    # scalar a => 2*2=4, z with 3 comps => 3*(2*4)=24, plus one nominal
    assert counts == {"nominal": 1, "perturbed": 28, "total": 29}
