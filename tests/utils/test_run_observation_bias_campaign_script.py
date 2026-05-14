from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "scripts"
    / "run_observation_bias_campaign.py"
)


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "run_observation_bias_campaign",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_config(path: Path) -> None:
    payload = {
        "experiment": {
            "kind": "observation_bias_campaign",
            "run_name": "unit_campaign",
            "subblocks": {
                "n_subblocks": 2,
                "n_frames": 1,
                "noise": "disabled",
                "phi_ref": "truth_when_available",
                "schur_curvature_method": "auto",
                "max_dense_dim": 40,
                "schur_damping": 1.0e-8,
                "trace_jitter": {
                    "x_sigma_as": 1.0e-3,
                    "y_sigma_as": 1.0e-3,
                    "pa_sigma_deg": 1.0e-4,
                },
            },
            "seeding": {
                "seed_policy": "different_jitter_different_noise",
                "base_seed": 42,
            },
            "observation_theta": {
                "source": {
                    "separation_as": True,
                    "log_flux_total": False,
                    "contrast": False,
                },
                "optics": {
                    "plate_scale_as_per_pix": False,
                    "primary_zernikes": {
                        "enabled": True,
                        "indices": "from_system",
                        "include": [0],
                        "exclude": [],
                    },
                    "secondary_zernikes": {
                        "enabled": True,
                        "indices": "from_system",
                        "include": [0],
                        "exclude": [],
                    },
                },
            },
            "eigenbasis": {
                "enabled": True,
                "sources": ["accumulated_information", "posterior_precision"],
                "whiten": True,
                "eig_floor_abs": 0.0,
                "eig_floor_rel": 1.0e-12,
                "top_k_contributors": 3,
            },
            "bias_cases": [
                {
                    "case_name": "zero_bias_full_zernike",
                    "theta_reference_offsets": {},
                },
                {
                    "case_name": "matched_pair",
                    "theta_reference_offsets": {
                        "optics.primary.zernike_coeffs_nm[0]": 5.0,
                        "optics.secondary.zernike_coeffs_nm[0]": 5.0,
                    },
                },
            ],
            "prior_draws": {"enabled": False},
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def enable_forecast(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["experiment"]["forecast"] = {
        "enabled": True,
        "modes": ["replicate", "fixed_information_score_noise"],
        "n_subblocks_grid": [1, 3, 5],
        "subblock_duration_s": 1.0,
        "single_observation_n_subblocks": 5,
        "replicate": {"enabled": True},
        "fixed_information_score_noise": {
            "enabled": True,
            "n_trials": 2,
            "seed": 2026,
            "score_noise_alpha": 0.5,
            "score_noise_eig_floor_abs": 0.0,
            "score_noise_eig_floor_rel": 1.0e-12,
            "truth_mode": "campaign_truth",
        },
        "plots": False,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_summary(path: Path, labels: tuple[str, ...], theta_ref: np.ndarray, truth: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(labels)
    info = np.eye(n, dtype=float)
    if n >= 3:
        common = np.array([0.0, 1.0, 1.0]) / np.sqrt(2.0)
        differential = np.array([0.0, 1.0, -1.0]) / np.sqrt(2.0)
        info += 4.0 * np.outer(common, common)
        info += 0.01 * np.outer(differential, differential)
    score = info @ (theta_ref - truth)
    payload = {
        "schema_version": "image_backed_subblock_summary.v1",
        "subblock_id": path.parent.parent.parent.name,
        "summary_kind": "synthetic_test",
        "theta_labels": list(labels),
        "theta_ref": theta_ref.tolist(),
        "reduced_information": info.tolist(),
        "reduced_score": score.tolist(),
        "summary_diagnostics": {},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_bias_case_parsing_validates_layout_keys(tmp_path: Path):
    module = load_module()
    config_path = tmp_path / "campaign.json"
    write_config(config_path)
    plan = module.build_campaign_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="unit_campaign",
        system_preset="SHERA_FLIGHT_3P",
    )

    assert plan.layout.labels == (
        "source.separation_as",
        "optics.primary.zernike_coeffs_nm[0]",
        "optics.secondary.zernike_coeffs_nm[0]",
    )
    assert plan.cases[1].theta_reference_offsets[
        "optics.primary.zernike_coeffs_nm[0]"
    ] == 5.0
    first_command = plan.subblock_commands["matched_pair"][0]
    assert "--trace-seed" in first_command
    assert "--render-seed" in first_command

    bad = json.loads(config_path.read_text(encoding="utf-8"))
    bad["experiment"]["bias_cases"][0]["theta_reference_offsets"] = {
        "source.contrast": 1.0
    }
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="not in the resolved observation theta layout"):
        module.build_campaign_plan(
            config_path=bad_path,
            results_root=tmp_path,
            run_name="bad",
            system_preset="SHERA_FLIGHT_3P",
        )


def test_command_construction_includes_schur_summary_arguments(tmp_path: Path):
    module = load_module()
    command = module.build_subblock_command(
        case_root_parent=tmp_path / "subblocks",
        case_subblock_name="case/subblock_000",
        theta_labels=(
            "source.separation_as",
            "optics.primary.zernike_coeffs_nm[0]",
        ),
        offsets={"optics.primary.zernike_coeffs_nm[0]": 5.0},
        subblock_cfg={
            "n_frames": 3,
            "noise": "enabled",
            "phi_ref": "recovered",
            "schur_curvature_method": "auto",
            "max_dense_dim": 40,
            "schur_damping": 1.0e-8,
            "trace_jitter": {
                "x_sigma_as": 1.0e-3,
                "y_sigma_as": 1.0e-3,
                "pa_sigma_deg": 1.0e-4,
            },
        },
        trace_seed=123,
        noise_seed=456,
    )

    assert "--mode" in command
    assert "schur_summary" in command
    assert "--theta-keys" in command
    assert "source.separation_as,optics.primary.zernike_coeffs_nm[0]" in command
    assert "--theta-reference-offset" in command
    assert "optics.primary.zernike_coeffs_nm[0]=5.0" in command
    assert "--phi-ref" in command
    assert "recovered" in command
    assert "--n-frames" in command
    assert "3" in command
    assert "--trace-seed" in command
    assert "123" in command
    assert "--render-seed" in command
    assert "456" in command
    assert "--trace-jitter-x-sigma-as" in command
    assert "--trace-jitter-y-sigma-as" in command
    assert "--trace-jitter-pa-sigma-deg" in command


def test_command_forwarding_includes_diagnostics_exposure_and_quality_flags(tmp_path: Path):
    module = load_module()
    command = module.build_subblock_command(
        case_root_parent=tmp_path / "subblocks",
        case_subblock_name="case/subblock_000",
        theta_labels=("source.separation_as",),
        offsets={},
        subblock_cfg={
            "n_frames": 3,
            "noise": "enabled",
            "phi_ref": "recovered",
            "exposure_time_s": 0.05,
            "reference_diagnostics_profile": "basic",
            "schur_frame_quality_policy": "mask",
            "schur_frame_chi2_threshold": 5.0,
            "schur_frame_quality_missing": "allow_all",
            "schur_frame_mask_denominator": "original",
            "schur_frame_mask_min_good_frames": 1,
            "variance_floor": 1.0,
        },
        trace_seed=123,
        noise_seed=456,
    )

    assert "--reference-diagnostics-profile" in command
    assert "basic" in command
    assert "--exposure-time-s" in command
    assert "0.05" in command
    assert "--schur-frame-quality-policy" in command
    assert "mask" in command
    assert "--schur-frame-chi2-threshold" in command
    assert "5.0" in command
    assert "--schur-frame-quality-missing" in command
    assert "allow_all" in command
    assert "--schur-frame-mask-denominator" in command
    assert "original" in command
    assert "--schur-frame-mask-min-good-frames" in command
    assert "--variance-floor" in command


def test_command_forwarding_includes_reference_optimizer_controls(tmp_path: Path):
    module = load_module()
    command = module.build_subblock_command(
        case_root_parent=tmp_path / "subblocks",
        case_subblock_name="case/subblock_000",
        theta_labels=("source.separation_as",),
        offsets={},
        subblock_cfg={
            "reference_optimizer_kind": "sgd",
            "reference_base_lr": 0.7,
            "reference_n_iter": 80,
            "reference_schedule_kind": "linear_warmup",
            "reference_schedule_warmup_steps": 10,
            "reference_schedule_start_factor": 0.125,
            "reference_preconditioning_enabled": True,
            "reference_preconditioning_method": "fisher",
            "reference_preconditioning_reference": "truth_when_available",
            "reference_preconditioning_damping": 1.0e-6,
            "reference_preconditioning_eig_floor_rel": 1.0e-8,
            "reference_preconditioning_eig_floor_abs": 1.0e-10,
            "reference_preconditioning_lr_clip": [0.1, 2.0],
        },
    )

    assert "--reference-optimizer-kind" in command
    assert "sgd" in command
    assert "--reference-base-lr" in command
    assert "0.7" in command
    assert "--reference-n-iter" in command
    assert "80" in command
    assert "--reference-schedule-kind" in command
    assert "linear_warmup" in command
    assert "--reference-schedule-warmup-steps" in command
    assert "10" in command
    assert "--reference-schedule-start-factor" in command
    assert "0.125" in command
    assert "--reference-preconditioning-enabled" in command
    assert "--reference-preconditioning-method" in command
    assert "--reference-preconditioning-reference" in command
    assert "--reference-preconditioning-damping" in command
    assert "--reference-preconditioning-eig-floor-rel" in command
    assert "--reference-preconditioning-eig-floor-abs" in command
    assert "--reference-preconditioning-lr-clip" in command
    assert "0.1,2.0" in command


def test_subblock_plan_records_forwarded_command_options(tmp_path: Path):
    module = load_module()
    config_path = tmp_path / "campaign.json"
    write_config(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["experiment"]["subblocks"].update(
        {
            "exposure_time_s": 0.05,
            "reference_diagnostics_profile": "basic",
            "schur_frame_quality_policy": "warn",
            "schur_frame_chi2_threshold": 5.0,
        }
    )
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    plan = module.build_campaign_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="forwarding_plan",
        system_preset="SHERA_FLIGHT_3P",
    )
    row = plan.subblock_plans["matched_pair"][0]
    assert row["exposure_time_s"] == pytest.approx(0.05)
    assert row["reference_diagnostics_profile"] == "basic"
    assert row["schur_frame_quality_policy"] == "warn"
    assert row["schur_frame_chi2_threshold"] == pytest.approx(5.0)
    payload = module._plan_payload(plan)
    assert payload["subblock_command_options"]["exposure_time_s"] == pytest.approx(0.05)
    assert "--exposure-time-s" in payload["subblock_command_options"]["forwarded_flags"]


def _prior_draw_only_payload(*, include_implicit_zero_bias: bool | None) -> dict[str, Any]:
    payload = {
        "experiment": {
            "kind": "observation_bias_campaign",
            "run_name": "prior_draw_only",
            "subblocks": {
                "n_subblocks": 1,
                "n_frames": 1,
                "noise": "disabled",
            },
            "observation_theta": {
                "source": {
                    "separation_as": True,
                    "log_flux_total": False,
                    "contrast": False,
                },
                "optics": {
                    "plate_scale_as_per_pix": False,
                    "primary_zernikes": {
                        "enabled": True,
                        "indices": "from_system",
                        "include": [0],
                        "exclude": [],
                    },
                    "secondary_zernikes": {
                        "enabled": True,
                        "indices": "from_system",
                        "include": [0],
                        "exclude": [],
                    },
                },
            },
            "bias_cases": [],
            "prior_draws": {
                "enabled": True,
                "n_cases": 1,
                "center": "truth",
                "distribution": "normal",
                "draw_seed": 12345,
                "case_name_template": "prior_draw_{draw_index:03d}",
                "sigmas": {
                    "source.separation_as": {"kind": "absolute", "sigma": 1.0e-5},
                    "optics.primary.zernike_coeffs_nm[*]": {
                        "kind": "absolute",
                        "sigma": 1.0e-1,
                    },
                    "optics.secondary.zernike_coeffs_nm[*]": {
                        "kind": "absolute",
                        "sigma": 1.0e-1,
                    },
                },
            },
        }
    }
    if include_implicit_zero_bias is not None:
        payload["experiment"]["case_generation"] = {
            "include_implicit_zero_bias": include_implicit_zero_bias
        }
    return payload


def test_implicit_zero_bias_can_be_disabled_for_prior_draw_only_runs(tmp_path: Path):
    module = load_module()
    config_path = tmp_path / "prior_draw_only.json"
    config_path.write_text(
        json.dumps(_prior_draw_only_payload(include_implicit_zero_bias=False)),
        encoding="utf-8",
    )
    plan = module.build_campaign_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="prior_draw_only",
        system_preset="SHERA_FLIGHT_3P",
    )
    assert [case.case_name for case in plan.cases] == ["prior_draw_000"]
    assert plan.case_generation["zero_bias_case_status"] == "disabled"


def test_implicit_zero_bias_default_is_preserved_for_prior_draw_only_runs(tmp_path: Path):
    module = load_module()
    config_path = tmp_path / "prior_draw_default.json"
    config_path.write_text(
        json.dumps(_prior_draw_only_payload(include_implicit_zero_bias=None)),
        encoding="utf-8",
    )
    plan = module.build_campaign_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="prior_draw_default",
        system_preset="SHERA_FLIGHT_3P",
    )
    assert [case.case_name for case in plan.cases] == [
        "zero_bias_full_zernike",
        "prior_draw_000",
    ]
    assert (
        plan.case_generation["zero_bias_case_status"]
        == "implicit_default_with_prior_draws"
    )


def test_no_cases_still_defaults_to_zero_bias(tmp_path: Path):
    module = load_module()
    config_path = tmp_path / "zero_default.json"
    payload = _prior_draw_only_payload(include_implicit_zero_bias=None)
    payload["experiment"]["prior_draws"] = {"enabled": False}
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    plan = module.build_campaign_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="zero_default",
        system_preset="SHERA_FLIGHT_3P",
    )
    assert [case.case_name for case in plan.cases] == ["zero_bias_full_zernike"]
    assert plan.cases[0].case_origin == "implicit_zero_bias"


def test_seed_policy_derivation_is_deterministic_and_distinct():
    module = load_module()
    first = module._derive_subblock_seeds(
        run_name="run",
        case_name="case",
        subblock_index=0,
        seed_policy="different_jitter_different_noise",
        base_seed=42,
    )
    second = module._derive_subblock_seeds(
        run_name="run",
        case_name="case",
        subblock_index=0,
        seed_policy="different_jitter_different_noise",
        base_seed=42,
    )
    third = module._derive_subblock_seeds(
        run_name="run",
        case_name="case",
        subblock_index=1,
        seed_policy="different_jitter_different_noise",
        base_seed=42,
    )
    assert first == second
    assert first["trace_seed"] != third["trace_seed"]
    assert first["noise_seed"] != third["noise_seed"]


def test_seed_policies_produce_expected_same_different_behavior():
    module = load_module()
    same_jitter_0 = module._derive_subblock_seeds(
        run_name="run",
        case_name="case",
        subblock_index=0,
        seed_policy="same_jitter_different_noise",
        base_seed=42,
    )
    same_jitter_1 = module._derive_subblock_seeds(
        run_name="run",
        case_name="case",
        subblock_index=1,
        seed_policy="same_jitter_different_noise",
        base_seed=42,
    )
    assert same_jitter_0["trace_seed"] == same_jitter_1["trace_seed"]
    assert same_jitter_0["noise_seed"] != same_jitter_1["noise_seed"]

    same_noise_0 = module._derive_subblock_seeds(
        run_name="run",
        case_name="case",
        subblock_index=0,
        seed_policy="different_jitter_same_noise",
        base_seed=42,
    )
    same_noise_1 = module._derive_subblock_seeds(
        run_name="run",
        case_name="case",
        subblock_index=1,
        seed_policy="different_jitter_same_noise",
        base_seed=42,
    )
    assert same_noise_0["trace_seed"] != same_noise_1["trace_seed"]
    assert same_noise_0["noise_seed"] == same_noise_1["noise_seed"]


def test_eigen_source_resolution_accepts_legacy_source_matrix():
    module = load_module()
    sources = module._resolve_eigen_sources({"source_matrix": "posterior_precision"})
    assert sources == ("posterior_precision",)


def test_dry_run_writes_plan_and_does_not_execute_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_module()
    config_path = tmp_path / "campaign.json"
    write_config(config_path)

    def fail_execute(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("subprocess execution should not run during dry-run")

    monkeypatch.setattr(module, "execute_subblocks", fail_execute)
    payload = module.run_observation_bias_campaign(
        config_path=config_path,
        results_root=tmp_path,
        run_name="dry",
        dry_run=True,
        system_preset="SHERA_FLIGHT_3P",
        quiet=True,
    )

    assert payload["run_root"].endswith("dry")
    assert payload["dimension_estimate"]["n_theta"] == 3
    assert (tmp_path / "dry" / "campaign_plan.json").exists()
    assert (tmp_path / "dry" / "subblock_plan.csv").exists()
    planned = payload["subblock_plan"]["matched_pair"][0]
    assert "trace_seed" in planned
    assert "noise_seed" in planned


def test_aggregate_update_from_synthetic_summaries(tmp_path: Path):
    module = load_module()
    config_path = tmp_path / "campaign.json"
    write_config(config_path)
    plan = module.build_campaign_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="aggregate",
        system_preset="SHERA_FLIGHT_3P",
    )
    case = plan.cases[1]
    labels = plan.layout.labels
    truth = plan.prior_truth.copy()
    theta_ref = truth.copy()
    theta_ref[1] += 5.0
    theta_ref[2] += 5.0
    for path in plan.summary_paths[case.case_name]:
        write_summary(path, labels, theta_ref, truth)

    result = module.aggregate_case(
        plan=plan,
        case=case,
        prior_source="summary_theta_ref",
    )

    case_root = Path(result["case_root"])
    assert (case_root / "posterior_by_label.csv").exists()
    assert (case_root / "eigenvalues_accumulated_information.csv").exists()
    assert (case_root / "eigenvalues_posterior_precision.csv").exists()
    assert (case_root / "weak_mode_summary_accumulated_information.csv").exists()
    assert (case_root / "weak_mode_summary_posterior_precision.csv").exists()
    contributors = (
        case_root / "eigenmode_contributors_accumulated_information.csv"
    ).read_text(encoding="utf-8")
    assert "optics.primary.zernike_coeffs_nm[0]" in contributors
    assert "optics.secondary.zernike_coeffs_nm[0]" in contributors
    matrix_diag = json.loads(
        (case_root / "matrix_diagnostics.json").read_text(encoding="utf-8")
    )
    assert "accumulated_information" in matrix_diag
    assert "posterior_precision" in matrix_diag


def test_forecast_grid_inserts_actual_count_and_rejects_invalid():
    module = load_module()
    assert module.parse_forecast_grid([1, 5, 1800, 5], actual_n_summaries=3) == (
        1,
        3,
        5,
        1800,
    )
    with pytest.raises(ValueError, match=">= 1"):
        module.parse_forecast_grid([0, 3], actual_n_summaries=2)


def test_replicate_summaries_for_count_preserves_template_order():
    module = load_module()
    labels = ("source.separation_as",)
    summary_a = module.SubblockSummary.from_reduced_form(
        subblock_id="A",
        theta_labels=labels,
        theta_ref=np.array([0.0]),
        reduced_information=np.array([[1.0]]),
        reduced_score=np.array([0.0]),
    )
    summary_b = module.SubblockSummary.from_reduced_form(
        subblock_id="B",
        theta_labels=labels,
        theta_ref=np.array([0.0]),
        reduced_information=np.array([[1.0]]),
        reduced_score=np.array([0.0]),
    )
    replicated = module.replicate_summaries_for_count([summary_a, summary_b], 5)
    assert [summary.subblock_id for summary in replicated] == ["A", "B", "A", "B", "A"]


def test_fixed_information_score_noise_synthesis_is_seed_deterministic():
    module = load_module()
    labels = ("source.separation_as", "optics.primary.zernike_coeffs_nm[0]")
    summary = module.SubblockSummary.from_reduced_form(
        subblock_id="template",
        theta_labels=labels,
        theta_ref=np.array([1.0, 2.0]),
        reduced_information=np.array([[2.0, 0.1], [0.1, 1.0]]),
        reduced_score=np.array([0.0, 0.0]),
    )
    kwargs = {
        "summaries": [summary],
        "n_subblocks": 3,
        "theta_true": np.array([1.0, 2.0]),
        "alpha": 0.5,
        "eig_floor_abs": 0.0,
        "eig_floor_rel": 1.0e-12,
    }
    first = module.synthesize_score_noise_summaries(
        **kwargs,
        rng=np.random.default_rng(123),
    )
    second = module.synthesize_score_noise_summaries(
        **kwargs,
        rng=np.random.default_rng(123),
    )
    third = module.synthesize_score_noise_summaries(
        **kwargs,
        rng=np.random.default_rng(456),
    )
    np.testing.assert_allclose(
        np.vstack([summary.reduced_score for summary in first]),
        np.vstack([summary.reduced_score for summary in second]),
    )
    assert not np.allclose(
        np.vstack([summary.reduced_score for summary in first]),
        np.vstack([summary.reduced_score for summary in third]),
    )


def test_forecast_outputs_are_written_for_both_modes(tmp_path: Path):
    module = load_module()
    config_path = tmp_path / "campaign.json"
    write_config(config_path)
    enable_forecast(config_path)
    plan = module.build_campaign_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="forecast_case",
        system_preset="SHERA_FLIGHT_3P",
    )
    case = plan.cases[1]
    labels = plan.layout.labels
    truth = plan.prior_truth.copy()
    theta_ref = truth.copy()
    for path in plan.summary_paths[case.case_name]:
        write_summary(path, labels, theta_ref, truth)

    result = module.aggregate_case(
        plan=plan,
        case=case,
        prior_source="summary_theta_ref",
    )
    case_root = Path(result["case_root"])
    assert (case_root / "forecast" / "replicate" / "forecast_results.csv").exists()
    assert (
        case_root
        / "forecast"
        / "fixed_information_score_noise"
        / "forecast_results.csv"
    ).exists()
    assert (
        case_root
        / "forecast"
        / "fixed_information_score_noise"
        / "trial_forecast_results.csv"
    ).exists()
    assert any(int(row["n_subblocks"]) == 5 for row in result["forecast_results"])


def test_aggregate_only_runs_forecast_without_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_module()
    config_path = tmp_path / "campaign.json"
    write_config(config_path)
    enable_forecast(config_path)
    plan = module.build_campaign_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="aggregate_only_forecast",
        system_preset="SHERA_FLIGHT_3P",
    )
    labels = plan.layout.labels
    truth = plan.prior_truth.copy()
    for case in plan.cases:
        for path in plan.summary_paths[case.case_name]:
            write_summary(path, labels, truth.copy(), truth)

    def fail_execute(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("subprocess execution should not run in aggregate-only")

    monkeypatch.setattr(module, "execute_subblocks", fail_execute)
    result = module.run_observation_bias_campaign(
        config_path=config_path,
        results_root=tmp_path,
        run_name="aggregate_only_forecast",
        aggregate_only=True,
        system_preset="SHERA_FLIGHT_3P",
        quiet=True,
    )
    run_root = Path(result["run_root"])
    assert (run_root / "forecast_results.csv").exists()
    rows = (run_root / "forecast_results.csv").read_text(encoding="utf-8")
    assert "1800" not in rows
    assert ",5," in rows or "\n5," in rows


def test_prior_draw_forecast_uses_case_specific_prior_sigma(tmp_path: Path):
    module = load_module()
    config_path = tmp_path / "campaign.json"
    write_config(config_path)
    enable_forecast(config_path)
    plan = module.build_campaign_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="forecast_prior",
        system_preset="SHERA_FLIGHT_3P",
    )
    base_case = plan.cases[0]
    labels = plan.layout.labels
    truth = plan.prior_truth.copy()
    for path in plan.summary_paths[base_case.case_name]:
        write_summary(path, labels, truth.copy(), truth)
    custom_case = module.BiasCase(
        case_name=base_case.case_name,
        theta_reference_offsets=base_case.theta_reference_offsets,
        case_origin="prior_draw",
        prior_sigma_by_label={label: 0.5 for label in labels},
        prior_draw_metadata={"draw_seed": 123, "draw_index": 0},
    )
    result = module.aggregate_case(
        plan=plan,
        case=custom_case,
        prior_source="summary_theta_ref",
    )
    assert result["forecast_results"]
    assert {
        row["prior_sigma_source"] for row in result["forecast_results"]
    } == {"prior_draw_config"}


def test_prior_draw_sigma_rule_expansion_and_fractional_calculation():
    module = load_module()
    labels = (
        "source.separation_as",
        "source.log_flux_total",
        "source.contrast",
        "optics.plate_scale_as_per_pix",
        "optics.primary.zernike_coeffs_nm[0]",
        "optics.secondary.zernike_coeffs_nm[0]",
    )
    truth = {
        "source.separation_as": 1.2e-3,
        "source.log_flux_total": 10.0,
        "source.contrast": 2.0e-2,
        "optics.plate_scale_as_per_pix": 3.5e-3,
        "optics.primary.zernike_coeffs_nm[0]": 0.0,
        "optics.secondary.zernike_coeffs_nm[0]": 0.0,
    }
    sigmas, meta = module._resolve_prior_draw_sigmas(
        labels=labels,
        truth_by_label=truth,
        sigmas_cfg={
            "source.separation_as": {"kind": "absolute", "sigma": 1.0e-5},
            "source.log_flux_total": {"kind": "absolute", "sigma": 2.0e-5},
            "source.contrast": {"kind": "fractional", "sigma": 1.0e-2},
            "optics.plate_scale_as_per_pix": {"kind": "fractional", "sigma": 1.0e-3},
            "optics.primary.zernike_coeffs_nm[*]": {"kind": "absolute", "sigma": 1.0e-1},
            "optics.secondary.zernike_coeffs_nm[*]": {"kind": "absolute", "sigma": 2.0e-1},
        },
    )
    assert sigmas["source.separation_as"] == pytest.approx(1.0e-5)
    assert sigmas["source.log_flux_total"] == pytest.approx(2.0e-5)
    assert sigmas["source.contrast"] == pytest.approx(abs(truth["source.contrast"]) * 1.0e-2)
    assert sigmas["optics.plate_scale_as_per_pix"] == pytest.approx(
        abs(truth["optics.plate_scale_as_per_pix"]) * 1.0e-3
    )
    assert sigmas["optics.primary.zernike_coeffs_nm[0]"] == pytest.approx(1.0e-1)
    assert sigmas["optics.secondary.zernike_coeffs_nm[0]"] == pytest.approx(2.0e-1)
    assert meta["optics.primary.zernike_coeffs_nm[0]"]["sigma_source_rule"] == "optics.primary.zernike_coeffs_nm[*]"


def test_prior_draw_generation_is_deterministic_and_offsets_match_z():
    module = load_module()
    labels = (
        "source.separation_as",
        "source.log_flux_total",
        "source.contrast",
    )
    truth = {
        "source.separation_as": 1.0e-3,
        "source.log_flux_total": 10.0,
        "source.contrast": 2.0e-2,
    }
    experiment_cfg = {
        "prior_draws": {
            "enabled": True,
            "n_cases": 2,
            "distribution": "normal",
            "center": "truth",
            "draw_seed": 123,
            "case_name_template": "prior_draw_{draw_index:03d}",
            "sigmas": {
                "source.separation_as": {"kind": "absolute", "sigma": 1.0e-5},
                "source.log_flux_total": {"kind": "absolute", "sigma": 1.0e-5},
                "source.contrast": {"kind": "fractional", "sigma": 1.0e-2},
            },
        }
    }
    cases_a, rows_a = module._generate_prior_draw_cases(
        experiment_cfg=experiment_cfg,
        labels=labels,
        truth_by_label=truth,
    )
    cases_b, rows_b = module._generate_prior_draw_cases(
        experiment_cfg=experiment_cfg,
        labels=labels,
        truth_by_label=truth,
    )
    assert [case.theta_reference_offsets for case in cases_a] == [
        case.theta_reference_offsets for case in cases_b
    ]
    assert rows_a == rows_b
    first_case = cases_a[0]
    first_rows = rows_a[first_case.case_name]
    for row in first_rows:
        assert row["theta_reference_offset"] == pytest.approx(
            row["draw_z"] * row["prior_sigma"]
        )


def test_aggregate_case_uses_case_specific_prior_sigma(tmp_path: Path):
    module = load_module()
    config_path = tmp_path / "campaign.json"
    write_config(config_path)
    plan = module.build_campaign_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="aggregate_prior",
        system_preset="SHERA_FLIGHT_3P",
    )
    base_case = plan.cases[0]
    labels = plan.layout.labels
    truth = plan.prior_truth.copy()
    theta_ref = truth.copy()
    for path in plan.summary_paths[base_case.case_name]:
        write_summary(path, labels, theta_ref, truth)

    custom_case = module.BiasCase(
        case_name=base_case.case_name,
        theta_reference_offsets=base_case.theta_reference_offsets,
        case_origin="prior_draw",
        prior_sigma_by_label={label: 0.5 for label in labels},
        prior_draw_metadata={"draw_seed": 123, "draw_index": 0},
    )
    result = module.aggregate_case(plan=plan, case=custom_case, prior_source="summary_theta_ref")
    update_summary = json.loads(
        (Path(result["case_root"]) / "observation_update_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert update_summary["prior"]["metadata"]["prior_sigma_source"] == "prior_draw_config"


def test_smoke_prescription_dry_run_plan_has_expected_forwarded_flags(tmp_path: Path):
    module = load_module()
    repo_root = Path(__file__).resolve().parents[2]
    config_path = (
        repo_root
        / "work"
        / "experiments"
        / "observation_bias_campaign_prior_draw_smoke.yaml"
    )
    plan = module.build_campaign_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="smoke_config",
        system_preset=None,
    )
    payload = module._plan_payload(plan)
    assert payload["dimension_estimate"]["n_theta"] == 20
    assert [case.case_name for case in plan.cases] == ["prior_draw_000"]
    command = plan.subblock_commands["prior_draw_000"][0]
    assert "--phi-ref" in command
    assert "recovered" in command
    assert "--trace-seed" in command
    assert "--render-seed" in command
    assert "--trace-jitter-x-sigma-as" in command
    assert "--trace-jitter-y-sigma-as" in command
    assert "--trace-jitter-pa-sigma-deg" in command
    assert "--reference-diagnostics-profile" in command
    assert "basic" in command
    assert "--exposure-time-s" in command
    assert "0.05" in command
    assert payload["case_generation"]["zero_bias_case_status"] == "disabled"
    assert 1800 in payload["forecast"]["n_subblocks_grid"]


def test_overnight_prescription_dry_run_plan_has_mask_quality_and_forecast(tmp_path: Path):
    module = load_module()
    repo_root = Path(__file__).resolve().parents[2]
    config_path = (
        repo_root
        / "work"
        / "experiments"
        / "observation_bias_campaign_prior_draw_overnight.yaml"
    )
    plan = module.build_campaign_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="overnight_config",
        system_preset=None,
    )
    payload = module._plan_payload(plan)
    assert len(plan.cases) == 5
    assert all(case.case_origin == "prior_draw" for case in plan.cases)
    assert len(plan.subblock_commands["prior_draw_000"]) == 5
    row = plan.subblock_plans["prior_draw_000"][0]
    assert row["n_frames"] == 20
    assert row["schur_frame_quality_policy"] == "mask"
    assert payload["forecast"]["n_subblocks_grid"][-1] == 1800
