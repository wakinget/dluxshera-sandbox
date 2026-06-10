from __future__ import annotations

import importlib.util
import json
import math
import sys
from types import SimpleNamespace
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


def write_summary(
    path: Path,
    labels: tuple[str, ...],
    theta_ref: np.ndarray,
    truth: np.ndarray,
    *,
    exposure_time_s: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(labels)
    info = np.eye(n, dtype=float)
    if n >= 3:
        common = np.zeros((n,), dtype=float)
        differential = np.zeros((n,), dtype=float)
        common[1:3] = np.array([1.0, 1.0]) / np.sqrt(2.0)
        differential[1:3] = np.array([1.0, -1.0]) / np.sqrt(2.0)
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
        "information_accounting": {
            "summary_information_scale": "summed_likelihood",
            "summary_frame_reduce": "sum",
            "summary_subblock_reduce": "sum",
        },
    }
    if exposure_time_s is not None:
        payload["metadata"] = {
            "prior_context": {
                "effective_store_values": {
                    "source.exposure_time_s": float(exposure_time_s),
                }
            },
            "system": {
                "resolved_config": {
                    "source": {"exposure_time_s": float(exposure_time_s)}
                }
            },
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


def test_trajectory_smear_dry_run_records_sidecars(tmp_path: Path):
    module = load_module()
    config_path = tmp_path / "campaign.json"
    write_config(config_path)
    airbus = tmp_path / "airbus.csv"
    airbus.write_text(
        "\n".join(
            [
                "0.0,0.0,0.0,0.0",
                "0.1,0.1,0.2,0.0",
                "0.2,0.2,0.4,0.0",
                "0.3,0.3,0.6,0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    subblocks = payload["experiment"]["subblocks"]
    subblocks["n_subblocks"] = 1
    subblocks["n_frames"] = 2
    subblocks["exposure_time_s"] = 0.05
    subblocks["trace_source"] = {
        "mode": "trajectory",
        "source": {"kind": "airbus_csv", "path": str(airbus), "sample_dt_s": 0.1},
        "window": {"start_s": 0.05, "n_subblocks": 1},
        "sampling": {
            "frame_dt_s": 0.05,
            "subblock_duration_s": 1.0,
            "n_frames_per_subblock": 2,
        },
        "output_keys": [
            "source.x_position_as",
            "source.y_position_as",
            "source.position_angle_deg",
        ],
    }
    subblocks["trajectory_processing"] = {"smear": {"enabled": True}}
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    plan = module.build_campaign_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="bias_smear",
        system_preset="SHERA_FLIGHT_3P",
    )

    row = next(iter(plan.subblock_plans.values()))[0]
    assert row["smear_enabled"] is True
    assert Path(row["smear_truth_csv"]).exists()
    assert Path(row["smear_model_csv"]).exists()
    assert Path(row["smear_provenance_json"]).exists()
    assert plan.trace_source_plan.summary["smear"]["enabled"] is True


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
    assert "--resource-time" not in command
    assert "--no-resource-time" not in command


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


def test_subblock_command_forwards_reference_early_stopping_flags(tmp_path: Path):
    module = load_module()
    command = module.build_subblock_command(
        case_root_parent=tmp_path / "subblocks",
        case_subblock_name="case/subblock_000",
        theta_labels=("source.separation_as",),
        offsets={},
        subblock_cfg={
            "reference_early_stopping_enabled": True,
            "reference_early_stopping_min_iter": 60,
            "reference_early_stopping_patience": 12,
            "reference_early_stopping_loss_rtol": 1.0e-8,
            "reference_early_stopping_loss_atol": 0.0,
            "reference_early_stopping_step_atol": 1.0e-10,
            "reference_early_stopping_grad_norm_atol": 1.0e-8,
        },
    )

    assert "--reference-early-stopping" in command
    expected_pairs = {
        "--reference-early-stopping-min-iter": "60",
        "--reference-early-stopping-patience": "12",
        "--reference-early-stopping-loss-rtol": "1e-08",
        "--reference-early-stopping-loss-atol": "0.0",
        "--reference-early-stopping-step-atol": "1e-10",
        "--reference-early-stopping-grad-norm-atol": "1e-08",
    }
    for flag, value in expected_pairs.items():
        index = command.index(flag)
        assert command[index + 1] == value


@pytest.mark.parametrize("enabled_value", [False, None])
def test_subblock_command_omits_reference_early_stopping_flag_when_disabled(
    tmp_path: Path,
    enabled_value: bool | None,
):
    module = load_module()
    subblock_cfg: dict[str, Any] = {}
    if enabled_value is not None:
        subblock_cfg["reference_early_stopping_enabled"] = enabled_value
    command = module.build_subblock_command(
        case_root_parent=tmp_path / "subblocks",
        case_subblock_name="case/subblock_000",
        theta_labels=("source.separation_as",),
        offsets={},
        subblock_cfg=subblock_cfg,
    )

    assert "--reference-early-stopping" not in command


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
            "reference_early_stopping_enabled": True,
            "reference_early_stopping_min_iter": 60,
            "reference_early_stopping_patience": 12,
            "reference_early_stopping_loss_rtol": 1.0e-8,
            "reference_early_stopping_loss_atol": 0.0,
            "reference_early_stopping_step_atol": 1.0e-10,
            "reference_early_stopping_grad_norm_atol": 1.0e-8,
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
    assert row["reference_early_stopping_enabled"] is True
    assert row["reference_early_stopping_min_iter"] == 60
    assert row["reference_early_stopping_patience"] == 12
    assert row["reference_early_stopping_loss_rtol"] == pytest.approx(1.0e-8)
    assert row["reference_early_stopping_loss_atol"] == pytest.approx(0.0)
    assert row["reference_early_stopping_step_atol"] == pytest.approx(1.0e-10)
    assert row["reference_early_stopping_grad_norm_atol"] == pytest.approx(1.0e-8)
    payload = module._plan_payload(plan)
    assert payload["subblock_command_options"]["exposure_time_s"] == pytest.approx(0.05)
    assert "--exposure-time-s" in payload["subblock_command_options"]["forwarded_flags"]
    assert "--reference-early-stopping" in payload["subblock_command_options"]["forwarded_flags"]


def test_campaign_truth_context_uses_subblock_exposure_for_log_flux(tmp_path: Path):
    module = load_module()
    config_path = tmp_path / "campaign.json"
    write_config(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["experiment"]["subblocks"]["exposure_time_s"] = 0.05
    payload["experiment"]["observation_theta"]["source"]["log_flux_total"] = True
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    plan = module.build_campaign_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="flux_exposure",
        system_preset="SHERA_FLIGHT_3P",
    )

    assert plan.config["system"]["source"]["exposure_time_s"] == pytest.approx(0.05)
    assert plan.layout_metadata["resolved_system"]["source"]["exposure_time_s"] == pytest.approx(0.05)
    command = plan.subblock_commands["matched_pair"][0]
    assert command[command.index("--exposure-time-s") + 1] == "0.05"
    log_flux_index = plan.layout.labels.index("source.log_flux_total")
    truth_log_flux = float(plan.prior_truth[log_flux_index])
    assert truth_log_flux < 8.0
    assert truth_log_flux + math.log10(1800.0 / 0.05) > 11.0


def test_iterative_log_flux_offsets_use_posterior_table_truth_values():
    module = load_module()
    labels = ("source.log_flux_total",)
    stale_campaign_truth = {"source.log_flux_total": 11.571953773498535}
    posterior_rows = {
        "source.log_flux_total": {
            "truth_value": "7.01565177202808",
            "posterior_mean": "7.0133428774766475",
        }
    }

    posterior_truth = module._posterior_truth_by_label(
        labels=labels,
        posterior_rows=posterior_rows,
        fallback_truth=stale_campaign_truth,
    )
    offsets, status = module.posterior_offsets_from_rows(
        labels=labels,
        posterior_rows_by_label=posterior_rows,
        truth_by_label=posterior_truth,
    )

    assert posterior_truth["source.log_flux_total"] == pytest.approx(7.01565177202808)
    assert offsets["source.log_flux_total"] == pytest.approx(-0.002308894551432239)
    assert status["source.log_flux_total"] == "ok"


def test_iterative_exposure_context_diagnostics_and_guard(tmp_path: Path):
    module = load_module()
    config_path = tmp_path / "campaign.json"
    write_config(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["experiment"]["subblocks"]["exposure_time_s"] = 0.05
    payload["experiment"]["observation_theta"]["source"]["log_flux_total"] = True
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    plan = module.build_campaign_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="flux_guard",
        system_preset="SHERA_FLIGHT_3P",
    )
    case = plan.cases[0]
    labels = plan.layout.labels
    truth = plan.prior_truth.copy()
    for path in plan.summary_paths[case.case_name]:
        write_summary(path, labels, truth.copy(), truth, exposure_time_s=0.05)

    diagnostics = module.iterative_context_diagnostics(
        plan=plan,
        summary_paths=plan.summary_paths[case.case_name],
    )
    assert diagnostics["source.exposure_time_s"]["consistent"] is True
    assert diagnostics["source.exposure_time_s"]["campaign"] == pytest.approx(0.05)
    assert diagnostics["source.exposure_time_s"]["summary_values"] == [0.05, 0.05]
    assert diagnostics["source.exposure_time_s"]["truth_or_prior_store"] == pytest.approx(0.05)
    module.validate_iterative_log_flux_exposure_context(
        plan=plan,
        context_diagnostics=diagnostics,
    )

    bad_system = json.loads(json.dumps(plan.config["system"]))
    bad_system["source"]["exposure_time_s"] = 1800.0
    bad_metadata = json.loads(json.dumps(plan.layout_metadata))
    bad_metadata["resolved_system"]["source"]["exposure_time_s"] = 1800.0
    bad_plan = module.replace(
        plan,
        config={**plan.config, "system": bad_system},
        layout_metadata=bad_metadata,
    )
    bad_diagnostics = module.iterative_context_diagnostics(
        plan=bad_plan,
        summary_paths=plan.summary_paths[case.case_name],
    )
    assert bad_diagnostics["source.exposure_time_s"]["consistent"] is False
    assert bad_diagnostics["source.log_flux_total"][
        "expected_log10_offset_if_truth_or_prior_vs_campaign"
    ] == pytest.approx(math.log10(1800.0 / 0.05))
    with pytest.raises(RuntimeError, match="consistent exposure context"):
        module.validate_iterative_log_flux_exposure_context(
            plan=bad_plan,
            context_diagnostics=bad_diagnostics,
        )


def test_imported_binary_iterative_failure_bundle_has_exposure_ratio_signature():
    bundle = (
        Path(__file__).resolve().parents[2]
        / "Results"
        / "hpc_imports"
        / "binary_iterative_cluster_validation_v1_failure_context_20260609_141255"
    )
    update_path = (
        bundle
        / "cases"
        / "binary_iter_validation_draw_000"
        / "windows"
        / "window_000"
        / "iterative_reference_update.json"
    )
    if not update_path.exists():
        pytest.skip("Imported binary iterative failure bundle is not available.")

    update = json.loads(update_path.read_text(encoding="utf-8"))
    observed = float(update["posterior_offsets"]["source.log_flux_total"])
    expected_context_delta = -math.log10(1800.0 / 0.05)

    assert observed == pytest.approx(expected_context_delta, abs=3.0e-3)
    assert update["current_offsets"]["source.log_flux_total"] == pytest.approx(
        0.0,
        abs=1.0e-6,
    )


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


def test_no_resource_time_executes_without_external_time_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module()
    config_path = tmp_path / "campaign.json"
    write_config(config_path)
    plan = module.build_campaign_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="no_resource_time",
        system_preset="SHERA_FLIGHT_3P",
    )
    calls: list[str | bool | None] = []

    def fake_run_subprocess_with_diagnostics(**kwargs: Any) -> Any:
        calls.append(kwargs["resource_time"])
        Path(kwargs["stdout_log"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["stdout_log"]).write_text("", encoding="utf-8")
        Path(kwargs["stderr_log"]).write_text("", encoding="utf-8")
        return SimpleNamespace(
            return_code=0,
            failure_class=None,
            failure_hint=None,
            stdout_log=str(kwargs["stdout_log"]),
            stderr_log=str(kwargs["stderr_log"]),
            last_stderr_line=None,
            resource_time={"resource_time_mode_effective": "disabled"},
            stderr_tail=[],
        )

    monkeypatch.setattr(
        module,
        "run_subprocess_with_diagnostics",
        fake_run_subprocess_with_diagnostics,
    )
    module.execute_subblocks(
        plan,
        resume=False,
        max_workers=1,
        fail_fast=True,
        quiet=True,
        resource_time="disabled",
    )
    assert calls
    assert set(calls) == {"disabled"}


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

def test_truth_realization_disabled_returns_empty_overrides():
    module = load_module()
    labels = (
        "source.separation_as",
        "optics.primary.zernike_coeffs_nm[0]",
        "optics.secondary.zernike_coeffs_nm[0]",
    )
    base_truth = {label: 0.0 for label in labels}
    result = module._realize_campaign_truth(
        experiment_cfg={"truth_realization": {"enabled": False}},
        labels=labels,
        base_truth_by_label=base_truth,
    )
    assert result.truth_overrides_by_label == {}
    assert result.rows == []
    assert result.summary["enabled"] is False


def test_truth_realization_draws_deterministically_by_mirror_sigma():
    module = load_module()
    labels = (
        "source.separation_as",
        "optics.primary.zernike_coeffs_nm[0]",
        "optics.primary.zernike_coeffs_nm[1]",
        "optics.secondary.zernike_coeffs_nm[0]",
    )
    base_truth = {label: 0.0 for label in labels}
    cfg = {
        "truth_realization": {
            "enabled": True,
            "seed": 7,
            "mode": "zernike_per_coefficient_sigma",
            "zernikes": {
                "primary": {"enabled": True, "indices": "from_observation_theta", "mean_nm": 0.0, "sigma_nm": 5.0},
                "secondary": {"enabled": True, "indices": "from_observation_theta", "mean_nm": 0.0, "sigma_nm": 2.0},
            },
        }
    }
    first = module._realize_campaign_truth(experiment_cfg=cfg, labels=labels, base_truth_by_label=base_truth)
    second = module._realize_campaign_truth(experiment_cfg=cfg, labels=labels, base_truth_by_label=base_truth)
    assert first.truth_overrides_by_label == second.truth_overrides_by_label
    assert "source.separation_as" not in first.truth_overrides_by_label
    rows = {row["theta_label"]: row for row in first.rows}
    assert rows["optics.primary.zernike_coeffs_nm[0]"]["sigma_nm"] == pytest.approx(5.0)
    assert rows["optics.secondary.zernike_coeffs_nm[0]"]["sigma_nm"] == pytest.approx(2.0)


def test_prior_draw_rows_use_realized_truth_values(tmp_path: Path):
    module = load_module()
    config_path = tmp_path / "campaign_truth_realized.json"
    write_config(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["experiment"]["bias_cases"] = []
    payload["experiment"]["prior_draws"] = {
        "enabled": True,
        "n_cases": 1,
        "center": "truth",
        "distribution": "normal",
        "draw_seed": 11,
        "case_name_template": "draw_{draw_index:03d}",
        "sigmas": {
            "source.separation_as": {"kind": "absolute", "sigma": 1.0e-5},
            "optics.primary.zernike_coeffs_nm[*]": {"kind": "absolute", "sigma": 0.1},
            "optics.secondary.zernike_coeffs_nm[*]": {"kind": "absolute", "sigma": 0.1},
        },
    }
    payload["experiment"]["truth_realization"] = {
        "enabled": True,
        "seed": 20260521,
        "mode": "zernike_per_coefficient_sigma",
        "zernikes": {
            "primary": {"enabled": True, "indices": "from_observation_theta", "mean_nm": 0.0, "sigma_nm": 5.0},
            "secondary": {"enabled": True, "indices": "from_observation_theta", "mean_nm": 0.0, "sigma_nm": 2.0},
        },
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    plan = module.build_campaign_plan(config_path=config_path, results_root=tmp_path, run_name="truth_realized", system_preset="SHERA_FLIGHT_3P")
    row = plan.prior_draw_rows_by_case["draw_000"][1]
    assert row["truth_value"] == pytest.approx(float(plan.prior_truth[1]))
    assert row["theta_reference_offset"] == pytest.approx(row["reference_value"] - row["truth_value"])


def test_bias_parser_supports_resource_time_flags() -> None:
    module = load_module()
    parser = module._build_parser()
    assert parser.parse_args(["--no-resource-time"]).resource_time == "disabled"
    assert parser.parse_args(["--resource-time"]).resource_time == "enabled"
    assert parser.parse_args(["--resource-time", "auto"]).resource_time == "auto"
    parsed = parser.parse_args(["--no-resource-time", "--dry-run"])
    assert parsed.resource_time == "disabled"
    assert parsed.dry_run is True


def test_aggregate_only_rejects_mismatched_existing_case_set(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "campaign.json"
    write_config(config_path)
    run_root = tmp_path / "unit_campaign"
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "campaign_plan.json").write_text(
        json.dumps({"summary_paths": {"different_case": ["dummy.json"]}}),
        encoding="utf-8",
    )
    args = module._build_parser().parse_args(
        ["--config", str(config_path), "--run-name", "unit_campaign", "--aggregate-only"]
    )
    with pytest.raises(ValueError, match="stored plan validation failed"):
        module.run_observation_bias_campaign(
            config_path=config_path,
            results_root=tmp_path,
            run_name="unit_campaign",
            aggregate_only=True,
            args=args,
        )


def test_observation_bias_plan_includes_high_order_wfe_templates_and_provenance(tmp_path: Path):
    module = load_module()
    config_path = tmp_path / "campaign_high_order.json"
    write_config(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["experiment"]["high_order_wfe"] = {
        "enabled": True,
        "truth": {
            "enabled": True,
            "mirrors": ["primary", "secondary"],
            "mode": "synthetic",
            "npix": 16,
            "amplitude_nm_rms": 1.0,
            "pairing": "independent",
        },
        "inference": {
            "enabled": True,
            "mode": "knowledge_error",
            "knowledge_error": {"enabled": True, "amplitude_nm_rms": 0.3},
        },
        "artifacts": {"write_maps": False, "write_summary_json": True},
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    plan = module.build_campaign_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="unit_campaign_howfe",
        system_preset="SHERA_FLIGHT_3P",
    )
    plan_payload = module._plan_payload(plan)
    first_command = plan.subblock_commands["zero_bias_full_zernike"][0]

    assert plan_payload["high_order_wfe"]["provenance"]["enabled"] is True
    assert plan.subblock_plans["zero_bias_full_zernike"][0]["high_order_wfe_enabled"] is True
    assert "--render-template" in first_command
    assert str(plan.run_root / "templates" / "render_template.json") in first_command
    render_template = json.loads(
        (plan.run_root / "templates" / "render_template.json").read_text(encoding="utf-8")
    )
    inference_template = json.loads(
        (plan.run_root / "templates" / "inference_template.json").read_text(encoding="utf-8")
    )
    assert render_template["system"]["optics"]["high_order_wfe"]["enabled"] is True
    assert inference_template["system"] != render_template["system"]


def test_aggregate_only_requires_stored_model_split_artifacts(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "campaign.json"
    write_config(config_path)

    module.run_observation_bias_campaign(
        config_path=config_path,
        results_root=tmp_path,
        run_name="unit_campaign",
        dry_run=True,
        aggregate_only=False,
        resume=False,
        max_workers=1,
        quiet=True,
        resource_time="disabled",
    )
    run_root = tmp_path / "unit_campaign"
    missing = run_root / "model_split" / "model_split.json"
    missing.unlink()

    with pytest.raises(FileNotFoundError, match="stored model-split artifacts"):
        module.run_observation_bias_campaign(
            config_path=config_path,
            results_root=tmp_path,
            run_name="unit_campaign",
            dry_run=False,
            aggregate_only=True,
            resume=False,
            max_workers=1,
            quiet=True,
            resource_time="disabled",
        )
