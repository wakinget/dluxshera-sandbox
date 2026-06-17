from __future__ import annotations

import importlib.util
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "scripts"
    / "run_single_star_calibration_demo.py"
)


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "run_single_star_calibration_demo",
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
            "kind": "single_star_calibration_demo",
            "run_name": "unit_cal",
            "calibration_source": {
                "mode": "alpha_cen_a_placeholder",
                "source_kind": "single_star",
                "x_position_as": 0.0,
                "y_position_as": 0.0,
                "position_angle_deg": 0.0,
                "n_lambda": 3,
            },
            "subblocks": {
                "n_subblocks": 1,
                "n_frames": 2,
                "noise": "disabled",
                "phi_ref": "truth_when_available",
                "schur_curvature_method": "structured_independent_frames",
                "max_dense_dim": 40,
                "schur_damping": 1.0e-8,
                "exposure_time_s": 0.05,
            },
            "seeding": {
                "seed_policy": "different_jitter_different_noise",
                "base_seed": 42,
            },
            "observation_theta": {
                "source": {"log_flux_total": True},
                "optics": {
                    "plate_scale_as_per_pix": True,
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
            "prior": {
                "sigma": {
                    "source.log_flux_total": {"kind": "absolute", "sigma": 1.0e-5},
                    "optics.plate_scale_as_per_pix": {
                        "kind": "fractional",
                        "sigma": 1.0e-5,
                    },
                    "optics.primary.zernike_coeffs_nm[*]": {
                        "kind": "absolute",
                        "sigma": 1.0,
                    },
                    "optics.secondary.zernike_coeffs_nm[*]": {
                        "kind": "absolute",
                        "sigma": 1.0,
                    },
                }
            },
            "case_generation": {
                "mode": "prior_draw",
                "n_cases": 1,
                "seed": 123,
                "draw_scale": 0.5,
                "include_zero_bias_case": True,
            },
            "forecast": {"enabled": False, "subblock_duration_s": 1.0},
            "eigenbasis": {"enabled": False},
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize(
    "update_mode",
    ["physical_full", "eigen_full", "eigen_damped", "eigen_truncated"],
)
def test_update_policy_parser_accepts_supported_modes(update_mode: str):
    module = load_module()
    policy = module._resolve_update_policy(
        {
            "update_policy": {
                "update_mode": update_mode,
                "update_gain": 0.5,
                "eigenbasis": {
                    "basis_source": "posterior_precision",
                    "gate_source": "accumulated_information",
                    "eig_floor_rel": 1.0e-9,
                    "min_kept_modes": 1,
                },
            }
        }
    )

    assert policy.update_mode == update_mode
    assert policy.update_gain == 0.5
    assert policy.min_kept_modes == 1


def test_update_policy_parser_defaults_to_physical_full():
    module = load_module()
    policy = module._resolve_update_policy({})

    assert policy.update_mode == "physical_full"
    assert policy.update_gain == 1.0


def write_truth_realization_config(
    path: Path,
    *,
    primary_sigma_nm: float,
    secondary_sigma_nm: float,
    n_cases: int = 1,
    draw_scale: float = 1.0,
) -> None:
    write_config(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["experiment"]["case_generation"].update(
        {
            "mode": "prior_draw",
            "n_cases": int(n_cases),
            "seed": 123,
            "draw_scale": float(draw_scale),
            "include_zero_bias_case": True,
        }
    )
    payload["experiment"]["prior"]["sigma"].update(
        {
            "optics.primary.zernike_coeffs_nm[*]": {
                "kind": "absolute",
                "sigma": 0.01,
                "unit": "nm",
            },
            "optics.secondary.zernike_coeffs_nm[*]": {
                "kind": "absolute",
                "sigma": 0.01,
                "unit": "nm",
            },
        }
    )
    payload["experiment"]["truth_realization"] = {
        "enabled": True,
        "seed": 20260727,
        "mode": "zernike_per_coefficient_sigma",
        "combine_with_system_truth": False,
        "zernikes": {
            "primary": {
                "enabled": True,
                "indices": "from_observation_theta",
                "mean_nm": 0.0,
                "sigma_nm": float(primary_sigma_nm),
            },
            "secondary": {
                "enabled": True,
                "indices": "from_observation_theta",
                "mean_nm": 0.0,
                "sigma_nm": float(secondary_sigma_nm),
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_summary(path: Path, labels: tuple[str, ...], theta_ref: np.ndarray, truth: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    info = np.eye(len(labels), dtype=float) * 4.0
    score = info @ (theta_ref - truth)
    payload = {
        "schema_version": "image_backed_subblock_summary.v1",
        "subblock_id": "synthetic",
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
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_default_plan_is_single_star_and_excludes_binary_terms(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "config.json"
    write_config(config_path)

    plan = module.build_calibration_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="unit_cal",
        system_preset="SHERA_FLIGHT_3P",
    )

    labels = plan.layout.labels
    assert plan.system_cfg["source"]["kind"] == "single_star"
    assert plan.system_cfg["source"]["x_position_as"] == 0.0
    assert plan.system_cfg["source"]["y_position_as"] == 0.0
    assert "source.log_flux_total" in labels
    assert "optics.plate_scale_as_per_pix" in labels
    assert "optics.primary.zernike_coeffs_nm[0]" in labels
    assert "optics.secondary.zernike_coeffs_nm[0]" in labels
    assert "source.separation_as" not in labels
    assert "source.contrast" not in labels
    payload = module._plan_payload(plan)
    assert payload["source_kind"] == "single_star"
    assert payload["local_eliminated_keys"] == list(module.ACTIVE_FRAME_KEYS)
    assert payload["active_frame_keys"] == [
        "source.x_position_as",
        "source.y_position_as",
    ]
    assert "source.position_angle_deg" not in payload["active_frame_keys"]
    assert payload["dimension_estimate"]["frame_phi_dim"] == 2
    assert payload["dimension_estimate"]["n_phi"] == 4
    template_inference = (
        plan.run_root / "templates" / "inference_template.json"
    )
    inference_payload = json.loads(template_inference.read_text(encoding="utf-8"))
    assert inference_payload["experiment"]["inference"]["active"]["frame_keys"] == [
        "source.x_position_as",
        "source.y_position_as",
    ]
    assert "source.position_angle_deg" not in inference_payload["experiment"]["inference"]["active"]["frame_keys"]


def test_single_star_forward_spec_does_not_require_binary_fields(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "config.json"
    write_config(config_path)
    plan = module.build_calibration_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="unit_cal",
        system_preset="SHERA_FLIGHT_3P",
    )
    spec = module.compose_forward_spec(plan.system_cfg)
    store = module._refreshed_store_from_system_cfg(plan.system_cfg)
    assert store.get("source.log_flux_total") == plan.system_cfg["source"]["log_flux_total"]
    assert "source.separation_as" not in spec
    assert "source.contrast" not in spec


def test_single_star_scalar_truth_matches_refreshed_store(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "config.json"
    write_config(config_path)
    plan = module.build_calibration_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="unit_cal",
        system_preset="SHERA_FLIGHT_3P",
    )
    store = module._refreshed_store_from_system_cfg(plan.system_cfg)
    truth_by_label = {
        label: float(plan.truth_vector[index])
        for index, label in enumerate(plan.layout.labels)
    }

    assert truth_by_label["source.log_flux_total"] == pytest.approx(
        float(store.get("source.log_flux_total")),
        abs=0.0,
    )
    assert truth_by_label["optics.plate_scale_as_per_pix"] == pytest.approx(
        float(store.get("optics.plate_scale_as_per_pix")),
        abs=0.0,
    )

    pixel_pitch = float(store.get("detector.pixel_pitch_m"))
    m1_f = float(store.get("optics.m1_focal_length_m"))
    m2_f = float(store.get("optics.m2_focal_length_m"))
    sep = float(store.get("optics.m1_m2_separation_m"))
    focal_length = 1.0 / ((1.0 / m1_f) + (1.0 / m2_f) - sep / (m1_f * m2_f))
    expected_plate_scale = pixel_pitch / focal_length * module.ARCSEC_PER_RAD
    assert truth_by_label["optics.plate_scale_as_per_pix"] == pytest.approx(
        expected_plate_scale,
        rel=0.0,
        abs=1.0e-15,
    )


def test_single_star_log_flux_matches_rendered_source_flux(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "config.json"
    write_config(config_path)
    plan = module.build_calibration_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="unit_cal",
        system_preset="SHERA_FLIGHT_3P",
    )
    spec = module.compose_forward_spec(plan.system_cfg)
    store = module.ParameterStore.from_spec_defaults(spec).refresh_derived(spec)
    source = module.build_single_star_source(store, cfg=plan.system_cfg)

    expected_flux = 10.0 ** float(store.get("source.log_flux_total"))
    assert float(np.asarray(source.flux)) == pytest.approx(expected_flux)
    assert float(np.sum(np.asarray(source.weights))) == pytest.approx(1.0)


def test_prior_draw_cases_are_reproducible_and_zero_bias_is_zero(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "config.json"
    write_config(config_path)
    plan_a = module.build_calibration_plan(
        config_path=config_path,
        results_root=tmp_path / "a",
        run_name="unit_cal",
        system_preset="SHERA_FLIGHT_3P",
    )
    plan_b = module.build_calibration_plan(
        config_path=config_path,
        results_root=tmp_path / "b",
        run_name="unit_cal",
        system_preset="SHERA_FLIGHT_3P",
    )
    zero = next(case for case in plan_a.cases if case.case_origin == "zero_bias")
    draw_a = next(case for case in plan_a.cases if case.case_origin == "prior_draw")
    draw_b = next(case for case in plan_b.cases if case.case_origin == "prior_draw")
    assert zero.theta_reference_offsets == {}
    assert draw_a.theta_reference_offsets == draw_b.theta_reference_offsets
    assert any(
        abs(value) > 0.0
        for label, value in draw_a.theta_reference_offsets.items()
        if "zernike_coeffs_nm" in label
    )


def test_truth_realization_config_builds_nonzero_render_truth(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "truth.json"
    write_truth_realization_config(
        config_path,
        primary_sigma_nm=0.3,
        secondary_sigma_nm=0.0,
    )
    plan = module.build_calibration_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="truth_case",
        system_preset="SHERA_FLIGHT_3P",
    )
    assert plan.truth_realization["enabled"] is True
    rows = plan.truth_realization_rows
    primary = [row for row in rows if row["mirror"] == "primary"]
    secondary = [row for row in rows if row["mirror"] == "secondary"]
    assert primary
    assert any(abs(float(row["truth_offset"])) > 0.0 for row in primary)
    assert secondary
    assert all(float(row["truth_offset"]) == pytest.approx(0.0) for row in secondary)
    truth_by_label = {
        label: float(plan.truth_vector[index])
        for index, label in enumerate(plan.layout.labels)
    }
    assert truth_by_label["optics.primary.zernike_coeffs_nm[0]"] == pytest.approx(
        float(primary[0]["realized_truth_value"])
    )
    assert plan.system_cfg["optics"]["primary"]["zernike_coeffs_nm"][0] == pytest.approx(
        truth_by_label["optics.primary.zernike_coeffs_nm[0]"]
    )
    render_payload = json.loads(
        (plan.run_root / "templates" / "render_template.json").read_text(encoding="utf-8")
    )
    assert render_payload["system"]["optics"]["primary"]["zernike_coeffs_nm"][0] == pytest.approx(
        truth_by_label["optics.primary.zernike_coeffs_nm[0]"]
    )


def test_truth_realization_zero_sigma_control_writes_zero_offsets(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "zero_truth.json"
    write_truth_realization_config(
        config_path,
        primary_sigma_nm=0.0,
        secondary_sigma_nm=0.0,
    )
    plan = module.build_calibration_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="zero_truth",
        system_preset="SHERA_FLIGHT_3P",
    )
    assert plan.truth_realization_rows
    assert all(
        float(row["truth_offset"]) == pytest.approx(0.0)
        for row in plan.truth_realization_rows
    )


def test_truth_realization_prior_draws_are_centered_on_realized_truth(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "truth_prior.json"
    write_truth_realization_config(
        config_path,
        primary_sigma_nm=0.3,
        secondary_sigma_nm=0.0,
        draw_scale=1.0,
    )
    plan = module.build_calibration_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="truth_prior",
        system_preset="SHERA_FLIGHT_3P",
    )
    truth_by_label = {
        label: float(plan.truth_vector[index])
        for index, label in enumerate(plan.layout.labels)
    }
    zero = next(case for case in plan.cases if case.case_origin == "zero_bias")
    assert zero.theta_reference_offsets == {}
    primary_label = "optics.primary.zernike_coeffs_nm[0]"
    primary_draw = next(
        row for row in plan.prior_draw_rows if row["theta_label"] == primary_label
    )
    assert float(primary_draw["truth_value"]) == pytest.approx(truth_by_label[primary_label])
    assert float(primary_draw["reference_value"]) == pytest.approx(
        truth_by_label[primary_label] + float(primary_draw["theta_reference_offset"])
    )
    assert float(primary_draw["truth_value"]) != pytest.approx(0.0)
    draw_case = next(case for case in plan.cases if case.case_origin == "prior_draw")
    command = " ".join(plan.subblock_commands[draw_case.case_name][0])
    assert f"--theta-reference-offset {primary_label}=" in command
    zero_command = " ".join(plan.subblock_commands[zero.case_name][0])
    assert "--theta-reference-offset" not in zero_command
    assert "--trace-jitter-pa-sigma-deg" not in zero_command


def test_truth_realization_dry_run_writes_artifacts(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "truth_dry.json"
    write_truth_realization_config(
        config_path,
        primary_sigma_nm=0.3,
        secondary_sigma_nm=0.0,
    )
    payload = module.main(
        [
            "--config",
            str(config_path),
            "--results-root",
            str(tmp_path),
            "--run-name",
            "truth_dry",
            "--dry-run",
            "--quiet",
        ]
    )
    run_root = Path(payload["run_root"])
    for name in [
        "campaign_plan.json",
        "resolved_config.json",
        "calibration_cases.csv",
        "prior_draws.csv",
        "subblock_plan.csv",
        "truth_realization_by_label.csv",
    ]:
        assert (run_root / name).exists()
    truth_rows = _read_csv(run_root / "truth_realization_by_label.csv")
    assert any(row["mirror"] == "primary" for row in truth_rows)
    assert any(abs(float(row["truth_offset"])) > 0.0 for row in truth_rows)


def test_command_construction_uses_single_star_schur_summary(tmp_path: Path) -> None:
    module = load_module()
    template_paths = {
        "trace": tmp_path / "trace.json",
        "render": tmp_path / "render.json",
        "inference": tmp_path / "inference.json",
    }
    command = module.build_subblock_command(
        case_root_parent=tmp_path / "subblocks",
        case_subblock_name="case/subblock_000000",
        template_paths=template_paths,
        theta_labels=(
            "source.log_flux_total",
            "optics.plate_scale_as_per_pix",
            "optics.primary.zernike_coeffs_nm[0]",
            "optics.secondary.zernike_coeffs_nm[0]",
        ),
        layout_metadata={
            "primary_zernike_indices": [0],
            "secondary_zernike_indices": [0],
        },
        offsets={"optics.primary.zernike_coeffs_nm[0]": 1.0},
        subblock_cfg={
            "n_frames": 3,
            "noise": "disabled",
            "phi_ref": "recovered",
            "schur_curvature_method": "structured_independent_frames",
            "max_dense_dim": 40,
            "schur_damping": 1.0e-8,
        },
        trace_seed=1,
        noise_seed=2,
    )

    joined = " ".join(command)
    assert "run_obs_subblock_study.py" in joined
    assert "--mode schur_summary" in joined
    assert "--enable-zernikes" in command
    assert "--zernike-indices" in command
    assert "source.separation_as" not in joined
    assert "source.contrast" not in joined
    assert "--phi-ref recovered" in joined
    assert "--schur-curvature-method structured_independent_frames" in joined
    assert "--use-render-variance" not in joined


def test_subblock_command_uses_render_variance_for_noisy_single_star_by_default(tmp_path: Path) -> None:
    module = load_module()
    template_paths = {
        "trace": tmp_path / "trace.json",
        "render": tmp_path / "render.json",
        "inference": tmp_path / "inference.json",
    }
    command = module.build_subblock_command(
        case_root_parent=tmp_path / "subblocks",
        case_subblock_name="case/subblock_000000",
        template_paths=template_paths,
        theta_labels=("source.log_flux_total",),
        layout_metadata={"primary_zernike_indices": [], "secondary_zernike_indices": []},
        offsets={},
        subblock_cfg={"noise": "enabled"},
        trace_seed=1,
        noise_seed=2,
    )

    assert "--use-render-variance" in command


def test_cli_accepts_memory_diagnostics_flag() -> None:
    module = load_module()
    args = module._build_parser().parse_args(["--memory-diagnostics"])
    assert bool(args.memory_diagnostics) is True


def test_cli_accepts_no_resource_time_dry_run() -> None:
    module = load_module()
    args = module._build_parser().parse_args(["--no-resource-time", "--dry-run"])
    assert args.resource_time is False
    assert args.dry_run is True


def test_plan_forwards_memory_diagnostics_to_subblock_commands(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "config.json"
    write_config(config_path)
    args = module._build_parser().parse_args(["--memory-diagnostics"])
    plan = module.build_calibration_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="unit_cal",
        system_preset="SHERA_FLIGHT_3P",
        args=args,
    )
    command = plan.subblock_commands["zero_bias_case"][0]
    assert "--memory-diagnostics" in command


def test_cli_override_schur_method_is_honored_in_plan(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "config.json"
    write_config(config_path)
    args = module._build_parser().parse_args(["--schur-curvature-method", "dense"])
    plan = module.build_calibration_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="unit_cal",
        system_preset="SHERA_FLIGHT_3P",
        args=args,
    )
    row = plan.subblock_rows[0]
    assert row["schur_curvature_method_requested"] == "dense"
    assert row["schur_route_source"] == "user_request"
    assert "--schur-curvature-method" in plan.subblock_commands["zero_bias_case"][0]
    assert "dense" in plan.subblock_commands["zero_bias_case"][0]


def test_dry_run_writes_plan_without_executing_subprocesses(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "config.json"
    write_config(config_path)
    payload = module.main(
        [
            "--config",
            str(config_path),
            "--results-root",
            str(tmp_path),
            "--run-name",
            "dry_run_case",
            "--dry-run",
        ]
    )
    assert payload["run_root"].endswith("dry_run_case")
    run_root = Path(payload["run_root"])
    assert (run_root / "subblock_plan.csv").exists()
    assert not (run_root / "subblock_status.csv").exists()


def test_trajectory_smear_dry_run_records_sidecars(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "config.json"
    write_config(config_path)
    airbus = tmp_path / "airbus.csv"
    airbus.write_text(
        "0.0,0.0,0.0,0.0\n0.1,0.1,0.2,0.0\n0.2,0.2,0.4,0.0\n",
        encoding="utf-8",
    )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    subblocks = payload["experiment"]["subblocks"]
    subblocks["trace_source"] = {
        "mode": "trajectory",
        "source": {"kind": "airbus_csv", "path": str(airbus), "sample_dt_s": 0.1},
        "window": {"start_s": 0.05, "n_subblocks": 1},
        "sampling": {
            "frame_dt_s": 0.05,
            "subblock_duration_s": 1.0,
            "n_frames_per_subblock": 2,
        },
        "output_keys": ["source.x_position_as", "source.y_position_as"],
    }
    subblocks["trajectory_processing"] = {"smear": {"enabled": True}}
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    plan = module.build_calibration_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="single_smear",
        system_preset="SHERA_FLIGHT_3P",
    )

    row = plan.subblock_rows[0]
    assert row["smear_enabled"] is True
    assert Path(row["smear_truth_csv"]).exists()
    assert Path(row["smear_model_csv"]).exists()
    assert Path(row["smear_provenance_json"]).exists()
    assert plan.trace_source_plan.summary["smear"]["enabled"] is True


def test_failed_subblock_records_status_before_fail_fast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module()
    config_path = tmp_path / "config.json"
    write_config(config_path)
    plan = module.build_calibration_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="failed_status",
        system_preset="SHERA_FLIGHT_3P",
    )

    def _fail_subprocess(**kwargs):
        kwargs["stderr_log"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["stderr_log"].write_text("Traceback\nAttributeError: missing reference flag\n")
        kwargs["stdout_log"].write_text("")
        kwargs["diagnostics_json"].write_text("{}")
        return SimpleNamespace(
            return_code=1,
            elapsed_seconds=0.1,
            stdout_log=str(kwargs["stdout_log"]),
            stderr_log=str(kwargs["stderr_log"]),
            last_stderr_line="AttributeError: missing reference flag",
            failure_class="nonzero_exit",
            failure_hint="Process exited with nonzero status.",
            memory_sampler={"peak_total_rss_mb": None},
            resource_time={"maximum_resident_set_mb": 156.0},
        )

    monkeypatch.setattr(module, "run_subprocess_with_diagnostics", _fail_subprocess)

    with pytest.raises(RuntimeError, match="child stderr tail"):
        module.execute_subblocks(
            plan,
            resume=False,
            max_workers=1,
            fail_fast=True,
            quiet=True,
            memory_diagnostics=False,
            resource_time=False,
        )

    rows = list(module.csv.DictReader((plan.run_root / "subblock_status.csv").open()))
    assert rows[0]["status"] == "failed"
    assert rows[0]["return_code"] == "1"
    assert rows[0]["last_stderr_line"] == "AttributeError: missing reference flag"
    assert (plan.run_root / "memory_failure_summary.csv").exists()
    assert (plan.run_root / "progress.json").exists()


def test_trace_policy_allows_inert_pa_truth_without_solving(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "config.json"
    write_config(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["experiment"]["subblocks"]["trace_jitter"] = {
        "x_sigma_as": 1.0e-3,
        "y_sigma_as": 1.0e-3,
        "pa_sigma_deg": 1.0e-4,
        "pa_mode": "inert_diagnostic",
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    plan = module.build_calibration_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="unit_cal",
        system_preset="SHERA_FLIGHT_3P",
    )
    trace_template = json.loads(
        (plan.run_root / "templates" / "trace_template.json").read_text(encoding="utf-8")
    )
    assert "source.position_angle_deg" in trace_template["experiment"]["trace"]["varying_keys"]
    inference_template = json.loads(
        (plan.run_root / "templates" / "inference_template.json").read_text(encoding="utf-8")
    )
    assert "source.position_angle_deg" not in inference_template["experiment"]["inference"]["active"]["frame_keys"]
    plan_payload = module._plan_payload(plan)
    assert plan_payload["single_star_pa_policy"]["status"] == "inactive"
    assert "source.position_angle_deg" in plan_payload["inactive_truth_keys"]


def test_dimension_estimate_tracks_xy_only_phi_layout(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "config.json"
    write_config(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["experiment"]["subblocks"]["n_frames"] = 3
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    plan_3 = module.build_calibration_plan(
        config_path=config_path,
        results_root=tmp_path / "three",
        run_name="unit_cal",
        system_preset="SHERA_FLIGHT_3P",
    )
    plan_payload_3 = module._plan_payload(plan_3)
    assert plan_payload_3["dimension_estimate"]["frame_phi_dim"] == 2
    assert plan_payload_3["dimension_estimate"]["n_phi"] == 6

    default_plan = module.build_calibration_plan(
        config_path=None,
        results_root=tmp_path / "default",
        run_name="unit_default",
        system_preset="SHERA_FLIGHT_3P",
    )
    default_payload = module._plan_payload(default_plan)
    assert default_payload["n_frames"] == 20
    assert default_payload["dimension_estimate"]["frame_phi_dim"] == 2
    assert default_payload["dimension_estimate"]["n_phi"] == 40


def test_truth_comparison_active_key_filter_excludes_pa() -> None:
    module = load_module()
    columns = [
        "source.x_position_as_truth",
        "source.x_position_as_recovered",
        "source.x_position_as_residual",
        "source.y_position_as_truth",
        "source.y_position_as_recovered",
        "source.y_position_as_residual",
        "source.position_angle_deg_truth",
        "source.position_angle_deg_recovered",
        "source.position_angle_deg_residual",
    ]
    selected = module.select_active_truth_comparison_keys(columns)
    assert selected == ["source.x_position_as", "source.y_position_as"]


def test_aggregate_math_does_not_assume_binary_labels(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "config.json"
    write_config(config_path)
    plan = module.build_calibration_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="unit_cal",
        system_preset="SHERA_FLIGHT_3P",
    )
    case = next(case for case in plan.cases if case.case_origin == "prior_draw")
    labels = plan.layout.labels
    truth = np.asarray(plan.truth_vector, dtype=float)
    theta_ref = truth + np.asarray(
        [case.theta_reference_offsets.get(label, 0.0) for label in labels],
        dtype=float,
    )
    write_summary(plan.summary_paths[case.case_name][0], labels, theta_ref, truth)

    result = module.aggregate_case(plan, case)

    posterior_csv = Path(result["posterior_by_parameter_csv"])
    rows = posterior_csv.read_text(encoding="utf-8").splitlines()
    assert posterior_csv.exists()
    assert "correction_fraction" in rows[0]
    assert "posterior_error_over_sigma" in rows[0]
    assert "source.separation_as" not in posterior_csv.read_text(encoding="utf-8")
    assert (plan.run_root / "cases" / case.case_name / "posterior_history.csv").exists()
    assert (
        plan.run_root
        / "cases"
        / case.case_name
        / "eigen_update_diagnostics.json"
    ).exists()
    assert (
        plan.run_root / "cases" / case.case_name / "eigen_update_modes.csv"
    ).exists()


def test_zero_bias_posterior_metrics_mark_fractions_undefined(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "config.json"
    write_config(config_path)
    plan = module.build_calibration_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="unit_cal",
        system_preset="SHERA_FLIGHT_3P",
    )
    zero = next(case for case in plan.cases if case.case_origin == "zero_bias")
    labels = plan.layout.labels
    truth = np.asarray(plan.truth_vector, dtype=float)
    write_summary(plan.summary_paths[zero.case_name][0], labels, truth, truth)

    result = module.aggregate_case(plan, zero)
    rows = _read_csv(Path(result["posterior_by_parameter_csv"]))
    scalar_rows = [
        row
        for row in rows
        if row["theta_label"]
        in {"source.log_flux_total", "optics.plate_scale_as_per_pix"}
    ]
    assert scalar_rows
    for row in scalar_rows:
        assert abs(float(row["injected_bias"])) == pytest.approx(0.0)
        assert np.isnan(float(row["correction_fraction"]))
        assert np.isnan(float(row["residual_fraction"]))
        assert row["moves_toward_truth"] == ""
    assert (
        plan.run_root
        / "cases"
        / zero.case_name
        / "single_star_consistency_diagnostics.json"
    ).exists()


def test_subblock_command_forwards_reference_early_stopping_flags(tmp_path: Path) -> None:
    module = load_module()
    template_paths = {
        "trace": tmp_path / "trace.json",
        "render": tmp_path / "render.json",
        "inference": tmp_path / "inference.json",
    }
    command = module.build_subblock_command(
        case_root_parent=tmp_path / "subblocks",
        case_subblock_name="case/subblock_000000",
        template_paths=template_paths,
        theta_labels=("source.log_flux_total",),
        layout_metadata={"primary_zernike_indices": [], "secondary_zernike_indices": []},
        offsets={},
        subblock_cfg={
            "reference_early_stopping_enabled": True,
            "reference_early_stopping_min_iter": 10,
            "reference_early_stopping_patience": 5,
            "reference_early_stopping_loss_rtol": 1.0e-6,
            "reference_early_stopping_loss_atol": 1.0e-9,
            "reference_early_stopping_step_atol": 1.0e-10,
            "reference_early_stopping_grad_norm_atol": 1.0e-8,
        },
        trace_seed=1,
        noise_seed=2,
    )

    joined = " ".join(command)
    assert "--reference-early-stopping" in command
    assert "--reference-early-stopping-min-iter 10" in joined
    assert "--reference-early-stopping-patience 5" in joined
    assert "--reference-early-stopping-loss-rtol 1e-06" in joined
    assert "--reference-early-stopping-loss-atol 1e-09" in joined
    assert "--reference-early-stopping-step-atol 1e-10" in joined
    assert "--reference-early-stopping-grad-norm-atol 1e-08" in joined


def test_subblock_command_omits_reference_early_stopping_flags_by_default(tmp_path: Path) -> None:
    module = load_module()
    template_paths = {
        "trace": tmp_path / "trace.json",
        "render": tmp_path / "render.json",
        "inference": tmp_path / "inference.json",
    }
    command = module.build_subblock_command(
        case_root_parent=tmp_path / "subblocks",
        case_subblock_name="case/subblock_000000",
        template_paths=template_paths,
        theta_labels=("source.log_flux_total",),
        layout_metadata={"primary_zernike_indices": [], "secondary_zernike_indices": []},
        offsets={},
        subblock_cfg={},
        trace_seed=1,
        noise_seed=2,
    )

    joined = " ".join(command)
    assert "--reference-early-stopping" not in joined
    assert "--reference-early-stopping-min-iter" not in joined


def test_aggregate_only_rejects_mismatched_existing_case_set(tmp_path: Path) -> None:
    module = load_module()
    plan = module.build_calibration_plan(
        config_path=None,
        results_root=tmp_path,
        run_name="agg_mismatch",
        system_preset=None,
        args=module._build_parser().parse_args(["--run-name", "agg_mismatch", "--aggregate-only"]),
    )
    plan.run_root.mkdir(parents=True, exist_ok=True)
    (plan.run_root / "campaign_plan.json").write_text(
        json.dumps({"summary_paths": {"different_case": ["dummy.json"]}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="different case set"):
        module.main(["--results-root", str(tmp_path), "--run-name", "agg_mismatch", "--aggregate-only"])


def test_single_star_plan_includes_high_order_wfe_templates_and_provenance(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "config_high_order.json"
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

    plan = module.build_calibration_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="unit_cal_howfe",
        system_preset="SHERA_FLIGHT_3P",
    )
    plan_payload = module._plan_payload(plan)

    assert plan_payload["high_order_wfe"]["provenance"]["enabled"] is True
    assert plan.subblock_rows[0]["high_order_wfe_enabled"] is True
    render_template = json.loads(
        (plan.run_root / "templates" / "render_template.json").read_text(encoding="utf-8")
    )
    inference_template = json.loads(
        (plan.run_root / "templates" / "inference_template.json").read_text(encoding="utf-8")
    )
    assert render_template["system"]["optics"]["high_order_wfe"]["enabled"] is True
    assert inference_template["system"]["optics"]["high_order_wfe"]["enabled"] is True
    assert inference_template["system"] != render_template["system"]
