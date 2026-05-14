from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "scripts"
    / "run_scalar_bias_capture_study.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "run_scalar_bias_capture_study_script",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config(module, tmp_path: Path):
    return module.ScalarBiasStudyConfig(
        results_root=tmp_path,
        run_name="unit",
        parameters=("optics.plate_scale_as_per_pix",),
        theta_keys=module.DEFAULT_PARAMETERS,
        bias_ppm_grid=(-10.0, 0.0, 10.0),
        n_frames=20,
        noise="disabled",
        trace_template=None,
        seed=123,
        trace_seed_policy="same_trace_all_cases",
        max_workers=1,
        dry_run=True,
    )


def test_parse_ppm_grid_valid_and_rejects_duplicates_and_malformed():
    module = _load_script_module()

    assert module.parse_ppm_grid("-10,0,10") == (-10.0, 0.0, 10.0)

    with pytest.raises(ValueError, match="Duplicate PPM"):
        module.parse_ppm_grid("-10,0,-10")
    with pytest.raises(ValueError, match="Malformed PPM"):
        module.parse_ppm_grid("-10,nope,10")


def test_parse_parameter_list_accepts_defaults_and_rejects_bad_inputs():
    module = _load_script_module()

    assert module.parse_parameter_list(None) == module.DEFAULT_PARAMETERS
    with pytest.raises(ValueError, match="duplicates"):
        module.parse_parameter_list("source.separation_as,source.separation_as")
    with pytest.raises(ValueError, match="Unsupported"):
        module.parse_parameter_list("source.separation_as,optics.not_a_key")


def test_compute_reference_offset_uses_physical_ppm_and_log_flux_raw_flux_ppm():
    module = _load_script_module()

    reference, offset, expression = module.compute_reference_offset(
        "source.separation_as",
        2.0,
        10.0,
    )
    assert reference == pytest.approx(2.0 * (1.0 + 10.0e-6))
    assert offset == pytest.approx(2.0 * 10.0e-6)
    assert expression == "truth_value * ppm * 1e-6"

    reference, offset, expression = module.compute_reference_offset(
        "source.log_flux_total",
        5.0,
        -30.0,
    )
    assert offset == pytest.approx(math.log10(1.0 - 30.0e-6))
    assert reference == pytest.approx(5.0 + math.log10(1.0 - 30.0e-6))
    assert expression == "log10(1 + ppm * 1e-6)"

    with pytest.raises(ValueError, match="greater than -1e6"):
        module.compute_reference_offset("source.log_flux_total", 5.0, -1_000_000.0)


def test_case_naming_helpers_are_deterministic():
    module = _load_script_module()

    assert module.slugify_parameter("optics.plate_scale_as_per_pix") == "optics_plate_scale_as_per_pix"
    assert module.format_signed_ppm(10.0) == "p10"
    assert module.format_signed_ppm(-10.0) == "m10"
    assert module.format_signed_ppm(0.0) == "z0"


def test_build_case_plan_and_command_include_required_schur_flags(tmp_path):
    module = _load_script_module()
    config = _config(module, tmp_path)
    truth_values = {
        "source.separation_as": 1.0,
        "source.log_flux_total": 5.0,
        "source.contrast": 0.2,
        "optics.plate_scale_as_per_pix": 0.01,
    }

    cases = module.build_case_plan(config, truth_values)

    assert len(cases) == 3
    assert str(cases[0].command_path).startswith(str(config.run_root / "commands"))
    assert str(cases[0].stdout_log).startswith(str(config.run_root / "logs"))
    assert str(cases[0].stderr_log).startswith(str(config.run_root / "logs"))
    command = list(cases[0].command)
    assert "--phi-ref" in command
    assert command[command.index("--phi-ref") + 1] == "recovered"
    assert "--reference-schedule-kind" in command
    assert "linear_warmup" in command
    assert "--reference-diagnostics-profile" in command
    assert "basic" in command
    assert "--theta-reference-offset" in command
    offset_arg = command[command.index("--theta-reference-offset") + 1]
    assert offset_arg.startswith("optics.plate_scale_as_per_pix=")
    assert float(offset_arg.split("=", 1)[1]) == pytest.approx(-1.0e-7)
    assert "--theta-reference-offset" not in cases[1].command
    assert cases[2].case_name == "optics_plate_scale_as_per_pix_ppm_p10_20f_noiseless"


def test_zero_bias_reference_n_iter_only_applies_to_zero_case(tmp_path):
    module = _load_script_module()
    config = module.ScalarBiasStudyConfig(
        **{
            **_config(module, tmp_path).__dict__,
            "reference_n_iter": None,
            "zero_bias_reference_n_iter": 200,
        }
    )
    truth_values = {
        "source.separation_as": 1.0,
        "source.log_flux_total": 5.0,
        "source.contrast": 0.2,
        "optics.plate_scale_as_per_pix": 0.01,
    }
    cases = module.build_case_plan(config, truth_values)
    assert "--reference-n-iter" not in cases[0].command
    assert "--reference-n-iter" in cases[1].command
    assert cases[1].command[cases[1].command.index("--reference-n-iter") + 1] == "200"
    assert "--reference-n-iter" not in cases[2].command


def test_write_command_file_contains_shell_safe_command_and_offset(tmp_path):
    module = _load_script_module()
    config = _config(module, tmp_path)
    truth_values = {
        "source.separation_as": 1.0,
        "source.log_flux_total": 5.0,
        "source.contrast": 0.2,
        "optics.plate_scale_as_per_pix": 0.01,
    }
    spec = module.build_case_plan(config, truth_values)[0]
    module.write_command_file(spec)
    text = spec.command_path.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in text
    assert "run_obs_subblock_study.py" in text
    assert "--theta-reference-offset" in text
    assert "optics.plate_scale_as_per_pix=" in text


def test_run_case_writes_stdout_stderr_and_tail_lines(tmp_path):
    module = _load_script_module()
    summary = tmp_path / "summary.json"
    command = (
        sys.executable,
        "-c",
        (
            "from pathlib import Path; import sys; "
            f"Path({str(summary)!r}).parent.mkdir(parents=True, exist_ok=True); "
            f"Path({str(summary)!r}).write_text('{{}}', encoding='utf-8'); "
            "print('hello stdout'); print('hello stderr', file=sys.stderr)"
        ),
    )
    spec = module.ScalarBiasCaseSpec(
        case_id=1,
        case_name="fake",
        biased_parameter="source.separation_as",
        bias_ppm=0.0,
        fractional_bias=0.0,
        truth_value=1.0,
        reference_value=1.0,
        theta_reference_offset=0.0,
        theta_reference_offset_expression="truth_value * ppm * 1e-6",
        n_frames=1,
        noise="disabled",
        trace_seed=0,
        render_seed=0,
        theta_keys=module.DEFAULT_PARAMETERS,
        phi_ref="recovered",
        reference_n_iter=None,
        results_root=tmp_path,
        case_root=tmp_path / "case",
        summary_json_expected=summary,
        command=command,
        command_path=tmp_path / "commands" / "fake.sh",
        stdout_log=tmp_path / "logs" / "fake.stdout.log",
        stderr_log=tmp_path / "logs" / "fake.stderr.log",
    )
    result = module.run_case(spec, quiet=True, resume=False)
    assert result.status == "ok"
    assert result.return_code == 0
    assert result.stdout_log.exists()
    assert result.stderr_log.exists()
    assert result.last_stdout_line == "hello stdout"
    assert result.last_stderr_line == "hello stderr"


def test_run_case_failure_records_log_pointers(tmp_path):
    module = _load_script_module()
    spec = module.ScalarBiasCaseSpec(
        case_id=1,
        case_name="fakefail",
        biased_parameter="source.separation_as",
        bias_ppm=0.0,
        fractional_bias=0.0,
        truth_value=1.0,
        reference_value=1.0,
        theta_reference_offset=0.0,
        theta_reference_offset_expression="truth_value * ppm * 1e-6",
        n_frames=1,
        noise="disabled",
        trace_seed=0,
        render_seed=0,
        theta_keys=module.DEFAULT_PARAMETERS,
        phi_ref="recovered",
        reference_n_iter=None,
        results_root=tmp_path,
        case_root=tmp_path / "case",
        summary_json_expected=tmp_path / "missing" / "subblock_summary.json",
        command=(sys.executable, "-c", "import sys; print('boom', file=sys.stderr); sys.exit(7)"),
        command_path=tmp_path / "commands" / "fakefail.sh",
        stdout_log=tmp_path / "logs" / "fakefail.stdout.log",
        stderr_log=tmp_path / "logs" / "fakefail.stderr.log",
    )
    result = module.run_case(spec, quiet=True, resume=False)
    assert result.status == "failed"
    assert result.return_code == 7
    assert "stderr_log=" in result.failure_reason
    assert result.last_stderr_line == "boom"


def test_registration_absorption_metrics_from_truth_comparison_csv(tmp_path):
    module = _load_script_module()
    spec = module.ScalarBiasCaseSpec(
        case_id=1,
        case_name="case",
        biased_parameter="source.separation_as",
        bias_ppm=10.0,
        fractional_bias=10.0e-6,
        truth_value=1.0,
        reference_value=1.00001,
        theta_reference_offset=1.0e-5,
        theta_reference_offset_expression="truth_value * ppm * 1e-6",
        n_frames=20,
        noise="disabled",
        trace_seed=0,
        render_seed=0,
        theta_keys=module.DEFAULT_PARAMETERS,
        phi_ref="recovered",
        reference_n_iter=None,
        results_root=tmp_path,
        case_root=tmp_path / "case",
        summary_json_expected=tmp_path / "case" / "study" / "schur_summary" / "subblock_summary.json",
        command=("python3", "study.py"),
        command_path=tmp_path / "commands" / "case.sh",
        stdout_log=tmp_path / "logs" / "case.stdout.log",
        stderr_log=tmp_path / "logs" / "case.stderr.log",
    )
    comparison = tmp_path / "truth_comparison.csv"
    with comparison.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source.x_position_as_residual",
                "source.y_position_as_residual",
                "source.position_angle_deg_residual",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source.x_position_as_residual": "1.0",
                "source.y_position_as_residual": "2.0",
                "source.position_angle_deg_residual": "3.0",
            }
        )
        writer.writerow(
            {
                "source.x_position_as_residual": "-1.0",
                "source.y_position_as_residual": "4.0",
                "source.position_angle_deg_residual": "-3.0",
            }
        )

    theta_context = {
        "planned_truth_value": 1.0,
        "planned_reference_value": 1.00001,
        "planned_theta_reference_offset": 1.0e-5,
        "actual_reference_value": 1.00002,
        "effective_truth_value": 1.00001,
        "reference_value_source": "summary_theta_ref",
        "truth_value_source": "actual_reference_minus_injected_offset",
        "planned_minus_actual_reference": -1.0e-5,
        "planned_actual_reference_abs_diff": 1.0e-5,
        "planned_actual_reference_rel_diff": 1.0e-5,
        "planned_actual_reference_mismatch_warning": True,
    }
    row = module.compute_registration_absorption_metrics(
        spec,
        {
            "subblock_summary_json": spec.summary_json_expected,
            "truth_comparison_csv": comparison,
            "recovered_trace_csv": None,
            "manifest_json": None,
        },
        {"schur_summary": {"frame_quality": {"bad_frame_count": 1, "good_frame_count": 19}}},
        theta_context,
    )

    assert row["mean_dx_as"] == pytest.approx(0.0)
    assert row["std_dx_as"] == pytest.approx(1.0)
    assert row["rms_dy_as"] == pytest.approx(math.sqrt((4.0 + 16.0) / 2.0))
    assert row["max_abs_dpa_deg"] == pytest.approx(3.0)
    assert row["frame_quality_bad_frame_count"] == 1
    assert "combined_registration_rms" not in row


def test_nuisance_columns_do_not_include_combined_registration_rms():
    module = _load_script_module()
    assert "combined_registration_rms" not in module.NUISANCE_COLUMNS


def test_theta_reference_consistency_passed_helper_supports_multiple_shapes():
    module = _load_script_module()
    assert module._theta_reference_consistency_passed(
        {"metadata": {"theta_reference_consistency_passed": True}}
    ) is True
    assert module._theta_reference_consistency_passed(
        {"metadata": {"theta_reference_consistency": {"passed": False}}}
    ) is False
    assert module._theta_reference_consistency_passed(
        {"theta_reference_consistency": {"passed": True}}
    ) is True
    assert module._theta_reference_consistency_passed({}) == ""


def test_correction_and_science_leakage_metrics_are_computed():
    module = _load_script_module()
    summary_payload = {
        "theta_labels": list(module.DEFAULT_PARAMETERS),
        "theta_ref": [1.1, 5.0, 0.2, 0.01],
        "summary_diagnostics": {
            "score_norm": 2.0,
            "rank_estimate": 4,
            "min_eigenvalue": 0.1,
            "condition_number": 10.0,
        },
        "metadata": {"theta_reference_consistency_passed": True},
    }
    spec = SimpleNamespace(
        case_id=1,
        case_name="case",
        biased_parameter="source.separation_as",
        bias_ppm=100000.0,
        truth_value=1.0,
        reference_value=1.1,
        theta_reference_offset=0.1,
    )
    update = SimpleNamespace(posterior=SimpleNamespace(metadata={"solve_method": "solve"}))
    posterior_mean = np.array([1.02, 5.0, 0.2, 0.01])
    posterior_sigma = np.array([0.01, 0.1, 0.1, 0.001])

    theta_context = {
        "planned_truth_value": 1.0,
        "planned_reference_value": 1.1,
        "planned_theta_reference_offset": 0.1,
        "actual_reference_value": 1.1,
        "effective_truth_value": 1.0,
        "reference_value_source": "summary_theta_ref",
        "truth_value_source": "actual_reference_minus_injected_offset",
        "planned_minus_actual_reference": 0.0,
        "planned_actual_reference_abs_diff": 0.0,
        "planned_actual_reference_rel_diff": 0.0,
        "planned_actual_reference_mismatch_warning": False,
        "actual_reference_by_label": {
            "source.separation_as": 1.1,
            "source.log_flux_total": 5.0,
            "source.contrast": 0.2,
            "optics.plate_scale_as_per_pix": 0.01,
        },
        "effective_truth_by_label": {
            "source.separation_as": 1.0,
            "source.log_flux_total": 5.0,
            "source.contrast": 0.2,
            "optics.plate_scale_as_per_pix": 0.01,
        },
    }
    correction = module.compute_correction_response_metrics(
        spec,
        summary_payload,
        posterior_mean,
        posterior_sigma,
        update,
        {"surrogate_validation_csv": None},
        theta_context,
    )
    science = module.compute_science_leakage_metrics(
        spec,
        summary_payload,
        posterior_mean,
        posterior_sigma,
        theta_context,
    )

    assert correction["posterior_shift_biased_parameter"] == pytest.approx(-0.08)
    assert correction["correction_fraction_biased_parameter"] == pytest.approx(0.8)
    assert correction["residual_fraction_biased_parameter"] == pytest.approx(0.2)
    assert correction["moves_biased_parameter_toward_truth"] is True
    assert science["posterior_separation_error_microas"] == pytest.approx(20000.0)
    assert science["separation_correction_fraction_if_biased"] == pytest.approx(0.8)


def test_effective_truth_reference_use_actual_summary_theta_ref_for_biased_parameter():
    module = _load_script_module()
    spec = SimpleNamespace(
        biased_parameter="optics.plate_scale_as_per_pix",
        theta_reference_offset=1.0e-6,
        truth_value=0.2,
        reference_value=0.200001,
    )
    summary_payload = {
        "theta_labels": list(module.DEFAULT_PARAMETERS),
        "theta_ref": [9.0, 7.0, 3.0, 0.1232069],
    }
    ctx = module._effective_theta_context(spec, summary_payload)
    assert ctx["actual_reference_value"] == pytest.approx(0.1232069)
    assert ctx["effective_truth_value"] == pytest.approx(0.1232059)
    assert ctx["planned_minus_actual_reference"] == pytest.approx(0.200001 - 0.1232069)


def test_non_biased_parameter_effective_truth_equals_actual_reference():
    module = _load_script_module()
    spec = SimpleNamespace(
        biased_parameter="optics.plate_scale_as_per_pix",
        theta_reference_offset=1.0e-6,
        truth_value=0.2,
        reference_value=0.200001,
    )
    summary_payload = {
        "theta_labels": list(module.DEFAULT_PARAMETERS),
        "theta_ref": [9.5, 7.1, 3.1, 0.1232069],
    }
    ctx = module._effective_theta_context(spec, summary_payload)
    assert ctx["effective_truth_by_label"]["source.separation_as"] == pytest.approx(9.5)
    assert ctx["actual_reference_by_label"]["source.separation_as"] == pytest.approx(9.5)


def test_science_leakage_uses_effective_truth_and_removes_false_offset():
    module = _load_script_module()
    summary_payload = {
        "theta_labels": list(module.DEFAULT_PARAMETERS),
        "theta_ref": [9.7650001, 7.0, 3.0, 0.1232069],
    }
    spec = SimpleNamespace(
        case_id=1,
        case_name="case",
        biased_parameter="optics.plate_scale_as_per_pix",
        bias_ppm=10.0,
        truth_value=0.1232059,
        reference_value=0.1232069,
        theta_reference_offset=1.0e-6,
    )
    posterior_mean = np.array([9.7650001, 7.0, 3.0, 0.12320695])
    posterior_sigma = np.array([1.0e-3, 0.1, 0.1, 1.0e-6])
    ctx = {
        "actual_reference_by_label": {
            "source.separation_as": 9.7650001,
            "source.log_flux_total": 7.0,
            "source.contrast": 3.0,
            "optics.plate_scale_as_per_pix": 0.1232069,
        },
        "effective_truth_by_label": {
            "source.separation_as": 9.7650001,
            "source.log_flux_total": 7.0,
            "source.contrast": 3.0,
            "optics.plate_scale_as_per_pix": 0.1232059,
        },
        "planned_truth_value": 0.1232059,
    }
    science = module.compute_science_leakage_metrics(
        spec,
        summary_payload,
        posterior_mean,
        posterior_sigma,
        ctx,
    )
    assert science["posterior_separation_error_as"] == pytest.approx(0.0)
    assert science["posterior_separation_error_microas"] == pytest.approx(0.0)


def test_chi2_extraction_prefers_manifest_metrics(tmp_path):
    module = _load_script_module()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "metrics": {
                    "chi2": {
                        "initial_model": {"block_reduced_chi2": 12.5},
                        "final_model": {"block_reduced_chi2": 0.75},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    spec = module.ScalarBiasCaseSpec(
        case_id=1,
        case_name="case",
        biased_parameter="source.separation_as",
        bias_ppm=10.0,
        fractional_bias=1.0e-5,
        truth_value=1.0,
        reference_value=1.00001,
        theta_reference_offset=1.0e-5,
        theta_reference_offset_expression="truth_value * ppm * 1e-6",
        n_frames=1,
        noise="disabled",
        trace_seed=0,
        render_seed=0,
        theta_keys=module.DEFAULT_PARAMETERS,
        phi_ref="recovered",
        reference_n_iter=None,
        results_root=tmp_path,
        case_root=tmp_path / "case",
        summary_json_expected=tmp_path / "case" / "study" / "schur_summary" / "subblock_summary.json",
        command=(sys.executable, "-c", "print('x')"),
        command_path=tmp_path / "commands" / "case.sh",
        stdout_log=tmp_path / "logs" / "case.stdout.log",
        stderr_log=tmp_path / "logs" / "case.stderr.log",
    )
    row = module.compute_registration_absorption_metrics(
        spec,
        {
            "subblock_summary_json": spec.summary_json_expected,
            "truth_comparison_csv": None,
            "recovered_trace_csv": None,
            "truth_trace_csv": None,
            "manifest_json": manifest,
        },
        {"schur_summary": {"frame_quality": {}}},
        {
            "planned_truth_value": 1.0,
            "planned_reference_value": 1.00001,
            "planned_theta_reference_offset": 1.0e-5,
            "actual_reference_value": 1.00002,
            "effective_truth_value": 1.00001,
            "reference_value_source": "summary_theta_ref",
            "truth_value_source": "actual_reference_minus_injected_offset",
            "planned_minus_actual_reference": -1.0e-5,
            "planned_actual_reference_abs_diff": 1.0e-5,
            "planned_actual_reference_rel_diff": 1.0e-5,
            "planned_actual_reference_mismatch_warning": True,
        },
    )
    assert row["initial_block_reduced_chi2"] == pytest.approx(12.5)
    assert row["final_block_reduced_chi2"] == pytest.approx(0.75)


def test_parser_no_longer_exposes_study_script_option():
    module = _load_script_module()
    parser = module._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--study-script", "x.py", "--run-name", "x"])


def test_write_plan_and_aggregate_json_stable_artifacts(tmp_path):
    module = _load_script_module()
    config = _config(module, tmp_path)
    truth_values = {
        "source.separation_as": 1.0,
        "source.log_flux_total": 5.0,
        "source.contrast": 0.2,
        "optics.plate_scale_as_per_pix": 0.01,
    }
    cases = module.build_case_plan(config, truth_values)

    module.write_plan(config, cases, truth_values)
    summary = module.aggregate_cases(config, cases[:1], truth_values)

    assert (config.run_root / "run_plan.csv").exists()
    assert (config.run_root / "run_plan.json").exists()
    assert (config.run_root / "case_status.csv").exists()
    assert summary["counts"]["planned"] == 1
    assert summary["counts"]["aggregate_failed"] == 1
    assert "nuisance_absorption_sensitivity_csv" in summary["artifacts"]
    payload = json.loads((config.run_root / "aggregate_summary.json").read_text(encoding="utf-8"))
    assert payload["config"]["run_name"] == "unit"
    assert (config.run_root / "commands").exists()
    assert (config.run_root / "logs").exists()
