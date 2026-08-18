from __future__ import annotations

import importlib.util
import csv
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from dluxshera.utils.iterative_campaigns import apply_physical_reference_update


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "scripts"
    / "run_observation_bias_campaign.py"
)


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "run_observation_bias_campaign_iterative_test",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_iterative_config(path: Path) -> None:
    payload = {
        "experiment": {
            "kind": "observation_bias_campaign",
            "run_name": "binary_iterative_unit",
            "subblocks": {
                "n_frames": 3,
                "noise": "disabled",
                "phi_ref": "truth_when_available",
                "schur_curvature_method": "auto",
                "max_dense_dim": 40,
                "schur_damping": 1.0e-8,
                "summary_information_scale": "summed_likelihood",
                "trace_source": {"mode": "iid_jitter"},
            },
            "iterative": {
                "enabled": True,
                "windows_per_draw": 2,
                "subblocks_per_window": 1,
                "update_gain": 1.0,
                "update_mode": "physical_full",
                "carry_prior_mean_with_reference": True,
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
                        "enabled": False,
                        "indices": "from_system",
                        "include": [],
                        "exclude": [],
                    },
                },
            },
            "bias_cases": [
                {
                    "case_name": "sep_bias",
                    "theta_reference_offsets": {"source.separation_as": 2.0e-6},
                }
            ],
            "case_generation": {"include_implicit_zero_bias": False},
            "prior_draws": {"enabled": False},
            "forecast": {"enabled": False, "plots": False},
            "eigenbasis": {"enabled": False},
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_apply_physical_reference_update_ignores_missing_and_nonfinite() -> None:
    current = {"a": 2.0, "b": 5.0}
    posterior = {
        "a": {"theta_label": "a", "posterior_mean": "13.0"},
        "b": {"theta_label": "b", "posterior_mean": "nan"},
        "c": {"theta_label": "c", "posterior_mean": "99.0"},
    }
    truth = {"a": 10.0, "b": 100.0}

    updated = apply_physical_reference_update(
        current_offsets=current,
        posterior_rows_by_label=posterior,
        truth_by_label=truth,
        update_gain=0.5,
    )

    assert updated["a"] == pytest.approx(2.5)
    assert updated["b"] == pytest.approx(5.0)
    assert "c" not in updated


def test_binary_iterative_dry_run_writes_stable_window_contract(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "iterative.json"
    write_iterative_config(config_path)

    payload = module.run_observation_bias_campaign(
        config_path=config_path,
        results_root=tmp_path,
        run_name="iterative_plan",
        dry_run=True,
        system_preset="SHERA_FLIGHT_3P",
        quiet=True,
    )
    run_root = tmp_path / "iterative_plan"

    assert payload["iterative"]["enabled"] is True
    assert payload["iterative"]["windows_per_draw"] == 2
    assert (run_root / "campaign_plan.json").exists()
    assert (run_root / "iterative_plan.csv").exists()
    assert (run_root / "expected_outputs.csv").exists()
    expected = payload["expected_outputs"]
    assert len(expected) == 2
    assert {row["window_index"] for row in expected} == {0, 1}
    assert all("case_posterior_path" in row for row in expected)
    assert all("summary_path" in row for row in expected)
    assert all("window_case_name" in row for row in expected)
    assert all("window_summary_path" in row for row in expected)
    assert all("iterative_reference_update_path" in row for row in expected)
    assert all("realized_command_path" in row for row in expected)
    first_iter = payload["iterative_plan"][0]
    assert first_iter["theta_reference_offsets_window0_json"]
    assert first_iter["trace_source_mode"] == "iid_jitter"
    assert first_iter["realized_after_reference_update"] is True


def test_binary_iterative_aggregate_only_reports_missing_outputs(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "iterative.json"
    write_iterative_config(config_path)
    module.run_observation_bias_campaign(
        config_path=config_path,
        results_root=tmp_path,
        run_name="missing_plan",
        dry_run=True,
        system_preset="SHERA_FLIGHT_3P",
        quiet=True,
    )

    status = module.run_observation_bias_campaign(
        config_path=config_path,
        results_root=tmp_path,
        run_name="missing_plan",
        aggregate_only=True,
        system_preset="SHERA_FLIGHT_3P",
        quiet=True,
    )

    analysis = tmp_path / "missing_plan" / "analysis"
    assert status["missing_summaries"] == 2
    assert status["missing_posterior_tables"] == 2
    assert status["missing_outputs_by_kind"]["iterative_reference_update"] == 2
    assert (analysis / "missing_outputs.csv").exists()
    assert (analysis / "output_inventory.csv").exists()
    assert (analysis / "aggregate_status.json").exists()


def test_binary_iterative_accepts_eigen_damped_update_mode(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "iterative.json"
    write_iterative_config(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["experiment"]["iterative"]["update_mode"] = "eigen_damped"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    result = module.run_observation_bias_campaign(
        config_path=config_path,
        results_root=tmp_path,
        run_name="eigen_damped_mode",
        dry_run=True,
        system_preset="SHERA_FLIGHT_3P",
        quiet=True,
    )

    assert result["iterative"]["update_mode"] == "eigen_damped"
    assert result["iterative"]["update_policy"]["update_mode"] == "eigen_damped"
    assert "eigenbasis" in result["iterative"]
    assert result["iterative"]["eigenbasis"]["basis_source"] == "posterior_precision"


def test_binary_iterative_rejects_unknown_update_mode(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "iterative.json"
    write_iterative_config(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["experiment"]["iterative"]["update_mode"] = "eigen_future"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="experiment.iterative.update_mode must be one of"):
        module.run_observation_bias_campaign(
            config_path=config_path,
            results_root=tmp_path,
            run_name="bad_mode",
            dry_run=True,
            system_preset="SHERA_FLIGHT_3P",
            quiet=True,
        )


def test_binary_iterative_aggregate_only_reconstructs_synthetic_diagnostics(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "iterative.json"
    write_iterative_config(config_path)
    payload = module.run_observation_bias_campaign(
        config_path=config_path,
        results_root=tmp_path,
        run_name="synthetic_iterative",
        dry_run=True,
        system_preset="SHERA_FLIGHT_3P",
        quiet=True,
    )
    plan = module.build_campaign_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="synthetic_iterative",
        system_preset="SHERA_FLIGHT_3P",
    )
    truth_sep = float(plan.prior_truth[plan.layout.labels.index("source.separation_as")])

    posterior_offsets = [1.0e-6, 0.25e-6]
    for row, sep_offset in zip(payload["expected_outputs"], posterior_offsets, strict=True):
        posterior_path = Path(row["case_posterior_path"])
        posterior_path.parent.mkdir(parents=True, exist_ok=True)
        posterior_path.write_text(
            "case_name,theta_label,truth_value,reference_value,posterior_mean,posterior_sigma\n"
            f"sep_bias,source.separation_as,{truth_sep},0.0,{truth_sep + sep_offset},1e-7\n"
            "sep_bias,optics.primary.zernike_coeffs_nm[0],0.0,0.0,0.0,1.0\n",
            encoding="utf-8",
        )
        Path(row["summary_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(row["summary_path"]).write_text("{}", encoding="utf-8")
        Path(row["window_summary_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(row["window_summary_path"]).write_text("case_name\nsep_bias\n", encoding="utf-8")

    status = module.run_observation_bias_campaign(
        config_path=config_path,
        results_root=tmp_path,
        run_name="synthetic_iterative",
        aggregate_only=True,
        system_preset="SHERA_FLIGHT_3P",
        quiet=True,
    )

    diag_path = tmp_path / "synthetic_iterative" / "analysis" / "iterative_window_diagnostics.csv"
    text = diag_path.read_text(encoding="utf-8")
    with diag_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert status["iterative_window_diagnostic_rows"] == 2
    assert "separation_posterior_update_microas" in text
    assert "separation_applied_update_microas" in text
    assert "residual_norm_decreased_from_previous_window" in text
    assert rows[0]["separation_next_reference_improved"] == "True"
    assert rows[1]["residual_norm_decreased_from_previous_window"] == "True"


def test_binary_iterative_synthetic_wrong_direction_flags_not_improved(tmp_path: Path) -> None:
    module = load_module()
    config_path = tmp_path / "iterative.json"
    write_iterative_config(config_path)
    payload = module.run_observation_bias_campaign(
        config_path=config_path,
        results_root=tmp_path,
        run_name="synthetic_wrong_direction",
        dry_run=True,
        system_preset="SHERA_FLIGHT_3P",
        quiet=True,
    )
    plan = module.build_campaign_plan(
        config_path=config_path,
        results_root=tmp_path,
        run_name="synthetic_wrong_direction",
        system_preset="SHERA_FLIGHT_3P",
    )
    truth_sep = float(plan.prior_truth[plan.layout.labels.index("source.separation_as")])

    for row in payload["expected_outputs"]:
        posterior_path = Path(row["case_posterior_path"])
        posterior_path.parent.mkdir(parents=True, exist_ok=True)
        posterior_path.write_text(
            "case_name,theta_label,truth_value,reference_value,posterior_mean,posterior_sigma\n"
            f"sep_bias,source.separation_as,{truth_sep},0.0,{truth_sep + 3.0e-6},1e-7\n"
            "sep_bias,optics.primary.zernike_coeffs_nm[0],0.0,0.0,0.0,1.0\n",
            encoding="utf-8",
        )
        Path(row["summary_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(row["summary_path"]).write_text("{}", encoding="utf-8")
        Path(row["window_summary_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(row["window_summary_path"]).write_text("case_name\nsep_bias\n", encoding="utf-8")

    module.run_observation_bias_campaign(
        config_path=config_path,
        results_root=tmp_path,
        run_name="synthetic_wrong_direction",
        aggregate_only=True,
        system_preset="SHERA_FLIGHT_3P",
        quiet=True,
    )

    diag_path = tmp_path / "synthetic_wrong_direction" / "analysis" / "iterative_window_diagnostics.csv"
    with diag_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["separation_next_reference_improved"] == "False"
    assert rows[0]["separation_update_sign_toward_truth"] == "False"
