from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "scripts"
    / "run_observation_bias_campaign.py"
)


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location("run_observation_bias_campaign", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_iterative_forecast_config_normalization_validates() -> None:
    module = load_module()
    disabled = module._resolve_iterative_forecast_config({"subblocks": {"n_subblocks": 1}})
    assert disabled["enabled"] is False
    assert disabled["actual_windows"] == 1

    enabled = module._resolve_iterative_forecast_config(
        {
            "seed": 7,
            "iterative": {
                "enabled": True,
                "windows_per_draw": 2,
                "subblocks_per_window": 3,
            },
            "iterative_forecast": {
                "enabled": True,
                "actual_windows": 2,
                "projected_windows": 5,
                "subblocks_per_window": 3,
                "modes": ["replicate_information", "stochastic_score_noise"],
                "stochastic_trials": 4,
            },
        }
    )
    assert enabled["actual_subblocks_total_per_case"] == 6
    assert enabled["projected_subblocks_total"] == 15

    with pytest.raises(ValueError, match="projected_windows"):
        module._resolve_iterative_forecast_config(
            {
                "iterative": {"subblocks_per_window": 1},
                "iterative_forecast": {
                    "enabled": True,
                    "actual_windows": 3,
                    "projected_windows": 2,
                    "subblocks_per_window": 1,
                },
            }
        )
    with pytest.raises(ValueError, match="stochastic_trials"):
        module._resolve_iterative_forecast_config(
            {
                "iterative": {"subblocks_per_window": 1},
                "iterative_forecast": {
                    "enabled": True,
                    "actual_windows": 1,
                    "projected_windows": 2,
                    "subblocks_per_window": 1,
                    "modes": ["stochastic_score_noise"],
                    "stochastic_trials": 0,
                },
            }
        )


def _fake_plan(module: Any, tmp_path: Path):
    case = module.BiasCase(
        case_name="m1_0p3nm_m2_0p3nm_draw_000",
        theta_reference_offsets={"source.separation_as": 100.0e-6},
        case_origin="prior_draw",
        prior_sigma_by_label={"source.separation_as": 100.0e-6},
        prior_draw_metadata={
            "condition_name": "m1_0p3nm_m2_0p3nm",
            "draw_index": 0,
        },
    )
    return module.CampaignPlan(
        run_root=tmp_path,
        layout=SimpleNamespace(labels=("source.separation_as",), size=1, to_dict=lambda: {"labels": ["source.separation_as"]}),
        layout_metadata={},
        prior_truth=pd.Series([0.0]).to_numpy(),
        cases=(case,),
        subblock_commands={},
        summary_paths={},
        subblock_plans={},
        prior_draw_rows_by_case={},
        config={
            "experiment": {
                "seed": 11,
                "subblocks": {"phi_ref": "truth_when_available"},
                "detector_calibration_knowledge_error": {
                    "enabled": True,
                    "apply_to": "inference",
                    "pixel_offsets": {"sigma_pix": 0.001},
                    "pixel_response": {"sigma_fractional": 0.001},
                },
                "iterative": {
                    "enabled": True,
                    "windows_per_draw": 2,
                    "subblocks_per_window": 1,
                    "update_mode": "eigen_damped",
                    "eigenbasis": {"damping_mode": "information"},
                },
                "iterative_forecast": {
                    "enabled": True,
                    "actual_windows": 2,
                    "projected_windows": 4,
                    "subblocks_per_window": 1,
                    "observation_duration_s": 4.0,
                    "modes": ["replicate_information", "stochastic_score_noise"],
                    "stochastic_trials": 8,
                    "seed": 123,
                    "plots": False,
                },
            }
        },
        partition={},
        case_generation={},
        truth_realization={},
        truth_realization_rows=[],
        trace_source_plan=SimpleNamespace(summary={}),
        iterative={
            "enabled": True,
            "windows_per_draw": 2,
            "subblocks_per_window": 1,
            "update_mode": "eigen_damped",
            "eigenbasis": {"damping_mode": "information"},
        },
        iterative_plan_rows=[],
        expected_output_rows=[],
    )


def _diagnostics() -> list[dict[str, Any]]:
    return [
        {
            "case_name": "m1_0p3nm_m2_0p3nm_draw_000",
            "window_index": 0,
            "separation_reference_error_before_microas": 100.0,
            "separation_next_reference_error_microas": 80.0,
            "posterior_sigma_separation_microas": 20.0,
            "reference_error_norm_before": 100.0e-6,
            "next_reference_error_norm": 80.0e-6,
            "update_cosine_with_ideal": 0.9,
        },
        {
            "case_name": "m1_0p3nm_m2_0p3nm_draw_000",
            "window_index": 1,
            "separation_reference_error_before_microas": 80.0,
            "separation_next_reference_error_microas": 40.0,
            "posterior_sigma_separation_microas": 14.1421356237,
            "reference_error_norm_before": 80.0e-6,
            "next_reference_error_norm": 40.0e-6,
            "update_cosine_with_ideal": 0.8,
        },
    ]


def test_iterative_forecast_writes_required_artifacts_and_trials(tmp_path: Path) -> None:
    module = load_module()
    plan = _fake_plan(module, tmp_path)
    summary = module.write_iterative_observation_forecast_artifacts(plan, _diagnostics())
    assert summary["enabled"] is True
    assert summary["n_final_rows"] == 1
    assert summary["n_trial_rows"] == 8

    final = pd.read_csv(tmp_path / "analysis/final_observation_summary.csv")
    evolution = pd.read_csv(tmp_path / "analysis/window_evolution_actual_and_projected.csv")
    trials = pd.read_csv(tmp_path / "analysis/projected_observation_forecast_trials.csv")
    payload = json.loads((tmp_path / "analysis/final_observation_summary.json").read_text())

    assert len(evolution) == 4
    assert set(evolution["window_kind"]) == {"actual", "projected"}
    assert final.loc[0, "actual_windows"] == 2
    assert final.loc[0, "projected_windows"] == 4
    assert final.loc[0, "projected_subblocks_total"] == 4
    assert final.loc[0, "phi_ref"] == "truth_when_available"
    assert final.loc[0, "update_mode"] == "eigen_damped"
    assert final.loc[0, "eigen_damping_mode"] == "information"
    assert final.loc[0, "separation_prior_sigma_microas"] == pytest.approx(100.0)
    assert final.loc[0, "projected_final_posterior_sigma_separation_microas"] < final.loc[0, "actual_final_posterior_sigma_separation_microas"]
    assert "projected_p50_separation_error_microas" in final.columns
    assert len(trials) == 8
    assert payload["trial_rows"] == 8


def test_iterative_forecast_stochastic_seed_reproducibility(tmp_path: Path) -> None:
    module = load_module()
    plan = _fake_plan(module, tmp_path / "a")
    module.write_iterative_observation_forecast_artifacts(plan, _diagnostics())
    first = pd.read_csv(tmp_path / "a/analysis/projected_observation_forecast_trials.csv")

    plan_same = _fake_plan(module, tmp_path / "b")
    module.write_iterative_observation_forecast_artifacts(plan_same, _diagnostics())
    second = pd.read_csv(tmp_path / "b/analysis/projected_observation_forecast_trials.csv")
    assert first["projected_final_separation_error_microas"].tolist() == second["projected_final_separation_error_microas"].tolist()

    plan_diff = _fake_plan(module, tmp_path / "c")
    plan_diff.config["experiment"]["iterative_forecast"]["seed"] = 124
    module.write_iterative_observation_forecast_artifacts(plan_diff, _diagnostics())
    third = pd.read_csv(tmp_path / "c/analysis/projected_observation_forecast_trials.csv")
    assert first["projected_final_separation_error_microas"].tolist() != third["projected_final_separation_error_microas"].tolist()
