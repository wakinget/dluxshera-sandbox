from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "scripts"
    / "sweep_obs_subblock_adam.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "sweep_obs_subblock_adam_script",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _minimal_base_config() -> dict:
    module = _load_script_module()
    return {
        "system": {"preset": "SHERA_TESTBED_3P"},
        "experiment": {
            "kind": "subblock_inference",
            "inference": {
                "data": {"cube": "cube.fits"},
                "active": {
                    "frame_keys": list(module.CURRENT_TOY_FRAME_KEYS),
                    "shared_keys": [],
                },
                "objective": {"kind": "nll", "reduce": "sum"},
                "optimizer": {
                    "kind": "sgd",
                    "base_lr": 0.5,
                    "n_iter": 40,
                    "preconditioning": {"enabled": False},
                },
                "diagnostics": {"plots": True},
            },
            "outputs": {"outdir": "Results/subblock_inference"},
        },
    }


def test_final_truth_score_normalizes_key_columns():
    module = _load_script_module()
    residuals = np.asarray(
        [
            [1.0e-3, 0.0, 1.0e-2],
            [0.0, -2.0e-3, 0.0],
        ]
    )

    score = module.final_truth_score(residuals, module.CURRENT_TOY_FRAME_KEYS)
    metrics = module.key_residual_metrics(residuals, module.CURRENT_TOY_FRAME_KEYS)

    assert score == pytest.approx(1.0)
    assert metrics["rms_residual.source.x_position_as"] == pytest.approx(
        math.sqrt((1.0e-3**2 + 0.0) / 2.0)
    )
    assert metrics["max_abs_residual.source.y_position_as"] == pytest.approx(2.0e-3)
    assert metrics["rms_residual.source.position_angle_deg"] == pytest.approx(
        math.sqrt((1.0e-2**2 + 0.0) / 2.0)
    )


def test_convergence_and_stability_helpers():
    module = _load_script_module()

    assert module.iter_to_90pct_improvement([10.0, 6.0, 1.9, 1.0]) == 2
    assert module.iter_to_90pct_improvement([1.0, 1.0, 1.0]) == 0
    assert module.tail_std_last_k([5.0, 4.0, 3.0, 2.0, 2.0, 2.0], k=3) == pytest.approx(0.0)
    assert module.max_overshoot_ratio([10.0, 12.0, 8.0, 5.0]) == pytest.approx(1.2)
    assert module.max_overshoot_ratio([0.0, 0.0]) == pytest.approx(1.0)


def _one_component_history(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape((len(values), 1, 1))


def test_settling_iter_tol_on_synthetic_histories():
    module = _load_script_module()

    assert module.settling_iter_tol(_one_component_history([1.0, 0.2, 0.09, 0.05])) == 2
    assert module.settling_iter_tol(_one_component_history([0.5, -0.15, 0.08, -0.05])) == 2
    assert module.settling_iter_tol(
        _one_component_history([0.5, -0.4, 0.3, -0.2, 0.09, -0.08])
    ) == 4
    assert module.settling_iter_tol(_one_component_history([0.04, -0.04, 0.03])) == 0
    assert module.settling_iter_tol(_one_component_history([0.2, 0.11, 0.12])) == 2


def test_ringing_index_on_synthetic_histories():
    module = _load_script_module()

    assert module.ringing_index(_one_component_history([1.0, 0.5, 0.2, 0.0])) == 0.0
    assert module.ringing_index(_one_component_history([1.0, -0.5, -0.2])) == pytest.approx(0.5)
    assert module.ringing_index(_one_component_history([1.0, -0.8, 0.6, -0.4])) == pytest.approx(1.8)
    assert module.ringing_index(_one_component_history([0.04, -0.04, 0.03])) == 0.0


def test_normalized_residual_history_uses_truth_score_scales():
    module = _load_script_module()
    residuals = np.asarray(
        [
            [[1.0e-3, -2.0e-3, 0.5e-2]],
            [[0.0, 1.0e-3, -1.0e-2]],
        ]
    )

    normalized = module.normalize_residual_history(
        residuals,
        module.CURRENT_TOY_FRAME_KEYS,
    )

    assert normalized.shape == residuals.shape
    assert normalized[0, 0, 0] == pytest.approx(1.0)
    assert normalized[0, 0, 1] == pytest.approx(-2.0)
    assert normalized[0, 0, 2] == pytest.approx(0.5)
    assert normalized[1, 0, 2] == pytest.approx(-1.0)


def test_score_curve_decodes_frame_major_theta_history():
    module = _load_script_module()
    truth = np.asarray(
        [
            [1.0e-3, 2.0e-3, 1.0e-2],
            [3.0e-3, 4.0e-3, 2.0e-2],
        ]
    )
    theta0 = np.zeros(6)
    theta_trace = np.vstack([0.5 * truth.ravel(), truth.ravel()])

    curve = module.score_curve_from_theta_history(
        theta0=theta0,
        theta_trace=theta_trace,
        truth_matrix=truth,
        active_keys=module.CURRENT_TOY_FRAME_KEYS,
    )

    assert curve.shape == (3,)
    assert curve[-1] == pytest.approx(0.0)
    assert curve[0] > curve[1] > curve[2]


def test_patch_config_for_adam_point_keeps_base_config_pristine(tmp_path):
    module = _load_script_module()
    base_cfg = _minimal_base_config()
    point = module.AdamSweepPoint(base_lr=3.0e-4, b1=0.7, b2=0.99, eps=1.0e-8)

    patched = module.patch_config_for_adam_point(
        base_cfg,
        point,
        runs_root=tmp_path / "runs",
        per_run_plots=False,
    )

    assert module.validate_current_toy_config(base_cfg) == module.CURRENT_TOY_FRAME_KEYS
    assert base_cfg["experiment"]["inference"]["objective"]["reduce"] == "sum"
    assert base_cfg["experiment"]["inference"]["optimizer"]["kind"] == "sgd"
    optimizer = patched["experiment"]["inference"]["optimizer"]
    assert patched["experiment"]["inference"]["objective"]["reduce"] == "mean"
    assert optimizer["kind"] == "adam"
    assert optimizer["base_lr"] == pytest.approx(3.0e-4)
    assert optimizer["kwargs"] == {"b1": 0.7, "b2": 0.99, "eps": 1.0e-8}
    assert optimizer["n_iter"] == 40
    assert optimizer["preconditioning"] == {"enabled": False}
    assert patched["experiment"]["inference"]["diagnostics"]["plots"] is False
    assert patched["experiment"]["outputs"]["outdir"] == str(tmp_path / "runs")


def test_aggregate_outputs_write_results_manifest_and_ranked_summary(tmp_path):
    module = _load_script_module()
    point_a = module.AdamSweepPoint(base_lr=1.0e-3, b1=0.9, b2=0.999, eps=1.0e-8)
    point_b = module.AdamSweepPoint(base_lr=3.0e-3, b1=0.7, b2=0.99, eps=1.0e-8)
    rows = [
        {
            "run_id": point_a.run_id,
            "completed": True,
            "truth_metrics_available": True,
            "status": "ok",
            "optimizer.kind": "adam",
            "optimizer.base_lr": point_a.base_lr,
            "optimizer.kwargs.b1": point_a.b1,
            "optimizer.kwargs.b2": point_a.b2,
            "optimizer.kwargs.eps": point_a.eps,
            "objective.reduce": "mean",
            "final_truth_score": 0.5,
            "iter_to_90pct_improvement": 7,
            "settling_iter_tol": 8,
            "ringing_index": 0.3,
            "tail_std_last_k": 0.01,
            "max_overshoot_ratio": 1.0,
        },
        {
            "run_id": point_b.run_id,
            "completed": True,
            "truth_metrics_available": True,
            "status": "ok",
            "optimizer.kind": "adam",
            "optimizer.base_lr": point_b.base_lr,
            "optimizer.kwargs.b1": point_b.b1,
            "optimizer.kwargs.b2": point_b.b2,
            "optimizer.kwargs.eps": point_b.eps,
            "objective.reduce": "mean",
            "final_truth_score": 0.25,
            "iter_to_90pct_improvement": 9,
            "settling_iter_tol": 12,
            "ringing_index": 0.5,
            "tail_std_last_k": 0.02,
            "max_overshoot_ratio": 1.1,
        },
    ]

    outputs = module.write_aggregate_outputs(
        output_dir=tmp_path,
        rows=rows,
        base_config_path=tmp_path / "base_config.json",
        grid=[point_a, point_b],
        started_at="2026-04-16T00:00:00",
        completed_at="2026-04-16T00:01:00",
        write_plots=False,
    )

    assert outputs["results_csv"].exists()
    assert outputs["ranked_summary_csv"].exists()
    assert outputs["manifest_json"].exists()
    assert outputs["recommendation_json"].exists()
    assert outputs["recommendation_md"].exists()

    with outputs["results_csv"].open("r", encoding="utf-8", newline="") as handle:
        results_header = next(csv.reader(handle))
    assert "settling_iter_tol" in results_header
    assert "ringing_index" in results_header

    with outputs["ranked_summary_csv"].open("r", encoding="utf-8", newline="") as handle:
        ranked_rows = list(csv.DictReader(handle))
    assert "settling_iter_tol" in ranked_rows[0]
    assert "ringing_index" in ranked_rows[0]
    assert ranked_rows[0]["run_id"] == point_b.run_id
    assert ranked_rows[0]["rank"] == "1"

    manifest = json.loads(outputs["manifest_json"].read_text(encoding="utf-8"))
    assert manifest["schema_version"] == module.MANIFEST_SCHEMA_VERSION
    assert manifest["outputs"]["results_csv"] == module.RESULTS_CSV
    assert manifest["recommendation"]["run_id"] == point_b.run_id
    assert manifest["recommendation"]["optimizer"]["base_lr"] == pytest.approx(3.0e-3)
    assert manifest["recommendation"]["metrics"]["settling_iter_tol"] == pytest.approx(12)
    assert manifest["recommendation"]["metrics"]["ringing_index"] == pytest.approx(0.5)


def test_ranking_uses_oscillation_metrics_after_final_and_quickness_ties():
    module = _load_script_module()
    smooth = {
        "run_id": "smooth",
        "completed": True,
        "final_truth_score": 0.1,
        "iter_to_90pct_improvement": 5,
        "settling_iter_tol": 6,
        "ringing_index": 0.2,
        "tail_std_last_k": 0.1,
        "max_overshoot_ratio": 1.2,
    }
    ringing = {
        "run_id": "ringing",
        "completed": True,
        "final_truth_score": 0.1,
        "iter_to_90pct_improvement": 5,
        "settling_iter_tol": 9,
        "ringing_index": 1.5,
        "tail_std_last_k": 0.01,
        "max_overshoot_ratio": 1.0,
    }

    ranked = module.rank_rows([ringing, smooth])

    assert ranked[0]["run_id"] == "smooth"
    assert ranked[0]["rank"] == 1


def test_run_sweep_dry_run_reports_planned_shape(tmp_path):
    module = _load_script_module()
    cfg_path = tmp_path / "base_config.json"
    cfg_path.write_text(json.dumps(_minimal_base_config(), indent=2), encoding="utf-8")
    grid = [module.AdamSweepPoint(base_lr=1.0e-3, b1=0.9, b2=0.999, eps=1.0e-8)]

    result = module.run_sweep(
        base_config_path=cfg_path,
        results_dir=tmp_path / "Results",
        experiment_name="dry_run_sweep",
        grid=grid,
        no_progress=True,
        per_run_plots=False,
        summary_plots=False,
        dry_run=True,
        fail_fast=False,
    )

    assert result["dry_run"] is True
    assert result["run_count"] == 1
    assert result["run_ids"] == [grid[0].run_id]
    assert result["output_dir"].endswith("dry_run_sweep")
