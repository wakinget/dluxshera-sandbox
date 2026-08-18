from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "scripts"
    / "sweep_obs_subblock_sgd.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "sweep_obs_subblock_sgd_script",
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
                "objective": {
                    "kind": "nll",
                    "frame_reduce": "sum",
                    "subblock_reduce": "sum",
                },
                "optimizer": {
                    "kind": "adam",
                    "base_lr": 0.1,
                    "n_iter": 40,
                    "preconditioning": {"enabled": False},
                },
                "diagnostics": {"plots": True},
            },
            "outputs": {"outdir": "Results/subblock_inference"},
        },
    }


def test_build_sgd_grid_constructs_explicit_product():
    module = _load_script_module()

    grid = module.build_sgd_grid(
        base_lrs=(0.25, 0.5),
        momentums=(0.0, 0.8),
        nesterov_values=(False, True),
    )

    assert len(grid) == 8
    assert grid[0] == module.SGDSweepPoint(
        base_lr=0.25,
        momentum=0.0,
        nesterov=False,
    )
    assert grid[-1] == module.SGDSweepPoint(
        base_lr=0.5,
        momentum=0.8,
        nesterov=True,
    )
    assert grid[-1].optimizer_kwargs == {"momentum": 0.8, "nesterov": True}
    assert grid[-1].run_id == "sgd_lr0p5_mom0p8_nesttrue"


def test_build_sgd_grid_validates_ranges():
    module = _load_script_module()

    with pytest.raises(ValueError, match="base learning rates"):
        module.build_sgd_grid(base_lrs=(0.0,))
    with pytest.raises(ValueError, match="momentum"):
        module.build_sgd_grid(momentums=(1.0,))
    with pytest.raises(ValueError, match="nesterov"):
        module.build_sgd_grid(nesterov_values=())


def test_parse_bool_list_accepts_common_nesterov_spellings():
    module = _load_script_module()

    parsed = module._parse_bool_list(
        "true,false,1,0,yes,no,on,off",
        label="--nesterov-values",
    )

    assert parsed == (True, False, True, False, True, False, True, False)
    with pytest.raises(ValueError, match="comma-separated list of bools"):
        module._parse_bool_list("true,maybe", label="--nesterov-values")


def test_patch_config_for_sgd_point_keeps_base_config_pristine(tmp_path):
    module = _load_script_module()
    base_cfg = _minimal_base_config()
    point = module.SGDSweepPoint(base_lr=0.75, momentum=0.8, nesterov=True)

    patched = module.patch_config_for_sgd_point(
        base_cfg,
        point,
        runs_root=tmp_path / "runs",
        per_run_plots=False,
    )

    assert module.validate_current_toy_config(base_cfg) == module.CURRENT_TOY_FRAME_KEYS
    assert base_cfg["experiment"]["inference"]["objective"]["frame_reduce"] == "sum"
    assert base_cfg["experiment"]["inference"]["optimizer"]["kind"] == "adam"

    inference = patched["experiment"]["inference"]
    optimizer = inference["optimizer"]
    assert inference["objective"]["frame_reduce"] == "mean"
    assert inference["objective"]["subblock_reduce"] == "sum"
    assert optimizer["kind"] == "sgd"
    assert optimizer["base_lr"] == pytest.approx(0.75)
    assert optimizer["kwargs"] == {"momentum": 0.8, "nesterov": True}
    assert optimizer["n_iter"] == 40
    assert optimizer["preconditioning"] == {"enabled": False}
    assert inference["diagnostics"]["plots"] is False
    assert patched["experiment"]["outputs"]["outdir"] == str(tmp_path / "runs")


def test_validate_current_toy_config_rejects_broadened_scope():
    module = _load_script_module()
    valid_cfg = _minimal_base_config()
    assert module.validate_current_toy_config(valid_cfg) == module.CURRENT_TOY_FRAME_KEYS

    wrong_kind = _minimal_base_config()
    wrong_kind["experiment"]["kind"] = "other"
    with pytest.raises(ValueError, match="subblock_inference"):
        module.validate_current_toy_config(wrong_kind)

    shared = _minimal_base_config()
    shared["experiment"]["inference"]["active"]["shared_keys"] = ["source.flux"]
    with pytest.raises(ValueError, match="shared_keys"):
        module.validate_current_toy_config(shared)

    frame_keys = _minimal_base_config()
    frame_keys["experiment"]["inference"]["active"]["frame_keys"] = [
        "source.x_position_as"
    ]
    with pytest.raises(ValueError, match="registration-only toy frame keys"):
        module.validate_current_toy_config(frame_keys)


def test_recommendation_outputs_use_sgd_shape(tmp_path):
    module = _load_script_module()
    point_a = module.SGDSweepPoint(base_lr=0.5, momentum=0.0, nesterov=False)
    point_b = module.SGDSweepPoint(base_lr=0.75, momentum=0.8, nesterov=True)
    rows = [
        {
            "run_id": point_a.run_id,
            "completed": True,
            "truth_metrics_available": True,
            "status": "ok",
            "optimizer.kind": "sgd",
            "optimizer.base_lr": point_a.base_lr,
            "optimizer.kwargs.momentum": point_a.momentum,
            "optimizer.kwargs.nesterov": point_a.nesterov,
            "objective.frame_reduce": "mean",
            "objective.subblock_reduce": "sum",
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
            "optimizer.kind": "sgd",
            "optimizer.base_lr": point_b.base_lr,
            "optimizer.kwargs.momentum": point_b.momentum,
            "optimizer.kwargs.nesterov": point_b.nesterov,
            "objective.frame_reduce": "mean",
            "objective.subblock_reduce": "sum",
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

    with outputs["results_csv"].open("r", encoding="utf-8", newline="") as handle:
        results_header = next(csv.reader(handle))
    assert "optimizer.kwargs.momentum" in results_header
    assert "optimizer.kwargs.nesterov" in results_header

    manifest = json.loads(outputs["manifest_json"].read_text(encoding="utf-8"))
    assert manifest["schema_version"] == module.MANIFEST_SCHEMA_VERSION
    assert manifest["grid"]["optimizer.kind"] == "sgd"
    assert manifest["grid"]["momentums"] == [0.0, 0.8]
    assert manifest["grid"]["nesterov_values"] == [False, True]
    assert manifest["recommendation"]["run_id"] == point_b.run_id
    assert manifest["recommendation"]["optimizer"] == {
        "kind": "sgd",
        "base_lr": 0.75,
        "kwargs": {"momentum": 0.8, "nesterov": True},
    }

    recommendation_md = outputs["recommendation_md"].read_text(encoding="utf-8")
    assert "kind: sgd" in recommendation_md
    assert "momentum: 0.8" in recommendation_md
    assert "nesterov: true" in recommendation_md


def test_run_sweep_dry_run_reports_planned_shape(tmp_path):
    module = _load_script_module()
    cfg_path = tmp_path / "base_config.json"
    cfg_path.write_text(json.dumps(_minimal_base_config(), indent=2), encoding="utf-8")
    grid = [module.SGDSweepPoint(base_lr=0.5, momentum=0.8, nesterov=True)]

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
