"""Smoke tests for observation sub-block inference workflow."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def _load_recipe(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load recipe at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _base_system_block() -> dict[str, object]:
    return {
        "preset": "SHERA_TESTBED_3P",
        "source": {"n_lambda": 1},
        "detector": {"layers": [{"name": "downsample", "kind": "Downsample", "kernel_size": 3}]},
    }


def _build_inference_config(
    *,
    cube_path: Path,
    trace_path: Path | None,
    manifest_path: Path | None,
    outputs_outdir: Path,
    n_iter: int,
    write_plots: bool,
    enable_preconditioning: bool = False,
    active_frame_keys: list[str] | None = None,
    optimizer_schedule: dict[str, object] | None = None,
    early_stopping: dict[str, object] | None = None,
) -> dict[str, object]:
    data: dict[str, object] = {"cube": str(cube_path)}
    if trace_path is not None:
        data["truth_trace"] = str(trace_path)
    if manifest_path is not None:
        data["manifest"] = str(manifest_path)
    frame_keys = active_frame_keys or [
        "source.x_position_as",
        "source.y_position_as",
        "source.position_angle_deg",
    ]

    return {
        "system": _base_system_block(),
        "experiment": {
            "kind": "subblock_inference",
            "inference": {
                "data": data,
                "validate": {
                    "require_contiguous_frame_index": True,
                    "require_monotonic_time": True,
                },
                "init": {
                    "frame": {
                        "mode": "shared_guess",
                        "values": {
                            "source.x_position_as": 0.0,
                            "source.y_position_as": 0.0,
                            "source.position_angle_deg": 90.0,
                        },
                    },
                    "shared": {},
                },
                "active": {
                    "frame_keys": frame_keys,
                    "shared_keys": [],
                },
                "priors": {"frame": {}, "shared": {}},
                "temporal": {"frame_model": {"kind": "independent"}},
                "objective": {
                    "kind": "nll",
                    "frame_reduce": "sum",
                    "subblock_reduce": "sum",
                    "noise_model": {
                        "kind": "gaussian",
                        "variance_model": "scalar",
                        "scalar": 1.0,
                    },
                },
                "optimizer": {
                    "kind": "adam",
                    "base_lr": 0.01,
                    "n_iter": n_iter,
                    "schedule": optimizer_schedule,
                    "early_stopping": early_stopping,
                    "preconditioning": {
                        "enabled": enable_preconditioning,
                        "damping": 1e-6,
                        "eig_floor_rel": 1e-6,
                        "eig_floor_abs": 1e-8,
                    },
                },
                "diagnostics": {"plots": write_plots},
            },
            "outputs": {
                "outdir": str(outputs_outdir),
                "file_prefix": "obs_subblock_inference",
            },
        },
    }


def test_observation_subblock_inference_recipe_smoke_with_truth_outputs(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    trace_recipe = _load_recipe(
        repo_root / "examples" / "recipes" / "observation_subblock.py",
        "observation_subblock_recipe_for_inference_smoke",
    )
    inference_recipe = _load_recipe(
        repo_root / "examples" / "recipes" / "observation_subblock_inference.py",
        "observation_subblock_inference_recipe_smoke",
    )

    render_cfg_path = (
        repo_root
        / "examples"
        / "recipes"
        / "observation_subblock_template"
        / "subblock_generation_prescription.yaml"
    )
    render_result = trace_recipe.generate_obs_subblock(
        config_path=render_cfg_path,
        results_dir=tmp_path / "render_results",
        run_name="render_for_inference",
        show_progress=False,
    )

    cube_path = Path(render_result["artifacts"]["cube_fits"])
    cfg = _build_inference_config(
        cube_path=cube_path,
        trace_path=None,
        manifest_path=None,
        outputs_outdir=tmp_path / "inference_results",
        n_iter=10,
        write_plots=True,
        enable_preconditioning=True,
    )
    cfg_path = tmp_path / "inference_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    result = inference_recipe.main(
        [
            "--config",
            str(cfg_path),
            "--run-name",
            "inference_smoke",
            "--no-progress",
        ]
    )

    artifacts = {name: Path(path) for name, path in result["artifacts"].items()}
    for name in (
        "recovered_trace_csv",
        "truth_comparison_csv",
        "manifest_json",
        "loss_history_png",
        "image_fit_png",
        "trace_comparison_png",
        "trace_residuals_png",
    ):
        assert name in artifacts
        assert artifacts[name].exists()
        assert artifacts[name].stat().st_size > 0

    recovered_rows = _read_csv_rows(artifacts["recovered_trace_csv"])
    comparison_rows = _read_csv_rows(artifacts["truth_comparison_csv"])
    assert len(recovered_rows) == result["frame_count"]
    assert len(comparison_rows) == result["frame_count"]
    assert "frame_chi2" in recovered_rows[0]
    assert "frame_reduced_chi2" in recovered_rows[0]
    assert "frame_chi2_dof_pixels" in recovered_rows[0]
    assert "frame_chi2" in comparison_rows[0]
    assert "frame_reduced_chi2" in comparison_rows[0]
    assert "frame_chi2_dof_pixels" in comparison_rows[0]

    manifest = json.loads(artifacts["manifest_json"].read_text(encoding="utf-8"))
    for key in (
        "schema_version",
        "created_at",
        "generator",
        "frame_count",
        "infer_keys",
        "inputs",
        "active",
        "init",
        "priors",
        "temporal",
        "objective",
        "optimizer",
        "metrics",
        "system",
        "artifacts",
    ):
        assert key in manifest
    assert manifest["truth_comparison_available"] is True
    assert manifest["inputs"]["manifest_auto_discovered"] is True
    assert manifest["inputs"]["config_path"].endswith("inference_config.json")
    assert manifest["system"]["resolved_config"]["preset"] == "SHERA_TESTBED_3P"
    assert manifest["init"]["frame"]["mode"] == "shared_guess"
    assert manifest["init"]["frame"]["values"]["source.x_position_as"] == 0.0
    assert manifest["init"]["frame"]["values"]["source.y_position_as"] == 0.0
    assert manifest["init"]["frame"]["values"]["source.position_angle_deg"] == 90.0
    assert manifest["objective"]["frame_reduce"] == "sum"
    assert manifest["objective"]["subblock_reduce"] == "sum"
    assert "reduce" not in manifest["objective"]
    chi2_metrics = manifest["metrics"]["chi2"]
    assert "positive finite variance" in chi2_metrics["metric_notes"]
    assert chi2_metrics["per_frame_csv_columns"] == [
        "frame_chi2",
        "frame_reduced_chi2",
        "frame_chi2_dof_pixels",
    ]
    assert len(chi2_metrics["final_model"]["per_frame_reduced_chi2"]) == int(
        result["frame_count"]
    )
    assert chi2_metrics["initial_model"]["block_sum_chi2"] is not None
    assert chi2_metrics["final_model"]["block_reduced_chi2"] is not None
    precond_meta = manifest["optimizer"]["preconditioning"]
    assert precond_meta["enabled"] is True
    assert precond_meta["theta_dim"] > 0
    assert precond_meta["lr_vec_max"] >= precond_meta["lr_vec_min"] > 0.0

    assert np.isfinite(float(result["initial_loss"]))
    assert np.isfinite(float(result["final_loss"]))
    assert len(result["chi2"]["final_model"]["per_frame_chi2"]) == int(result["frame_count"])
    assert float(result["final_loss"]) <= float(result["initial_loss"])


def test_observation_subblock_inference_recipe_records_early_stopping_metadata(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    render_recipe = _load_recipe(
        repo_root / "examples" / "recipes" / "observation_subblock.py",
        "observation_subblock_recipe_for_early_stopping_smoke",
    )
    inference_recipe = _load_recipe(
        repo_root / "examples" / "recipes" / "observation_subblock_inference.py",
        "observation_subblock_inference_recipe_early_stopping_smoke",
    )

    render_cfg_path = (
        repo_root
        / "examples"
        / "recipes"
        / "observation_subblock_template"
        / "subblock_generation_prescription.yaml"
    )
    render_result = render_recipe.generate_obs_subblock(
        config_path=render_cfg_path,
        results_dir=tmp_path / "render_results_early_stopping",
        run_name="render_for_early_stopping",
        show_progress=False,
    )

    cfg = _build_inference_config(
        cube_path=Path(render_result["artifacts"]["cube_fits"]),
        trace_path=Path(render_result["artifacts"]["frame_truth_csv"]),
        manifest_path=Path(render_result["artifacts"]["manifest_json"]),
        outputs_outdir=tmp_path / "inference_results_early_stopping",
        n_iter=5,
        write_plots=False,
        early_stopping={
            "enabled": True,
            "min_iter": 2,
            "patience": 2,
            "loss_rtol": 1.0e-8,
            "loss_atol": 0.0,
            "step_atol": 1.0e-10,
            "grad_norm_atol": 1.0e-8,
        },
    )
    cfg_path = tmp_path / "inference_early_stopping_config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    result = inference_recipe.main(
        ["--config", str(cfg_path), "--no-progress"],
    )

    manifest = json.loads(
        Path(result["artifacts"]["manifest_json"]).read_text(encoding="utf-8")
    )
    early_stopping = manifest["optimizer"]["early_stopping"]
    assert early_stopping["enabled"] is True
    assert early_stopping["min_iter"] == 2
    assert early_stopping["patience"] == 2
    assert early_stopping["loss_rtol"] == pytest.approx(1.0e-8)
    assert early_stopping["source"] == "experiment.inference.optimizer.early_stopping"
    assert manifest["optimizer"]["early_stopping_result"]["enabled"] is True
    assert result["early_stopping"]["enabled"] is True
    assert "early_stopping" not in result["trace_history"]


def test_observation_subblock_inference_recipe_writes_schedule_artifact_when_configured(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    render_recipe = _load_recipe(
        repo_root / "examples" / "recipes" / "observation_subblock.py",
        "observation_subblock_recipe_for_schedule_smoke",
    )
    inference_recipe = _load_recipe(
        repo_root / "examples" / "recipes" / "observation_subblock_inference.py",
        "observation_subblock_inference_recipe_schedule_smoke",
    )

    render_cfg_path = (
        repo_root
        / "examples"
        / "recipes"
        / "observation_subblock_template"
        / "subblock_generation_prescription.yaml"
    )
    render_result = render_recipe.generate_obs_subblock(
        config_path=render_cfg_path,
        results_dir=tmp_path / "render_results_schedule",
        run_name="render_for_schedule",
        show_progress=False,
    )

    cfg = _build_inference_config(
        cube_path=Path(render_result["artifacts"]["cube_fits"]),
        trace_path=None,
        manifest_path=None,
        outputs_outdir=tmp_path / "inference_results_schedule",
        n_iter=6,
        write_plots=False,
        enable_preconditioning=True,
        optimizer_schedule={
            "kind": "linear_warmup",
            "warmup_steps": 2,
            "start_factor": 0.25,
        },
    )
    cfg_path = tmp_path / "inference_schedule_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    result = inference_recipe.main(
        [
            "--config",
            str(cfg_path),
            "--run-name",
            "inference_schedule",
            "--no-progress",
        ]
    )

    artifacts = {name: Path(path) for name, path in result["artifacts"].items()}
    assert artifacts["optimizer_schedule_csv"].exists()

    schedule_rows = _read_csv_rows(artifacts["optimizer_schedule_csv"])
    assert len(schedule_rows) == 6
    assert float(schedule_rows[0]["schedule_factor"]) == pytest.approx(0.25)
    assert float(schedule_rows[0]["scalar_lr"]) == pytest.approx(0.0025)
    assert float(schedule_rows[-1]["schedule_factor"]) == pytest.approx(1.0)

    manifest = json.loads(artifacts["manifest_json"].read_text(encoding="utf-8"))
    schedule_meta = manifest["optimizer"]["schedule"]
    assert schedule_meta["enabled"] is True
    assert schedule_meta["kind"] == "linear_warmup"
    assert schedule_meta["first_factor"] == pytest.approx(0.25)
    assert schedule_meta["last_factor"] == pytest.approx(1.0)
    assert schedule_meta["first_scalar_lr"] == pytest.approx(0.0025)
    assert schedule_meta["last_scalar_lr"] == pytest.approx(0.01)


def test_observation_subblock_inference_accepts_partial_truth_trace(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    render_recipe = _load_recipe(
        repo_root / "examples" / "recipes" / "observation_subblock.py",
        "observation_subblock_recipe_for_partial_truth",
    )
    inference_recipe = _load_recipe(
        repo_root / "examples" / "recipes" / "observation_subblock_inference.py",
        "observation_subblock_inference_recipe_partial_truth",
    )

    render_cfg_path = (
        repo_root
        / "examples"
        / "recipes"
        / "observation_subblock_template"
        / "subblock_generation_prescription.yaml"
    )
    render_result = render_recipe.generate_obs_subblock(
        config_path=render_cfg_path,
        results_dir=tmp_path / "render_results_partial_truth",
        run_name="render_for_partial_truth",
        show_progress=False,
    )

    cfg = _build_inference_config(
        cube_path=Path(render_result["artifacts"]["cube_fits"]),
        trace_path=Path(render_result["artifacts"]["frame_truth_csv"]),
        manifest_path=None,
        outputs_outdir=tmp_path / "inference_results_partial_truth",
        n_iter=1,
        write_plots=False,
        enable_preconditioning=True,
        active_frame_keys=[
            "source.x_position_as",
            "source.y_position_as",
            "source.position_angle_deg",
            "source.log_flux_total",
        ],
    )
    cfg_path = tmp_path / "inference_partial_truth_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    result = inference_recipe.main(
        [
            "--config",
            str(cfg_path),
            "--run-name",
            "inference_partial_truth",
            "--no-progress",
        ]
    )

    artifacts = {name: Path(path) for name, path in result["artifacts"].items()}
    assert artifacts["truth_comparison_csv"].exists()
    comparison_rows = _read_csv_rows(artifacts["truth_comparison_csv"])
    assert "source.log_flux_total_truth" in comparison_rows[0]
    assert result["truth"]["frame_key_sources"] == {
        "source.x_position_as": "trace_csv",
        "source.y_position_as": "trace_csv",
        "source.position_angle_deg": "trace_csv",
        "source.log_flux_total": "resolved_store",
    }

    manifest = json.loads(artifacts["manifest_json"].read_text(encoding="utf-8"))
    assert manifest["truth"]["frame_key_sources"]["source.log_flux_total"] == (
        "resolved_store"
    )
    assert manifest["truth"]["complete_for_active_frame_keys"] is True
    assert manifest["truth_comparison_available"] is True
    assert manifest["optimizer"]["preconditioning"]["reference_source"] == "truth_mixed"


def test_observation_subblock_inference_recipe_without_truth_still_writes_core_outputs(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    render_recipe = _load_recipe(
        repo_root / "examples" / "recipes" / "observation_subblock.py",
        "observation_subblock_recipe_for_inference_no_truth",
    )
    inference_recipe = _load_recipe(
        repo_root / "examples" / "recipes" / "observation_subblock_inference.py",
        "observation_subblock_inference_recipe_no_truth",
    )

    render_cfg_path = (
        repo_root
        / "examples"
        / "recipes"
        / "observation_subblock_template"
        / "subblock_generation_prescription.yaml"
    )
    render_result = render_recipe.generate_obs_subblock(
        config_path=render_cfg_path,
        results_dir=tmp_path / "render_results_no_truth",
        run_name="render_for_inference_no_truth",
        show_progress=False,
    )
    cube_path = Path(render_result["artifacts"]["cube_fits"])
    standalone_cube_path = tmp_path / "standalone_cube.fits"
    standalone_cube_path.write_bytes(cube_path.read_bytes())

    cfg = _build_inference_config(
        cube_path=standalone_cube_path,
        trace_path=None,
        manifest_path=None,
        outputs_outdir=tmp_path / "inference_results_no_truth",
        n_iter=4,
        write_plots=False,
        enable_preconditioning=False,
    )
    cfg_path = tmp_path / "inference_no_truth_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    result = inference_recipe.main(
        [
            "--config",
            str(cfg_path),
            "--run-name",
            "inference_no_truth",
            "--no-progress",
        ]
    )

    artifacts = {name: Path(path) for name, path in result["artifacts"].items()}
    assert "recovered_trace_csv" in artifacts
    assert "manifest_json" in artifacts
    assert "truth_comparison_csv" not in artifacts

    recovered_rows = _read_csv_rows(artifacts["recovered_trace_csv"])
    assert len(recovered_rows) == int(result["frame_count"])
    assert "frame_chi2" in recovered_rows[0]
    assert "frame_reduced_chi2" in recovered_rows[0]
    assert "frame_chi2_dof_pixels" in recovered_rows[0]

    manifest = json.loads(artifacts["manifest_json"].read_text(encoding="utf-8"))
    assert manifest["truth_comparison_available"] is False
    assert manifest["frame_count"] == int(result["frame_count"])
    assert manifest["inputs"]["manifest_auto_discovered"] is False
    assert manifest["objective"]["frame_reduce"] == "sum"
    assert manifest["objective"]["subblock_reduce"] == "sum"
    assert len(manifest["metrics"]["chi2"]["final_model"]["per_frame_chi2"]) == int(
        result["frame_count"]
    )
