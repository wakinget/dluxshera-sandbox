"""Smoke tests for observation sub-block inference workflow."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np


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
) -> dict[str, object]:
    inputs: dict[str, object] = {"cube": str(cube_path)}
    if trace_path is not None:
        inputs["trace"] = str(trace_path)
    if manifest_path is not None:
        inputs["manifest"] = str(manifest_path)

    return {
        "system": _base_system_block(),
        "experiment": {
            "kind": "observation_subblock_inference",
            "truth": {"source": {"exposure_time_s": 0.05}},
            "observation_subblock_inference": {
                "inputs": inputs,
                "validate": {
                    "require_contiguous_frame_index": True,
                    "require_monotonic_time": True,
                },
                "init": {
                    "x_position_as": 0.0,
                    "y_position_as": 0.0,
                    "position_angle_deg": 90.0,
                },
                "loss": {"reduce": "sum", "variance": 1.0},
                "optimizer": {"kind": "adam", "base_lr": 0.01, "n_iter": n_iter},
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
        / "prescription.yaml"
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
    )
    cfg_path = tmp_path / "inference_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    result = inference_recipe.generate_obs_subblock_inference(
        config_path=cfg_path,
        run_name="inference_smoke",
        show_progress=False,
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

    manifest = json.loads(artifacts["manifest_json"].read_text(encoding="utf-8"))
    for key in (
        "schema_version",
        "created_at",
        "generator",
        "frame_count",
        "infer_keys",
        "inputs",
        "init",
        "optimizer",
        "loss",
        "metrics",
        "shared_truth",
        "system",
        "artifacts",
    ):
        assert key in manifest
    assert manifest["truth_comparison_available"] is True
    assert manifest["inputs"]["manifest_auto_discovered"] is True
    assert manifest["inputs"]["config_path"].endswith("inference_config.json")
    assert manifest["system"]["resolved_config"]["preset"] == "SHERA_TESTBED_3P"
    assert manifest["shared_truth"]["source"]["exposure_time_s"] == 0.05
    assert manifest["init"]["x_position_as"] == 0.0
    assert manifest["init"]["y_position_as"] == 0.0
    assert manifest["init"]["position_angle_deg"] == 90.0

    assert np.isfinite(float(result["initial_loss"]))
    assert np.isfinite(float(result["final_loss"]))
    assert float(result["final_loss"]) <= float(result["initial_loss"])


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
        / "prescription.yaml"
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
    )
    cfg_path = tmp_path / "inference_no_truth_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    result = inference_recipe.generate_obs_subblock_inference(
        config_path=cfg_path,
        run_name="inference_no_truth",
        show_progress=False,
    )

    artifacts = {name: Path(path) for name, path in result["artifacts"].items()}
    assert "recovered_trace_csv" in artifacts
    assert "manifest_json" in artifacts
    assert "truth_comparison_csv" not in artifacts

    recovered_rows = _read_csv_rows(artifacts["recovered_trace_csv"])
    assert len(recovered_rows) == int(result["frame_count"])

    manifest = json.loads(artifacts["manifest_json"].read_text(encoding="utf-8"))
    assert manifest["truth_comparison_available"] is False
    assert manifest["frame_count"] == int(result["frame_count"])
    assert manifest["inputs"]["manifest_auto_discovered"] is False
