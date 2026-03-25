"""Smoke tests for observation sub-block trace generation workflow."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from dluxshera.utils.obs_subblock_trace import (
    APPLIED_V1_VARYING_KEYS,
    load_obs_subblock_trace_csv,
)


def _load_recipe(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load recipe at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_observation_subblock_trace_recipe_generates_loader_compatible_csv(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    trace_recipe_path = repo_root / "examples" / "recipes" / "observation_subblock_trace.py"
    trace_recipe = _load_recipe(trace_recipe_path, "observation_subblock_trace_recipe")
    trace_template_cfg = (
        repo_root
        / "examples"
        / "recipes"
        / "observation_subblock_trace_template"
        / "prescription.yaml"
    )

    result = trace_recipe.generate_obs_subblock_trace(
        config_path=trace_template_cfg,
        results_dir=tmp_path,
        run_name="trace_smoke",
    )

    trace_csv = Path(result["artifacts"]["trace_csv"])
    manifest_path = Path(result["artifacts"]["manifest_json"])

    assert trace_csv.exists()
    assert manifest_path.exists()

    trace = load_obs_subblock_trace_csv(trace_csv)
    assert trace.frame_count == 3
    assert trace.required_columns == (
        "frame_index",
        "time_s",
        "source.x_position_as",
        "source.y_position_as",
        "source.position_angle_deg",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in (
        "schema_version",
        "created_at",
        "generator",
        "frame_count",
        "applied_varying_keys",
        "trace",
        "artifacts",
    ):
        assert key in manifest
    assert manifest["applied_varying_keys"] == list(APPLIED_V1_VARYING_KEYS)
    assert manifest["trace"]["path"].endswith("_frame_truth.csv")


def test_generated_trace_can_be_rendered_and_tracks_requested_vs_applied_keys(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    trace_recipe = _load_recipe(
        repo_root / "examples" / "recipes" / "observation_subblock_trace.py",
        "observation_subblock_trace_recipe_render",
    )
    render_recipe = _load_recipe(
        repo_root / "examples" / "recipes" / "observation_subblock.py",
        "observation_subblock_recipe_from_trace",
    )

    trace_cfg = {
        "experiment": {
            "kind": "observation_subblock_trace",
            "seed": 101,
            "observation_subblock_trace": {
                "n_frames": 2,
                "dt_s": 0.05,
                "keys": {
                    "source.x_position_as": {"mode": "explicit", "values": [0.0, 0.01]},
                    "source.y_position_as": {
                        "mode": "linear_drift",
                        "start": 0.0,
                        "rate_per_s": -0.02,
                    },
                    "source.position_angle_deg": {
                        "mode": "iid_jitter",
                        "center": 90.0,
                        "sigma": 0.0,
                    },
                },
            },
            "outputs": {"outdir": str(tmp_path / "trace_out"), "write_manifest": True},
        }
    }
    trace_cfg_path = tmp_path / "trace_recipe.json"
    trace_cfg_path.write_text(json.dumps(trace_cfg, indent=2), encoding="utf-8")

    trace_result = trace_recipe.generate_obs_subblock_trace(
        config_path=trace_cfg_path,
        run_name="generated_trace",
    )
    trace_csv = Path(trace_result["artifacts"]["trace_csv"])
    assert trace_csv.exists()

    renderer_cfg = {
        "system": {
            "preset": "SHERA_TESTBED_3P",
            "source": {"n_lambda": 1},
            "detector": {
                "layers": [
                    {"name": "downsample", "kernel_size": 3},
                ]
            },
        },
        "experiment": {
            "kind": "observation_subblock",
            "seed": 33,
            "truth": {"source": {"exposure_time_s": 0.05}},
            "observation_subblock": {
                "varying_keys": ["source.x_position_as", "metadata.only"],
                "trace": {"format": "csv", "path": str(trace_csv)},
                "validate": {
                    "require_contiguous_frame_index": True,
                    "require_monotonic_time": True,
                },
            },
            "outputs": {
                "outdir": str(tmp_path / "render_out"),
                "file_prefix": "obs_subblock",
                "frame_truth_format": "csv",
            },
            "noise": {"enabled": False},
        },
    }
    renderer_cfg_path = tmp_path / "renderer_recipe.json"
    renderer_cfg_path.write_text(json.dumps(renderer_cfg, indent=2), encoding="utf-8")

    render_result = render_recipe.generate_obs_subblock(
        config_path=renderer_cfg_path,
        run_name="render_from_generated_trace",
    )

    manifest_path = Path(render_result["artifacts"]["manifest_json"])
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["applied_varying_keys"] == list(APPLIED_V1_VARYING_KEYS)
    assert manifest["requested_varying_keys"] == ["source.x_position_as", "metadata.only"]
