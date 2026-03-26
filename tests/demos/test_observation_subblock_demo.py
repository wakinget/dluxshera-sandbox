"""Smoke test for the observation sub-block recipe."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from astropy.io import fits


def _load_recipe_module():
    repo_root = Path(__file__).resolve().parents[2]
    recipe_path = repo_root / "examples" / "recipes" / "observation_subblock.py"
    spec = importlib.util.spec_from_file_location("observation_subblock_recipe", recipe_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load recipe at {recipe_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_observation_subblock_recipe_runs_and_writes_artifacts(tmp_path):
    recipe = _load_recipe_module()
    repo_root = Path(__file__).resolve().parents[2]
    config_path = (
        repo_root
        / "examples"
        / "recipes"
        / "observation_subblock_template"
        / "prescription.yaml"
    )

    result = recipe.generate_obs_subblock(
        config_path=config_path,
        results_dir=tmp_path,
        run_name="smoke_obs_subblock",
    )

    cube_path = Path(result["artifacts"]["cube_fits"])
    truth_path = Path(result["artifacts"]["frame_truth_csv"])
    manifest_path = Path(result["artifacts"]["manifest_json"])

    assert cube_path.exists()
    assert truth_path.exists()
    assert manifest_path.exists()

    with fits.open(cube_path) as hdul:
        cube = hdul[0].data
    assert cube is not None
    assert cube.ndim == 3
    assert cube.shape[0] == 3

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in (
        "schema_version",
        "created_at",
        "generator",
        "frame_count",
        "varying_keys",
        "applied_varying_keys",
        "trace",
        "artifacts",
    ):
        assert key in manifest
    assert manifest["applied_varying_keys"] == [
        "source.x_position_as",
        "source.y_position_as",
        "source.position_angle_deg",
    ]


def test_observation_subblock_varying_keys_are_applied_and_validated(tmp_path):
    recipe = _load_recipe_module()

    trace_path = tmp_path / "frame_truth.csv"
    trace_path.write_text(
        "\n".join(
            [
                "frame_index,time_s,source.x_position_as,source.y_position_as,source.position_angle_deg,optics.plate_scale_as_per_pix",
                "0,0.0,0.0,0.0,90.0,0.11",
                "1,0.1,0.1,-0.1,90.2,0.12",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    base_cfg = {
        "system": {"preset": "SHERA_TESTBED_3P"},
        "experiment": {
            "kind": "observation_subblock",
            "seed": 42,
            "truth": {"source": {"exposure_time_s": 0.05}},
            "observation_subblock": {
                "trace": {"format": "csv", "path": str(trace_path)},
                "validate": {
                    "require_contiguous_frame_index": True,
                    "require_monotonic_time": True,
                },
            },
            "outputs": {
                "outdir": str(tmp_path / "out"),
                "file_prefix": "obs_subblock",
                "frame_truth_format": "csv",
            },
            "noise": {"enabled": False},
        },
    }

    cfg_missing_varying = tmp_path / "prescription_missing_varying.json"
    cfg_missing_varying.write_text(json.dumps(base_cfg, indent=2), encoding="utf-8")
    result_missing = recipe.generate_obs_subblock(
        config_path=cfg_missing_varying,
        dry_run=True,
    )
    assert result_missing["frame_count"] == 2

    cfg_custom = dict(base_cfg)
    cfg_custom["experiment"] = dict(base_cfg["experiment"])
    cfg_custom["experiment"]["observation_subblock"] = dict(
        base_cfg["experiment"]["observation_subblock"]
    )
    cfg_custom["experiment"]["observation_subblock"]["varying_keys"] = [
        "source.x_position_as",
        "optics.plate_scale_as_per_pix",
    ]
    cfg_custom_path = tmp_path / "prescription_custom_varying.json"
    cfg_custom_path.write_text(json.dumps(cfg_custom, indent=2), encoding="utf-8")
    result_custom = recipe.generate_obs_subblock(
        config_path=cfg_custom_path,
        dry_run=True,
    )
    assert result_custom["frame_count"] == 2

    cfg_invalid = dict(base_cfg)
    cfg_invalid["experiment"] = dict(base_cfg["experiment"])
    cfg_invalid["experiment"]["observation_subblock"] = dict(
        base_cfg["experiment"]["observation_subblock"]
    )
    cfg_invalid["experiment"]["observation_subblock"]["varying_keys"] = [
        "metadata.only",
    ]
    cfg_invalid_path = tmp_path / "prescription_invalid_varying.json"
    cfg_invalid_path.write_text(json.dumps(cfg_invalid, indent=2), encoding="utf-8")
    try:
        recipe.generate_obs_subblock(
            config_path=cfg_invalid_path,
            dry_run=True,
        )
    except ValueError as exc:
        assert "Unsupported observation-subblock varying key" in str(exc)
    else:
        raise AssertionError("Expected unsupported varying key to raise ValueError.")
