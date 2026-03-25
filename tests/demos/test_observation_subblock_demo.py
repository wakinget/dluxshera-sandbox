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
        "trace",
        "artifacts",
    ):
        assert key in manifest
