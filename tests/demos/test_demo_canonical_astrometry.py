"""Smoke test for the canonical astrometry recipe."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def load_recipe_module():
    repo_root = Path(__file__).resolve().parents[2]
    recipe_path = repo_root / "examples" / "recipes" / "canonical_astrometry.py"
    spec = importlib.util.spec_from_file_location("canonical_astrometry_recipe", recipe_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load recipe at {recipe_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canonical_astrometry_recipe_runs(tmp_path):
    recipe = load_recipe_module()
    recipe.main(fast=True, save_plots=True, add_noise=False, results_dir=tmp_path)
    assert (tmp_path / "initial_psf_comparison.png").exists()
