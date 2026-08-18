"""Smoke test for the two-plane astrometry recipe."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path


def load_recipe_module():
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg", force=True)

    repo_root = Path(__file__).resolve().parents[2]
    recipe_path = repo_root / "examples" / "recipes" / "twoplane_astrometry.py"
    spec = importlib.util.spec_from_file_location("twoplane_astrometry_recipe", recipe_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load recipe at {recipe_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_twoplane_astrometry_recipe_runs(tmp_path):
    recipe = load_recipe_module()
    config_path = tmp_path / "twoplane_astrometry_config.json"
    config_path.write_text(
        json.dumps(
            {
                "experiment": {
                    "add_noise": False,
                    "outputs": {"save_plots": True},
                    "optimizer": {"kind": "gd", "n_iter_fast": 1, "base_lr": 0.1},
                    "init": {"mode": "truth"},
                }
            }
        ),
        encoding="utf-8",
    )

    recipe.main(
        config_path=config_path,
        fast=True,
        results_dir=tmp_path,
        use_eigen=False,
    )
    assert (tmp_path / "initial_psf_comparison.png").exists()
