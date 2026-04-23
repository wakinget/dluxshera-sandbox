from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")


def _load_recipe_module():
    repo_root = Path(__file__).resolve().parents[2]
    recipe_path = repo_root / "examples" / "recipes" / "observation_subblock_inference.py"
    spec = importlib.util.spec_from_file_location(
        "observation_subblock_inference_image_fit_tests",
        recipe_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load recipe at {recipe_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plot_image_fit_writes_zscore_diagnostic(tmp_path):
    recipe = _load_recipe_module()
    data_cube = np.arange(3 * 4 * 5, dtype=float).reshape((3, 4, 5))
    model_cube = data_cube - 0.5
    variance_cube = np.full_like(data_cube, 4.0)
    output_path = tmp_path / "image_fit.png"

    recipe._plot_image_fit(
        data_cube=data_cube,
        model_cube=model_cube,
        variance_cube=variance_cube,
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_image_fit_handles_nonpositive_variance(tmp_path):
    recipe = _load_recipe_module()
    data_cube = np.ones((2, 4, 4), dtype=float)
    model_cube = np.zeros_like(data_cube)
    variance_cube = np.ones_like(data_cube)
    variance_cube[0, 0, 0] = 0.0
    variance_cube[1, 0, 1] = -1.0
    output_path = tmp_path / "image_fit_nonpositive_variance.png"

    recipe._plot_image_fit(
        data_cube=data_cube,
        model_cube=model_cube,
        variance_cube=variance_cube,
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_image_fit_rejects_shape_mismatch(tmp_path):
    recipe = _load_recipe_module()
    data_cube = np.ones((2, 4, 4), dtype=float)
    model_cube = np.ones((2, 4, 4), dtype=float)
    variance_cube = np.ones((2, 4, 5), dtype=float)

    with pytest.raises(ValueError, match="same shape"):
        recipe._plot_image_fit(
            data_cube=data_cube,
            model_cube=model_cube,
            variance_cube=variance_cube,
            output_path=tmp_path / "unused.png",
        )
