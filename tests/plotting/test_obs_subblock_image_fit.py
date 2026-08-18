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


def test_plot_image_fit_uses_narrowed_residual_and_fixed_zscore_limits(
    tmp_path, monkeypatch
):
    recipe = _load_recipe_module()
    data_cube = np.arange(3 * 2 * 2, dtype=float).reshape((3, 2, 2))
    residual_cube = np.array(
        [
            [[1.0, -2.0], [0.5, 0.0]],
            [[4.0, -1.0], [0.25, -0.5]],
            [[0.0, 0.75], [-3.0, 2.0]],
        ],
        dtype=float,
    )
    model_cube = data_cube - residual_cube
    variance_cube = np.full_like(data_cube, 0.01)
    output_path = tmp_path / "image_fit_limits.png"

    from matplotlib.axes import Axes

    original_imshow = Axes.imshow
    residual_clims = []

    def recording_imshow(self, image, *args, **kwargs):
        if kwargs.get("cmap") == "RdBu_r":
            residual_clims.append((kwargs.get("vmin"), kwargs.get("vmax")))
        return original_imshow(self, image, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", recording_imshow)

    recipe._plot_image_fit(
        data_cube=data_cube,
        model_cube=model_cube,
        variance_cube=variance_cube,
        output_path=output_path,
    )

    assert output_path.exists()
    zlim = recipe.DEFAULT_IMAGE_FIT_ZSCORE_LIMIT
    assert residual_clims == [
        (-2.0, 2.0),
        (-zlim, zlim),
        (-2.0, 2.0),
        (-zlim, zlim),
        (-2.0, 2.0),
        (-zlim, zlim),
    ]


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


def test_variance_flooring_keeps_chi2_summary_finite():
    recipe = _load_recipe_module()
    data_cube = np.array(
        [
            [[0.0, -1.0], [2.0, 3.0]],
            [[4.0, 0.0], [1.0, -2.0]],
        ],
        dtype=float,
    )
    model_cube = np.zeros_like(data_cube)
    variance_cube = recipe._build_variance_cube(
        data_cube=data_cube,
        noise_model_cfg={"variance_model": "data"},
    )

    summary = recipe.summarize_framewise_chi2(
        data_cube,
        model_cube,
        variance_cube=variance_cube,
    )

    assert np.all(variance_cube >= 1.0)
    assert np.all(np.isfinite(summary.per_frame_chi2))
    assert np.all(np.isfinite(summary.per_frame_reduced_chi2))


def test_data_variance_uses_default_floor():
    recipe = _load_recipe_module()
    data_cube = np.array([[[0.0, 0.5, 1.0, 2.0]]], dtype=float)

    variance_cube = recipe._build_variance_cube(
        data_cube=data_cube,
        noise_model_cfg={"kind": "gaussian", "variance_model": "data"},
    )

    np.testing.assert_allclose(variance_cube, np.array([[[1.0, 1.0, 1.0, 2.0]]]))


def test_data_variance_uses_explicit_floor():
    recipe = _load_recipe_module()
    data_cube = np.array([[[0.0, 0.5, 1.0, 2.0]]], dtype=float)

    variance_cube = recipe._build_variance_cube(
        data_cube=data_cube,
        noise_model_cfg={
            "kind": "gaussian",
            "variance_model": "data",
            "variance_floor": 0.25,
        },
    )

    np.testing.assert_allclose(variance_cube, np.array([[[0.25, 0.5, 1.0, 2.0]]]))


@pytest.mark.parametrize("variance_floor", [0, -1, "not-a-number"])
def test_data_variance_rejects_invalid_floor(variance_floor):
    recipe = _load_recipe_module()
    data_cube = np.array([[[0.0, 0.5, 1.0, 2.0]]], dtype=float)

    with pytest.raises(ValueError, match="variance_floor"):
        recipe._build_variance_cube(
            data_cube=data_cube,
            noise_model_cfg={
                "kind": "gaussian",
                "variance_model": "data",
                "variance_floor": variance_floor,
            },
        )


def test_scalar_variance_ignores_variance_floor():
    recipe = _load_recipe_module()
    data_cube = np.array([[[0.0, 0.5, 1.0, 2.0]]], dtype=float)

    variance_cube = recipe._build_variance_cube(
        data_cube=data_cube,
        noise_model_cfg={
            "kind": "gaussian",
            "variance_model": "scalar",
            "scalar": 3.0,
            "variance_floor": -1.0,
        },
    )

    np.testing.assert_allclose(variance_cube, np.full_like(data_cube, 3.0))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_data_variance_rejects_nonfinite_data(bad_value):
    recipe = _load_recipe_module()
    data_cube = np.array([[[1.0, bad_value]]], dtype=float)

    with pytest.raises(ValueError, match="non-finite"):
        recipe._build_variance_cube(
            data_cube=data_cube,
            noise_model_cfg={"kind": "gaussian", "variance_model": "data"},
        )
