from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


RECIPE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "recipes"
    / "observation_subblock_inference.py"
)


def _load_recipe_module():
    spec = importlib.util.spec_from_file_location(
        "observation_subblock_inference_starting_guess_csv_tests",
        RECIPE_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _layout(recipe, keys: tuple[str, ...], n_frame: int):
    specs = tuple(
        recipe.ActiveKeySpec(
            canonical=key,
            address=recipe.parse_obs_subblock_key_address(key),
            kind="scalar",
        )
        for key in keys
    )
    return recipe.ActiveStateLayout(frame_specs=specs, shared_specs=(), n_frame=n_frame)


def test_starting_guess_csv_reads_mapped_columns(tmp_path):
    recipe = _load_recipe_module()
    csv_path = tmp_path / "starting_guess_prediction.csv"
    csv_path.write_text(
        "frame_index,source.x_position_as_linear_fit,source.y_position_as_linear_fit\n"
        "0,1.0,2.0\n"
        "1,1.5,2.5\n",
        encoding="utf-8",
    )

    matrix = recipe._load_starting_guess_frame_matrix(
        {
            "path": "starting_guess_prediction.csv",
            "columns": {
                "source.x_position_as": "source.x_position_as_linear_fit",
                "source.y_position_as": "source.y_position_as_linear_fit",
            },
        },
        layout=_layout(recipe, ("source.x_position_as", "source.y_position_as"), 2),
        config_path=tmp_path / "inference_config.json",
    )

    assert np.allclose(matrix, [[1.0, 2.0], [1.5, 2.5]])


def test_starting_guess_csv_missing_active_key_mapping_fails(tmp_path):
    recipe = _load_recipe_module()
    csv_path = tmp_path / "starting_guess_prediction.csv"
    csv_path.write_text(
        "frame_index,source.x_position_as_linear_fit\n0,1.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing: source.y_position_as"):
        recipe._load_starting_guess_frame_matrix(
            {
                "path": "starting_guess_prediction.csv",
                "columns": {
                    "source.x_position_as": "source.x_position_as_linear_fit",
                },
            },
            layout=_layout(recipe, ("source.x_position_as", "source.y_position_as"), 1),
            config_path=tmp_path / "inference_config.json",
        )


def test_starting_guess_csv_row_count_must_match_layout(tmp_path):
    recipe = _load_recipe_module()
    csv_path = tmp_path / "starting_guess_prediction.csv"
    csv_path.write_text(
        "frame_index,source.x_position_as_linear_fit\n0,1.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="row count"):
        recipe._load_starting_guess_frame_matrix(
            {
                "path": "starting_guess_prediction.csv",
                "columns": {
                    "source.x_position_as": "source.x_position_as_linear_fit",
                },
            },
            layout=_layout(recipe, ("source.x_position_as",), 2),
            config_path=tmp_path / "inference_config.json",
        )
