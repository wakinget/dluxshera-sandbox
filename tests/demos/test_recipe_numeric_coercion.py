from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_recipe(path_parts: tuple[str, ...], module_name: str):
    repo_root = Path(__file__).resolve().parents[2]
    recipe_path = repo_root.joinpath(*path_parts)
    spec = importlib.util.spec_from_file_location(module_name, recipe_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load recipe at {recipe_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _subblock_experiment_cfg() -> dict:
    return {
        "kind": "subblock_inference",
        "truth": {},
        "inference": {
            "data": {"cube": "cube.fits"},
            "active": {
                "frame_keys": ["source.x_position_as"],
                "shared_keys": [],
            },
            "init": {
                "frame": {
                    "mode": "shared_guess",
                    "values": {"source.x_position_as": "0.0"},
                },
                "shared": {},
            },
            "priors": {"frame": {}, "shared": {}},
            "temporal": {"frame_model": {"kind": "independent"}},
            "objective": {
                "kind": "nll",
                "reduce": "sum",
                "noise_model": {
                    "kind": "gaussian",
                    "variance_model": "scalar",
                    "scalar": "1e-3",
                },
            },
            "optimizer": {
                "kind": "adam",
                "base_lr": "1e-3",
                "n_iter": 3,
                "kwargs": {
                    "b1": "0.9",
                    "eps": "1e-8",
                    "eps_root": "1.0e-8",
                },
                "preconditioning": {
                    "enabled": True,
                    "damping": "5e-4",
                    "eig_floor_rel": "1e-6",
                    "eig_floor_abs": "1e-8",
                    "lr_clip": ["1e-4", "1.0"],
                },
            },
        },
    }


def test_subblock_validation_normalizes_scientific_optimizer_values():
    recipe = _load_recipe(
        ("examples", "recipes", "observation_subblock_inference.py"),
        "observation_subblock_inference_numeric_tests",
    )

    validated = recipe._validate_experiment_cfg(_subblock_experiment_cfg())
    optimizer = validated["inference"]["optimizer"]
    preconditioning = optimizer["preconditioning"]

    assert optimizer["base_lr"] == pytest.approx(1e-3)
    assert optimizer["kwargs"]["b1"] == pytest.approx(0.9)
    assert optimizer["kwargs"]["eps"] == pytest.approx(1e-8)
    assert optimizer["kwargs"]["eps_root"] == pytest.approx(1.0e-8)
    assert preconditioning["damping"] == pytest.approx(5e-4)
    assert preconditioning["eig_floor_rel"] == pytest.approx(1e-6)
    assert preconditioning["eig_floor_abs"] == pytest.approx(1e-8)
    assert preconditioning["lr_clip"] == pytest.approx([1e-4, 1.0])


@pytest.mark.parametrize("bad_value", ["abc", "", True])
def test_subblock_validation_rejects_bad_optimizer_eps(bad_value):
    recipe = _load_recipe(
        ("examples", "recipes", "observation_subblock_inference.py"),
        "observation_subblock_inference_bad_numeric_tests",
    )
    cfg = _subblock_experiment_cfg()
    cfg["inference"]["optimizer"]["kwargs"]["eps"] = bad_value

    with pytest.raises(ValueError, match="experiment.inference.optimizer.kwargs.eps"):
        recipe._validate_experiment_cfg(cfg)


def test_canonical_validation_normalizes_optimizer_kwargs():
    recipe = _load_recipe(
        ("examples", "recipes", "canonical_astrometry.py"),
        "canonical_astrometry_numeric_tests",
    )

    validated = recipe._validate_experiment(
        {
            "seed": 1,
            "infer_keys": ["source.x_position_as"],
            "optimizer": {
                "kind": "adam",
                "base_lr": "1e-3",
                "kwargs": {"eps": "1e-8", "b1": "0.9"},
            },
        }
    )

    assert validated["optimizer"]["base_lr"] == pytest.approx(1e-3)
    assert validated["optimizer"]["kwargs"]["eps"] == pytest.approx(1e-8)
    assert validated["optimizer"]["kwargs"]["b1"] == pytest.approx(0.9)


def test_prescribed_mc_run_spec_normalizes_optimizer_kwargs():
    recipe = _load_recipe(
        ("examples", "recipes", "prescribed_monte_carlo.py"),
        "prescribed_monte_carlo_numeric_tests",
    )

    resolved = recipe._resolve_run_spec_with_id(
        {
            "defaults": {
                "seed": 7,
                "optimizer": {
                    "kind": "sgd",
                    "n_iter": 5,
                    "base_lr": "1e-3",
                    "kwargs": {"momentum": "0.05"},
                },
            }
        },
        {},
        index=0,
        run_id_index=0,
    )

    assert resolved["optimizer"]["base_lr"] == pytest.approx(1e-3)
    assert resolved["optimizer"]["kwargs"]["momentum"] == pytest.approx(0.05)
