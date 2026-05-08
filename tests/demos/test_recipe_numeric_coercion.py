from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
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
                "frame_reduce": "sum",
                "subblock_reduce": "sum",
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
                "schedule": {
                    "kind": "linear_warmup",
                    "warmup_steps": "2",
                    "start_factor": "0.25",
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
    objective = validated["inference"]["objective"]
    preconditioning = optimizer["preconditioning"]

    assert objective["frame_reduce"] == "sum"
    assert objective["subblock_reduce"] == "sum"
    assert optimizer["base_lr"] == pytest.approx(1e-3)
    assert optimizer["kwargs"]["b1"] == pytest.approx(0.9)
    assert optimizer["kwargs"]["eps"] == pytest.approx(1e-8)
    assert optimizer["kwargs"]["eps_root"] == pytest.approx(1.0e-8)
    assert optimizer["schedule"]["kind"] == "linear_warmup"
    assert optimizer["schedule"]["warmup_steps"] == 2
    assert optimizer["schedule"]["start_factor"] == pytest.approx(0.25)
    assert preconditioning["method"] == "auto"
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


def test_subblock_validation_rejects_unknown_preconditioning_method():
    recipe = _load_recipe(
        ("examples", "recipes", "observation_subblock_inference.py"),
        "observation_subblock_inference_bad_precond_method_tests",
    )
    cfg = _subblock_experiment_cfg()
    cfg["inference"]["optimizer"]["preconditioning"]["method"] = "not_a_method"

    with pytest.raises(ValueError, match="preconditioning.method"):
        recipe._validate_experiment_cfg(cfg)


def test_subblock_validation_rejects_bad_schedule_kind():
    recipe = _load_recipe(
        ("examples", "recipes", "observation_subblock_inference.py"),
        "observation_subblock_inference_bad_schedule_tests",
    )
    cfg = _subblock_experiment_cfg()
    cfg["inference"]["optimizer"]["schedule"]["kind"] = "bad_schedule"

    with pytest.raises(ValueError, match="optimizer.schedule.kind"):
        recipe._validate_experiment_cfg(cfg)


def test_subblock_validation_maps_legacy_reduce_to_frame_and_subblock_defaults():
    recipe = _load_recipe(
        ("examples", "recipes", "observation_subblock_inference.py"),
        "observation_subblock_inference_legacy_reduce_tests",
    )
    cfg = _subblock_experiment_cfg()
    objective = cfg["inference"]["objective"]
    objective.pop("frame_reduce")
    objective.pop("subblock_reduce")
    objective["reduce"] = "mean"

    validated = recipe._validate_experiment_cfg(cfg)

    assert validated["inference"]["objective"]["frame_reduce"] == "mean"
    assert validated["inference"]["objective"]["subblock_reduce"] == "sum"
    assert "reduce" not in validated["inference"]["objective"]


def test_subblock_validation_prefers_new_reduction_fields_over_legacy_reduce():
    recipe = _load_recipe(
        ("examples", "recipes", "observation_subblock_inference.py"),
        "observation_subblock_inference_reduce_precedence_tests",
    )
    cfg = _subblock_experiment_cfg()
    cfg["inference"]["objective"]["reduce"] = "sum"
    cfg["inference"]["objective"]["frame_reduce"] = "mean"
    cfg["inference"]["objective"]["subblock_reduce"] = "mean"

    validated = recipe._validate_experiment_cfg(cfg)

    assert validated["inference"]["objective"]["frame_reduce"] == "mean"
    assert validated["inference"]["objective"]["subblock_reduce"] == "mean"


def test_subblock_term_reduction_helper_respects_sum_and_mean():
    recipe = _load_recipe(
        ("examples", "recipes", "observation_subblock_inference.py"),
        "observation_subblock_inference_reduction_helper_tests",
    )

    per_frame_terms = recipe.jnp.asarray(np.array([2.0, 4.0, 8.0], dtype=float))

    assert float(recipe._reduce_subblock_terms(per_frame_terms, reduce="sum")) == pytest.approx(
        14.0
    )
    assert float(
        recipe._reduce_subblock_terms(per_frame_terms, reduce="mean")
    ) == pytest.approx(14.0 / 3.0)


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
