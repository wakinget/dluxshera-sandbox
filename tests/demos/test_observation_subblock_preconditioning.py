"""Focused tests for observation sub-block theta preconditioning helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np



def _load_recipe_module():
    repo_root = Path(__file__).resolve().parents[2]
    recipe_path = repo_root / "examples" / "recipes" / "observation_subblock_inference.py"
    spec = importlib.util.spec_from_file_location(
        "observation_subblock_inference_recipe_precond_tests",
        recipe_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load recipe at {recipe_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_theta_preconditioning_bundle_shapes_match_theta_dim():
    recipe = _load_recipe_module()

    curvature = np.array([[4.0, 1.2], [1.2, 2.5]], dtype=float)

    def loss_fn(theta):
        return 0.5 * theta @ curvature @ theta

    theta_ref = np.array([0.25, -0.75], dtype=float)
    bundle = recipe._build_theta_preconditioning_bundle(
        loss_fn=loss_fn,
        theta_ref=theta_ref,
        base_lr=0.1,
        cfg={
            "damping": 1e-6,
            "eig_floor_rel": 1e-6,
            "eig_floor_abs": 1e-8,
            "lr_clip": [1e-4, 1.0],
        },
    )

    assert bundle.fim.shape == (theta_ref.size, theta_ref.size)
    assert bundle.eigvals.shape == (theta_ref.size,)
    assert bundle.eigvals_stable.shape == (theta_ref.size,)
    assert bundle.lr_vec.shape == (theta_ref.size,)
    assert np.all(np.isfinite(bundle.lr_vec))
    assert np.all(bundle.lr_vec > 0.0)


def test_pack_unpack_active_state_with_frame_and_shared_roundtrips():
    recipe = _load_recipe_module()

    layout = recipe.ActiveStateLayout(
        frame_specs=(
            recipe.ActiveKeySpec(
                canonical="source.x_position_as",
                address=recipe.ObsSubblockKeyAddress(base_key="source.x_position_as", index=None),
                kind="primitive",
            ),
            recipe.ActiveKeySpec(
                canonical="source.y_position_as",
                address=recipe.ObsSubblockKeyAddress(base_key="source.y_position_as", index=None),
                kind="primitive",
            ),
        ),
        shared_specs=(
            recipe.ActiveKeySpec(
                canonical="source.log_flux_total",
                address=recipe.ObsSubblockKeyAddress(base_key="source.log_flux_total", index=None),
                kind="primitive",
            ),
        ),
        n_frame=3,
    )
    state = recipe.ActiveState(
        frame=np.array([[0.1, -0.1], [0.2, -0.2], [0.3, -0.3]], dtype=float),
        shared=np.array([9.5], dtype=float),
    )

    theta = recipe._pack_active_state(layout, state)
    restored = recipe._unpack_active_state(layout, theta)

    assert theta.shape == (layout.theta_size,)
    np.testing.assert_allclose(np.asarray(restored.frame), state.frame)
    np.testing.assert_allclose(np.asarray(restored.shared), state.shared)
