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


def _active_spec(recipe, key: str) -> object:
    return recipe.ActiveKeySpec(
        canonical=key,
        address=recipe.parse_obs_subblock_varying_keys([key])[0],
        kind="primitive",
    )


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
    assert bundle.fim_diag.shape == (theta_ref.size,)
    assert bundle.curvature_vec.shape == (theta_ref.size,)
    assert bundle.lr_vec.shape == (theta_ref.size,)
    assert np.all(np.isfinite(bundle.lr_vec))
    assert np.all(bundle.lr_vec > 0.0)


def test_theta_preconditioning_lr_vec_is_scale_only_not_base_lr_scaled(monkeypatch):
    recipe = _load_recipe_module()

    curvature = np.array([[4.0, 1.2], [1.2, 2.5]], dtype=float)
    monkeypatch.setattr(recipe, "fim_theta", lambda loss_fn, theta_ref: curvature)

    def loss_fn(theta):
        return 0.5 * theta @ recipe.jnp.asarray(curvature) @ theta

    theta_ref = np.array([0.25, -0.75], dtype=float)
    cfg = {
        "damping": 1e-6,
        "eig_floor_rel": 1e-6,
        "eig_floor_abs": 1e-8,
        "lr_clip": None,
    }
    bundle_low_lr = recipe._build_theta_preconditioning_bundle(
        loss_fn=loss_fn,
        theta_ref=theta_ref,
        base_lr=0.1,
        cfg=cfg,
    )
    bundle_high_lr = recipe._build_theta_preconditioning_bundle(
        loss_fn=loss_fn,
        theta_ref=theta_ref,
        base_lr=0.2,
        cfg=cfg,
    )

    np.testing.assert_allclose(bundle_low_lr.curvature_vec, np.diag(curvature))
    np.testing.assert_allclose(
        bundle_low_lr.lr_vec_unclipped,
        1.0 / (np.diag(curvature) + 1e-12),
    )
    np.testing.assert_allclose(bundle_low_lr.lr_vec, bundle_low_lr.lr_vec_unclipped)
    np.testing.assert_allclose(bundle_high_lr.lr_vec, bundle_low_lr.lr_vec)


def test_theta_preconditioning_matches_canonical_fim_diag_formula(monkeypatch):
    recipe = _load_recipe_module()

    curvature = np.array([[9.0, 4.0], [4.0, 16.0]], dtype=float)
    monkeypatch.setattr(recipe, "fim_theta", lambda loss_fn, theta_ref: curvature)

    def loss_fn(theta):
        return 0.5 * theta @ recipe.jnp.asarray(curvature) @ theta

    theta_ref = np.array([0.25, -0.75], dtype=float)
    bundle = recipe._build_theta_preconditioning_bundle(
        loss_fn=loss_fn,
        theta_ref=theta_ref,
        base_lr=0.1,
        cfg={
            "damping": 1e-6,
            "eig_floor_rel": 1e-6,
            "eig_floor_abs": 1e-8,
            "lr_clip": None,
        },
    )

    expected_curvature = np.array([9.0, 16.0])
    expected_lr_vec = 1.0 / (expected_curvature + 1e-12)
    np.testing.assert_allclose(bundle.curvature_vec, expected_curvature)
    np.testing.assert_allclose(bundle.lr_vec, expected_lr_vec)


def test_theta_preconditioning_reference_uses_truth_trace_when_available():
    recipe = _load_recipe_module()

    layout = recipe.ActiveStateLayout(
        frame_specs=(
            recipe.ActiveKeySpec(
                canonical="source.x_position_as",
                address=recipe.ObsSubblockKeyAddress(
                    base_key="source.x_position_as",
                    index=None,
                ),
                kind="primitive",
            ),
        ),
        shared_specs=(),
        n_frame=1,
    )
    theta0 = recipe.jnp.asarray([0.0])
    initial_state = recipe.ActiveState(
        frame=recipe.jnp.asarray([[0.0]]),
        shared=recipe.jnp.asarray([]),
    )
    trace = recipe.ObsSubblockTrace(
        rows=(
            {
                "frame_index": 0,
                "time_s": 0.0,
                "source.x_position_as": 0.123,
            },
        ),
        required_columns=("frame_index", "time_s", "source.x_position_as"),
        extra_columns=(),
        source_path=Path("truth.csv"),
    )
    truth = recipe._build_truth_frame_matrix(
        trace,
        layout=layout,
        base_store=recipe.ParameterStore.from_dict({}),
        n_frame=1,
    )

    theta_ref, source = recipe._resolve_theta_preconditioning_reference(
        layout=layout,
        theta0=theta0,
        initial_state=initial_state,
        truth=truth,
        reference_mode="truth_when_available",
    )

    assert source == "truth_trace"
    np.testing.assert_allclose(np.asarray(theta_ref), [0.123])


def test_truth_frame_matrix_fills_missing_active_key_from_resolved_store():
    recipe = _load_recipe_module()
    layout = recipe.ActiveStateLayout(
        frame_specs=(
            _active_spec(recipe, "source.x_position_as"),
            _active_spec(recipe, "source.log_flux_total"),
        ),
        shared_specs=(),
        n_frame=2,
    )
    trace = recipe.ObsSubblockTrace(
        rows=(
            {"frame_index": 0, "time_s": 0.0, "source.x_position_as": "0.1"},
            {"frame_index": 1, "time_s": 1.0, "source.x_position_as": "0.2"},
        ),
        required_columns=("frame_index", "time_s"),
        extra_columns=("source.x_position_as",),
        source_path=Path("truth.csv"),
    )

    truth = recipe._build_truth_frame_matrix(
        trace,
        layout=layout,
        base_store=recipe.ParameterStore.from_dict({"source.log_flux_total": 9.5}),
        n_frame=2,
    )

    assert truth.sources == {
        "source.x_position_as": "trace_csv",
        "source.log_flux_total": "resolved_store",
    }
    assert truth.complete is True
    np.testing.assert_allclose(truth.matrix, [[0.1, 9.5], [0.2, 9.5]])


def test_truth_frame_matrix_marks_missing_unrecoverable_key_unavailable(capsys):
    recipe = _load_recipe_module()
    layout = recipe.ActiveStateLayout(
        frame_specs=(
            _active_spec(recipe, "source.x_position_as"),
            _active_spec(recipe, "source.log_flux_total"),
        ),
        shared_specs=(),
        n_frame=1,
    )
    trace = recipe.ObsSubblockTrace(
        rows=({"frame_index": 0, "time_s": 0.0, "source.x_position_as": "0.1"},),
        required_columns=("frame_index", "time_s"),
        extra_columns=("source.x_position_as",),
        source_path=Path("truth.csv"),
    )

    truth = recipe._build_truth_frame_matrix(
        trace,
        layout=layout,
        base_store=recipe.ParameterStore.from_dict({}),
        n_frame=1,
    )

    assert truth.sources["source.x_position_as"] == "trace_csv"
    assert truth.sources["source.log_flux_total"] == "unavailable"
    assert truth.complete is False
    assert truth.unavailable_keys == ("source.log_flux_total",)
    assert np.isnan(truth.matrix[0, 1])
    assert "truth unavailable for active frame keys" in capsys.readouterr().out


def test_theta_preconditioning_reference_uses_mixed_truth_sources_when_complete():
    recipe = _load_recipe_module()
    layout = recipe.ActiveStateLayout(
        frame_specs=(
            _active_spec(recipe, "source.x_position_as"),
            _active_spec(recipe, "source.log_flux_total"),
        ),
        shared_specs=(),
        n_frame=2,
    )
    initial_state = recipe.ActiveState(
        frame=recipe.jnp.asarray([[0.0, 0.0], [0.0, 0.0]]),
        shared=recipe.jnp.asarray([]),
    )
    trace = recipe.ObsSubblockTrace(
        rows=(
            {"frame_index": 0, "time_s": 0.0, "source.x_position_as": "0.1"},
            {"frame_index": 1, "time_s": 1.0, "source.x_position_as": "0.2"},
        ),
        required_columns=("frame_index", "time_s"),
        extra_columns=("source.x_position_as",),
        source_path=Path("truth.csv"),
    )
    truth = recipe._build_truth_frame_matrix(
        trace,
        layout=layout,
        base_store=recipe.ParameterStore.from_dict({"source.log_flux_total": 9.5}),
        n_frame=2,
    )

    theta_ref, source = recipe._resolve_theta_preconditioning_reference(
        layout=layout,
        theta0=recipe.jnp.zeros(layout.theta_size),
        initial_state=initial_state,
        truth=truth,
        reference_mode="truth_when_available",
    )

    assert source == "truth_mixed"
    np.testing.assert_allclose(np.asarray(theta_ref), [0.1, 9.5, 0.2, 9.5])


def test_theta_preconditioning_reference_falls_back_for_incomplete_truth(capsys):
    recipe = _load_recipe_module()
    layout = recipe.ActiveStateLayout(
        frame_specs=(
            _active_spec(recipe, "source.x_position_as"),
            _active_spec(recipe, "source.log_flux_total"),
        ),
        shared_specs=(),
        n_frame=1,
    )
    initial_state = recipe.ActiveState(
        frame=recipe.jnp.asarray([[0.0, 0.0]]),
        shared=recipe.jnp.asarray([]),
    )
    trace = recipe.ObsSubblockTrace(
        rows=({"frame_index": 0, "time_s": 0.0, "source.x_position_as": "0.1"},),
        required_columns=("frame_index", "time_s"),
        extra_columns=("source.x_position_as",),
        source_path=Path("truth.csv"),
    )
    truth = recipe._build_truth_frame_matrix(
        trace,
        layout=layout,
        base_store=recipe.ParameterStore.from_dict({}),
        n_frame=1,
    )

    theta_ref, source = recipe._resolve_theta_preconditioning_reference(
        layout=layout,
        theta0=recipe.jnp.asarray([3.0, 4.0]),
        initial_state=initial_state,
        truth=truth,
        reference_mode="truth_when_available",
    )

    assert source == "initial_fallback_incomplete_truth"
    np.testing.assert_allclose(np.asarray(theta_ref), [3.0, 4.0])
    assert "preconditioning reference is incomplete" in capsys.readouterr().out


def test_preconditioned_sgd_effective_step_scales_linearly_with_base_lr(monkeypatch):
    recipe = _load_recipe_module()

    curvature = np.array([[3.0, 0.4], [0.4, 1.5]], dtype=float)
    monkeypatch.setattr(recipe, "fim_theta", lambda loss_fn, theta_ref: curvature)

    def loss_fn(theta):
        return 0.5 * theta @ recipe.jnp.asarray(curvature) @ theta

    theta0 = np.array([0.5, -0.25], dtype=float)
    cfg = {
        "damping": 1e-6,
        "eig_floor_rel": 1e-6,
        "eig_floor_abs": 1e-8,
        "lr_clip": None,
    }
    bundle_low_lr = recipe._build_theta_preconditioning_bundle(
        loss_fn=loss_fn,
        theta_ref=theta0,
        base_lr=0.1,
        cfg=cfg,
    )
    bundle_high_lr = recipe._build_theta_preconditioning_bundle(
        loss_fn=loss_fn,
        theta_ref=theta0,
        base_lr=0.2,
        cfg=cfg,
    )

    step_low_lr = recipe._optimizer_first_step(
        loss_fn=loss_fn,
        theta0=recipe.jnp.asarray(theta0),
        learning_rate=0.1,
        lr_vec=bundle_low_lr.lr_vec,
        optimizer_kind="sgd",
        optimizer_kwargs={},
    )
    step_high_lr = recipe._optimizer_first_step(
        loss_fn=loss_fn,
        theta0=recipe.jnp.asarray(theta0),
        learning_rate=0.2,
        lr_vec=bundle_high_lr.lr_vec,
        optimizer_kind="sgd",
        optimizer_kwargs={},
    )

    np.testing.assert_allclose(bundle_high_lr.lr_vec, bundle_low_lr.lr_vec)
    np.testing.assert_allclose(
        step_high_lr["delta0"],
        2.0 * step_low_lr["delta0"],
        rtol=1e-6,
    )


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
