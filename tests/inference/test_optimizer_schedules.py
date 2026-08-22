from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from dluxshera.inference.optimization import run_shera_gd
from dluxshera.inference.run_artifacts import load_meta, load_summary, load_trace
from dluxshera.inference.schedules import (
    build_scalar_lr_schedule,
    build_schedule_factor_history,
)


def test_no_schedule_returns_all_ones_and_disabled_metadata():
    history, meta = build_schedule_factor_history(None, n_iter=5)

    np.testing.assert_allclose(history, np.ones(5))
    assert meta["enabled"] is False
    assert meta["kind"] == "none"
    assert meta["normalized_config"] is None


def test_constant_schedule_defaults_to_all_ones():
    history, meta = build_schedule_factor_history({"kind": "constant"}, n_iter=4)
    schedule_fn, _ = build_scalar_lr_schedule({"kind": "constant"}, n_iter=4)

    np.testing.assert_allclose(history, np.ones(4))
    assert schedule_fn(0) == pytest.approx(1.0)
    assert schedule_fn(3) == pytest.approx(1.0)
    assert meta["enabled"] is True


def test_linear_warmup_schedule_ramps_and_then_stays_at_one():
    history, _ = build_schedule_factor_history(
        {
            "kind": "linear_warmup",
            "warmup_steps": 4,
            "start_factor": 0.25,
        },
        n_iter=7,
    )

    np.testing.assert_allclose(
        history,
        np.array([0.25, 0.4375, 0.625, 0.8125, 1.0, 1.0, 1.0]),
    )


@pytest.mark.parametrize(
    "cfg, match",
    [
        (
            {"kind": "linear_warmup", "warmup_steps": 0, "start_factor": 0.25},
            "warmup_steps",
        ),
        (
            {"kind": "linear_warmup", "warmup_steps": 3, "start_factor": 1.25},
            "start_factor",
        ),
    ],
)
def test_linear_warmup_schedule_validates_inputs(cfg, match):
    with pytest.raises(ValueError, match=match):
        build_schedule_factor_history(cfg, n_iter=6)


def test_piecewise_constant_schedule_uses_zero_based_boundaries():
    history, _ = build_schedule_factor_history(
        {
            "kind": "piecewise_constant",
            "boundaries": [2, 5],
            "factors": [1.0, 0.5, 0.1],
        },
        n_iter=8,
    )

    np.testing.assert_allclose(
        history,
        np.array([1.0, 1.0, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1]),
    )


def test_exponential_decay_supports_staircase_and_continuous_modes():
    staircase, _ = build_schedule_factor_history(
        {
            "kind": "exponential_decay",
            "decay_rate": 0.5,
            "transition_steps": 2,
            "staircase": True,
        },
        n_iter=5,
    )
    continuous, _ = build_schedule_factor_history(
        {
            "kind": "exponential_decay",
            "decay_rate": 0.5,
            "transition_steps": 2,
            "staircase": False,
        },
        n_iter=5,
    )

    np.testing.assert_allclose(staircase, np.array([1.0, 1.0, 0.5, 0.5, 0.25]))
    np.testing.assert_allclose(
        continuous,
        np.array([1.0, np.sqrt(0.5), 0.5, 0.5 * np.sqrt(0.5), 0.25]),
    )


def test_cosine_decay_starts_at_one_and_ends_at_min_factor():
    history, _ = build_schedule_factor_history(
        {"kind": "cosine_decay", "min_factor": 0.2},
        n_iter=5,
    )

    assert history[0] == pytest.approx(1.0)
    assert history[-1] == pytest.approx(0.2)


def test_linear_warmup_cosine_decay_warms_then_decays():
    history, _ = build_schedule_factor_history(
        {
            "kind": "linear_warmup_cosine_decay",
            "warmup_steps": 2,
            "start_factor": 0.0,
            "min_factor": 0.1,
        },
        n_iter=6,
    )

    np.testing.assert_allclose(history[:3], np.array([0.0, 0.5, 1.0]))
    assert history[-1] == pytest.approx(0.1)
    assert np.all(np.diff(history[2:]) <= 1.0e-12)


def test_invalid_schedule_kind_and_fields_raise_clear_errors():
    with pytest.raises(ValueError, match="Unsupported .*kind"):
        build_schedule_factor_history({"kind": "bad_schedule"}, n_iter=4)

    with pytest.raises(ValueError, match="Supported fields"):
        build_schedule_factor_history(
            {"kind": "cosine_decay", "min_factor": 0.1, "extra": 1},
            n_iter=4,
        )


def test_run_shera_gd_without_schedule_matches_constant_lr_behavior():
    theta0 = np.array([1.0], dtype=float)

    def loss_fn(theta):
        return 0.5 * jnp.sum(theta**2)

    theta_final, history = run_shera_gd(
        loss_fn=loss_fn,
        theta0=theta0,
        learning_rate=0.1,
        num_steps=2,
        optimizer_kind="sgd",
        return_artifacts=False,
        show_progress=False,
    )

    assert theta_final.shape == theta0.shape
    assert "scalar_lr" not in history
    np.testing.assert_allclose(np.asarray(history["loss"]).shape, (2,))


def test_run_shera_gd_uses_scheduled_scalar_lr_without_preconditioning():
    theta0 = np.array([1.0], dtype=float)
    scalar_lr_history = np.array([0.1, 0.2], dtype=float)
    schedule_factor_history = scalar_lr_history / 0.5

    def loss_fn(theta):
        return 0.5 * jnp.sum(theta**2)

    theta_final, history = run_shera_gd(
        loss_fn=loss_fn,
        theta0=theta0,
        learning_rate=0.5,
        scalar_lr_history=scalar_lr_history,
        schedule_factor_history=schedule_factor_history,
        schedule_meta={"enabled": True, "kind": "test"},
        num_steps=2,
        optimizer_kind="sgd",
        return_artifacts=False,
        show_progress=False,
    )

    assert float(theta_final[0]) == pytest.approx(0.72, abs=1e-7)
    np.testing.assert_allclose(history["scalar_lr"], scalar_lr_history)
    np.testing.assert_allclose(history["schedule_factor"], schedule_factor_history)


def test_run_shera_gd_uses_scheduled_scalar_lr_with_preconditioning():
    theta0 = np.array([1.0, 1.0], dtype=float)
    scalar_lr_history = np.array([0.1], dtype=float)
    lr_vec = np.array([2.0, 0.5], dtype=float)

    def loss_fn(theta):
        return 0.5 * jnp.sum(theta**2)

    theta_final, history = run_shera_gd(
        loss_fn=loss_fn,
        theta0=theta0,
        learning_rate=0.25,
        lr_vec=lr_vec,
        scalar_lr_history=scalar_lr_history,
        schedule_factor_history=scalar_lr_history / 0.25,
        schedule_meta={"enabled": True, "kind": "test"},
        num_steps=1,
        optimizer_kind="sgd",
        return_artifacts=False,
        show_progress=False,
    )

    np.testing.assert_allclose(theta_final, np.array([0.8, 0.95]), atol=1e-7)
    np.testing.assert_allclose(history["scalar_lr"], scalar_lr_history)


def test_run_shera_gd_applies_scalar_schedule_to_adam():
    theta0 = np.array([1.0, -0.5], dtype=float)
    scalar_lr_history = np.array([0.05, 0.05], dtype=float)

    def loss_fn(theta):
        return 0.5 * jnp.sum(theta**2)

    theta_scheduled, history_scheduled = run_shera_gd(
        loss_fn=loss_fn,
        theta0=theta0,
        learning_rate=0.2,
        scalar_lr_history=scalar_lr_history,
        schedule_factor_history=scalar_lr_history / 0.2,
        schedule_meta={"enabled": True, "kind": "constant"},
        num_steps=2,
        optimizer_kind="adam",
        return_artifacts=False,
        show_progress=False,
    )
    theta_constant, history_constant = run_shera_gd(
        loss_fn=loss_fn,
        theta0=theta0,
        learning_rate=0.05,
        num_steps=2,
        optimizer_kind="adam",
        return_artifacts=False,
        show_progress=False,
    )

    np.testing.assert_allclose(theta_scheduled, theta_constant, atol=1e-7)
    np.testing.assert_allclose(history_scheduled["loss"], history_constant["loss"], atol=1e-7)
    np.testing.assert_allclose(history_scheduled["scalar_lr"], scalar_lr_history)


def test_run_shera_gd_writes_lr_history_artifacts_when_schedule_is_configured(tmp_path):
    theta0 = np.array([1.0], dtype=float)
    scalar_lr_history = np.array([0.1, 0.1, 0.05], dtype=float)
    schedule_factor_history = scalar_lr_history / 0.2
    run_dir = tmp_path / "scheduled_run"

    def loss_fn(theta):
        return 0.5 * jnp.sum(theta**2)

    run_shera_gd(
        loss_fn=loss_fn,
        theta0=theta0,
        learning_rate=0.2,
        scalar_lr_history=scalar_lr_history,
        schedule_factor_history=schedule_factor_history,
        schedule_meta={
            "enabled": True,
            "kind": "piecewise_constant",
            "normalized_config": {"kind": "piecewise_constant"},
        },
        num_steps=3,
        optimizer_kind="sgd",
        run_dir=run_dir,
        return_artifacts=False,
        show_progress=False,
    )

    trace = load_trace(run_dir)
    meta = load_meta(run_dir)

    np.testing.assert_allclose(trace["base_lr"], np.full(3, 0.2))
    np.testing.assert_allclose(trace["scalar_lr"], scalar_lr_history)
    np.testing.assert_allclose(trace["schedule_factor"], schedule_factor_history)
    assert meta["optimizer"]["schedule"]["kind"] == "piecewise_constant"


def test_run_shera_gd_summary_final_loss_matches_returned_theta(tmp_path):
    theta0 = np.array([1.0], dtype=float)
    run_dir = tmp_path / "final_loss_alignment"

    def loss_fn(theta):
        return 0.5 * jnp.sum(theta**2)

    theta_final, history = run_shera_gd(
        loss_fn=loss_fn,
        theta0=theta0,
        learning_rate=0.1,
        num_steps=2,
        optimizer_kind="sgd",
        run_dir=run_dir,
        return_artifacts=False,
        show_progress=False,
    )

    expected_final_loss = float(loss_fn(theta_final))
    trace = load_trace(run_dir)
    summary = load_summary(run_dir)
    meta = load_meta(run_dir)

    assert summary["loss_init"] == pytest.approx(float(loss_fn(theta0)))
    assert summary["loss_final"] == pytest.approx(expected_final_loss)
    assert meta["optimizer"]["early_stopping"]["final_loss"] == pytest.approx(
        expected_final_loss
    )
    assert float(history["loss"][-1]) == pytest.approx(expected_final_loss)
    assert float(trace["loss"][-1]) == pytest.approx(expected_final_loss)
    np.testing.assert_allclose(trace["theta"][-1], theta_final)


def test_run_shera_gd_restore_best_final_loss_matches_returned_theta(tmp_path):
    theta0 = np.array([1.0], dtype=float)
    run_dir = tmp_path / "restore_best_final_loss_alignment"

    def loss_fn(theta):
        return jnp.sum(theta**2)

    theta_final, history = run_shera_gd(
        loss_fn=loss_fn,
        theta0=theta0,
        learning_rate=2.0,
        num_steps=4,
        optimizer_kind="sgd",
        run_dir=run_dir,
        return_artifacts=False,
        show_progress=False,
        early_stopping={
            "enabled": True,
            "patience": 1,
            "step_atol": 100.0,
            "restore_best": True,
        },
    )

    expected_final_loss = float(loss_fn(theta_final))
    trace = load_trace(run_dir)
    summary = load_summary(run_dir)
    meta = load_meta(run_dir)

    np.testing.assert_allclose(theta_final, theta0)
    assert history["early_stopping"]["stopped_early"] is True
    assert history["early_stopping"]["restored_best"] is True
    assert summary["loss_final"] == pytest.approx(expected_final_loss)
    assert meta["optimizer"]["early_stopping"]["final_loss"] == pytest.approx(
        expected_final_loss
    )
    assert float(trace["loss"][-1]) == pytest.approx(expected_final_loss)
    np.testing.assert_allclose(trace["theta"][-1], theta_final)


def test_run_shera_gd_truncates_scheduled_histories_after_early_stopping():
    theta0 = np.array([1.0], dtype=float)
    scalar_lr_history = np.array([0.2, 0.15, 0.1, 0.05, 0.025], dtype=float)
    schedule_factor_history = scalar_lr_history / 0.2

    def loss_fn(theta):
        return 0.5 * jnp.sum(theta**2)

    _theta, history = run_shera_gd(
        loss_fn=loss_fn,
        theta0=theta0,
        learning_rate=0.2,
        scalar_lr_history=scalar_lr_history,
        schedule_factor_history=schedule_factor_history,
        schedule_meta={"enabled": True, "kind": "test"},
        num_steps=5,
        optimizer_kind="sgd",
        return_artifacts=False,
        show_progress=False,
        early_stopping={
            "enabled": True,
            "patience": 2,
            "step_atol": 1.0,
        },
    )

    actual_steps = int(history["early_stopping"]["actual_n_iter"])
    assert actual_steps < 5
    assert history["scalar_lr"].shape == (actual_steps,)
    assert history["schedule_factor"].shape == (actual_steps,)
    np.testing.assert_allclose(history["scalar_lr"], scalar_lr_history[:actual_steps])
    np.testing.assert_allclose(
        history["schedule_factor"],
        schedule_factor_history[:actual_steps],
    )


def test_run_shera_gd_writes_truncated_lr_artifacts_after_early_stopping(tmp_path):
    theta0 = np.array([1.0], dtype=float)
    scalar_lr_history = np.array([0.2, 0.15, 0.1, 0.05], dtype=float)
    schedule_factor_history = scalar_lr_history / 0.2
    run_dir = tmp_path / "early_stop_scheduled_run"

    def loss_fn(theta):
        return 0.5 * jnp.sum(theta**2)

    run_shera_gd(
        loss_fn=loss_fn,
        theta0=theta0,
        learning_rate=0.2,
        scalar_lr_history=scalar_lr_history,
        schedule_factor_history=schedule_factor_history,
        schedule_meta={"enabled": True, "kind": "test"},
        num_steps=4,
        optimizer_kind="sgd",
        run_dir=run_dir,
        return_artifacts=False,
        show_progress=False,
        early_stopping={
            "enabled": True,
            "patience": 2,
            "step_atol": 1.0,
        },
    )

    trace = load_trace(run_dir)
    meta = load_meta(run_dir)
    actual_steps = int(meta["optimizer"]["early_stopping"]["actual_optimizer_steps"])
    assert actual_steps < 4
    assert trace["scalar_lr"].shape == (actual_steps,)
    assert trace["schedule_factor"].shape == (actual_steps,)
    assert meta["optimizer"]["actual_num_steps"] == actual_steps
    assert meta["optimizer"]["early_stopping"]["triggered"] is True
    np.testing.assert_allclose(trace["scalar_lr"], scalar_lr_history[:actual_steps])


def test_run_shera_gd_keeps_full_scheduled_histories_without_early_stopping():
    theta0 = np.array([1.0], dtype=float)
    scalar_lr_history = np.array([0.2, 0.15, 0.1], dtype=float)
    schedule_factor_history = scalar_lr_history / 0.2

    def loss_fn(theta):
        return 0.5 * jnp.sum(theta**2)

    _theta, history = run_shera_gd(
        loss_fn=loss_fn,
        theta0=theta0,
        learning_rate=0.2,
        scalar_lr_history=scalar_lr_history,
        schedule_factor_history=schedule_factor_history,
        schedule_meta={"enabled": True, "kind": "test"},
        num_steps=3,
        optimizer_kind="sgd",
        return_artifacts=False,
        show_progress=False,
    )

    assert history["loss"].shape == (3,)
    np.testing.assert_allclose(history["scalar_lr"], scalar_lr_history)
    np.testing.assert_allclose(history["schedule_factor"], schedule_factor_history)


def test_run_shera_gd_history_length_error_has_context():
    theta0 = np.array([1.0], dtype=float)

    def loss_fn(theta):
        return 0.5 * jnp.sum(theta**2)

    with pytest.raises(ValueError) as excinfo:
        run_shera_gd(
            loss_fn=loss_fn,
            theta0=theta0,
            learning_rate=0.2,
            scalar_lr_history=np.array([0.2], dtype=float),
            schedule_factor_history=np.array([1.0], dtype=float),
            schedule_meta={"enabled": True, "kind": "test"},
            num_steps=3,
            optimizer_kind="sgd",
            return_artifacts=False,
            show_progress=False,
        )

    message = str(excinfo.value)
    assert "scalar_lr_history" in message
    assert "expected actual_steps=3" in message
    assert "got 1" in message
    assert "reference_n_iter=3" in message
    assert "early_stopping=False" in message
