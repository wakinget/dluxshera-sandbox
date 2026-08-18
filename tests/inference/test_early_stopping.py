import jax.numpy as jnp
from dluxshera.inference.optimization import run_shera_gd


def test_early_stopping_disabled_runs_full_iterations():
    loss_fn = lambda t: jnp.sum((t - 1.0) ** 2)
    _, hist = run_shera_gd(loss_fn=loss_fn, theta0=jnp.array([5.0]), learning_rate=0.1, num_steps=8, return_artifacts=False, show_progress=False)
    assert hist["loss"].shape[0] == 8


def test_early_stopping_loss_rtol_patience_triggers():
    loss_fn = lambda t: jnp.array(1.0)
    _, hist = run_shera_gd(
        loss_fn=loss_fn,
        theta0=jnp.array([0.0]),
        learning_rate=0.1,
        num_steps=20,
        return_artifacts=False,
        show_progress=False,
        early_stopping={"enabled": True, "min_iter": 0, "patience": 3, "loss_rtol": 1e-12},
    )
    es = hist["early_stopping"]
    assert es["stopped_early"] is True
    assert hist["loss"].shape[0] < 20


def test_early_stopping_min_iter_is_respected():
    loss_fn = lambda t: jnp.array(1.0)
    _, hist = run_shera_gd(
        loss_fn=loss_fn,
        theta0=jnp.array([0.0]),
        learning_rate=0.1,
        num_steps=20,
        return_artifacts=False,
        show_progress=False,
        early_stopping={"enabled": True, "min_iter": 10, "patience": 3, "loss_rtol": 1e-12},
    )
    assert hist["early_stopping"]["actual_n_iter"] >= 10


def test_early_stopping_non_finite_loss_records_reason():
    loss_fn = lambda t: jnp.where(t[0] > 0.5, jnp.array(float("nan")), jnp.sum((t - 1.0) ** 2))
    _, hist = run_shera_gd(
        loss_fn=loss_fn,
        theta0=jnp.array([0.0]),
        learning_rate=1.0,
        num_steps=10,
        return_artifacts=False,
        show_progress=False,
        early_stopping={"enabled": True, "require_finite_loss": True},
    )
    assert hist["early_stopping"]["stop_reason"] == "non_finite_loss"
