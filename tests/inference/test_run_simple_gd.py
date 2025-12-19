import jax.numpy as jnp
from dluxshera.inference.optimization import run_simple_gd


def test_run_simple_gd_converges_on_quadratic():
    # Simple convex quadratic: minimum at theta_true
    theta_true = jnp.array([1.5, -0.5])

    def loss_fn(theta):
        return 0.5 * jnp.sum((theta - theta_true) ** 2)

    theta0 = jnp.array([0.0, 0.0])

    theta_final, history = run_simple_gd(
        loss_fn,
        theta0,
        learning_rate=0.5,
        num_steps=50,
    )

    # Loss should strictly decrease overall
    loss_start = float(history["loss"][0])
    loss_end = float(history["loss"][-1])
    assert loss_end < loss_start

    # Final theta should be closer than where it started
    init_err = jnp.linalg.norm(theta0 - theta_true)
    err = jnp.linalg.norm(theta_final - theta_true)
    assert err < init_err
    assert err < 1e-1
