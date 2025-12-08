import jax
import jax.numpy as jnp

from dluxshera.optics.config import SheraThreePlaneConfig, SHERA_TESTBED_CONFIG
from dluxshera.params.spec import build_inference_spec_basic
from dluxshera.params.store import ParameterStore
from dluxshera.params.packing import pack_params
from dluxshera.core.binder import SheraThreePlaneBinder
from dluxshera.inference.optimization import (
    make_binder_image_nll_fn,
    loss_canonical,
)


def _make_minimal_inference_setup(cfg: SheraThreePlaneConfig):
    """
    Build a simple, self-consistent inference spec + store + synthetic data
    suitable for testing the canonical θ-space loss.

    This mirrors the structure used in the existing image NLL and FIM tests,
    but keeps everything local to this file for clarity.
    """
    # Baseline inference spec & store
    inference_spec = build_inference_spec_basic()
    base_store = ParameterStore.from_spec_defaults(inference_spec)

    # Construct Binder and generate a synthetic "truth" image from the baseline
    binder = SheraThreePlaneBinder(cfg, inference_spec, base_store)
    image_truth = binder.forward(base_store)

    # Use the noiseless model PSF as data; simple homogeneous variance
    data = image_truth
    var = jnp.ones_like(data)

    return inference_spec, base_store, data, var


def test_loss_canonical_matches_binder_nll_and_is_jittable():
    # 1) Build minimal setup
    cfg = SHERA_TESTBED_CONFIG
    inference_spec, base_store, data, var = _make_minimal_inference_setup(cfg)

    # 2) Choose a small inference subset
    infer_keys = ("binary.separation_as", "binary.x_position", "binary.y_position")
    inf_spec_subset = inference_spec.subset(infer_keys)

    # (Optional) pack θ₀ using the subset helper, if you want a cross-check:
    # theta0_subset = pack_params(inf_spec_subset, base_store)

    # 3) Reference Binder-based θ → NLL
    loss_fn_binder, theta0 = make_binder_image_nll_fn(
        cfg,
        inference_spec,
        base_store,
        infer_keys,
        data,
        var,
        noise_model="gaussian",
        reduce="sum",
    )
    loss_ref = loss_fn_binder(theta0)

    # 4) Canonical loss should agree with Binder NLL for the same θ
    loss_val = loss_canonical(
        theta0,
        cfg,
        inference_spec,   # full spec
        infer_keys,       # explicit subset
        base_store,
        data,
        var,
        noise_model="gaussian",
        reduce="sum",
    )

    assert jnp.allclose(loss_val, loss_ref, rtol=1e-6, atol=1e-6)

    # 5) Gradient + JIT smoke tests (as before)
    # JIT + grad tests should be done on the Binder-based θ → NLL,
    #    which *is* the JAX-friendly canonical loss.
    def loss_wrapped(th):
        return loss_fn_binder(th)

    grad_val = jax.grad(loss_wrapped)(theta0)
    assert grad_val.shape == theta0.shape
    assert jnp.all(jnp.isfinite(grad_val))

    loss_jitted = jax.jit(loss_wrapped)
    loss_jit_val = loss_jitted(theta0)
    assert jnp.allclose(loss_jit_val, loss_ref, rtol=1e-5, atol=1e-3)
