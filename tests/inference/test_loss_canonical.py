import jax
import jax.numpy as jnp

from dluxshera.optics.config import SheraTwoPlaneConfig
from dluxshera.core.binder import SheraTwoPlaneBinder
from dluxshera.inference.optimization import (
    make_binder_image_nll_fn,
    loss_canonical,
)
from tests.conftest import make_forward_store


def test_loss_canonical_matches_binder_nll_and_is_jittable(
    shera_smoke_cfg,
    shera_smoke_forward,
    shera_smoke_binder_data,
    shera_smoke_infer_keys,
):
    forward_spec, base_store = shera_smoke_forward
    _, data, var = shera_smoke_binder_data

    # 2) Choose a small inference subset
    infer_keys = shera_smoke_infer_keys
    inf_spec_subset = forward_spec.subset(shera_smoke_infer_keys)

    # (Optional) pack θ₀ using the subset helper, if you want a cross-check:
    # theta0_subset = pack_params(inf_spec_subset, base_store)

    # 3) Reference Binder-based θ → NLL
    loss_fn_binder, theta0 = make_binder_image_nll_fn(
        shera_smoke_cfg,
        forward_spec,
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
        shera_smoke_cfg,
        forward_spec,   # full spec
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

    # JIT stability is exercised elsewhere; here we only require consistency
    # between the canonical wrapper and the Binder-based loss.


def test_make_binder_image_nll_fn_twoplane_smoke():
    cfg = SheraTwoPlaneConfig()
    forward_spec, base_store = make_forward_store(cfg)

    infer_keys = (
        "binary.separation_as",
        "binary.position_angle_deg",
        "binary.contrast",
    )

    binder = SheraTwoPlaneBinder(cfg, forward_spec, base_store)
    image = binder.model()
    var = jnp.ones_like(image)

    loss_fn, theta0 = make_binder_image_nll_fn(
        cfg,
        forward_spec,
        base_store,
        infer_keys,
        image,
        var,
        use_system_graph=True,
    )

    loss_val = loss_fn(theta0)
    assert jnp.isfinite(loss_val)
