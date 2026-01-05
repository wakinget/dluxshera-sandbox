from __future__ import annotations

import jax.numpy as jnp

from dluxshera.core.binder import SheraThreePlaneBinder
from dluxshera.core.binder import SheraTwoPlaneBinder
from dluxshera.optics import builder
from dluxshera.optics.builder import clear_twoplane_optics_cache
from dluxshera.optics.config import SheraTwoPlaneConfig
from dluxshera.params.packing import pack_params as store_pack_params
from dluxshera.params.packing import unpack_params as store_unpack_params
from dluxshera.params.spec import build_inference_spec_basic
from tests.conftest import make_forward_store


def test_plate_scale_updates_psf(shera_smoke_cfg, shera_smoke_updates):
    forward_spec, forward_store = make_forward_store(
        shera_smoke_cfg, updates=shera_smoke_updates
    )
    binder = SheraThreePlaneBinder(
        shera_smoke_cfg,
        forward_spec,
        forward_store,
        use_system_graph=False,
    )

    inference_spec = build_inference_spec_basic()
    sub_spec = inference_spec.subset(["system.plate_scale_as_per_pix"])
    theta = store_pack_params(sub_spec, forward_store)

    delta = jnp.array([1e-3])
    store0 = store_unpack_params(sub_spec, theta, forward_store)
    store1 = store_unpack_params(sub_spec, theta + delta, forward_store)

    assert store0.get("system.plate_scale_as_per_pix") != store1.get(
        "system.plate_scale_as_per_pix"
    )

    psf0 = binder.model(store0)
    psf1 = binder.model(store1)

    diff = jnp.max(jnp.abs(psf1 - psf0))
    assert diff > 1e-9


def test_twoplane_plate_scale_updates_without_cache_rebuild():
    clear_twoplane_optics_cache()

    cfg = SheraTwoPlaneConfig(
        pupil_npix=64,
        psf_npix=64,
        oversample=1,
        n_lambda=1,
    )
    forward_spec, forward_store = make_forward_store(cfg)
    binder = SheraTwoPlaneBinder(
        cfg,
        forward_spec,
        forward_store,
        use_system_graph=False,
    )

    store_a = forward_store.replace(
        {"system.plate_scale_as_per_pix": cfg.plate_scale_as_per_pix}
    )
    store_b = forward_store.replace(
        {"system.plate_scale_as_per_pix": cfg.plate_scale_as_per_pix + 1e-3}
    )

    psf_a = binder.model(store_a)
    assert len(builder._TWOPLANE_CACHE) == 1

    psf_b = binder.model(store_b)
    assert len(builder._TWOPLANE_CACHE) == 1

    diff = jnp.max(jnp.abs(psf_b - psf_a))
    assert diff > 1e-9
