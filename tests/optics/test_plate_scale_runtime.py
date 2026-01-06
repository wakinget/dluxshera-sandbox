from __future__ import annotations

import jax.numpy as jnp

from dluxshera.core.binder import SheraThreePlaneBinder
from dluxshera.core.binder import SheraTwoPlaneBinder
from dluxshera.optics import builder
from dataclasses import replace

from dluxshera.optics.builder import clear_threeplane_optics_cache
from dluxshera.optics.builder import clear_twoplane_optics_cache
from dluxshera.optics.config import SHERA_TESTBED_CONFIG
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


def test_runtime_bindings_update_cached_optics():
    clear_threeplane_optics_cache()
    clear_twoplane_optics_cache()

    cfg = replace(
        SHERA_TESTBED_CONFIG,
        pupil_npix=64,
        psf_npix=64,
        oversample=1,
        n_lambda=1,
    )

    forward_spec, forward_store = make_forward_store(cfg)
    n_m1 = len(cfg.primary_noll_indices)
    n_m2 = len(cfg.secondary_noll_indices)

    coeffs_a = jnp.zeros(n_m1)
    coeffs_b = jnp.ones(n_m1)
    sec_coeffs_a = jnp.zeros(n_m2)
    sec_coeffs_b = jnp.ones(n_m2)
    plate_scale = forward_store.get("system.plate_scale_as_per_pix")

    store_a = forward_store.replace(
        {
            "primary.zernike_coeffs_nm": coeffs_a,
            "secondary.zernike_coeffs_nm": sec_coeffs_a,
            "system.plate_scale_as_per_pix": plate_scale,
        }
    )
    store_b = forward_store.replace(
        {
            "primary.zernike_coeffs_nm": coeffs_b,
            "secondary.zernike_coeffs_nm": sec_coeffs_b,
            "system.plate_scale_as_per_pix": plate_scale + 1e-3,
        }
    )

    optics_a = builder.build_shera_threeplane_optics(cfg, store_a, forward_spec)
    assert len(builder._THREEPLANE_CACHE) == 1
    optics_b = builder.build_shera_threeplane_optics(cfg, store_b, forward_spec)
    assert len(builder._THREEPLANE_CACHE) == 1

    assert not jnp.allclose(
        optics_a.p1_layers["m1_aperture"].coefficients,
        optics_b.p1_layers["m1_aperture"].coefficients,
    )
    assert not jnp.allclose(
        optics_a.p2_layers["m2_aperture"].coefficients,
        optics_b.p2_layers["m2_aperture"].coefficients,
    )
    assert optics_a.psf_pixel_scale != optics_b.psf_pixel_scale

    twoplane_cfg = SheraTwoPlaneConfig(
        pupil_npix=64,
        psf_npix=64,
        oversample=1,
        n_lambda=1,
        primary_noll_indices=cfg.primary_noll_indices,
    )
    twoplane_spec, twoplane_store = make_forward_store(twoplane_cfg)
    twoplane_store_a = twoplane_store.replace(
        {
            "primary.zernike_coeffs_nm": coeffs_a,
            "system.plate_scale_as_per_pix": twoplane_cfg.plate_scale_as_per_pix,
        }
    )
    twoplane_store_b = twoplane_store.replace(
        {
            "primary.zernike_coeffs_nm": coeffs_b,
            "system.plate_scale_as_per_pix": twoplane_cfg.plate_scale_as_per_pix
            + 1e-3,
        }
    )

    twoplane_a = builder.build_shera_twoplane_optics(
        twoplane_cfg, twoplane_store_a, twoplane_spec
    )
    assert len(builder._TWOPLANE_CACHE) == 1
    twoplane_b = builder.build_shera_twoplane_optics(
        twoplane_cfg, twoplane_store_b, twoplane_spec
    )
    assert len(builder._TWOPLANE_CACHE) == 1

    assert not jnp.allclose(
        twoplane_a.layers["aperture"].coefficients,
        twoplane_b.layers["aperture"].coefficients,
    )
    assert twoplane_a.psf_pixel_scale != twoplane_b.psf_pixel_scale
