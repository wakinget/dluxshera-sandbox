from dataclasses import replace

import jax.numpy as jnp

from dluxshera.builders import optics as builder
from dluxshera.components.optics import (
    build_threeplane_optics_contract,
    build_twoplane_optics_contract,
)
from dluxshera.systems.three_plane import SHERA_TESTBED_CONFIG
from dluxshera.systems.two_plane import SheraTwoPlaneConfig
from tests.conftest import make_forward_store


def test_threeplane_contract_structural_and_bindings():
    cfg = SHERA_TESTBED_CONFIG
    contract = build_threeplane_optics_contract(cfg)

    assert contract.get("system.pupil_npix").structural is True
    assert contract.get("system.m1_focal_length_m").structural is True
    assert contract.get("system.dp_path").structural is True
    assert contract.get("imaging.throughput").structural is False

    assert contract.get("primary.zernike_coeffs_nm").binding == "p1_layers.m1_aperture.coefficients"
    assert contract.get("secondary.zernike_coeffs_nm").binding == "p2_layers.m2_aperture.coefficients"
    assert contract.get("system.plate_scale_as_per_pix").binding == "psf_pixel_scale"


def test_twoplane_contract_structural_and_bindings():
    cfg = SheraTwoPlaneConfig(primary_noll_indices=(4, 5, 6))
    contract = build_twoplane_optics_contract(cfg)

    assert contract.get("system.pupil_npix").structural is True
    assert contract.get("system.primary_noll_indices").structural is True
    assert contract.get("system.plate_scale_as_per_pix").structural is False

    assert contract.get("primary.zernike_coeffs_nm").binding == "layers.aperture.coefficients"
    assert contract.get("system.plate_scale_as_per_pix").binding == "psf_pixel_scale"


def test_structural_hash_changes_for_structural_field():
    cfg = SHERA_TESTBED_CONFIG
    tweaked = replace(cfg, pupil_npix=cfg.pupil_npix + 1)

    assert builder.structural_hash_from_config(cfg) != builder.structural_hash_from_config(tweaked)


def test_runtime_binding_updates_cached_optics_fields():
    builder.clear_threeplane_optics_cache()

    cfg = replace(SHERA_TESTBED_CONFIG, pupil_npix=64, psf_npix=64, oversample=1, n_lambda=1)
    forward_spec, forward_store = make_forward_store(cfg)

    plate_scale = forward_store.get("system.plate_scale_as_per_pix")
    n_m1 = len(cfg.primary_noll_indices)

    store_a = forward_store.replace(
        {
            "primary.zernike_coeffs_nm": jnp.zeros(n_m1),
            "system.plate_scale_as_per_pix": plate_scale,
        }
    )
    store_b = forward_store.replace(
        {
            "primary.zernike_coeffs_nm": jnp.ones(n_m1),
            "system.plate_scale_as_per_pix": plate_scale + 1e-3,
        }
    )

    optics_a = builder.build_shera_threeplane_optics(cfg, store_a, forward_spec)
    optics_b = builder.build_shera_threeplane_optics(cfg, store_b, forward_spec)

    assert not jnp.allclose(
        optics_a.p1_layers["m1_aperture"].coefficients,
        optics_b.p1_layers["m1_aperture"].coefficients,
    )
    assert optics_a.psf_pixel_scale != optics_b.psf_pixel_scale
