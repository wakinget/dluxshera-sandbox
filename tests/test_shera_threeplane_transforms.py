import math

import pytest

import math
from dataclasses import replace

import pytest

from dluxshera.optics.config import SHERA_TESTBED_CONFIG
from dluxshera.optics.builder import build_shera_threeplane_optics
from dluxshera.params.spec import build_forward_model_spec_from_config
from dluxshera.params.store import ParameterStore, refresh_derived
from dluxshera.params.transforms import DEFAULT_SYSTEM_ID, TRANSFORMS

# IMPORTANT: ensure transforms are registered at import time
import dluxshera.params.shera_threeplane_transforms  # noqa: F401


def _build_forward_model_store(cfg=SHERA_TESTBED_CONFIG):
    """Helper to build a forward-model spec + store from a config."""
    spec = build_forward_model_spec_from_config(cfg)
    store = ParameterStore.from_spec_defaults(spec)
    return spec, store


def test_forward_spec_includes_binary_astrometry_primitives():
    spec, store = _build_forward_model_store()

    expected_keys = {
        "binary.x_position_as",
        "binary.y_position_as",
        "binary.separation_as",
        "binary.position_angle_deg",
        "binary.contrast",
    }

    assert expected_keys.issubset(set(spec.keys()))
    for key in expected_keys:
        assert key in store


def test_forward_spec_zernike_coeffs_follow_noll_indices():
    cfg = SHERA_TESTBED_CONFIG
    spec = build_forward_model_spec_from_config(cfg)
    store = ParameterStore.from_spec_defaults(spec)

    n_m1 = len(cfg.primary_noll_indices)
    n_m2 = len(cfg.secondary_noll_indices)

    field_m1 = spec.get("primary.zernike_coeffs")
    field_m2 = spec.get("secondary.zernike_coeffs")

    assert field_m1.shape == (n_m1,)
    assert field_m2.shape == (n_m2,)

    assert store.get("primary.zernike_coeffs") == tuple([0.0] * n_m1)
    assert store.get("secondary.zernike_coeffs") == tuple([0.0] * n_m2)


def test_forward_spec_omits_zernike_when_basis_absent():
    cfg_empty = replace(
        SHERA_TESTBED_CONFIG,
        primary_noll_indices=(),
        secondary_noll_indices=(),
    )
    spec = build_forward_model_spec_from_config(cfg_empty)
    store = ParameterStore.from_spec_defaults(spec)

    assert "primary.zernike_coeffs" not in spec.keys()
    assert "secondary.zernike_coeffs" not in spec.keys()
    assert "primary.zernike_coeffs" not in store
    assert "secondary.zernike_coeffs" not in store


def test_system_focal_length_matches_analytic():
    """
    Check that system.focal_length_m from the transform matches the
    analytic two-mirror formula used in the legacy model.
    """
    cfg = SHERA_TESTBED_CONFIG
    _, store = _build_forward_model_store(cfg)

    # Value from the transform registry
    f_eff = TRANSFORMS.compute("system.focal_length_m", store)

    # Analytic reference using the same relation as SheraThreePlaneSystem
    f1 = cfg.m1_focal_length_m
    f2 = cfg.m2_focal_length_m
    sep = cfg.m1_m2_separation_m

    denom = (1.0 / f1) + (1.0 / f2) - sep / (f1 * f2)
    f_expected = 1.0 / denom

    assert math.isclose(f_eff, f_expected, rel_tol=1e-12, abs_tol=0.0)


def test_plate_scale_matches_legacy_optics():
    """
    Check that system.plate_scale_as_per_pix from the transform matches
    the PSF pixel scale computed by SheraThreePlaneSystem.
    """
    cfg = SHERA_TESTBED_CONFIG
    _, store = _build_forward_model_store(cfg)

    plate_from_transform = TRANSFORMS.compute(
        "system.plate_scale_as_per_pix", store
    )

    # Build the legacy optics system and use its psf_pixel_scale as reference
    optics = build_shera_threeplane_optics(cfg)
    plate_from_optics = float(optics.psf_pixel_scale)

    assert math.isclose(
        plate_from_transform,
        plate_from_optics,
        rel_tol=1e-10,
        abs_tol=0.0,
    )


def test_binary_log_flux_total_matches_formula():
    """
    Check that binary.log_flux_total from the transform matches the
    simple collecting-area × band × exposure × throughput formula.
    """
    cfg = SHERA_TESTBED_CONFIG
    _, store = _build_forward_model_store(cfg)

    logF = TRANSFORMS.compute("binary.log_flux_total", store)

    D = float(store.get("system.m1_diameter_m"))
    bandwidth_m = float(store.get("band.bandwidth_m"))
    t_exp = float(store.get("imaging.exposure_time_s"))
    throughput = float(store.get("imaging.throughput"))
    flux_density = float(store.get("binary.spectral_flux_density"))

    area = math.pi * (D / 2.0) ** 2
    total_flux = flux_density * bandwidth_m * area * t_exp * throughput
    expected_logF = math.log10(total_flux)

    assert math.isclose(logF, expected_logF, rel_tol=1e-12, abs_tol=0.0)


def test_refresh_derived_populates_forward_model_keys():
    spec, store = _build_forward_model_store()

    assert "system.plate_scale_as_per_pix" not in store
    assert "binary.log_flux_total" not in store

    refreshed = refresh_derived(
        store,
        spec,
        TRANSFORMS,
        system_id=DEFAULT_SYSTEM_ID,
        include_derived=True,
    )

    plate_scale = TRANSFORMS.compute("system.plate_scale_as_per_pix", store)
    log_flux = TRANSFORMS.compute("binary.log_flux_total", store)

    assert refreshed.get("system.plate_scale_as_per_pix") == pytest.approx(plate_scale)
    assert refreshed.get("binary.log_flux_total") == pytest.approx(log_flux)
