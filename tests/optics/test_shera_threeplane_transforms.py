import math
from contextlib import ExitStack
from importlib import resources
from pathlib import Path

import pytest

from dluxshera.components.sources import get_target_spec
from dluxshera.systems.three_plane import (
    SHERA_TESTBED_CONFIG,
    build_forward_spec_from_config,
)
from dluxshera.builders.optics import build_shera_threeplane_optics
from dluxshera.params.store import ParameterStore, refresh_derived
from dluxshera.params.transform_registry import TRANSFORMS
from dluxshera.utils.source_photometry import (
    build_wavelength_grid_m,
    derive_source_photometry,
    target_sed_root,
)


def _build_forward_model_store(cfg=SHERA_TESTBED_CONFIG):
    """Helper to build a forward-model spec + store from a config."""
    spec = build_forward_spec_from_config(cfg)
    store = ParameterStore.from_spec_defaults(spec)
    return spec, store


def test_forward_spec_includes_binary_astrometry_primitives():
    spec, store = _build_forward_model_store()

    expected_keys = {
        "source.x_position_as",
        "source.y_position_as",
        "source.separation_as",
        "source.position_angle_deg",
        "source.contrast",
        "source.log_flux_total",
    }

    assert expected_keys.issubset(set(spec.keys()))
    for key in expected_keys - {"source.log_flux_total"}:
        assert key in store


def test_forward_spec_zernike_coeffs_follow_noll_indices():
    cfg = SHERA_TESTBED_CONFIG
    spec = build_forward_spec_from_config(cfg)
    store = ParameterStore.from_spec_defaults(spec)

    n_m1 = len(cfg.primary_noll_indices)
    n_m2 = len(cfg.secondary_noll_indices)

    field_m1 = spec.get("optics.primary.zernike_coeffs_nm")
    field_m2 = spec.get("optics.secondary.zernike_coeffs_nm")

    assert field_m1.shape == (n_m1,)
    assert field_m2.shape == (n_m2,)

    assert store.get("optics.primary.zernike_coeffs_nm") == pytest.approx([0.0] * n_m1)
    assert store.get("optics.secondary.zernike_coeffs_nm") == pytest.approx([0.0] * n_m2)


def test_forward_spec_omits_zernike_when_basis_absent():
    cfg_empty = SHERA_TESTBED_CONFIG.replace(
        primary_noll_indices=(),
        secondary_noll_indices=(),
    )
    spec = build_forward_spec_from_config(cfg_empty)
    store = ParameterStore.from_spec_defaults(spec)

    assert "optics.primary.zernike_coeffs_nm" not in spec.keys()
    assert "optics.secondary.zernike_coeffs_nm" not in spec.keys()
    assert "optics.primary.zernike_coeffs_nm" not in store
    assert "optics.secondary.zernike_coeffs_nm" not in store


def test_system_focal_length_matches_analytic():
    """
    Check that optics.focal_length_m from the transform matches the
    analytic two-mirror formula used in the legacy model.
    """
    cfg = SHERA_TESTBED_CONFIG
    _, store = _build_forward_model_store(cfg)

    # Value from the transform registry
    f_eff = TRANSFORMS.compute("optics.focal_length_m", store)

    # Analytic reference using the same relation as SheraThreePlaneOptics
    f1 = cfg.m1_focal_length_m
    f2 = cfg.m2_focal_length_m
    sep = cfg.m1_m2_separation_m

    denom = (1.0 / f1) + (1.0 / f2) - sep / (f1 * f2)
    f_expected = 1.0 / denom

    assert math.isclose(f_eff, f_expected, rel_tol=5e-6, abs_tol=0.0)


def test_plate_scale_matches_legacy_optics():
    """
    Check that optics.plate_scale_as_per_pix from the transform matches
    the PSF pixel scale computed by SheraThreePlaneOptics.
    """
    cfg = SHERA_TESTBED_CONFIG
    _, store = _build_forward_model_store(cfg)

    plate_from_transform = TRANSFORMS.compute(
        "optics.plate_scale_as_per_pix", store
    )

    # Build the legacy optics system and use its psf_pixel_scale as reference
    optics = build_shera_threeplane_optics(cfg)
    plate_from_optics = float(optics.psf_pixel_scale)

    assert math.isclose(
        plate_from_transform,
        plate_from_optics,
        rel_tol=5e-6,
        abs_tol=0.0,
    )


def test_source_log_flux_total_matches_formula():
    """
    Check that source.log_flux_total from the transform matches
    explicit SED-backed photometry integration for Alpha Cen.
    """
    cfg = SHERA_TESTBED_CONFIG
    _, store = _build_forward_model_store(cfg)

    logF = TRANSFORMS.compute("source.log_flux_total", store)

    target = get_target_spec(str(store.get("source.target")))
    D = float(store.get("optics.m1_diameter_m"))
    wavelength_m = float(store.get("source.wavelength_m"))
    bandwidth_m = float(store.get("source.bandwidth_m"))
    n_lambda = int(store.get("source.n_lambda"))
    t_exp = float(store.get("source.exposure_time_s"))
    throughput = float(store.get("optics.throughput"))
    area_m2 = math.pi * (D / 2.0) ** 2
    wavelength_grid_m = build_wavelength_grid_m(
        wavelength_m=wavelength_m,
        bandwidth_m=bandwidth_m,
        n_lambda=n_lambda,
    )

    sed_root = target_sed_root()
    sed_a_ref = sed_root.joinpath(target.sed_a_file)
    sed_b_ref = sed_root.joinpath(target.sed_b_file)
    with ExitStack() as stack:
        sed_a_path = Path(stack.enter_context(resources.as_file(sed_a_ref)))
        sed_b_path = Path(stack.enter_context(resources.as_file(sed_b_ref)))
        expected = derive_source_photometry(
            wavelength_grid_m=wavelength_grid_m,
            bandwidth_m=bandwidth_m,
            collecting_area_m2=area_m2,
            exposure_time_s=t_exp,
            throughput=throughput,
            sed_a_path=sed_a_path,
            sed_b_path=sed_b_path,
            vmag_a=target.vmag_a,
            vmag_b=target.vmag_b,
        )

    assert expected.mode == "sed"
    assert math.isclose(logF, expected.log_flux_total, rel_tol=1e-12, abs_tol=0.0)


def test_refresh_derived_populates_forward_model_keys():
    spec, store = _build_forward_model_store()

    assert "optics.plate_scale_as_per_pix" not in store
    assert "source.log_flux_total" not in store

    refreshed = refresh_derived(
        store,
        spec,
        include_derived=True,
    )

    plate_scale = TRANSFORMS.compute("optics.plate_scale_as_per_pix", store)
    log_flux = TRANSFORMS.compute("source.log_flux_total", store)

    assert refreshed.get("optics.plate_scale_as_per_pix") == pytest.approx(plate_scale)
    assert refreshed.get("source.log_flux_total") == pytest.approx(log_flux)
