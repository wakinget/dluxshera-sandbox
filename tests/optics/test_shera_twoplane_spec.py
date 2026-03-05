import pytest

import dluxshera.params.transforms  # noqa: F401
from dluxshera.systems.two_plane import (
    SheraTwoPlaneConfig,
    build_forward_spec_from_config,
)
from dluxshera.params.spec import (
    ParamSpec,
    build_inference_spec_basic,
)
from dluxshera.params.store import ParameterStore, refresh_derived
from dluxshera.params.transform_registry import DERIVED_RESOLVER, DEFAULT_SYSTEM_ID


def test_twoplane_config_defaults():
    cfg = SheraTwoPlaneConfig()

    assert cfg.pupil_npix == 256
    assert cfg.psf_npix == 256
    assert cfg.oversample == 3
    assert cfg.m1_diameter_m == pytest.approx(0.09)
    assert cfg.plate_scale_as_per_pix == pytest.approx(0.355)
    assert cfg.primary_noll_indices == ()


def test_twoplane_forward_spec_structure_with_primary_basis():
    cfg = SheraTwoPlaneConfig(primary_noll_indices=(2, 3))
    spec = build_forward_spec_from_config(cfg)

    expected_binary_keys = {
        "source.x_position_as",
        "source.y_position_as",
        "source.separation_as",
        "source.position_angle_deg",
        "source.contrast",
    }

    assert expected_binary_keys.issubset(set(spec.keys()))
    assert "secondary.zernike_coeffs_nm" not in spec

    primary_field = spec.get("primary.zernike_coeffs_nm")
    assert primary_field.shape == (2,)
    assert primary_field.default == (0.0, 0.0)

    plate_scale_field = spec.get("optics.plate_scale_as_per_pix")
    assert plate_scale_field.kind == "primitive"
    assert plate_scale_field.default == cfg.plate_scale_as_per_pix

    log_flux_field = spec.get("source.log_flux_total")
    assert log_flux_field.kind == "derived"
    assert log_flux_field.transform == "source.log_flux_total"


def test_twoplane_forward_spec_refresh():
    cfg = SheraTwoPlaneConfig()
    spec = build_forward_spec_from_config(cfg)

    store = ParameterStore.from_spec_defaults(spec)
    assert "source.log_flux_total" not in store

    refreshed = refresh_derived(
        store,
        spec,
        resolver=DERIVED_RESOLVER,
        system_id=DEFAULT_SYSTEM_ID,
        include_derived=True,
    )

    assert refreshed.get("optics.plate_scale_as_per_pix") == pytest.approx(
        cfg.plate_scale_as_per_pix
    )
    assert "source.log_flux_total" in refreshed
    assert refreshed.get("source.log_flux_total") > 0.0


def test_inference_spec_secondary_toggle():
    spec_with_secondary = build_inference_spec_basic()
    assert "secondary.zernike_coeffs_nm" in spec_with_secondary

    spec_without_secondary: ParamSpec = build_inference_spec_basic(
        include_secondary=False
    )
    assert "secondary.zernike_coeffs_nm" not in spec_without_secondary
    # Ensure shared astrometry keys remain
    assert "source.separation_as" in spec_without_secondary
    assert "optics.plate_scale_as_per_pix" in spec_without_secondary
