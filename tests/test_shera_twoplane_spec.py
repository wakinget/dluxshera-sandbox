import pytest

import dluxshera.params.shera_threeplane_transforms  # noqa: F401
from dluxshera.optics.config import SheraTwoPlaneConfig
from dluxshera.params.spec import (
    ParamSpec,
    build_inference_spec_basic,
    build_shera_twoplane_forward_spec_from_config,
)
from dluxshera.params.store import ParameterStore, refresh_derived
from dluxshera.params.transforms import DERIVED_RESOLVER, DEFAULT_SYSTEM_ID


def test_twoplane_config_defaults():
    cfg = SheraTwoPlaneConfig()

    assert cfg.pupil_npix == 256
    assert cfg.psf_npix == 256
    assert cfg.oversample == 3
    assert cfg.m1_diameter_m == pytest.approx(0.09)
    assert cfg.plate_scale_as_per_pix == pytest.approx(0.3547)
    assert cfg.primary_noll_indices == ()


def test_twoplane_forward_spec_structure_with_primary_basis():
    cfg = SheraTwoPlaneConfig(primary_noll_indices=(2, 3))
    spec = build_shera_twoplane_forward_spec_from_config(cfg)

    expected_binary_keys = {
        "binary.x_position_as",
        "binary.y_position_as",
        "binary.separation_as",
        "binary.position_angle_deg",
        "binary.contrast",
    }

    assert expected_binary_keys.issubset(set(spec.keys()))
    assert "secondary.zernike_coeffs" not in spec

    primary_field = spec.get("primary.zernike_coeffs")
    assert primary_field.shape == (2,)
    assert primary_field.default == (0.0, 0.0)

    plate_scale_field = spec.get("system.plate_scale_as_per_pix")
    assert plate_scale_field.kind == "primitive"
    assert plate_scale_field.default == cfg.plate_scale_as_per_pix

    log_flux_field = spec.get("binary.log_flux_total")
    assert log_flux_field.kind == "derived"
    assert log_flux_field.transform == "binary_log_flux_total"


def test_twoplane_forward_spec_refresh():
    cfg = SheraTwoPlaneConfig()
    spec = build_shera_twoplane_forward_spec_from_config(cfg)

    store = ParameterStore.from_spec_defaults(spec)
    assert "binary.log_flux_total" not in store

    refreshed = refresh_derived(
        store, spec, DERIVED_RESOLVER, DEFAULT_SYSTEM_ID, include_derived=True
    )

    assert refreshed.get("system.plate_scale_as_per_pix") == pytest.approx(
        cfg.plate_scale_as_per_pix
    )
    assert "binary.log_flux_total" in refreshed
    assert refreshed.get("binary.log_flux_total") > 0.0


def test_inference_spec_secondary_toggle():
    spec_with_secondary = build_inference_spec_basic()
    assert "secondary.zernike_coeffs" in spec_with_secondary

    spec_without_secondary: ParamSpec = build_inference_spec_basic(
        include_secondary=False
    )
    assert "secondary.zernike_coeffs" not in spec_without_secondary
    # Ensure shared astrometry keys remain
    assert "binary.separation_as" in spec_without_secondary
    assert "system.plate_scale_as_per_pix" in spec_without_secondary
