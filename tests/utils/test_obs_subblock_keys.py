from __future__ import annotations

import numpy as np
import pytest

from dluxshera.config.io import load_user_config
from dluxshera.config.resolver import resolve_config
from dluxshera.params.store import ParameterStore
from dluxshera.systems.base import compose_forward_spec
from dluxshera.utils.obs_subblock_keys import (
    apply_obs_subblock_overrides_preserving_derived,
    parse_obs_subblock_key_address,
    parse_obs_subblock_varying_keys,
    split_obs_subblock_frame_overrides,
    validate_supported_obs_subblock_key_addresses,
)


def _forward_spec_and_store():
    cfg = resolve_config(
        load_user_config(
            config_path=None,
            system_preset="SHERA_TESTBED_3P",
            experiment_preset=None,
        )
    )
    spec = compose_forward_spec(cfg["system"])
    store = ParameterStore.from_spec_defaults(spec).refresh_derived(spec)
    return spec, store


def test_parse_scalar_and_indexed_key_addresses():
    scalar = parse_obs_subblock_key_address("source.x_position_as")
    indexed = parse_obs_subblock_key_address("optics.primary.zernike_coeffs_nm[3]")

    assert scalar.base_key == "source.x_position_as"
    assert scalar.index is None
    assert scalar.canonical == "source.x_position_as"
    assert indexed.base_key == "optics.primary.zernike_coeffs_nm"
    assert indexed.index == 3
    assert indexed.canonical == "optics.primary.zernike_coeffs_nm[3]"


def test_parse_malformed_indexed_key_rejected():
    with pytest.raises(ValueError, match="Invalid observation-subblock key syntax"):
        parse_obs_subblock_key_address("optics.primary.zernike_coeffs_nm[abc]")


def test_unsupported_and_structural_keys_rejected():
    spec, store = _forward_spec_and_store()
    unsupported = parse_obs_subblock_varying_keys(["detector.read_noise_e"])
    with pytest.raises(ValueError, match="Unsupported observation-subblock varying key"):
        validate_supported_obs_subblock_key_addresses(unsupported)

    structural = parse_obs_subblock_varying_keys(["optics.psf_npix"])
    with pytest.raises(ValueError, match="Unsupported observation-subblock varying key"):
        validate_supported_obs_subblock_key_addresses(
            structural,
            forward_spec=spec,
            reference_store=store,
        )


def test_index_out_of_bounds_rejected_when_store_available():
    spec, store = _forward_spec_and_store()
    addresses = parse_obs_subblock_varying_keys(["optics.primary.zernike_coeffs_nm[999]"])
    with pytest.raises(ValueError, match="out of bounds"):
        validate_supported_obs_subblock_key_addresses(
            addresses,
            forward_spec=spec,
            reference_store=store,
        )


def test_explicit_plate_scale_override_survives_refresh():
    spec, base_store = _forward_spec_and_store()
    addresses = parse_obs_subblock_varying_keys(
        ["source.x_position_as", "optics.plate_scale_as_per_pix"]
    )
    row = {
        "source.x_position_as": 0.015,
        "optics.plate_scale_as_per_pix": 0.111,
    }
    primitive_overrides, derived_overrides = split_obs_subblock_frame_overrides(
        base_store=base_store,
        forward_spec=spec,
        addresses=addresses,
        values_by_key=row,
    )
    frame_store = apply_obs_subblock_overrides_preserving_derived(
        base_store,
        forward_spec=spec,
        primitive_overrides=primitive_overrides,
        derived_overrides=derived_overrides,
    )
    assert np.isclose(
        float(np.asarray(frame_store.get("optics.plate_scale_as_per_pix"))),
        0.111,
    )


def test_explicit_log_flux_override_survives_refresh():
    spec, base_store = _forward_spec_and_store()
    addresses = parse_obs_subblock_varying_keys(
        ["source.y_position_as", "source.log_flux_total"]
    )
    row = {
        "source.y_position_as": -0.02,
        "source.log_flux_total": 12.5,
    }
    primitive_overrides, derived_overrides = split_obs_subblock_frame_overrides(
        base_store=base_store,
        forward_spec=spec,
        addresses=addresses,
        values_by_key=row,
    )
    frame_store = apply_obs_subblock_overrides_preserving_derived(
        base_store,
        forward_spec=spec,
        primitive_overrides=primitive_overrides,
        derived_overrides=derived_overrides,
    )
    assert np.isclose(
        float(np.asarray(frame_store.get("source.log_flux_total"))),
        12.5,
    )


def test_absent_derived_override_falls_back_to_refreshed_value():
    spec, base_store = _forward_spec_and_store()
    addresses = parse_obs_subblock_varying_keys(["source.x_position_as"])
    row = {"source.x_position_as": 0.05}
    primitive_overrides, derived_overrides = split_obs_subblock_frame_overrides(
        base_store=base_store,
        forward_spec=spec,
        addresses=addresses,
        values_by_key=row,
    )
    frame_store = apply_obs_subblock_overrides_preserving_derived(
        base_store,
        forward_spec=spec,
        primitive_overrides=primitive_overrides,
        derived_overrides=derived_overrides,
    )
    refreshed_expected = (
        base_store.replace({"source.x_position_as": 0.05}).refresh_derived(spec)
    )
    assert np.isclose(
        float(np.asarray(frame_store.get("source.log_flux_total"))),
        float(np.asarray(refreshed_expected.get("source.log_flux_total"))),
    )


def test_indexed_zernike_components_update_independently():
    spec, base_store = _forward_spec_and_store()
    addresses = parse_obs_subblock_varying_keys(
        [
            "optics.primary.zernike_coeffs_nm[1]",
            "optics.primary.zernike_coeffs_nm[4]",
            "optics.secondary.zernike_coeffs_nm[2]",
        ]
    )
    row = {
        "optics.primary.zernike_coeffs_nm[1]": 10.0,
        "optics.primary.zernike_coeffs_nm[4]": -5.0,
        "optics.secondary.zernike_coeffs_nm[2]": 3.5,
    }
    primitive_overrides, derived_overrides = split_obs_subblock_frame_overrides(
        base_store=base_store,
        forward_spec=spec,
        addresses=addresses,
        values_by_key=row,
    )
    frame_store = apply_obs_subblock_overrides_preserving_derived(
        base_store,
        forward_spec=spec,
        primitive_overrides=primitive_overrides,
        derived_overrides=derived_overrides,
    )

    primary = np.asarray(frame_store.get("optics.primary.zernike_coeffs_nm"))
    secondary = np.asarray(frame_store.get("optics.secondary.zernike_coeffs_nm"))
    assert np.isclose(primary[1], 10.0)
    assert np.isclose(primary[4], -5.0)
    assert np.isclose(secondary[2], 3.5)
