from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dluxshera.config.io import load_user_config
from dluxshera.config.resolver import resolve_config
from dluxshera.params.store import ParameterStore
from dluxshera.params.transforms import transform_source_raw_fluxes
from dluxshera.systems.base import compose_forward_spec
from dluxshera.utils.obs_subblock_keys import (
    apply_jax_safe_source_photometry_update,
    apply_obs_subblock_overrides_preserving_derived,
    apply_obs_subblock_runtime_overrides_without_refresh,
    get_obs_subblock_mapping_value,
    get_obs_subblock_store_value,
    parse_obs_subblock_key_address,
    parse_obs_subblock_varying_keys,
    set_obs_subblock_mapping_value,
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


def test_get_obs_subblock_mapping_value_supports_scalar_and_indexed_addresses():
    mapping = {
        "source": {"log_flux_total": 12.5},
        "optics": {
            "primary": {"zernike_coeffs_nm": [0.0, 1.5, -2.0, 3.5]},
        },
    }

    assert get_obs_subblock_mapping_value(
        mapping,
        address=parse_obs_subblock_key_address("source.log_flux_total"),
    ) == pytest.approx(12.5)
    assert get_obs_subblock_mapping_value(
        mapping,
        address=parse_obs_subblock_key_address("optics.primary.zernike_coeffs_nm[3]"),
    ) == pytest.approx(3.5)
    assert (
        get_obs_subblock_mapping_value(
            mapping,
            address=parse_obs_subblock_key_address("source.contrast"),
        )
        is None
    )


def test_set_obs_subblock_mapping_value_patches_scalar_and_indexed_addresses():
    mapping = {
        "system": {
            "source": {"contrast": 0.1},
            "optics": {
                "primary": {"zernike_coeffs_nm": [0.0, 1.0, 2.0, 3.0]},
            },
        }
    }

    set_obs_subblock_mapping_value(
        mapping["system"],
        address=parse_obs_subblock_key_address("source.contrast"),
        value=0.25,
    )
    set_obs_subblock_mapping_value(
        mapping["system"],
        address=parse_obs_subblock_key_address("optics.primary.zernike_coeffs_nm[2]"),
        value=-4.5,
    )

    assert mapping["system"]["source"]["contrast"] == pytest.approx(0.25)
    assert mapping["system"]["optics"]["primary"]["zernike_coeffs_nm"] == [
        0.0,
        1.0,
        -4.5,
        3.0,
    ]


def test_set_obs_subblock_mapping_value_can_seed_missing_indexed_vector_from_store():
    _spec, store = _forward_spec_and_store()
    mapping = {"optics": {"kind": "three_plane", "primary": {}}}
    address = parse_obs_subblock_key_address("optics.primary.zernike_coeffs_nm[3]")
    reference_vector = np.asarray(store.get(address.base_key), dtype=float)

    set_obs_subblock_mapping_value(
        mapping,
        address=address,
        value=7.25,
        reference_vector=reference_vector,
    )

    patched = np.asarray(mapping["optics"]["primary"]["zernike_coeffs_nm"], dtype=float)
    assert patched.shape == reference_vector.shape
    assert patched[3] == pytest.approx(7.25)
    assert np.allclose(np.delete(patched, 3), np.delete(reference_vector, 3))


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


def test_jax_safe_source_photometry_update_matches_transform_semantics():
    store = ParameterStore.from_dict(
        {
            "source.log_flux_total": 12.0,
            "source.contrast": 3.0,
            "source.raw_fluxes": np.zeros(2, dtype=float),
        }
    )

    updated = apply_jax_safe_source_photometry_update(
        store,
        log_flux_total=12.3,
        contrast=2.5,
    )
    expected = transform_source_raw_fluxes(
        {
            "source.log_flux_total": 12.3,
            "source.contrast": 2.5,
        }
    )

    np.testing.assert_allclose(updated.get("source.raw_fluxes"), expected)
    assert float(np.asarray(updated.get("source.log_flux_total"))) == pytest.approx(12.3)
    assert float(np.asarray(updated.get("source.contrast"))) == pytest.approx(2.5)


def test_jax_safe_source_photometry_update_supports_gradients():
    store = ParameterStore.from_dict(
        {
            "source.log_flux_total": 12.0,
            "source.contrast": 3.0,
            "source.raw_fluxes": np.zeros(2, dtype=float),
        }
    )

    def _loss_log_flux(log_flux_total):
        updated = apply_jax_safe_source_photometry_update(
            store,
            log_flux_total=log_flux_total,
            contrast=3.0,
        )
        return jnp.sum(jnp.asarray(updated.get("source.raw_fluxes"), dtype=float))

    def _loss_contrast(contrast):
        updated = apply_jax_safe_source_photometry_update(
            store,
            log_flux_total=12.0,
            contrast=contrast,
        )
        return jnp.asarray(updated.get("source.raw_fluxes"), dtype=float)[0]

    grad_log_flux = jax.grad(_loss_log_flux)(jnp.asarray(12.0, dtype=float))
    grad_contrast = jax.grad(_loss_contrast)(jnp.asarray(3.0, dtype=float))

    assert np.isfinite(np.asarray(grad_log_flux)).all()
    assert np.isfinite(np.asarray(grad_contrast)).all()
    assert float(np.asarray(grad_log_flux)) > 0.0


def test_runtime_overrides_without_refresh_preserve_active_source_values():
    spec, base_store = _forward_spec_and_store()

    updated = apply_obs_subblock_runtime_overrides_without_refresh(
        base_store,
        overrides_flat={
            "source.log_flux_total": 12.5,
            "source.contrast": 2.75,
            "optics.plate_scale_as_per_pix": 0.111,
        },
        forward_spec=spec,
    )

    expected_raw_fluxes = transform_source_raw_fluxes(
        {
            "source.log_flux_total": 12.5,
            "source.contrast": 2.75,
        }
    )
    assert float(np.asarray(updated.get("source.log_flux_total"))) == pytest.approx(12.5)
    assert float(np.asarray(updated.get("source.contrast"))) == pytest.approx(2.75)
    assert float(np.asarray(updated.get("optics.plate_scale_as_per_pix"))) == pytest.approx(0.111)
    np.testing.assert_allclose(updated.get("source.raw_fluxes"), expected_raw_fluxes)


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


def test_get_obs_subblock_store_value_supports_indexed_candidates():
    _spec, store = _forward_spec_and_store()
    address = parse_obs_subblock_key_address("optics.primary.zernike_coeffs_nm[1]")

    assert get_obs_subblock_store_value(store, address=address) == pytest.approx(
        float(np.asarray(store.get(address.base_key))[1])
    )
