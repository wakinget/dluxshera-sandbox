import jax.numpy as jnp

from dluxshera.params.packing import pack_params, unpack_params
from dluxshera.params.spec import make_inference_subspec
from dluxshera.params.store import ParameterStore
from dluxshera.systems.three_plane import SHERA_TESTBED_CONFIG, build_forward_spec_from_config


def _build_forward_store():
    forward_spec = build_forward_spec_from_config(SHERA_TESTBED_CONFIG)
    base_store = ParameterStore.from_spec_defaults(forward_spec).replace(
        {
            "source.log_flux_total": 8.0,
            "source.contrast": 2.5,
            "source.separation_as": 10.0,
            "source.position_angle_deg": 90.0,
        }
    )
    base_store = base_store.refresh_derived(forward_spec)
    return forward_spec, base_store


def test_subset_layout_matches_legacy_helper_layout():
    forward_spec, base_store = _build_forward_store()
    infer_keys = [
        "source.separation_as",
        "source.raw_fluxes",
        "optics.plate_scale_as_per_pix",
    ]

    legacy_subspec = make_inference_subspec(base_spec=forward_spec, infer_keys=infer_keys)
    new_subspec = forward_spec.subset(infer_keys)

    assert list(legacy_subspec.keys()) == list(new_subspec.keys()) == infer_keys

    legacy_theta = pack_params(legacy_subspec, base_store)
    new_theta = pack_params(new_subspec, base_store)

    assert legacy_theta.shape == new_theta.shape
    assert jnp.allclose(legacy_theta, new_theta)


def test_forward_subset_allows_derived_key_pack_unpack_roundtrip():
    forward_spec, base_store = _build_forward_store()

    infer_keys = ["source.raw_fluxes"]
    subspec = forward_spec.subset(infer_keys)

    assert list(subspec.keys()) == infer_keys
    assert subspec.get("source.raw_fluxes").kind == "derived"

    theta = pack_params(subspec, base_store)
    unpacked = unpack_params(subspec, theta, base_store)

    assert jnp.allclose(unpacked.get("source.raw_fluxes"), base_store.get("source.raw_fluxes"))


def test_pack_unpack_store_delta_from_forward_subset():
    forward_spec, base_store = _build_forward_store()
    infer_keys = ["source.separation_as", "source.position_angle_deg"]
    subspec = forward_spec.subset(infer_keys)

    theta0 = pack_params(subspec, base_store)
    theta1 = theta0 + jnp.array([0.25, -1.5], dtype=theta0.dtype)

    updated_store = unpack_params(subspec, theta1, base_store)

    assert float(updated_store.get("source.separation_as")) == float(theta1[0])
    assert float(updated_store.get("source.position_angle_deg")) == float(theta1[1])
    # Non-inferred keys are preserved from the base store.
    assert float(updated_store.get("source.log_flux_total")) == float(base_store.get("source.log_flux_total"))
