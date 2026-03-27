from __future__ import annotations

import jax.numpy as jnp

from dluxshera.builders.detector import build_detector
from dluxshera.params.store import ParameterStore
from dluxshera.systems import SheraBinder
from dluxshera.systems.two_plane import (
    SHERA_TWOPLANE_SYSTEM_ID,
    SheraTwoPlaneConfig,
    build_forward_spec_from_config,
)


def _make_twoplane_binder() -> tuple[SheraBinder, float]:
    cfg = SheraTwoPlaneConfig(
        pupil_npix=64,
        psf_npix=64,
        oversample=1,
        n_lambda=1,
        primary_noll_indices=(),
    )
    forward_spec = build_forward_spec_from_config(cfg)
    forward_store = ParameterStore.from_spec_defaults(forward_spec).refresh_derived(
        forward_spec,
        system_id=SHERA_TWOPLANE_SYSTEM_ID,
    )
    binder = SheraBinder(cfg, forward_spec, forward_store)
    return binder, float(forward_store.get("detector.layers.jitter.sigma"))


def test_build_detector_returns_jitter_contract_with_scalar_sigma_field():
    cfg = SheraTwoPlaneConfig(psf_npix=32, oversample=1)

    _detector, detector_contract = build_detector(cfg)

    assert "detector.layers.jitter.sigma" in detector_contract
    field = detector_contract.get("detector.layers.jitter.sigma")
    assert field.shape == ()
    assert float(field.default) == 1e-12


def test_runtime_jitter_override_changes_model_output():
    binder, _default_sigma = _make_twoplane_binder()

    low_sigma = ParameterStore.from_dict({"detector.layers.jitter.sigma": 1e-6})
    high_sigma = ParameterStore.from_dict({"detector.layers.jitter.sigma": 2.0})

    image_low = binder.model(store_delta=low_sigma)
    image_high = binder.model(store_delta=high_sigma)

    diff_norm = float(jnp.linalg.norm(image_high - image_low))
    assert diff_norm > 0.0
    assert not jnp.allclose(image_high, image_low)


def test_no_jitter_override_matches_explicit_default_sigma():
    binder, default_sigma = _make_twoplane_binder()

    baseline = binder.model()
    explicit_default = binder.model(
        store_delta=ParameterStore.from_dict({"detector.layers.jitter.sigma": default_sigma})
    )

    assert jnp.allclose(baseline, explicit_default)
