from __future__ import annotations

import jax.numpy as jnp

from dluxshera.builders.detector import apply_runtime_bindings, build_detector
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


def _make_convolution_binder(
    detector_layers,
) -> tuple[SheraBinder, SheraTwoPlaneConfig]:
    cfg = SheraTwoPlaneConfig(
        pupil_npix=64,
        psf_npix=64,
        oversample=1,
        n_lambda=1,
        primary_noll_indices=(),
        detector_layers=detector_layers,
    )
    forward_spec = build_forward_spec_from_config(cfg)
    forward_store = ParameterStore.from_spec_defaults(forward_spec).refresh_derived(
        forward_spec,
        system_id=SHERA_TWOPLANE_SYSTEM_ID,
    )
    return SheraBinder(cfg, forward_spec, forward_store), cfg


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


def test_build_detector_returns_convolution_contract_with_name_scoped_fields():
    cfg = SheraTwoPlaneConfig(
        psf_npix=32,
        oversample=1,
        detector_layers=[
            {
                "name": "diffusion",
                "kind": "ApplyConvolution",
                "kernel": {
                    "kind": "gaussian",
                    "sigma_x": 0.25,
                    "sigma_y": 0.15,
                    "theta_deg": 10.0,
                    "kernel_size": 9,
                    "units": "psf_pix",
                },
            }
        ],
    )

    _detector, detector_contract = build_detector(cfg)

    assert "detector.layers.diffusion.sigma_x" in detector_contract
    assert "detector.layers.diffusion.sigma_y" in detector_contract
    assert "detector.layers.diffusion.theta_deg" in detector_contract
    assert "detector.layers.diffusion.kernel_size" in detector_contract


def test_runtime_convolution_override_changes_model_output():
    binder, _cfg = _make_convolution_binder(
        [
            {
                "name": "diffusion",
                "kind": "ApplyConvolution",
                "kernel": {
                    "kind": "gaussian",
                    "sigma_x": 0.25,
                    "sigma_y": 0.15,
                    "theta_deg": 0.0,
                    "kernel_size": 9,
                    "units": "psf_pix",
                },
            }
        ]
    )

    mild = ParameterStore.from_dict(
        {
            "detector.layers.diffusion.sigma_x": 0.25,
            "detector.layers.diffusion.sigma_y": 0.15,
            "detector.layers.diffusion.theta_deg": 0.0,
        }
    )
    strong = ParameterStore.from_dict(
        {
            "detector.layers.diffusion.sigma_x": 1.25,
            "detector.layers.diffusion.sigma_y": 0.35,
            "detector.layers.diffusion.theta_deg": 45.0,
        }
    )

    image_mild = binder.model(store_delta=mild)
    image_strong = binder.model(store_delta=strong)

    diff_norm = float(jnp.linalg.norm(image_strong - image_mild))
    assert diff_norm > 0.0
    assert not jnp.allclose(image_strong, image_mild)


def test_runtime_bindings_update_only_the_named_convolution_layer():
    cfg = SheraTwoPlaneConfig(
        psf_npix=32,
        oversample=1,
        detector_layers=[
            {
                "name": "diffusion_a",
                "kind": "ApplyConvolution",
                "kernel": {
                    "kind": "gaussian",
                    "sigma_x": 0.20,
                    "sigma_y": 0.10,
                    "theta_deg": 0.0,
                    "kernel_size": 7,
                    "units": "psf_pix",
                },
            },
            {
                "name": "diffusion_b",
                "kind": "ApplyConvolution",
                "kernel": {
                    "kind": "gaussian",
                    "sigma_x": 0.30,
                    "sigma_y": 0.20,
                    "theta_deg": 25.0,
                    "kernel_size": 9,
                    "units": "psf_pix",
                },
            },
        ],
    )

    detector, _contract = build_detector(cfg)
    updated = apply_runtime_bindings(
        detector,
        ParameterStore.from_dict({"detector.layers.diffusion_b.sigma_x": 1.5}),
    )

    assert float(updated.layers["diffusion_a"].sigma_x) == float(detector.layers["diffusion_a"].sigma_x)
    assert float(updated.layers["diffusion_a"].sigma_y) == float(detector.layers["diffusion_a"].sigma_y)
    assert float(updated.layers["diffusion_b"].sigma_x) == 1.5
    assert float(updated.layers["diffusion_b"].sigma_y) == float(detector.layers["diffusion_b"].sigma_y)
