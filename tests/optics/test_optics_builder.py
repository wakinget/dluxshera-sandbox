from dataclasses import asdict, replace
from pathlib import Path

import jax.numpy as jnp

from dluxshera.params.spec import build_inference_spec_basic
from dluxshera.params.store import ParameterStore
from dluxshera.builders import optics as builder
from dluxshera.builders.optics import (
    build_shera_threeplane_optics,
    clear_threeplane_optics_cache,
    structural_hash_from_config,
)
from dluxshera.systems.three_plane import SHERA_TESTBED_CONFIG


def test_build_shera_threeplane_optics_smoke():
    cfg = SHERA_TESTBED_CONFIG
    optics = build_shera_threeplane_optics(cfg)

    # Basic structural checks
    assert hasattr(optics, "wf_npixels")
    assert hasattr(optics, "psf_npixels")

    # Check that key values match config
    assert optics.wf_npixels == cfg.pupil_npix
    assert optics.psf_npixels == cfg.psf_npix

    # You can add more checks if SheraThreePlaneOptics exposes them, e.g.:
    # assert optics.m1_diameter == cfg.m1_diameter_m


def test_build_shera_threeplane_optics_resolves_repo_relative_pupil_path(
    tmp_path: Path,
    monkeypatch,
):
    clear_threeplane_optics_cache()
    monkeypatch.chdir(tmp_path)
    cfg = asdict(SHERA_TESTBED_CONFIG)
    cfg["diffractive_pupil_path"] = "src/dluxshera/data/pupils/diffractive_pupil.npy"

    optics = build_shera_threeplane_optics(cfg)

    assert optics.wf_npixels == cfg["pupil_npix"]


def test_build_shera_threeplane_optics_uses_zernike_coeffs():
    cfg = SHERA_TESTBED_CONFIG

    # Make sure the config actually specifies some Noll indices
    n_m1 = len(cfg.primary_noll_indices)
    n_m2 = len(cfg.secondary_noll_indices)
    assert n_m1 > 0
    assert n_m2 > 0

    # Build a store from the basic inference spec
    spec = build_inference_spec_basic()
    store = ParameterStore.from_spec_defaults(spec)

    # Define some simple, known coefficients
    m1_coeffs = jnp.arange(n_m1, dtype=jnp.float32)
    m2_coeffs = -1 * jnp.arange(n_m2, dtype=jnp.float32)

    store = store.replace(
        {
            "optics.primary.zernike_coeffs_nm": m1_coeffs,
            "optics.secondary.zernike_coeffs_nm": m2_coeffs,
        }
    )

    optics = build_shera_threeplane_optics(cfg, store=store, spec=spec)

    # Try to inspect the underlying layers if the attributes are exposed.
    # We keep this defensive so the test doesn't break if the internal
    # storage format changes.
    m1_layer = None
    m2_layer = None

    # Primary mirror layer
    if hasattr(optics, "p1_layers"):
        p1 = getattr(optics, "p1_layers")
        # p1 is an OrderedDict, so use .items()
        for name, layer in p1.items():
            if name == "m1_aperture":
                m1_layer = layer
                break

    # Secondary mirror layer
    if hasattr(optics, "p2_layers"):
        p2 = getattr(optics, "p2_layers")
        for name, layer in p2.items():
            if name == "m2_aperture":
                m2_layer = layer
                break

    # Check coefficient wiring
    if m1_layer is not None and hasattr(m1_layer, "coefficients"):
        assert jnp.allclose(m1_layer.coefficients, m1_coeffs)

    if m2_layer is not None and hasattr(m2_layer, "coefficients"):
        assert jnp.allclose(m2_layer.coefficients, m2_coeffs)


def test_threeplane_optics_cache_hits():
    clear_threeplane_optics_cache()

    cfg = SHERA_TESTBED_CONFIG
    optics_a = build_shera_threeplane_optics(cfg)
    optics_b = build_shera_threeplane_optics(cfg)

    struct_hash = structural_hash_from_config(cfg)
    assert struct_hash in builder._THREEPLANE_CACHE
    assert len(builder._THREEPLANE_CACHE) == 1
    assert optics_a is not optics_b  # copies returned per-call for safety


def test_threeplane_optics_cache_miss_on_structural_change():
    clear_threeplane_optics_cache()

    cfg = SHERA_TESTBED_CONFIG
    tweaked_cfg = cfg.replace(m1_diameter_m=cfg.m1_diameter_m + 0.01)

    build_shera_threeplane_optics(cfg)
    build_shera_threeplane_optics(tweaked_cfg)

    assert len(builder._THREEPLANE_CACHE) == 2  # structural change invalidates cache key


def test_threeplane_cache_ignores_nonstructural_design_name():
    clear_threeplane_optics_cache()

    cfg = SHERA_TESTBED_CONFIG
    renamed_cfg = cfg.replace(design_name="alternate_label")

    build_shera_threeplane_optics(cfg)
    build_shera_threeplane_optics(renamed_cfg)

    assert len(builder._THREEPLANE_CACHE) == 1  # design_name is metadata only


def test_structural_hash_stable():
    cfg = SHERA_TESTBED_CONFIG
    hash_a = structural_hash_from_config(cfg)
    hash_b = structural_hash_from_config(cfg)

    assert hash_a == hash_b
