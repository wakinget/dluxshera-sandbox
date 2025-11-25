import jax.numpy as jnp

from dluxshera.optics.config import SHERA_TESTBED_CONFIG
from dluxshera.optics.builder import build_shera_threeplane_optics


def test_build_shera_threeplane_optics_smoke():
    cfg = SHERA_TESTBED_CONFIG
    optics = build_shera_threeplane_optics(cfg)

    # Basic structural checks
    assert hasattr(optics, "wf_npixels")
    assert hasattr(optics, "psf_npixels")

    # Check that key values match config
    assert optics.wf_npixels == cfg.pupil_npix
    assert optics.psf_npixels == cfg.psf_npix

    # You can add more checks if SheraThreePlaneSystem exposes them, e.g.:
    # assert optics.m1_diameter == cfg.m1_diameter_m
