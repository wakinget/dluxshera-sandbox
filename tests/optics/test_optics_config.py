import pytest
from dataclasses import FrozenInstanceError

from dluxshera.optics.config import SHERA_TESTBED_CONFIG, SheraTwoPlaneConfig


def test_threeplane_config_is_frozen():
    cfg = SHERA_TESTBED_CONFIG

    with pytest.raises(FrozenInstanceError):
        cfg.psf_npix = 999  # type: ignore[misc]


def test_replace_returns_new_instance_and_preserves_original():
    cfg = SHERA_TESTBED_CONFIG

    tweaked = cfg.replace(psf_npix=128)

    assert tweaked is not cfg
    assert tweaked.psf_npix == 128
    assert cfg.psf_npix != tweaked.psf_npix
    assert tweaked.primary_noll_indices == cfg.primary_noll_indices


def test_replace_rejects_unknown_fields():
    cfg = SHERA_TESTBED_CONFIG

    with pytest.raises(TypeError):
        cfg.replace(not_a_field=1)  # type: ignore[arg-type]


def test_twoplane_replace_allows_immutability():
    cfg = SheraTwoPlaneConfig()

    updated = cfg.replace(oversample=5)

    assert updated is not cfg
    assert updated.oversample == 5
    assert cfg.oversample != updated.oversample
