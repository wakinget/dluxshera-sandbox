# tests/test_binder_shared_behaviour.py

import pytest

from dluxshera.systems import SheraBinder
from dluxshera.systems.three_plane import SHERA_TESTBED_CONFIG
from dluxshera.systems import SheraBinder
from dluxshera.systems.two_plane import SheraTwoPlaneConfig
from dluxshera.params.store import ParameterStore
from tests.conftest import make_forward_store


@pytest.mark.parametrize(
    "binder_cls,cfg",
    [
        (SheraBinder, SHERA_TESTBED_CONFIG),
        (SheraBinder, SheraTwoPlaneConfig()),
    ],
)
def test_binder_merge_overlay_is_shared(binder_cls, cfg):
    forward_spec, forward_store = make_forward_store(cfg)
    binder = binder_cls(cfg, forward_spec, forward_store)

    base_contrast = forward_store.get("source.contrast")
    delta_store = ParameterStore.from_dict({"source.contrast": base_contrast + 0.5})

    eff_store = binder._merge_store(delta_store)

    assert eff_store.get("source.contrast") == pytest.approx(base_contrast + 0.5)
    assert eff_store.get("source.separation_as") == forward_store.get(
        "source.separation_as"
    )
    # Ensure the base store remains unchanged to preserve immutability expectations
    assert binder.base_forward_store.get("source.contrast") == base_contrast


@pytest.mark.parametrize(
    "binder_cls,cfg",
    [
        (SheraBinder, SHERA_TESTBED_CONFIG),
        (SheraBinder, SheraTwoPlaneConfig()),
    ],
)
def test_binder_get_reads_cfg_and_store(binder_cls, cfg):
    forward_spec, forward_store = make_forward_store(cfg)
    binder = binder_cls(cfg, forward_spec, forward_store)

    psf_npix_value = binder.get("psf_npix")
    assert psf_npix_value == binder.cfg.psf_npix

    plate_scale_path = "optics.plate_scale_as_per_pix"
    plate_scale_value = binder.get(plate_scale_path)
    assert plate_scale_value == binder.base_forward_store.get(plate_scale_path)

    values = binder.get(["psf_npix", plate_scale_path])
    assert values == [psf_npix_value, plate_scale_value]


@pytest.mark.parametrize(
    "binder_cls,cfg",
    [
        (SheraBinder, SHERA_TESTBED_CONFIG),
        (SheraBinder, SheraTwoPlaneConfig()),
    ],
)
def test_binder_cfg_field_forwarding(binder_cls, cfg):
    forward_spec, forward_store = make_forward_store(cfg)
    binder = binder_cls(cfg, forward_spec, forward_store)

    assert binder.cfg is cfg
    assert binder.psf_npix == binder.optics.psf_npixels
    assert binder.psf_npix == binder.cfg.psf_npix

    with pytest.raises(AttributeError):
        binder.this_does_not_exist
