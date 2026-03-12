import pytest

from dluxshera.systems import SheraBinder
from dluxshera.systems.three_plane import SHERA_TESTBED_CONFIG
from tests.conftest import make_forward_store


def _make_binder():
    forward_spec, forward_store = make_forward_store(SHERA_TESTBED_CONFIG)
    return SheraBinder(
        SHERA_TESTBED_CONFIG,
        forward_spec,
        forward_store,
    )


def test_binder_ns_returns_store_namespace():
    binder = _make_binder()

    optics_ns = binder.ns("optics")

    assert optics_ns.plate_scale_as_per_pix == binder.base_forward_store.get(
        "optics.plate_scale_as_per_pix"
    )


@pytest.mark.parametrize("prefix", ["does_not_exist", "cfg"])
def test_binder_ns_validation(prefix):
    binder = _make_binder()

    with pytest.raises(ValueError):
        binder.ns(prefix)


def test_binder_store_prefix_attr_access():
    binder = _make_binder()

    optics = binder.optics

    assert optics is binder.telescope.optics
    assert binder.source is binder.telescope.source
    assert binder.detector is binder.telescope.detector


def test_binder_store_prefix_missing_attr_raises_attribute_error():
    binder = _make_binder()

    with pytest.raises(AttributeError):
        binder.this_prefix_does_not_exist
