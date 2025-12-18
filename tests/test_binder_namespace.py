import pytest

from dluxshera.core.binder import SheraThreePlaneBinder
from dluxshera.optics.config import SHERA_TESTBED_CONFIG
from .helpers import make_forward_store


def _make_binder():
    forward_spec, forward_store = make_forward_store(SHERA_TESTBED_CONFIG)
    return SheraThreePlaneBinder(
        SHERA_TESTBED_CONFIG,
        forward_spec,
        forward_store,
        use_system_graph=False,
    )


def test_binder_ns_returns_store_namespace():
    binder = _make_binder()

    system_ns = binder.ns("system")

    assert (
        system_ns.plate_scale_as_per_pix
        == binder.base_forward_store.get("system.plate_scale_as_per_pix")
    )


@pytest.mark.parametrize("prefix", ["does_not_exist", "cfg"])
def test_binder_ns_validation(prefix):
    binder = _make_binder()

    with pytest.raises(ValueError):
        binder.ns(prefix)
