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


def test_unambiguous_leaf_attr_resolves_to_store_value():
    binder = _make_binder()

    assert (
        binder.plate_scale_as_per_pix
        == binder.base_forward_store.get("system.plate_scale_as_per_pix")
    )


def test_ambiguous_leaf_attr_raises_attribute_error():
    binder = _make_binder()

    with pytest.raises(AttributeError) as excinfo:
        _ = binder.zernike_coeffs_nm

    message = str(excinfo.value)
    assert "Ambiguous leaf name" in message
    assert "primary.zernike_coeffs_nm" in message
    assert "secondary.zernike_coeffs_nm" in message
    assert "binder.get(\"<full.key>\")" in message
