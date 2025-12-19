import pytest

from dluxshera.params.store import ParameterStore
from dluxshera.params.store_namespace import StoreNamespace


def test_attribute_access_returns_prefixed_value():
    store = ParameterStore.from_dict({"system.plate_scale_as_per_pix": 0.025})

    ns = StoreNamespace(store, "system")

    assert ns.plate_scale_as_per_pix == store.get("system.plate_scale_as_per_pix")


def test_missing_leaf_raises_attribute_error():
    store = ParameterStore.from_dict({})
    ns = StoreNamespace(store, "system")

    with pytest.raises(AttributeError):
        _ = ns.missing_leaf
