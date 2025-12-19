import pytest

from dluxshera.core.binder import SheraThreePlaneBinder
from dluxshera.optics.config import SHERA_TESTBED_CONFIG
from tests.conftest import make_forward_store


def _make_binder():
    forward_spec, forward_store = make_forward_store(SHERA_TESTBED_CONFIG)
    return SheraThreePlaneBinder(
        SHERA_TESTBED_CONFIG,
        forward_spec,
        forward_store,
        use_system_graph=False,
    )


def test_leaf_index_includes_plate_scale_key():
    binder = _make_binder()

    leaf_index = binder._leaf_index()

    assert "plate_scale_as_per_pix" in leaf_index
    assert "system.plate_scale_as_per_pix" in leaf_index["plate_scale_as_per_pix"]


def test_leaf_index_collects_zernike_leaves():
    binder = _make_binder()

    leaf_index = binder._leaf_index()

    zernike_paths = leaf_index.get("zernike_coeffs_nm", [])

    assert set(zernike_paths) == {
        "primary.zernike_coeffs_nm",
        "secondary.zernike_coeffs_nm",
    }
