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


def test_ambiguous_leaf_attr_raises_attribute_error():
    binder = _make_binder()

    with pytest.raises(AttributeError) as excinfo:
        _ = binder.zernike_coeffs_nm

    message = str(excinfo.value)
    assert "Ambiguous leaf name" in message
    assert "optics.primary.zernike_coeffs_nm" in message
    assert "optics.secondary.zernike_coeffs_nm" in message
    assert "binder.get(\"<full.key>\")" in message


def test_bound_leaf_prefers_runtime_binding_over_store():
    binder = _make_binder()
    runtime_value = binder.optics.psf_npixels

    # Diverge the store to ensure runtime wins.
    binder.base_forward_store = binder.base_forward_store.replace(
        {"optics.psf_npix": runtime_value + 5}
    )

    assert binder.psf_npix == runtime_value


def test_default_binding_falls_back_to_leaf_name_runtime_first():
    binder = _make_binder()
    runtime_value = binder.source.contrast

    binder.base_forward_store = binder.base_forward_store.replace(
        {"source.contrast": runtime_value + 0.25}
    )

    assert binder.contrast == runtime_value


def test_store_fallback_when_runtime_missing(monkeypatch):
    binder = _make_binder()
    expected = binder.base_forward_store.get("optics.plate_scale_as_per_pix")

    monkeypatch.setattr(binder, "_read_runtime_value", lambda field: (False, None))

    assert binder.plate_scale_as_per_pix == expected
