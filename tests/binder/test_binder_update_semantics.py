import pytest
import jax.numpy as jnp

from dluxshera.params.store import ParameterStore
from dluxshera.params.transform_registry import DEFAULT_SYSTEM_ID
from dluxshera.systems import SheraBinder
from dluxshera.systems.three_plane import SHERA_TESTBED_CONFIG
from dluxshera.systems.two_plane import SHERA_TWOPLANE_SYSTEM_ID, SheraTwoPlaneConfig
from tests.conftest import make_forward_store


@pytest.fixture(params=[SHERA_TESTBED_CONFIG, SheraTwoPlaneConfig(pupil_npix=96, psf_npix=96, oversample=2, n_lambda=2)])
def cfg(request):
    return request.param


def _refresh_store(store, spec):
    system_id = DEFAULT_SYSTEM_ID if spec.system_id == SHERA_TWOPLANE_SYSTEM_ID else None
    return store.refresh_derived(spec, system_id=system_id)


def test_model_runtime_overlay_avoids_rebuild(monkeypatch, cfg):
    forward_spec, forward_store = make_forward_store(cfg)
    binder = SheraBinder(cfg, forward_spec, forward_store)

    def _boom(*_args, **_kwargs):
        raise AssertionError("model() unexpectedly rebuilt the telescope")

    monkeypatch.setattr(binder, "_build_telescope", _boom)

    base_contrast = forward_store.get("source.contrast")
    delta = ParameterStore.from_dict({"source.contrast": base_contrast + 0.5})

    binder.model(delta)


def test_update_store_runtime_avoids_optics_rebuild(monkeypatch, cfg):
    forward_spec, forward_store = make_forward_store(cfg)
    binder = SheraBinder(cfg, forward_spec, forward_store)

    def _boom(*_args, **_kwargs):
        raise AssertionError("update_store() unexpectedly rebuilt optics")

    monkeypatch.setattr(binder, "_build_optics", _boom)

    base_contrast = forward_store.get("source.contrast")
    updated_store = forward_store.replace({"source.contrast": base_contrast + 0.25})

    updated_binder = binder.update_store(updated_store)

    assert updated_binder.base_forward_store.get("source.contrast") == pytest.approx(
        base_contrast + 0.25
    )


def test_structural_update_requires_allow_rebuild(cfg):
    forward_spec, forward_store = make_forward_store(cfg)
    binder = SheraBinder(cfg, forward_spec, forward_store)

    new_value = forward_store.get("optics.m1_diameter_m") + 0.01
    updated_store = forward_store.replace({"optics.m1_diameter_m": new_value})
    updated_store = _refresh_store(updated_store, forward_spec)

    with pytest.raises(ValueError, match="optics.m1_diameter_m"):
        binder.update_store(updated_store)


def test_structural_update_rebuilds_optics_when_allowed(monkeypatch, cfg):
    forward_spec, forward_store = make_forward_store(cfg)
    binder = SheraBinder(cfg, forward_spec, forward_store)

    new_value = forward_store.get("optics.m1_diameter_m") + 0.01
    updated_store = forward_store.replace({"optics.m1_diameter_m": new_value})
    updated_store = _refresh_store(updated_store, forward_spec)

    calls = {"count": 0}
    original = binder._build_optics

    def _wrapped(store):
        calls["count"] += 1
        return original(store)

    monkeypatch.setattr(binder, "_build_optics", _wrapped)

    updated_binder = binder.update_store(updated_store, allow_rebuild=True)

    assert calls["count"] == 1
    assert updated_binder.base_forward_store.get("optics.m1_diameter_m") == pytest.approx(
        new_value
    )


def test_mixed_updates_structural_dominate(cfg):
    forward_spec, forward_store = make_forward_store(cfg)
    binder = SheraBinder(cfg, forward_spec, forward_store)

    delta = ParameterStore.from_dict(
        {
            "optics.m1_diameter_m": forward_store.get("optics.m1_diameter_m") + 0.02,
            "source.contrast": forward_store.get("source.contrast") + 0.1,
        }
    )

    with pytest.raises(ValueError, match="optics.m1_diameter_m"):
        binder.model(delta)


def test_strip_structural_removes_only_structural_keys(cfg):
    forward_spec, forward_store = make_forward_store(cfg)
    binder = SheraBinder(cfg, forward_spec, forward_store)

    structural_key = next(iter(binder.structural_store_keys()))
    non_structural_key = next(
        key
        for key, field in forward_spec.items()
        if not field.structural and key in forward_store.keys()
    )

    base_non_structural = forward_store.get(non_structural_key)
    delta = ParameterStore.from_dict(
        {
            structural_key: forward_store.get(structural_key),
            non_structural_key: base_non_structural,
        }
    )

    stripped = binder.strip_structural(delta)

    assert structural_key not in stripped.keys()
    assert stripped.get(non_structural_key) == pytest.approx(base_non_structural)
    # Original delta remains unchanged
    assert jnp.all(delta.get(structural_key) == forward_store.get(structural_key))
