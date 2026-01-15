import pytest

from dluxshera.params.store import ParameterStore
from dluxshera.params.transform_registry import DEFAULT_SYSTEM_ID
from dluxshera.systems.three_plane import SheraThreePlaneBinder
from dluxshera.systems.two_plane import (
    SHERA_TWOPLANE_SYSTEM_ID,
    SheraTwoPlaneBinder,
    SheraTwoPlaneConfig,
)
from tests.conftest import make_forward_store


def _runtime_cfg_for_binder(binder_cls, request):
    if binder_cls is SheraThreePlaneBinder:
        return request.getfixturevalue("shera_smoke_cfg")
    return SheraTwoPlaneConfig(pupil_npix=96, psf_npix=96, oversample=2, n_lambda=2)


def _refresh_store(store, spec):
    system_id = DEFAULT_SYSTEM_ID if spec.system_id == SHERA_TWOPLANE_SYSTEM_ID else None
    return store.refresh_derived(spec, system_id=system_id)


@pytest.mark.parametrize(
    "binder_cls",
    [SheraThreePlaneBinder, SheraTwoPlaneBinder],
)
def test_model_runtime_overlay_avoids_rebuild(monkeypatch, binder_cls, request):
    cfg = _runtime_cfg_for_binder(binder_cls, request)
    forward_spec, forward_store = make_forward_store(cfg)
    binder = binder_cls(cfg, forward_spec, forward_store)

    def _boom(*_args, **_kwargs):
        raise AssertionError("model() unexpectedly rebuilt the telescope")

    monkeypatch.setattr(binder, "_build_telescope", _boom)

    base_contrast = forward_store.get("binary.contrast")
    delta = ParameterStore.from_dict({"binary.contrast": base_contrast + 0.5})

    binder.model(delta)


@pytest.mark.parametrize(
    "binder_cls",
    [SheraThreePlaneBinder, SheraTwoPlaneBinder],
)
def test_update_store_runtime_avoids_optics_rebuild(monkeypatch, binder_cls, request):
    cfg = _runtime_cfg_for_binder(binder_cls, request)
    forward_spec, forward_store = make_forward_store(cfg)
    binder = binder_cls(cfg, forward_spec, forward_store)

    def _boom(*_args, **_kwargs):
        raise AssertionError("update_store() unexpectedly rebuilt optics")

    monkeypatch.setattr(binder, "_build_optics", _boom)

    base_contrast = forward_store.get("binary.contrast")
    updated_store = forward_store.replace({"binary.contrast": base_contrast + 0.25})

    updated_binder = binder.update_store(updated_store)

    assert updated_binder.base_forward_store.get("binary.contrast") == pytest.approx(
        base_contrast + 0.25
    )


@pytest.mark.parametrize(
    "binder_cls",
    [SheraThreePlaneBinder, SheraTwoPlaneBinder],
)
def test_structural_update_requires_allow_rebuild(binder_cls, request):
    cfg = _runtime_cfg_for_binder(binder_cls, request)
    forward_spec, forward_store = make_forward_store(cfg)
    binder = binder_cls(cfg, forward_spec, forward_store)

    new_value = forward_store.get("system.m1_diameter_m") + 0.01
    updated_store = forward_store.replace({"system.m1_diameter_m": new_value})
    updated_store = _refresh_store(updated_store, forward_spec)

    with pytest.raises(ValueError, match="system.m1_diameter_m"):
        binder.update_store(updated_store)


@pytest.mark.parametrize(
    "binder_cls",
    [SheraThreePlaneBinder, SheraTwoPlaneBinder],
)
def test_structural_update_rebuilds_optics_when_allowed(monkeypatch, binder_cls, request):
    cfg = _runtime_cfg_for_binder(binder_cls, request)
    forward_spec, forward_store = make_forward_store(cfg)
    binder = binder_cls(cfg, forward_spec, forward_store)

    new_value = forward_store.get("system.m1_diameter_m") + 0.01
    updated_store = forward_store.replace({"system.m1_diameter_m": new_value})
    updated_store = _refresh_store(updated_store, forward_spec)

    calls = {"count": 0}
    original = binder._build_optics

    def _wrapped(store):
        calls["count"] += 1
        return original(store)

    monkeypatch.setattr(binder, "_build_optics", _wrapped)

    updated_binder = binder.update_store(updated_store, allow_rebuild=True)

    assert calls["count"] == 1
    assert updated_binder.base_forward_store.get("system.m1_diameter_m") == pytest.approx(
        new_value
    )


@pytest.mark.parametrize(
    "binder_cls",
    [SheraThreePlaneBinder, SheraTwoPlaneBinder],
)
def test_mixed_updates_structural_dominate(binder_cls, request):
    cfg = _runtime_cfg_for_binder(binder_cls, request)
    forward_spec, forward_store = make_forward_store(cfg)
    binder = binder_cls(cfg, forward_spec, forward_store)

    delta = ParameterStore.from_dict(
        {
            "system.m1_diameter_m": forward_store.get("system.m1_diameter_m") + 0.02,
            "binary.contrast": forward_store.get("binary.contrast") + 0.1,
        }
    )

    with pytest.raises(ValueError, match="system.m1_diameter_m"):
        binder.model(delta)
