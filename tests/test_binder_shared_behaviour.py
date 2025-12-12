# tests/test_binder_shared_behaviour.py

import pytest

from dluxshera.core.binder import SheraThreePlaneBinder, SheraTwoPlaneBinder
from dluxshera.optics.config import SHERA_TESTBED_CONFIG, SheraTwoPlaneConfig
from dluxshera.params.store import ParameterStore
from .helpers import make_forward_store


@pytest.mark.parametrize(
    "binder_cls,cfg",
    [
        (SheraThreePlaneBinder, SHERA_TESTBED_CONFIG),
        (SheraTwoPlaneBinder, SheraTwoPlaneConfig()),
    ],
)
def test_binder_merge_overlay_is_shared(binder_cls, cfg):
    forward_spec, forward_store = make_forward_store(cfg)
    binder = binder_cls(cfg, forward_spec, forward_store, use_system_graph=False)

    base_contrast = forward_store.get("binary.contrast")
    delta_store = ParameterStore.from_dict({"binary.contrast": base_contrast + 0.5})

    eff_store = binder._merge_store(delta_store)

    assert eff_store.get("binary.contrast") == pytest.approx(base_contrast + 0.5)
    assert eff_store.get("binary.separation_as") == forward_store.get(
        "binary.separation_as"
    )
    # Ensure the base store remains unchanged to preserve immutability expectations
    assert binder.base_forward_store.get("binary.contrast") == base_contrast
