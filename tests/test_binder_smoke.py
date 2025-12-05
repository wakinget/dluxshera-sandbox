# tests/test_binder_smoke.py

import jax.numpy as jnp

from dluxshera.optics.config import SHERA_TESTBED_CONFIG
from dluxshera.params.spec import build_inference_spec_basic
from dluxshera.params.store import ParameterStore
from dluxshera.core.binder import SheraThreePlaneBinder


def test_shera_threeplane_binder_smoke():
    cfg = SHERA_TESTBED_CONFIG
    spec = build_inference_spec_basic()
    store = ParameterStore.from_spec_defaults(spec)

    binder = SheraThreePlaneBinder(cfg, spec, store)

    img = binder.forward(store)

    assert img.ndim == 2  # simple shape sanity check
    assert jnp.all(jnp.isfinite(img))
