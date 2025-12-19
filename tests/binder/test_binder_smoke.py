# tests/test_binder_smoke.py

import jax.numpy as jnp
import pytest

from dluxshera.core.binder import SheraThreePlaneBinder
from dluxshera.core.binder import SheraTwoPlaneBinder
from dluxshera.optics.config import SheraTwoPlaneConfig
from tests.conftest import make_forward_store


@pytest.mark.slow
def test_shera_threeplane_binder_smoke(shera_smoke_cfg):
    cfg = shera_smoke_cfg
    forward_spec, forward_store = make_forward_store(cfg)

    binder = SheraThreePlaneBinder(cfg, forward_spec, forward_store)

    img = binder.model()

    assert img.ndim == 2  # simple shape sanity check
    assert jnp.all(jnp.isfinite(img))


def test_shera_twoplane_binder_smoke():
    cfg = SheraTwoPlaneConfig()
    forward_spec, forward_store = make_forward_store(cfg)

    binder = SheraTwoPlaneBinder(cfg, forward_spec, forward_store)

    img = binder.model()

    assert img.ndim == 2
    assert jnp.all(jnp.isfinite(img))
