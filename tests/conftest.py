from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional

import jax.numpy as jnp
import pytest

from dluxshera.core.binder import SheraThreePlaneBinder
from dluxshera.core.builder import build_shera_threeplane_model
from dluxshera.optics.config import SHERA_TESTBED_CONFIG, SheraThreePlaneConfig, SheraTwoPlaneConfig
from dluxshera.params.spec import (
    ParamSpec,
    build_forward_model_spec_from_config,
    build_inference_spec_basic,
    build_shera_twoplane_forward_spec_from_config,
    SHERA_TWOPLANE_SYSTEM_ID,
)
from dluxshera.params.store import ParameterStore
from dluxshera.params.transforms import DEFAULT_SYSTEM_ID


def make_forward_store(
    cfg,
    updates: Optional[Dict[str, object]] = None,
) -> tuple[ParamSpec, ParameterStore]:
    """Build a forward-style spec + store with deriveds refreshed."""

    if isinstance(cfg, SheraTwoPlaneConfig):
        spec = build_shera_twoplane_forward_spec_from_config(cfg)
    else:
        spec = build_forward_model_spec_from_config(cfg)
    store = ParameterStore.from_spec_defaults(spec)
    if updates:
        store = store.replace(updates)
    system_id = DEFAULT_SYSTEM_ID if spec.system_id == SHERA_TWOPLANE_SYSTEM_ID else None
    store = store.refresh_derived(spec, system_id=system_id)
    return spec, store


def inference_store_from_forward(
    forward_store: ParameterStore,
    inference_spec: Optional[ParamSpec] = None,
) -> tuple[ParamSpec, ParameterStore]:
    """Project a forward store onto an inference spec for legacy helpers."""

    spec = inference_spec or build_inference_spec_basic()
    base = ParameterStore.from_spec_defaults(spec)
    updates = {key: forward_store.get(key) for key in spec.keys() if key in forward_store}
    return spec, base.replace(updates)


@pytest.fixture(scope="session")
def shera_smoke_cfg():
    """Lightweight SHERA config for smoke tests (smaller grids/n_lambda)."""

    return replace(
        SHERA_TESTBED_CONFIG,
        pupil_npix=128,
        psf_npix=128,
        oversample=2,
        n_lambda=2,
    )


@pytest.fixture(scope="session")
def shera_smoke_updates(shera_smoke_cfg):
    updates = {
        "binary.separation_as": 10.0,
        "binary.position_angle_deg": 90.0,
        "binary.x_position_as": 0.0,
        "binary.y_position_as": 0.0,
        "binary.contrast": 3.0,
    }

    n_m1 = len(shera_smoke_cfg.primary_noll_indices)
    n_m2 = len(shera_smoke_cfg.secondary_noll_indices)
    if n_m1 > 0:
        updates["primary.zernike_coeffs_nm"] = jnp.zeros(n_m1)
    if n_m2 > 0:
        updates["secondary.zernike_coeffs_nm"] = jnp.zeros(n_m2)

    return updates


@pytest.fixture(scope="session")
def shera_smoke_forward(shera_smoke_cfg, shera_smoke_updates):
    return make_forward_store(shera_smoke_cfg, updates=shera_smoke_updates)


@pytest.fixture(scope="session")
def shera_smoke_inference(shera_smoke_forward):
    forward_spec, forward_store = shera_smoke_forward
    return inference_store_from_forward(forward_store)


@pytest.fixture(scope="session")
def shera_smoke_infer_keys():
    return (
        "binary.separation_as",
        "binary.x_position_as",
        "binary.y_position_as",
    )


@pytest.fixture(scope="session")
def shera_smoke_binder_data(shera_smoke_cfg, shera_smoke_forward):
    forward_spec, forward_store = shera_smoke_forward
    binder = SheraThreePlaneBinder(shera_smoke_cfg, forward_spec, forward_store)
    data = binder.model()
    var = jnp.ones_like(data)
    return binder, data, var


@pytest.fixture(scope="session")
def shera_smoke_model_data(shera_smoke_cfg, shera_smoke_inference):
    inference_spec, inference_store = shera_smoke_inference
    model = build_shera_threeplane_model(shera_smoke_cfg, inference_spec, inference_store)
    data = model.model()
    var = jnp.ones_like(data)
    return data, var
