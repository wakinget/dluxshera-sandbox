from __future__ import annotations

from typing import Dict, Optional

from dluxshera.optics.config import SheraThreePlaneConfig, SheraTwoPlaneConfig
from dluxshera.params.spec import (
    ParamSpec,
    build_forward_model_spec_from_config,
    build_inference_spec_basic,
    build_shera_twoplane_forward_spec_from_config,
)
from dluxshera.params.store import ParameterStore, refresh_derived
from dluxshera.params.transforms import DEFAULT_SYSTEM_ID, TRANSFORMS
import dluxshera.params.shera_threeplane_transforms  # Registers default transforms


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
    store = refresh_derived(store, spec, TRANSFORMS, system_id=DEFAULT_SYSTEM_ID)
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
