"""Optics builder responsibilities (structural assembly and caching)."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional, TYPE_CHECKING, Any

import numpy as np
import jax.numpy as jnp
import dLux.layers as dll

if TYPE_CHECKING:
    from ..systems.three_plane import SheraThreePlaneConfig
    from ..systems.two_plane import SheraTwoPlaneConfig
from ..components.optics import (
    SheraThreePlaneOptics,
    SheraTwoPlaneOptics,
    build_threeplane_optics_contract,
    build_twoplane_optics_contract,
)
from ..params.store import ParameterStore
from ..params.spec import ParamField, ParamSpec


_THREEPLANE_CACHE: dict[str, SheraThreePlaneOptics] = {}
_TWOPLANE_CACHE: dict[str, SheraTwoPlaneOptics] = {}
_CACHE_DISABLED_ENV = "DLUXSHERA_THREEPLANE_CACHE_DISABLED"
_TWOPLANE_CACHE_DISABLED_ENV = "DLUXSHERA_TWOPLANE_CACHE_DISABLED"


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        return [_normalize_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_normalize_json_value(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _config_attr_for_field(field: ParamField) -> str | None:
    attr = field.key.split(".", 1)[1] if "." in field.key else field.key
    alias = {
        "dp_path": "diffractive_pupil_path",
    }
    return alias.get(attr, attr)


def _structural_subset_from_contract(cfg: Any, contract: ParamSpec) -> dict[str, Any]:
    subset: dict[str, Any] = {}
    for field in contract.values():
        if not field.structural:
            continue
        attr = _config_attr_for_field(field)
        if attr is None:
            continue
        value = getattr(cfg, attr)
        subset[field.key] = _normalize_json_value(value)
    return subset


def _runtime_binding_fields(contract: ParamSpec) -> tuple[ParamField, ...]:
    return tuple(field for field in contract.values() if field.binding is not None)


def _binding_pairs(bindings_or_contract: ParamSpec | tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
    if isinstance(bindings_or_contract, ParamSpec):
        return tuple(
            (field.key, field.binding)
            for field in _runtime_binding_fields(bindings_or_contract)
            if field.binding is not None
        )
    return bindings_or_contract


def apply_runtime_bindings(
    optics,
    store: Optional[ParameterStore],
    bindings_or_contract: ParamSpec | tuple[tuple[str, str], ...],
):
    """Apply runtime ParameterStore overrides onto a cached optics object."""

    if store is None:
        return optics

    for store_key, set_path in _binding_pairs(bindings_or_contract):
        val = store.get(store_key, default=None)
        if val is None:
            continue
        optics = optics.set(set_path, jnp.asarray(val))
    return optics


def structural_hash_from_config(cfg: SheraThreePlaneConfig) -> str:
    """Return a deterministic structural hash for three-plane optics."""

    contract = build_threeplane_optics_contract(cfg)
    payload = {
        "optics_kind": "three_plane",
        "structural": _structural_subset_from_contract(cfg, contract),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def structural_hash_for_twoplane(cfg: SheraTwoPlaneConfig) -> str:
    """Return a deterministic structural hash for two-plane optics."""

    contract = build_twoplane_optics_contract(cfg)
    payload = {
        "optics_kind": "two_plane",
        "structural": _structural_subset_from_contract(cfg, contract),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def clear_threeplane_optics_cache() -> None:
    """Clear the cached Shera three-plane optics structures."""

    _THREEPLANE_CACHE.clear()


def clear_twoplane_optics_cache() -> None:
    """Clear the cached Shera two-plane optics structures."""

    _TWOPLANE_CACHE.clear()


def _load_diffractive_pupil_mask(cfg: SheraTwoPlaneConfig) -> dll.AberratedLayer:
    """Load the diffractive pupil mask for two-plane optics construction."""

    if cfg.diffractive_pupil_path is None:
        return dll.AberratedLayer(jnp.zeros((cfg.pupil_npix, cfg.pupil_npix)))

    mask_array = np.load(cfg.diffractive_pupil_path)
    return dll.AberratedLayer(jnp.asarray(mask_array))


def build_shera_threeplane_optics(
    cfg: SheraThreePlaneConfig,
    store: Optional[ParameterStore] = None,
    spec: Optional[ParamSpec] = None,
) -> SheraThreePlaneOptics:
    """Construct a three-plane optical system with structural caching."""

    contract = build_threeplane_optics_contract(cfg)

    if store is not None and spec is not None:
        store = store.validate_against(spec, allow_derived=True)

    cache_disabled = os.getenv(_CACHE_DISABLED_ENV, "").lower() in {"1", "true", "yes"}
    struct_hash = structural_hash_from_config(cfg)

    base_optics = None
    if not cache_disabled:
        base_optics = _THREEPLANE_CACHE.get(struct_hash)

    if base_optics is None:
        base_optics = SheraThreePlaneOptics(
            wf_npixels=cfg.pupil_npix,
            psf_npixels=cfg.psf_npix,
            oversample=cfg.oversample,
            detector_pixel_pitch=cfg.pixel_pitch_m,
            mask=cfg.diffractive_pupil_path,
            m1_noll_ind=tuple(cfg.primary_noll_indices)
            if cfg.primary_noll_indices
            else None,
            m2_noll_ind=tuple(cfg.secondary_noll_indices)
            if cfg.secondary_noll_indices
            else None,
            p1_diameter=cfg.m1_diameter_m,
            p2_diameter=cfg.m2_diameter_m,
            m1_focal_length=cfg.m1_focal_length_m,
            m2_focal_length=cfg.m2_focal_length_m,
            plane_separation=cfg.m1_m2_separation_m,
            n_struts=cfg.n_struts,
            strut_width=cfg.strut_width_m,
            strut_rotation_deg=cfg.strut_rotation_deg,
            dp_design_wavel=cfg.dp_design_wavelength_m,
        )
        if not cache_disabled:
            _THREEPLANE_CACHE[struct_hash] = base_optics

    optics = base_optics.set("wf_npixels", base_optics.wf_npixels)
    return apply_runtime_bindings(optics, store, contract)


def build_shera_twoplane_optics(
    cfg: SheraTwoPlaneConfig,
    store: Optional[ParameterStore] = None,
    spec: Optional[ParamSpec] = None,
) -> SheraTwoPlaneOptics:
    """Construct the Shera two-plane optical system with runtime overrides."""

    contract = build_twoplane_optics_contract(cfg)

    if store is not None and spec is not None:
        store = store.validate_against(spec, allow_derived=True)

    cache_disabled = os.getenv(_TWOPLANE_CACHE_DISABLED_ENV, "").lower() in {
        "1",
        "true",
        "yes",
    }
    struct_hash = structural_hash_for_twoplane(cfg)

    base_optics = None
    if not cache_disabled:
        base_optics = _TWOPLANE_CACHE.get(struct_hash)

    if base_optics is None:
        base_optics = SheraTwoPlaneOptics(
            wf_npixels=cfg.pupil_npix,
            psf_npixels=cfg.psf_npix,
            oversample=cfg.oversample,
            psf_pixel_scale=cfg.plate_scale_as_per_pix,
            mask=cfg.diffractive_pupil_path,
            m1_diameter=cfg.m1_diameter_m,
            m2_diameter=cfg.m2_diameter_m,
            n_struts=cfg.n_struts,
            strut_width=cfg.strut_width_m,
            strut_rotation_deg=cfg.strut_rotation_deg,
            dp_design_wavel=cfg.dp_design_wavelength_m,
            noll_indices=jnp.asarray(cfg.primary_noll_indices)
            if cfg.primary_noll_indices
            else None,
        )
        if not cache_disabled:
            _TWOPLANE_CACHE[struct_hash] = base_optics

    optics = base_optics.set("wf_npixels", base_optics.wf_npixels)
    return apply_runtime_bindings(optics, store, contract)


__all__ = [
    "apply_runtime_bindings",
    "build_shera_threeplane_optics",
    "build_shera_twoplane_optics",
    "clear_threeplane_optics_cache",
    "clear_twoplane_optics_cache",
    "structural_hash_from_config",
    "structural_hash_for_twoplane",
]
