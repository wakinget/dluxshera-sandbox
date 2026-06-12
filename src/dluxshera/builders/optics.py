"""Optics builder responsibilities (structural assembly and caching)."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Any
from collections.abc import Mapping

import numpy as np
import jax.numpy as jnp
import dLux.layers as dll

if TYPE_CHECKING:
    from ..systems.three_plane import SheraThreePlaneConfig
    from ..systems.two_plane import SheraTwoPlaneConfig
from ..components.optics import (
    SheraThreePlaneOptics,
    SheraTwoPlaneOptics,
)
from ..params.store import ParameterStore
from ..params.spec import ParamField, ParamSpec
from ..utils.high_order_wfe import realize_high_order_wfe_pair


_THREEPLANE_CACHE: dict[str, SheraThreePlaneOptics] = {}
_TWOPLANE_CACHE: dict[str, SheraTwoPlaneOptics] = {}
_CACHE_DISABLED_ENV = "DLUXSHERA_THREEPLANE_CACHE_DISABLED"
_TWOPLANE_CACHE_DISABLED_ENV = "DLUXSHERA_TWOPLANE_CACHE_DISABLED"


def _find_repo_root(start: Path) -> Path:
    """Walk parents until we find a repo marker. Fallback to start if none found."""

    start = start.resolve()
    for p in [start, *start.parents]:
        if (
            (p / ".git").exists()
            or (p / "pyproject.toml").exists()
            or (p / "setup.cfg").exists()
        ):
            return p
    return start


_REPO_ROOT = _find_repo_root(Path(__file__).resolve())


def _resolve_repo_path(path: str | Path | None) -> str | None:
    """Resolve package/repo-relative asset paths before handing them to dLux."""

    if path is None:
        return None
    p = Path(path).expanduser()
    if p.is_absolute():
        return str(p)
    return str((_REPO_ROOT / p).resolve())


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
        if isinstance(cfg, Mapping):
            value = cfg.get(attr)
        else:
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

    cfg_map = asdict(cfg) if is_dataclass(cfg) else cfg
    contract = SheraThreePlaneOptics.contract(cfg_map)
    payload = {
        "optics_kind": "three_plane",
        "structural": _structural_subset_from_contract(cfg_map, contract),
        "high_order_wfe": _normalize_json_value(cfg_map.get("high_order_wfe", {})),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def structural_hash_for_twoplane(cfg: SheraTwoPlaneConfig) -> str:
    """Return a deterministic structural hash for two-plane optics."""

    cfg_map = asdict(cfg) if is_dataclass(cfg) else cfg
    contract = SheraTwoPlaneOptics.contract(cfg_map)
    payload = {
        "optics_kind": "two_plane",
        "structural": _structural_subset_from_contract(cfg_map, contract),
        "high_order_wfe": _normalize_json_value(cfg_map.get("high_order_wfe", {})),
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

    mask_array = np.load(_resolve_repo_path(cfg.diffractive_pupil_path))
    return dll.AberratedLayer(jnp.asarray(mask_array))


def _surface_high_order_pair(optics_cfg: Mapping[str, Any], surface: str, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    how_cfg = optics_cfg.get("high_order_wfe") or {}
    if not how_cfg.get("enabled", False):
        z = np.zeros(shape)
        return z, z
    surf_cfg = how_cfg.get(surface) or {}
    if not surf_cfg.get("enabled", False):
        z = np.zeros(shape)
        return z, z
    real = realize_high_order_wfe_pair(
        shape,
        truth_cfg=surf_cfg.get("map") or surf_cfg,
        knowledge_cfg=surf_cfg.get("knowledge_error") or None,
    )
    return real.truth_opd_nm, real.inference_opd_nm


def _threeplane_cfg_map(cfg: Any) -> Mapping[str, Any]:
    if is_dataclass(cfg):
        cfg = asdict(cfg)
    if isinstance(cfg, Mapping):
        return cfg.get("optics", cfg)
    raise ValueError("build_shera_threeplane_optics expects a mapping or dataclass config.")


def build_shera_threeplane_optics(
    cfg: SheraThreePlaneConfig,
    store: Optional[ParameterStore] = None,
    spec: Optional[ParamSpec] = None,
) -> SheraThreePlaneOptics:
    """Construct a three-plane optical system with structural caching."""

    optics_cfg = _threeplane_cfg_map(cfg)

    contract = SheraThreePlaneOptics.contract(optics_cfg)

    if store is not None and spec is not None:
        store = store.validate_against(spec, allow_derived=True)

    cache_disabled = os.getenv(_CACHE_DISABLED_ENV, "").lower() in {"1", "true", "yes"}
    struct_hash = structural_hash_from_config(optics_cfg)

    base_optics = None
    if not cache_disabled:
        base_optics = _THREEPLANE_CACHE.get(struct_hash)

    if base_optics is None:
        m1_truth_nm, m1_inf_nm = _surface_high_order_pair(optics_cfg, "primary", (optics_cfg["pupil_npix"], optics_cfg["pupil_npix"]))
        m2_truth_nm, m2_inf_nm = _surface_high_order_pair(optics_cfg, "secondary", (optics_cfg["pupil_npix"], optics_cfg["pupil_npix"]))

        base_optics = SheraThreePlaneOptics(
            wf_npixels=optics_cfg["pupil_npix"],
            psf_npixels=optics_cfg["psf_npix"],
            oversample=optics_cfg["oversample"],
            detector_pixel_pitch=optics_cfg["pixel_pitch_m"],
            mask=_resolve_repo_path(
                optics_cfg.get("diffractive_pupil_path", optics_cfg.get("dp_path"))
            ),
            m1_noll_ind=tuple(optics_cfg.get("primary_noll_indices") or [])
            if optics_cfg.get("primary_noll_indices")
            else None,
            m2_noll_ind=tuple(optics_cfg.get("secondary_noll_indices") or [])
            if optics_cfg.get("secondary_noll_indices")
            else None,
            p1_diameter=optics_cfg["m1_diameter_m"],
            p2_diameter=optics_cfg["m2_diameter_m"],
            m1_focal_length=optics_cfg["m1_focal_length_m"],
            m2_focal_length=optics_cfg["m2_focal_length_m"],
            plane_separation=optics_cfg["m1_m2_separation_m"],
            n_struts=optics_cfg["n_struts"],
            strut_width=optics_cfg["strut_width_m"],
            strut_rotation_deg=optics_cfg["strut_rotation_deg"],
            dp_design_wavel=optics_cfg["dp_design_wavelength_m"],
            m1_high_order_wfe_opd_m=jnp.asarray(m1_inf_nm) * 1e-9,
            m2_high_order_wfe_opd_m=jnp.asarray(m2_inf_nm) * 1e-9,
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

    if is_dataclass(cfg):
        cfg = asdict(cfg)
    if isinstance(cfg, Mapping):
        optics_cfg = cfg.get("optics", cfg)
    else:
        optics_cfg = cfg

    contract = SheraTwoPlaneOptics.contract(optics_cfg)

    if store is not None and spec is not None:
        store = store.validate_against(spec, allow_derived=True)

    cache_disabled = os.getenv(_TWOPLANE_CACHE_DISABLED_ENV, "").lower() in {
        "1",
        "true",
        "yes",
    }
    struct_hash = structural_hash_for_twoplane(optics_cfg)

    base_optics = None
    if not cache_disabled:
        base_optics = _TWOPLANE_CACHE.get(struct_hash)

    if base_optics is None:
        m1_truth_nm, m1_inf_nm = _surface_high_order_pair(optics_cfg, "primary", (optics_cfg["pupil_npix"], optics_cfg["pupil_npix"]))

        base_optics = SheraTwoPlaneOptics(
            wf_npixels=optics_cfg["pupil_npix"],
            psf_npixels=optics_cfg["psf_npix"],
            oversample=optics_cfg["oversample"],
            psf_pixel_scale=optics_cfg["plate_scale_as_per_pix"],
            mask=_resolve_repo_path(
                optics_cfg.get("diffractive_pupil_path", optics_cfg.get("dp_path"))
            ),
            m1_diameter=optics_cfg["m1_diameter_m"],
            m2_diameter=optics_cfg["m2_diameter_m"],
            n_struts=optics_cfg["n_struts"],
            strut_width=optics_cfg["strut_width_m"],
            strut_rotation_deg=optics_cfg["strut_rotation_deg"],
            dp_design_wavel=optics_cfg["dp_design_wavelength_m"],
            m1_high_order_wfe_opd_m=jnp.asarray(m1_inf_nm) * 1e-9,
            high_order_wfe_opd_m=jnp.asarray(m1_inf_nm) * 1e-9,
            noll_indices=jnp.asarray(optics_cfg.get("primary_noll_indices") or [])
            if optics_cfg.get("primary_noll_indices")
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
