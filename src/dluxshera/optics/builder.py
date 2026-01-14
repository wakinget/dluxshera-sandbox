# src/dluxshera/optics/builder.py

from __future__ import annotations

import hashlib
import json
import os
import warnings
from typing import Optional

import numpy as np
import jax.numpy as jnp
import dLux.layers as dll

from .config import SheraThreePlaneConfig, SheraTwoPlaneConfig
from ..params.store import ParameterStore
from ..params.spec import ParamSpec

try:
    # Legacy three-plane optics implementation, now living in the
    # refactored package under dluxshera.optics.optical_systems.
    from .optical_systems import SheraThreePlaneOptics, SheraTwoPlaneOptics
except ImportError as e:  # pragma: no cover - hard failure, not a logic branch
    raise ImportError(
        "SheraThreePlaneOptics could not be imported from "
        "dluxshera.optics.optical_systems. Make sure "
        "optical_systems.py defines SheraThreePlaneOptics and is "
        "installed/importable as part of dluxshera."
    ) from e


# -----------------------------------------------------------------------------
# Structural hash / cache helpers
# -----------------------------------------------------------------------------

_THREEPLANE_CACHE: dict[str, SheraThreePlaneOptics] = {}
_TWOPLANE_CACHE: dict[str, SheraTwoPlaneOptics] = {}
_CACHE_DISABLED_ENV = "DLUXSHERA_THREEPLANE_CACHE_DISABLED"
_TWOPLANE_CACHE_DISABLED_ENV = "DLUXSHERA_TWOPLANE_CACHE_DISABLED"

# Runtime bindings apply post-cache overrides onto cached optics objects.
# They exist because structural caching keys off the config only, so any
# parameter baked into the cached optics (Zernike coefficients, plate scale,
# etc.) must be reapplied from the ParameterStore after cache lookup.
# Include only cached-optics knobs here (values that live inside the optics
# object). Do not include source/noise/detector parameters; those are consumed
# elsewhere and are not part of the cached optics state. When adding new
# bindings, use full optics.set paths and ensure the path is stable in the
# optics implementation so missing paths fail loudly.
THREEPLANE_RUNTIME_BINDINGS: tuple[tuple[str, str], ...] = (
    ("primary.zernike_coeffs_nm", "p1_layers.m1_aperture.coefficients"),
    ("secondary.zernike_coeffs_nm", "p2_layers.m2_aperture.coefficients"),
    ("system.plate_scale_as_per_pix", "psf_pixel_scale"),
)
TWOPLANE_RUNTIME_BINDINGS: tuple[tuple[str, str], ...] = (
    ("primary.zernike_coeffs_nm", "layers.aperture.coefficients"),
    ("system.plate_scale_as_per_pix", "psf_pixel_scale"),
)


def apply_runtime_bindings(optics, store, bindings):
    """Apply runtime ParameterStore overrides onto a cached optics object."""

    if store is None:
        return optics

    for store_key, set_path in bindings:
        val = store.get(store_key, default=None)
        if val is None:
            continue
        optics = optics.set(set_path, jnp.asarray(val))
    return optics


def _structural_subset(cfg: SheraThreePlaneConfig) -> dict:
    """Extract the structural subset of ``cfg`` as plain Python types."""

    return {
        "pupil_npix": int(cfg.pupil_npix),
        "psf_npix": int(cfg.psf_npix),
        "oversample": int(cfg.oversample),
        "wavelength_m": float(cfg.wavelength_m),
        "bandwidth_m": float(cfg.bandwidth_m),
        "n_lambda": int(cfg.n_lambda),
        "m1_diameter_m": float(cfg.m1_diameter_m),
        "m2_diameter_m": float(cfg.m2_diameter_m),
        "m1_focal_length_m": float(cfg.m1_focal_length_m),
        "m2_focal_length_m": float(cfg.m2_focal_length_m),
        "m1_m2_separation_m": float(cfg.m1_m2_separation_m),
        "pixel_pitch_m": float(cfg.pixel_pitch_m),
        "n_struts": int(cfg.n_struts),
        "strut_width_m": float(cfg.strut_width_m),
        "strut_rotation_deg": float(cfg.strut_rotation_deg),
        "primary_noll_indices": tuple(int(i) for i in cfg.primary_noll_indices),
        "secondary_noll_indices": tuple(int(i) for i in cfg.secondary_noll_indices),
        "diffractive_pupil_path": None if cfg.diffractive_pupil_path is None else str(cfg.diffractive_pupil_path),
        "dp_design_wavelength_m": None
        if cfg.dp_design_wavelength_m is None
        else float(cfg.dp_design_wavelength_m),
    }


def _twoplane_structural_subset(cfg: SheraTwoPlaneConfig) -> dict:
    """Extract structural fields for the two-plane optics.

    Parameters
    ----------
    cfg
        Two-plane configuration describing the fixed geometry and sampling.

    Notes
    -----
    Plate scale is intentionally excluded so it can be updated at runtime via
    the ParameterStore without invalidating the cached optics geometry.
    """

    return {
        "pupil_npix": int(cfg.pupil_npix),
        "psf_npix": int(cfg.psf_npix),
        "oversample": int(cfg.oversample),
        "wavelength_m": float(cfg.wavelength_m),
        "bandwidth_m": float(cfg.bandwidth_m),
        "n_lambda": int(cfg.n_lambda),
        "m1_diameter_m": float(cfg.m1_diameter_m),
        "central_obscuration_ratio": float(cfg.central_obscuration_ratio),
        "n_struts": int(cfg.n_struts),
        "strut_width_m": float(cfg.strut_width_m),
        "strut_rotation_deg": float(cfg.strut_rotation_deg),
        "primary_noll_indices": tuple(int(i) for i in cfg.primary_noll_indices),
        "diffractive_pupil_path": None
        if cfg.diffractive_pupil_path is None
        else str(cfg.diffractive_pupil_path),
        "dp_design_wavelength_m": None
        if cfg.dp_design_wavelength_m is None
        else float(cfg.dp_design_wavelength_m),
    }


def structural_hash_from_config(cfg: SheraThreePlaneConfig) -> str:
    """Return a deterministic structural hash for ``cfg``.

    The hash is stable across runs and depends only on structural fields. It is
    used to cache optics builds so that non-structural updates (e.g., Zernike
    coefficients in the ParameterStore) do not force a rebuild of the optics
    geometry.
    """

    payload = _structural_subset(cfg)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def structural_hash_for_twoplane(cfg: SheraTwoPlaneConfig) -> str:
    """Return a deterministic structural hash for the two-plane optics stack.

    Notes
    -----
    This hash includes only geometry, sampling, and diffractive pupil settings
    from ``cfg``. Runtime knobs (plate scale, Zernike coefficients) are applied
    after cached optics are materialized and are excluded from the hash.
    """

    payload = _twoplane_structural_subset(cfg)
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
        # Avoid reliance on external DP assets in tests/demos; default to a
        # clear pupil by supplying a zero-OPD mask explicitly.
        return dll.AberratedLayer(jnp.zeros((cfg.pupil_npix, cfg.pupil_npix)))

    mask_array = np.load(cfg.diffractive_pupil_path)
    return dll.AberratedLayer(jnp.asarray(mask_array))


def build_shera_threeplane_optics(
    cfg: SheraThreePlaneConfig,
    store: Optional[ParameterStore] = None,
    spec: Optional[ParamSpec] = None,
) -> SheraThreePlaneOptics:
    """
    Construct the legacy Shera three-plane optical system from a
    SheraThreePlaneConfig and (optionally) a ParameterStore.

    This is a compatibility wrapper around SheraThreePlaneOptics that:
      - translates from the new config schema (meters, degrees, tuples, etc.)
        into the argument conventions used by the existing optics class, and
      - optionally injects Zernike coefficients from a ParameterStore.

    Parameters
    ----------
    cfg:
        Structural configuration for the three-plane optics (geometry,
        grids, Zernike basis structure, DP file path, etc.).

    store:
        Optional ParameterStore holding numeric parameter values. If
        provided, this is used to populate `primary.zernike_coeffs_nm`
        and `secondary.zernike_coeffs_nm` (when present) into the optics.

    spec:
        Optional ParamSpec used to validate the store keys. If provided,
        `store.validate_against(spec)` is called before using values.
        This helps catch typos or misnamed parameters early.

    Notes
    -----
    - Structural fields for caching include the config geometry, sampling,
      Zernike basis indices, and diffractive pupil settings (see
      ``structural_hash_from_config``). Values pulled from the store are used
      only for non-structural coefficients and plate-scale overrides.
    - Zernike coefficients are non-structural; they are applied after loading
      the cached geometry so the cache remains reusable across coefficient
      updates.
    - `pixel_pitch_m` is stored and passed in meters, matching the
      SheraThreePlaneOptics convention.
    - SheraThreePlaneOptics has been updated to accept strut_rotation in
      degrees, so we may pass it directly from the config.
    - Primary and secondary Zernike bases are selected via the Noll index
      tuples in the config. If no secondary indices are provided, the
      secondary mirror currently has no Zernike basis (i.e., it is modeled
      as a pure transmissive layer).
    - Optics structures are cached by a structural hash (see
      ``structural_hash_from_config``). Zernike coefficients remain outside
      the structural hash and are applied to a copy of the cached structure.
    """

    if store is not None:
        # Optionally validate that the store keys are consistent with the spec.
        # Forward-model stores may legitimately contain derived values, so we
        # allow them here to keep the builder compatible with forward-style
        # binders.
        if spec is not None:
            store = store.validate_against(spec, allow_derived=True)

    # --- Construct or reuse the optics ---------------------------------
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

    # Create a shallow functional copy so callers cannot mutate the cached
    # structure. Using `.set` preserves JAX pytree semantics without
    # re-running the heavy constructor.
    optics = base_optics.set("wf_npixels", base_optics.wf_npixels)

    optics = apply_runtime_bindings(optics, store, THREEPLANE_RUNTIME_BINDINGS)

    return optics


def build_shera_twoplane_optics(
    cfg: SheraTwoPlaneConfig,
    store: Optional[ParameterStore] = None,
    spec: Optional[ParamSpec] = None,
) -> SheraTwoPlaneOptics:
    """
    Construct the Shera two-plane optical system with runtime overrides.

    Parameters
    ----------
    cfg
        Structural configuration for the two-plane optics (geometry, grids,
        diffractive pupil settings, Zernike basis selection).
    store
        Optional ParameterStore providing runtime overrides such as plate
        scale or Zernike coefficients. When provided, the store is validated
        against ``spec`` before its values are consumed.
    spec
        Optional ParamSpec used to validate the store keys.

    Notes
    -----
    - The cached structural geometry is keyed only on ``cfg`` fields (see
      ``structural_hash_for_twoplane``). Plate scale is applied as a runtime
      update on the returned optics instance.
    - Zernike coefficients are treated as non-structural and applied after
      caching to keep the base optics reusable across inference updates.
    """

    plate_scale = cfg.plate_scale_as_per_pix
    if store is not None:
        if spec is not None:
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
        strut_rotation = float(cfg.strut_rotation_deg)
        mask = _load_diffractive_pupil_mask(cfg)
        base_optics = SheraTwoPlaneOptics(
            wf_npixels=cfg.pupil_npix,
            psf_npixels=cfg.psf_npix,
            oversample=cfg.oversample,
            psf_pixel_scale=plate_scale,
            m1_diameter=cfg.m1_diameter_m,
            m2_diameter=cfg.central_obscuration_ratio * cfg.m1_diameter_m,
            n_struts=cfg.n_struts,
            strut_width=cfg.strut_width_m,
            strut_rotation=jnp.deg2rad(strut_rotation),
            mask=mask,
            dp_design_wavel=cfg.dp_design_wavelength_m,
            noll_indices=tuple(cfg.primary_noll_indices)
            if cfg.primary_noll_indices
            else None,
        )
        if not cache_disabled:
            _TWOPLANE_CACHE[struct_hash] = base_optics

    optics = base_optics.set("wf_npixels", base_optics.wf_npixels)

    optics = apply_runtime_bindings(optics, store, TWOPLANE_RUNTIME_BINDINGS)

    return optics


# -----------------------------------------------------------------------------
# Legacy Bridge
# -----------------------------------------------------------------------------


def build_legacy_shera_threeplane_model(
    cfg: SheraThreePlaneConfig,
    spec: ParamSpec,
    store: ParameterStore,
):
    """
    Legacy bridge from (config, ParamSpec, ParameterStore) to SheraThreePlane_Model.

    This helper exists for backwards compatibility with legacy
    ``SheraThreePlaneParams`` and ``SheraThreePlane_Model`` usage. New workflows
    should use the binder-based pipeline instead of this bridge.
    """

    from ..legacy.modeling import SheraThreePlane_Model
    from ..legacy.params import SheraThreePlaneParams

    warnings.warn(
        "build_legacy_shera_threeplane_model is deprecated and exists only for "
        "legacy SheraThreePlaneParams/SheraThreePlane_Model usage. Prefer the "
        "binder-based pipeline for new workflows.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Optional: ensure the store doesn’t contain unknown keys for this spec.
    store = store.validate_against(spec)

    # ------------------------------------------------------------------
    # 1) Seed legacy params from the point design, if available
    # ------------------------------------------------------------------
    point_design = cfg.design_name   # usually "shera_testbed" / "shera_flight"
    params = SheraThreePlaneParams(point_design=point_design)

    # ------------------------------------------------------------------
    # 2) Geometry + sampling from config
    # ------------------------------------------------------------------
    # Keep this explicit so it’s easy to see what is “owned” by config.
    params = params.set("pupil_npix", cfg.pupil_npix)
    params = params.set("psf_npix", cfg.psf_npix)
    params = params.set("pixel_size", cfg.pixel_pitch_m)

    params = params.set("p1_diameter", cfg.m1_diameter_m)
    params = params.set("p2_diameter", cfg.m2_diameter_m)
    params = params.set("m1_focal_length", cfg.m1_focal_length_m)
    params = params.set("m2_focal_length", cfg.m2_focal_length_m)
    params = params.set("plane_separation", cfg.m1_m2_separation_m)

    # Zernike basis structure: mirror the config’s Noll index tuples.
    if cfg.primary_noll_indices:
        params = params.set(
            "m1_zernike_noll",
            jnp.asarray(cfg.primary_noll_indices, dtype=jnp.int32),
        )
    else:
        params = params.set("m1_zernike_noll", None)

    if cfg.secondary_noll_indices:
        params = params.set(
            "m2_zernike_noll",
            jnp.asarray(cfg.secondary_noll_indices, dtype=jnp.int32),
        )
    else:
        params = params.set("m2_zernike_noll", None)

    # ------------------------------------------------------------------
    # 3) Bandpass from config (meters → nanometers for legacy params)
    # ------------------------------------------------------------------
    wavelength_nm = cfg.wavelength_m * 1e9
    bandwidth_nm = cfg.bandwidth_m * 1e9

    params = params.set("wavelength", wavelength_nm)
    params = params.set("bandwidth", bandwidth_nm)
    params = params.set("n_wavelengths", cfg.n_lambda)

    # ------------------------------------------------------------------
    # 4) Astrometry + photometry: new inference keys → legacy names
    # ------------------------------------------------------------------
    # These keys come from build_inference_spec_basic().
    # For now we assume they are present; if not, KeyError is a good signal.
    sep_as = store.get("binary.separation_as")
    pa_deg = store.get("binary.position_angle_deg")
    x_as = store.get("binary.x_position_as")
    y_as = store.get("binary.y_position_as")
    contrast = store.get("binary.contrast")
    log_flux = store.get("binary.log_flux_total")

    params = params.set("separation", sep_as)
    params = params.set("position_angle", pa_deg)
    params = params.set("x_position", x_as)
    params = params.set("y_position", y_as)
    params = params.set("contrast", contrast)
    params = params.set("log_flux", log_flux)

    # ------------------------------------------------------------------
    # 5) Wavefront: Zernike WFE coefficients (nm)
    # ------------------------------------------------------------------
    # These map cleanly onto the legacy params because SheraThreePlane_Model
    # scales the basis by 1e-9 so nm-valued coefficients become OPD in meters.
    m1_coeffs = store.get("primary.zernike_coeffs_nm", default=None)
    if m1_coeffs is not None:
        params = params.set("m1_zernike_amp", jnp.asarray(m1_coeffs))

    m2_coeffs = store.get("secondary.zernike_coeffs_nm", default=None)
    if m2_coeffs is not None:
        params = params.set("m2_zernike_amp", jnp.asarray(m2_coeffs))

    # (P0) We leave all the 1/f WFE knobs, RNG seeds, etc. at whatever
    # SheraThreePlaneParams(point_design=...) chose as defaults. Later we can
    # add them to a richer ParamSpec and map them here the same way.

    # ------------------------------------------------------------------
    # 6) Construct and return the legacy Telescope wrapper
    # ------------------------------------------------------------------
    model = SheraThreePlane_Model(params=params)
    return model
