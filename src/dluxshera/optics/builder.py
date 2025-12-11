# src/dluxshera/optics/builder.py

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

import jax.numpy as jnp

from .config import SheraThreePlaneConfig
from ..params.store import ParameterStore
from ..params.spec import ParamSpec

try:
    # Legacy three-plane optics implementation, now living in the
    # refactored package under dluxshera.optics.optical_systems.
    from .optical_systems import SheraThreePlaneSystem
except ImportError as e:  # pragma: no cover - hard failure, not a logic branch
    raise ImportError(
        "SheraThreePlaneSystem could not be imported from "
        "dluxshera.optics.optical_systems. Make sure "
        "optical_systems.py defines SheraThreePlaneSystem and is "
        "installed/importable as part of dluxshera."
    ) from e


# -----------------------------------------------------------------------------
# Structural hash / cache helpers
# -----------------------------------------------------------------------------

_THREEPLANE_CACHE: dict[str, SheraThreePlaneSystem] = {}
_CACHE_DISABLED_ENV = "DLUXSHERA_THREEPLANE_CACHE_DISABLED"


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


def clear_threeplane_optics_cache() -> None:
    """Clear the cached Shera three-plane optics structures."""

    _THREEPLANE_CACHE.clear()


def _construct_threeplane_optics(cfg: SheraThreePlaneConfig) -> SheraThreePlaneSystem:
    """Actual constructor for the Shera three-plane optics (structural only)."""

    return SheraThreePlaneSystem(
        wf_npixels=cfg.pupil_npix,
        psf_npixels=cfg.psf_npix,
        oversample=cfg.oversample,
        detector_pixel_pitch=cfg.pixel_pitch_m,
        mask=cfg.diffractive_pupil_path,
        m1_noll_ind=tuple(cfg.primary_noll_indices) if cfg.primary_noll_indices else None,
        m2_noll_ind=tuple(cfg.secondary_noll_indices) if cfg.secondary_noll_indices else None,
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



def build_shera_threeplane_optics(
    cfg: SheraThreePlaneConfig,
    store: Optional[ParameterStore] = None,
    spec: Optional[ParamSpec] = None,
) -> SheraThreePlaneSystem:
    """
    Construct the legacy Shera three-plane optical system from a
    SheraThreePlaneConfig and (optionally) a ParameterStore.

    This is a compatibility wrapper around SheraThreePlaneSystem that:
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
        provided, this is used to populate `primary.zernike_coeffs`
        and `secondary.zernike_coeffs` (when present) into the optics.

    spec:
        Optional ParamSpec used to validate the store keys. If provided,
        `store.validate_against(spec)` is called before using values.
        This helps catch typos or misnamed parameters early.

    Notes
    -----
    - `pixel_pitch_m` is stored and passed in meters, matching the
      SheraThreePlaneSystem convention.
    - SheraThreePlaneSystem has been updated to accept strut_rotation in
      degrees, so we may pass it directly from the config.
    - Primary and secondary Zernike bases are selected via the Noll index
      tuples in the config. If no secondary indices are provided, the
      secondary mirror currently has no Zernike basis (i.e., it is modeled
      as a pure transmissive layer).
    - Optics structures are cached by a structural hash (see
      ``structural_hash_from_config``). Zernike coefficients remain outside
      the structural hash and are applied to a copy of the cached structure.
    """

    # --- Zernike coefficients from the ParameterStore (optional) -------
    m1_coefficients = None
    m2_coefficients = None

    if store is not None:
        # Optionally validate that the store keys are consistent with the spec.
        # Forward-model stores may legitimately contain derived values, so we
        # allow them here to keep the builder compatible with forward-style
        # binders.
        if spec is not None:
            store = store.validate_against(spec, allow_derived=True)

        # Expected lengths based on the config's Noll index tuples.
        n_m1 = len(cfg.primary_noll_indices) if cfg.primary_noll_indices else 0
        n_m2 = len(cfg.secondary_noll_indices) if cfg.secondary_noll_indices else 0

        # Primary mirror coefficients
        if n_m1 > 0:
            coeffs = store.get("primary.zernike_coeffs", default=None)
            if coeffs is not None:
                coeffs = jnp.asarray(coeffs)
                if coeffs.shape[0] != n_m1:
                    raise ValueError(
                        f"primary.zernike_coeffs length {coeffs.shape[0]} does not "
                        f"match number of primary_noll_indices {n_m1}."
                    )
                m1_coefficients = coeffs

        # Secondary mirror coefficients
        if n_m2 > 0:
            coeffs = store.get("secondary.zernike_coeffs", default=None)
            if coeffs is not None:
                coeffs = jnp.asarray(coeffs)
                if coeffs.shape[0] != n_m2:
                    raise ValueError(
                        f"secondary.zernike_coeffs length {coeffs.shape[0]} does not "
                        f"match number of secondary_noll_indices {n_m2}."
                    )
                m2_coefficients = coeffs

    # --- Construct or reuse the optics ---------------------------------
    cache_disabled = os.getenv(_CACHE_DISABLED_ENV, "").lower() in {"1", "true", "yes"}
    struct_hash = structural_hash_from_config(cfg)

    base_optics = None
    if not cache_disabled:
        base_optics = _THREEPLANE_CACHE.get(struct_hash)

    if base_optics is None:
        base_optics = _construct_threeplane_optics(cfg)
        if not cache_disabled:
            _THREEPLANE_CACHE[struct_hash] = base_optics

    # Create a shallow functional copy so callers cannot mutate the cached
    # structure. Using `.set` preserves JAX pytree semantics without
    # re-running the heavy constructor.
    optics = base_optics.set("wf_npixels", base_optics.wf_npixels)

    # Apply Zernike coefficients without mutating the cached structure
    if m1_coefficients is not None and hasattr(optics, "m1_aperture"):
        m1_aperture = getattr(optics, "m1_aperture")
        if hasattr(m1_aperture, "coefficients"):
            optics = optics.set("p1_layers.m1_aperture.coefficients", m1_coefficients)

    if m2_coefficients is not None and hasattr(optics, "m2_aperture"):
        m2_aperture = getattr(optics, "m2_aperture")
        if hasattr(m2_aperture, "coefficients"):
            optics = optics.set("p2_layers.m2_aperture.coefficients", m2_coefficients)

    return optics
