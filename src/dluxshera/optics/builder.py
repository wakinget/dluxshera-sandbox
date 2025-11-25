# src/dluxshera/optics/builder.py

from __future__ import annotations

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
    - `detector_pixel_pitch_m` is stored and passed in meters, matching the
      SheraThreePlaneSystem convention.
    - SheraThreePlaneSystem has been updated to accept strut_rotation in
      degrees, so we may pass it directly from the config.
    - Primary and secondary Zernike bases are selected via the Noll index
      tuples in the config. If no secondary indices are provided, the
      secondary mirror currently has no Zernike basis (i.e., it is modeled
      as a pure transmissive layer).
    """

    # --- Noll indices: structural configuration ------------------------
    # Primary: tuple of Noll indices or None (no basis).
    if cfg.primary_noll_indices and len(cfg.primary_noll_indices) > 0:
        m1_noll_ind = tuple(cfg.primary_noll_indices)
    else:
        m1_noll_ind = None

    # Secondary: tuple of Noll indices or None (no basis).
    if cfg.secondary_noll_indices and len(cfg.secondary_noll_indices) > 0:
        m2_noll_ind = tuple(cfg.secondary_noll_indices)
    else:
        m2_noll_ind = None

    # --- Zernike coefficients from the ParameterStore (optional) -------
    m1_coefficients = None
    m2_coefficients = None

    if store is not None:
        # Optionally validate that the store keys are consistent with the spec.
        if spec is not None:
            store = store.validate_against(spec)

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


    # --- Construct the optics ------------------------------------------
    optics = SheraThreePlaneSystem(
        wf_npixels=cfg.pupil_npix,
        psf_npixels=cfg.psf_npix,
        oversample=cfg.oversample,
        detector_pixel_pitch=cfg.detector_pixel_pitch_m,
        mask=cfg.diffractive_pupil_path,
        m1_noll_ind=m1_noll_ind,
        m1_coefficients=m1_coefficients,
        m2_noll_ind=m2_noll_ind,
        m2_coefficients=m2_coefficients,
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

    return optics
