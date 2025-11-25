# src/dluxshera/optics/builder.py

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp

from .config import SheraThreePlaneConfig

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
) -> SheraThreePlaneSystem:
    """
    Construct the legacy Shera three-plane optical system from a
    SheraThreePlaneConfig.

    This is a thin compatibility wrapper around SheraThreePlaneSystem that
    translates from the new config schema (meters, degrees, tuples, etc.)
    into the argument conventions used by the existing optics class.

    Notes
    -----
    - `strut_rotation_deg` is stored in degrees in the config, while the
      legacy code currently expects radians, so we convert here. In the
      future we may update SheraThreePlaneSystem to accept degrees directly.
    """

    # --- Noll indices ---------------------------------------------------
    # Primary
    if cfg.primary_noll_indices and len(cfg.primary_noll_indices) > 0:
        m1_noll_ind = tuple(cfg.primary_noll_indices)
    else:
        m1_noll_ind = None

    # Secondary
    if cfg.secondary_noll_indices and len(cfg.secondary_noll_indices) > 0:
        m2_noll_ind = tuple(cfg.secondary_noll_indices)
    else:
        m2_noll_ind = None


    # --- Angles: degrees → radians -------------------------------------
    strut_rotation_rad = cfg.strut_rotation_deg * jnp.pi / 180.0

    # --- Construct the optics ------------------------------------------
    optics = SheraThreePlaneSystem(
        wf_npixels=cfg.pupil_npix,
        psf_npixels=cfg.psf_npix,
        oversample=cfg.oversample,
        detector_pixel_pitch=cfg.detector_pixel_pitch_m,
        mask=cfg.diffractive_pupil_path,
        m1_noll_ind=m1_noll_ind,
        m2_noll_ind=m2_noll_ind,
        p1_diameter=cfg.m1_diameter_m,
        p2_diameter=cfg.m2_diameter_m,
        m1_focal_length=cfg.m1_focal_length_m,
        m2_focal_length=cfg.m2_focal_length_m,
        plane_separation=cfg.m1_m2_separation_m,
        n_struts=cfg.n_struts,
        strut_width=cfg.strut_width_m,
        strut_rotation=strut_rotation_rad,
        dp_design_wavel=cfg.dp_design_wavelength_m,
    )

    return optics
