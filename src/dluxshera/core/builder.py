from __future__ import annotations

from typing import Optional

import jax.numpy as jnp

from dluxshera.optics.config import SheraThreePlaneConfig
from dluxshera.params.spec import ParamSpec
from dluxshera.params.store import ParameterStore

from .modeling import SheraThreePlane_Model, SheraThreePlaneParams


def build_shera_threeplane_model(
    cfg: SheraThreePlaneConfig,
    spec: ParamSpec,
    store: ParameterStore,
) -> SheraThreePlane_Model:
    """
    Bridge from (config, ParamSpec, ParameterStore) to the legacy
    SheraThreePlane_Model API.

    This helper:

      * creates a SheraThreePlaneParams seeded from the point design implied
        by `cfg.design_name` (e.g. "shera_testbed", "shera_flight"),
      * overwrites geometric / bandpass fields from the SheraThreePlaneConfig,
      * copies inference-space values in the ParameterStore into the legacy
        SheraThreePlaneParams naming scheme, and
      * constructs and returns a SheraThreePlane_Model.

    For P0 we:
      - treat the SheraThreePlaneParams defaults as the source of “extra”
        knobs (1/f WFE, RNG seeds, etc.),
      - only map the parameters we’ve explicitly refactored into the new
        ParamSpec/ParameterStore (astrometry, flux, Zernikes, bandpass,
        basic geometry).
    """

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
