from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .spec import ParamKey
from .transform_registry import DEFAULT_SYSTEM_ID, register_transform

# Type alias for the ctx mapping each transform receives
Ctx = Mapping[ParamKey, Any]

SYSTEM_IDS = (DEFAULT_SYSTEM_ID, "shera_twoplane")


def register_for_systems(
    key: ParamKey,
    *,
    depends_on: tuple[ParamKey, ...] = (),
    doc: str | None = None,
):
    def decorator(fn):
        for system_id in SYSTEM_IDS:
            register_transform(
                key,
                depends_on=depends_on,
                doc=doc,
                system_id=system_id,
            )(fn)
        return fn

    return decorator

# Conversion factor: radians → arcseconds
ARCSEC_PER_RAD = 206264.8062470963551565  # 180 / pi * 3600

# ---------------------------------------------------------------------------
# Effective focal length: system.focal_length_m
# ---------------------------------------------------------------------------


@register_for_systems(
    "optics.focal_length_m",
    depends_on=(
        "optics.m1_focal_length_m",
        "optics.m2_focal_length_m",
        "optics.m1_m2_separation_m",
    ),
)
def transform_3P_focal_length_m(ctx: Ctx) -> float:
    """
    Compute the effective telescope focal length for the Shera 3-plane system.

        1 / f_eff = 1 / f1 + 1 / f2 - sep / (f1 * f2)

    where:
        f1  = primary focal length
        f2  = secondary focal length
        sep = axial separation between mirrors
    """
    f1 = float(ctx["optics.m1_focal_length_m"])
    f2 = float(ctx["optics.m2_focal_length_m"])
    sep = float(ctx["optics.m1_m2_separation_m"])

    denom = (1.0 / f1) + (1.0 / f2) - sep / (f1 * f2)
    # Optionally: guard against denom ≈ 0.0 and raise a TransformError.
    f_eff = 1.0 / denom
    return f_eff


# ---------------------------------------------------------------------------
# Plate scale: system.plate_scale_as_per_pix
# ---------------------------------------------------------------------------


@register_for_systems(
    "optics.plate_scale_as_per_pix",
    depends_on=(
        "optics.focal_length_m",
        "optics.pixel_pitch_m",
    ),
)
def transform_3P_plate_scale_as_per_pix(ctx: Ctx) -> float:
    """
    Compute the geometric plate scale in arcseconds per pixel.

        plate_scale_rad_per_pix = pixel_pitch_m / f_eff
        plate_scale_as_per_pix  = plate_scale_rad_per_pix * ARCSEC_PER_RAD
    """
    f_eff = float(ctx["optics.focal_length_m"])
    pixel_pitch = float(ctx["optics.pixel_pitch_m"])

    plate_scale_rad = pixel_pitch / f_eff
    plate_scale_as = plate_scale_rad * ARCSEC_PER_RAD
    return plate_scale_as


# ---------------------------------------------------------------------------
# Log-flux: source.log_flux_total
# ---------------------------------------------------------------------------


@register_for_systems(
    "source.log_flux_total",
    depends_on=(
        "optics.m1_diameter_m",
        "source.bandwidth_m",
        "source.exposure_time_s",
        "optics.throughput",
        "source.spectral_flux_density",
    ),
)
def transform_source_log_flux_total(ctx: Ctx) -> float:
    """
    Compute the truth-level log10 total photon count over the exposure.

    Model:

        area         = π (D / 2)^2
        total_flux   = spectral_flux_density * bandwidth_m
                       * area * exposure_time_s * throughput
        log_flux_tot = log10(total_flux)

    where:
        D                      = primary mirror diameter [m]
        spectral_flux_density  = mean photon flux density at the pupil in
                                 ph/s/m^2 per *meter* of band
        bandwidth_m            = bandpass width [m]
        exposure_time_s        = integration time [s]
        throughput             = end-to-end efficiency (0–1)
    """
    D = float(ctx["optics.m1_diameter_m"])
    bandwidth_m = float(ctx["source.bandwidth_m"])
    t_exp = float(ctx["source.exposure_time_s"])
    throughput = float(ctx["optics.throughput"])
    flux_density = float(ctx["source.spectral_flux_density"])

    area = math.pi * (D / 2.0) ** 2
    total_flux = flux_density * bandwidth_m * area * t_exp * throughput

    # total_flux should be > 0 for physical configurations
    if not (total_flux > 0.0):
        # Optional guard; you could also just let log10 blow up.
        raise ValueError(
            f"Non-positive total_flux={total_flux} in source_log_flux_total "
            "(check flux_density, bandwidth, area, exposure_time, throughput)."
        )

    log_flux = math.log10(total_flux)
    return log_flux


# ---------------------------------------------------------------------------
# Raw fluxes: source.raw_fluxes
# ---------------------------------------------------------------------------


@register_for_systems(
    "source.raw_fluxes",
    depends_on=(
        "source.log_flux_total",
        "source.contrast",
    ),
)
def transform_source_raw_fluxes(ctx: Ctx) -> np.ndarray:
    """
    Compute raw fluxes for the binary pair (photons for star A and B).

    This mirrors the AlphaCen source model:

        total_flux = 10 ** log_flux_total
        flux_A = total_flux * contrast / (1 + contrast)
        flux_B = total_flux / (1 + contrast)
    """
    log_flux = float(ctx["source.log_flux_total"])
    contrast = float(ctx["source.contrast"])

    total_flux = 10.0 ** log_flux
    flux_B = total_flux / (1.0 + contrast)
    flux_A = contrast * flux_B

    return np.asarray([flux_A, flux_B])
