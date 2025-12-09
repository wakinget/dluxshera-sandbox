from __future__ import annotations

import math
from typing import Any, Mapping

from .transforms import DEFAULT_SYSTEM_ID, register_transform
from .spec import ParamKey

# Type alias for the ctx mapping each transform receives
Ctx = Mapping[ParamKey, Any]

# Conversion factor: radians → arcseconds
ARCSEC_PER_RAD = 206264.8062470963551565  # 180 / pi * 3600

# ---------------------------------------------------------------------------
# Effective focal length: system.focal_length_m
# ---------------------------------------------------------------------------

@register_transform(
    "system.focal_length_m",
    depends_on=(
        "system.m1_focal_length_m",
        "system.m2_focal_length_m",
        "system.m1_m2_separation_m",
    ),
    system_id=DEFAULT_SYSTEM_ID,
)
def transform_system_focal_length_m(ctx: Ctx) -> float:
    """
    Compute the effective telescope focal length for the Shera two-mirror relay.

        1 / f_eff = 1 / f1 + 1 / f2 - sep / (f1 * f2)

    where:
        f1  = primary focal length
        f2  = secondary focal length
        sep = axial separation between mirrors
    """
    f1 = float(ctx["system.m1_focal_length_m"])
    f2 = float(ctx["system.m2_focal_length_m"])
    sep = float(ctx["system.m1_m2_separation_m"])

    denom = (1.0 / f1) + (1.0 / f2) - sep / (f1 * f2)
    # Optionally: guard against denom ≈ 0.0 and raise a TransformError.
    f_eff = 1.0 / denom
    return f_eff


# ---------------------------------------------------------------------------
# Plate scale: system.plate_scale_as_per_pix
# ---------------------------------------------------------------------------

@register_transform(
    "system.plate_scale_as_per_pix",
    depends_on=(
        "system.focal_length_m",
        "system.pixel_pitch_m",
    ),
    system_id=DEFAULT_SYSTEM_ID,
)
def transform_system_plate_scale_as_per_pix(ctx: Ctx) -> float:
    """
    Compute the geometric plate scale in arcseconds per pixel.

        plate_scale_rad_per_pix = pixel_pitch_m / f_eff
        plate_scale_as_per_pix  = plate_scale_rad_per_pix * ARCSEC_PER_RAD

    This is equivalent to:

        dLux.utils.rad2arcsec(pixel_pitch_m / f_eff)

    but kept self-contained to avoid a heavy dependency on dLux in the
    parameter transforms layer.
    """
    f_eff = float(ctx["system.focal_length_m"])
    pixel_pitch = float(ctx["system.pixel_pitch_m"])

    plate_scale_rad = pixel_pitch / f_eff
    plate_scale_as = plate_scale_rad * ARCSEC_PER_RAD
    return plate_scale_as


# ---------------------------------------------------------------------------
# Log-flux: binary.log_flux_total
# ---------------------------------------------------------------------------

@register_transform(
    "binary.log_flux_total",
    depends_on=(
        "system.m1_diameter_m",
        "band.bandwidth_m",
        "imaging.exposure_time_s",
        "imaging.throughput",
        "binary.spectral_flux_density",
    ),
    system_id=DEFAULT_SYSTEM_ID,
)
def transform_binary_log_flux_total(ctx: Ctx) -> float:
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
    D = float(ctx["system.m1_diameter_m"])
    bandwidth_m = float(ctx["band.bandwidth_m"])
    t_exp = float(ctx["imaging.exposure_time_s"])
    throughput = float(ctx["imaging.throughput"])
    flux_density = float(ctx["binary.spectral_flux_density"])

    area = math.pi * (D / 2.0) ** 2
    total_flux = flux_density * bandwidth_m * area * t_exp * throughput

    # total_flux should be > 0 for physical configurations
    if not (total_flux > 0.0):
        # Optional guard; you could also just let log10 blow up.
        raise ValueError(
            f"Non-positive total_flux={total_flux} in binary_log_flux_total "
            f"(check flux_density, bandwidth, area, exposure_time, throughput)."
        )

    log_flux = math.log10(total_flux)
    return log_flux
