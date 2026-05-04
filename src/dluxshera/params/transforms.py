from __future__ import annotations

from contextlib import ExitStack
from importlib import resources
import math
from pathlib import Path
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np

from ..components.sources import get_target_spec
from ..utils.source_photometry import (
    build_wavelength_grid_m,
    derive_source_photometry,
    target_sed_root,
)
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
        "detector.pixel_pitch_m",
    ),
)
def transform_3P_plate_scale_as_per_pix(ctx: Ctx) -> float:
    """
    Compute the geometric plate scale in arcseconds per pixel.

        plate_scale_rad_per_pix = pixel_pitch_m / f_eff
        plate_scale_as_per_pix  = plate_scale_rad_per_pix * ARCSEC_PER_RAD
    """
    f_eff = float(ctx["optics.focal_length_m"])
    pixel_pitch = float(ctx["detector.pixel_pitch_m"])

    plate_scale_rad = pixel_pitch / f_eff
    plate_scale_as = plate_scale_rad * ARCSEC_PER_RAD
    return plate_scale_as


# ---------------------------------------------------------------------------
# Log-flux: source.log_flux_total
# ---------------------------------------------------------------------------


@register_for_systems(
    "source.log_flux_total",
    depends_on=(
        "source.target",
        "source.vmag_a",
        "source.vmag_b",
        "source.wavelength_m",
        "optics.m1_diameter_m",
        "source.bandwidth_m",
        "source.n_lambda",
        "source.exposure_time_s",
        "optics.throughput",
    ),
)
def transform_source_log_flux_total(ctx: Ctx) -> float:
    """
    Compute total detected ``log10(photons)`` from authoritative source photometry.

    This transform uses curated target component SEDs when available. If SEDs
    are missing, it falls back to a documented Johnson-V/Vega-style
    approximation using component V magnitudes.
    """
    target_key_raw = ctx.get("source.target", None)
    target_key = str(target_key_raw).strip() if target_key_raw not in (None, "") else None
    target_spec = get_target_spec(target_key) if target_key else None

    vmag_a_raw = ctx.get("source.vmag_a", None)
    vmag_b_raw = ctx.get("source.vmag_b", None)
    vmag_a = float(vmag_a_raw) if vmag_a_raw is not None else (target_spec.vmag_a if target_spec else None)
    vmag_b = float(vmag_b_raw) if vmag_b_raw is not None else (target_spec.vmag_b if target_spec else None)

    D = float(ctx["optics.m1_diameter_m"])
    wavelength_m = float(ctx["source.wavelength_m"])
    bandwidth_m = float(ctx["source.bandwidth_m"])
    n_lambda = int(ctx["source.n_lambda"])
    t_exp = float(ctx["source.exposure_time_s"])
    throughput = float(ctx["optics.throughput"])
    area_m2 = math.pi * (D / 2.0) ** 2
    wavelength_grid_m = build_wavelength_grid_m(
        wavelength_m=wavelength_m,
        bandwidth_m=bandwidth_m,
        n_lambda=n_lambda,
    )

    sed_a_ref = None
    sed_b_ref = None
    if target_spec and target_spec.sed_a_file and target_spec.sed_b_file:
        sed_root = target_sed_root()
        sed_a_ref = sed_root.joinpath(target_spec.sed_a_file)
        sed_b_ref = sed_root.joinpath(target_spec.sed_b_file)

    if sed_a_ref is not None and sed_b_ref is not None and sed_a_ref.is_file() and sed_b_ref.is_file():
        with ExitStack() as stack:
            sed_a_path = Path(stack.enter_context(resources.as_file(sed_a_ref)))
            sed_b_path = Path(stack.enter_context(resources.as_file(sed_b_ref)))
            photometry = derive_source_photometry(
                wavelength_grid_m=wavelength_grid_m,
                bandwidth_m=bandwidth_m,
                collecting_area_m2=area_m2,
                exposure_time_s=t_exp,
                throughput=throughput,
                sed_a_path=sed_a_path,
                sed_b_path=sed_b_path,
                vmag_a=vmag_a,
                vmag_b=vmag_b,
            )
    else:
        photometry = derive_source_photometry(
            wavelength_grid_m=wavelength_grid_m,
            bandwidth_m=bandwidth_m,
            collecting_area_m2=area_m2,
            exposure_time_s=t_exp,
            throughput=throughput,
            sed_a_path=None,
            sed_b_path=None,
            vmag_a=vmag_a,
            vmag_b=vmag_b,
        )

    return float(photometry.log_flux_total)


# ---------------------------------------------------------------------------
# Raw fluxes: source.raw_fluxes
# ---------------------------------------------------------------------------


def compute_source_raw_fluxes_from_log_flux_total_and_contrast(
    log_flux_total: Any,
    contrast: Any,
) -> jnp.ndarray:
    """Return source raw fluxes using the canonical Alpha Cen convention.

    Notes
    -----
    This helper is intentionally JAX-safe. It matches the documented transform
    semantics for ``source.raw_fluxes`` but avoids Python scalar coercion so it
    can be reused inside traced local inference objectives.
    """

    log_flux = jnp.asarray(log_flux_total, dtype=float)
    contrast_value = jnp.asarray(contrast, dtype=float)
    total_flux = jnp.power(jnp.asarray(10.0, dtype=float), log_flux)
    flux_b = total_flux / (jnp.asarray(1.0, dtype=float) + contrast_value)
    flux_a = contrast_value * flux_b
    return jnp.stack((flux_a, flux_b), axis=0)


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
    return np.asarray(
        compute_source_raw_fluxes_from_log_flux_total_and_contrast(
            log_flux,
            contrast,
        ),
        dtype=float,
    )
