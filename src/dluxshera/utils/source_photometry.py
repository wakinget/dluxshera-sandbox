"""Helpers for target photometry seeding from curated SEDs or V magnitudes.

This module centralizes source-photometry seeding for binary targets:

- curated SED-backed photometry (authoritative path),
- simple Johnson-V/Vega-style fallback from component V magnitudes.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Optional

import numpy as np


# Physical constants (exact SI values).
PLANCK_CONSTANT_J_S = 6.62607015e-34
SPEED_OF_LIGHT_M_PER_S = 299792458.0

# Fallback Johnson-V/Vega-style assumptions.
# - Zero-mag spectral irradiance near Johnson V:
#     F_lambda,0 ~= 3.631e-8 W / m^2 / um
# - Effective reference wavelength:
#     lambda_V ~= 550 nm
JOHNSON_V_ZERO_POINT_W_M2_UM = 3.631e-8
JOHNSON_V_REFERENCE_WAVELENGTH_M = 550e-9


@dataclass(frozen=True)
class SourcePhotometry:
    """Broadband source photometry seed for the AlphaCen-style binary model.

    Parameters
    ----------
    weights:
        Array of shape ``(2, n_lambda)`` containing normalized component
        spectral weights (A then B). Each row sums to 1.
    contrast:
        Broadband flux ratio ``A/B`` over the active modeled bandpass.
    log_flux_total:
        ``log10`` of total detected photons from both components over the
        exposure, after collecting area and throughput are applied.
    component_fluxes_ph_s_m2:
        Broadband component photon irradiances at the pupil
        ``(A, B)`` in ``photons / s / m^2``.
    mode:
        Seeding mode label: ``"sed"`` or ``"vmag_fallback"``.
    """

    weights: np.ndarray
    contrast: float
    log_flux_total: float
    component_fluxes_ph_s_m2: tuple[float, float]
    mode: str


def target_sed_root() -> resources.abc.Traversable:
    """Return the packaged directory containing curated target SED files."""

    return resources.files("dluxshera").joinpath("data", "target_seds")


def build_wavelength_grid_m(
    wavelength_m: float,
    bandwidth_m: float,
    n_lambda: int,
) -> np.ndarray:
    """Return the active model wavelength grid in meters.

    Parameters
    ----------
    wavelength_m:
        Band center wavelength in meters.
    bandwidth_m:
        Bandwidth in meters.
    n_lambda:
        Number of discrete wavelength samples.

    Returns
    -------
    np.ndarray
        Monotonic wavelength grid of shape ``(n_lambda,)`` in meters.
    """

    if n_lambda <= 0:
        raise ValueError("n_lambda must be positive.")
    half_bw = 0.5 * float(bandwidth_m)
    center = float(wavelength_m)
    return np.linspace(center - half_bw, center + half_bw, int(n_lambda))


def load_sed_photon_flux_density_per_nm(
    sed_path: Path,
    wavelength_grid_m: np.ndarray,
) -> np.ndarray:
    """Interpolate an SED onto the model grid as photon spectral irradiance.

    Parameters
    ----------
    sed_path:
        Path to a text SED table with at least two columns:
        wavelength in nm and flux density in ``W / m^2 / nm``.
    wavelength_grid_m:
        Model wavelength grid in meters.

    Returns
    -------
    np.ndarray
        Photon spectral irradiance on the model grid in
        ``photons / s / m^2 / nm``.
    """

    table = np.loadtxt(sed_path, ndmin=2)
    if table.shape[1] < 2:
        raise ValueError(
            f"SED file {sed_path} must contain wavelength_nm and flux columns."
        )

    wavelength_nm = np.asarray(table[:, 0], dtype=float)
    flux_w_m2_nm = np.asarray(table[:, 1], dtype=float)
    if wavelength_nm.size < 2:
        raise ValueError(f"SED file {sed_path} must contain at least two samples.")

    order = np.argsort(wavelength_nm)
    wavelength_nm = wavelength_nm[order]
    flux_w_m2_nm = flux_w_m2_nm[order]

    model_nm = np.asarray(wavelength_grid_m, dtype=float) * 1e9
    interp_w_m2_nm = np.interp(model_nm, wavelength_nm, flux_w_m2_nm, left=0.0, right=0.0)
    interp_w_m2_nm = np.clip(interp_w_m2_nm, a_min=0.0, a_max=None)

    if not np.all(np.isfinite(interp_w_m2_nm)):
        raise ValueError(f"Interpolated SED contains non-finite values: {sed_path}")

    photon_flux_per_nm = interp_w_m2_nm * np.asarray(wavelength_grid_m) / (
        PLANCK_CONSTANT_J_S * SPEED_OF_LIGHT_M_PER_S
    )
    return photon_flux_per_nm


def normalize_component_weights(photon_flux_per_nm: np.ndarray) -> np.ndarray:
    """Return normalized component spectral weights from photon spectral flux."""

    total = float(np.sum(photon_flux_per_nm))
    if not (total > 0.0):
        raise ValueError("Cannot normalize zero-sum photon spectral flux.")
    return np.asarray(photon_flux_per_nm, dtype=float) / total


def integrate_broadband_photon_flux_ph_s_m2(
    photon_flux_per_nm: np.ndarray,
    wavelength_grid_m: np.ndarray,
    bandwidth_m: float,
) -> float:
    """Integrate photon spectral irradiance over the active bandpass.

    Parameters
    ----------
    photon_flux_per_nm:
        Photon spectral irradiance sampled on ``wavelength_grid_m`` in
        ``photons / s / m^2 / nm``.
    wavelength_grid_m:
        Wavelength grid in meters.
    bandwidth_m:
        Total modeled bandwidth in meters.

    Returns
    -------
    float
        Broadband photon irradiance in ``photons / s / m^2``.
    """

    flux = np.asarray(photon_flux_per_nm, dtype=float)
    grid = np.asarray(wavelength_grid_m, dtype=float)
    if flux.shape != grid.shape:
        raise ValueError("photon_flux_per_nm and wavelength_grid_m must have identical shapes.")

    if flux.size == 1:
        bandwidth_nm = float(bandwidth_m) * 1e9
        return float(flux.reshape(())) * bandwidth_nm

    wavelength_nm = grid * 1e9
    return float(np.trapezoid(flux, wavelength_nm))


def derive_source_photometry(
    *,
    wavelength_grid_m: np.ndarray,
    bandwidth_m: float,
    collecting_area_m2: float,
    exposure_time_s: float,
    throughput: float,
    sed_a_path: Optional[Path],
    sed_b_path: Optional[Path],
    vmag_a: Optional[float],
    vmag_b: Optional[float],
) -> SourcePhotometry:
    """Derive source photometry seeds from SEDs or Johnson-V fallback.

    Parameters
    ----------
    wavelength_grid_m:
        Active wavelength grid in meters.
    bandwidth_m:
        Active modeled bandwidth in meters.
    collecting_area_m2:
        Effective collecting area in square meters.
    exposure_time_s:
        Exposure time in seconds.
    throughput:
        End-to-end throughput (0 to 1).
    sed_a_path:
        Optional component-A SED path. When both component paths exist,
        SED-backed photometry is used.
    sed_b_path:
        Optional component-B SED path.
    vmag_a:
        Optional component-A Johnson-V magnitude for fallback mode.
    vmag_b:
        Optional component-B Johnson-V magnitude for fallback mode.

    Returns
    -------
    SourcePhotometry
        Derived broadband ``contrast`` + ``log_flux_total`` and component
        normalized weights.

    Notes
    -----
    Fallback mode uses a pragmatic Johnson-V/Vega-style approximation:

    - ``F_lambda,0 = 3.631e-8 W / m^2 / um`` at ``lambda_V = 550 nm``
    - per-component flux density scales as ``10**(-0.4 * Vmag)``
    - photon conversion uses ``E_photon = h c / lambda_V``
    - spectral shape defaults to uniform weights across the active grid

    This fallback is intentionally simple and does not attempt to model
    detailed SED shape.
    """

    area = float(collecting_area_m2)
    t_exp = float(exposure_time_s)
    eff = float(throughput)
    if not (area > 0.0):
        raise ValueError("collecting_area_m2 must be positive.")
    if not (t_exp > 0.0):
        raise ValueError("exposure_time_s must be positive.")
    if not (eff > 0.0):
        raise ValueError("throughput must be positive.")

    if sed_a_path is not None and sed_b_path is not None and sed_a_path.is_file() and sed_b_path.is_file():
        photon_a = load_sed_photon_flux_density_per_nm(sed_a_path, wavelength_grid_m=wavelength_grid_m)
        photon_b = load_sed_photon_flux_density_per_nm(sed_b_path, wavelength_grid_m=wavelength_grid_m)

        flux_a = integrate_broadband_photon_flux_ph_s_m2(
            photon_a,
            wavelength_grid_m=wavelength_grid_m,
            bandwidth_m=bandwidth_m,
        )
        flux_b = integrate_broadband_photon_flux_ph_s_m2(
            photon_b,
            wavelength_grid_m=wavelength_grid_m,
            bandwidth_m=bandwidth_m,
        )
        weights = np.stack(
            [
                normalize_component_weights(photon_a),
                normalize_component_weights(photon_b),
            ],
            axis=0,
        )
        mode = "sed"
    else:
        if vmag_a is None or vmag_b is None:
            raise ValueError(
                "Cannot derive source photometry without SED files: fallback requires "
                "both component V magnitudes."
            )

        zero_point_w_m2_m = JOHNSON_V_ZERO_POINT_W_M2_UM * 1e6
        energy_a_w_m2_m = zero_point_w_m2_m * (10.0 ** (-0.4 * float(vmag_a)))
        energy_b_w_m2_m = zero_point_w_m2_m * (10.0 ** (-0.4 * float(vmag_b)))

        photon_a_per_m = energy_a_w_m2_m * JOHNSON_V_REFERENCE_WAVELENGTH_M / (
            PLANCK_CONSTANT_J_S * SPEED_OF_LIGHT_M_PER_S
        )
        photon_b_per_m = energy_b_w_m2_m * JOHNSON_V_REFERENCE_WAVELENGTH_M / (
            PLANCK_CONSTANT_J_S * SPEED_OF_LIGHT_M_PER_S
        )

        flux_a = photon_a_per_m * float(bandwidth_m)
        flux_b = photon_b_per_m * float(bandwidth_m)

        n_lambda = int(np.asarray(wavelength_grid_m).size)
        if n_lambda <= 0:
            raise ValueError("wavelength_grid_m must contain at least one sample.")
        weights = np.full((2, n_lambda), 1.0 / n_lambda, dtype=float)
        mode = "vmag_fallback"

    if not (flux_a > 0.0 and flux_b > 0.0):
        raise ValueError(
            "Derived non-positive component photon fluxes; check SED coverage or fallback magnitudes."
        )

    contrast = float(flux_a / flux_b)
    total_photons = (flux_a + flux_b) * area * t_exp * eff
    if not (total_photons > 0.0):
        raise ValueError(
            "Derived non-positive total photon count; check area/exposure/throughput."
        )

    return SourcePhotometry(
        weights=weights,
        contrast=contrast,
        log_flux_total=float(np.log10(total_photons)),
        component_fluxes_ph_s_m2=(float(flux_a), float(flux_b)),
        mode=mode,
    )


__all__ = [
    "JOHNSON_V_REFERENCE_WAVELENGTH_M",
    "JOHNSON_V_ZERO_POINT_W_M2_UM",
    "SourcePhotometry",
    "build_wavelength_grid_m",
    "derive_source_photometry",
    "integrate_broadband_photon_flux_ph_s_m2",
    "load_sed_photon_flux_density_per_nm",
    "normalize_component_weights",
    "target_sed_root",
]
