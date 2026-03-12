"""Source builder responsibilities (source assembly and runtime wiring)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dLuxToliman import AlphaCen

if TYPE_CHECKING:
    from ..systems.three_plane import SheraThreePlaneConfig
    from ..systems.two_plane import SheraTwoPlaneConfig
from ..params.store import ParameterStore


SOURCE_RUNTIME_BINDINGS: tuple[tuple[str, str], ...] = ()


def build_alpha_cen_source(
    store: ParameterStore,
    cfg: SheraThreePlaneConfig | SheraTwoPlaneConfig,
) -> AlphaCen:
    """
    Construct an AlphaCen source (from dLuxToliman) from a ParameterStore.

    Parameters
    ----------
    store :
        ParameterStore holding source parameters under the ``source.*``
        namespace. Required keys are:

        - 'source.wavelength_m'         (float, meters)
        - 'source.bandwidth_m'          (float, meters)
        - 'source.n_lambda'             (int-like scalar)
        - 'source.separation_as'        (float, arcseconds)
        - 'source.position_angle_deg'   (float, degrees East of North)
        - 'source.log_flux_total'       (float, log10 photons)
        - 'source.contrast'             (float, unitless flux ratio)

        Optional keys:
        - 'source.x_position_as'        (float, arcseconds; defaults to 0.0)
        - 'source.y_position_as'        (float, arcseconds; defaults to 0.0)

    cfg :
        Shera configuration object (kept for API compatibility); source
        structural wavelength settings are read from ``store``.

    Returns
    -------
    AlphaCen
        A dLuxToliman AlphaCen source object constructed directly from the
        ParameterStore values and the config-driven bandpass.

    Notes
    -----
    - This is a *P0 convenience builder*: it assumes that the effective
      log-flux has already been computed (or chosen) and stored under
      'source.log_flux_total'. It does *not* try to derive that from
      physical quantities like exposure time, aperture area, etc.

    - Longer-term, we expect a dedicated “UniverseSpec” and parameter
      transforms to handle the “truth-level” flux bookkeeping. This builder
      will then simply read whatever effective quantities the spec provides.
    """
    # Required parameters – let KeyError surface if they’re missing.
    wavelength_m = store.get("source.wavelength_m")
    bandwidth_m = store.get("source.bandwidth_m")
    n_lambda = store.get("source.n_lambda")
    separation_as = store.get("source.separation_as")
    position_angle_deg = store.get("source.position_angle_deg")
    log_flux_total = store.get("source.log_flux_total")
    contrast = store.get("source.contrast")

    # Optional centre; default to (0, 0) if not present
    x_position = store.get("source.x_position_as", default=0.0)
    y_position = store.get("source.y_position_as", default=0.0)

    center_nm = wavelength_m * 1e9
    bandwidth_nm = bandwidth_m * 1e9
    bandpass = (
        center_nm - bandwidth_nm / 2,
        center_nm + bandwidth_nm / 2,
    )

    return AlphaCen(
        n_wavels=int(n_lambda),
        separation=separation_as,       # arcsec
        position_angle=position_angle_deg,  # degrees
        x_position=x_position,
        y_position=y_position,
        log_flux=log_flux_total,        # log10 photons
        contrast=contrast,
        bandpass=bandpass,
    )


def apply_runtime_bindings(
    source: AlphaCen,
    store: ParameterStore | None,
    *,
    cfg: SheraThreePlaneConfig | SheraTwoPlaneConfig,
    bindings: tuple[tuple[str, str], ...] = SOURCE_RUNTIME_BINDINGS,
) -> AlphaCen:
    """Apply runtime ``source.*`` store overrides onto a cached source."""

    if store is None:
        return source

    if bindings:
        for store_key, set_path in bindings:
            val = store.get(store_key, default=None)
            if val is None:
                continue
            source = source.set(set_path, val)
        return source

    return build_alpha_cen_source(store, cfg=cfg)


__all__ = [
    "SOURCE_RUNTIME_BINDINGS",
    "apply_runtime_bindings",
    "build_alpha_cen_source",
]
