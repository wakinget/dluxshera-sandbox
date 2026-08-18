"""Single-star calibration source helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np

from dluxshera.components.sources import compute_source_flux_diagnostics
from dluxshera.params.store import ParameterStore
from dluxshera.systems.base import compose_forward_spec


ALPHA_CEN_A_PLACEHOLDER_NOTE = (
    "Alpha Cen A component placeholder until a calibration-star registry exists."
)


def prepare_alpha_cen_a_single_star_system_config(
    base_system_cfg: Mapping[str, Any],
    *,
    exposure_time_s: float | None = None,
    n_lambda: int | None = None,
) -> dict[str, Any]:
    """Return a centered Alpha Cen A-like ``single_star`` system config.

    The helper derives an Alpha Cen A component photon count from the existing
    Alpha Cen binary-target photometry path, then exposes that count through
    the public ``source.log_flux_total`` parameter used by ``single_star``.
    """

    alpha_cfg = deepcopy(dict(base_system_cfg))
    alpha_source = alpha_cfg.setdefault("source", {})
    if not isinstance(alpha_source, dict):
        raise ValueError("Expected system.source to be a mapping.")
    alpha_source["kind"] = "binary_target"
    alpha_source["target"] = "ALPHA_CEN"
    if exposure_time_s is not None:
        alpha_source["exposure_time_s"] = float(exposure_time_s)
    if n_lambda is not None:
        alpha_source["n_lambda"] = int(n_lambda)
    alpha_spec = compose_forward_spec(alpha_cfg)
    alpha_store = ParameterStore.from_spec_defaults(alpha_spec).refresh_derived(alpha_spec)
    alpha_flux_diag = compute_source_flux_diagnostics("binary_target", alpha_store)
    alpha_a_flux = float(np.asarray(alpha_flux_diag["component_fluxes"]["primary"]))
    if not np.isfinite(alpha_a_flux) or alpha_a_flux <= 0.0:
        raise ValueError("Alpha Cen A component flux must be positive and finite.")

    system_cfg = deepcopy(dict(base_system_cfg))
    source_cfg = system_cfg.setdefault("source", {})
    if not isinstance(source_cfg, dict):
        raise ValueError("Expected system.source to be a mapping.")
    source_cfg.clear()
    source_cfg.update(
        {
            "kind": "single_star",
            "wavelength_m": float(np.asarray(alpha_store.get("source.wavelength_m"))),
            "bandwidth_m": float(np.asarray(alpha_store.get("source.bandwidth_m"))),
            "n_lambda": int(np.asarray(alpha_store.get("source.n_lambda"))),
            "exposure_time_s": float(np.asarray(alpha_store.get("source.exposure_time_s"))),
            "x_position_as": 0.0,
            "y_position_as": 0.0,
            "position_angle_deg": 0.0,
            "log_flux_total": float(np.log10(alpha_a_flux)),
        }
    )
    if exposure_time_s is not None:
        source_cfg["exposure_time_s"] = float(exposure_time_s)
    if n_lambda is not None:
        source_cfg["n_lambda"] = int(n_lambda)
    return system_cfg


__all__ = [
    "ALPHA_CEN_A_PLACEHOLDER_NOTE",
    "prepare_alpha_cen_a_single_star_system_config",
]
