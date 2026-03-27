"""Source builder responsibilities (source assembly and runtime wiring)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
from dLuxToliman import AlphaCen

from ..components.sources import TargetSpec, get_target_spec

if TYPE_CHECKING:
    from ..systems.three_plane import SheraThreePlaneConfig
    from ..systems.two_plane import SheraTwoPlaneConfig
from ..params.store import ParameterStore


SOURCE_RUNTIME_BINDINGS: tuple[tuple[str, str], ...] = ()

TARGET_SED_DIR = Path(__file__).resolve().parents[1] / "data" / "target_seds"


def _cfg_get(root: Any, path: str, default=None):
    """Read dotted config values from mapping-like or dataclass-like objects."""
    parts = path.split(".")
    cur = root
    for key in parts:
        if cur is None:
            return default
        if isinstance(cur, Mapping):
            cur = cur.get(key, None)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur


def _extract_source_cfg(cfg: SheraThreePlaneConfig | SheraTwoPlaneConfig) -> Mapping[str, Any]:
    """Return the ``system.source`` mapping when present, else an empty mapping."""
    if is_dataclass(cfg):
        cfg_dict = asdict(cfg)
    else:
        cfg_dict = cfg

    if not isinstance(cfg_dict, Mapping):
        return {}

    if "system" in cfg_dict and isinstance(cfg_dict["system"], Mapping):
        src = cfg_dict["system"].get("source", {})
    else:
        src = cfg_dict.get("source", {})

    return src if isinstance(src, Mapping) else {}


def _store_or_cfg(
    store: ParameterStore,
    key: str,
    *,
    cfg: SheraThreePlaneConfig | SheraTwoPlaneConfig,
    source_cfg: Mapping[str, Any],
    default,
):
    """Return parameter value with precedence: store > source config > cfg attr > default."""
    sentinel = object()
    store_val = store.get(key, default=sentinel)
    if store_val is not sentinel:
        return store_val

    source_key = key.split(".", maxsplit=1)[-1]
    if source_key in source_cfg:
        return source_cfg[source_key]

    cfg_val = _cfg_get(cfg, source_key, default=sentinel)
    if cfg_val is not sentinel:
        return cfg_val

    return default


def _resolve_target_spec(source_cfg: Mapping[str, Any]) -> TargetSpec | None:
    """Resolve optional ``system.source.target`` into a curated ``TargetSpec``."""
    target_key = source_cfg.get("target")
    if not target_key:
        return None
    return get_target_spec(str(target_key))


def load_normalized_sed_weights(
    sed_path: Path,
    wavelength_grid_m: np.ndarray,
) -> np.ndarray:
    """Load a tabulated component SED and return normalized model-grid weights.

    Assumptions
    -----------
    - Input SED files are plain-text ``.dat`` tables with wavelength in **nm**
      (column 0) and flux density in ``W / m^2 / nm`` (column 1).
    - Additional columns (for example uncertainty) are ignored.
    - The model wavelength grid is provided in **meters**.

    The returned weights are non-negative and normalized to sum to 1.0 so they
    encode only chromatic shape; total source normalization remains controlled
    by ``source.log_flux_total`` and ``source.contrast``.
    """

    table = np.loadtxt(sed_path, ndmin=2)
    if table.shape[1] < 2:
        raise ValueError(
            f"SED file {sed_path} must contain at least two columns: wavelength_nm and flux."
        )

    wavelength_nm = np.asarray(table[:, 0], dtype=float)
    flux_density = np.asarray(table[:, 1], dtype=float)

    if wavelength_nm.size < 2:
        raise ValueError(f"SED file {sed_path} must contain at least two wavelength samples.")

    order = np.argsort(wavelength_nm)
    wavelength_nm = wavelength_nm[order]
    flux_density = flux_density[order]

    model_wavelengths_nm = np.asarray(wavelength_grid_m, dtype=float) * 1e9
    interpolated = np.interp(model_wavelengths_nm, wavelength_nm, flux_density, left=0.0, right=0.0)
    interpolated = np.clip(interpolated, a_min=0.0, a_max=None)

    if not np.all(np.isfinite(interpolated)):
        raise ValueError(f"Interpolated SED weights from {sed_path} contain non-finite values.")

    total = float(np.sum(interpolated))
    if not (total > 0.0):
        raise ValueError(
            "Interpolated SED weights sum to zero. Ensure the file wavelength range overlaps "
            "the model bandpass and flux values are positive."
        )

    return interpolated / total


def _resolve_component_weights(
    target_spec: TargetSpec | None,
    wavelength_grid_m: np.ndarray,
) -> jnp.ndarray | None:
    """Return component spectral weights for a target, or ``None`` for uniform fallback."""
    if target_spec is None:
        return None

    if not target_spec.sed_a_file or not target_spec.sed_b_file:
        return None

    sed_a_path = TARGET_SED_DIR / target_spec.sed_a_file
    sed_b_path = TARGET_SED_DIR / target_spec.sed_b_file

    # Lean fallback behaviour: if either curated file is unavailable, use the
    # source model's default uniform weighting instead of failing construction.
    if not sed_a_path.exists() or not sed_b_path.exists():
        return None

    weights_a = load_normalized_sed_weights(sed_a_path, wavelength_grid_m=wavelength_grid_m)
    weights_b = load_normalized_sed_weights(sed_b_path, wavelength_grid_m=wavelength_grid_m)

    return jnp.asarray(np.stack([weights_a, weights_b], axis=0))


def build_binary_target_source(
    store: ParameterStore,
    cfg: SheraThreePlaneConfig | SheraTwoPlaneConfig,
) -> AlphaCen:
    """Construct a binary-target source from runtime store values.

    The builder resolves optional ``system.source.target`` metadata to seed
    nominal defaults, but the authoritative runtime parameter semantics remain
    unchanged under ``source.*`` keys.
    """
    source_cfg = _extract_source_cfg(cfg)
    source_kind = str(source_cfg.get("kind", "")).lower()

    if "target" not in source_cfg and source_kind == "alpha_cen":
        source_cfg = dict(source_cfg)
        source_cfg["target"] = "ALPHA_CEN"

    target_spec = _resolve_target_spec(source_cfg)

    default_sep = target_spec.nominal_separation_as if target_spec else 10.0
    default_pa = target_spec.nominal_position_angle_deg if target_spec else 90.0
    default_contrast = (
        target_spec.nominal_contrast if target_spec and target_spec.nominal_contrast else 3.0
    )

    wavelength_m = _store_or_cfg(
        store,
        "source.wavelength_m",
        cfg=cfg,
        source_cfg=source_cfg,
        default=None,
    )
    bandwidth_m = _store_or_cfg(
        store,
        "source.bandwidth_m",
        cfg=cfg,
        source_cfg=source_cfg,
        default=None,
    )
    n_lambda = _store_or_cfg(
        store,
        "source.n_lambda",
        cfg=cfg,
        source_cfg=source_cfg,
        default=None,
    )
    separation_as = _store_or_cfg(
        store,
        "source.separation_as",
        cfg=cfg,
        source_cfg=source_cfg,
        default=default_sep,
    )
    position_angle_deg = _store_or_cfg(
        store,
        "source.position_angle_deg",
        cfg=cfg,
        source_cfg=source_cfg,
        default=default_pa,
    )
    log_flux_total = _store_or_cfg(
        store,
        "source.log_flux_total",
        cfg=cfg,
        source_cfg=source_cfg,
        default=None,
    )
    contrast = _store_or_cfg(
        store,
        "source.contrast",
        cfg=cfg,
        source_cfg=source_cfg,
        default=default_contrast,
    )

    if wavelength_m is None or bandwidth_m is None or n_lambda is None or log_flux_total is None:
        raise ValueError(
            "Missing required source values. Expected source.wavelength_m, source.bandwidth_m, "
            "source.n_lambda, and source.log_flux_total (in store or config)."
        )

    # Optional centre; default to (0, 0) if not present anywhere.
    x_position = _store_or_cfg(
        store,
        "source.x_position_as",
        cfg=cfg,
        source_cfg=source_cfg,
        default=0.0,
    )
    y_position = _store_or_cfg(
        store,
        "source.y_position_as",
        cfg=cfg,
        source_cfg=source_cfg,
        default=0.0,
    )

    center_nm = float(wavelength_m) * 1e9
    bandwidth_nm = float(bandwidth_m) * 1e9
    bandpass = (
        center_nm - bandwidth_nm / 2,
        center_nm + bandwidth_nm / 2,
    )

    model_wavelengths_m = np.linspace(
        bandpass[0] * 1e-9,
        bandpass[1] * 1e-9,
        int(n_lambda),
    )
    weights = _resolve_component_weights(target_spec, wavelength_grid_m=model_wavelengths_m)

    return AlphaCen(
        n_wavels=int(n_lambda),
        separation=separation_as,  # arcsec
        position_angle=position_angle_deg,  # degrees
        x_position=x_position,
        y_position=y_position,
        log_flux=log_flux_total,  # log10 photons
        contrast=contrast,
        bandpass=bandpass,
        weights=weights,
    )


def build_alpha_cen_source(
    store: ParameterStore,
    cfg: SheraThreePlaneConfig | SheraTwoPlaneConfig,
) -> AlphaCen:
    """Compatibility wrapper for the historical Alpha-Cen builder entry point."""
    return build_binary_target_source(store, cfg=cfg)


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

    return build_binary_target_source(store, cfg=cfg)


__all__ = [
    "SOURCE_RUNTIME_BINDINGS",
    "TARGET_SED_DIR",
    "apply_runtime_bindings",
    "build_alpha_cen_source",
    "build_binary_target_source",
    "load_normalized_sed_weights",
]
