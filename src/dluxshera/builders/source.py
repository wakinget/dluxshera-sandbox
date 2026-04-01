"""Source builder responsibilities (source assembly and runtime wiring)."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import asdict, is_dataclass
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
from dLuxToliman import AlphaCen

from ..components.sources import TargetSpec, get_target_spec
from ..utils.source_photometry import (
    build_wavelength_grid_m,
    derive_source_photometry,
    load_sed_photon_flux_density_per_nm,
    normalize_component_weights,
    target_sed_root,
)

if TYPE_CHECKING:
    from ..systems.three_plane import SheraThreePlaneConfig
    from ..systems.two_plane import SheraTwoPlaneConfig
from ..params.store import ParameterStore


SOURCE_RUNTIME_BINDINGS: tuple[tuple[str, str], ...] = ()


def _target_sed_root() -> resources.abc.Traversable:
    """Return the package-local directory containing curated target SED files.

    Returns
    -------
    importlib.resources.abc.Traversable
        Traversable directory rooted at ``dluxshera/data/target_seds``.

    Notes
    -----
    The implementation resolves from the installed/imported package via
    ``importlib.resources`` so lookup remains independent of the process CWD.
    During local source-tree execution this still points to the same on-disk
    directory under ``src/dluxshera/data/target_seds``.
    """

    return target_sed_root()


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


def _resolve_target_spec(target_key: Any) -> TargetSpec | None:
    """Resolve optional target key into a curated ``TargetSpec``."""
    if not target_key:
        return None
    return get_target_spec(str(target_key))


def load_normalized_sed_weights(
    sed_path: Path,
    wavelength_grid_m: np.ndarray,
) -> np.ndarray:
    """Load an SED and return normalized photon-count weights on the model grid.

    Assumptions
    -----------
    - Input SED files are plain-text ``.dat`` tables with wavelength in **nm**
      (column 0) and flux density in ``W / m^2 / nm`` (column 1).
    - Additional columns (for example uncertainty) are ignored.
    - The model wavelength grid is provided in **meters**.

    The returned weights are normalized photon-count weights (shape-only):
    they do not set total flux normalization or broadband contrast.
    """
    photon_flux = load_sed_photon_flux_density_per_nm(
        sed_path,
        wavelength_grid_m=wavelength_grid_m,
    )
    return normalize_component_weights(photon_flux)


def _derive_nominal_target_photometry(
    target_spec: TargetSpec | None,
    wavelength_grid_m: np.ndarray,
    bandwidth_m: float,
    collecting_area_m2: float,
    exposure_time_s: float,
    throughput: float,
    vmag_a: float | None,
    vmag_b: float | None,
):
    """Derive nominal source photometry from curated SEDs or V-mag fallback."""
    sed_a_ref = None
    sed_b_ref = None
    if target_spec and target_spec.sed_a_file and target_spec.sed_b_file:
        sed_root = _target_sed_root()
        sed_a_ref = sed_root.joinpath(target_spec.sed_a_file)
        sed_b_ref = sed_root.joinpath(target_spec.sed_b_file)

    if sed_a_ref is not None and sed_b_ref is not None and sed_a_ref.is_file() and sed_b_ref.is_file():
        with ExitStack() as stack:
            sed_a_path = Path(stack.enter_context(resources.as_file(sed_a_ref)))
            sed_b_path = Path(stack.enter_context(resources.as_file(sed_b_ref)))
            return derive_source_photometry(
                wavelength_grid_m=wavelength_grid_m,
                bandwidth_m=bandwidth_m,
                collecting_area_m2=collecting_area_m2,
                exposure_time_s=exposure_time_s,
                throughput=throughput,
                sed_a_path=sed_a_path,
                sed_b_path=sed_b_path,
                vmag_a=vmag_a,
                vmag_b=vmag_b,
            )

    return derive_source_photometry(
        wavelength_grid_m=wavelength_grid_m,
        bandwidth_m=bandwidth_m,
        collecting_area_m2=collecting_area_m2,
        exposure_time_s=exposure_time_s,
        throughput=throughput,
        sed_a_path=None,
        sed_b_path=None,
        vmag_a=vmag_a,
        vmag_b=vmag_b,
    )


def build_binary_target_source(
    store: ParameterStore,
    cfg: SheraThreePlaneConfig | SheraTwoPlaneConfig,
) -> AlphaCen:
    """Construct a binary-target source from runtime store values.

    Runtime source parameterization is unchanged:
    astrometry + ``log_flux_total`` + ``contrast`` + component weights.
    When runtime brightness terms are missing, nominal photometry is seeded
    from curated target SEDs (authoritative path) with Johnson-V fallback.
    """
    source_cfg = _extract_source_cfg(cfg)
    source_kind = str(source_cfg.get("kind", "")).lower()

    if "target" not in source_cfg and source_kind == "alpha_cen":
        source_cfg = dict(source_cfg)
        source_cfg["target"] = "ALPHA_CEN"

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
    target_key = _store_or_cfg(
        store,
        "source.target",
        cfg=cfg,
        source_cfg=source_cfg,
        default=source_cfg.get("target"),
    )
    target_spec = _resolve_target_spec(target_key)
    vmag_a = _store_or_cfg(
        store,
        "source.vmag_a",
        cfg=cfg,
        source_cfg=source_cfg,
        default=target_spec.vmag_a if target_spec else None,
    )
    vmag_b = _store_or_cfg(
        store,
        "source.vmag_b",
        cfg=cfg,
        source_cfg=source_cfg,
        default=target_spec.vmag_b if target_spec else None,
    )
    exposure_time_s = _store_or_cfg(
        store,
        "source.exposure_time_s",
        cfg=cfg,
        source_cfg=source_cfg,
        default=1.0,
    )
    throughput = _store_or_cfg(
        store,
        "optics.throughput",
        cfg=cfg,
        source_cfg=source_cfg,
        default=1.0,
    )
    m1_diameter_m = _store_or_cfg(
        store,
        "optics.m1_diameter_m",
        cfg=cfg,
        source_cfg=source_cfg,
        default=None,
    )
    separation_as = _store_or_cfg(
        store,
        "source.separation_as",
        cfg=cfg,
        source_cfg=source_cfg,
        default=target_spec.nominal_separation_as if target_spec else 10.0,
    )
    position_angle_deg = _store_or_cfg(
        store,
        "source.position_angle_deg",
        cfg=cfg,
        source_cfg=source_cfg,
        default=target_spec.nominal_position_angle_deg if target_spec else 90.0,
    )
    if wavelength_m is None or bandwidth_m is None or n_lambda is None:
        raise ValueError(
            "Missing required source values. Expected source.wavelength_m, source.bandwidth_m, "
            "and source.n_lambda (in store or config)."
        )
    if m1_diameter_m is None:
        raise ValueError("Missing required optics.m1_diameter_m for source photometry seeding.")

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

    model_wavelengths_m = build_wavelength_grid_m(
        wavelength_m=float(wavelength_m),
        bandwidth_m=float(bandwidth_m),
        n_lambda=int(n_lambda),
    )
    collecting_area_m2 = float(np.pi * (float(m1_diameter_m) / 2.0) ** 2)

    nominal_photometry = None
    nominal_error = None
    try:
        nominal_photometry = _derive_nominal_target_photometry(
            target_spec=target_spec,
            wavelength_grid_m=model_wavelengths_m,
            bandwidth_m=float(bandwidth_m),
            collecting_area_m2=collecting_area_m2,
            exposure_time_s=float(exposure_time_s),
            throughput=float(throughput),
            vmag_a=float(vmag_a) if vmag_a is not None else None,
            vmag_b=float(vmag_b) if vmag_b is not None else None,
        )
    except ValueError as exc:
        nominal_error = exc

    default_contrast = (
        float(nominal_photometry.contrast)
        if nominal_photometry is not None
        else (target_spec.nominal_contrast if target_spec and target_spec.nominal_contrast else 3.0)
    )
    contrast_sentinel = object()
    contrast = store.get("source.contrast", default=contrast_sentinel)
    if contrast is contrast_sentinel:
        if target_spec is None and "contrast" in source_cfg:
            contrast = source_cfg.get("contrast")
        else:
            contrast = default_contrast

    default_log_flux = float(nominal_photometry.log_flux_total) if nominal_photometry is not None else None
    log_flux_total = _store_or_cfg(
        store,
        "source.log_flux_total",
        cfg=cfg,
        source_cfg=source_cfg,
        default=default_log_flux,
    )
    if log_flux_total is None:
        if nominal_error is None:
            raise ValueError(
                "Missing source.log_flux_total and no nominal photometry could be derived."
            )
        raise ValueError(
            "Missing source.log_flux_total and nominal photometry seeding failed: "
            f"{nominal_error}"
        ) from nominal_error

    if nominal_photometry is not None:
        weights = jnp.asarray(nominal_photometry.weights)
    else:
        # If nominal photometry is unavailable but explicit flux parameters
        # are provided, keep source construction permissive with uniform shape.
        weights = jnp.asarray(np.full((2, int(n_lambda)), 1.0 / float(n_lambda)))

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
    "apply_runtime_bindings",
    "build_alpha_cen_source",
    "build_binary_target_source",
    "load_normalized_sed_weights",
]
