"""Source component contracts for external source implementations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

from ..params.spec import ParamField, ParamSpec

__all__ = ["build_alpha_cen_contract"]


def build_alpha_cen_contract(source_cfg: Mapping[str, Any]) -> ParamSpec:
    """Return the AlphaCen source parameter contract under ``source.*`` keys.

    Required keys:
      - ``source.wavelength_m``
      - ``source.bandwidth_m``
      - ``source.n_lambda``
      - ``source.separation_as``
      - ``source.position_angle_deg``
      - ``source.log_flux_total``
      - ``source.contrast``

    Optional keys:
      - ``source.x_position_as`` (defaults to 0.0)
      - ``source.y_position_as`` (defaults to 0.0)
    """

    if is_dataclass(source_cfg):
        source_cfg = asdict(source_cfg)

    if isinstance(source_cfg, Mapping) and "source" in source_cfg:
        source_cfg = source_cfg["source"]

    if not isinstance(source_cfg, Mapping):
        raise ValueError(
            "build_alpha_cen_contract expected a source mapping or a system "
            "mapping containing a 'source' key."
        )

    wavelength_m = source_cfg.get("wavelength_m")
    bandwidth_m = source_cfg.get("bandwidth_m")
    n_lambda = source_cfg.get("n_lambda")
    exposure_time_s = source_cfg.get("exposure_time_s", 1.0)
    spectral_flux_density = source_cfg.get("spectral_flux_density", 1.7227e17)

    log_flux_dependencies = (
        "optics.m1_diameter_m",
        "source.bandwidth_m",
        "source.exposure_time_s",
        "optics.throughput",
        "source.spectral_flux_density",
    )

    raw_flux_dependencies = (
        "source.log_flux_total",
        "source.contrast",
    )

    fields = [
        ParamField(
            "source.wavelength_m",
            group="source",
            kind="primitive",
            dtype=float,
            shape=(),
            default=wavelength_m,
            structural=True,
        ),
        ParamField(
            "source.bandwidth_m",
            group="source",
            kind="primitive",
            dtype=float,
            shape=(),
            default=bandwidth_m,
            structural=True,
        ),
        ParamField(
            "source.n_lambda",
            group="source",
            kind="primitive",
            dtype=int,
            shape=(),
            default=n_lambda,
            structural=True,
        ),
        ParamField(
            "source.exposure_time_s",
            group="source",
            kind="primitive",
            dtype=float,
            shape=(),
            default=exposure_time_s,
            structural=False,
        ),
        ParamField(
            "source.spectral_flux_density",
            group="source",
            kind="primitive",
            dtype=float,
            shape=(),
            default=spectral_flux_density,
            structural=False,
        ),
        ParamField(
            "source.separation_as",
            group="source",
            kind="primitive",
            dtype=float,
            shape=(),
            default=10.0,
            structural=False,
            binding="separation",
        ),
        ParamField(
            "source.position_angle_deg",
            group="source",
            kind="primitive",
            dtype=float,
            shape=(),
            default=90.0,
            structural=False,
            binding="position_angle",
        ),
        ParamField(
            "source.x_position_as",
            group="source",
            kind="primitive",
            dtype=float,
            shape=(),
            default=0.0,
            structural=False,
            binding="x_position",
        ),
        ParamField(
            "source.y_position_as",
            group="source",
            kind="primitive",
            dtype=float,
            shape=(),
            default=0.0,
            structural=False,
            binding="y_position",
        ),
        ParamField(
            "source.log_flux_total",
            group="source",
            kind="derived",
            dtype=float,
            shape=(),
            default=None,
            structural=False,
            transform="source.log_flux_total",
            depends_on=log_flux_dependencies,
            binding="log_flux",
        ),
        ParamField(
            "source.contrast",
            group="source",
            kind="primitive",
            dtype=float,
            shape=(),
            default=3.0,
            structural=False,
        ),
        ParamField(
            "source.raw_fluxes",
            group="source",
            kind="derived",
            dtype=float,
            shape=(2,),
            default=None,
            structural=False,
            transform="source.raw_fluxes",
            depends_on=raw_flux_dependencies,
        ),
    ]
    return ParamSpec(fields)
