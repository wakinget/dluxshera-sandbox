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
            "source.separation_as",
            group="source",
            kind="primitive",
            dtype=float,
            shape=(),
            default=10.0,
            structural=False,
        ),
        ParamField(
            "source.position_angle_deg",
            group="source",
            kind="primitive",
            dtype=float,
            shape=(),
            default=90.0,
            structural=False,
        ),
        ParamField(
            "source.x_position_as",
            group="source",
            kind="primitive",
            dtype=float,
            shape=(),
            default=0.0,
            structural=False,
        ),
        ParamField(
            "source.y_position_as",
            group="source",
            kind="primitive",
            dtype=float,
            shape=(),
            default=0.0,
            structural=False,
        ),
        ParamField(
            "source.log_flux_total",
            group="source",
            kind="primitive",
            dtype=float,
            shape=(),
            default=20.0,
            structural=False,
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
    ]
    return ParamSpec(fields)
