"""Source component contracts for external source implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..params.spec import ParamField, ParamSpec

if TYPE_CHECKING:
    from ..systems.three_plane import SheraThreePlaneConfig
    from ..systems.two_plane import SheraTwoPlaneConfig


__all__ = ["build_alpha_cen_contract"]


def build_alpha_cen_contract(
    source_cfg: "SheraThreePlaneConfig | SheraTwoPlaneConfig",
) -> ParamSpec:
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

    fields = [
        ParamField(
            "source.wavelength_m",
            group="source",
            kind="primitive",
            dtype=float,
            shape=(),
            default=source_cfg.wavelength_m,
            structural=True,
        ),
        ParamField(
            "source.bandwidth_m",
            group="source",
            kind="primitive",
            dtype=float,
            shape=(),
            default=source_cfg.bandwidth_m,
            structural=True,
        ),
        ParamField(
            "source.n_lambda",
            group="source",
            kind="primitive",
            dtype=int,
            shape=(),
            default=source_cfg.n_lambda,
            structural=True,
        ),
        ParamField(
            "source.separation_as",
            group="source",
            kind="primitive",
            dtype=float,
            shape=(),
            structural=False,
        ),
        ParamField(
            "source.position_angle_deg",
            group="source",
            kind="primitive",
            dtype=float,
            shape=(),
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
            structural=False,
        ),
        ParamField(
            "source.contrast",
            group="source",
            kind="primitive",
            dtype=float,
            shape=(),
            structural=False,
        ),
    ]
    return ParamSpec(fields)
