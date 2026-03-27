"""Source component contracts and target metadata for binary sources."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from ..params.spec import ParamField, ParamSpec


@dataclass(frozen=True)
class TargetSpec:
    """Lean static metadata for a curated binary target.

    Notes
    -----
    ``TargetSpec`` stores nominal/reference metadata only. Runtime inference
    parameters continue to live under ``source.*`` keys in the ``ParameterStore``
    (for example ``source.separation_as``, ``source.position_angle_deg``,
    ``source.log_flux_total``, and ``source.contrast``).
    """

    key: str
    display_name: str
    component_labels: tuple[str, str] = ("A", "B")
    reference_epoch: float | None = None
    nominal_separation_as: float | None = None
    nominal_position_angle_deg: float | None = None
    vmag_a: float | None = None
    vmag_b: float | None = None
    spectral_type_a: str | None = None
    spectral_type_b: str | None = None
    sed_a_file: str | None = None
    sed_b_file: str | None = None
    distance_pc: float | None = None
    wds_id: str | None = None
    notes: str | None = None

    @property
    def nominal_contrast(self) -> float | None:
        """Return nominal broadband contrast (A:B) derived from V magnitudes."""
        if self.vmag_a is None or self.vmag_b is None:
            return None
        return 10.0 ** (0.4 * (self.vmag_b - self.vmag_a))


TARGET_SPECS: dict[str, TargetSpec] = {
    "ALPHA_CEN": TargetSpec(
        key="ALPHA_CEN",
        display_name="Alpha Centauri",
        reference_epoch=2027.0,
        nominal_separation_as=9.765,
        nominal_position_angle_deg=14.508,
        vmag_a=0.002,
        vmag_b=1.350,
        spectral_type_a="G2V",
        spectral_type_b="K1V",
        sed_a_file="alpha_cen_a.dat",
        sed_b_file="alpha_cen_b.dat",
        notes="Nominal metadata for binary-source initialization.",
    ),
    "61_CYG": TargetSpec(
        key="61_CYG",
        display_name="61 Cygni",
        reference_epoch=2027.0,
        nominal_separation_as=32.185,
        nominal_position_angle_deg=154.399,
        vmag_a=5.211,
        vmag_b=6.043,
        spectral_type_a="K5V",
        spectral_type_b="K7V",
        sed_a_file="61_cyg_a.dat",
        sed_b_file="61_cyg_b.dat",
        notes="Nominal metadata for binary-source initialization.",
    ),
}


def get_target_spec(key: str) -> TargetSpec:
    """Return a curated ``TargetSpec`` by key.

    Parameters
    ----------
    key:
        Curated target key (case-insensitive), for example ``"ALPHA_CEN"``.
    """

    normalized = key.strip().upper()
    try:
        return TARGET_SPECS[normalized]
    except KeyError as exc:
        known = ", ".join(sorted(TARGET_SPECS))
        raise ValueError(
            f"Unknown target {key!r}. Supported curated targets: {known}."
        ) from exc


def _extract_source_mapping(source_cfg: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    """Return a source mapping from source-only or full-system configuration."""
    if is_dataclass(source_cfg):
        source_cfg = asdict(source_cfg)

    if isinstance(source_cfg, Mapping) and "source" in source_cfg:
        source_cfg = source_cfg["source"]

    if not isinstance(source_cfg, Mapping):
        raise ValueError(
            "build_binary_target_contract expected a source mapping or a system "
            "mapping containing a 'source' key."
        )
    return source_cfg


def build_binary_target_contract(source_cfg: Mapping[str, Any]) -> ParamSpec:
    """Return the generic binary-target parameter contract under ``source.*`` keys.

    The optional ``system.source.target`` key points to static ``TargetSpec``
    metadata that may seed nominal defaults for ``source.separation_as``,
    ``source.position_angle_deg``, and ``source.contrast``. Runtime values in
    the store remain authoritative.
    """

    source_cfg = _extract_source_mapping(source_cfg)

    target_key = source_cfg.get("target")
    target_spec = get_target_spec(str(target_key)) if target_key else None

    wavelength_m = source_cfg.get("wavelength_m")
    bandwidth_m = source_cfg.get("bandwidth_m")
    n_lambda = source_cfg.get("n_lambda")
    exposure_time_s = source_cfg.get("exposure_time_s", 1.0)
    spectral_flux_density = source_cfg.get("spectral_flux_density", 1.7227e17)

    default_separation = source_cfg.get(
        "separation_as",
        target_spec.nominal_separation_as if target_spec else 10.0,
    )
    default_pa = source_cfg.get(
        "position_angle_deg",
        target_spec.nominal_position_angle_deg if target_spec else 90.0,
    )
    default_contrast = source_cfg.get(
        "contrast",
        target_spec.nominal_contrast if target_spec and target_spec.nominal_contrast else 3.0,
    )

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
            default=default_separation,
            structural=False,
            binding="separation",
        ),
        ParamField(
            "source.position_angle_deg",
            group="source",
            kind="primitive",
            dtype=float,
            shape=(),
            default=default_pa,
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
            default=default_contrast,
            structural=False,
            binding="contrast",
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


def build_alpha_cen_contract(source_cfg: Mapping[str, Any]) -> ParamSpec:
    """Compatibility wrapper for the historical Alpha-Cen contract entry point."""
    source_cfg = _extract_source_mapping(source_cfg)
    if "target" not in source_cfg and str(source_cfg.get("kind", "")).lower() in {
        "alpha_cen",
        "binary",
        "binary_target",
    }:
        source_cfg = dict(source_cfg)
        source_cfg["target"] = "ALPHA_CEN"
    return build_binary_target_contract(source_cfg)


__all__ = [
    "TargetSpec",
    "TARGET_SPECS",
    "build_alpha_cen_contract",
    "build_binary_target_contract",
    "get_target_spec",
]
