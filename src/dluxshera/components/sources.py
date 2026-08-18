"""Source component contracts, diagnostics, and target metadata."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import asdict, dataclass, is_dataclass
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..params.spec import ParamField, ParamSpec
from ..utils.source_photometry import (
    build_wavelength_grid_m,
    derive_source_photometry,
    target_sed_root,
)

if TYPE_CHECKING:
    import jax.numpy as jnp

    from ..params.store import ParameterStore


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


# Target values are taken from a google spreadsheet "MO Targets and sensitivity (TOLIMAN/SHERA)"
# shared by Eric Mamajek 03/26/2026
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
        sed_a_file="alfCenA_SED.dat",
        sed_b_file="alfCenB_SED.dat",
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
        sed_a_file="61CygA_SED.dat",
        sed_b_file="61CygB_SED.dat",
        notes="Nominal metadata for binary-source initialization.",
    ),
    "70_OPH": TargetSpec(
        key="70_OPH",
        display_name="70 Ophiuchi",
        reference_epoch=2027.0,
        nominal_separation_as=6.714,
        nominal_position_angle_deg=115.704,
        vmag_a=4.220,
        vmag_b=6.061,
        spectral_type_a="K0V",
        spectral_type_b="K4V",
        sed_a_file="70OphA_SED.dat",
        sed_b_file="70OphB_SED.dat",
        notes="Nominal metadata for binary-source initialization.",
    ),
    "36_OPH": TargetSpec(
        key="36_OPH",
        display_name="36 Ophiuchi",
        reference_epoch=2027.0,
        nominal_separation_as=5.236,
        nominal_position_angle_deg=137.129,
        vmag_a=5.070,
        vmag_b=5.110,
        spectral_type_a="K1V",
        spectral_type_b="K1V",
        sed_a_file="36OphA_SED.dat",
        sed_b_file="36OphB_SED.dat",
        notes="Nominal metadata for binary-source initialization.",
    ),
    "XI_BOO": TargetSpec(
        key="XI_BOO",
        display_name="Xi Boötis",
        reference_epoch=2027.0,
        nominal_separation_as=4.690,
        nominal_position_angle_deg=286.461,
        vmag_a=4.540,
        vmag_b=6.979,
        spectral_type_a="G8V",
        spectral_type_b="K5V",
        sed_a_file="xiBooA_SED.dat",
        sed_b_file="xiBooB_SED.dat",
        notes="Nominal metadata for binary-source initialization.",
    ),
    "P_ERI": TargetSpec(
        key="P_ERI",
        display_name="p Eridani",
        reference_epoch=2027.0,
        nominal_separation_as=11.444,
        nominal_position_angle_deg=184.024,
        vmag_a=5.764,
        vmag_b=5.876,
        spectral_type_a="K2V",
        spectral_type_b="K2V",
        sed_a_file="pEriA_SED.dat",
        sed_b_file="pEriB_SED.dat",
        notes="Nominal metadata for binary-source initialization.",
    ),
    "HR_2667_2668": TargetSpec(
        key="HR_2667_2668",
        display_name="HR 2667 / HR 2668",
        component_labels=("HR 2667", "HR 2668"),
        reference_epoch=2027.0,
        nominal_separation_as=21.916,
        nominal_position_angle_deg=123.243,
        vmag_a=5.560,
        vmag_b=6.830,
        spectral_type_a="G1.5V",
        spectral_type_b="K0.5V",
        sed_a_file="HR2667_SED.dat",
        sed_b_file="HR2668_SED.dat",
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


def linear_total_flux_from_log10(log_flux_total: Any) -> jnp.ndarray:
    """Return linear total photons from a public log10-flux parameter.

    Parameters
    ----------
    log_flux_total:
        dLuxShera public total-flux parameter, expressed as
        ``log10(total photons)``.

    Returns
    -------
    jax.Array
        Linear total photon count.
    """

    import jax.numpy as jnp

    return jnp.power(jnp.asarray(10.0, dtype=float), jnp.asarray(log_flux_total, dtype=float))


def binary_component_fluxes_from_total_and_contrast(
    total_flux: Any,
    contrast: Any,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return primary/secondary fluxes from total flux and A/B contrast.

    Parameters
    ----------
    total_flux:
        Total binary flux in photons.
    contrast:
        Primary-to-secondary flux ratio.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        ``(primary_flux, secondary_flux)`` using the AlphaCen convention where
        ``primary / secondary == contrast`` and the components sum to
        ``total_flux``.
    """

    import jax.numpy as jnp

    total = jnp.asarray(total_flux, dtype=float)
    ratio = jnp.asarray(contrast, dtype=float)
    secondary = total / (jnp.asarray(1.0, dtype=float) + ratio)
    primary = ratio * secondary
    return primary, secondary


def binary_mean_flux_from_total_and_contrast(
    total_flux: Any,
    contrast: Any,
) -> jnp.ndarray:
    """Return dLux ``BinarySource.mean_flux`` from dLuxShera total flux.

    Parameters
    ----------
    total_flux:
        Public dLuxShera total binary flux in photons.
    contrast:
        Primary-to-secondary flux ratio. Present for convention clarity; the
        current dLux ``BinarySource`` convention uses the average component
        flux, so the result is independent of contrast.

    Returns
    -------
    jax.Array
        Mean component flux, equal to ``total_flux / 2``.
    """

    import jax.numpy as jnp

    del contrast
    return jnp.asarray(total_flux, dtype=float) / jnp.asarray(2.0, dtype=float)


def compute_source_flux_diagnostics(
    source_kind: str,
    store: ParameterStore,
) -> dict[str, Any]:
    """Return source-aware flux diagnostics from a parameter store.

    Parameters
    ----------
    source_kind:
        Source kind, for example ``"single_star"``, ``"binary"``, or
        ``"binary_target"``.
    store:
        Store containing the public dLuxShera source parameters.

    Returns
    -------
    dict[str, Any]
        Mapping with ``total_flux`` and ``component_fluxes``. Binary sources
        also include ``contrast``.

    Raises
    ------
    ValueError
        If the source kind is unknown.
    KeyError
        If required store keys are missing.
    """

    kind = str(source_kind).lower()
    total_flux = linear_total_flux_from_log10(store.get("source.log_flux_total"))

    if kind == "single_star":
        return {
            "total_flux": total_flux,
            "component_fluxes": {"star": total_flux},
        }

    if kind in {"binary", "binary_target", "alpha_cen"}:
        contrast = store.get("source.contrast")
        primary, secondary = binary_component_fluxes_from_total_and_contrast(
            total_flux,
            contrast,
        )
        return {
            "total_flux": total_flux,
            "contrast": contrast,
            "component_fluxes": {
                "primary": primary,
                "secondary": secondary,
            },
        }

    raise ValueError(
        "Unknown source kind for flux diagnostics: "
        f"{source_kind!r}. Supported kinds: binary_target, binary, single_star."
    )


def _derive_nominal_contrast(
    *,
    target_spec: TargetSpec | None,
    wavelength_m: float | None,
    bandwidth_m: float | None,
    n_lambda: int | None,
    vmag_a: float | None,
    vmag_b: float | None,
) -> float:
    """Return nominal ``A/B`` contrast from SEDs with V-mag fallback."""
    if wavelength_m is None or bandwidth_m is None or n_lambda is None:
        if target_spec and target_spec.nominal_contrast:
            return float(target_spec.nominal_contrast)
        return 3.0

    wavelength_grid_m = build_wavelength_grid_m(
        wavelength_m=float(wavelength_m),
        bandwidth_m=float(bandwidth_m),
        n_lambda=int(n_lambda),
    )

    sed_a_ref = None
    sed_b_ref = None
    if target_spec and target_spec.sed_a_file and target_spec.sed_b_file:
        sed_root = target_sed_root()
        sed_a_ref = sed_root.joinpath(target_spec.sed_a_file)
        sed_b_ref = sed_root.joinpath(target_spec.sed_b_file)

    try:
        if sed_a_ref is not None and sed_b_ref is not None and sed_a_ref.is_file() and sed_b_ref.is_file():
            with ExitStack() as stack:
                sed_a_path = Path(stack.enter_context(resources.as_file(sed_a_ref)))
                sed_b_path = Path(stack.enter_context(resources.as_file(sed_b_ref)))
                photometry = derive_source_photometry(
                    wavelength_grid_m=wavelength_grid_m,
                    bandwidth_m=float(bandwidth_m),
                    collecting_area_m2=1.0,
                    exposure_time_s=1.0,
                    throughput=1.0,
                    sed_a_path=sed_a_path,
                    sed_b_path=sed_b_path,
                    vmag_a=vmag_a,
                    vmag_b=vmag_b,
                )
        else:
            photometry = derive_source_photometry(
                wavelength_grid_m=wavelength_grid_m,
                bandwidth_m=float(bandwidth_m),
                collecting_area_m2=1.0,
                exposure_time_s=1.0,
                throughput=1.0,
                sed_a_path=None,
                sed_b_path=None,
                vmag_a=vmag_a,
                vmag_b=vmag_b,
            )
    except ValueError:
        if target_spec and target_spec.nominal_contrast:
            return float(target_spec.nominal_contrast)
        return 3.0

    return float(photometry.contrast)


def _base_spectral_fields(
    source_cfg: Mapping[str, Any],
    *,
    log_flux_kind: str = "primitive",
) -> list[ParamField]:
    """Return common wavelength/exposure/log-flux source fields."""

    log_flux_default = source_cfg.get("log_flux_total", 6.0)
    log_flux_kwargs: dict[str, Any] = {}
    if log_flux_kind == "derived":
        log_flux_kwargs = {
            "default": None,
            "transform": "source.log_flux_total",
            "depends_on": (
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
        }
    else:
        log_flux_kwargs = {"default": log_flux_default}

    return [
        ParamField(
            "source.wavelength_m",
            group="source",
            kind="primitive",
            dtype=float,
            shape=(),
            default=source_cfg.get("wavelength_m"),
            structural=True,
        ),
        ParamField(
            "source.bandwidth_m",
            group="source",
            kind="primitive",
            dtype=float,
            shape=(),
            default=source_cfg.get("bandwidth_m"),
            structural=True,
        ),
        ParamField(
            "source.n_lambda",
            group="source",
            kind="primitive",
            dtype=int,
            shape=(),
            default=source_cfg.get("n_lambda"),
            structural=True,
        ),
        ParamField(
            "source.exposure_time_s",
            group="source",
            kind="primitive",
            dtype=float,
            shape=(),
            default=source_cfg.get("exposure_time_s", 1.0),
            structural=False,
        ),
        ParamField(
            "source.log_flux_total",
            group="source",
            kind=log_flux_kind,
            dtype=float,
            shape=(),
            structural=False,
            binding="log_flux",
            **log_flux_kwargs,
        ),
    ]


def _position_fields(
    source_cfg: Mapping[str, Any],
    *,
    include_separation: bool,
    include_contrast: bool,
) -> list[ParamField]:
    """Return common source astrometry fields."""

    fields = [
        ParamField(
            "source.x_position_as",
            group="source",
            kind="primitive",
            dtype=float,
            shape=(),
            default=source_cfg.get("x_position_as", 0.0),
            structural=False,
            binding="x_position",
        ),
        ParamField(
            "source.y_position_as",
            group="source",
            kind="primitive",
            dtype=float,
            shape=(),
            default=source_cfg.get("y_position_as", 0.0),
            structural=False,
            binding="y_position",
        ),
        ParamField(
            "source.position_angle_deg",
            group="source",
            kind="primitive",
            dtype=float,
            shape=(),
            default=source_cfg.get("position_angle_deg", 0.0),
            structural=False,
            binding="position_angle",
        ),
    ]

    if include_separation:
        fields.insert(
            0,
            ParamField(
                "source.separation_as",
                group="source",
                kind="primitive",
                dtype=float,
                shape=(),
                default=source_cfg.get("separation_as", 10.0),
                structural=False,
                binding="separation",
            ),
        )

    if include_contrast:
        fields.append(
            ParamField(
                "source.contrast",
                group="source",
                kind="primitive",
                dtype=float,
                shape=(),
                default=source_cfg.get("contrast", 1.0),
                structural=False,
                binding="contrast",
            )
        )

    return fields


def build_single_star_contract(source_cfg: Mapping[str, Any]) -> ParamSpec:
    """Return the calibration-friendly single-star source contract.

    The public flux parameter is ``source.log_flux_total``. No binary-only
    separation, contrast, or target photometry keys are required.
    """

    source_cfg = _extract_source_mapping(source_cfg)
    return ParamSpec(
        [
            *_base_spectral_fields(source_cfg, log_flux_kind="primitive"),
            *_position_fields(
                source_cfg,
                include_separation=False,
                include_contrast=False,
            ),
        ]
    )


def build_binary_contract(source_cfg: Mapping[str, Any]) -> ParamSpec:
    """Return the generic dLux ``BinarySource`` contract.

    This path is independent of the curated target registry. The public flux
    parameter remains ``source.log_flux_total`` and is converted to dLux
    ``mean_flux`` only inside the source builder.
    """

    source_cfg = _extract_source_mapping(source_cfg)
    return ParamSpec(
        [
            *_base_spectral_fields(source_cfg, log_flux_kind="primitive"),
            *_position_fields(
                source_cfg,
                include_separation=True,
                include_contrast=True,
            ),
        ]
    )


def build_binary_target_contract(source_cfg: Mapping[str, Any]) -> ParamSpec:
    """Return the generic binary-target parameter contract under ``source.*`` keys.

    Curated target photometry defaults are SED-authoritative where available.
    If SEDs are missing or omitted, defaults fall back to Johnson-V magnitudes
    and uniform component weights.

    Runtime values in the store remain authoritative for source primitives.
    """

    source_cfg = _extract_source_mapping(source_cfg)

    target_key = source_cfg.get("target")
    target_spec = get_target_spec(str(target_key)) if target_key else None

    wavelength_m = source_cfg.get("wavelength_m")
    bandwidth_m = source_cfg.get("bandwidth_m")
    n_lambda = source_cfg.get("n_lambda")
    exposure_time_s = source_cfg.get("exposure_time_s", 1.0)
    default_target = str(target_key).strip().upper() if target_key else None
    default_vmag_a = source_cfg.get("vmag_a", target_spec.vmag_a if target_spec else None)
    default_vmag_b = source_cfg.get("vmag_b", target_spec.vmag_b if target_spec else None)

    default_separation = source_cfg.get(
        "separation_as",
        target_spec.nominal_separation_as if target_spec else 10.0,
    )
    default_pa = source_cfg.get(
        "position_angle_deg",
        target_spec.nominal_position_angle_deg if target_spec else 90.0,
    )
    if target_spec is None and source_cfg.get("contrast") is not None:
        default_contrast = source_cfg.get("contrast")
    else:
        default_contrast = _derive_nominal_contrast(
            target_spec=target_spec,
            wavelength_m=wavelength_m,
            bandwidth_m=bandwidth_m,
            n_lambda=n_lambda,
            vmag_a=default_vmag_a,
            vmag_b=default_vmag_b,
        )

    log_flux_dependencies = (
        "source.target",
        "source.vmag_a",
        "source.vmag_b",
        "source.wavelength_m",
        "optics.m1_diameter_m",
        "source.bandwidth_m",
        "source.n_lambda",
        "source.exposure_time_s",
        "optics.throughput",
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
            "source.target",
            group="source",
            kind="primitive",
            dtype=str,
            shape=(),
            default=default_target,
            structural=False,
        ),
        ParamField(
            "source.vmag_a",
            group="source",
            kind="primitive",
            dtype=float,
            shape=(),
            default=default_vmag_a,
            structural=False,
        ),
        ParamField(
            "source.vmag_b",
            group="source",
            kind="primitive",
            dtype=float,
            shape=(),
            default=default_vmag_b,
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
    "binary_component_fluxes_from_total_and_contrast",
    "binary_mean_flux_from_total_and_contrast",
    "build_alpha_cen_contract",
    "build_binary_contract",
    "build_binary_target_contract",
    "build_single_star_contract",
    "compute_source_flux_diagnostics",
    "get_target_spec",
    "linear_total_flux_from_log10",
]
