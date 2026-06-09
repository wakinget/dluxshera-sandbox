"""Source-level effective spectral response utilities.

This module builds deterministic detected spectral mixtures for future SHERA
truth/knowledge deck generation. It deliberately stays NumPy-only and does not
instantiate dLux sources, optical layers, detector layers, or active inference
parameters.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib import resources
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..components.sources import get_target_spec
from .source_photometry import load_sed_photon_flux_density_per_nm, target_sed_root

__all__ = [
    "EffectiveSpectrum",
    "SpectralDeck",
    "SpectralComparison",
    "SourceSpectralDeck",
    "DEFAULT_DETECTOR_QE_PATH",
    "DEFAULT_FILTER_RESPONSE_PATH",
    "resolve_response_curve_path",
    "load_response_curve_csv",
    "interpolate_response_curve",
    "build_effective_spectrum",
    "build_truth_inference_spectral_deck",
    "resolve_source_sed_components",
    "build_target_aware_spectral_deck",
    "write_spectral_deck_artifacts",
]

SCHEMA_VERSION = "spectral_throughput_deck.v1"
DEFAULT_FILTER_RESPONSE_PATH = "data/filter_response/SHERA Notch Filter V2.csv"
DEFAULT_FILTER_RESPONSE_V1_PATH = "data/filter_response/SHERA Notch Filter V1.csv"
DEFAULT_DETECTOR_QE_PATH = "data/detector_qe/LTN4323_QE.csv"
DETECTOR_QE_PROXY_ASSUMPTION = (
    "LTN4323 QE curve used as near-term proxy for HWK4123 detector QE because "
    "the detector models are nearly identical in relevant specifications."
)
WAVELENGTH_UNIT_TO_M = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "um": 1e-6,
    "micron": 1e-6,
    "microns": 1e-6,
    "nm": 1e-9,
    "nanometer": 1e-9,
    "nanometers": 1e-9,
    "angstrom": 1e-10,
    "angstroms": 1e-10,
    "a": 1e-10,
}


@dataclass(frozen=True)
class EffectiveSpectrum:
    """Represent one source-level effective detected spectrum.

    Use this object as the reusable boundary between spectral-throughput deck
    realization and future dLux source construction. ``weights`` are normalized
    to sum to one and can be consumed as chromatic mixture weights; the scalar
    ``flux_factor`` records the integrated detected response before
    normalization.

    Parameters
    ----------
    label:
        Human-readable component or model label.
    wavelengths_m:
        Wavelength samples in meters.
    weights:
        Sample-normalized spectral weights. The sum is one.
    flux_factor:
        Trapezoidal integral of ``raw_response`` over wavelength in nm.
    raw_response:
        Unnormalized detected spectral contribution sampled on
        ``wavelengths_m``.
    diagnostics:
        Scalar spectral moments, normalization checks, and leakage fractions.
    provenance:
        Source paths, component labels, units, and assumptions.
    """

    label: str
    wavelengths_m: np.ndarray
    weights: np.ndarray
    flux_factor: float
    raw_response: np.ndarray
    diagnostics: dict[str, Any]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class SpectralComparison:
    """Represent comparison diagnostics for a truth/inference spectrum pair."""

    metrics: dict[str, Any]


@dataclass(frozen=True)
class SpectralDeck:
    """Represent a truth/inference spectral-throughput deck.

    The deck is intentionally limited to effective source spectra and comparison
    diagnostics. It does not mutate source configs or wire the results into image
    rendering; future deck generators can serialize this object and then decide
    how to inject ``wavelengths_m`` and ``weights`` into system configs.
    """

    truth: EffectiveSpectrum
    inference: EffectiveSpectrum
    comparison: dict[str, Any]
    schema_version: str
    provenance: dict[str, Any]


@dataclass(frozen=True)
class SourceSpectralDeck:
    """Represent source spectra by component for source-config integration.

    This object is the component-aware counterpart to :class:`SpectralDeck`.
    Single-star decks contain one ``"star"`` component. Binary-like decks
    contain ``"primary"`` and ``"secondary"`` components built on shared truth
    and inference wavelength grids so they can be applied to one dLux source
    config as row-normalized component weights.
    """

    source_kind: str
    target: str | None
    truth_by_component: dict[str, EffectiveSpectrum]
    inference_by_component: dict[str, EffectiveSpectrum]
    comparison_by_component: dict[str, dict[str, Any]]
    combined_comparison: dict[str, Any]
    schema_version: str
    provenance: dict[str, Any]


# -----------------------------------------------------------------------------
# Private helpers


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _repo_root() -> Path:
    """Return the source-tree repository root when running from checkout."""

    return Path(__file__).resolve().parents[3]


def resolve_response_curve_path(path: str | Path) -> Path:
    """Resolve response-curve paths used by spectral deck configs.

    The resolver accepts absolute paths, paths relative to the current working
    directory, and template-facing ``data/...`` paths. In this checkout the
    response CSVs live under ``src/dluxshera/data``; ``data/...`` is therefore
    treated as an auditable shorthand for package data.
    """

    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw
    if raw.is_file():
        return raw

    repo_candidate = _repo_root() / raw
    if repo_candidate.is_file():
        return repo_candidate

    parts = raw.parts
    if parts and parts[0] == "data":
        package_candidate = Path(__file__).resolve().parents[1] / raw
        if package_candidate.is_file():
            return package_candidate

    return raw


def _unit_scale(wavelength_unit: str) -> float:
    key = str(wavelength_unit).strip().lower()
    if key not in WAVELENGTH_UNIT_TO_M:
        supported = ", ".join(sorted(WAVELENGTH_UNIT_TO_M))
        raise ValueError(f"Unsupported wavelength_unit={wavelength_unit!r}; expected one of {supported}.")
    return WAVELENGTH_UNIT_TO_M[key]


def _as_1d_float_array(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _validate_wavelength_grid(wavelengths_m: Any) -> np.ndarray:
    wavelengths = _as_1d_float_array(wavelengths_m, name="wavelengths_m")
    if np.any(wavelengths <= 0.0):
        raise ValueError("wavelengths_m must be strictly positive.")
    if wavelengths.size > 1 and np.any(np.diff(wavelengths) <= 0.0):
        raise ValueError("wavelengths_m must be strictly increasing.")
    return wavelengths


def _validate_nonnegative(values: np.ndarray, *, name: str, clip_negative: bool) -> np.ndarray:
    if np.any(values < 0.0):
        if not clip_negative:
            minimum = float(np.min(values))
            raise ValueError(f"{name} contains negative response values; minimum={minimum}.")
        values = np.clip(values, a_min=0.0, a_max=None)
    return values


def _validate_response_upper_bound(values: np.ndarray, *, name: str, allow_above_one: bool) -> None:
    if not allow_above_one and np.any(values > 1.0):
        maximum = float(np.max(values))
        raise ValueError(f"{name} contains values above 1.0; maximum={maximum}.")


def _resolve_wavelength_grid(config: Mapping[str, Any], *, default_min_nm: float, default_max_nm: float) -> np.ndarray:
    n_lambda = int(config.get("n_lambda", 30))
    if n_lambda <= 0:
        raise ValueError("n_lambda must be positive.")
    minimum = config.get("wavelength_min_nm", None)
    maximum = config.get("wavelength_max_nm", None)
    min_nm = default_min_nm if minimum is None else float(minimum)
    max_nm = default_max_nm if maximum is None else float(maximum)
    if not max_nm > min_nm:
        raise ValueError("wavelength_max_nm must be greater than wavelength_min_nm.")
    return np.linspace(min_nm, max_nm, n_lambda, dtype=float) * 1e-9


def _resolve_sampled_curve(spec: Mapping[str, Any], *, wavelengths_m: np.ndarray, kind: str) -> tuple[np.ndarray, dict[str, Any]]:
    label = str(spec.get("label", kind))
    fill_value = float(spec.get("fill_value", 0.0))
    clip_negative = bool(spec.get("clip_negative", False))
    allow_above_one = bool(spec.get("allow_above_one", False))

    if "path" in spec and spec.get("path") is not None:
        configured_path = Path(spec["path"])
        path = resolve_response_curve_path(configured_path)
        curve_wavelengths, curve_values = load_response_curve_csv(
            path,
            wavelength_column=str(spec.get("wavelength_column", "wavelength")),
            response_column=str(spec.get("response_column", "response")),
            wavelength_unit=str(spec.get("wavelength_unit", "nm")),
            response_scale=float(spec.get("response_scale", 1.0)),
            clip_negative=clip_negative,
            allow_above_one=allow_above_one,
        )
        values = interpolate_response_curve(
            wavelengths_m,
            curve_wavelengths,
            curve_values,
            fill_value=fill_value,
            clip_negative=clip_negative,
            allow_above_one=allow_above_one,
            component_label=label,
        )
        provenance = {
            "label": label,
            "kind": kind,
            "path": str(configured_path),
            "resolved_path": str(path),
            "sha256": _sha256_file(path) if path.is_file() else None,
            "wavelength_column": spec.get("wavelength_column", "wavelength"),
            "response_column": spec.get("response_column", "response"),
            "wavelength_unit": spec.get("wavelength_unit", "nm"),
            "response_unit": spec.get("response_unit", "dimensionless"),
            "response_scale": float(spec.get("response_scale", 1.0)),
            "fill_value": fill_value,
        }
        if "assumption" in spec:
            provenance["assumption"] = spec["assumption"]
        if "detector_model_proxy_for" in spec:
            provenance["detector_model_proxy_for"] = spec["detector_model_proxy_for"]
        return values, provenance

    if "callable" in spec and spec.get("callable") is not None:
        fn = spec["callable"]
        if not callable(fn):
            raise ValueError(f"{label} callable response component is not callable.")
        values = _as_1d_float_array(fn(wavelengths_m), name=f"{label} response")
        if values.shape != wavelengths_m.shape:
            raise ValueError(f"{label} callable response shape {values.shape} does not match wavelengths {wavelengths_m.shape}.")
        values = _validate_nonnegative(values, name=label, clip_negative=clip_negative)
        _validate_response_upper_bound(values, name=label, allow_above_one=allow_above_one)
        return values, {"label": label, "kind": kind, "source": "callable"}

    if "wavelengths_m" in spec and "response" in spec:
        curve_wavelengths = _validate_wavelength_grid(spec["wavelengths_m"])
        curve_values = _as_1d_float_array(spec["response"], name=f"{label} response")
        values = interpolate_response_curve(
            wavelengths_m,
            curve_wavelengths,
            curve_values,
            fill_value=fill_value,
            clip_negative=clip_negative,
            allow_above_one=allow_above_one,
            component_label=label,
        )
        return values, {"label": label, "kind": kind, "source": "sampled_array", "fill_value": fill_value}

    if "response" in spec:
        raw = spec["response"]
        if np.isscalar(raw):
            values = np.full_like(wavelengths_m, float(raw), dtype=float)
        else:
            values = _as_1d_float_array(raw, name=f"{label} response")
            if values.shape != wavelengths_m.shape:
                raise ValueError(f"{label} response shape {values.shape} does not match wavelengths {wavelengths_m.shape}.")
        values = _validate_nonnegative(values, name=label, clip_negative=clip_negative)
        _validate_response_upper_bound(values, name=label, allow_above_one=allow_above_one)
        return values, {"label": label, "kind": kind, "source": "direct_array"}

    raise ValueError(f"Response component {label!r} must provide path, callable, or response samples.")


def _resolve_sed(sed: Any, wavelengths_m: np.ndarray, *, input_kind: str) -> tuple[np.ndarray, dict[str, Any]]:
    if input_kind != "photon_flux_density_per_nm":
        raise ValueError(
            "Only input_kind='photon_flux_density_per_nm' is supported in v1; "
            "energy-flux SED conversion must be requested explicitly in a future task."
        )

    if sed is None:
        return np.ones_like(wavelengths_m, dtype=float), {"source": "flat_unit_sed", "input_kind": input_kind}

    if isinstance(sed, (str, Path)):
        path = Path(sed)
        values = load_sed_photon_flux_density_per_nm(path, wavelengths_m)
        return values, {
            "source": "sed_file",
            "path": str(path),
            "sha256": _sha256_file(path) if path.is_file() else None,
            "input_kind": input_kind,
            "wavelength_unit": "nm",
            "flux_unit": "photons / s / m^2 / nm",
        }

    if callable(sed):
        values = _as_1d_float_array(sed(wavelengths_m), name="sed")
        if values.shape != wavelengths_m.shape:
            raise ValueError(f"Callable SED shape {values.shape} does not match wavelengths {wavelengths_m.shape}.")
        values = _validate_nonnegative(values, name="sed", clip_negative=False)
        return values, {"source": "callable", "input_kind": input_kind}

    if isinstance(sed, Mapping):
        if "path" in sed and sed.get("path") is not None:
            return _resolve_sed(Path(sed["path"]), wavelengths_m, input_kind=str(sed.get("input_kind", input_kind)))
        if "callable" in sed and sed.get("callable") is not None:
            return _resolve_sed(sed["callable"], wavelengths_m, input_kind=str(sed.get("input_kind", input_kind)))
        if "wavelengths_m" in sed and "values" in sed:
            sed_wavelengths = _validate_wavelength_grid(sed["wavelengths_m"])
            sed_values = _as_1d_float_array(sed["values"], name="sed values")
            if sed_wavelengths.shape != sed_values.shape:
                raise ValueError("SED wavelengths_m and values must have identical shapes.")
            values = np.interp(wavelengths_m, sed_wavelengths, sed_values, left=0.0, right=0.0)
            values = _validate_nonnegative(values, name="sed", clip_negative=False)
            return values, {"source": "sampled_array", "input_kind": input_kind, "fill_value": 0.0}
        if "values" in sed:
            return _resolve_sed(sed["values"], wavelengths_m, input_kind=str(sed.get("input_kind", input_kind)))

    if np.isscalar(sed):
        value = float(sed)
        if value < 0.0:
            raise ValueError("Scalar SED must be non-negative.")
        return np.full_like(wavelengths_m, value, dtype=float), {"source": "scalar", "input_kind": input_kind}

    values = _as_1d_float_array(sed, name="sed")
    if values.shape != wavelengths_m.shape:
        raise ValueError(f"SED shape {values.shape} does not match wavelengths {wavelengths_m.shape}.")
    values = _validate_nonnegative(values, name="sed", clip_negative=False)
    return values, {"source": "direct_array", "input_kind": input_kind}


def _integrate_raw_response(raw_response: np.ndarray, wavelengths_m: np.ndarray) -> float:
    if raw_response.size == 1:
        return float(raw_response[0])
    return float(np.trapezoid(raw_response, wavelengths_m * 1e9))


def _integrate_fraction_in_band(raw_response: np.ndarray, wavelengths_m: np.ndarray, in_band_nm: tuple[float, float] | None) -> tuple[float | None, float | None]:
    if in_band_nm is None:
        return None, None
    total = _integrate_raw_response(raw_response, wavelengths_m)
    if not total > 0.0:
        return 0.0, 1.0
    wavelength_nm = wavelengths_m * 1e9
    mask = (wavelength_nm >= float(in_band_nm[0])) & (wavelength_nm <= float(in_band_nm[1]))
    if not np.any(mask):
        return 0.0, 1.0
    if np.count_nonzero(mask) == 1:
        in_band = float(raw_response[mask][0])
        total_sum = float(np.sum(raw_response))
        fraction = 0.0 if total_sum <= 0.0 else in_band / total_sum
    else:
        fraction = _integrate_raw_response(raw_response[mask], wavelengths_m[mask]) / total
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return fraction, 1.0 - fraction


def _spectrum_diagnostics(
    *,
    wavelengths_m: np.ndarray,
    weights: np.ndarray,
    raw_response: np.ndarray,
    flux_factor: float,
    in_band_nm: tuple[float, float] | None,
) -> dict[str, Any]:
    wavelength_nm = wavelengths_m * 1e9
    lambda_eff_nm = float(np.sum(weights * wavelength_nm))
    variance = float(np.sum(weights * (wavelength_nm - lambda_eff_nm) ** 2))
    peak_index = int(np.argmax(weights))
    in_band_fraction, out_of_band_fraction = _integrate_fraction_in_band(raw_response, wavelengths_m, in_band_nm)

    diagnostics: dict[str, Any] = {
        "n_lambda": int(wavelengths_m.size),
        "lambda_min_nm": float(wavelength_nm[0]),
        "lambda_max_nm": float(wavelength_nm[-1]),
        "lambda_eff_nm": lambda_eff_nm,
        "bandwidth_rms_nm": float(np.sqrt(max(variance, 0.0))),
        "flux_factor": float(flux_factor),
        "weights_sum": float(np.sum(weights)),
        "raw_response_sum": float(np.sum(raw_response)),
        "raw_response_integral": float(flux_factor),
        "raw_response_integral_unit": "sample_units * nm",
        "peak_weight": float(weights[peak_index]),
        "peak_wavelength_nm": float(wavelength_nm[peak_index]),
        "weight_normalization": "sample_sum",
    }
    if in_band_fraction is not None:
        diagnostics["in_band_fraction"] = float(in_band_fraction)
        diagnostics["out_of_band_fraction"] = float(out_of_band_fraction)
        diagnostics["in_band_min_nm"] = float(in_band_nm[0])
        diagnostics["in_band_max_nm"] = float(in_band_nm[1])
    return diagnostics


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if callable(value):
        return getattr(value, "__name__", repr(value))
    return value


def _spectrum_to_record(spectrum: EffectiveSpectrum) -> dict[str, Any]:
    record = asdict(spectrum)
    return _json_ready(record)


def _as_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(config or {})


# -----------------------------------------------------------------------------
# Public API


def load_response_curve_csv(
    path: str | Path,
    *,
    wavelength_column: str = "wavelength",
    response_column: str = "response",
    wavelength_unit: str = "nm",
    response_scale: float = 1.0,
    clip_negative: bool = False,
    allow_above_one: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a dimensionless response curve from a CSV file.

    Use this for detector QE, filter transmission, mirror reflectance, or other
    multiplicative throughput curves. Wavelengths are returned in meters;
    response values are dimensionless and validated to be finite and
    non-negative by default.

    Parameters
    ----------
    path:
        CSV file path. Headered CSV files should contain ``wavelength_column``
        and ``response_column``. Headerless two-column files are also accepted
        when the default column names are not present.
    wavelength_column:
        Name of the wavelength column for headered CSV files.
    response_column:
        Name of the response column for headered CSV files.
    wavelength_unit:
        Explicit wavelength unit: ``m``, ``um``, ``nm``, or ``angstrom``.
    response_scale:
        Multiplicative scale applied after parsing. Use ``0.01`` for percent
        transmission columns such as ``T (%)``.
    clip_negative:
        If true, negative response samples are clipped to zero. By default they
        raise a ``ValueError``.
    allow_above_one:
        If true, response values above one are accepted. By default they raise a
        ``ValueError`` because QE/filter/reflectance curves should normally be
        throughput factors.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Sorted ``(wavelengths_m, response)`` arrays.
    """

    csv_path = resolve_response_curve_path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Response curve CSV does not exist: {csv_path}")

    wavelengths: list[float] = []
    responses: list[float] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))

    def _clean(cell: str) -> str:
        return str(cell).strip().lstrip("\ufeff")

    header_index = None
    wavelength_index = None
    response_index = None
    for idx, row in enumerate(rows):
        cleaned = [_clean(cell) for cell in row]
        if wavelength_column in cleaned and response_column in cleaned:
            header_index = idx
            wavelength_index = cleaned.index(wavelength_column)
            response_index = cleaned.index(response_column)
            break

    if header_index is not None:
        assert wavelength_index is not None
        assert response_index is not None
        for row in rows[header_index + 1 :]:
            if not row or not any(_clean(cell) for cell in row):
                continue
            if len(row) <= max(wavelength_index, response_index):
                continue
            wavelength_cell = _clean(row[wavelength_index])
            response_cell = _clean(row[response_index])
            if not wavelength_cell or not response_cell or wavelength_cell.startswith("#"):
                continue
            wavelengths.append(float(wavelength_cell))
            responses.append(float(response_cell) * float(response_scale))
    else:
        # Backward-compatible headerless two-column format.
        for row in rows:
            if not row or row[0].strip().startswith("#"):
                continue
            if len(row) < 2:
                raise ValueError(f"Headerless response CSV {csv_path} must have at least two columns.")
            try:
                wavelengths.append(float(_clean(row[0])))
                responses.append(float(_clean(row[1])) * float(response_scale))
            except ValueError as exc:
                raise ValueError(
                    f"Response CSV {csv_path} must contain columns "
                    f"{wavelength_column!r} and {response_column!r}."
                ) from exc

    wavelength_arr = _as_1d_float_array(wavelengths, name="response wavelengths") * _unit_scale(wavelength_unit)
    response_arr = _as_1d_float_array(responses, name="response values")
    if wavelength_arr.shape != response_arr.shape:
        raise ValueError("Response wavelength and value arrays must have identical shapes.")
    if wavelength_arr.size < 2:
        raise ValueError("Response curve must contain at least two samples.")
    response_arr = _validate_nonnegative(response_arr, name=str(csv_path), clip_negative=clip_negative)
    _validate_response_upper_bound(response_arr, name=str(csv_path), allow_above_one=allow_above_one)

    order = np.argsort(wavelength_arr)
    wavelength_arr = wavelength_arr[order]
    response_arr = response_arr[order]
    if np.any(np.diff(wavelength_arr) <= 0.0):
        raise ValueError(f"Response curve wavelengths must be unique and increasing after sorting: {csv_path}")
    return wavelength_arr, response_arr


def interpolate_response_curve(
    wavelengths_m: Sequence[float],
    curve_wavelengths_m: Sequence[float],
    curve_response: Sequence[float],
    *,
    fill_value: float = 0.0,
    clip_negative: bool = False,
    allow_above_one: bool = False,
    component_label: str = "response",
) -> np.ndarray:
    """Interpolate a dimensionless response curve onto a wavelength grid.

    Out-of-range values default to zero so truncated inference bands can assume
    no modeled response outside their explicit grid. Callers can change
    ``fill_value`` for special cases, but the default is the conservative deck
    behavior used by the full-fidelity spectral mismatch study.
    """

    target = _validate_wavelength_grid(wavelengths_m)
    curve_wavelengths = _validate_wavelength_grid(curve_wavelengths_m)
    curve_values = _as_1d_float_array(curve_response, name=f"{component_label} response")
    if curve_wavelengths.shape != curve_values.shape:
        raise ValueError("curve_wavelengths_m and curve_response must have identical shapes.")
    curve_values = _validate_nonnegative(curve_values, name=component_label, clip_negative=clip_negative)
    _validate_response_upper_bound(curve_values, name=component_label, allow_above_one=allow_above_one)
    interpolated = np.interp(target, curve_wavelengths, curve_values, left=fill_value, right=fill_value)
    interpolated = _validate_nonnegative(interpolated, name=component_label, clip_negative=clip_negative)
    _validate_response_upper_bound(interpolated, name=component_label, allow_above_one=allow_above_one)
    return interpolated


def build_effective_spectrum(
    *,
    label: str,
    wavelengths_m: Sequence[float] | None = None,
    wavelength_min_nm: float | None = None,
    wavelength_max_nm: float | None = None,
    n_lambda: int | None = None,
    sed: Any = None,
    sed_input_kind: str = "photon_flux_density_per_nm",
    detector_qe: Mapping[str, Any] | None = None,
    filter_response: Mapping[str, Any] | None = None,
    mirror_reflectance_components: Sequence[Mapping[str, Any]] | None = None,
    response_components: Sequence[Mapping[str, Any]] | None = None,
    in_band_nm: tuple[float, float] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> EffectiveSpectrum:
    """Build one effective detected source spectrum.

    Compose a source SED with detector QE, filter response, optional mirror
    reflectance terms, and additional user-supplied throughput curves. The SED
    convention is photon spectral flux density per nm. Energy-flux SEDs are not
    silently converted in v1.

    Parameters
    ----------
    label:
        Spectrum label, for example ``"truth"`` or ``"inference"``.
    wavelengths_m:
        Explicit wavelength grid in meters. If omitted, ``wavelength_min_nm``,
        ``wavelength_max_nm``, and ``n_lambda`` are used.
    wavelength_min_nm, wavelength_max_nm:
        Wavelength range in nm used when ``wavelengths_m`` is omitted.
    n_lambda:
        Number of wavelength samples used when ``wavelengths_m`` is omitted.
    sed:
        Source SED as ``None``/scalar/array/callable/path/mapping. Paths are
        interpreted through existing packaged SED utilities as photon flux
        density per nm after interpolation.
    sed_input_kind:
        Must be ``"photon_flux_density_per_nm"`` in v1.
    detector_qe, filter_response:
        Optional response-component mappings. They accept ``path`` CSV inputs,
        ``callable`` inputs, sampled ``wavelengths_m`` plus ``response``, or a
        scalar/direct response array.
    mirror_reflectance_components, response_components:
        Additional multiplicative response components.
    in_band_nm:
        Optional diagnostic band. Fractions inside/outside this range are
        integrated from the raw response.
    provenance:
        Additional provenance fields copied into the returned object.

    Returns
    -------
    EffectiveSpectrum
        Spectrum with sample-normalized weights, raw response, flux factor, and
        diagnostics.
    """

    if wavelengths_m is None:
        if wavelength_min_nm is None or wavelength_max_nm is None or n_lambda is None:
            raise ValueError("Provide wavelengths_m or wavelength_min_nm, wavelength_max_nm, and n_lambda.")
        wavelengths = _resolve_wavelength_grid(
            {"wavelength_min_nm": wavelength_min_nm, "wavelength_max_nm": wavelength_max_nm, "n_lambda": n_lambda},
            default_min_nm=float(wavelength_min_nm),
            default_max_nm=float(wavelength_max_nm),
        )
    else:
        wavelengths = _validate_wavelength_grid(wavelengths_m)

    sed_values, sed_provenance = _resolve_sed(sed, wavelengths, input_kind=sed_input_kind)
    raw_response = np.array(sed_values, dtype=float, copy=True)

    component_specs: list[tuple[str, Mapping[str, Any]]] = []
    if detector_qe is not None:
        component_specs.append(("detector_qe", detector_qe))
    if filter_response is not None:
        component_specs.append(("filter_response", filter_response))
    for component in mirror_reflectance_components or ():
        component_specs.append(("mirror_reflectance", component))
    for component in response_components or ():
        component_specs.append(("user_response", component))

    component_provenance: list[dict[str, Any]] = []
    for kind, spec in component_specs:
        values, comp_prov = _resolve_sampled_curve(spec, wavelengths_m=wavelengths, kind=kind)
        raw_response *= values
        component_provenance.append(comp_prov)

    if not np.all(np.isfinite(raw_response)):
        raise ValueError("Effective raw_response contains non-finite values.")
    if np.any(raw_response < 0.0):
        raise ValueError("Effective raw_response contains negative values.")

    weight_total = float(np.sum(raw_response))
    if not weight_total > 0.0:
        raise ValueError("Cannot normalize an effective spectrum with zero summed raw response.")
    weights = raw_response / weight_total
    flux_factor = _integrate_raw_response(raw_response, wavelengths)
    if not flux_factor > 0.0:
        raise ValueError("Effective spectrum flux_factor must be positive.")

    diagnostics = _spectrum_diagnostics(
        wavelengths_m=wavelengths,
        weights=weights,
        raw_response=raw_response,
        flux_factor=flux_factor,
        in_band_nm=in_band_nm,
    )
    spectrum_provenance = {
        "schema_version": SCHEMA_VERSION,
        "label": str(label),
        "sed": sed_provenance,
        "response_components": component_provenance,
        "assumptions": {
            "log_flux_total_and_contrast": "detected_post_response_band_integrated",
            "spectral_shape_active_inference_parameter": False,
            "weight_normalization": "sample_sum",
            "flux_factor_integration_unit": "nm",
        },
    }
    spectrum_provenance.update(dict(provenance or {}))
    return EffectiveSpectrum(
        label=str(label),
        wavelengths_m=wavelengths,
        weights=weights,
        flux_factor=float(flux_factor),
        raw_response=raw_response,
        diagnostics=diagnostics,
        provenance=spectrum_provenance,
    )


def build_truth_inference_spectral_deck(
    *,
    sed: Any = None,
    truth_config: Mapping[str, Any] | None = None,
    inference_config: Mapping[str, Any] | None = None,
    detector_qe: Mapping[str, Any] | None = None,
    filter_response: Mapping[str, Any] | None = None,
    mirror_reflectance_components: Sequence[Mapping[str, Any]] | None = None,
    response_components: Sequence[Mapping[str, Any]] | None = None,
    inference_detector_qe: Mapping[str, Any] | None = None,
    inference_filter_response: Mapping[str, Any] | None = None,
    inference_mirror_reflectance_components: Sequence[Mapping[str, Any]] | None = None,
    inference_response_components: Sequence[Mapping[str, Any]] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> SpectralDeck:
    """Build a matched or mismatched truth/inference spectral deck.

    Truth defaults to a broad, dense 500--700 nm grid with 30 samples. Inference
    defaults to a narrower 525--675 nm grid with 7 samples and recomputes the
    effective spectrum under its own assumptions instead of slicing truth
    weights. Null wavelength bounds in config-like inputs resolve to these
    defaults so the full-fidelity YAML skeleton can be consumed before real
    response files exist.
    """

    truth_cfg = _as_config(truth_config)
    inference_cfg = _as_config(inference_config)
    truth_wavelengths = _resolve_wavelength_grid(truth_cfg, default_min_nm=500.0, default_max_nm=700.0)
    inference_wavelengths = _resolve_wavelength_grid(inference_cfg, default_min_nm=525.0, default_max_nm=675.0)
    inference_band = (float(inference_wavelengths[0] * 1e9), float(inference_wavelengths[-1] * 1e9))

    truth = build_effective_spectrum(
        label=str(truth_cfg.get("label", "truth")),
        wavelengths_m=truth_wavelengths,
        sed=sed,
        detector_qe=detector_qe,
        filter_response=filter_response,
        mirror_reflectance_components=mirror_reflectance_components,
        response_components=response_components,
        in_band_nm=inference_band,
        provenance={"model_role": "truth", "config": _json_ready(truth_cfg)},
    )
    inference = build_effective_spectrum(
        label=str(inference_cfg.get("label", "inference")),
        wavelengths_m=inference_wavelengths,
        sed=sed,
        detector_qe=inference_detector_qe if inference_detector_qe is not None else detector_qe,
        filter_response=inference_filter_response if inference_filter_response is not None else filter_response,
        mirror_reflectance_components=(
            inference_mirror_reflectance_components
            if inference_mirror_reflectance_components is not None
            else mirror_reflectance_components
        ),
        response_components=inference_response_components if inference_response_components is not None else response_components,
        in_band_nm=inference_band,
        provenance={
            "model_role": "inference",
            "config": _json_ready(inference_cfg),
            "out_of_band_response": inference_cfg.get("out_of_band_response", "zero"),
            "renormalize_weights": bool(inference_cfg.get("renormalize_weights", True)),
        },
    )
    comparison = compare_effective_spectra(truth, inference, inference_band_nm=inference_band)
    deck_provenance = {
        "generated_by": "dluxshera.utils.spectral_response.build_truth_inference_spectral_deck",
        "generated_at": _now_iso(),
        "active_inference_parameters": [],
        "spectral_shape_active_inference_parameter": False,
        "assumptions": {
            "log_flux_total_and_contrast": "detected_post_response_band_integrated",
            "inference_out_of_band_response": "zero",
            "inference_spectrum_recomputed_not_sliced": True,
        },
    }
    deck_provenance.update(dict(provenance or {}))
    return SpectralDeck(
        truth=truth,
        inference=inference,
        comparison=comparison,
        schema_version=SCHEMA_VERSION,
        provenance=deck_provenance,
    )


def _target_sed_path(filename: str) -> Path:
    ref = target_sed_root().joinpath(filename)
    with resources.as_file(ref) as path:
        resolved = Path(path)
        if not resolved.is_file():
            raise FileNotFoundError(f"Packaged target SED does not exist: {filename}")
        return resolved


def resolve_source_sed_components(
    source_cfg: Mapping[str, Any],
    *,
    sed_mode: str = "target",
    target: str | None = None,
    sed_path: str | Path | None = None,
    sed_a_path: str | Path | None = None,
    sed_b_path: str | Path | None = None,
    generic_binary_fallback: str = "require_explicit",
) -> dict[str, Any]:
    """Resolve source config into component SED paths and provenance.

    Binary target and ``alpha_cen`` sources use the existing curated
    ``TargetSpec`` registry. Generic binaries require explicit A/B SED paths by
    default; the render smoke can opt into ``generic_binary_fallback="alpha_cen"``
    for convenience with explicit provenance.
    """

    kind = str(source_cfg.get("kind", "binary_target")).lower()
    resolved_target = target or source_cfg.get("target")
    if kind == "alpha_cen" and not resolved_target:
        resolved_target = "ALPHA_CEN"
    if kind == "single_star" and not resolved_target:
        resolved_target = "ALPHA_CEN"

    if sed_mode in {"synthetic-ramp", "flat"}:
        components = {"star": None} if kind == "single_star" else {"primary": None, "secondary": None}
        return {
            "mode": sed_mode,
            "source_kind": kind,
            "target": resolved_target,
            "components": components,
            "shared_across_binary_components": kind != "single_star",
            "component_specific": False,
            "input_kind": "photon_flux_density_per_nm",
        }

    if kind == "single_star":
        path = Path(sed_path) if sed_path is not None else _target_sed_path("alfCenA_SED.dat")
        return {
            "mode": "single_star_default" if sed_path is None else "explicit",
            "source_kind": kind,
            "target": resolved_target,
            "components": {
                "star": {
                    "sed_path": str(sed_path or "data/target_seds/alfCenA_SED.dat"),
                    "resolved_sed_path": str(path),
                    "target_sed_label": "ALPHA_CEN_A" if sed_path is None else path.name,
                    "placeholder_note": "Alpha Cen A SED used as single-star calibration placeholder.",
                }
            },
            "shared_across_binary_components": False,
            "component_specific": False,
            "input_kind": "photon_flux_density_per_nm",
        }

    if sed_mode == "shared":
        if sed_path is None:
            raise ValueError("sed_mode='shared' requires sed_path.")
        path = Path(sed_path)
        return {
            "mode": "shared",
            "source_kind": kind,
            "target": resolved_target,
            "components": {
                "primary": {"sed_path": str(sed_path), "resolved_sed_path": str(path), "target_sed_label": path.name},
                "secondary": {"sed_path": str(sed_path), "resolved_sed_path": str(path), "target_sed_label": path.name},
            },
            "shared_across_binary_components": True,
            "component_specific": False,
            "warning": "Shared SED fallback used for binary source; component-specific SEDs are preferred.",
            "input_kind": "photon_flux_density_per_nm",
        }

    if sed_a_path is not None and sed_b_path is not None:
        path_a = Path(sed_a_path)
        path_b = Path(sed_b_path)
        return {
            "mode": "explicit",
            "source_kind": kind,
            "target": resolved_target,
            "components": {
                "primary": {"sed_path": str(sed_a_path), "resolved_sed_path": str(path_a), "target_sed_label": path_a.name},
                "secondary": {"sed_path": str(sed_b_path), "resolved_sed_path": str(path_b), "target_sed_label": path_b.name},
            },
            "shared_across_binary_components": False,
            "component_specific": True,
            "input_kind": "photon_flux_density_per_nm",
        }
    if sed_mode == "explicit":
        raise ValueError("sed_mode='explicit' requires sed_a_path and sed_b_path for binary-like sources.")

    if kind == "binary" and not resolved_target:
        if generic_binary_fallback == "alpha_cen":
            resolved_target = "ALPHA_CEN"
        else:
            raise ValueError(
                "Generic binary spectral SED resolution requires sed_a_path and "
                "sed_b_path, or generic_binary_fallback='alpha_cen' for smoke use."
            )

    if not resolved_target:
        raise ValueError("Target-aware binary SED resolution requires source.target or target.")

    spec = get_target_spec(str(resolved_target))
    if not spec.sed_a_file or not spec.sed_b_file:
        raise ValueError(f"Target {resolved_target!r} does not define component SED files.")
    path_a = _target_sed_path(spec.sed_a_file)
    path_b = _target_sed_path(spec.sed_b_file)
    return {
        "mode": "target" if not (kind == "binary" and generic_binary_fallback == "alpha_cen") else "smoke_alpha_cen_fallback",
        "source_kind": kind,
        "target": spec.key,
        "components": {
            "primary": {
                "sed_path": f"data/target_seds/{spec.sed_a_file}",
                "resolved_sed_path": str(path_a),
                "target_sed_label": f"{spec.key}:primary",
            },
            "secondary": {
                "sed_path": f"data/target_seds/{spec.sed_b_file}",
                "resolved_sed_path": str(path_b),
                "target_sed_label": f"{spec.key}:secondary",
            },
        },
        "shared_across_binary_components": False,
        "component_specific": True,
        "input_kind": "photon_flux_density_per_nm",
    }


def build_target_aware_spectral_deck(
    *,
    source_cfg: Mapping[str, Any],
    truth_config: Mapping[str, Any] | None = None,
    inference_config: Mapping[str, Any] | None = None,
    detector_qe: Mapping[str, Any] | None = None,
    filter_response: Mapping[str, Any] | None = None,
    mirror_reflectance_components: Sequence[Mapping[str, Any]] | None = None,
    response_components: Sequence[Mapping[str, Any]] | None = None,
    inference_detector_qe: Mapping[str, Any] | None = None,
    inference_filter_response: Mapping[str, Any] | None = None,
    sed_mode: str = "target",
    target: str | None = None,
    sed_path: str | Path | None = None,
    sed_a_path: str | Path | None = None,
    sed_b_path: str | Path | None = None,
    generic_binary_fallback: str = "require_explicit",
    provenance: Mapping[str, Any] | None = None,
) -> SourceSpectralDeck:
    """Build truth/inference spectra for each source component."""

    truth_cfg = _as_config(truth_config)
    inference_cfg = _as_config(inference_config)
    truth_wavelengths = _resolve_wavelength_grid(truth_cfg, default_min_nm=500.0, default_max_nm=700.0)
    inference_wavelengths = _resolve_wavelength_grid(inference_cfg, default_min_nm=525.0, default_max_nm=675.0)
    inference_band = (float(inference_wavelengths[0] * 1e9), float(inference_wavelengths[-1] * 1e9))
    sed_resolution = resolve_source_sed_components(
        source_cfg,
        sed_mode=sed_mode,
        target=target,
        sed_path=sed_path,
        sed_a_path=sed_a_path,
        sed_b_path=sed_b_path,
        generic_binary_fallback=generic_binary_fallback,
    )

    truth_by_component: dict[str, EffectiveSpectrum] = {}
    inference_by_component: dict[str, EffectiveSpectrum] = {}
    comparisons: dict[str, dict[str, Any]] = {}

    components = sed_resolution["components"]
    for label, sed_info in components.items():
        if sed_resolution["mode"] == "synthetic-ramp":
            sed: Any = lambda wavelengths_m: wavelengths_m * 1e9
        elif sed_resolution["mode"] == "flat":
            sed = 1.0
        else:
            sed = Path(sed_info["resolved_sed_path"])
        truth = build_effective_spectrum(
            label=f"truth_{label}",
            wavelengths_m=truth_wavelengths,
            sed=sed,
            detector_qe=detector_qe,
            filter_response=filter_response,
            mirror_reflectance_components=mirror_reflectance_components,
            response_components=response_components,
            in_band_nm=inference_band,
            provenance={"model_role": "truth", "component_label": label, "sed_component": _json_ready(sed_info)},
        )
        inference = build_effective_spectrum(
            label=f"inference_{label}",
            wavelengths_m=inference_wavelengths,
            sed=sed,
            detector_qe=inference_detector_qe if inference_detector_qe is not None else detector_qe,
            filter_response=inference_filter_response if inference_filter_response is not None else filter_response,
            mirror_reflectance_components=mirror_reflectance_components,
            response_components=response_components,
            in_band_nm=inference_band,
            provenance={"model_role": "inference", "component_label": label, "sed_component": _json_ready(sed_info)},
        )
        truth_by_component[label] = truth
        inference_by_component[label] = inference
        comparisons[label] = compare_effective_spectra(truth, inference, inference_band_nm=inference_band)

    combined: dict[str, Any] = {}
    if {"primary", "secondary"}.issubset(truth_by_component):
        p_truth = truth_by_component["primary"]
        s_truth = truth_by_component["secondary"]
        p_inf = inference_by_component["primary"]
        s_inf = inference_by_component["secondary"]
        combined = {
            "truth_primary_minus_secondary_lambda_eff_nm": float(p_truth.diagnostics["lambda_eff_nm"] - s_truth.diagnostics["lambda_eff_nm"]),
            "inference_primary_minus_secondary_lambda_eff_nm": float(p_inf.diagnostics["lambda_eff_nm"] - s_inf.diagnostics["lambda_eff_nm"]),
            "truth_flux_factor_ratio_secondary_over_primary": float(s_truth.flux_factor / p_truth.flux_factor),
            "inference_flux_factor_ratio_secondary_over_primary": float(s_inf.flux_factor / p_inf.flux_factor),
            "component_weights_differ_truth": bool(not np.allclose(p_truth.weights, s_truth.weights)),
            "component_weights_differ_inference": bool(not np.allclose(p_inf.weights, s_inf.weights)),
        }

    deck_provenance = {
        "generated_by": "dluxshera.utils.spectral_response.build_target_aware_spectral_deck",
        "generated_at": _now_iso(),
        "sed_resolution": _json_ready(sed_resolution),
        "active_inference_parameters": [],
        "spectral_shape_active_inference_parameter": False,
        "assumptions": {
            "log_flux_total_and_contrast": "detected_post_response_band_integrated",
            "component_flux_factors": "diagnostic_provenance_only",
        },
    }
    deck_provenance.update(dict(provenance or {}))
    return SourceSpectralDeck(
        source_kind=str(source_cfg.get("kind", "binary_target")).lower(),
        target=sed_resolution.get("target"),
        truth_by_component=truth_by_component,
        inference_by_component=inference_by_component,
        comparison_by_component=comparisons,
        combined_comparison=combined,
        schema_version=SCHEMA_VERSION,
        provenance=deck_provenance,
    )


def compare_effective_spectra(
    truth: EffectiveSpectrum,
    inference: EffectiveSpectrum,
    *,
    inference_band_nm: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Return truth-vs-inference spectral-shape comparison metrics."""

    truth_nm = truth.wavelengths_m * 1e9
    inference_nm = inference.wavelengths_m * 1e9
    common_min = min(float(truth_nm[0]), float(inference_nm[0]))
    common_max = max(float(truth_nm[-1]), float(inference_nm[-1]))
    n_common = max(int(truth_nm.size), int(inference_nm.size), 64)
    common_nm = np.linspace(common_min, common_max, n_common)
    truth_proj = np.interp(common_nm, truth_nm, truth.weights, left=0.0, right=0.0)
    inference_proj = np.interp(common_nm, inference_nm, inference.weights, left=0.0, right=0.0)
    if np.sum(truth_proj) > 0.0:
        truth_proj = truth_proj / np.sum(truth_proj)
    if np.sum(inference_proj) > 0.0:
        inference_proj = inference_proj / np.sum(inference_proj)
    diff = inference_proj - truth_proj

    if inference_band_nm is None:
        inference_band_nm = (float(inference_nm[0]), float(inference_nm[-1]))
    _in_band, truth_out = _integrate_fraction_in_band(truth.raw_response, truth.wavelengths_m, inference_band_nm)

    metrics = {
        "truth_lambda_eff_nm": truth.diagnostics["lambda_eff_nm"],
        "inference_lambda_eff_nm": inference.diagnostics["lambda_eff_nm"],
        "delta_lambda_eff_nm": float(inference.diagnostics["lambda_eff_nm"] - truth.diagnostics["lambda_eff_nm"]),
        "truth_bandwidth_rms_nm": truth.diagnostics["bandwidth_rms_nm"],
        "inference_bandwidth_rms_nm": inference.diagnostics["bandwidth_rms_nm"],
        "delta_bandwidth_rms_nm": float(inference.diagnostics["bandwidth_rms_nm"] - truth.diagnostics["bandwidth_rms_nm"]),
        "truth_flux_factor": float(truth.flux_factor),
        "inference_flux_factor": float(inference.flux_factor),
        "flux_factor_ratio_inference_over_truth": float(inference.flux_factor / truth.flux_factor),
        "truth_out_of_inference_band_fraction": float(0.0 if truth_out is None else truth_out),
        "shape_l1_common_grid": float(np.sum(np.abs(diff))),
        "shape_l2_common_grid": float(np.sqrt(np.sum(diff**2))),
        "shape_max_abs_common_grid": float(np.max(np.abs(diff))),
        "common_grid_n_lambda": int(n_common),
        "common_grid_min_nm": float(common_min),
        "common_grid_max_nm": float(common_max),
        "weight_projection": "linear_interpolation_then_sample_sum_normalization",
    }
    return metrics


def write_spectral_deck_artifacts(deck: SpectralDeck | SourceSpectralDeck, outdir: str | Path) -> dict[str, Path]:
    """Write CSV/JSON spectral deck artifacts under ``outdir``.

    The artifact contract is intentionally small and stable for v1:
    ``truth_weights.csv``, ``inference_weights.csv``, ``spectral_moments.json``,
    ``spectral_comparison.json``, and ``spectral_deck_manifest.json``.
    """

    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)

    def write_weights(path: Path, spectra: Mapping[str, EffectiveSpectrum]) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "wavelength_m",
                    "wavelength_nm",
                    "weight",
                    "raw_response",
                    "normalized_weight",
                    "in_band",
                    "component_label",
                ],
            )
            writer.writeheader()
            for component_label, spectrum in spectra.items():
                in_min = spectrum.diagnostics.get("in_band_min_nm", None)
                in_max = spectrum.diagnostics.get("in_band_max_nm", None)
                for wavelength_m, weight, raw in zip(spectrum.wavelengths_m, spectrum.weights, spectrum.raw_response):
                    wavelength_nm = float(wavelength_m * 1e9)
                    in_band = True if in_min is None or in_max is None else bool(in_min <= wavelength_nm <= in_max)
                    writer.writerow(
                        {
                            "wavelength_m": f"{float(wavelength_m):.17g}",
                            "wavelength_nm": f"{wavelength_nm:.17g}",
                            "weight": f"{float(weight):.17g}",
                            "raw_response": f"{float(raw):.17g}",
                            "normalized_weight": f"{float(weight):.17g}",
                            "in_band": str(in_band).lower(),
                            "component_label": component_label,
                        }
                    )

    truth_weights = root / "truth_weights.csv"
    inference_weights = root / "inference_weights.csv"
    moments = root / "spectral_moments.json"
    comparison = root / "spectral_comparison.json"
    manifest = root / "spectral_deck_manifest.json"

    if isinstance(deck, SourceSpectralDeck):
        truth_spectra = deck.truth_by_component
        inference_spectra = deck.inference_by_component
        truth_diag = {label: spec.diagnostics for label, spec in truth_spectra.items()}
        inference_diag = {label: spec.diagnostics for label, spec in inference_spectra.items()}
        comparison_payload_value = {
            "by_component": deck.comparison_by_component,
            "combined": deck.combined_comparison,
        }
        manifest_truth = {label: _spectrum_to_record(spec) for label, spec in truth_spectra.items()}
        manifest_inference = {label: _spectrum_to_record(spec) for label, spec in inference_spectra.items()}
    else:
        truth_spectra = {deck.truth.label: deck.truth}
        inference_spectra = {deck.inference.label: deck.inference}
        truth_diag = deck.truth.diagnostics
        inference_diag = deck.inference.diagnostics
        comparison_payload_value = deck.comparison
        manifest_truth = _spectrum_to_record(deck.truth)
        manifest_inference = _spectrum_to_record(deck.inference)

    write_weights(truth_weights, truth_spectra)
    write_weights(inference_weights, inference_spectra)

    generated_at = _now_iso()
    moments_payload = {
        "schema_version": deck.schema_version,
        "generated_at": generated_at,
        "truth": truth_diag,
        "inference": inference_diag,
        "note": "Spectral shape is not an active inference parameter in v1.",
    }
    comparison_payload = {
        "schema_version": deck.schema_version,
        "generated_at": generated_at,
        "comparison": comparison_payload_value,
    }
    manifest_payload = {
        "schema_version": deck.schema_version,
        "generated_at": generated_at,
        "artifacts": {
            "truth_weights_csv": truth_weights.name,
            "inference_weights_csv": inference_weights.name,
            "spectral_moments_json": moments.name,
            "spectral_comparison_json": comparison.name,
            "spectral_deck_manifest_json": manifest.name,
        },
        "provenance": _json_ready(deck.provenance),
        "truth": manifest_truth,
        "inference": manifest_inference,
        "note": "source.log_flux_total and contrast are interpreted as detected post-response band-integrated quantities.",
    }

    moments.write_text(json.dumps(_json_ready(moments_payload), indent=2, sort_keys=True) + "\n")
    comparison.write_text(json.dumps(_json_ready(comparison_payload), indent=2, sort_keys=True) + "\n")
    manifest.write_text(json.dumps(_json_ready(manifest_payload), indent=2, sort_keys=True) + "\n")
    return {
        "truth_weights": truth_weights,
        "inference_weights": inference_weights,
        "spectral_moments": moments,
        "spectral_comparison": comparison,
        "spectral_deck_manifest": manifest,
    }
