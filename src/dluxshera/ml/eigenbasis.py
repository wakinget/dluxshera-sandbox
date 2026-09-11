from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dluxshera.datasets.schema import json_ready, read_json, write_json

from .catalog import SampleCatalog

__all__ = [
    "SCIENCE_EIGENBASIS_SCHEMA_VERSION",
    "SCIENCE_FIM_SOURCE_SCHEMA_VERSION",
    "ScienceEigenbasis",
    "build_science_mode_weights",
    "build_science_eigenbasis",
    "build_science_eigenbasis_from_source",
    "build_s10_nominal_physical_fim_source",
    "load_science_eigenbasis",
    "science_eigenbasis_content_sha256",
    "science_fim_sanity_summary",
    "science_fim_source_content_sha256",
    "science_mode_weight_summary",
    "validate_science_eigenbasis_expectations",
    "validate_science_mode_weights_nonuniform",
    "write_science_eigenbasis",
]

SCIENCE_EIGENBASIS_SCHEMA_VERSION = "dluxshera_ml_science_eigenbasis/1"
SCIENCE_FIM_SOURCE_SCHEMA_VERSION = "dluxshera_ml_science_fim_source/1"
PHYSICAL_THETA_COORDINATE_SPACE = "physical_theta"
FISHER_SCALED_Z_COORDINATE_SPACE = "fisher_scaled_z"
PREPARED_FISHER_SCALED_SCIENCE_SPACE = "prepared_v4_fisher_scaled_science_delta"
DEFAULT_COORDINATE_CONVENTION = (
    "delta_z_science = z_B - z_A in prepared-catalog parameter order"
)
DEFAULT_NORMALIZATION = "orthonormal eigenvectors stored as columns, sorted by descending eigenvalue"
DEFAULT_FISHER_SCALE_RTOL = 1.0e-8
DEFAULT_FISHER_SCALE_ATOL = 1.0e-12
DEFAULT_PSD_NEGATIVE_EIGENVALUE_TOL = 1.0e-10
DEFAULT_UNIFORM_WEIGHT_RTOL = 1.0e-7
DEFAULT_UNIFORM_WEIGHT_ATOL = 1.0e-8


def _stable_sha256(payload: Any) -> str:
    raw = json.dumps(json_ready(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _stable_matrix(values: np.ndarray) -> list[list[float]]:
    return np.asarray(values, dtype=np.float64).tolist()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_info() -> dict[str, Any]:
    root = _repo_root()
    info: dict[str, Any] = {
        "source_commit": os.environ.get("DLUXSHERA_SOURCE_COMMIT")
        or os.environ.get("ML_SOURCE_COMMIT"),
        "source_archive_id": os.environ.get("DLUXSHERA_SOURCE_ARCHIVE_ID")
        or os.environ.get("ML_SOURCE_ARCHIVE_ID"),
    }
    for key, cmd in {
        "commit": ["git", "-C", str(root), "rev-parse", "HEAD"],
        "branch": ["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"],
        "dirty": ["git", "-C", str(root), "status", "--short"],
    }.items():
        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            info[key] = None
        else:
            info[key] = bool(result.stdout.strip()) if key == "dirty" else result.stdout.strip()
    return info


def build_science_mode_weights(
    eigenvalues: np.ndarray,
    *,
    mode: str,
    strength: float = 0.5,
    eigenvalue_floor: float = 1.0e-6,
    weight_cap: float = 10.0,
) -> np.ndarray:
    """Return normalized eigenmode weights with mean active weight one."""
    values = np.asarray(eigenvalues, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("eigenvalues must be a non-empty 1D array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("eigenvalues contain non-finite values.")
    active = np.maximum(values, float(eigenvalue_floor))
    if mode == "ordinary":
        weights = np.ones_like(active)
    elif mode == "strong_mode_weighted":
        ref = float(np.mean(active))
        weights = np.power(active / ref, float(strength))
    elif mode == "weak_mode_weighted":
        ref = float(np.mean(active))
        weights = np.power(ref / active, float(strength))
        weights = np.minimum(weights, float(weight_cap))
    else:
        raise ValueError(f"Unsupported science_loss.mode {mode!r}.")
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 1.0)
    return weights / float(np.mean(weights))


def validate_science_mode_weights_nonuniform(
    weights: Sequence[float],
    *,
    mode: str,
    rtol: float = DEFAULT_UNIFORM_WEIGHT_RTOL,
    atol: float = DEFAULT_UNIFORM_WEIGHT_ATOL,
) -> None:
    """Reject weighted objectives that are numerically equivalent to ordinary MSE."""
    arr = np.asarray(weights, dtype=np.float64)
    if mode == "ordinary":
        return
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f"science_loss.mode={mode!r} produced invalid mode weights.")
    if np.allclose(arr, np.ones_like(arr), rtol=rtol, atol=atol):
        raise ValueError(
            f"science_loss.mode={mode!r} produced effectively uniform mode weights; "
            "weighted S10 objectives must not collapse to ordinary MSE."
        )


def science_mode_weight_summary(
    eigenvalues: Sequence[float],
    *,
    mode: str,
    strength: float,
    eigenvalue_floor: float,
    weight_cap: float,
) -> dict[str, Any]:
    weights = build_science_mode_weights(
        np.asarray(eigenvalues, dtype=np.float64),
        mode=mode,
        strength=strength,
        eigenvalue_floor=eigenvalue_floor,
        weight_cap=weight_cap,
    )
    return {
        "mode": str(mode),
        "strength": float(strength),
        "eigenvalue_floor": float(eigenvalue_floor),
        "weight_cap": float(weight_cap),
        "min": float(np.min(weights)),
        "max": float(np.max(weights)),
        "std": float(np.std(weights)),
        "mean": float(np.mean(weights)),
        "uniform_allclose_to_one": bool(
            np.allclose(
                weights,
                np.ones_like(weights),
                rtol=DEFAULT_UNIFORM_WEIGHT_RTOL,
                atol=DEFAULT_UNIFORM_WEIGHT_ATOL,
            )
        ),
    }


def science_fim_sanity_summary(matrix: Sequence[Sequence[float]]) -> dict[str, Any]:
    arr = _as_square_matrix(matrix, dim=len(matrix), field="science FIM sanity matrix")
    eigenvalues = np.linalg.eigvalsh(arr)
    min_eig = float(np.min(eigenvalues)) if eigenvalues.size else 0.0
    max_eig = float(np.max(eigenvalues)) if eigenvalues.size else 0.0
    ratio = float(max_eig / min_eig) if min_eig > 0.0 else None
    diag = np.diag(arr)
    denom = np.sqrt(np.outer(diag, diag))
    corr = np.divide(arr, denom, out=np.zeros_like(arr), where=denom > 0.0)
    if corr.size:
        corr = corr.copy()
        np.fill_diagonal(corr, 0.0)
    return {
        "min_eigenvalue": min_eig,
        "max_eigenvalue": max_eig,
        "condition_number_or_eigenvalue_ratio": ratio,
        "max_off_diagonal_abs_correlation": float(np.max(np.abs(corr))) if corr.size else 0.0,
    }


@dataclass(frozen=True)
class ScienceEigenbasis:
    """Fixed science-coordinate eigenbasis used for ML loss weighting."""

    artifact_id: str
    source_matrix_coordinate_space: str
    eigenbasis_coordinate_space: str
    parameter_labels: tuple[str, ...]
    fisher_scales: np.ndarray
    coordinate_convention: str
    curvature_matrix: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    normalization: str
    physical_fim_identity: Mapping[str, Any]
    transformed_fz_identity: Mapping[str, Any]
    nominal_provenance: Mapping[str, Any]
    weighting_variance_convention: Mapping[str, Any]
    source_provenance: Mapping[str, Any]
    generated_at: str
    content_identity: Mapping[str, Any]

    def __post_init__(self) -> None:
        labels = tuple(str(v) for v in self.parameter_labels)
        object.__setattr__(self, "parameter_labels", labels)
        dim = len(labels)
        matrix = np.asarray(self.curvature_matrix, dtype=np.float64)
        values = np.asarray(self.eigenvalues, dtype=np.float64)
        vectors = np.asarray(self.eigenvectors, dtype=np.float64)
        fisher_scales = np.asarray(self.fisher_scales, dtype=np.float64)
        if matrix.shape != (dim, dim):
            raise ValueError(f"curvature_matrix shape {matrix.shape} does not match {dim} labels.")
        if fisher_scales.shape != (dim,):
            raise ValueError(f"fisher_scales shape {fisher_scales.shape} does not match {dim} labels.")
        if not np.all(np.isfinite(fisher_scales)) or np.any(fisher_scales <= 0.0):
            raise ValueError("fisher_scales must be finite and positive.")
        if values.shape != (dim,):
            raise ValueError(f"eigenvalues shape {values.shape} does not match {dim} labels.")
        if vectors.shape != (dim, dim):
            raise ValueError(f"eigenvectors shape {vectors.shape} does not match {dim} labels.")
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(values)) or not np.all(np.isfinite(vectors)):
            raise ValueError("Science eigenbasis contains non-finite values.")
        if not np.allclose(matrix, matrix.T, rtol=1.0e-8, atol=1.0e-10):
            raise ValueError("curvature_matrix must be symmetric.")
        if np.any(values[:-1] < values[1:] - 1.0e-10):
            raise ValueError("eigenvalues must be sorted in descending order.")
        if not np.allclose(vectors.T @ vectors, np.eye(dim), rtol=1.0e-6, atol=1.0e-6):
            raise ValueError("eigenvectors must be orthonormal columns.")
        if self.eigenbasis_coordinate_space != PREPARED_FISHER_SCALED_SCIENCE_SPACE:
            raise ValueError(
                "Science eigenbasis must be expressed in the prepared Fisher-scaled "
                f"science space, got {self.eigenbasis_coordinate_space!r}."
            )
        object.__setattr__(self, "curvature_matrix", matrix)
        object.__setattr__(self, "eigenvalues", values)
        object.__setattr__(self, "eigenvectors", vectors)
        object.__setattr__(self, "fisher_scales", fisher_scales)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCIENCE_EIGENBASIS_SCHEMA_VERSION,
            "artifact_id": self.artifact_id,
            "source_matrix_coordinate_space": self.source_matrix_coordinate_space,
            "eigenbasis_coordinate_space": self.eigenbasis_coordinate_space,
            "parameter_labels": list(self.parameter_labels),
            "fisher_scales": self.fisher_scales.astype(float).tolist(),
            "coordinate_convention": self.coordinate_convention,
            "curvature_matrix": _stable_matrix(self.curvature_matrix),
            "eigenvalues": self.eigenvalues.astype(float).tolist(),
            "eigenvectors": _stable_matrix(self.eigenvectors),
            "normalization": self.normalization,
            "physical_fim_identity": dict(self.physical_fim_identity),
            "transformed_fz_identity": dict(self.transformed_fz_identity),
            "nominal_provenance": dict(self.nominal_provenance),
            "weighting_variance_convention": dict(self.weighting_variance_convention),
            "source_provenance": dict(self.source_provenance),
            "generated_at": self.generated_at,
            "content_identity": dict(self.content_identity),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ScienceEigenbasis":
        if payload.get("schema_version") != SCIENCE_EIGENBASIS_SCHEMA_VERSION:
            raise ValueError(f"Unsupported science eigenbasis schema {payload.get('schema_version')!r}.")
        return cls(
            artifact_id=str(payload["artifact_id"]),
            source_matrix_coordinate_space=str(payload["source_matrix_coordinate_space"]),
            eigenbasis_coordinate_space=str(payload["eigenbasis_coordinate_space"]),
            parameter_labels=tuple(str(v) for v in payload["parameter_labels"]),
            fisher_scales=np.asarray(payload["fisher_scales"], dtype=np.float64),
            coordinate_convention=str(payload["coordinate_convention"]),
            curvature_matrix=np.asarray(payload["curvature_matrix"], dtype=np.float64),
            eigenvalues=np.asarray(payload["eigenvalues"], dtype=np.float64),
            eigenvectors=np.asarray(payload["eigenvectors"], dtype=np.float64),
            normalization=str(payload["normalization"]),
            physical_fim_identity=dict(payload.get("physical_fim_identity", {})),
            transformed_fz_identity=dict(payload.get("transformed_fz_identity", {})),
            nominal_provenance=dict(payload.get("nominal_provenance", {})),
            weighting_variance_convention=dict(payload.get("weighting_variance_convention", {})),
            source_provenance=dict(payload.get("source_provenance", {})),
            generated_at=str(payload.get("generated_at", "")),
            content_identity=dict(payload.get("content_identity", {})),
        )

    def validate_catalog(self, catalog: SampleCatalog) -> None:
        labels = tuple(str(v) for v in catalog.parameter_labels)
        if self.parameter_labels != labels:
            raise ValueError(
                "Science eigenbasis parameter_labels do not match prepared catalog "
                f"({self.parameter_labels} != {labels})."
            )
        if self.eigenbasis_coordinate_space != PREPARED_FISHER_SCALED_SCIENCE_SPACE:
            raise ValueError(
                "Science eigenbasis coordinate space must resolve to the prepared "
                f"Fisher-scaled science space, got {self.eigenbasis_coordinate_space!r}."
            )
        try:
            np.testing.assert_allclose(
                self.fisher_scales,
                np.asarray(catalog.fisher_sigmas, dtype=np.float64),
                rtol=DEFAULT_FISHER_SCALE_RTOL,
                atol=DEFAULT_FISHER_SCALE_ATOL,
                err_msg="Science eigenbasis fisher_scales do not match prepared V4 Fisher scales.",
            )
        except AssertionError as exc:
            raise ValueError(str(exc)) from exc


def science_eigenbasis_content_sha256(basis: ScienceEigenbasis | Mapping[str, Any]) -> str:
    payload = basis.to_dict() if isinstance(basis, ScienceEigenbasis) else dict(basis)
    stable = dict(payload)
    stable.pop("generated_at", None)
    stable.pop("content_identity", None)
    return _stable_sha256(stable)


def _orient_eigenvectors(vectors: np.ndarray) -> np.ndarray:
    out = np.asarray(vectors, dtype=np.float64).copy()
    for col in range(out.shape[1]):
        pivot = int(np.argmax(np.abs(out[:, col])))
        if out[pivot, col] < 0.0:
            out[:, col] *= -1.0
    return out


def science_fim_source_content_sha256(source: Mapping[str, Any]) -> str:
    stable = dict(source)
    stable.pop("generated_at", None)
    stable.pop("content_identity", None)
    return _stable_sha256(stable)


def _matrix_identity(matrix: np.ndarray, *, coordinate_space: str) -> dict[str, Any]:
    return {
        "algorithm": "sha256/json-canonical/matrix-v1",
        "coordinate_space": coordinate_space,
        "shape": list(np.asarray(matrix).shape),
        "sha256": _stable_sha256(_stable_matrix(matrix)),
    }


def _vector_space_identity(catalog: SampleCatalog) -> dict[str, Any]:
    vector_spaces = dict(catalog.vector_spaces)
    transform = dict(vector_spaces.get("transforms", {}).get("fisher_diagonal_scale", {}) or {})
    science_identity = vector_spaces.get("science_vector_space_identity")
    return {
        "prepared_dataset": {
            "artifact_id": catalog.artifact_id,
            "prepared_dataset_hash": catalog.prepared_dataset_hash,
        },
        "science_vector_space_id": vector_spaces.get("science_vector_space_id")
        or catalog.manifest.get("source_dataset", {}).get("science_vector_space_id"),
        "fisher_diagonal_scale": {
            "type": transform.get("type"),
            "source_space": transform.get("source_space"),
            "destination_space": transform.get("destination_space"),
            "forward_mode": transform.get("forward_mode"),
            "scale_source": transform.get("scale_source"),
            "scales_sha256": _stable_sha256(
                [float(v) for v in np.asarray(catalog.fisher_sigmas, dtype=np.float64)]
            ),
        },
        "science_vector_space_identity_hash": None
        if science_identity is None
        else _stable_sha256(science_identity),
    }


def _validate_psd(matrix: np.ndarray, *, name: str) -> None:
    values = np.linalg.eigvalsh(np.asarray(matrix, dtype=np.float64))
    max_abs = max(float(np.max(np.abs(values))) if values.size else 0.0, 1.0)
    lower_bound = -DEFAULT_PSD_NEGATIVE_EIGENVALUE_TOL * max_abs
    if float(np.min(values)) < lower_bound:
        raise ValueError(
            f"{name} is materially indefinite: minimum eigenvalue "
            f"{float(np.min(values)):.6g} is below tolerance {lower_bound:.6g}."
        )


def _as_square_matrix(value: Any, *, dim: int, field: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (dim, dim):
        raise ValueError(f"{field} shape {matrix.shape} does not match ({dim}, {dim}).")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{field} contains non-finite values.")
    if not np.allclose(matrix, matrix.T, rtol=1.0e-8, atol=1.0e-10):
        raise ValueError(f"{field} must be symmetric.")
    return 0.5 * (matrix + matrix.T)


def _validate_source_labels(
    source: Mapping[str, Any],
    catalog: SampleCatalog,
) -> tuple[str, ...]:
    labels = tuple(str(v) for v in source.get("parameter_labels", ()))
    if not labels:
        raise ValueError("Science FIM source must define parameter_labels.")
    expected = tuple(str(v) for v in catalog.parameter_labels)
    if labels != expected:
        raise ValueError(
            "Science FIM source parameter_labels must match prepared catalog ordering "
            f"({labels} != {expected})."
        )
    return labels


def _diagonal_fisher_sigmas(f_theta: np.ndarray) -> np.ndarray:
    diag = np.diag(np.asarray(f_theta, dtype=np.float64))
    if np.any(diag <= 0.0):
        raise ValueError("Physical FIM diagonal must be positive to derive Fisher sigmas.")
    return 1.0 / np.sqrt(diag)


def _validate_fisher_scales(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    rtol: float = DEFAULT_FISHER_SCALE_RTOL,
    atol: float = DEFAULT_FISHER_SCALE_ATOL,
) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    abs_diff = np.abs(actual - expected)
    rel_diff = abs_diff / np.maximum(np.abs(expected), np.finfo(np.float64).tiny)
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        worst = int(np.argmax(abs_diff))
        raise ValueError(
            "Physical nominal FIM diagonal-derived Fisher sigmas do not match "
            "authoritative prepared V4 Fisher scales "
            f"(worst index {worst}: {actual[worst]} != {expected[worst]}, "
            f"rtol={rtol}, atol={atol})."
        )
    return {
        "rtol": float(rtol),
        "atol": float(atol),
        "max_abs_diff": float(np.max(abs_diff)) if abs_diff.size else 0.0,
        "max_rel_diff": float(np.max(rel_diff)) if rel_diff.size else 0.0,
        "status": "PASS",
    }


def _compute_s10_nominal_full_physical_fim(catalog: SampleCatalog) -> dict[str, Any]:
    from work.experiments.generate_training_dataset_v3 import (
        compute_s10_nominal_science_fim_source_inputs,
    )

    return compute_s10_nominal_science_fim_source_inputs(
        catalog_labels=catalog.parameter_labels,
    )


def _resolve_source_fim(
    *,
    source: Mapping[str, Any],
    catalog: SampleCatalog,
) -> tuple[str, np.ndarray | None, np.ndarray, dict[str, Any]]:
    labels = _validate_source_labels(source, catalog)
    coordinate_space = source.get("coordinate_space", source.get("source_matrix_coordinate_space"))
    if coordinate_space in (None, ""):
        raise ValueError("Science FIM source must declare coordinate_space.")
    coordinate_space = str(coordinate_space)
    dim = len(labels)
    matrix_value = source.get("curvature_matrix", source.get("fim"))
    if matrix_value is None:
        raise ValueError("Science FIM source must define curvature_matrix or fim.")
    matrix = _as_square_matrix(matrix_value, dim=dim, field="source FIM")
    fisher_scales = np.asarray(catalog.fisher_sigmas, dtype=np.float64)
    scale_identity = _vector_space_identity(catalog)
    diagnostics: dict[str, Any] = {"prepared_fisher_scale_identity": scale_identity}
    if coordinate_space == PHYSICAL_THETA_COORDINATE_SPACE:
        _validate_psd(matrix, name="physical nominal FIM")
        derived_sigmas = _diagonal_fisher_sigmas(matrix)
        diagnostics["fisher_scale_compatibility"] = _validate_fisher_scales(
            derived_sigmas,
            fisher_scales,
        )
        diagnostics["diagonal_derived_fisher_scales"] = derived_sigmas.astype(float).tolist()
        d = np.diag(fisher_scales)
        f_z = d @ matrix @ d
        _validate_psd(f_z, name="Fisher-scaled nominal FIM")
        return coordinate_space, matrix, 0.5 * (f_z + f_z.T), diagnostics
    if coordinate_space == FISHER_SCALED_Z_COORDINATE_SPACE:
        declared = source.get("fisher_scale_identity", source.get("prepared_fisher_scale_identity"))
        if not isinstance(declared, Mapping):
            raise ValueError(
                "Fisher-scaled-z FIM sources must declare fisher_scale_identity."
            )
        declared_labels = tuple(str(v) for v in declared.get("parameter_labels", labels))
        if declared_labels != labels:
            raise ValueError("Fisher-scaled-z FIM source scaling identity labels do not match.")
        declared_scales = np.asarray(declared.get("fisher_scales", []), dtype=np.float64)
        if declared_scales.shape != fisher_scales.shape:
            raise ValueError("Fisher-scaled-z FIM source scaling identity has wrong scale count.")
        diagnostics["fisher_scale_compatibility"] = _validate_fisher_scales(
            declared_scales,
            fisher_scales,
        )
        _validate_psd(matrix, name="Fisher-scaled nominal FIM")
        return coordinate_space, None, matrix, diagnostics
    raise ValueError(f"Unsupported science FIM coordinate_space {coordinate_space!r}.")


def build_science_eigenbasis_from_source(
    *,
    artifact_id: str,
    source: Mapping[str, Any],
    catalog: SampleCatalog,
    coordinate_convention: str = DEFAULT_COORDINATE_CONVENTION,
    normalization: str = DEFAULT_NORMALIZATION,
    source_provenance: Mapping[str, Any] | None = None,
) -> ScienceEigenbasis:
    """Build an S10-v1 eigenbasis from a coordinate-declared science FIM source."""
    coordinate_space, f_theta, f_z, diagnostics = _resolve_source_fim(
        source=source,
        catalog=catalog,
    )
    provenance = dict(source.get("source_provenance", {}) or {})
    provenance.update(dict(source_provenance or {}))
    provenance["source_artifact_id"] = source.get("artifact_id")
    if "content_identity" in source:
        provenance["source_content_identity"] = dict(source["content_identity"])
    provenance["coordinate_validation"] = diagnostics
    physical_identity = (
        _matrix_identity(f_theta, coordinate_space=PHYSICAL_THETA_COORDINATE_SPACE)
        if f_theta is not None
        else dict(source.get("physical_fim_identity", {}))
    )
    return build_science_eigenbasis(
        artifact_id=artifact_id,
        parameter_labels=tuple(str(v) for v in source["parameter_labels"]),
        curvature_matrix=f_z,
        source_matrix_coordinate_space=coordinate_space,
        eigenbasis_coordinate_space=PREPARED_FISHER_SCALED_SCIENCE_SPACE,
        fisher_scales=np.asarray(catalog.fisher_sigmas, dtype=np.float64),
        physical_fim_identity=physical_identity,
        transformed_fz_identity=_matrix_identity(
            f_z,
            coordinate_space=PREPARED_FISHER_SCALED_SCIENCE_SPACE,
        ),
        nominal_provenance=dict(source.get("nominal_provenance", {})),
        weighting_variance_convention=dict(source.get("weighting_variance_convention", {})),
        coordinate_convention=coordinate_convention,
        normalization=normalization,
        source_provenance=provenance,
    )


def validate_science_eigenbasis_expectations(
    basis: ScienceEigenbasis,
    *,
    expected: Mapping[str, Any] | None,
) -> None:
    """Validate a supplied eigenbasis against study-declared identity fields."""
    if not expected:
        return
    checks = {
        "artifact_id": basis.artifact_id,
        "source_matrix_coordinate_space": basis.source_matrix_coordinate_space,
        "eigenbasis_coordinate_space": basis.eigenbasis_coordinate_space,
        "coordinate_convention": basis.coordinate_convention,
    }
    for key, actual in checks.items():
        wanted = expected.get(key)
        if wanted not in (None, "") and str(wanted) != str(actual):
            raise ValueError(
                f"Science eigenbasis {key} {actual!r} does not match declared {wanted!r}."
            )
    schema = expected.get("schema_version")
    if schema not in (None, "") and str(schema) != SCIENCE_EIGENBASIS_SCHEMA_VERSION:
        raise ValueError(
            f"Science eigenbasis schema declaration {schema!r} does not match "
            f"{SCIENCE_EIGENBASIS_SCHEMA_VERSION!r}."
        )
    expected_source_id = expected.get("source_fim_artifact_id")
    if expected_source_id not in (None, ""):
        declared_source = basis.source_provenance.get("source_artifact_id")
        if declared_source in (None, ""):
            raise ValueError("Science eigenbasis is missing source_artifact_id provenance.")
        if str(declared_source) != str(expected_source_id):
            raise ValueError(
                f"Science eigenbasis source FIM artifact {declared_source!r} does not match "
                f"declared {expected_source_id!r}."
            )


def build_s10_nominal_physical_fim_source(
    *,
    catalog: SampleCatalog,
    artifact_id: str = "s10_science_fim_source",
) -> dict[str, Any]:
    """Return the S10-v1 nominal physical-theta science FIM source payload."""
    labels = tuple(str(v) for v in catalog.parameter_labels)
    fisher_scales = np.asarray(catalog.fisher_sigmas, dtype=np.float64)
    computed = _compute_s10_nominal_full_physical_fim(catalog)
    computed_labels = tuple(str(v) for v in computed.get("parameter_labels", ()))
    if computed_labels != labels:
        raise ValueError(
            "S10 nominal FIM source labels do not match prepared catalog labels "
            f"({computed_labels} != {labels})."
        )
    f_theta = _as_square_matrix(
        computed["fim_theta"],
        dim=len(labels),
        field="S10 nominal physical FIM",
    )
    _validate_psd(f_theta, name="S10 nominal physical FIM")
    derived = _diagonal_fisher_sigmas(f_theta)
    compatibility = _validate_fisher_scales(derived, fisher_scales)
    d = np.diag(fisher_scales)
    f_z = 0.5 * (d @ f_theta @ d + (d @ f_theta @ d).T)
    _validate_psd(f_z, name="S10 transformed Fisher-scaled FIM")
    transformed_sanity = science_fim_sanity_summary(f_z)
    payload: dict[str, Any] = {
        "schema_version": SCIENCE_FIM_SOURCE_SCHEMA_VERSION,
        "artifact_id": str(artifact_id),
        "coordinate_space": PHYSICAL_THETA_COORDINATE_SPACE,
        "parameter_labels": list(labels),
        "curvature_matrix": _stable_matrix(f_theta),
        "fisher_scales": fisher_scales.astype(float).tolist(),
        "diagonal_derived_fisher_scales": derived.astype(float).tolist(),
        "physical_fim_identity": _matrix_identity(
            f_theta,
            coordinate_space=PHYSICAL_THETA_COORDINATE_SPACE,
        ),
        "prepared_fisher_scale_identity": {
            **_vector_space_identity(catalog),
            "parameter_labels": list(labels),
            "fisher_scales": fisher_scales.astype(float).tolist(),
        },
        "diagonal_fisher_scale_compatibility": compatibility,
        "nominal_provenance": {
            "prepared_dataset_artifact_id": catalog.artifact_id,
            "prepared_dataset_hash": catalog.prepared_dataset_hash,
            "science_dim": catalog.science_dim,
            "science_target": "delta_z_science = z_B - z_A",
            "nuisance_policy": "registration nuisance parameters held fixed for S10-v1",
            "system_preset": (
                computed.get("system_cfg", {}).get("preset")
                if isinstance(computed.get("system_cfg"), Mapping)
                else None
            ),
            "nominal_parameter_labels": list(labels),
            "theta_ref": np.asarray(computed.get("theta_ref", []), dtype=np.float64).astype(float).tolist(),
            "index_map": computed.get("index_map", {}),
            "image_shape": list(computed.get("image_shape", [])),
        },
        "weighting_variance_convention": {
            "fim_diagonal_definition": "sigma_Fisher_i = 1 / sqrt(F_theta[ii,ii])",
            "sigma_source": "PREP-V4-v1 vector_spaces fisher_diagonal_scale.scales",
            "variance_model": "nominal binder image used as Gaussian per-pixel variance",
            "loss_convention": computed.get("loss_convention", {}),
            "basis_scope": "20-dimensional science block only; no nuisance Schur marginalization",
        },
        "source_provenance": {
            "source_implementation": computed.get("source_implementation"),
            "script_version": computed.get("script_version"),
            "prescription_path": computed.get("prescription_path"),
            "prescription_sha256": computed.get("prescription_sha256"),
            "git_info": computed.get("git_info") or _git_info(),
            "runtime_git_info": _git_info(),
            "environment_commit": os.environ.get("DLUXSHERA_SOURCE_COMMIT")
            or os.environ.get("ML_SOURCE_COMMIT"),
            "fim_display_labels": computed.get("fim_display_labels", []),
            "sweep_keys": computed.get("sweep_keys", []),
        },
        "diagnostics": {
            "physical_fim": science_fim_sanity_summary(f_theta),
            "transformed_fz": transformed_sanity,
        },
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    payload["transformed_fz_identity"] = _matrix_identity(
        f_z,
        coordinate_space=PREPARED_FISHER_SCALED_SCIENCE_SPACE,
    )
    payload["content_identity"] = {
        "algorithm": "sha256/json-canonical/science-fim-source-v1",
        "sha256": science_fim_source_content_sha256(payload),
        "excludes": ["generated_at", "content_identity"],
    }
    return payload


def build_science_eigenbasis(
    *,
    artifact_id: str,
    parameter_labels: Sequence[str],
    curvature_matrix: Sequence[Sequence[float]],
    source_matrix_coordinate_space: str = FISHER_SCALED_Z_COORDINATE_SPACE,
    eigenbasis_coordinate_space: str = PREPARED_FISHER_SCALED_SCIENCE_SPACE,
    fisher_scales: Sequence[float] | None = None,
    physical_fim_identity: Mapping[str, Any] | None = None,
    transformed_fz_identity: Mapping[str, Any] | None = None,
    nominal_provenance: Mapping[str, Any] | None = None,
    weighting_variance_convention: Mapping[str, Any] | None = None,
    coordinate_convention: str = DEFAULT_COORDINATE_CONVENTION,
    normalization: str = DEFAULT_NORMALIZATION,
    source_provenance: Mapping[str, Any] | None = None,
) -> ScienceEigenbasis:
    """Build a deterministic fixed eigenbasis from a Fisher-scaled-z FIM."""
    labels = tuple(str(v) for v in parameter_labels)
    matrix = np.asarray(curvature_matrix, dtype=np.float64)
    if matrix.shape != (len(labels), len(labels)):
        raise ValueError(f"curvature_matrix shape {matrix.shape} does not match {len(labels)} labels.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("curvature_matrix contains non-finite values.")
    matrix = 0.5 * (matrix + matrix.T)
    _validate_psd(matrix, name="science eigenbasis curvature_matrix")
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    psd_floor = -DEFAULT_PSD_NEGATIVE_EIGENVALUE_TOL * max(
        float(np.max(np.abs(eigenvalues))),
        1.0,
    )
    eigenvalues = np.where((eigenvalues < 0.0) & (eigenvalues >= psd_floor), 0.0, eigenvalues)
    eigenvectors = _orient_eigenvectors(eigenvectors[:, order])
    scales = (
        np.ones((len(labels),), dtype=np.float64)
        if fisher_scales is None
        else np.asarray(fisher_scales, dtype=np.float64)
    )
    basis = ScienceEigenbasis(
        artifact_id=str(artifact_id),
        source_matrix_coordinate_space=str(source_matrix_coordinate_space),
        eigenbasis_coordinate_space=str(eigenbasis_coordinate_space),
        parameter_labels=labels,
        fisher_scales=scales,
        coordinate_convention=str(coordinate_convention),
        curvature_matrix=matrix,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        normalization=str(normalization),
        physical_fim_identity=dict(physical_fim_identity or {}),
        transformed_fz_identity=dict(
            transformed_fz_identity
            or _matrix_identity(matrix, coordinate_space=str(eigenbasis_coordinate_space))
        ),
        nominal_provenance=dict(nominal_provenance or {}),
        weighting_variance_convention=dict(weighting_variance_convention or {}),
        source_provenance=dict(source_provenance or {}),
        generated_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        content_identity={},
    )
    payload = basis.to_dict()
    payload["content_identity"] = {
        "algorithm": "sha256/json-canonical/science-eigenbasis-v1",
        "sha256": science_eigenbasis_content_sha256(payload),
        "excludes": ["generated_at", "content_identity"],
    }
    return ScienceEigenbasis.from_dict(payload)


def write_science_eigenbasis(path: Path, basis: ScienceEigenbasis, *, overwrite: bool = False) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it.")
    write_json(path, basis.to_dict())


def load_science_eigenbasis(path: Path, *, catalog: SampleCatalog | None = None) -> ScienceEigenbasis:
    payload = read_json(Path(path))
    basis = ScienceEigenbasis.from_dict(payload)
    identity = payload.get("content_identity", {})
    if isinstance(identity, Mapping) and identity.get("sha256"):
        actual = science_eigenbasis_content_sha256(basis)
        if str(identity["sha256"]) != actual:
            raise ValueError(
                "Science eigenbasis content_identity.sha256 does not match content "
                f"({identity['sha256']} != {actual})."
            )
    if catalog is not None:
        basis.validate_catalog(catalog)
    return basis
