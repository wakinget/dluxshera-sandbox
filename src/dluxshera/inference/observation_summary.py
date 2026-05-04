from __future__ import annotations

import json
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dluxshera.inference.observation_belief import (
    MatrixDiagnostics,
    SchurReductionResult,
    SubblockSummary,
    schur_reduce_information,
)

__all__ = [
    "CombinedLocalParameterLayout",
    "ImageBackedSubblockSummaryArtifact",
    "LocalQuadraticBlocks",
    "SchurReducedLocalQuadratic",
    "build_combined_local_parameter_layout",
    "inspect_subblock_summary_artifact",
    "load_subblock_summary_artifact_payload",
    "load_subblock_summary",
    "partition_local_curvature",
    "schur_reduce_local_quadratic",
    "validate_subblock_summary_artifact",
]


def _as_label_tuple(labels: Sequence[str], *, name: str) -> tuple[str, ...]:
    values = tuple(str(label) for label in labels)
    if not values:
        raise ValueError(f"{name} must contain at least one label.")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate labels.")
    return values


def _as_vector(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _as_square_matrix(
    values: Sequence[Sequence[float]] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values.")
    return 0.5 * (matrix + matrix.T)


def _compute_matrix_diagnostics(
    matrix: np.ndarray,
    *,
    rcond: float | None = None,
) -> MatrixDiagnostics:
    matrix = _as_square_matrix(matrix, name="matrix")
    if matrix.size == 0:
        return MatrixDiagnostics(
            rank_estimate=0,
            min_eigenvalue=0.0,
            max_eigenvalue=0.0,
            condition_number=1.0,
            trace=0.0,
            frobenius_norm=0.0,
        )

    eigenvalues = np.linalg.eigvalsh(matrix)
    scale = np.finfo(float).eps * max(matrix.shape) if rcond is None else float(rcond)
    tolerance = scale * max(float(np.max(np.abs(eigenvalues))), 1.0)
    active = eigenvalues > tolerance
    positive = eigenvalues[active]
    if positive.size == 0:
        condition_number = float("inf")
    else:
        condition_number = float(np.max(positive) / np.min(positive))

    return MatrixDiagnostics(
        rank_estimate=int(np.count_nonzero(active)),
        min_eigenvalue=float(np.min(eigenvalues)),
        max_eigenvalue=float(np.max(eigenvalues)),
        condition_number=condition_number,
        trace=float(np.trace(matrix)),
        frobenius_norm=float(np.linalg.norm(matrix)),
    )


def _load_summary_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}.")
    return payload


def _resolve_matrix_artifact_path(
    summary_json_path: Path,
    payload: Mapping[str, Any],
) -> Path:
    matrix_artifact_value = payload.get("matrix_artifact_path")
    if not isinstance(matrix_artifact_value, str) or not matrix_artifact_value.strip():
        raise ValueError("Summary JSON must include a non-empty matrix_artifact_path.")
    matrix_path = Path(matrix_artifact_value)
    if not matrix_path.is_absolute():
        matrix_path = (summary_json_path.parent / matrix_path).resolve()
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Subblock summary matrix sidecar does not exist: {matrix_path}"
        )
    return matrix_path


def _top_entries(
    labels: Sequence[str],
    values: np.ndarray,
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    vector = _as_vector(values, name="values")
    if len(labels) != vector.size:
        raise ValueError("labels length must match values size.")
    order = np.argsort(np.abs(vector))[::-1][:top_k]
    return [
        {
            "label": str(labels[index]),
            "value": float(vector[index]),
            "abs_value": float(abs(vector[index])),
        }
        for index in order
    ]


@dataclass(frozen=True)
class CombinedLocalParameterLayout:
    """Describe the packed local parameter layout ``[Theta, phi]``."""

    theta_labels: tuple[str, ...]
    phi_labels: tuple[str, ...]
    combined_labels: tuple[str, ...]
    theta_slice: slice
    phi_slice: slice

    def __post_init__(self) -> None:
        theta_labels = _as_label_tuple(self.theta_labels, name="theta_labels")
        phi_labels = _as_label_tuple(self.phi_labels, name="phi_labels")
        combined_labels = _as_label_tuple(self.combined_labels, name="combined_labels")
        if combined_labels != theta_labels + phi_labels:
            raise ValueError("combined_labels must equal theta_labels + phi_labels.")
        if self.theta_slice != slice(0, len(theta_labels)):
            raise ValueError("theta_slice must cover the leading Theta block.")
        if self.phi_slice != slice(len(theta_labels), len(combined_labels)):
            raise ValueError("phi_slice must cover the trailing phi block.")
        object.__setattr__(self, "theta_labels", theta_labels)
        object.__setattr__(self, "phi_labels", phi_labels)
        object.__setattr__(self, "combined_labels", combined_labels)

    @property
    def n_theta(self) -> int:
        return len(self.theta_labels)

    @property
    def n_phi(self) -> int:
        return len(self.phi_labels)

    @property
    def size(self) -> int:
        return len(self.combined_labels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "theta_labels": list(self.theta_labels),
            "phi_labels": list(self.phi_labels),
            "combined_labels": list(self.combined_labels),
            "theta_slice": [self.theta_slice.start, self.theta_slice.stop],
            "phi_slice": [self.phi_slice.start, self.phi_slice.stop],
            "n_theta": int(self.n_theta),
            "n_phi": int(self.n_phi),
            "size": int(self.size),
        }


def build_combined_local_parameter_layout(
    theta_labels: Sequence[str],
    phi_labels: Sequence[str],
) -> CombinedLocalParameterLayout:
    """Return the deterministic packed layout for one local quadratic."""

    theta = _as_label_tuple(theta_labels, name="theta_labels")
    phi = _as_label_tuple(phi_labels, name="phi_labels")
    return CombinedLocalParameterLayout(
        theta_labels=theta,
        phi_labels=phi,
        combined_labels=theta + phi,
        theta_slice=slice(0, len(theta)),
        phi_slice=slice(len(theta), len(theta) + len(phi)),
    )


@dataclass(frozen=True)
class LocalQuadraticBlocks:
    """Partition one dense combined quadratic into Theta/phi blocks."""

    layout: CombinedLocalParameterLayout
    combined_gradient: np.ndarray
    combined_curvature: np.ndarray
    g_theta: np.ndarray
    g_phi: np.ndarray
    h_tt: np.ndarray
    h_tp: np.ndarray
    h_pp: np.ndarray

    def __post_init__(self) -> None:
        gradient = _as_vector(self.combined_gradient, name="combined_gradient")
        curvature = _as_square_matrix(self.combined_curvature, name="combined_curvature")
        if gradient.shape != (self.layout.size,):
            raise ValueError("combined_gradient shape must match combined layout size.")
        if curvature.shape != (self.layout.size, self.layout.size):
            raise ValueError("combined_curvature shape must match combined layout size.")
        object.__setattr__(self, "combined_gradient", gradient)
        object.__setattr__(self, "combined_curvature", curvature)
        object.__setattr__(self, "g_theta", _as_vector(self.g_theta, name="g_theta"))
        object.__setattr__(self, "g_phi", _as_vector(self.g_phi, name="g_phi"))
        object.__setattr__(self, "h_tt", _as_square_matrix(self.h_tt, name="h_tt"))
        object.__setattr__(self, "h_tp", np.asarray(self.h_tp, dtype=float))
        object.__setattr__(self, "h_pp", _as_square_matrix(self.h_pp, name="h_pp"))
        if self.g_theta.shape != (self.layout.n_theta,):
            raise ValueError("g_theta shape must match n_theta.")
        if self.g_phi.shape != (self.layout.n_phi,):
            raise ValueError("g_phi shape must match n_phi.")
        if self.h_tt.shape != (self.layout.n_theta, self.layout.n_theta):
            raise ValueError("h_tt shape must match n_theta.")
        if self.h_tp.shape != (self.layout.n_theta, self.layout.n_phi):
            raise ValueError("h_tp shape must match (n_theta, n_phi).")
        if self.h_pp.shape != (self.layout.n_phi, self.layout.n_phi):
            raise ValueError("h_pp shape must match n_phi.")


def partition_local_curvature(
    *,
    layout: CombinedLocalParameterLayout,
    combined_gradient: Sequence[float] | np.ndarray,
    combined_curvature: Sequence[Sequence[float]] | np.ndarray,
) -> LocalQuadraticBlocks:
    """Partition one dense combined gradient/Hessian into Theta/phi blocks."""

    gradient = _as_vector(combined_gradient, name="combined_gradient")
    curvature = _as_square_matrix(combined_curvature, name="combined_curvature")
    if gradient.shape != (layout.size,):
        raise ValueError("combined_gradient shape must match combined layout size.")
    if curvature.shape != (layout.size, layout.size):
        raise ValueError("combined_curvature shape must match combined layout size.")

    return LocalQuadraticBlocks(
        layout=layout,
        combined_gradient=gradient,
        combined_curvature=curvature,
        g_theta=gradient[layout.theta_slice],
        g_phi=gradient[layout.phi_slice],
        h_tt=curvature[np.ix_(range(layout.n_theta), range(layout.n_theta))],
        h_tp=curvature[np.ix_(range(layout.n_theta), range(layout.n_theta, layout.size))],
        h_pp=curvature[np.ix_(range(layout.n_theta, layout.size), range(layout.n_theta, layout.size))],
    )


@dataclass(frozen=True)
class SchurReducedLocalQuadratic:
    """Store the reduced local quadratic over observation-level Theta."""

    blocks: LocalQuadraticBlocks
    reduced_information: np.ndarray
    reduced_score: np.ndarray
    schur_result: SchurReductionResult
    h_pp_solve_method: str
    used_pseudoinverse: bool
    symmetry_residual: float
    psd_tolerance: float
    psd_within_tolerance: bool
    reduced_diagnostics: MatrixDiagnostics

    def to_diagnostics_dict(self) -> dict[str, Any]:
        return {
            "h_pp_solve_method": self.h_pp_solve_method,
            "used_pseudoinverse": bool(self.used_pseudoinverse),
            "symmetry_residual": float(self.symmetry_residual),
            "psd_tolerance": float(self.psd_tolerance),
            "psd_within_tolerance": bool(self.psd_within_tolerance),
            "n_theta": int(self.blocks.layout.n_theta),
            "n_phi": int(self.blocks.layout.n_phi),
            "h_tt": _compute_matrix_diagnostics(self.blocks.h_tt).to_dict(),
            "h_pp": self.schur_result.nuisance_diagnostics.to_dict(),
            "reduced_information": self.reduced_diagnostics.to_dict(),
            "schur": self.schur_result.to_dict(),
        }


def schur_reduce_local_quadratic(
    *,
    blocks: LocalQuadraticBlocks,
    damping: float = 0.0,
    rcond: float | None = None,
) -> SchurReducedLocalQuadratic:
    """Schur-reduce the local nuisance block from a dense quadratic."""

    schur_result = schur_reduce_information(
        blocks.h_tt,
        blocks.h_tp,
        blocks.h_pp,
        damping=damping,
        rcond=rcond,
    )

    if blocks.layout.n_phi == 0:
        solved_g_phi = np.zeros((0,), dtype=float)
        solve_method = "no_nuisance_block"
        used_pseudoinverse = False
    else:
        h_pp_damped = blocks.h_pp.copy()
        if damping > 0.0:
            h_pp_damped += float(damping) * np.eye(blocks.layout.n_phi, dtype=float)
        try:
            if schur_result.nuisance_diagnostics.rank_estimate < blocks.layout.n_phi:
                raise np.linalg.LinAlgError("nuisance block is numerically rank-deficient")
            solved_g_phi = np.linalg.solve(h_pp_damped, blocks.g_phi)
            solve_method = "solve"
            used_pseudoinverse = False
        except np.linalg.LinAlgError:
            pinv = np.linalg.pinv(
                h_pp_damped,
                rcond=np.finfo(float).eps if rcond is None else float(rcond),
                hermitian=True,
            )
            solved_g_phi = pinv @ blocks.g_phi
            solve_method = "pinv"
            used_pseudoinverse = True

    reduced_score = blocks.g_theta - blocks.h_tp @ solved_g_phi
    reduced_score = _as_vector(reduced_score, name="reduced_score")
    reduced_information = schur_result.reduced_information
    symmetry_residual = float(
        np.max(np.abs(reduced_information - reduced_information.T))
    )
    eigenvalues = np.linalg.eigvalsh(reduced_information)
    psd_tolerance = max(
        np.finfo(float).eps * max(reduced_information.shape),
        np.finfo(float).eps,
    ) * max(
        float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0,
        1.0,
    )
    reduced_diagnostics = _compute_matrix_diagnostics(reduced_information, rcond=rcond)
    return SchurReducedLocalQuadratic(
        blocks=blocks,
        reduced_information=reduced_information,
        reduced_score=reduced_score,
        schur_result=schur_result,
        h_pp_solve_method=solve_method,
        used_pseudoinverse=used_pseudoinverse or schur_result.used_pseudoinverse,
        symmetry_residual=symmetry_residual,
        psd_tolerance=psd_tolerance,
        psd_within_tolerance=bool(np.min(eigenvalues) >= -psd_tolerance),
        reduced_diagnostics=reduced_diagnostics,
    )


@dataclass(frozen=True)
class ImageBackedSubblockSummaryArtifact:
    """Bundle one reduced summary with its image-backed sidecar metadata."""

    summary: SubblockSummary
    layout: CombinedLocalParameterLayout
    theta_ref: np.ndarray
    phi_ref: np.ndarray
    reduced: SchurReducedLocalQuadratic
    metadata: dict[str, Any] = field(default_factory=dict)
    combined_gradient: np.ndarray | None = None
    combined_curvature: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "theta_ref", _as_vector(self.theta_ref, name="theta_ref"))
        object.__setattr__(self, "phi_ref", _as_vector(self.phi_ref, name="phi_ref"))
        if self.theta_ref.shape != (self.layout.n_theta,):
            raise ValueError("theta_ref shape must match n_theta.")
        if self.phi_ref.shape != (self.layout.n_phi,):
            raise ValueError("phi_ref shape must match n_phi.")
        if self.combined_gradient is not None:
            gradient = _as_vector(self.combined_gradient, name="combined_gradient")
            if gradient.shape != (self.layout.size,):
                raise ValueError("combined_gradient shape must match combined layout size.")
            object.__setattr__(self, "combined_gradient", gradient)
        if self.combined_curvature is not None:
            curvature = _as_square_matrix(self.combined_curvature, name="combined_curvature")
            if curvature.shape != (self.layout.size, self.layout.size):
                raise ValueError("combined_curvature shape must match combined layout size.")
            object.__setattr__(self, "combined_curvature", curvature)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def npz_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "theta_ref": self.theta_ref,
            "phi_ref": self.phi_ref,
            "reduced_information": self.summary.reduced_information,
            "reduced_score": self.summary.reduced_score,
            "h_tt": self.reduced.blocks.h_tt,
            "h_tp": self.reduced.blocks.h_tp,
            "h_pp": self.reduced.blocks.h_pp,
            "g_theta": self.reduced.blocks.g_theta,
            "g_phi": self.reduced.blocks.g_phi,
        }
        if self.combined_gradient is not None:
            payload["combined_gradient"] = self.combined_gradient
        if self.combined_curvature is not None:
            payload["combined_curvature"] = self.combined_curvature
        return payload

    def to_json_dict(
        self,
        *,
        matrix_artifact_path: str,
    ) -> dict[str, Any]:
        created_at = self.metadata.get(
            "created_at",
            datetime.now(timezone.utc).isoformat(),
        )
        generator = self.metadata.get("generator")
        payload = {
            "schema_version": "image_backed_subblock_summary.v1",
            "created_at": str(created_at),
            "generator": None if generator is None else str(generator),
            "subblock_id": self.summary.subblock_id,
            "summary_kind": self.summary.summary_kind,
            "case_root": self.metadata.get("case_root"),
            "cube_path": self.metadata.get("cube_path"),
            "manifest_path": self.metadata.get("manifest_path"),
            "truth_trace_path": self.metadata.get("truth_trace_path"),
            "config_path": self.metadata.get("config_path"),
            "objective": self.metadata.get("objective"),
            "system": self.metadata.get("system"),
            "prior_context": self.metadata.get("prior_context"),
            "recovered_reference": self.metadata.get("recovered_reference"),
            "theta_labels": list(self.summary.theta_labels),
            "phi_labels": list(self.layout.phi_labels),
            "combined_labels": list(self.layout.combined_labels),
            "theta_ref": self.theta_ref.tolist(),
            "phi_ref": self.phi_ref.tolist(),
            "dimensions": {
                "n_theta": int(self.layout.n_theta),
                "n_phi": int(self.layout.n_phi),
                "combined_dim": int(self.layout.size),
            },
            "diagnostics": self.reduced.to_diagnostics_dict(),
            "summary_diagnostics": dict(self.summary.diagnostics),
            "matrix_artifact_path": str(matrix_artifact_path),
            "metadata": dict(self.metadata),
        }
        return payload

    def write(
        self,
        *,
        summary_json_path: Path,
        matrix_npz_path: Path,
    ) -> None:
        summary_json_path.parent.mkdir(parents=True, exist_ok=True)
        matrix_npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(matrix_npz_path, **self.npz_payload())
        payload = self.to_json_dict(matrix_artifact_path=matrix_npz_path.name)
        with summary_json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


def load_subblock_summary(summary_json_path: Path | str) -> SubblockSummary:
    """Load a reduced summary artifact into the observation-belief contract."""

    path = Path(summary_json_path).resolve()
    payload = _load_summary_json(path)

    theta_labels = payload.get("theta_labels")
    theta_ref = payload.get("theta_ref")
    if not isinstance(theta_labels, list) or theta_ref is None:
        raise ValueError("Summary JSON must include theta_labels and theta_ref.")

    reduced_information = payload.get("reduced_information")
    reduced_score = payload.get("reduced_score")
    if reduced_information is None or reduced_score is None:
        matrix_path = _resolve_matrix_artifact_path(path, payload)
        with np.load(matrix_path) as arrays:
            reduced_information = np.asarray(arrays["reduced_information"], dtype=float)
            reduced_score = np.asarray(arrays["reduced_score"], dtype=float)

    return SubblockSummary.from_reduced_form(
        subblock_id=str(payload.get("subblock_id", path.stem)),
        theta_labels=tuple(str(label) for label in theta_labels),
        theta_ref=np.asarray(theta_ref, dtype=float),
        reduced_information=np.asarray(reduced_information, dtype=float),
        reduced_score=np.asarray(reduced_score, dtype=float),
        summary_kind=str(payload.get("summary_kind", "image_backed_schur")),
        diagnostics=payload.get("summary_diagnostics", payload.get("diagnostics", {})),
    )


def load_subblock_summary_artifact_payload(
    summary_json_path: Path | str,
) -> dict[str, Any]:
    """Load the raw JSON payload for one image-backed summary artifact."""

    return _load_summary_json(Path(summary_json_path).resolve())


def validate_subblock_summary_artifact(
    summary_json_path: Path | str,
) -> dict[str, Any]:
    """Validate one image-backed summary JSON/NPZ artifact pair."""

    path = Path(summary_json_path).resolve()
    payload = _load_summary_json(path)
    matrix_path = _resolve_matrix_artifact_path(path, payload)
    summary = load_subblock_summary(path)
    theta_labels = tuple(summary.theta_labels)
    phi_labels_raw = payload.get("phi_labels", [])
    if not isinstance(phi_labels_raw, list):
        raise ValueError("Summary JSON phi_labels must be a list when present.")
    phi_labels = tuple(str(label) for label in phi_labels_raw)
    combined_labels_raw = payload.get("combined_labels", [])
    if combined_labels_raw and not isinstance(combined_labels_raw, list):
        raise ValueError("Summary JSON combined_labels must be a list when present.")
    combined_labels = tuple(str(label) for label in combined_labels_raw)
    if combined_labels and len(combined_labels) != len(theta_labels) + len(phi_labels):
        raise ValueError("combined_labels length must match theta_labels + phi_labels.")

    with np.load(matrix_path) as arrays:
        required_keys = (
            "theta_ref",
            "phi_ref",
            "reduced_information",
            "reduced_score",
            "h_tt",
            "h_tp",
            "h_pp",
            "g_theta",
            "g_phi",
        )
        missing = [key for key in required_keys if key not in arrays]
        if missing:
            raise ValueError(
                "Subblock summary matrix sidecar is missing required arrays: "
                + ", ".join(sorted(missing))
            )
        theta_ref = _as_vector(np.asarray(arrays["theta_ref"], dtype=float), name="theta_ref")
        phi_ref = _as_vector(np.asarray(arrays["phi_ref"], dtype=float), name="phi_ref")
        reduced_information = _as_square_matrix(
            np.asarray(arrays["reduced_information"], dtype=float),
            name="reduced_information",
        )
        reduced_score = _as_vector(
            np.asarray(arrays["reduced_score"], dtype=float),
            name="reduced_score",
        )
        h_tt = _as_square_matrix(np.asarray(arrays["h_tt"], dtype=float), name="h_tt")
        h_tp = np.asarray(arrays["h_tp"], dtype=float)
        h_pp = _as_square_matrix(np.asarray(arrays["h_pp"], dtype=float), name="h_pp")
        g_theta = _as_vector(np.asarray(arrays["g_theta"], dtype=float), name="g_theta")
        g_phi = _as_vector(np.asarray(arrays["g_phi"], dtype=float), name="g_phi")

    n_theta = len(theta_labels)
    n_phi = len(phi_labels)
    if theta_ref.shape != (n_theta,):
        raise ValueError("theta_ref shape does not match theta_labels length.")
    if reduced_information.shape != (n_theta, n_theta):
        raise ValueError("reduced_information shape does not match theta_labels length.")
    if reduced_score.shape != (n_theta,):
        raise ValueError("reduced_score shape does not match theta_labels length.")
    if g_theta.shape != (n_theta,):
        raise ValueError("g_theta shape does not match theta_labels length.")
    if phi_ref.shape != (n_phi,):
        raise ValueError("phi_ref shape does not match phi_labels length.")
    if h_tt.shape != (n_theta, n_theta):
        raise ValueError("h_tt shape does not match theta_labels length.")
    if h_tp.shape != (n_theta, n_phi):
        raise ValueError("h_tp shape does not match theta/phi dimensions.")
    if h_pp.shape != (n_phi, n_phi):
        raise ValueError("h_pp shape does not match phi_labels length.")
    if g_phi.shape != (n_phi,):
        raise ValueError("g_phi shape does not match phi_labels length.")

    diag_payload = payload.get("diagnostics", {})
    summary_diag_payload = payload.get("summary_diagnostics", {})
    objective_payload = payload.get("objective")
    objective_kind = None
    variance_model = None
    if isinstance(objective_payload, Mapping):
        objective_kind = objective_payload.get("objective_kind_used")
        inference_objective = objective_payload.get("inference_objective")
        if isinstance(inference_objective, Mapping):
            variance_model = (
                inference_objective.get("noise_model", {}) or {}
            ).get("variance_model")

    reduced_diag = _compute_matrix_diagnostics(reduced_information)
    h_pp_diag = _compute_matrix_diagnostics(h_pp)
    top_score_entries = _top_entries(theta_labels, reduced_score)
    top_info_diagonal_entries = _top_entries(theta_labels, np.diag(reduced_information))
    return {
        "summary_json_path": str(path),
        "matrix_sidecar_path": str(matrix_path),
        "schema_version": payload.get("schema_version"),
        "subblock_id": summary.subblock_id,
        "theta_labels": list(theta_labels),
        "phi_labels": list(phi_labels),
        "combined_labels": list(combined_labels),
        "dimensions": {
            "n_theta": int(n_theta),
            "n_phi": int(n_phi),
            "combined_dim": int(n_theta + n_phi),
        },
        "provenance": {
            "generator": payload.get("generator"),
            "case_root": payload.get("case_root"),
            "cube_path": payload.get("cube_path"),
            "manifest_path": payload.get("manifest_path"),
            "config_path": payload.get("config_path"),
            "objective_kind": objective_kind,
            "variance_model": variance_model,
        },
        "schur": {
            "damping": summary_diag_payload.get("damping_value"),
            "h_pp_rank": int(h_pp_diag.rank_estimate),
            "h_pp_condition_number": float(h_pp_diag.condition_number),
            "h_pp_min_eigenvalue": float(h_pp_diag.min_eigenvalue),
            "h_pp_max_eigenvalue": float(h_pp_diag.max_eigenvalue),
            "reduced_information_rank": int(reduced_diag.rank_estimate),
            "reduced_information_condition_number": float(
                reduced_diag.condition_number
            ),
            "reduced_information_min_eigenvalue": float(reduced_diag.min_eigenvalue),
            "reduced_information_max_eigenvalue": float(reduced_diag.max_eigenvalue),
            "symmetry_residual": diag_payload.get("symmetry_residual"),
            "psd_within_tolerance": diag_payload.get("psd_within_tolerance"),
            "psd_tolerance": diag_payload.get("psd_tolerance"),
            "used_pseudoinverse": diag_payload.get("used_pseudoinverse"),
        },
        "reduced_score_norm": float(np.linalg.norm(reduced_score)),
        "top_reduced_score_entries": top_score_entries,
        "top_reduced_information_diagonal_entries": top_info_diagonal_entries,
    }


def inspect_subblock_summary_artifact(
    summary_json_path: Path | str,
) -> dict[str, Any]:
    """Return a compact inspection report for one image-backed summary artifact."""

    return validate_subblock_summary_artifact(summary_json_path)
