from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "accumulate_summary_information",
    "build_system_observation_theta_layout",
    "build_prior_whitened_information_gain_matrix",
    "MatrixDiagnostics",
    "ObservationBeliefState",
    "ObservationEigenBasis",
    "ObservationThetaLayout",
    "ObservationUpdateResult",
    "ObservationUpdateStep",
    "SchurReductionResult",
    "SubblockSummary",
    "build_observation_eigenbasis",
    "infer_indexed_parameter_indices",
    "infer_system_zernike_indices",
    "schur_reduce_information",
    "update_observation_belief",
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


def _as_positive_vector(
    values: Sequence[float] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    vector = _as_vector(values, name=name)
    if np.any(vector <= 0.0):
        raise ValueError(f"{name} values must be strictly positive.")
    return vector


def _normalize_index_selection(
    available: Sequence[int],
    *,
    include: Sequence[int] | None = None,
    exclude: Sequence[int] | None = None,
    name: str,
) -> tuple[int, ...]:
    available_tuple = tuple(int(index) for index in available)
    if len(set(available_tuple)) != len(available_tuple):
        raise ValueError(f"{name} available indices contain duplicates.")
    available_set = set(available_tuple)

    if include is None:
        selected = list(available_tuple)
    else:
        include_tuple = tuple(int(index) for index in include)
        if len(set(include_tuple)) != len(include_tuple):
            raise ValueError(f"{name}.include contains duplicates.")
        invalid = sorted(set(include_tuple) - available_set)
        if invalid:
            raise ValueError(
                f"{name}.include contains indices outside the resolved system: "
                + ", ".join(str(index) for index in invalid)
            )
        selected = [index for index in available_tuple if index in set(include_tuple)]

    exclude_tuple = tuple(int(index) for index in (exclude or ()))
    if len(set(exclude_tuple)) != len(exclude_tuple):
        raise ValueError(f"{name}.exclude contains duplicates.")
    invalid_exclude = sorted(set(exclude_tuple) - available_set)
    if invalid_exclude:
        raise ValueError(
            f"{name}.exclude contains indices outside the resolved system: "
            + ", ".join(str(index) for index in invalid_exclude)
        )
    exclude_set = set(exclude_tuple)
    return tuple(index for index in selected if index not in exclude_set)


def _store_get_optional(store: Any, key: str) -> Any:
    try:
        return store.get(key)
    except (AttributeError, KeyError):
        return None


def infer_indexed_parameter_indices(store: Any, key: str) -> tuple[int, ...]:
    """Infer valid zero-based indices for a vector-valued store parameter.

    Parameters
    ----------
    store :
        Resolved parameter store or store-like object exposing ``get(key)``.
    key :
        Vector-valued parameter key.

    Returns
    -------
    tuple of int
        ``range(len(store[key]))`` for a one-dimensional resolved value.
    """

    value = _store_get_optional(store, str(key))
    if value is None:
        return ()
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"Store key {key!r} must resolve to a 1D vector.")
    return tuple(range(int(array.shape[0])))


def infer_system_zernike_indices(store: Any, *, optic: str) -> tuple[int, ...]:
    """Infer the coefficient indices for one system-defined Zernike family."""

    if optic not in {"primary", "secondary"}:
        raise ValueError("optic must be either 'primary' or 'secondary'.")
    return infer_indexed_parameter_indices(
        store,
        f"optics.{optic}.zernike_coeffs_nm",
    )


def _infer_system_zernike_noll_indices(store: Any, *, optic: str) -> tuple[int, ...] | None:
    value = _store_get_optional(store, f"optics.{optic}_noll_indices")
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim != 1:
        return None
    return tuple(int(item) for item in array.tolist())


def _zernike_group_indices_from_config(
    store: Any,
    *,
    optic: str,
    group_config: Mapping[str, Any],
) -> tuple[int, ...]:
    enabled = bool(group_config.get("enabled", False))
    if not enabled:
        return ()

    raw_policy = group_config.get("indices", "from_system")
    system_available = infer_system_zernike_indices(store, optic=optic)
    if raw_policy is None or raw_policy == "from_system":
        available = system_available
    elif isinstance(raw_policy, str):
        tokens = [token.strip() for token in raw_policy.split(",") if token.strip()]
        available = tuple(int(token) for token in tokens)
    else:
        available = tuple(int(index) for index in raw_policy)
    if system_available:
        invalid_available = sorted(set(available) - set(system_available))
        if invalid_available:
            raise ValueError(
                f"observation_theta.optics.{optic}_zernikes.indices contains "
                "indices outside the resolved system: "
                + ", ".join(str(index) for index in invalid_available)
            )

    include_raw = group_config.get("include")
    include = None if include_raw is None else tuple(int(index) for index in include_raw)
    exclude = tuple(int(index) for index in group_config.get("exclude", ()) or ())
    return _normalize_index_selection(
        available,
        include=include,
        exclude=exclude,
        name=f"observation_theta.optics.{optic}_zernikes",
    )


def build_system_observation_theta_layout(
    store: Any,
    *,
    config: Mapping[str, Any] | None = None,
) -> tuple[ObservationThetaLayout, dict[str, Any]]:
    """Build an observation layout from a resolved system/store.

    The default layout includes the canonical source scalars, plate scale, and
    every primary/secondary Zernike coefficient present in the resolved store.
    Optional config masks can include/exclude source groups, plate scale, and
    individual Zernike coefficient indices while keeping the native basis in
    physical parameter labels.

    Parameters
    ----------
    store :
        Resolved parameter store or store-like object.
    config :
        Optional ``observation_theta`` mapping.

    Returns
    -------
    layout, metadata :
        The physical-basis layout plus JSON-friendly Zernike provenance.
    """

    theta_cfg = dict(config or {})
    source_cfg = theta_cfg.get("source", {})
    optics_cfg = theta_cfg.get("optics", {})
    if source_cfg is None:
        source_cfg = {}
    if optics_cfg is None:
        optics_cfg = {}
    if not isinstance(source_cfg, Mapping):
        raise ValueError("observation_theta.source must be a mapping.")
    if not isinstance(optics_cfg, Mapping):
        raise ValueError("observation_theta.optics must be a mapping.")

    primary_cfg = optics_cfg.get(
        "primary_zernikes",
        {"enabled": True, "indices": "from_system", "include": None, "exclude": []},
    )
    secondary_cfg = optics_cfg.get(
        "secondary_zernikes",
        {"enabled": True, "indices": "from_system", "include": None, "exclude": []},
    )
    if not isinstance(primary_cfg, Mapping):
        raise ValueError("observation_theta.optics.primary_zernikes must be a mapping.")
    if not isinstance(secondary_cfg, Mapping):
        raise ValueError(
            "observation_theta.optics.secondary_zernikes must be a mapping."
        )

    primary_indices = _zernike_group_indices_from_config(
        store,
        optic="primary",
        group_config=primary_cfg,
    )
    secondary_indices = _zernike_group_indices_from_config(
        store,
        optic="secondary",
        group_config=secondary_cfg,
    )

    layout = ObservationThetaLayout.from_config(
        {
            "theta_layout": {
                "source": {
                    "separation_as": bool(source_cfg.get("separation_as", True)),
                    "log_flux_total": bool(source_cfg.get("log_flux_total", True)),
                    "contrast": bool(source_cfg.get("contrast", True)),
                },
                "optics": {
                    "plate_scale_as_per_pix": bool(
                        optics_cfg.get("plate_scale_as_per_pix", True)
                    ),
                    "primary_zernikes": {
                        "enabled": bool(primary_indices),
                        "indices": list(primary_indices),
                    },
                    "secondary_zernikes": {
                        "enabled": bool(secondary_indices),
                        "indices": list(secondary_indices),
                    },
                },
            }
        }
    )
    primary_noll = _infer_system_zernike_noll_indices(store, optic="primary")
    secondary_noll = _infer_system_zernike_noll_indices(store, optic="secondary")
    metadata: dict[str, Any] = {
        "zernike_index_source": "resolved_store",
        "primary_zernike_indices": list(primary_indices),
        "secondary_zernike_indices": list(secondary_indices),
        "theta_layout": layout.to_dict(),
    }
    if primary_noll is not None:
        metadata["primary_zernike_noll_indices"] = list(primary_noll)
    if secondary_noll is not None:
        metadata["secondary_zernike_noll_indices"] = list(secondary_noll)
    return layout, metadata


def _matrix_rank_tolerance(matrix: np.ndarray, *, rcond: float | None) -> float:
    if matrix.size == 0:
        return 0.0
    if rcond is None:
        scale = np.finfo(float).eps * max(matrix.shape)
    else:
        scale = float(rcond)
    spectrum = np.linalg.eigvalsh(matrix)
    max_eig = float(np.max(np.abs(spectrum))) if spectrum.size else 0.0
    return scale * max(max_eig, 1.0)


@dataclass(frozen=True)
class MatrixDiagnostics:
    """Summarize the spectrum and effective rank of one symmetric matrix.

    Use this for JSON-friendly health checks on reduced Fisher blocks and
    accumulated precision matrices. The diagnostics intentionally stay small and
    stable so they can be attached to both synthetic summaries and future
    image-backed Schur products.

    Parameters
    ----------
    rank_estimate :
        Numerical rank estimated from the eigenspectrum under the supplied or
        default rank tolerance.
    min_eigenvalue :
        Smallest eigenvalue of the symmetrized matrix.
    max_eigenvalue :
        Largest eigenvalue of the symmetrized matrix.
    condition_number :
        Ratio of the largest to smallest eigenvalue above tolerance. ``inf``
        denotes a singular or effectively singular matrix.
    trace :
        Matrix trace.
    frobenius_norm :
        Frobenius norm of the matrix.
    """

    rank_estimate: int
    min_eigenvalue: float
    max_eigenvalue: float
    condition_number: float
    trace: float
    frobenius_norm: float

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-friendly payload for artifact writers."""

        return {
            "rank_estimate": int(self.rank_estimate),
            "min_eigenvalue": float(self.min_eigenvalue),
            "max_eigenvalue": float(self.max_eigenvalue),
            "condition_number": float(self.condition_number),
            "trace": float(self.trace),
            "frobenius_norm": float(self.frobenius_norm),
        }


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
    tolerance = _matrix_rank_tolerance(matrix, rcond=rcond)
    active = np.abs(eigenvalues) > tolerance
    rank_estimate = int(np.count_nonzero(active))

    positive = eigenvalues[eigenvalues > tolerance]
    if positive.size == 0:
        condition_number = float("inf")
    else:
        condition_number = float(np.max(positive) / np.min(positive))

    return MatrixDiagnostics(
        rank_estimate=rank_estimate,
        min_eigenvalue=float(np.min(eigenvalues)),
        max_eigenvalue=float(np.max(eigenvalues)),
        condition_number=condition_number,
        trace=float(np.trace(matrix)),
        frobenius_norm=float(np.linalg.norm(matrix)),
    )


@dataclass(frozen=True)
class ObservationThetaLayout:
    """Define the canonical observation-level physical parameter layout.

    The observation-level accumulator operates on a fixed ordered vector of
    physical parameter labels. This object owns that order and validates all
    vectors or matrices that claim to live in the observation-level basis.

    The initial implementation is config-driven and intentionally narrow: it
    understands the canonical source terms, optional plate scale, and optional
    indexed primary/secondary Zernike families. It does not depend on a live
    :class:`ParameterStore`.
    """

    labels: tuple[str, ...]
    label_groups: tuple[str, ...]
    group_indices: dict[str, tuple[int, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        labels = _as_label_tuple(self.labels, name="labels")
        label_groups = tuple(str(group) for group in self.label_groups)
        if len(label_groups) != len(labels):
            raise ValueError("label_groups must align one-to-one with labels.")

        normalized_groups: dict[str, list[int]] = {}
        for index, group in enumerate(label_groups):
            normalized_groups.setdefault(group, []).append(index)

        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "label_groups", label_groups)
        object.__setattr__(
            self,
            "group_indices",
            {name: tuple(indices) for name, indices in normalized_groups.items()},
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ObservationThetaLayout:
        """Build a layout from a nested config-style mapping.

        Parameters
        ----------
        config :
            Mapping shaped like the proposed ``theta_layout`` block. Passing the
            full outer mapping or the inner ``theta_layout`` mapping is allowed.

        Returns
        -------
        ObservationThetaLayout
            Canonical physical-basis layout in deterministic label order.
        """

        theta_cfg = config.get("theta_layout", config)
        if not isinstance(theta_cfg, Mapping):
            raise ValueError("theta_layout config must be a mapping.")

        source_cfg = theta_cfg.get("source", {})
        optics_cfg = theta_cfg.get("optics", {})
        if not isinstance(source_cfg, Mapping):
            raise ValueError("theta_layout.source must be a mapping.")
        if not isinstance(optics_cfg, Mapping):
            raise ValueError("theta_layout.optics must be a mapping.")

        labels: list[str] = []
        groups: list[str] = []

        source_keys = (
            ("separation_as", "source.separation_as"),
            ("log_flux_total", "source.log_flux_total"),
            ("contrast", "source.contrast"),
        )
        for config_key, label in source_keys:
            if bool(source_cfg.get(config_key, False)):
                labels.append(label)
                groups.append("source")

        if bool(optics_cfg.get("plate_scale_as_per_pix", False)):
            labels.append("optics.plate_scale_as_per_pix")
            groups.append("optics.plate_scale")

        def _extend_zernikes(
            group_name: str,
            base_key: str,
            *,
            config_key: str,
        ) -> None:
            raw_group = optics_cfg.get(config_key, {})
            if raw_group is None:
                return
            if not isinstance(raw_group, Mapping):
                raise ValueError(f"theta_layout.optics.{config_key} must be a mapping.")
            if not bool(raw_group.get("enabled", False)):
                return
            raw_indices = raw_group.get("indices", ())
            indices = tuple(int(index) for index in raw_indices)
            if not indices:
                raise ValueError(
                    f"theta_layout.optics.{config_key}.indices must be non-empty "
                    "when the group is enabled."
                )
            if len(set(indices)) != len(indices):
                raise ValueError(
                    f"theta_layout.optics.{config_key}.indices contains duplicates."
                )
            for index in indices:
                labels.append(f"{base_key}[{index}]")
                groups.append(group_name)

        _extend_zernikes(
            "optics.primary_zernikes",
            "optics.primary.zernike_coeffs_nm",
            config_key="primary_zernikes",
        )
        _extend_zernikes(
            "optics.secondary_zernikes",
            "optics.secondary.zernike_coeffs_nm",
            config_key="secondary_zernikes",
        )

        return cls(labels=tuple(labels), label_groups=tuple(groups))

    @property
    def size(self) -> int:
        """Return the number of enabled physical parameters."""

        return len(self.labels)

    def validate_vector(
        self,
        values: Sequence[float] | np.ndarray,
        *,
        name: str = "theta",
    ) -> np.ndarray:
        """Return a validated vector in this layout's basis."""

        vector = _as_vector(values, name=name)
        if vector.shape != (self.size,):
            raise ValueError(
                f"{name} must have shape ({self.size},); received {vector.shape}."
            )
        return vector

    def validate_matrix(
        self,
        values: Sequence[Sequence[float]] | np.ndarray,
        *,
        name: str = "matrix",
    ) -> np.ndarray:
        """Return a validated square matrix in this layout's basis."""

        matrix = _as_square_matrix(values, name=name)
        expected_shape = (self.size, self.size)
        if matrix.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}; received {matrix.shape}."
            )
        return matrix

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-friendly description of the layout."""

        return {
            "labels": list(self.labels),
            "size": int(self.size),
            "label_groups": list(self.label_groups),
            "group_indices": {
                key: list(indices) for key, indices in self.group_indices.items()
            },
        }


@dataclass(frozen=True)
class SchurReductionResult:
    """Return one reduced information matrix plus solver diagnostics."""

    reduced_information: np.ndarray
    nuisance_diagnostics: MatrixDiagnostics
    reduced_diagnostics: MatrixDiagnostics
    damping: float
    rcond: float | None
    solve_method: str
    used_pseudoinverse: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly diagnostic snapshot."""

        return {
            "damping": float(self.damping),
            "rcond": None if self.rcond is None else float(self.rcond),
            "solve_method": self.solve_method,
            "used_pseudoinverse": bool(self.used_pseudoinverse),
            "nuisance": self.nuisance_diagnostics.to_dict(),
            "reduced": self.reduced_diagnostics.to_dict(),
        }


def schur_reduce_information(
    h_tt: Sequence[Sequence[float]] | np.ndarray,
    h_tp: Sequence[Sequence[float]] | np.ndarray,
    h_pp: Sequence[Sequence[float]] | np.ndarray,
    *,
    damping: float = 0.0,
    rcond: float | None = None,
) -> SchurReductionResult:
    """Return the Schur-reduced information on the retained ``theta`` block.

    This helper computes the reduced curvature

    ``S = H_tt - H_tp @ H_pp^{-1} @ H_pt``

    without explicitly forming ``H_pp^{-1}`` in the default path. It prefers a
    direct linear solve and falls back to a pseudo-inverse when the nuisance
    block is singular or numerically rank-deficient.

    Parameters
    ----------
    h_tt :
        Curvature block for retained observation-level parameters.
    h_tp :
        Cross block from retained parameters to nuisance parameters.
    h_pp :
        Curvature block for nuisance parameters to be eliminated.
    damping :
        Optional non-negative diagonal damping added to ``h_pp`` before the
        solve.
    rcond :
        Optional relative tolerance used for effective-rank diagnostics and the
        pseudo-inverse fallback.

    Returns
    -------
    SchurReductionResult
        Reduced information matrix and diagnostics for both the nuisance block
        and the reduced result.
    """

    h_tt_matrix = _as_square_matrix(h_tt, name="h_tt")
    h_tp_matrix = np.asarray(h_tp, dtype=float)
    if h_tp_matrix.ndim != 2:
        raise ValueError("h_tp must be a 2D matrix.")
    if not np.all(np.isfinite(h_tp_matrix)):
        raise ValueError("h_tp contains non-finite values.")
    h_pp_matrix = _as_square_matrix(h_pp, name="h_pp")

    theta_dim = h_tt_matrix.shape[0]
    nuisance_dim = h_pp_matrix.shape[0]
    if h_tp_matrix.shape != (theta_dim, nuisance_dim):
        raise ValueError(
            "h_tp must have shape "
            f"({theta_dim}, {nuisance_dim}); received {h_tp_matrix.shape}."
        )
    if damping < 0.0:
        raise ValueError("damping must be non-negative.")

    if nuisance_dim == 0:
        reduced = h_tt_matrix.copy()
        reduced_diag = _compute_matrix_diagnostics(reduced, rcond=rcond)
        return SchurReductionResult(
            reduced_information=reduced,
            nuisance_diagnostics=_compute_matrix_diagnostics(h_pp_matrix, rcond=rcond),
            reduced_diagnostics=reduced_diag,
            damping=float(damping),
            rcond=rcond,
            solve_method="no_nuisance_block",
            used_pseudoinverse=False,
        )

    h_pp_damped = h_pp_matrix.copy()
    if damping > 0.0:
        h_pp_damped = h_pp_damped + float(damping) * np.eye(nuisance_dim, dtype=float)

    nuisance_diag = _compute_matrix_diagnostics(h_pp_damped, rcond=rcond)
    used_pseudoinverse = False
    solve_method = "solve"
    try:
        if nuisance_diag.rank_estimate < nuisance_dim:
            raise np.linalg.LinAlgError("nuisance block is numerically rank-deficient")
        solved = np.linalg.solve(h_pp_damped, h_tp_matrix.T)
    except np.linalg.LinAlgError:
        used_pseudoinverse = True
        solve_method = "pinv"
        pinv = np.linalg.pinv(
            h_pp_damped,
            rcond=np.finfo(float).eps if rcond is None else float(rcond),
            hermitian=True,
        )
        solved = pinv @ h_tp_matrix.T

    reduced = h_tt_matrix - h_tp_matrix @ solved
    reduced = 0.5 * (reduced + reduced.T)
    reduced_diag = _compute_matrix_diagnostics(reduced, rcond=rcond)
    return SchurReductionResult(
        reduced_information=reduced,
        nuisance_diagnostics=nuisance_diag,
        reduced_diagnostics=reduced_diag,
        damping=float(damping),
        rcond=rcond,
        solve_method=solve_method,
        used_pseudoinverse=used_pseudoinverse,
    )


@dataclass(frozen=True)
class SubblockSummary:
    """Store one reduced local quadratic summary in physical parameter space.

    The summary lives in the observation-level physical basis and uses the
    convention

    ``L(theta) ~= const + g.T @ (theta - theta_ref) + 0.5 * dtheta.T @ S @ dtheta``

    where ``g`` is the objective gradient at ``theta_ref`` and ``S`` is the
    reduced information matrix after local nuisance terms have been eliminated.
    """

    subblock_id: str
    theta_labels: tuple[str, ...]
    theta_ref: np.ndarray
    reduced_information: np.ndarray
    reduced_score: np.ndarray
    summary_kind: str = "synthetic_schur"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        labels = _as_label_tuple(self.theta_labels, name="theta_labels")
        theta_ref = _as_vector(self.theta_ref, name="theta_ref")
        reduced_information = _as_square_matrix(
            self.reduced_information,
            name="reduced_information",
        )
        reduced_score = _as_vector(self.reduced_score, name="reduced_score")
        theta_size = len(labels)
        if theta_ref.shape != (theta_size,):
            raise ValueError(
                "theta_ref shape must match theta_labels; "
                f"received {theta_ref.shape} for {theta_size} labels."
            )
        if reduced_information.shape != (theta_size, theta_size):
            raise ValueError(
                "reduced_information shape must match theta_labels; "
                f"received {reduced_information.shape} for {theta_size} labels."
            )
        if reduced_score.shape != (theta_size,):
            raise ValueError(
                "reduced_score shape must match theta_labels; "
                f"received {reduced_score.shape} for {theta_size} labels."
            )

        payload = dict(self.diagnostics)
        object.__setattr__(self, "theta_labels", labels)
        object.__setattr__(self, "theta_ref", theta_ref)
        object.__setattr__(self, "reduced_information", reduced_information)
        object.__setattr__(self, "reduced_score", reduced_score)
        object.__setattr__(self, "diagnostics", payload)

    @classmethod
    def from_reduced_form(
        cls,
        *,
        subblock_id: str,
        theta_labels: Sequence[str],
        theta_ref: Sequence[float] | np.ndarray,
        reduced_information: Sequence[Sequence[float]] | np.ndarray,
        reduced_score: Sequence[float] | np.ndarray,
        summary_kind: str = "synthetic_schur",
        damping_used: float = 0.0,
        diagnostics: Mapping[str, Any] | None = None,
        rcond: float | None = None,
    ) -> SubblockSummary:
        """Build one summary and attach standard matrix diagnostics."""

        info = _as_square_matrix(reduced_information, name="reduced_information")
        score = _as_vector(reduced_score, name="reduced_score")
        diag = _compute_matrix_diagnostics(info, rcond=rcond)
        payload = dict(diagnostics or {})
        payload.setdefault("summary_kind", str(summary_kind))
        payload.setdefault("rank_estimate", int(diag.rank_estimate))
        payload.setdefault("min_eigenvalue", float(diag.min_eigenvalue))
        payload.setdefault("max_eigenvalue", float(diag.max_eigenvalue))
        payload.setdefault("condition_number", float(diag.condition_number))
        payload.setdefault("trace", float(diag.trace))
        payload.setdefault("frobenius_norm", float(diag.frobenius_norm))
        payload.setdefault("damping_used", bool(damping_used > 0.0))
        payload.setdefault("damping_value", float(damping_used))
        payload.setdefault("score_norm", float(np.linalg.norm(score)))
        return cls(
            subblock_id=str(subblock_id),
            theta_labels=tuple(theta_labels),
            theta_ref=np.asarray(theta_ref, dtype=float),
            reduced_information=info,
            reduced_score=score,
            summary_kind=str(summary_kind),
            diagnostics=payload,
        )

    @property
    def theta_size(self) -> int:
        """Return the number of reduced observation-level parameters."""

        return len(self.theta_labels)

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        """Return a JSON-friendly summary payload."""

        payload: dict[str, Any] = {
            "subblock_id": self.subblock_id,
            "theta_labels": list(self.theta_labels),
            "summary_kind": self.summary_kind,
            "diagnostics": dict(self.diagnostics),
        }
        if include_arrays:
            payload["theta_ref"] = self.theta_ref.tolist()
            payload["reduced_score"] = self.reduced_score.tolist()
            payload["reduced_information"] = self.reduced_information.tolist()
        return payload


@dataclass(frozen=True)
class ObservationBeliefState:
    """Store one information-form belief state in physical parameter space.

    This is the observation-level accumulator state used by the synthetic demo.
    The state stores the physical canonical labels, a posterior or prior mean,
    and the associated precision matrix. Covariance storage is optional because
    some workflows may only materialize it on demand.
    """

    theta_labels: tuple[str, ...]
    mean: np.ndarray
    precision: np.ndarray
    covariance: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        labels = _as_label_tuple(self.theta_labels, name="theta_labels")
        mean = _as_vector(self.mean, name="mean")
        precision = _as_square_matrix(self.precision, name="precision")
        if mean.shape != (len(labels),):
            raise ValueError("mean shape must match theta_labels.")
        if precision.shape != (len(labels), len(labels)):
            raise ValueError("precision shape must match theta_labels.")

        covariance = self.covariance
        if covariance is not None:
            covariance = _as_square_matrix(covariance, name="covariance")
            if covariance.shape != precision.shape:
                raise ValueError("covariance shape must match precision shape.")

        object.__setattr__(self, "theta_labels", labels)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_diagonal_prior(
        cls,
        *,
        theta_labels: Sequence[str],
        mean: Sequence[float] | np.ndarray,
        sigma: Sequence[float] | np.ndarray,
        metadata: Mapping[str, Any] | None = None,
    ) -> ObservationBeliefState:
        """Build a Gaussian prior with diagonal covariance."""

        labels = _as_label_tuple(theta_labels, name="theta_labels")
        mean_vector = _as_vector(mean, name="mean")
        sigma_vector = _as_vector(sigma, name="sigma")
        if mean_vector.shape != (len(labels),):
            raise ValueError("mean shape must match theta_labels.")
        if sigma_vector.shape != (len(labels),):
            raise ValueError("sigma shape must match theta_labels.")
        if np.any(sigma_vector <= 0.0):
            raise ValueError("sigma values must be strictly positive.")

        variance = np.square(sigma_vector)
        precision = np.diag(1.0 / variance)
        covariance = np.diag(variance)
        meta = dict(metadata or {})
        meta.setdefault("prior_sigma", sigma_vector.tolist())
        return cls(
            theta_labels=labels,
            mean=mean_vector,
            precision=precision,
            covariance=covariance,
            metadata=meta,
        )

    @property
    def theta_size(self) -> int:
        """Return the number of parameters in the belief state."""

        return len(self.theta_labels)

    @property
    def information_vector(self) -> np.ndarray:
        """Return the information vector ``eta = Lambda @ mean``."""

        return self.precision @ self.mean

    def sigma(self) -> np.ndarray:
        """Return posterior or prior standard deviations."""

        if self.covariance is None:
            covariance = np.linalg.pinv(self.precision, hermitian=True)
        else:
            covariance = self.covariance
        return np.sqrt(np.clip(np.diag(covariance), 0.0, None))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly description of the belief state."""

        return {
            "theta_labels": list(self.theta_labels),
            "mean": self.mean.tolist(),
            "precision": self.precision.tolist(),
            "covariance": None if self.covariance is None else self.covariance.tolist(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ObservationUpdateStep:
    """Capture the posterior state after one cumulative summary update."""

    n_subblocks: int
    subblock_id: str | None
    mean: np.ndarray
    precision: np.ndarray
    covariance: np.ndarray
    information_vector: np.ndarray
    diagnostics: MatrixDiagnostics

    def __post_init__(self) -> None:
        mean = _as_vector(self.mean, name="mean")
        precision = _as_square_matrix(self.precision, name="precision")
        covariance = _as_square_matrix(self.covariance, name="covariance")
        information_vector = _as_vector(
            self.information_vector,
            name="information_vector",
        )
        theta_size = mean.size
        expected = (theta_size, theta_size)
        if precision.shape != expected or covariance.shape != expected:
            raise ValueError("precision/covariance shapes must match mean length.")
        if information_vector.shape != (theta_size,):
            raise ValueError("information_vector shape must match mean length.")

        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "information_vector", information_vector)

    def sigma(self) -> np.ndarray:
        """Return standard deviations after this cumulative update."""

        return np.sqrt(np.clip(np.diag(self.covariance), 0.0, None))


@dataclass(frozen=True)
class ObservationUpdateResult:
    """Return the accumulated observation-level posterior and update trace."""

    prior: ObservationBeliefState
    posterior: ObservationBeliefState
    summaries: tuple[SubblockSummary, ...]
    information_vector: np.ndarray
    cumulative_steps: tuple[ObservationUpdateStep, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        information_vector = _as_vector(
            self.information_vector,
            name="information_vector",
        )
        if information_vector.shape != (self.posterior.theta_size,):
            raise ValueError("information_vector shape must match posterior size.")
        object.__setattr__(self, "information_vector", information_vector)
        object.__setattr__(self, "metadata", dict(self.metadata))


def accumulate_summary_information(
    theta_labels: Sequence[str],
    summaries: Sequence[SubblockSummary],
) -> np.ndarray:
    """Accumulate reduced information matrices into one global basis.

    This helper is useful for diagnostics that should look only at the evidence
    contributed by sub-block summaries, excluding the prior precision.
    """

    global_labels = _as_label_tuple(theta_labels, name="theta_labels")
    accumulated = np.zeros((len(global_labels), len(global_labels)), dtype=float)
    for summary in summaries:
        summary_precision, _, _ = _summary_to_global_arrays(
            summary,
            global_labels=global_labels,
        )
        accumulated += summary_precision
    return 0.5 * (accumulated + accumulated.T)


def build_prior_whitened_information_gain_matrix(
    information: Sequence[Sequence[float]] | np.ndarray,
    prior_sigma: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return the prior-whitened information-gain matrix.

    For diagonal prior covariance with standard deviations ``prior_sigma``, the
    whitened information gain is

    ``Lambda_gain = diag(prior_sigma) @ information @ diag(prior_sigma)``.
    """

    information_matrix = _as_square_matrix(information, name="information")
    sigma_vector = _as_positive_vector(prior_sigma, name="prior_sigma")
    if information_matrix.shape != (sigma_vector.size, sigma_vector.size):
        raise ValueError("information shape must match prior_sigma length.")
    whitening = np.diag(sigma_vector)
    gain = whitening @ information_matrix @ whitening
    return 0.5 * (gain + gain.T)


def _summary_to_global_arrays(
    summary: SubblockSummary,
    *,
    global_labels: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    label_to_index = {label: index for index, label in enumerate(global_labels)}
    global_dim = len(global_labels)
    info = np.zeros((global_dim, global_dim), dtype=float)
    score = np.zeros((global_dim,), dtype=float)
    theta_ref = np.zeros((global_dim,), dtype=float)

    summary_indices: list[int] = []
    for label in summary.theta_labels:
        if label not in label_to_index:
            raise ValueError(
                f"Summary {summary.subblock_id!r} contains label {label!r} "
                "outside the prior belief state."
            )
        summary_indices.append(label_to_index[label])

    index_array = np.asarray(summary_indices, dtype=int)
    info[np.ix_(index_array, index_array)] = summary.reduced_information
    score[index_array] = summary.reduced_score
    theta_ref[index_array] = summary.theta_ref
    return info, score, theta_ref


def _solve_symmetric_system(
    precision: np.ndarray,
    information_vector: np.ndarray,
    *,
    rcond: float | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        mean = np.linalg.solve(precision, information_vector)
        covariance = np.linalg.inv(precision)
        solve_method = "solve"
    except np.linalg.LinAlgError:
        pinv = np.linalg.pinv(
            precision,
            rcond=np.finfo(float).eps if rcond is None else float(rcond),
            hermitian=True,
        )
        mean = pinv @ information_vector
        covariance = pinv
        solve_method = "pinv"

    return mean, 0.5 * (covariance + covariance.T), solve_method


def update_observation_belief(
    prior: ObservationBeliefState,
    summaries: Sequence[SubblockSummary],
    *,
    damping: float = 0.0,
) -> ObservationUpdateResult:
    """Combine a prior belief state with many reduced local summaries.

    This implements the information-form convention described in the design
    note. For each local quadratic summary

    ``L_b(theta) ~= const + g_b.T @ (theta - theta_ref_b) + 0.5 * dtheta.T @ S_b @ dtheta``

    the corresponding global information contribution is

    ``eta_b = S_b @ theta_ref_b - g_b``.

    Parameters
    ----------
    prior :
        Prior belief state in physical parameter space.
    summaries :
        Sequence of reduced sub-block summaries. Summaries may use a different
        label order or even a strict subset of the prior labels.
    damping :
        Optional non-negative diagonal damping applied to the accumulated
        posterior precision before the solve.

    Returns
    -------
    ObservationUpdateResult
        Posterior belief state plus the cumulative update trace.
    """

    if damping < 0.0:
        raise ValueError("damping must be non-negative.")

    theta_labels = prior.theta_labels
    raw_precision = prior.precision.copy()
    raw_information = prior.information_vector.copy()
    cumulative_steps: list[ObservationUpdateStep] = []
    solve_method = "solve"

    for step_index, summary in enumerate(summaries, start=1):
        summary_precision, summary_score, summary_theta_ref = _summary_to_global_arrays(
            summary,
            global_labels=theta_labels,
        )
        raw_precision += summary_precision
        raw_information += summary_precision @ summary_theta_ref - summary_score

        solve_precision = raw_precision.copy()
        if damping > 0.0:
            solve_precision += float(damping) * np.eye(prior.theta_size, dtype=float)

        mean, covariance, solve_method = _solve_symmetric_system(
            solve_precision,
            raw_information,
        )
        diagnostics = _compute_matrix_diagnostics(solve_precision)
        cumulative_steps.append(
            ObservationUpdateStep(
                n_subblocks=step_index,
                subblock_id=summary.subblock_id,
                mean=mean,
                precision=solve_precision,
                covariance=covariance,
                information_vector=raw_information.copy(),
                diagnostics=diagnostics,
            )
        )

    posterior_precision = raw_precision.copy()
    if damping > 0.0:
        posterior_precision += float(damping) * np.eye(prior.theta_size, dtype=float)
    posterior_mean, posterior_covariance, solve_method = _solve_symmetric_system(
        posterior_precision,
        raw_information,
    )
    posterior = ObservationBeliefState(
        theta_labels=theta_labels,
        mean=posterior_mean,
        precision=posterior_precision,
        covariance=posterior_covariance,
        metadata={
            "damping": float(damping),
            "n_summaries": int(len(summaries)),
            "posterior_precision_diagnostics": _compute_matrix_diagnostics(
                posterior_precision
            ).to_dict(),
            "solve_method": solve_method,
        },
    )
    return ObservationUpdateResult(
        prior=prior,
        posterior=posterior,
        summaries=tuple(summaries),
        information_vector=raw_information,
        cumulative_steps=tuple(cumulative_steps),
        metadata={
            "damping": float(damping),
            "n_summaries": int(len(summaries)),
            "solve_method": solve_method,
            "raw_precision_diagnostics": _compute_matrix_diagnostics(
                raw_precision
            ).to_dict(),
        },
    )


@dataclass(frozen=True)
class ObservationEigenBasis:
    """Represent posterior precision eigenmodes in the physical basis.

    The belief state remains stored in the physical canonical parameter basis.
    This object is a diagnostic transform layered on top of that storage: it
    exposes constrained and weak directions, lets callers project deltas into
    eigen coordinates, and provides table-ready mode summaries.
    """

    labels: tuple[str, ...]
    eigenvalues: np.ndarray
    effective_eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    weak_mode_mask: np.ndarray
    condition_number: float
    eig_floor_abs: float
    eig_floor_rel: float

    def __post_init__(self) -> None:
        labels = _as_label_tuple(self.labels, name="labels")
        eigenvalues = _as_vector(self.eigenvalues, name="eigenvalues")
        effective_eigenvalues = _as_vector(
            self.effective_eigenvalues,
            name="effective_eigenvalues",
        )
        weak_mode_mask = np.asarray(self.weak_mode_mask, dtype=bool)
        eigenvectors = np.asarray(self.eigenvectors, dtype=float)
        if eigenvectors.ndim != 2:
            raise ValueError("eigenvectors must be a 2D matrix.")
        expected = (len(labels), len(labels))
        if eigenvectors.shape != expected:
            raise ValueError(
                f"eigenvectors must have shape {expected}; received {eigenvectors.shape}."
            )
        if eigenvalues.shape != (len(labels),):
            raise ValueError("eigenvalues shape must match labels.")
        if effective_eigenvalues.shape != eigenvalues.shape:
            raise ValueError("effective_eigenvalues shape must match eigenvalues.")
        if weak_mode_mask.shape != eigenvalues.shape:
            raise ValueError("weak_mode_mask shape must match eigenvalues.")

        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "eigenvalues", eigenvalues)
        object.__setattr__(self, "effective_eigenvalues", effective_eigenvalues)
        object.__setattr__(self, "eigenvectors", eigenvectors)
        object.__setattr__(self, "weak_mode_mask", weak_mode_mask)

    def sigma_along_modes(self) -> np.ndarray:
        """Return floored sigmas along each eigenmode for stable transforms."""

        return 1.0 / np.sqrt(np.clip(self.effective_eigenvalues, 1.0e-30, None))

    def raw_sigma_along_modes(self) -> np.ndarray:
        """Return raw sigmas along each eigenmode when defined."""

        raw_sigma = np.full(self.eigenvalues.shape, np.inf, dtype=float)
        positive = self.eigenvalues > 0.0
        raw_sigma[positive] = 1.0 / np.sqrt(self.eigenvalues[positive])
        return raw_sigma

    def floored_sigma_along_modes(self) -> np.ndarray:
        """Return floored sigmas along each eigenmode."""

        return self.sigma_along_modes()

    def was_floored(self) -> np.ndarray:
        """Return a mask for modes whose raw eigenvalues were floored."""

        return self.effective_eigenvalues > self.eigenvalues

    def physical_delta_to_eigen(
        self,
        delta: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Project a physical-basis delta into eigen coordinates."""

        vector = _as_vector(delta, name="delta")
        if vector.shape != (len(self.labels),):
            raise ValueError("delta shape must match eigenbasis labels.")
        return self.eigenvectors.T @ vector

    def eigen_delta_to_physical(
        self,
        eigen_delta: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Map an eigen-coordinate delta back into physical coordinates."""

        vector = _as_vector(eigen_delta, name="eigen_delta")
        if vector.shape != (len(self.labels),):
            raise ValueError("eigen_delta shape must match eigenbasis labels.")
        return self.eigenvectors @ vector

    def mode_contributors(
        self,
        mode_index: int,
        *,
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        """Return the dominant physical contributors to one eigenmode."""

        if mode_index < 0 or mode_index >= len(self.labels):
            raise IndexError("mode_index out of range.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        vector = self.eigenvectors[:, int(mode_index)]
        order = np.argsort(np.abs(vector))[::-1][:top_k]
        return [(self.labels[index], float(vector[index])) for index in order]

    def to_rows(self, *, top_k: int = 3) -> list[dict[str, Any]]:
        """Return one CSV-ready row per eigenmode."""

        raw_sigmas = self.raw_sigma_along_modes()
        floored_sigmas = self.floored_sigma_along_modes()
        floored_mask = self.was_floored()
        rows: list[dict[str, Any]] = []
        for mode_index, eigenvalue in enumerate(self.eigenvalues):
            contributors = self.mode_contributors(mode_index, top_k=top_k)
            row: dict[str, Any] = {
                "mode_index": int(mode_index),
                "raw_eigenvalue": float(eigenvalue),
                "raw_sigma_along_mode": float(raw_sigmas[mode_index]),
                "floored_eigenvalue": float(self.effective_eigenvalues[mode_index]),
                "floored_sigma_along_mode": float(floored_sigmas[mode_index]),
                "was_floored": bool(floored_mask[mode_index]),
                "weak_mode": bool(self.weak_mode_mask[mode_index]),
                "top_contributors": "; ".join(
                    f"{label}:{coefficient:+.6f}"
                    for label, coefficient in contributors
                ),
            }
            for contributor_index, (label, coefficient) in enumerate(
                contributors,
                start=1,
            ):
                row[f"top_label_{contributor_index}"] = label
                row[f"top_coeff_{contributor_index}"] = float(coefficient)
            rows.append(row)
        return rows


def build_observation_eigenbasis(
    precision: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[str],
    *,
    eig_floor_abs: float = 0.0,
    eig_floor_rel: float = 0.0,
) -> ObservationEigenBasis:
    """Build a diagnostic eigenbasis from an observation-level precision.

    The returned eigenvalues are sorted from strongest to weakest constraint,
    meaning descending by eigenvalue.
    """

    if eig_floor_abs < 0.0 or eig_floor_rel < 0.0:
        raise ValueError("eig_floor_abs and eig_floor_rel must be non-negative.")

    label_tuple = _as_label_tuple(labels, name="labels")
    precision_matrix = _as_square_matrix(precision, name="precision")
    if precision_matrix.shape != (len(label_tuple), len(label_tuple)):
        raise ValueError("precision shape must match labels.")

    eigenvalues, eigenvectors = np.linalg.eigh(precision_matrix)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    max_eigenvalue = float(np.max(eigenvalues)) if eigenvalues.size else 0.0
    floor = max(float(eig_floor_abs), float(eig_floor_rel) * max(max_eigenvalue, 0.0))
    effective = np.maximum(eigenvalues, floor)
    weak_mode_mask = eigenvalues <= floor if floor > 0.0 else eigenvalues <= 0.0
    diagnostics = _compute_matrix_diagnostics(precision_matrix)
    return ObservationEigenBasis(
        labels=label_tuple,
        eigenvalues=eigenvalues,
        effective_eigenvalues=effective,
        eigenvectors=eigenvectors,
        weak_mode_mask=weak_mode_mask,
        condition_number=float(diagnostics.condition_number),
        eig_floor_abs=float(eig_floor_abs),
        eig_floor_rel=float(eig_floor_rel),
    )
