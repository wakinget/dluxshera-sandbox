from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from dluxshera.datasets import ArrayShardReader
from dluxshera.datasets.schema import read_json, read_jsonl

__all__ = ["SampleCatalog", "load_sample_catalog"]


SCIENCE_GROUP_FIELD = "group_ids.physical_delta_sha256"
NUISANCE_GROUP_FIELD = "nuisance_id"
PREPARED_ARTIFACT_ID = "PREP-V3-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_digest(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _field(row: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    value: Any = row
    for key in dotted.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def _science_group_id(row: Mapping[str, Any], physical_vector: np.ndarray) -> str:
    explicit = _field(row, SCIENCE_GROUP_FIELD)
    if explicit not in (None, ""):
        return str(explicit)
    payload = json.dumps(
        [float(v) for v in physical_vector],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _nuisance_group_id(row: Mapping[str, Any], nuisance_vector: np.ndarray | None) -> str:
    explicit = row.get(NUISANCE_GROUP_FIELD)
    if explicit not in (None, ""):
        return str(explicit)
    explicit = _field(row, "group_ids.nuisance")
    if explicit not in (None, ""):
        return str(explicit)
    if nuisance_vector is not None:
        payload = json.dumps(
            [float(v) for v in nuisance_vector],
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return "__none__"


def _vector(row: Mapping[str, Any], field: str, *, dtype: Any = np.float32) -> np.ndarray:
    value = row.get(field)
    if value is None:
        raise ValueError(f"Prepared index row {row.get('sample_id')} is missing {field!r}.")
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim != 1:
        raise ValueError(f"Prepared index field {field!r} must be 1D, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Prepared index field {field!r} contains non-finite values.")
    return arr


def _optional_vector(
    row: Mapping[str, Any],
    field: str,
    *,
    expected_dim: int | None,
    dtype: Any = np.float32,
) -> np.ndarray | None:
    value = row.get(field)
    if value is None:
        return None
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim != 1:
        raise ValueError(f"Prepared index field {field!r} must be 1D, got shape {arr.shape}.")
    if expected_dim is not None and arr.shape[0] != expected_dim:
        raise ValueError(
            f"Prepared index field {field!r} has dimension {arr.shape[0]}, expected {expected_dim}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Prepared index field {field!r} contains non-finite values.")
    return arr


def _as_string(value: Any) -> str:
    return "" if value is None else str(value)


def _as_int(value: Any, *, missing: int = -1) -> int:
    return missing if value is None else int(value)


def _vector_space_labels(vector_spaces: Mapping[str, Any], space_key: str) -> tuple[str, ...]:
    space = vector_spaces.get("spaces", {}).get(space_key)
    if not isinstance(space, Mapping):
        return ()
    return tuple(str(component["label"]) for component in space.get("components", []))


def _fisher_sigmas(vector_spaces: Mapping[str, Any], dim: int) -> np.ndarray:
    transform = vector_spaces.get("transforms", {}).get("fisher_diagonal_scale", {})
    scales = np.asarray(transform.get("scales", []), dtype=np.float64)
    if scales.shape != (dim,):
        raise ValueError(
            "vector_spaces.json fisher_diagonal_scale scales are missing or have "
            f"shape {scales.shape}, expected ({dim},)."
        )
    if not np.all(np.isfinite(scales)) or np.any(scales == 0.0):
        raise ValueError("Fisher sigma scales must be finite and non-zero.")
    return scales


@dataclass(frozen=True)
class SampleCatalog:
    """Hold compact metadata needed to build ML image pairs.

    The catalog streams the prepared ``index.jsonl`` once and retains NumPy
    arrays plus small string-index mappings.  It deliberately does not keep the
    original JSON dictionaries, and it does not know anything about PyTorch or
    Siamese training loops.
    """

    root: Path
    artifact_id: str
    prepared_dataset_hash: str
    manifest: Mapping[str, Any]
    vector_spaces: Mapping[str, Any]
    sample_ids: np.ndarray
    array_indices: np.ndarray
    science_group_ids: np.ndarray
    nuisance_group_ids: np.ndarray
    dataset_families: np.ndarray
    sample_roles: np.ndarray
    pair_ids: np.ndarray
    grid_i_indices: np.ndarray
    grid_j_indices: np.ndarray
    fisher_scaled_deltas: np.ndarray
    physical_deltas: np.ndarray
    nuisance_vectors: np.ndarray
    nuisance_sigma_vectors: np.ndarray
    sample_shape: tuple[int, ...]
    parameter_labels: tuple[str, ...]
    nuisance_labels: tuple[str, ...]
    fisher_sigmas: np.ndarray
    science_group_policy: str = SCIENCE_GROUP_FIELD
    nuisance_group_policy: str = NUISANCE_GROUP_FIELD

    def __post_init__(self) -> None:
        n = len(self.sample_ids)
        if n == 0:
            raise ValueError("SampleCatalog requires at least one sample.")
        for name, values in {
            "array_indices": self.array_indices,
            "science_group_ids": self.science_group_ids,
            "nuisance_group_ids": self.nuisance_group_ids,
            "dataset_families": self.dataset_families,
            "pair_ids": self.pair_ids,
            "fisher_scaled_deltas": self.fisher_scaled_deltas,
            "physical_deltas": self.physical_deltas,
        }.items():
            if len(values) != n:
                raise ValueError(f"{name} length {len(values)} does not match sample count {n}.")
        if self.fisher_scaled_deltas.ndim != 2:
            raise ValueError("fisher_scaled_deltas must be a 2D array.")
        if self.physical_deltas.shape != self.fisher_scaled_deltas.shape:
            raise ValueError("physical_deltas shape must match fisher_scaled_deltas.")
        if self.fisher_sigmas.shape != (self.science_dim,):
            raise ValueError("fisher_sigmas dimension must match science_dim.")

    @property
    def sample_count(self) -> int:
        """Return the number of prepared samples represented by this catalog."""
        return int(self.sample_ids.shape[0])

    @property
    def science_dim(self) -> int:
        """Return the Fisher-scaled science vector dimension."""
        return int(self.fisher_scaled_deltas.shape[1])

    @property
    def nuisance_dim(self) -> int:
        """Return the nuisance vector dimension, or zero if unavailable."""
        if self.nuisance_vectors.ndim != 2:
            return 0
        return int(self.nuisance_vectors.shape[1])

    @property
    def sample_id_to_index(self) -> dict[str, int]:
        """Return a stable sample-id lookup dictionary."""
        return {str(sample_id): int(idx) for idx, sample_id in enumerate(self.sample_ids)}

    @property
    def science_group_count(self) -> int:
        """Return the number of unique science-state identities."""
        return int(len(set(str(v) for v in self.science_group_ids)))

    @property
    def nuisance_group_count(self) -> int:
        """Return the number of unique nuisance identities."""
        return int(len(set(str(v) for v in self.nuisance_group_ids)))

    def image_reader(self, *, cache_size: int = 4) -> ArrayShardReader:
        """Return an ``ArrayShardReader`` for the catalog's prepared root."""
        return ArrayShardReader(self.root, cache_size=cache_size)

    def indices_for_groups(
        self,
        *,
        science_groups: Iterable[str] | None = None,
        nuisance_groups: Iterable[str] | None = None,
    ) -> np.ndarray:
        """Return sample indices matching optional science and nuisance groups."""
        mask = np.ones((self.sample_count,), dtype=bool)
        if science_groups is not None:
            allowed = {str(v) for v in science_groups}
            mask &= np.asarray([str(v) in allowed for v in self.science_group_ids])
        if nuisance_groups is not None:
            allowed = {str(v) for v in nuisance_groups}
            mask &= np.asarray([str(v) in allowed for v in self.nuisance_group_ids])
        return np.flatnonzero(mask).astype(np.int64)

    def physical_from_z(self, z_delta: np.ndarray) -> np.ndarray:
        """Transform Fisher-scaled deltas back to native physical coordinates."""
        return np.asarray(z_delta, dtype=np.float64) * self.fisher_sigmas

    def summary(self) -> dict[str, Any]:
        """Return compact human-readable catalog metadata."""
        return {
            "artifact_id": self.artifact_id,
            "prepared_dataset_hash": self.prepared_dataset_hash,
            "root": str(self.root),
            "sample_count": self.sample_count,
            "sample_shape": list(self.sample_shape),
            "science_dim": self.science_dim,
            "science_group_count": self.science_group_count,
            "nuisance_dim": self.nuisance_dim,
            "nuisance_group_count": self.nuisance_group_count,
            "dataset_families": {
                str(value): int(np.count_nonzero(self.dataset_families == value))
                for value in sorted(set(self.dataset_families.tolist()))
            },
            "science_group_policy": self.science_group_policy,
            "nuisance_group_policy": self.nuisance_group_policy,
        }


def load_sample_catalog(prepared_root: Path, *, artifact_id: str = PREPARED_ARTIFACT_ID) -> SampleCatalog:
    """Stream a prepared SHERA dataset index into compact ML catalog arrays."""
    root = Path(prepared_root).resolve()
    manifest_path = root / "manifest.json"
    vector_spaces_path = root / "vector_spaces.json"
    index_path = root / "index.jsonl"
    manifest = read_json(manifest_path)
    vector_spaces = read_json(vector_spaces_path)
    if not index_path.exists():
        raise FileNotFoundError(f"Prepared dataset is missing {index_path}.")

    sample_ids: list[str] = []
    array_indices: list[int] = []
    science_group_ids: list[str] = []
    nuisance_group_ids: list[str] = []
    dataset_families: list[str] = []
    sample_roles: list[str] = []
    pair_ids: list[str] = []
    grid_i_indices: list[int] = []
    grid_j_indices: list[int] = []
    z_rows: list[np.ndarray] = []
    theta_rows: list[np.ndarray] = []
    nuisance_rows: list[np.ndarray] = []
    nuisance_sigma_rows: list[np.ndarray] = []
    nuisance_dim: int | None = None

    for row_number, row in enumerate(read_jsonl(index_path), start=1):
        sample_id = row.get("sample_id")
        if sample_id in (None, ""):
            raise ValueError(f"{index_path} row {row_number} is missing sample_id.")
        array_index = int(row.get("array_index", row.get("sample_index", row_number - 1)))
        z = _vector(row, "fisher_scaled_delta", dtype=np.float32)
        theta = _vector(row, "physical_delta", dtype=np.float32)
        if theta.shape != z.shape:
            raise ValueError(
                f"{index_path} row {row_number} physical_delta shape {theta.shape} "
                f"does not match fisher_scaled_delta shape {z.shape}."
            )
        nuisance = _optional_vector(
            row,
            "nuisance_vector",
            expected_dim=nuisance_dim,
            dtype=np.float32,
        )
        if nuisance is not None and nuisance_dim is None:
            nuisance_dim = int(nuisance.shape[0])
        nuisance_sigma = _optional_vector(
            row,
            "nuisance_sigma_vector",
            expected_dim=nuisance_dim,
            dtype=np.float32,
        )
        if nuisance is None:
            nuisance = np.zeros((0 if nuisance_dim is None else nuisance_dim,), dtype=np.float32)
        if nuisance_sigma is None:
            nuisance_sigma = np.zeros_like(nuisance)

        sample_ids.append(str(sample_id))
        array_indices.append(array_index)
        science_group_ids.append(_science_group_id(row, theta))
        nuisance_group_ids.append(_nuisance_group_id(row, nuisance))
        dataset_families.append(_as_string(row.get("dataset_family")))
        sample_roles.append(_as_string(row.get("sample_role")))
        pair_ids.append(_as_string(row.get("pair_id")))
        grid_i_indices.append(_as_int(row.get("grid_i_index")))
        grid_j_indices.append(_as_int(row.get("grid_j_index")))
        z_rows.append(z)
        theta_rows.append(theta)
        nuisance_rows.append(nuisance)
        nuisance_sigma_rows.append(nuisance_sigma)

    if not sample_ids:
        raise ValueError(f"{index_path} contains no samples.")
    z_array = np.vstack(z_rows).astype(np.float32, copy=False)
    theta_array = np.vstack(theta_rows).astype(np.float32, copy=False)
    if nuisance_dim is None:
        nuisance_dim = 0
    nuisance_array = (
        np.zeros((len(sample_ids), 0), dtype=np.float32)
        if nuisance_dim == 0
        else np.vstack(nuisance_rows).astype(np.float32, copy=False)
    )
    nuisance_sigma_array = (
        np.zeros((len(sample_ids), 0), dtype=np.float32)
        if nuisance_dim == 0
        else np.vstack(nuisance_sigma_rows).astype(np.float32, copy=False)
    )
    sample_shape = tuple(
        int(v)
        for v in manifest.get("array_storage", {}).get(
            "sample_shape",
            manifest.get("sample_shape", ()),
        )
    )
    if not sample_shape:
        sample_shape = tuple(
            int(v)
            for v in read_json(root / "array_shards_manifest.json").get("sample_shape", ())
        )
    prepared_dataset_hash = _stable_digest(
        {
            "manifest_sha256": _sha256_file(manifest_path),
            "vector_spaces_sha256": _sha256_file(vector_spaces_path),
            "source_dataset": manifest.get("source_dataset", {}),
            "array_storage": manifest.get("array_storage", {}),
            "sample_count": len(sample_ids),
        }
    )
    parameter_labels = _vector_space_labels(vector_spaces, "fisher_scaled_delta")
    if parameter_labels and len(parameter_labels) != z_array.shape[1]:
        raise ValueError(
            f"vector_spaces.json has {len(parameter_labels)} Fisher labels but index vectors have "
            f"dimension {z_array.shape[1]}."
        )
    if not parameter_labels:
        parameter_labels = tuple(f"z[{idx}]" for idx in range(z_array.shape[1]))
    nuisance_labels = _vector_space_labels(vector_spaces, "registration_nuisance")

    return SampleCatalog(
        root=root,
        artifact_id=str(artifact_id),
        prepared_dataset_hash=prepared_dataset_hash,
        manifest=manifest,
        vector_spaces=vector_spaces,
        sample_ids=np.asarray(sample_ids, dtype=object),
        array_indices=np.asarray(array_indices, dtype=np.int64),
        science_group_ids=np.asarray(science_group_ids, dtype=object),
        nuisance_group_ids=np.asarray(nuisance_group_ids, dtype=object),
        dataset_families=np.asarray(dataset_families, dtype=object),
        sample_roles=np.asarray(sample_roles, dtype=object),
        pair_ids=np.asarray(pair_ids, dtype=object),
        grid_i_indices=np.asarray(grid_i_indices, dtype=np.int32),
        grid_j_indices=np.asarray(grid_j_indices, dtype=np.int32),
        fisher_scaled_deltas=z_array,
        physical_deltas=theta_array,
        nuisance_vectors=nuisance_array,
        nuisance_sigma_vectors=nuisance_sigma_array,
        sample_shape=sample_shape,
        parameter_labels=parameter_labels,
        nuisance_labels=nuisance_labels,
        fisher_sigmas=_fisher_sigmas(vector_spaces, z_array.shape[1]),
    )
