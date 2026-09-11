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
    explicit = row.get("science_state_id")
    if explicit not in (None, ""):
        return str(explicit)
    explicit = _field(row, "group_ids.science")
    if explicit not in (None, ""):
        return str(explicit)
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
    explicit = row.get("nuisance_state_id")
    if explicit not in (None, ""):
        return str(explicit)
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


def _optional_int(row: Mapping[str, Any], *keys: str, missing: int = -1) -> int:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return int(value)
    return int(missing)


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
    """Describe and query one prepared SHERA ML image dataset.

    ``SampleCatalog`` is the preferred high-level entry point for ML data
    consumers.  It streams the prepared dataset ``index.jsonl`` once and keeps
    compact NumPy arrays plus stable string identities for each rendered sample.
    It is intentionally model-architecture neutral: use it from notebooks,
    custom training loops, pair samplers, PyTorch datasets, or any other code
    that needs controlled access to SHERA prepared images and metadata.

    A prepared sample combines two independent identities.  The *science group*
    identifies the physical science state, such as the Fisher-scaled or native
    physical perturbation vector.  The *nuisance group* identifies a rendering
    or registration nuisance realization for that same science state.  Keeping
    these identities separate lets callers ask for all nuisance realizations of
    one science state, or compare different science states under a fixed
    nuisance realization.

    Key indexing conventions
    ------------------------
    - Catalog row indices address arrays stored on this object, for example
      ``catalog.fisher_scaled_deltas[catalog_index]``.
    - ``array_indices`` are global prepared-sample indices into the sharded
      image store.  Pass those values to :class:`dluxshera.datasets.ArrayShardReader`.
    - ``sample_ids`` are stable string identifiers from the prepared index and
      should be preferred when saving selections outside a Python session.

    Notes
    -----
    The catalog does not retain the original per-row JSON dictionaries.  Use
    :meth:`sample_metadata` for a compact notebook-friendly metadata record, or
    inspect ``manifest`` and ``vector_spaces`` for full prepared-artifact
    provenance.
    """

    root: Path
    artifact_id: str
    prepared_dataset_hash: str
    manifest: Mapping[str, Any]
    vector_spaces: Mapping[str, Any]
    sample_ids: np.ndarray
    array_indices: np.ndarray
    dataset_versions: np.ndarray
    render_state_ids: np.ndarray
    science_state_ids: np.ndarray
    nuisance_state_ids: np.ndarray
    science_group_ids: np.ndarray
    nuisance_group_ids: np.ndarray
    dataset_families: np.ndarray
    sample_roles: np.ndarray
    pair_ids: np.ndarray
    grid_i_indices: np.ndarray
    grid_j_indices: np.ndarray
    science_sequence_indices: np.ndarray
    joint_train_sequence_indices: np.ndarray
    nuisance_bank_indices: np.ndarray
    radial_bin_labels: np.ndarray
    fisher_scaled_deltas: np.ndarray
    physical_deltas: np.ndarray
    native_science_vectors: np.ndarray
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
            "dataset_versions": self.dataset_versions,
            "render_state_ids": self.render_state_ids,
            "science_state_ids": self.science_state_ids,
            "nuisance_state_ids": self.nuisance_state_ids,
            "science_group_ids": self.science_group_ids,
            "nuisance_group_ids": self.nuisance_group_ids,
            "dataset_families": self.dataset_families,
            "sample_roles": self.sample_roles,
            "pair_ids": self.pair_ids,
            "science_sequence_indices": self.science_sequence_indices,
            "joint_train_sequence_indices": self.joint_train_sequence_indices,
            "nuisance_bank_indices": self.nuisance_bank_indices,
            "radial_bin_labels": self.radial_bin_labels,
            "fisher_scaled_deltas": self.fisher_scaled_deltas,
            "physical_deltas": self.physical_deltas,
            "native_science_vectors": self.native_science_vectors,
        }.items():
            if len(values) != n:
                raise ValueError(f"{name} length {len(values)} does not match sample count {n}.")
        if self.fisher_scaled_deltas.ndim != 2:
            raise ValueError("fisher_scaled_deltas must be a 2D array.")
        if self.physical_deltas.shape != self.fisher_scaled_deltas.shape:
            raise ValueError("physical_deltas shape must match fisher_scaled_deltas.")
        if self.native_science_vectors.shape != self.fisher_scaled_deltas.shape:
            raise ValueError("native_science_vectors shape must match fisher_scaled_deltas.")
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
    def science_groups(self) -> tuple[str, ...]:
        """Return sorted science-group identifiers present in the catalog.

        Returns
        -------
        tuple of str
            Unique science-state/group identities.  Each value can be supplied
            to :meth:`indices_for_groups` as a ``science_groups`` filter.
        """
        return tuple(sorted(set(str(v) for v in self.science_group_ids)))

    @property
    def nuisance_groups(self) -> tuple[str, ...]:
        """Return sorted nuisance-group identifiers present in the catalog.

        Returns
        -------
        tuple of str
            Unique nuisance-state/group identities.  Each value can be supplied
            to :meth:`indices_for_groups` as a ``nuisance_groups`` filter.
        """
        return tuple(sorted(set(str(v) for v in self.nuisance_group_ids)))

    @property
    def science_group_count(self) -> int:
        """Return the number of unique science-state identities."""
        return int(len(self.science_groups))

    @property
    def nuisance_group_count(self) -> int:
        """Return the number of unique nuisance identities."""
        return int(len(self.nuisance_groups))

    def image_reader(self, *, cache_size: int = 4) -> ArrayShardReader:
        """Open a sharded image reader for this prepared dataset.

        Parameters
        ----------
        cache_size:
            Maximum number of shard memory maps retained by the reader.

        Returns
        -------
        ArrayShardReader
            Reader rooted at ``catalog.root``.  Use it as a context manager when
            possible so cached memory maps are released promptly.

        Examples
        --------
        Read the image corresponding to catalog row ``catalog_index``::

            catalog_index = 0
            array_index = int(catalog.array_indices[catalog_index])
            with catalog.image_reader() as reader:
                image = reader[array_index]

        Notes
        -----
        :class:`ArrayShardReader` expects global prepared-sample indices, not
        catalog row indices.  In current prepared artifacts those values often
        match, but callers should use ``catalog.array_indices`` for portable
        code.
        """
        return ArrayShardReader(self.root, cache_size=cache_size)

    def indices_for_groups(
        self,
        *,
        science_groups: Iterable[str] | None = None,
        nuisance_groups: Iterable[str] | None = None,
    ) -> np.ndarray:
        """Return catalog row indices matching science/nuisance filters.

        Parameters
        ----------
        science_groups:
            Optional science group ids to keep.  ``None`` keeps all science
            groups.
        nuisance_groups:
            Optional nuisance group ids to keep.  ``None`` keeps all nuisance
            groups.

        Returns
        -------
        numpy.ndarray
            One-dimensional ``int64`` array of catalog row indices.  Use these
            indices to select metadata arrays on ``SampleCatalog``.  Convert to
            image-store indices with :meth:`array_indices_for_groups` or
            ``catalog.array_indices[indices]`` before reading images.

        Examples
        --------
        All nuisance realizations for one science state::

            science_id = catalog.science_groups[0]
            rows = catalog.indices_for_groups(science_groups=[science_id])

        All science states rendered at one fixed nuisance realization::

            nuisance_id = catalog.nuisance_groups[0]
            rows = catalog.indices_for_groups(nuisance_groups=[nuisance_id])
        """
        mask = np.ones((self.sample_count,), dtype=bool)
        if science_groups is not None:
            allowed = {str(v) for v in science_groups}
            mask &= np.asarray([str(v) in allowed for v in self.science_group_ids])
        if nuisance_groups is not None:
            allowed = {str(v) for v in nuisance_groups}
            mask &= np.asarray([str(v) in allowed for v in self.nuisance_group_ids])
        return np.flatnonzero(mask).astype(np.int64)

    def array_indices_for_groups(
        self,
        *,
        science_groups: Iterable[str] | None = None,
        nuisance_groups: Iterable[str] | None = None,
    ) -> np.ndarray:
        """Return global image-store indices matching science/nuisance filters.

        This is a convenience wrapper around :meth:`indices_for_groups` for the
        common notebook workflow where the next operation is ``reader[index]``.

        Parameters
        ----------
        science_groups:
            Optional science group ids to keep.
        nuisance_groups:
            Optional nuisance group ids to keep.

        Returns
        -------
        numpy.ndarray
            One-dimensional ``int64`` array of global prepared-sample indices
            suitable for :class:`dluxshera.datasets.ArrayShardReader`.
        """
        rows = self.indices_for_groups(
            science_groups=science_groups,
            nuisance_groups=nuisance_groups,
        )
        return self.array_indices[rows].astype(np.int64, copy=False)

    def sample_index(self, sample_id: str) -> int:
        """Return the catalog row index for a stable prepared ``sample_id``.

        Parameters
        ----------
        sample_id:
            Stable sample identifier from ``catalog.sample_ids``.

        Returns
        -------
        int
            Catalog row index for selecting metadata arrays on this object.

        Raises
        ------
        KeyError
            If ``sample_id`` is not present in this catalog.
        """
        lookup = self.sample_id_to_index
        try:
            return lookup[str(sample_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown prepared sample_id {sample_id!r}.") from exc

    def sample_metadata(self, index: int) -> dict[str, Any]:
        """Return compact metadata for one catalog row.

        Parameters
        ----------
        index:
            Catalog row index, not a shard-reader index.

        Returns
        -------
        dict
            JSON-friendly metadata containing stable sample ids, science and
            nuisance group ids, dataset-family fields, vector values, and the
            corresponding ``array_index`` to use with an image reader.

        Raises
        ------
        IndexError
            If ``index`` is outside ``[0, sample_count)``.
        """
        i = int(index)
        if i < 0 or i >= self.sample_count:
            raise IndexError(f"catalog index {index} out of range for {self.sample_count} samples.")
        return {
            "sample_id": str(self.sample_ids[i]),
            "catalog_index": i,
            "array_index": int(self.array_indices[i]),
            "science_group_id": str(self.science_group_ids[i]),
            "nuisance_group_id": str(self.nuisance_group_ids[i]),
            "science_state_id": str(self.science_state_ids[i]),
            "nuisance_state_id": str(self.nuisance_state_ids[i]),
            "dataset_version": str(self.dataset_versions[i]),
            "dataset_family": str(self.dataset_families[i]),
            "sample_role": str(self.sample_roles[i]),
            "pair_id": str(self.pair_ids[i]),
            "grid_i_index": int(self.grid_i_indices[i]),
            "grid_j_index": int(self.grid_j_indices[i]),
            "science_sequence_index": int(self.science_sequence_indices[i]),
            "joint_train_sequence_index": int(self.joint_train_sequence_indices[i]),
            "nuisance_bank_index": int(self.nuisance_bank_indices[i]),
            "radial_bin": str(self.radial_bin_labels[i]),
            "fisher_scaled_delta": self.fisher_scaled_deltas[i].astype(float).tolist(),
            "physical_delta": self.physical_deltas[i].astype(float).tolist(),
            "native_science_vector": self.native_science_vectors[i].astype(float).tolist(),
            "nuisance_vector": self.nuisance_vectors[i].astype(float).tolist(),
            "nuisance_sigma_vector": self.nuisance_sigma_vectors[i].astype(float).tolist(),
        }

    def physical_from_z(self, z_delta: np.ndarray) -> np.ndarray:
        """Transform Fisher-scaled science deltas to native physical coordinates.

        Parameters
        ----------
        z_delta:
            Fisher-scaled science vector or array of vectors.  The last
            dimension must match ``catalog.science_dim``.

        Returns
        -------
        numpy.ndarray
            Native physical-coordinate deltas with the same shape as
            ``z_delta``.
        """
        return np.asarray(z_delta, dtype=np.float64) * self.fisher_sigmas

    def summary(self) -> dict[str, Any]:
        """Return compact human-readable dataset metadata.

        Returns
        -------
        dict
            Summary of prepared artifact identity, root path, sample/image
            shape, science/nuisance dimensionality, group counts, dataset
            families, sample roles, and grouping policies.
        """
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
            "dataset_versions": {
                str(value): int(np.count_nonzero(self.dataset_versions == value))
                for value in sorted(set(self.dataset_versions.tolist()))
                if str(value)
            },
            "sample_roles": {
                str(value): int(np.count_nonzero(self.sample_roles == value))
                for value in sorted(set(self.sample_roles.tolist()))
                if str(value)
            },
            "science_group_policy": self.science_group_policy,
            "nuisance_group_policy": self.nuisance_group_policy,
        }


def _prepared_dataset_hash(root: Path, manifest: Mapping[str, Any]) -> str:
    content_identity = manifest.get("content_identity", {})
    if isinstance(content_identity, Mapping) and content_identity.get("sha256"):
        if manifest.get("schema_version") == "shera_prepared_dataset/1" and manifest.get("artifact_id") == "PREP-V4-v1":
            from dluxshera.datasets.prepared_v4 import validate_prepared_v4_dataset_identity

            validate_prepared_v4_dataset_identity(root)
        return str(content_identity["sha256"])
    manifest_path = root / "manifest.json"
    vector_spaces_path = root / "vector_spaces.json"
    return _stable_digest(
        {
            "manifest_sha256": _sha256_file(manifest_path),
            "vector_spaces_sha256": _sha256_file(vector_spaces_path),
            "source_dataset": manifest.get("source_dataset", {}),
            "array_storage": manifest.get("array_storage", {}),
            "sample_count": manifest.get("array_storage", {}).get("sample_count"),
        }
    )


def load_sample_catalog(prepared_root: Path, *, artifact_id: str | None = None) -> SampleCatalog:
    """Load a prepared SHERA ML dataset catalog.

    Parameters
    ----------
    prepared_root:
        Directory containing ``manifest.json``, ``vector_spaces.json``,
        ``index.jsonl``, ``array_shards_manifest.json``, and ``shards/``.
    artifact_id:
        Optional fallback artifact id used only when the prepared manifest does
        not provide one.

    Returns
    -------
    SampleCatalog
        Compact, architecture-neutral metadata catalog for sample selection,
        grouping, image access, pair construction, and split validation.

    Raises
    ------
    FileNotFoundError
        If required prepared-dataset files are missing.
    ValueError
        If required vector fields are missing, malformed, non-finite, or
        inconsistent with ``vector_spaces.json``.

    Notes
    -----
    The loader streams ``index.jsonl`` but returns in-memory metadata arrays.
    Image arrays remain in sharded ``.npy`` files and are accessed lazily via
    :meth:`SampleCatalog.image_reader`.
    """
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
    dataset_versions: list[str] = []
    render_state_ids: list[str] = []
    science_state_ids: list[str] = []
    nuisance_state_ids: list[str] = []
    science_group_ids: list[str] = []
    nuisance_group_ids: list[str] = []
    dataset_families: list[str] = []
    sample_roles: list[str] = []
    pair_ids: list[str] = []
    grid_i_indices: list[int] = []
    grid_j_indices: list[int] = []
    science_sequence_indices: list[int] = []
    joint_train_sequence_indices: list[int] = []
    nuisance_bank_indices: list[int] = []
    radial_bin_labels: list[str] = []
    z_rows: list[np.ndarray] = []
    theta_rows: list[np.ndarray] = []
    native_rows: list[np.ndarray] = []
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
        native_science = _optional_vector(
            row,
            "native_science_vector",
            expected_dim=z.shape[0],
            dtype=np.float32,
        )
        if native_science is None:
            native_science = _optional_vector(
                row,
                "ordered_physical_science_vector",
                expected_dim=z.shape[0],
                dtype=np.float32,
            )
        if native_science is None:
            native_science = theta
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
        science_group = _science_group_id(row, theta)
        nuisance_group = _nuisance_group_id(row, nuisance)
        science_group_ids.append(science_group)
        nuisance_group_ids.append(nuisance_group)
        dataset_versions.append(_as_string(row.get("dataset_version")))
        render_state_ids.append(_as_string(row.get("render_state_id", sample_id)))
        science_state_ids.append(_as_string(row.get("science_state_id", science_group)))
        nuisance_state_ids.append(_as_string(row.get("nuisance_state_id", nuisance_group)))
        dataset_families.append(_as_string(row.get("dataset_family")))
        sample_roles.append(_as_string(row.get("sample_role", row.get("split_role"))))
        pair_ids.append(_as_string(row.get("pair_id")))
        grid_i_indices.append(_as_int(row.get("grid_i_index")))
        grid_j_indices.append(_as_int(row.get("grid_j_index")))
        science_sequence_indices.append(
            _optional_int(row, "science_plan_row_index", "global_sequence_index")
        )
        joint_train_sequence_indices.append(
            _optional_int(row, "joint_train_sequence_index", "v4_joint_train_sequence_index")
        )
        nuisance_bank_indices.append(_optional_int(row, "nuisance_bank_index"))
        radial_bin_labels.append(_as_string(row.get("radial_bin")))
        z_rows.append(z)
        theta_rows.append(theta)
        native_rows.append(native_science)
        nuisance_rows.append(nuisance)
        nuisance_sigma_rows.append(nuisance_sigma)

    if not sample_ids:
        raise ValueError(f"{index_path} contains no samples.")
    z_array = np.vstack(z_rows).astype(np.float32, copy=False)
    theta_array = np.vstack(theta_rows).astype(np.float32, copy=False)
    native_array = np.vstack(native_rows).astype(np.float32, copy=False)
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
    prepared_dataset_hash = _prepared_dataset_hash(root, manifest)
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
        artifact_id=str(
            manifest.get("artifact_id")
            or (manifest.get("prepared_dataset", {}) or {}).get("artifact_id")
            or artifact_id
            or PREPARED_ARTIFACT_ID
        ),
        prepared_dataset_hash=prepared_dataset_hash,
        manifest=manifest,
        vector_spaces=vector_spaces,
        sample_ids=np.asarray(sample_ids, dtype=object),
        array_indices=np.asarray(array_indices, dtype=np.int64),
        dataset_versions=np.asarray(dataset_versions, dtype=object),
        render_state_ids=np.asarray(render_state_ids, dtype=object),
        science_state_ids=np.asarray(science_state_ids, dtype=object),
        nuisance_state_ids=np.asarray(nuisance_state_ids, dtype=object),
        science_group_ids=np.asarray(science_group_ids, dtype=object),
        nuisance_group_ids=np.asarray(nuisance_group_ids, dtype=object),
        dataset_families=np.asarray(dataset_families, dtype=object),
        sample_roles=np.asarray(sample_roles, dtype=object),
        pair_ids=np.asarray(pair_ids, dtype=object),
        grid_i_indices=np.asarray(grid_i_indices, dtype=np.int32),
        grid_j_indices=np.asarray(grid_j_indices, dtype=np.int32),
        science_sequence_indices=np.asarray(science_sequence_indices, dtype=np.int64),
        joint_train_sequence_indices=np.asarray(joint_train_sequence_indices, dtype=np.int64),
        nuisance_bank_indices=np.asarray(nuisance_bank_indices, dtype=np.int32),
        radial_bin_labels=np.asarray(radial_bin_labels, dtype=object),
        fisher_scaled_deltas=z_array,
        physical_deltas=theta_array,
        native_science_vectors=native_array,
        nuisance_vectors=nuisance_array,
        nuisance_sigma_vectors=nuisance_sigma_array,
        sample_shape=sample_shape,
        parameter_labels=parameter_labels,
        nuisance_labels=nuisance_labels,
        fisher_sigmas=_fisher_sigmas(vector_spaces, z_array.shape[1]),
    )
