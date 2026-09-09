from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np
from astropy.io import fits

from .arrays import ArrayShardReader, ArrayShardStore, DEFAULT_TARGET_SHARD_BYTES
from .master_v4 import (
    FrozenMasterV4,
    PRODUCTION_IDENTITIES,
    FrozenV4Identities,
    render_output_paths,
    render_state_id,
)
from .schema import json_ready, read_json, read_jsonl, write_json, write_jsonl
from .validation import compare_arrays

__all__ = [
    "PREPARED_V4_ARTIFACT_ID",
    "V4_TRAIN_PREFIXES",
    "PreparedV4Summary",
    "prepare_shera_v4_dataset",
    "validate_prepared_v4_dataset_identity",
]

PREPARED_V4_ARTIFACT_ID = "PREP-V4-v1"
V4_TRAIN_PREFIXES = (4096, 8192, 16384, 32768, 65536)
PREPARED_SCHEMA_VERSION = "shera_prepared_dataset/1"
PREPARATION_STATE_SCHEMA_VERSION = "shera_prepared_v4_preparation_state/1"


@dataclass(frozen=True)
class PreparedV4Summary:
    """Summarize a V4 prepared working dataset."""

    outdir: Path
    plan_root: Path
    source_root: Path
    total_source_sample_count: int
    sample_count: int
    sample_shape: tuple[int, ...]
    source_dtypes: tuple[str, ...]
    storage_dtype: str
    shard_count: int
    validation_sample_count: int
    dry_run: bool = False
    existing_complete: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary."""
        return {
            "outdir": str(self.outdir),
            "plan_root": str(self.plan_root),
            "source_root": str(self.source_root),
            "total_source_sample_count": int(self.total_source_sample_count),
            "sample_count": int(self.sample_count),
            "sample_shape": list(self.sample_shape),
            "source_dtypes": list(self.source_dtypes),
            "storage_dtype": self.storage_dtype,
            "shard_count": int(self.shard_count),
            "validation_sample_count": int(self.validation_sample_count),
            "dry_run": bool(self.dry_run),
            "existing_complete": bool(self.existing_complete),
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_sha256(payload: Any) -> str:
    raw = json.dumps(json_ready(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _jsonl_row_count(path: Path) -> int:
    with Path(path).open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_info(repo_root: Path | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {}
    root = _repo_root() if repo_root is None else Path(repo_root)
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


def _portable_path(path: Path, *, base: Path) -> str:
    resolved_path = path.resolve()
    resolved_base = base.resolve()
    try:
        return os.path.relpath(resolved_path, start=resolved_base)
    except ValueError:
        return str(resolved_path)


def _read_fits_array(path: Path) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
        if data is None:
            raise ValueError(f"FITS file {path} does not contain primary image data.")
        arr = np.asarray(data).copy()
    if arr.ndim != 2:
        raise ValueError(f"V4 FITS image {path} must be 2D, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"V4 FITS image {path} contains non-finite values.")
    return arr


def _read_sidecar(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"V4 render sidecar {path} must contain a JSON object.")
    return dict(payload)


def _nuisance_rows(master: FrozenMasterV4) -> tuple[dict[str, Any], ...]:
    rows = [dict(row) for row in master.nuisance_bank["states"]]
    return tuple(sorted(rows, key=lambda row: int(row["bank_index"])))


def _science_plan_path(plan_root: Path, entry: Mapping[str, Any]) -> Path:
    plan_path = Path(str(entry.get("plan_path", "")))
    return plan_path if plan_path.is_absolute() else plan_root / plan_path


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path} line {line_number} is not a JSON object.")
            yield dict(payload)


def _v4_metadata_rows(
    *,
    plan_root: Path,
    source_root: Path,
    master: FrozenMasterV4,
    max_samples: int | None,
) -> Iterator[dict[str, Any]]:
    nuisance_rows = _nuisance_rows(master)
    nuisance_count = len(nuisance_rows)
    count = 0
    for entry in master.render_contract["families"]:
        family = str(entry["family"])
        split = str(entry["split_role"])
        science_start = int(entry["science_start_index"])
        plan_path = _science_plan_path(plan_root, entry)
        for plan_row_index, science in enumerate(_read_jsonl(plan_path)):
            science_global_index = science_start + plan_row_index
            for nuisance in nuisance_rows:
                if max_samples is not None and count >= int(max_samples):
                    return
                nuisance_index = int(nuisance["bank_index"])
                render_index = science_global_index * nuisance_count + nuisance_index
                fits_path, metadata_path = render_output_paths(
                    output_root=source_root,
                    dataset_family=family,
                    split_role=split,
                    render_index=render_index,
                )
                rel_fits = _portable_path(fits_path, base=source_root)
                rel_metadata = _portable_path(metadata_path, base=source_root)
                render_id = render_state_id(
                    str(science["science_state_id"]),
                    str(nuisance["nuisance_state_id"]),
                    str(master.render_contract["render_system_contract_hash"]),
                )
                fisher = [float(v) for v in science["ordered_fisher_scaled_delta"]]
                physical_delta = [
                    float(v)
                    for v in science.get(
                        "ordered_physical_delta_vector",
                        science["ordered_physical_science_vector"],
                    )
                ]
                native = [float(v) for v in science["ordered_physical_science_vector"]]
                row = {
                    "sample_id": render_id,
                    "source_sample_id": render_id,
                    "sample_index": count,
                    "render_index": render_index,
                    "science_global_index": science_global_index,
                    "science_plan_row_index": plan_row_index,
                    "nuisance_bank_index": nuisance_index,
                    "dataset_version": science.get("dataset_version", "shera_ml_master_v4"),
                    "dataset_family": family,
                    "sample_role": split,
                    "split_role": split,
                    "science_state_id": science["science_state_id"],
                    "nuisance_state_id": nuisance["nuisance_state_id"],
                    "render_state_id": render_id,
                    "source_fits_path": rel_fits,
                    "source_metadata_path": rel_metadata,
                    "source_fits_provenance": {
                        "path_kind": "source-root-relative",
                        "render_index": render_index,
                        "render_state_id": render_id,
                    },
                    "source_metadata_provenance": {
                        "path_kind": "source-root-relative",
                        "render_index": render_index,
                        "render_state_id": render_id,
                    },
                    "science_vector_space_id": science["science_vector_space_id"],
                    "nuisance_vector_space_id": nuisance["nuisance_vector_space_id"],
                    "render_system_contract_hash": master.render_contract[
                        "render_system_contract_hash"
                    ],
                    "parameter_space_identity": master.manifest.get(
                        "science_vector_space_id"
                    ),
                    "fisher_scale_identity": master.manifest.get(
                        "science_vector_space_id"
                    ),
                    "native_science_vector": native,
                    "ordered_physical_science_vector": native,
                    "physical_delta": physical_delta,
                    "fisher_scaled_delta": fisher,
                    "nuisance_vector": [
                        float(v) for v in nuisance["ordered_physical_nuisance_vector"]
                    ],
                    "nuisance_sigma_vector": [
                        float(v)
                        for v in nuisance.get("ordered_fisher_scaled_nuisance_vector", [])
                    ],
                    "radial_bin": science.get("radial_bin"),
                    "radial_bin_index": science.get("radial_bin_index"),
                    "requested_fisher_radius": science.get("requested_fisher_radius"),
                    "actual_fisher_radius": science.get("actual_fisher_radius"),
                    "joint_train_sequence_index": (
                        int(science.get("global_sequence_index", plan_row_index))
                        if family == "joint_full_v4" and split == "train"
                        else None
                    ),
                    "joint_train_prefixes": (
                        [
                            prefix
                            for prefix in V4_TRAIN_PREFIXES
                            if int(science.get("global_sequence_index", plan_row_index)) < prefix
                        ]
                        if family == "joint_full_v4" and split == "train"
                        else []
                    ),
                    "v4_sampling_provenance": {
                        key: science.get(key)
                        for key in (
                            "sampling_family_contract_identity",
                            "sampling_method",
                            "sampling_seed",
                            "scale_multiplier",
                            "scale_stratum",
                            "within_stratum_sequence_index",
                            "radial_bin",
                            "radial_bin_index",
                            "direction_sequence_index",
                            "attempt_count",
                        )
                        if key in science
                    },
                    "group_ids": {
                        "sample": render_id,
                        "science": science["science_state_id"],
                        "nuisance": nuisance["nuisance_state_id"],
                        "render": render_id,
                        "physical_delta_sha256": science["science_state_id"],
                    },
                }
                yield row
                count += 1


def _v4_array_rows(source_root: Path, metadata_rows: Iterable[Mapping[str, Any]]) -> Iterator[np.ndarray]:
    for row in metadata_rows:
        yield _read_fits_array(source_root / str(row["source_fits_path"]))


def _validate_source_sidecar(
    *,
    source_root: Path,
    row: Mapping[str, Any],
) -> None:
    fits_path = source_root / str(row["source_fits_path"])
    metadata_path = source_root / str(row["source_metadata_path"])
    if not fits_path.exists():
        raise FileNotFoundError(f"Missing expected V4 FITS render: {fits_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing expected V4 render sidecar: {metadata_path}")
    sidecar = _read_sidecar(metadata_path)
    checks = {
        "render_state_id": row["render_state_id"],
        "science_state_id": row["science_state_id"],
        "nuisance_state_id": row["nuisance_state_id"],
        "dataset_family": row["dataset_family"],
        "split_role": row["split_role"],
        "render_system_contract_hash": row["render_system_contract_hash"],
    }
    for key, expected in checks.items():
        if key not in sidecar:
            raise ValueError(
                f"V4 source sidecar {metadata_path} is missing mandatory identity "
                f"field {key!r}."
            )
        if str(sidecar.get(key)) != str(expected):
            raise ValueError(
                f"V4 source sidecar {metadata_path} {key} mismatch: "
                f"expected {expected!r}, got {sidecar.get(key)!r}."
            )


def _source_audit_indices(sample_count: int, *, mode: str, seed: int, samples: int) -> set[int]:
    mode = str(mode)
    if mode in {"off", "none"}:
        return set()
    if mode in {"full", "deep"}:
        return set(range(int(sample_count)))
    if mode not in {"sample", "fast"}:
        raise ValueError("source_audit must be one of 'off', 'sample', or 'full'.")
    count = min(int(sample_count), max(1, int(samples)))
    if count == int(sample_count):
        return set(range(int(sample_count)))
    rng = np.random.default_rng(int(seed))
    anchors = {0, int(sample_count) - 1}
    anchors.update(int(v) for v in rng.choice(int(sample_count), size=count, replace=False))
    return anchors


def _probe_first_array(source_root: Path, first_row: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    path = source_root / str(first_row["source_fits_path"])
    if not path.exists():
        raise FileNotFoundError(f"Missing source V4 FITS for first selected render: {path}")
    array = _read_fits_array(path)
    return array, {
        "policy": "first_render_index_in_v4_render_contract_order",
        "source_fits_path": first_row["source_fits_path"],
        "render_index": first_row["render_index"],
        "render_state_id": first_row["render_state_id"],
        "shape_dtype_validation": "provisional_probe_only",
    }


def _validation_indices(sample_count: int, validation_samples: int, seed: int) -> tuple[int, ...]:
    count = min(max(0, int(validation_samples)), int(sample_count))
    if count == 0:
        return ()
    if count == sample_count:
        return tuple(range(sample_count))
    rng = np.random.default_rng(int(seed))
    return tuple(sorted(int(v) for v in rng.choice(sample_count, size=count, replace=False)))


def _validate_precision(
    *,
    source_root: Path,
    outdir: Path,
    storage_dtype: np.dtype,
    metadata_rows: Iterable[Mapping[str, Any]],
    prepared_sample_count: int,
    validation_samples: int,
    seed: int,
) -> dict[str, Any]:
    sample_count = int(prepared_sample_count)
    wanted = set(_validation_indices(sample_count, validation_samples, seed))
    rows: list[dict[str, Any]] = []
    if not wanted:
        return {
            "schema_version": "array_precision_validation/1",
            "policy": "informational_metrics_only",
            "storage_dtype": str(storage_dtype),
            "sample_count": sample_count,
            "validation_sample_count": 0,
            "validation_seed": int(seed),
        }
    prepared_iter = iter(read_jsonl(outdir / "index.jsonl"))
    with ArrayShardReader(outdir, cache_size=4) as reader:
        for index, source_row in enumerate(metadata_rows):
            try:
                index_row = next(prepared_iter)
            except StopIteration as exc:
                raise ValueError("Prepared index ended before validation stream.") from exc
            if index not in wanted:
                continue
            source = _read_fits_array(source_root / str(source_row["source_fits_path"]))
            readback = reader.get(index)
            expected_cast = source.astype(storage_dtype, copy=False)
            comparison = compare_arrays(source, readback)
            rows.append(
                {
                    "sample_index": index,
                    "sample_id": source_row["sample_id"],
                    "prepared_index_sample_id": index_row.get("sample_id"),
                    "source_sample_id_matches_index": (
                        source_row["sample_id"] == index_row.get("source_sample_id")
                    ),
                    "source_fits_path": source_row["source_fits_path"],
                    "readback_dtype": str(readback.dtype),
                    "readback_matches_expected_cast": bool(
                        np.array_equal(readback, expected_cast)
                    ),
                    **comparison.to_dict(),
                }
            )
    finite = [row for row in rows if row.get("max_abs_error") is not None]
    relative_l2 = [
        float(row["relative_l2_error"])
        for row in finite
        if row.get("relative_l2_error") is not None
    ]
    summary = {
        "schema_version": "array_precision_validation/1",
        "policy": "informational_metrics_only",
        "storage_dtype": str(storage_dtype),
        "sample_count": sample_count,
        "validation_sample_count": len(rows),
        "validation_seed": int(seed),
        "max_abs_error": None if not finite else max(float(row["max_abs_error"]) for row in finite),
        "max_relative_l2_error": None if not relative_l2 else max(relative_l2),
        "mean_rms_error": None if not finite else float(np.mean([float(row["rms_error"]) for row in finite])),
    }
    validation_dir = outdir / "validation"
    write_json(validation_dir / "precision_summary.json", summary)
    write_jsonl(validation_dir / "precision_samples.jsonl", rows)
    return summary


def _shard_manifest_scientific_payload(shard_manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(shard_manifest)
    extra = dict(payload.get("extra", {}))
    extra.pop("source_dataset_root", None)
    extra.pop("plan_root", None)
    payload["extra"] = extra
    return payload


def _prepared_content_tree(
    *,
    outdir: Path,
    manifest: Mapping[str, Any],
    shard_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    shards = []
    for shard in shard_manifest.get("shards", []):
        row = dict(shard)
        path = outdir / str(row["path"])
        row["file_size_bytes"] = int(path.stat().st_size)
        row["sha256"] = row.get("sha256") or _sha256_file(path)
        shards.append(row)
    shard_payload = _shard_manifest_scientific_payload(
        {**dict(shard_manifest), "shards": shards}
    )
    source = dict(manifest.get("source_dataset", {}))
    source.pop("root", None)
    return {
        "schema_version": "prepared-v4-content-tree/1",
        "index": {
            "path": "index.jsonl",
            "sha256": _sha256_file(outdir / "index.jsonl"),
            "row_count": _jsonl_row_count(outdir / "index.jsonl"),
        },
        "vector_spaces": {
            "path": "vector_spaces.json",
            "sha256": _sha256_file(outdir / "vector_spaces.json"),
        },
        "array_shards_manifest": {
            "path": "array_shards_manifest.json",
            "scientific_sha256": _stable_sha256(shard_payload),
            "shard_count": int(shard_manifest.get("shard_count", len(shards))),
            "shards": shards,
        },
        "source_dataset": {
            key: source.get(key)
            for key in (
                "dataset_version",
                "master_scientific_content_hash",
                "render_contract_hash",
                "render_system_contract_hash",
                "nuisance_bank_hash",
                "science_vector_space_id",
                "nuisance_vector_space_id",
                "prepared_sample_count",
                "total_source_sample_count",
                "selection_policy",
            )
        },
        "array_representation": {
            "storage_dtype": manifest.get("array_storage", {}).get("storage_dtype"),
            "sample_shape": manifest.get("array_storage", {}).get("sample_shape"),
            "sample_count": manifest.get("array_storage", {}).get("sample_count"),
            "target_shard_bytes": manifest.get("array_storage", {}).get("target_shard_bytes"),
            "max_samples_per_shard": manifest.get("array_storage", {}).get("max_samples_per_shard"),
            "dtype_conversion": manifest.get("array_storage", {}).get("dtype_conversion"),
        },
    }


def _content_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    stable = dict(payload)
    stable.pop("prepared_at", None)
    stable.pop("content_identity", None)
    source = dict(stable.get("source_dataset", {}))
    source.pop("root", None)
    stable["source_dataset"] = source
    return {
        "algorithm": "sha256/json-canonical/prepared-v4-v1",
        "sha256": _stable_sha256(stable),
        "excludes": ["prepared_at", "content_identity", "source_dataset.root"],
    }


def validate_prepared_v4_dataset_identity(root: Path, *, deep: bool = False) -> dict[str, Any]:
    """Validate persisted V4 prepared content identity.

    Normal validation recomputes small-file hashes and the recorded shard
    manifest identity.  ``deep=True`` also rehashes every shard file, which is
    appropriate after staging or before production training but can be more
    expensive for the full V4 dataset.
    """
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema_version") != PREPARED_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported prepared V4 schema {manifest.get('schema_version')!r}."
        )
    shard_manifest = read_json(root / "array_shards_manifest.json")
    tree = _prepared_content_tree(outdir=root, manifest=manifest, shard_manifest=shard_manifest)
    recorded_tree = manifest.get("content_tree")
    if recorded_tree != tree:
        raise ValueError("Prepared V4 content_tree does not match materialized index/shard metadata.")
    expected_identity = manifest.get("content_identity", {}).get("sha256")
    actual_identity = _content_identity(manifest)["sha256"]
    if expected_identity and str(expected_identity) != str(actual_identity):
        raise ValueError(
            "Prepared V4 content_identity.sha256 does not match manifest content "
            f"({expected_identity} != {actual_identity})."
        )
    if deep:
        for shard in tree["array_shards_manifest"]["shards"]:
            path = root / str(shard["path"])
            actual = _sha256_file(path)
            if actual != str(shard["sha256"]):
                raise ValueError(
                    f"Prepared V4 shard hash mismatch for {path}: "
                    f"expected {shard['sha256']}, got {actual}."
                )
    return tree


def _complete_summary(outdir: Path, *, plan_root: Path, source_root: Path) -> PreparedV4Summary:
    manifest = read_json(outdir / "manifest.json")
    storage = manifest.get("array_storage", {})
    return PreparedV4Summary(
        outdir=outdir,
        plan_root=plan_root,
        source_root=source_root,
        total_source_sample_count=int(
            storage.get(
                "total_source_sample_count",
                manifest.get("source_dataset", {}).get("total_source_sample_count", 0),
            )
        ),
        sample_count=int(storage.get("sample_count", 0)),
        sample_shape=tuple(int(v) for v in storage.get("sample_shape", ())),
        source_dtypes=tuple(str(v) for v in storage.get("source_dtypes", ())),
        storage_dtype=str(storage.get("storage_dtype", "")),
        shard_count=int(storage.get("shard_count", 0)),
        validation_sample_count=int(
            manifest.get("validation", {}).get("summary", {}).get(
                "validation_sample_count",
                0,
            )
        ),
        existing_complete=True,
    )


def _validate_complete_compatible(
    *,
    outdir: Path,
    master: FrozenMasterV4,
    selected: int,
    total: int,
    storage_dtype: np.dtype,
    sample_shape: tuple[int, ...],
    target_shard_bytes: int,
    max_samples_per_shard: int | None,
) -> None:
    manifest = read_json(outdir / "manifest.json")
    source = dict(manifest.get("source_dataset", {}))
    storage = dict(manifest.get("array_storage", {}))
    expected_source = {
        "dataset_version": master.manifest["dataset_version"],
        "master_scientific_content_hash": master.manifest["master_scientific_content_hash"],
        "render_contract_hash": master.manifest["render_contract_hash"],
        "render_system_contract_hash": master.manifest["render_system_contract_hash"],
        "nuisance_bank_hash": master.manifest["selected_nuisance_bank_hash"],
        "science_vector_space_id": master.manifest["science_vector_space_id"],
        "nuisance_vector_space_id": master.manifest["nuisance_vector_space_id"],
        "prepared_sample_count": selected,
        "total_source_sample_count": total,
    }
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            raise ValueError(
                f"Existing prepared V4 artifact is incompatible for {key}: "
                f"expected {expected!r}, got {source.get(key)!r}."
            )
    expected_storage = {
        "storage_dtype": str(storage_dtype),
        "sample_shape": list(sample_shape),
        "sample_count": selected,
        "total_source_sample_count": total,
        "target_shard_bytes": int(target_shard_bytes),
        "max_samples_per_shard": max_samples_per_shard,
    }
    for key, expected in expected_storage.items():
        if storage.get(key) != expected:
            raise ValueError(
                f"Existing prepared V4 artifact is incompatible for array_storage.{key}: "
                f"expected {expected!r}, got {storage.get(key)!r}."
            )
    validate_prepared_v4_dataset_identity(outdir)


def _shard_file_ok(
    path: Path,
    *,
    sample_count: int,
    sample_shape: tuple[int, ...],
    storage_dtype: np.dtype,
) -> bool:
    if not path.exists():
        return False
    try:
        array = np.load(path, mmap_mode="r")
    except Exception:
        return False
    return (
        tuple(int(v) for v in array.shape) == (int(sample_count), *sample_shape)
        and str(array.dtype) == str(storage_dtype)
    )


def _preparation_request_identity(
    *,
    master: FrozenMasterV4,
    selected: int,
    total: int,
    storage_dtype: np.dtype,
    sample_shape: tuple[int, ...],
    target_shard_bytes: int,
    max_samples_per_shard: int | None,
    samples_per_shard: int,
) -> dict[str, Any]:
    return {
        "schema_version": "shera_prepared_v4_request_identity/1",
        "dataset_version": master.manifest["dataset_version"],
        "master_scientific_content_hash": master.manifest["master_scientific_content_hash"],
        "render_contract_hash": master.manifest["render_contract_hash"],
        "render_system_contract_hash": master.manifest["render_system_contract_hash"],
        "nuisance_bank_hash": master.manifest["selected_nuisance_bank_hash"],
        "science_vector_space_id": master.manifest["science_vector_space_id"],
        "nuisance_vector_space_id": master.manifest["nuisance_vector_space_id"],
        "source_corpus": {
            "render_count": int(total),
            "families": master.render_contract["families"],
        },
        "selection": {
            "prepared_sample_count": int(selected),
            "total_source_sample_count": int(total),
        },
        "array_storage": {
            "storage_dtype": str(storage_dtype),
            "sample_shape": list(sample_shape),
            "target_shard_bytes": int(target_shard_bytes),
            "max_samples_per_shard": max_samples_per_shard,
            "samples_per_shard": int(samples_per_shard),
        },
    }


def _load_preparation_state(
    path: Path,
    *,
    request_identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    state = read_json(path)
    if state.get("schema_version") != PREPARATION_STATE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported V4 preparation state schema {state.get('schema_version')!r}."
        )
    previous = state.get("request_identity")
    if dict(previous or {}) != dict(request_identity):
        raise ValueError(
            "Existing V4 preparation_state.json is incompatible with this resume request."
        )
    return dict(state)


def _write_preparation_state(path: Path, state: Mapping[str, Any]) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    write_json(tmp_path, dict(state))
    tmp_path.replace(path)


def _verified_completed_shards(
    *,
    outdir: Path,
    state: Mapping[str, Any] | None,
    sample_shape: tuple[int, ...],
    storage_dtype: np.dtype,
) -> dict[str, dict[str, Any]]:
    if state is None:
        return {}
    verified: dict[str, dict[str, Any]] = {}
    for shard in state.get("completed_shards", []):
        row = dict(shard)
        shard_id = str(row.get("shard_id"))
        path = outdir / str(row.get("path", ""))
        sample_count = int(row.get("sample_count", 0))
        if not _shard_file_ok(
            path,
            sample_count=sample_count,
            sample_shape=sample_shape,
            storage_dtype=storage_dtype,
        ):
            continue
        actual_hash = _sha256_file(path)
        if actual_hash != str(row.get("sha256")):
            continue
        verified[shard_id] = row
    return verified


def _write_prepared_index_row(
    handle: Any,
    *,
    row: Mapping[str, Any],
    global_index: int,
    shard_id: str,
    shard_path: str,
    shard_offset: int,
    source_dtype: str,
    storage_dtype: np.dtype,
    sample_shape: tuple[int, ...],
) -> None:
    payload = dict(row)
    payload.update(
        {
            "sample_index": int(payload.get("sample_index", global_index)),
            "array_index": int(global_index),
            "shard_id": shard_id,
            "shard_path": shard_path,
            "shard_offset": int(shard_offset),
            "source_dtype": str(source_dtype),
            "storage_dtype": str(storage_dtype),
            "sample_shape": list(sample_shape),
        }
    )
    handle.write(json.dumps(json_ready(payload), sort_keys=True) + "\n")


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.unlink(missing_ok=True)
    with tmp_path.open("wb") as handle:
        np.save(handle, array)
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)


def _write_v4_shards(
    *,
    source_root: Path,
    outdir: Path,
    metadata_rows: Iterable[Mapping[str, Any]],
    selected: int,
    sample_shape: tuple[int, ...],
    source_dtype: str,
    storage_dtype: np.dtype,
    target_shard_bytes: int,
    max_samples_per_shard: int | None,
    samples_per_shard: int,
    resume: bool,
    extra_manifest: Mapping[str, Any],
    source_audit_mode: str,
    source_audit_samples: int,
    seed: int,
    request_identity: Mapping[str, Any],
) -> dict[str, Any]:
    shards_dir = outdir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    index_path = outdir / "index.jsonl"
    manifest_path = outdir / "array_shards_manifest.json"
    state_path = outdir / "preparation_state.json"
    index_tmp = index_path.with_name(index_path.name + ".tmp")
    index_tmp.unlink(missing_ok=True)
    manifest_path.unlink(missing_ok=True)
    state = _load_preparation_state(
        state_path,
        request_identity=request_identity,
    ) if resume else None
    verified_shards = _verified_completed_shards(
        outdir=outdir,
        state=state,
        sample_shape=sample_shape,
        storage_dtype=storage_dtype,
    )
    completed_by_id: dict[str, dict[str, Any]] = dict(verified_shards)
    state_payload: dict[str, Any] = {
        "schema_version": PREPARATION_STATE_SCHEMA_VERSION,
        "request_identity": json_ready(dict(request_identity)),
        "completed_shard_ids": sorted(completed_by_id),
        "completed_shards": [completed_by_id[key] for key in sorted(completed_by_id)],
        "completed_shard_hashes": {
            key: completed_by_id[key].get("sha256") for key in sorted(completed_by_id)
        },
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    audit_indices = _source_audit_indices(
        selected,
        mode=source_audit_mode,
        seed=seed,
        samples=source_audit_samples,
    )
    records: list[dict[str, Any]] = []
    all_source_dtypes = {source_dtype}
    with index_tmp.open("w", encoding="utf-8") as index_handle:
        chunk: list[np.ndarray] = []
        chunk_start = 0
        chunk_shard_index = 0
        current_reuse = False
        for global_index, row in enumerate(metadata_rows):
            if global_index >= int(selected):
                break
            shard_index = global_index // int(samples_per_shard)
            shard_start = shard_index * int(samples_per_shard)
            shard_count = min(int(samples_per_shard), int(selected) - shard_start)
            shard_id = f"shard_{shard_index:05d}"
            rel_path = f"shards/{shard_id}.npy"
            path = outdir / rel_path
            if shard_index != chunk_shard_index:
                raise AssertionError("V4 metadata ordering produced a non-contiguous shard stream.")
            if global_index == shard_start:
                current_reuse = bool(
                    resume
                    and shard_id in verified_shards
                    and str(verified_shards[shard_id].get("path")) == rel_path
                    and int(verified_shards[shard_id].get("sample_count", -1)) == shard_count
                )
            if global_index in audit_indices:
                _validate_source_sidecar(source_root=source_root, row=row)
            if not current_reuse:
                array = _read_fits_array(source_root / str(row["source_fits_path"]))
                all_source_dtypes.add(str(array.dtype))
                chunk.append(array.astype(storage_dtype, copy=False))
            _write_prepared_index_row(
                index_handle,
                row=row,
                global_index=global_index,
                shard_id=shard_id,
                shard_path=rel_path,
                shard_offset=global_index - shard_start,
                source_dtype=source_dtype,
                storage_dtype=storage_dtype,
                sample_shape=sample_shape,
            )
            at_end = (
                (global_index + 1 == int(selected))
                or ((global_index + 1) % int(samples_per_shard) == 0)
            )
            if at_end:
                if not current_reuse:
                    if path.exists():
                        path.unlink()
                    _atomic_save_npy(path, np.stack(chunk, axis=0))
                shard_row = {
                    "shard_id": shard_id,
                    "path": rel_path,
                    "start_index": chunk_start,
                    "stop_index": chunk_start + shard_count,
                    "sample_count": shard_count,
                    "sample_shape": list(sample_shape),
                    "source_dtypes": sorted(all_source_dtypes),
                    "storage_dtype": str(storage_dtype),
                    "file_size_bytes": int(path.stat().st_size),
                    "sha256": _sha256_file(path),
                }
                records.append(shard_row)
                completed_by_id[shard_id] = shard_row
                state_payload.update(
                    {
                        "completed_shard_ids": sorted(completed_by_id),
                        "completed_shards": [
                            completed_by_id[key] for key in sorted(completed_by_id)
                        ],
                        "completed_shard_hashes": {
                            key: completed_by_id[key].get("sha256")
                            for key in sorted(completed_by_id)
                        },
                        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    }
                )
                _write_preparation_state(state_path, state_payload)
                chunk = []
                chunk_start += shard_count
                chunk_shard_index += 1
    index_tmp.replace(index_path)
    manifest = {
        "schema_version": "array_shard_store/1",
        "storage_format": "npy",
        "manifest_path": "array_shards_manifest.json",
        "index_path": "index.jsonl",
        "shards_dir": "shards",
        "sample_count": int(selected),
        "sample_shape": list(sample_shape),
        "source_dtypes": sorted(all_source_dtypes),
        "storage_dtype": str(storage_dtype),
        "target_shard_bytes": int(target_shard_bytes),
        "max_samples_per_shard": max_samples_per_shard,
        "samples_per_shard": int(samples_per_shard),
        "shard_count": len(records),
        "shards": records,
        "extra": json_ready(dict(extra_manifest)),
    }
    write_json(manifest_path, manifest)
    return manifest


def prepare_shera_v4_dataset(
    *,
    source_root: Path,
    plan_root: Path,
    outdir: Path,
    dtype: str | np.dtype = "float32",
    target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
    max_samples_per_shard: int | None = 1024,
    validation_samples: int = 32,
    seed: int = 0,
    max_samples: int | None = None,
    source_audit: str = "sample",
    source_audit_samples: int = 256,
    overwrite: bool = False,
    resume: bool = False,
    dry_run: bool = False,
    expected: FrozenV4Identities | None = PRODUCTION_IDENTITIES,
) -> PreparedV4Summary:
    """Prepare frozen V4 raw FITS renders into the reusable ML shard format.

    Rows remain sample-centric: one prepared row corresponds to one rendered
    science/nuisance image, not to a Siamese pair. The ordering follows the
    frozen compact render contract, preserving V4 science splits and the joint
    training-prefix row order from the state plan.
    """
    source_root = Path(source_root).resolve()
    plan_root = Path(plan_root).resolve()
    outdir = Path(outdir).resolve()
    storage_dtype = np.dtype(dtype)
    if str(storage_dtype) not in {"float32", "float64"}:
        raise ValueError("dtype must be float32 or float64 for the V4 preparation workflow.")
    if max_samples is not None and int(max_samples) < 1:
        raise ValueError("max_samples must be >= 1 when provided.")
    if max_samples_per_shard is not None and int(max_samples_per_shard) < 1:
        raise ValueError("max_samples_per_shard must be >= 1 when provided.")

    master = FrozenMasterV4(plan_root, expected=expected)
    total = int(master.render_count)
    selected = total if max_samples is None else min(total, int(max_samples))
    first_row = next(
        _v4_metadata_rows(
            plan_root=plan_root,
            source_root=source_root,
            master=master,
            max_samples=selected,
        )
    )
    first_array, probe_info = _probe_first_array(source_root, first_row)
    sample_shape = tuple(int(v) for v in first_array.shape)
    source_dtypes = (str(first_array.dtype),)
    samples_per_shard = max(
        1,
        int(target_shard_bytes)
        // (int(np.prod(sample_shape, dtype=np.int64)) * int(storage_dtype.itemsize)),
    )
    if max_samples_per_shard is not None:
        samples_per_shard = min(samples_per_shard, int(max_samples_per_shard))
    expected_shards = int(np.ceil(selected / samples_per_shard))

    if dry_run:
        return PreparedV4Summary(
            outdir=outdir,
            plan_root=plan_root,
            source_root=source_root,
            total_source_sample_count=total,
            sample_count=selected,
            sample_shape=sample_shape,
            source_dtypes=source_dtypes,
            storage_dtype=str(storage_dtype),
            shard_count=expected_shards,
            validation_sample_count=min(selected, max(0, int(validation_samples))),
            dry_run=True,
        )

    if outdir.exists() and any(outdir.iterdir()):
        complete = all(
            (outdir / name).exists()
            for name in ("manifest.json", "vector_spaces.json", "index.jsonl", "array_shards_manifest.json")
        )
        if resume and complete:
            _validate_complete_compatible(
                outdir=outdir,
                master=master,
                selected=selected,
                total=total,
                storage_dtype=storage_dtype,
                sample_shape=sample_shape,
                target_shard_bytes=target_shard_bytes,
                max_samples_per_shard=max_samples_per_shard,
            )
            return _complete_summary(outdir, plan_root=plan_root, source_root=source_root)
        if not overwrite and not resume:
            raise FileExistsError(
                f"Output directory {outdir} already exists and is non-empty. "
                "Use overwrite=True to replace it, or resume=True to continue from verified shard boundaries."
            )
        if overwrite:
            shutil.rmtree(outdir)
        else:
            for name in (
                "manifest.json",
                "vector_spaces.json",
                "array_shards_manifest.json",
                "index.jsonl",
            ):
                (outdir / name).unlink(missing_ok=True)
            shutil.rmtree(outdir / "validation", ignore_errors=True)
            for tmp_path in sorted(outdir.glob("*.tmp")):
                tmp_path.unlink(missing_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    vector_spaces = dict(master.vector_spaces)
    spaces = dict(vector_spaces.get("spaces", {}))
    if "fisher_scaled_delta" not in spaces and "fisher_scaled_science_delta" in spaces:
        spaces["fisher_scaled_delta"] = spaces["fisher_scaled_science_delta"]
    if "physical_delta" not in spaces and "physical_science_state" in spaces:
        spaces["physical_delta"] = spaces["physical_science_state"]
    vector_spaces["spaces"] = spaces
    transforms = dict(vector_spaces.get("transforms", {}))
    if "fisher_diagonal_scale" not in transforms:
        dim = len(spaces.get("fisher_scaled_delta", {}).get("components", []))
        transforms["fisher_diagonal_scale"] = {
            "type": "DiagonalScaleTransform",
            "scales": [1.0] * dim,
            "forward_mode": "divide",
        }
    vector_spaces["transforms"] = transforms
    vector_spaces["prepared_aliases"] = {
        "physical_delta": "ordered_physical_delta_vector",
        "native_science_vector": "ordered_physical_science_vector",
        "fisher_scaled_delta": "ordered_fisher_scaled_delta",
    }
    write_json(outdir / "vector_spaces.json", vector_spaces)
    for name in (
        "freeze_manifest.json",
        "render_contract.json",
        "render_system_contract.json",
        "nuisance_bank.json",
    ):
        write_json(
            outdir / "provenance" / name,
            {
                "path": _portable_path(plan_root / name, base=plan_root),
                "sha256": _sha256_file(plan_root / name),
                "payload": read_json(plan_root / name),
            },
        )

    extra_manifest = {
        "source_dataset_root": _portable_path(source_root, base=outdir),
        "plan_root": _portable_path(plan_root, base=outdir),
        "selection_policy": {
            "type": "render_contract_prefix" if max_samples is not None else "all",
            "requested_max_samples": max_samples,
            "prepared_sample_count": selected,
            "total_source_sample_count": total,
        },
        "dtype_conversion": {
            "source_dtypes": list(source_dtypes),
            "storage_dtype": str(storage_dtype),
            "lossless": all(np.dtype(dtype) == storage_dtype for dtype in source_dtypes),
        },
        "source_sidecar_audit": {
            "mode": str(source_audit),
            "sample_count": int(source_audit_samples),
            "seed": int(seed),
            "checked_indices": sorted(
                _source_audit_indices(
                    selected,
                    mode=str(source_audit),
                    seed=int(seed),
                    samples=int(source_audit_samples),
                )
            )[:1024],
        },
        "restart_policy": {
            "resume_reuses_verified_complete_shards": True,
            "current_incomplete_shard": "regenerated_atomically",
            "index": "regenerated_from_deterministic_v4_order",
        },
    }
    request_identity = _preparation_request_identity(
        master=master,
        selected=selected,
        total=total,
        storage_dtype=storage_dtype,
        sample_shape=sample_shape,
        target_shard_bytes=target_shard_bytes,
        max_samples_per_shard=max_samples_per_shard,
        samples_per_shard=samples_per_shard,
    )
    shard_manifest = _write_v4_shards(
        source_root=source_root,
        outdir=outdir,
        metadata_rows=_v4_metadata_rows(
            plan_root=plan_root,
            source_root=source_root,
            master=master,
            max_samples=selected,
        ),
        selected=selected,
        sample_shape=sample_shape,
        source_dtype=str(first_array.dtype),
        storage_dtype=storage_dtype,
        target_shard_bytes=target_shard_bytes,
        max_samples_per_shard=max_samples_per_shard,
        samples_per_shard=samples_per_shard,
        resume=resume,
        extra_manifest=extra_manifest,
        source_audit_mode=str(source_audit),
        source_audit_samples=int(source_audit_samples),
        seed=int(seed),
        request_identity=request_identity,
    )
    observed_source_dtypes = tuple(str(dtype) for dtype in shard_manifest["source_dtypes"])
    validation_summary = _validate_precision(
        source_root=source_root,
        outdir=outdir,
        storage_dtype=storage_dtype,
        metadata_rows=_v4_metadata_rows(
            plan_root=plan_root,
            source_root=source_root,
            master=master,
            max_samples=selected,
        ),
        prepared_sample_count=selected,
        validation_samples=validation_samples,
        seed=seed,
    )
    dtype_conversion = {
        "source_dtypes": list(observed_source_dtypes),
        "storage_dtype": str(storage_dtype),
        "lossless": all(np.dtype(dtype) == storage_dtype for dtype in observed_source_dtypes),
    }
    shard_manifest["extra"]["dtype_conversion"] = dtype_conversion
    write_json(outdir / "array_shards_manifest.json", shard_manifest)
    prepared_manifest = {
        "schema_version": PREPARED_SCHEMA_VERSION,
        "artifact_id": PREPARED_V4_ARTIFACT_ID,
        "prepared_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_dataset": {
            "root": _portable_path(source_root, base=outdir),
            "dataset_version": master.manifest["dataset_version"],
            "master_scientific_content_hash": master.manifest[
                "master_scientific_content_hash"
            ],
            "render_contract_hash": master.manifest["render_contract_hash"],
            "render_system_contract_hash": master.manifest["render_system_contract_hash"],
            "nuisance_bank_hash": master.manifest["selected_nuisance_bank_hash"],
            "science_vector_space_id": master.manifest["science_vector_space_id"],
            "nuisance_vector_space_id": master.manifest["nuisance_vector_space_id"],
            "total_source_sample_count": total,
            "prepared_sample_count": selected,
            "selection_policy": shard_manifest["extra"]["selection_policy"],
        },
        "canonical_policy": {
            "canonical_artifacts": ["source FITS", "source JSON sidecars", "V4 state plans"],
            "prepared_arrays": "derived reproducible working representation",
            "mutates_source_dataset": False,
        },
        "v4_science_splits": {
            "policy": "preserve frozen raw V4 state-plan split_role",
            "no_random_resplit": True,
            "families": master.render_contract["families"],
            "joint_train_prefixes": list(V4_TRAIN_PREFIXES),
        },
        "array_storage": {
            "manifest": "array_shards_manifest.json",
            "index": "index.jsonl",
            "shards_dir": "shards",
            "storage_dtype": str(storage_dtype),
            "source_dtypes": list(observed_source_dtypes),
            "sample_shape": list(sample_shape),
            "sample_count": selected,
            "total_source_sample_count": total,
            "shard_count": shard_manifest["shard_count"],
            "target_shard_bytes": int(target_shard_bytes),
            "max_samples_per_shard": max_samples_per_shard,
            "source_probe": probe_info,
            "dtype_conversion": dtype_conversion,
        },
        "vector_spaces": "vector_spaces.json",
        "validation": {
            "precision_summary": "validation/precision_summary.json",
            "precision_samples": "validation/precision_samples.jsonl",
            "summary": validation_summary,
        },
        "index_format": {
            "path": "index.jsonl",
            "format": "jsonl",
            "row_semantics": "one rendered source sample per row",
            "parquet_required": False,
        },
        "tool": {
            "module": "dluxshera.datasets.prepared_v4.prepare_shera_v4_dataset",
            "repo": _git_info(),
        },
    }
    prepared_manifest["content_tree"] = _prepared_content_tree(
        outdir=outdir,
        manifest=prepared_manifest,
        shard_manifest=shard_manifest,
    )
    prepared_manifest["content_identity"] = _content_identity(prepared_manifest)
    write_json(outdir / "manifest.json", prepared_manifest)
    validate_prepared_v4_dataset_identity(outdir)
    return PreparedV4Summary(
        outdir=outdir,
        plan_root=plan_root,
        source_root=source_root,
        total_source_sample_count=total,
        sample_count=selected,
        sample_shape=sample_shape,
        source_dtypes=observed_source_dtypes,
        storage_dtype=str(storage_dtype),
        shard_count=int(shard_manifest["shard_count"]),
        validation_sample_count=int(validation_summary["validation_sample_count"]),
    )
