from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from astropy.io import fits

from .arrays import ArrayShardReader, ArrayShardStore, DEFAULT_TARGET_SHARD_BYTES
from .schema import (
    VectorComponentSpec,
    VectorSpaceSpec,
    json_ready,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from .transforms import DiagonalScaleTransform
from .validation import compare_arrays

__all__ = [
    "PreparedV3Summary",
    "build_shera_v3_vector_spaces",
    "prepare_shera_v3_dataset",
    "shera_v3_index_rows",
]

PREPARED_SCHEMA_VERSION = "shera_prepared_dataset/1"


@dataclass(frozen=True)
class PreparedV3Summary:
    """Summarize a V3 prepared working dataset."""

    outdir: Path
    total_source_sample_count: int
    sample_count: int
    sample_shape: tuple[int, ...]
    source_dtypes: tuple[str, ...]
    storage_dtype: str
    shard_count: int
    validation_sample_count: int
    source_probe_policy: str
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary."""
        return {
            "outdir": str(self.outdir),
            "total_source_sample_count": self.total_source_sample_count,
            "sample_count": self.sample_count,
            "sample_shape": list(self.sample_shape),
            "source_dtypes": list(self.source_dtypes),
            "storage_dtype": self.storage_dtype,
            "shard_count": self.shard_count,
            "validation_sample_count": self.validation_sample_count,
            "source_probe_policy": self.source_probe_policy,
            "dry_run": self.dry_run,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_info(repo_root: Path | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {}
    root = Path(repo_root) if repo_root is not None else _repo_root()
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
            if key == "dirty":
                info[key] = bool(result.stdout.strip())
            else:
                info[key] = result.stdout.strip() or None
    return info


def _portable_path(path: Path, *, base: Path) -> str:
    resolved_path = path.resolve()
    resolved_base = base.resolve()
    try:
        return os.path.relpath(resolved_path, start=resolved_base)
    except ValueError:
        return str(resolved_path)


def _source_artifacts(source_root: Path) -> tuple[Path, Path, Path]:
    manifest_path = source_root / "manifest.json"
    parameter_space_path = source_root / "parameter_space.json"
    samples_path = source_root / "samples.jsonl"
    missing = [
        str(path.relative_to(source_root))
        for path in (manifest_path, parameter_space_path, samples_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Source root {source_root} is missing required V3 artifacts: {', '.join(missing)}."
        )
    return manifest_path, parameter_space_path, samples_path


@dataclass(frozen=True)
class _SourcePlan:
    total_source_sample_count: int
    prepared_sample_count: int
    selection_policy: Mapping[str, Any]
    completeness: Mapping[str, Any]
    nuisance_labels: tuple[str, ...]
    first_selected_row: Mapping[str, Any]


def _iter_samples(source_root: Path) -> Iterable[dict[str, Any]]:
    _, _, samples_path = _source_artifacts(source_root)
    yield from read_jsonl(samples_path)


def _iter_selected_samples(
    source_root: Path,
    *,
    max_samples: int | None = None,
) -> Iterable[dict[str, Any]]:
    for row_index, row in enumerate(_iter_samples(source_root)):
        if max_samples is not None and row_index >= int(max_samples):
            break
        yield row


def _mapping_field(
    row: Mapping[str, Any],
    key: str,
    *,
    row_index: int,
) -> Mapping[str, Any]:
    value = row.get(key, {}) or {}
    if not isinstance(value, Mapping):
        raise ValueError(f"samples.jsonl row {row_index + 1} field {key!r} must be a JSON object.")
    return value


def _manifest_nuisance_labels(source_manifest: Mapping[str, Any]) -> tuple[str, ...]:
    labels: list[str] = []
    seen: set[str] = set()
    nuisance_cfg = source_manifest.get("nuisance_config", {})
    if isinstance(nuisance_cfg, Mapping):
        for key in nuisance_cfg.get("keys", []) or []:
            label = str(key)
            if label not in seen:
                seen.add(label)
                labels.append(label)
    return tuple(labels)


def _source_plan(
    *,
    source_root: Path,
    source_manifest: Mapping[str, Any],
    parameter_labels: Iterable[str],
    max_samples: int | None,
    allow_incomplete_source: bool,
) -> _SourcePlan:
    if max_samples is not None and int(max_samples) < 1:
        raise ValueError("max_samples must be >= 1 when provided.")
    parameter_label_set = {str(label) for label in parameter_labels}
    nuisance_labels = list(_manifest_nuisance_labels(source_manifest))
    seen_nuisance_labels = set(nuisance_labels)
    first_selected_row: dict[str, Any] | None = None
    total = 0
    with tempfile.TemporaryDirectory(prefix="dluxshera_sample_ids_") as tmpdir:
        db_path = Path(tmpdir) / "sample_ids.sqlite"
        connection = sqlite3.connect(db_path)
        try:
            connection.execute("CREATE TABLE sample_ids (sample_id TEXT PRIMARY KEY)")
            cursor = connection.cursor()
            for row_index, row in enumerate(_iter_samples(source_root)):
                sample_id = row.get("sample_id")
                if sample_id in (None, ""):
                    raise ValueError(
                        f"samples.jsonl row {row_index + 1} is missing sample_id."
                    )
                sample_id_text = str(sample_id)
                try:
                    cursor.execute(
                        "INSERT INTO sample_ids(sample_id) VALUES (?)",
                        (sample_id_text,),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError(
                        f"samples.jsonl contains duplicate sample_id {sample_id_text!r}."
                    ) from exc
                if "sample_index" in row and int(row["sample_index"]) != row_index:
                    raise ValueError(
                        f"samples.jsonl row {row_index + 1} has sample_index={row['sample_index']}, "
                        f"expected {row_index}."
                    )
                theta_delta = _mapping_field(row, "theta_delta", row_index=row_index)
                unknown_delta_keys = sorted(
                    str(key)
                    for key in theta_delta
                    if str(key) not in parameter_label_set
                )
                if unknown_delta_keys:
                    raise ValueError(
                        "samples.jsonl row "
                        f"{row_index + 1} theta_delta contains keys not present in "
                        "parameter_space.json: "
                        + ", ".join(unknown_delta_keys)
                    )
                for field in (
                    "registration_nuisance_values",
                    "registration_nuisance_sigma_values",
                ):
                    for key in _mapping_field(row, field, row_index=row_index):
                        label = str(key)
                        if label not in seen_nuisance_labels:
                            seen_nuisance_labels.add(label)
                            nuisance_labels.append(label)
                if max_samples is None or row_index < int(max_samples):
                    if first_selected_row is None:
                        first_selected_row = dict(row)
                total += 1
        finally:
            connection.close()

    if total == 0:
        raise ValueError(f"{source_root / 'samples.jsonl'} contains no rendered samples.")
    if max_samples is not None:
        if int(max_samples) > total:
            raise ValueError(
                f"max_samples={max_samples} exceeds source sample count {total}."
            )

    rendered_sample_count = source_manifest.get("rendered_sample_count")
    if rendered_sample_count is not None and int(rendered_sample_count) != total:
        raise ValueError(
            "Source manifest rendered_sample_count does not match samples.jsonl row count "
            f"({rendered_sample_count} vs {total})."
        )
    render_complete = source_manifest.get("render_complete")
    if render_complete is False and not allow_incomplete_source:
        raise ValueError(
            "Source manifest reports render_complete=false; pass allow_incomplete_source=True "
            "only for explicitly partial development datasets."
        )
    next_sample_index = source_manifest.get("next_sample_index")
    if (
        next_sample_index is not None
        and rendered_sample_count is not None
        and int(next_sample_index) != int(rendered_sample_count)
    ):
        raise ValueError(
            "Source manifest next_sample_index does not match rendered_sample_count "
            f"({next_sample_index} vs {rendered_sample_count})."
        )
    render_target = source_manifest.get("render_target_sample_count")
    if (
        render_complete is True
        and render_target is not None
        and rendered_sample_count is not None
        and int(rendered_sample_count) < int(render_target)
    ):
        raise ValueError(
            "Source manifest reports render_complete=true but rendered_sample_count is "
            f"less than render_target_sample_count ({rendered_sample_count} vs {render_target})."
        )
    prepared_count = total if max_samples is None else int(max_samples)
    if first_selected_row is None:
        raise ValueError("Cannot prepare an empty selected source sample set.")
    return _SourcePlan(
        total_source_sample_count=total,
        prepared_sample_count=prepared_count,
        selection_policy={
            "type": "prefix",
            "requested_max_samples": max_samples,
            "prepared_sample_count": prepared_count,
            "total_source_sample_count": total,
            "description": "first N source samples in samples.jsonl order"
            if max_samples is not None
            else "all source samples",
        },
        completeness={
            "allow_incomplete_source": bool(allow_incomplete_source),
            "manifest_fields_present": {
                key: key in source_manifest
                for key in (
                    "rendered_sample_count",
                    "render_complete",
                    "next_sample_index",
                    "render_target_sample_count",
                )
            },
            "rendered_sample_count": rendered_sample_count,
            "render_complete": render_complete,
            "next_sample_index": next_sample_index,
            "render_target_sample_count": render_target,
            "samples_jsonl_row_count": total,
        },
        nuisance_labels=tuple(nuisance_labels),
        first_selected_row=first_selected_row,
    )


def _read_fits_array(path: Path) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
        if data is None:
            raise ValueError(f"FITS file {path} does not contain primary image data.")
        return np.asarray(data).copy()


def _sample_fits_path(source_root: Path, row: Mapping[str, Any]) -> Path:
    raw = row.get("fits_path")
    if raw in (None, ""):
        raise ValueError(f"Sample {row.get('sample_id', '<unknown>')} does not define fits_path.")
    path = Path(str(raw))
    return path if path.is_absolute() else source_root / path


def _parameter_records(source_root: Path) -> list[dict[str, Any]]:
    _, parameter_space_path, _ = _source_artifacts(source_root)
    payload = read_json(parameter_space_path)
    records = payload.get("parameters") if isinstance(payload, Mapping) else None
    if not isinstance(records, list) or not records:
        raise ValueError(f"{parameter_space_path} must contain a non-empty 'parameters' list.")
    return [dict(record) for record in records]


def build_shera_v3_vector_spaces(
    parameter_records: Iterable[Mapping[str, Any]],
    *,
    nuisance_labels: Iterable[str] = (),
) -> tuple[
    VectorSpaceSpec,
    VectorSpaceSpec,
    VectorSpaceSpec | None,
    VectorSpaceSpec | None,
    DiagonalScaleTransform,
]:
    """Build V3 physical, Fisher-scaled, and nuisance vector metadata."""
    physical_components: list[VectorComponentSpec] = []
    fisher_components: list[VectorComponentSpec] = []
    scales: list[float] = []
    for idx, record in enumerate(parameter_records):
        label = str(record["label"])
        sigma = float(record["parameter_sigma"])
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(f"parameter_sigma for {label!r} must be finite and > 0.")
        common = {
            "label": label,
            "index": idx,
            "source_key": record.get("base_key"),
            "component_index": record.get("component_index"),
            "display_label": record.get("display_label"),
            "unit": record.get("units"),
            "group": record.get("group"),
            "reference_value": record.get("nominal_value"),
            "metadata": {
                "sweep_source_key": record.get("sweep_source_key"),
                "sweep_config": record.get("sweep_config"),
                "noll_index": record.get("noll_index"),
                "min_abs_delta": record.get("min_abs_delta"),
                "max_abs_delta": record.get("max_abs_delta"),
            },
        }
        physical_components.append(
            VectorComponentSpec(
                **common,
                scale={
                    "kind": "fisher_diagonal_sigma",
                    "value": sigma,
                    "forward_usage": "delta_divided_by_sigma",
                },
            )
        )
        fisher_components.append(
            VectorComponentSpec(
                **{key: value for key, value in common.items() if key != "unit"},
                unit="fisher_sigma",
                scale={"kind": "dimensionless", "source_sigma": sigma},
            )
        )
        scales.append(sigma)

    physical_space = VectorSpaceSpec(
        name="shera_v3_physical_delta",
        components=tuple(physical_components),
        description="Scalarized V3 SHERA physical parameter deltas in canonical parameter_space.json order.",
        metadata={"source": "parameter_space.json"},
    )
    fisher_space = VectorSpaceSpec(
        name="shera_v3_fisher_scaled_delta",
        components=tuple(fisher_components),
        description="Physical deltas divided by the V3 Fisher-diagonal parameter sigma.",
        metadata={"scale_source": "parameter_space.json parameter_sigma"},
    )
    nuisance_components = tuple(
        VectorComponentSpec(label=str(label), index=idx, group="registration")
        for idx, label in enumerate(nuisance_labels)
    )
    nuisance_space = (
        None
        if not nuisance_components
        else VectorSpaceSpec(
            name="shera_v3_registration_nuisance",
            components=nuisance_components,
            description="Registration nuisance offsets recorded by V3 sample metadata.",
            metadata={"source": "samples.jsonl registration_nuisance_values"},
        )
    )
    nuisance_sigma_space = (
        None
        if not nuisance_components
        else VectorSpaceSpec(
            name="shera_v3_registration_nuisance_sigma",
            components=tuple(
                VectorComponentSpec(label=str(label), index=idx, group="registration")
                for idx, label in enumerate(nuisance_labels)
            ),
            description=(
                "Registration nuisance draw offsets in V3 generator sigma coordinates."
            ),
            metadata={"source": "samples.jsonl registration_nuisance_sigma_values"},
        )
    )
    transform = DiagonalScaleTransform(
        source_space=physical_space,
        destination_space=fisher_space,
        name="shera_v3_fisher_diagonal_scale",
        scales=tuple(scales),
        forward_mode="divide",
        metadata={"scale_source": "parameter_space.json parameter_sigma"},
    )
    return physical_space, fisher_space, nuisance_space, nuisance_sigma_space, transform


def _vector_from_mapping(
    values: Mapping[str, Any],
    space: VectorSpaceSpec,
    *,
    default: float = 0.0,
) -> list[float]:
    out = []
    for component in space.components:
        value = values.get(component.label, default)
        out.append(float(value))
    return out


def _metadata_rows(
    *,
    rows: Iterable[Mapping[str, Any]],
    physical_space: VectorSpaceSpec,
    fisher_transform: DiagonalScaleTransform,
    nuisance_space: VectorSpaceSpec | None,
    nuisance_sigma_space: VectorSpaceSpec | None,
) -> Iterable[dict[str, Any]]:
    for row in rows:
        physical_vector = np.asarray(
            _vector_from_mapping(row.get("theta_delta", {}) or {}, physical_space),
            dtype=float,
        )
        fisher_vector = fisher_transform.forward(physical_vector)
        out = {
            "sample_id": row.get("sample_id"),
            "source_sample_id": row.get("sample_id"),
            "source_sample_index": row.get("sample_index"),
            "source_fits_path": str(row.get("fits_path")),
            "source_metadata_path": row.get("metadata_path"),
            "dataset_family": row.get("dataset_family"),
            "sample_role": row.get("sample_role"),
            "pair_id": row.get("pair_id"),
            "pair_label_i": row.get("pair_label_i"),
            "pair_label_j": row.get("pair_label_j"),
            "grid_i_index": row.get("grid_i_index"),
            "grid_j_index": row.get("grid_j_index"),
            "grid_i_sigma": row.get("grid_i_sigma"),
            "grid_j_sigma": row.get("grid_j_sigma"),
            "delta_i": row.get("delta_i"),
            "delta_j": row.get("delta_j"),
            "delta_units": row.get("delta_units"),
            "nuisance_id": row.get("nuisance_id"),
            "seed": row.get("seed"),
            "registration_nuisance_values": row.get("registration_nuisance_values"),
            "registration_nuisance_sigma_values": row.get("registration_nuisance_sigma_values"),
            "skipped_nuisance_keys": row.get("skipped_nuisance_keys"),
            "split": row.get("split"),
            "active_count": row.get("active_count"),
            "active_labels": row.get("active_labels"),
            "active_mask": row.get("active_mask"),
            "theta_sigma": row.get("theta_sigma"),
            "theta_nominal": row.get("theta_nominal"),
            "theta_applied": row.get("theta_applied"),
            "controlled_labels": row.get("controlled_labels"),
            "physical_delta": physical_vector.tolist(),
            "fisher_scaled_delta": fisher_vector.tolist(),
            "group_ids": {
                "sample": row.get("sample_id"),
                "pair": row.get("pair_id"),
                "nuisance": row.get("nuisance_id"),
                "physical_delta_sha256": hashlib.sha256(
                    json.dumps(
                        json_ready(physical_vector.tolist()),
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            },
        }
        if nuisance_space is not None:
            out["nuisance_vector"] = _vector_from_mapping(
                row.get("registration_nuisance_values", {}) or {},
                nuisance_space,
            )
        if nuisance_sigma_space is not None:
            out["nuisance_sigma_vector"] = _vector_from_mapping(
                row.get("registration_nuisance_sigma_values", {}) or {},
                nuisance_sigma_space,
            )
        yield out


def shera_v3_index_rows(source_root: Path) -> Iterable[dict[str, Any]]:
    """Yield V3 source sample records in source order."""
    yield from _iter_samples(Path(source_root))


def _array_rows(
    *,
    source_root: Path,
    rows: Iterable[Mapping[str, Any]],
    first_array: np.ndarray | None = None,
) -> Iterable[np.ndarray]:
    for index, row in enumerate(rows):
        if index == 0 and first_array is not None:
            yield first_array
            continue
        yield _read_fits_array(_sample_fits_path(source_root, row))


def _probe_source_array(
    source_root: Path,
    first_row: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    path = _sample_fits_path(source_root, first_row)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing source FITS file for sample {first_row.get('sample_id')}: {path}"
        )
    array = _read_fits_array(path)
    return array, {
        "policy": "first_selected_source_sample",
        "source_sample_id": first_row.get("sample_id"),
        "source_fits_path": str(first_row.get("fits_path")),
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


def _same_numeric_dtype(source_dtype: str, storage_dtype: np.dtype) -> bool:
    source = np.dtype(source_dtype)
    storage = np.dtype(storage_dtype)
    return source.kind == storage.kind and source.itemsize == storage.itemsize


def _validate_precision(
    *,
    source_root: Path,
    outdir: Path,
    storage_dtype: np.dtype,
    max_samples: int | None,
    prepared_sample_count: int,
    validation_samples: int,
    seed: int,
) -> dict[str, Any]:
    sample_count = int(prepared_sample_count)
    indices = set(_validation_indices(sample_count, validation_samples, seed))
    validation_rows: list[dict[str, Any]] = []
    source_iter = iter(_iter_selected_samples(source_root, max_samples=max_samples))
    prepared_iter = iter(read_jsonl(outdir / "index.jsonl"))
    with ArrayShardReader(outdir, cache_size=4) as reader:
        for index in range(sample_count):
            try:
                row = next(source_iter)
            except StopIteration as exc:
                raise ValueError(
                    f"Selected source stream ended before prepared sample count {sample_count}."
                ) from exc
            try:
                index_row = next(prepared_iter)
            except StopIteration as exc:
                raise ValueError(
                    f"Prepared index row count is less than prepared sample count {sample_count}."
                ) from exc
            if index not in indices:
                continue
            source = _read_fits_array(_sample_fits_path(source_root, row))
            readback = reader.get(index)
            expected_cast = source.astype(storage_dtype, copy=False)
            comparison = compare_arrays(source, readback)
            validation_rows.append(
                {
                    "sample_index": index,
                    "sample_id": row.get("sample_id"),
                    "prepared_index_sample_id": index_row.get("sample_id"),
                    "source_sample_id_matches_index": (
                        row.get("sample_id") == index_row.get("source_sample_id")
                    ),
                    "source_fits_path": row.get("fits_path"),
                    "readback_dtype": str(readback.dtype),
                    "readback_matches_expected_cast": bool(
                        np.array_equal(readback, expected_cast)
                    ),
                    **comparison.to_dict(),
                }
            )
    try:
        next(prepared_iter)
    except StopIteration:
        pass
    else:
        raise ValueError(
            f"Prepared index row count exceeds prepared sample count {sample_count}."
        )
    finite_metrics = [row for row in validation_rows if row.get("max_abs_error") is not None]
    relative_l2_values = [
        float(row["relative_l2_error"])
        for row in finite_metrics
        if row.get("relative_l2_error") is not None
    ]
    summary = {
        "schema_version": "array_precision_validation/1",
        "policy": "informational_metrics_only",
        "storage_dtype": str(storage_dtype),
        "sample_count": sample_count,
        "validation_sample_count": len(validation_rows),
        "validation_seed": int(seed),
        "max_abs_error": (
            None
            if not finite_metrics
            else max(float(row["max_abs_error"]) for row in finite_metrics)
        ),
        "max_relative_l2_error": None if not relative_l2_values else max(relative_l2_values),
        "mean_rms_error": (
            None
            if not finite_metrics
            else float(np.mean([float(row["rms_error"]) for row in finite_metrics]))
        ),
    }
    validation_dir = outdir / "validation"
    write_json(validation_dir / "precision_summary.json", summary)
    write_jsonl(validation_dir / "precision_samples.jsonl", validation_rows)
    return summary


def _ensure_output_dir(outdir: Path, *, overwrite: bool, dry_run: bool) -> None:
    if dry_run:
        return
    if outdir.exists() and any(outdir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory {outdir} already exists and is non-empty. Use overwrite=True to replace it."
            )
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)


def prepare_shera_v3_dataset(
    *,
    source_root: Path,
    outdir: Path,
    dtype: str | np.dtype = "float32",
    target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
    max_samples_per_shard: int | None = None,
    validation_samples: int = 16,
    seed: int = 0,
    max_samples: int | None = None,
    allow_incomplete_source: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
) -> PreparedV3Summary:
    """Prepare a V3 canonical FITS dataset into sharded working arrays.

    The original V3 FITS files and metadata remain canonical.  This workflow
    streams source FITS images into derived ``.npy`` shards, writes a
    sample-centric JSONL index, records vector-space metadata, and emits
    deterministic precision metrics for dtype conversion.
    """
    source_root = Path(source_root).resolve()
    outdir = Path(outdir).resolve()
    storage_dtype = np.dtype(dtype)
    if str(storage_dtype) not in {"float32", "float64"}:
        raise ValueError("dtype must be float32 or float64 for the V3 preparation workflow.")
    if int(target_shard_bytes) < 1:
        raise ValueError("target_shard_bytes must be >= 1.")
    if max_samples_per_shard is not None and int(max_samples_per_shard) < 1:
        raise ValueError("max_samples_per_shard must be >= 1 when provided.")
    manifest_path, parameter_space_path, samples_path = _source_artifacts(source_root)
    source_manifest = read_json(manifest_path)
    if not isinstance(source_manifest, Mapping):
        raise ValueError(f"{manifest_path} must contain a JSON object.")
    parameter_records = _parameter_records(source_root)
    source_plan = _source_plan(
        source_root=source_root,
        source_manifest=source_manifest,
        parameter_labels=(record["label"] for record in parameter_records),
        max_samples=max_samples,
        allow_incomplete_source=allow_incomplete_source,
    )
    (
        physical_space,
        fisher_space,
        nuisance_space,
        nuisance_sigma_space,
        fisher_transform,
    ) = build_shera_v3_vector_spaces(
        parameter_records,
        nuisance_labels=source_plan.nuisance_labels,
    )
    first_array, probe_info = _probe_source_array(
        source_root,
        source_plan.first_selected_row,
    )
    sample_shape = tuple(int(v) for v in first_array.shape)
    probed_source_dtypes = (str(first_array.dtype),)
    samples_per_shard = max(
        1,
        int(target_shard_bytes)
        // (int(np.prod(sample_shape, dtype=np.int64)) * int(storage_dtype.itemsize)),
    )
    if max_samples_per_shard is not None:
        samples_per_shard = min(samples_per_shard, int(max_samples_per_shard))
    expected_shards = int(np.ceil(source_plan.prepared_sample_count / samples_per_shard))

    if dry_run:
        return PreparedV3Summary(
            outdir=outdir,
            total_source_sample_count=source_plan.total_source_sample_count,
            sample_count=source_plan.prepared_sample_count,
            sample_shape=sample_shape,
            source_dtypes=probed_source_dtypes,
            storage_dtype=str(storage_dtype),
            shard_count=expected_shards,
            validation_sample_count=min(
                source_plan.prepared_sample_count,
                max(0, int(validation_samples)),
            ),
            source_probe_policy=str(probe_info["policy"]),
            dry_run=True,
        )

    _ensure_output_dir(outdir, overwrite=overwrite, dry_run=dry_run)
    write_json(
        outdir / "vector_spaces.json",
        {
            "schema_version": "shera_v3_vector_spaces/1",
            "spaces": {
                "physical_delta": physical_space.to_dict(),
                "fisher_scaled_delta": fisher_space.to_dict(),
                "registration_nuisance": (
                    None if nuisance_space is None else nuisance_space.to_dict()
                ),
                "registration_nuisance_sigma": (
                    None if nuisance_sigma_space is None else nuisance_sigma_space.to_dict()
                ),
            },
            "transforms": {"fisher_diagonal_scale": fisher_transform.to_dict()},
        },
    )
    write_json(
        outdir / "provenance" / "source_manifest.json",
        {
            "path": _portable_path(manifest_path, base=source_root),
            "sha256": _sha256_file(manifest_path),
            "payload": source_manifest,
        },
    )
    write_json(
        outdir / "provenance" / "source_parameter_space.json",
        {
            "path": _portable_path(parameter_space_path, base=source_root),
            "sha256": _sha256_file(parameter_space_path),
            "payload": {"parameters": parameter_records},
        },
    )
    write_json(
        outdir / "provenance" / "source_samples.json",
        {
            "path": _portable_path(samples_path, base=source_root),
            "sha256": _sha256_file(samples_path),
            "row_count": source_plan.total_source_sample_count,
        },
    )
    store = ArrayShardStore(
        outdir,
        storage_dtype=storage_dtype,
        target_shard_bytes=target_shard_bytes,
        max_samples_per_shard=max_samples_per_shard,
        manifest_name="array_shards_manifest.json",
        index_name="index.jsonl",
    )
    shard_manifest = store.write(
        _array_rows(
            source_root=source_root,
            rows=_iter_selected_samples(source_root, max_samples=max_samples),
            first_array=first_array,
        ),
        sample_metadata=_metadata_rows(
            rows=_iter_selected_samples(source_root, max_samples=max_samples),
            physical_space=physical_space,
            fisher_transform=fisher_transform,
            nuisance_space=nuisance_space,
            nuisance_sigma_space=nuisance_sigma_space,
        ),
        extra_manifest={
            "source_dataset_root": _portable_path(source_root, base=outdir),
            "source_samples_path": _portable_path(samples_path, base=source_root),
            "selection_policy": source_plan.selection_policy,
            "dtype_conversion": {
                "source_dtypes": list(probed_source_dtypes),
                "storage_dtype": str(storage_dtype),
                "lossless": all(
                    _same_numeric_dtype(dtype, storage_dtype)
                    for dtype in probed_source_dtypes
                ),
            },
        },
    )
    validation_summary = _validate_precision(
        source_root=source_root,
        outdir=outdir,
        storage_dtype=storage_dtype,
        max_samples=max_samples,
        prepared_sample_count=source_plan.prepared_sample_count,
        validation_samples=validation_samples,
        seed=seed,
    )
    observed_source_dtypes = tuple(str(dtype) for dtype in shard_manifest["source_dtypes"])
    dtype_conversion = {
        "source_dtypes": list(observed_source_dtypes),
        "storage_dtype": str(storage_dtype),
        "lossless": all(
            _same_numeric_dtype(dtype, storage_dtype) for dtype in observed_source_dtypes
        ),
    }
    shard_manifest["extra"]["dtype_conversion"] = dtype_conversion
    write_json(outdir / "array_shards_manifest.json", shard_manifest)
    prepared_manifest = {
        "schema_version": PREPARED_SCHEMA_VERSION,
        "prepared_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_dataset": {
            "root": _portable_path(source_root, base=outdir),
            "manifest_path": _portable_path(manifest_path, base=source_root),
            "manifest_sha256": _sha256_file(manifest_path),
            "parameter_space_path": _portable_path(parameter_space_path, base=source_root),
            "parameter_space_sha256": _sha256_file(parameter_space_path),
            "samples_path": _portable_path(samples_path, base=source_root),
            "samples_sha256": _sha256_file(samples_path),
            "total_source_sample_count": source_plan.total_source_sample_count,
            "prepared_sample_count": source_plan.prepared_sample_count,
            "selection_policy": source_plan.selection_policy,
            "completeness": source_plan.completeness,
            "schema_version": source_manifest.get("schema_version")
            if isinstance(source_manifest, Mapping)
            else None,
            "generator": source_manifest.get("generator")
            if isinstance(source_manifest, Mapping)
            else None,
        },
        "canonical_policy": {
            "canonical_artifacts": ["source FITS", "manifest.json", "parameter_space.json", "samples.jsonl"],
            "prepared_arrays": "derived reproducible working representation",
            "mutates_source_dataset": False,
        },
        "array_storage": {
            "manifest": "array_shards_manifest.json",
            "index": "index.jsonl",
            "shards_dir": "shards",
            "storage_dtype": str(storage_dtype),
            "source_dtypes": list(observed_source_dtypes),
            "sample_shape": list(sample_shape),
            "sample_count": source_plan.prepared_sample_count,
            "total_source_sample_count": source_plan.total_source_sample_count,
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
        "source_metadata": {
            "dataset_family_counts": source_manifest.get("dataset_family_counts")
            if isinstance(source_manifest, Mapping)
            else None,
            "parameter_space_summary": source_manifest.get("parameter_space_summary")
            if isinstance(source_manifest, Mapping)
            else None,
            "rendered_sample_count": source_manifest.get("rendered_sample_count")
            if isinstance(source_manifest, Mapping)
            else None,
            "render_complete": source_manifest.get("render_complete"),
            "next_sample_index": source_manifest.get("next_sample_index"),
            "render_target_sample_count": source_manifest.get("render_target_sample_count"),
        },
        "tool": {
            "module": "dluxshera.datasets.shera.prepare_shera_v3_dataset",
            "repo": _git_info(),
        },
    }
    write_json(outdir / "manifest.json", prepared_manifest)
    return PreparedV3Summary(
        outdir=outdir,
        total_source_sample_count=source_plan.total_source_sample_count,
        sample_count=source_plan.prepared_sample_count,
        sample_shape=sample_shape,
        source_dtypes=observed_source_dtypes,
        storage_dtype=str(storage_dtype),
        shard_count=int(shard_manifest["shard_count"]),
        validation_sample_count=int(validation_summary["validation_sample_count"]),
        source_probe_policy=str(probe_info["policy"]),
    )


def main(argv: list[str] | None = None) -> int:
    """Run the V3 preparation CLI."""
    parser = argparse.ArgumentParser(description="Prepare a SHERA V3 FITS dataset into sharded working arrays.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--target-shard-bytes", type=int, default=DEFAULT_TARGET_SHARD_BYTES)
    parser.add_argument("--max-samples-per-shard", type=int, default=None)
    parser.add_argument("--validation-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--allow-incomplete-source", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args(argv)

    summary = prepare_shera_v3_dataset(
        source_root=args.source_root,
        outdir=args.outdir,
        dtype=args.dtype,
        target_shard_bytes=args.target_shard_bytes,
        max_samples_per_shard=args.max_samples_per_shard,
        validation_samples=args.validation_samples,
        seed=args.seed,
        max_samples=args.max_samples,
        allow_incomplete_source=args.allow_incomplete_source,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(json.dumps(json_ready(summary.to_dict()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
