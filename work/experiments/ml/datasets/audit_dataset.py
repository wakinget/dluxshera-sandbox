from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

AUDIT_SCHEMA_VERSION = "dluxshera_dataset_audit/1"
PERCENTILES = (0.0, 1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 100.0)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {line_number} is not valid JSON: {exc}") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"{path} line {line_number} is not a JSON object.")
            yield dict(row)


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _status(value: Any, status: str, source: str | None = None) -> dict[str, Any]:
    out = {"status": status, "value": _json_ready(value)}
    if source is not None:
        out["source"] = source
    return out


def _field(row: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    value: Any = row
    for key in dotted.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def _stable_id(values: Iterable[float]) -> str:
    payload = json.dumps([float(v) for v in values], separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _vector_from_mapping(values: Any, labels: tuple[str, ...]) -> np.ndarray | None:
    if not labels:
        return None
    if not isinstance(values, Mapping):
        values = {}
    return np.asarray([float(values.get(label, 0.0)) for label in labels], dtype=np.float64)


def _vector_from_value(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 1:
        return None
    return arr


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"status": "unavailable", "count": 0}
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"status": "confirmed", "count": int(arr.size), "finite_count": 0}
    return {
        "status": "confirmed",
        "count": int(arr.size),
        "finite_count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "percentiles": {str(p): float(np.percentile(finite, p)) for p in PERCENTILES},
    }


def _vector_summary(matrix: list[np.ndarray], labels: tuple[str, ...]) -> list[dict[str, Any]]:
    if not matrix or not labels:
        return []
    arr = np.vstack(matrix)
    rows: list[dict[str, Any]] = []
    for idx, label in enumerate(labels):
        col = arr[:, idx]
        finite = col[np.isfinite(col)]
        rows.append(
            {
                "index": idx,
                "label": label,
                "count": int(col.size),
                "finite_count": int(finite.size),
                "nonfinite_count": int(col.size - finite.size),
                "min": None if finite.size == 0 else float(np.min(finite)),
                "max": None if finite.size == 0 else float(np.max(finite)),
                "mean": None if finite.size == 0 else float(np.mean(finite)),
                "std": None if finite.size == 0 else float(np.std(finite)),
                "p01": None if finite.size == 0 else float(np.percentile(finite, 1.0)),
                "p50": None if finite.size == 0 else float(np.percentile(finite, 50.0)),
                "p99": None if finite.size == 0 else float(np.percentile(finite, 99.0)),
            }
        )
    return rows


def _counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter, key=lambda item: str(item))}


def _labels_from_parameter_space(path: Path) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    if not path.exists():
        return (), []
    payload = _read_json(path)
    records = payload.get("parameters", []) if isinstance(payload, Mapping) else []
    if not isinstance(records, list):
        return (), []
    labels = tuple(str(record["label"]) for record in records if isinstance(record, Mapping) and "label" in record)
    return labels, [dict(record) for record in records if isinstance(record, Mapping)]


def _labels_from_vector_spaces(path: Path, key: str) -> tuple[str, ...]:
    if not path.exists():
        return ()
    payload = _read_json(path)
    space = payload.get("spaces", {}).get(key) if isinstance(payload, Mapping) else None
    if not isinstance(space, Mapping):
        return ()
    return tuple(str(component["label"]) for component in space.get("components", []) if isinstance(component, Mapping) and "label" in component)


def _sigmas_from_parameter_records(records: list[dict[str, Any]], labels: tuple[str, ...]) -> np.ndarray | None:
    by_label = {str(record.get("label")): record for record in records}
    sigmas: list[float] = []
    for label in labels:
        record = by_label.get(label)
        if record is None or record.get("parameter_sigma") is None:
            return None
        sigmas.append(float(record["parameter_sigma"]))
    arr = np.asarray(sigmas, dtype=np.float64)
    if arr.shape != (len(labels),) or not np.all(np.isfinite(arr)) or np.any(arr == 0.0):
        return None
    return arr


def _sigmas_from_vector_spaces(path: Path, dim: int) -> np.ndarray | None:
    if not path.exists():
        return None
    payload = _read_json(path)
    transform = payload.get("transforms", {}).get("fisher_diagonal_scale", {}) if isinstance(payload, Mapping) else {}
    try:
        scales = np.asarray(transform.get("scales", []), dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if scales.shape != (dim,) or not np.all(np.isfinite(scales)) or np.any(scales == 0.0):
        return None
    return scales


@dataclass
class _Accumulator:
    science_labels: tuple[str, ...]
    nuisance_labels: tuple[str, ...]
    sigmas: np.ndarray | None
    sample_count: int = 0
    sample_ids: Counter[str] = field(default_factory=Counter)
    sample_indices: list[int] = field(default_factory=list)
    families: Counter[str] = field(default_factory=Counter)
    roles: Counter[str] = field(default_factory=Counter)
    splits: Counter[str] = field(default_factory=Counter)
    active_counts: Counter[int] = field(default_factory=Counter)
    active_parameters: Counter[str] = field(default_factory=Counter)
    pair_counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    triple_counts: Counter[tuple[str, str, str]] = field(default_factory=Counter)
    nuisance_ids: Counter[str] = field(default_factory=Counter)
    nuisance_vectors_by_id: dict[str, set[tuple[float, ...]]] = field(default_factory=lambda: defaultdict(set))
    science_ids: Counter[str] = field(default_factory=Counter)
    science_nuisance_ids: Counter[tuple[str, str]] = field(default_factory=Counter)
    physical_vectors: list[np.ndarray] = field(default_factory=list)
    fisher_vectors: list[np.ndarray] = field(default_factory=list)
    nuisance_vectors: list[np.ndarray] = field(default_factory=list)
    image_shapes: Counter[tuple[int, ...]] = field(default_factory=Counter)
    vector_dimension_mismatches: list[dict[str, Any]] = field(default_factory=list)
    nonfinite_records: list[dict[str, Any]] = field(default_factory=list)
    missing_render_refs: int = 0
    missing_metadata_refs: int = 0

    def add_record(self, row: Mapping[str, Any], row_number: int, root: Path, *, prepared: bool) -> None:
        self.sample_count += 1
        sample_id = str(row.get("sample_id", f"__missing_row_{row_number}"))
        self.sample_ids[sample_id] += 1
        if row.get("sample_index") is not None:
            try:
                self.sample_indices.append(int(row["sample_index"]))
            except (TypeError, ValueError):
                self.vector_dimension_mismatches.append({"row": row_number, "field": "sample_index", "value": row.get("sample_index")})
        self.families[str(row.get("dataset_family", "unavailable"))] += 1
        self.roles[str(row.get("sample_role", "unavailable"))] += 1
        self.splits[str(row.get("split", "unavailable"))] += 1
        if row.get("fits_path") in (None, "") and row.get("source_fits_path") in (None, "") and not prepared:
            self.missing_render_refs += 1
        if row.get("metadata_path") in (None, "") and row.get("source_metadata_path") in (None, "") and not prepared:
            self.missing_metadata_refs += 1

        physical = self._physical_vector(row, prepared=prepared)
        fisher = self._fisher_vector(row, physical=physical, prepared=prepared)
        nuisance = self._nuisance_vector(row, prepared=prepared)
        self._record_vector("physical", physical, len(self.science_labels), row_number)
        self._record_vector("fisher_scaled", fisher, len(self.science_labels), row_number)
        self._record_vector("nuisance", nuisance, len(self.nuisance_labels), row_number)
        if physical is not None and physical.shape == (len(self.science_labels),):
            self.physical_vectors.append(physical)
            science_id = _field(row, "group_ids.physical_delta_sha256")
            if science_id in (None, ""):
                science_id = _stable_id(physical)
            self.science_ids[str(science_id)] += 1
        else:
            science_id = None
        if fisher is not None and fisher.shape == (len(self.science_labels),):
            self.fisher_vectors.append(fisher)
        if nuisance is not None and nuisance.shape == (len(self.nuisance_labels),):
            self.nuisance_vectors.append(nuisance)

        nuisance_id = row.get("nuisance_id", _field(row, "group_ids.nuisance"))
        nuisance_id_text = "unavailable" if nuisance_id in (None, "") else str(nuisance_id)
        self.nuisance_ids[nuisance_id_text] += 1
        if nuisance is not None and nuisance.shape == (len(self.nuisance_labels),):
            self.nuisance_vectors_by_id[nuisance_id_text].add(tuple(float(v) for v in nuisance))
        if science_id is not None:
            self.science_nuisance_ids[(str(science_id), nuisance_id_text)] += 1

        active_labels = self._active_labels(row)
        if active_labels is not None:
            self.active_counts[len(active_labels)] += 1
            for label in active_labels:
                self.active_parameters[str(label)] += 1
            if len(active_labels) == 2:
                self.pair_counts[tuple(sorted(str(v) for v in active_labels))] += 1
            elif len(active_labels) == 3:
                self.triple_counts[tuple(sorted(str(v) for v in active_labels))] += 1
                for pair in self._pairs(active_labels):
                    self.pair_counts[pair] += 1
        shape = row.get("image_shape")
        if shape is None:
            shape = _field(row, "array_storage.sample_shape")
        if isinstance(shape, (list, tuple)):
            try:
                self.image_shapes[tuple(int(v) for v in shape)] += 1
            except (TypeError, ValueError):
                pass

    def _record_vector(self, name: str, vector: np.ndarray | None, expected_dim: int, row_number: int) -> None:
        if vector is None:
            return
        if vector.shape != (expected_dim,):
            self.vector_dimension_mismatches.append(
                {"row": row_number, "field": name, "observed": list(vector.shape), "expected": [expected_dim]}
            )
            return
        if not np.all(np.isfinite(vector)):
            self.nonfinite_records.append({"row": row_number, "field": name})

    def _physical_vector(self, row: Mapping[str, Any], *, prepared: bool) -> np.ndarray | None:
        if prepared:
            return _vector_from_value(row.get("physical_delta"))
        return _vector_from_mapping(row.get("theta_delta", {}) or {}, self.science_labels)

    def _fisher_vector(self, row: Mapping[str, Any], *, physical: np.ndarray | None, prepared: bool) -> np.ndarray | None:
        if prepared:
            return _vector_from_value(row.get("fisher_scaled_delta"))
        sigma_map = row.get("theta_sigma")
        if isinstance(sigma_map, Mapping):
            return _vector_from_mapping(sigma_map, self.science_labels)
        if physical is not None and self.sigmas is not None and physical.shape == self.sigmas.shape:
            return physical / self.sigmas
        return None

    def _nuisance_vector(self, row: Mapping[str, Any], *, prepared: bool) -> np.ndarray | None:
        if prepared:
            return _vector_from_value(row.get("nuisance_vector"))
        return _vector_from_mapping(row.get("registration_nuisance_values", {}) or {}, self.nuisance_labels)

    def _active_labels(self, row: Mapping[str, Any]) -> tuple[str, ...] | None:
        labels = row.get("active_labels")
        if isinstance(labels, list):
            return tuple(str(label) for label in labels)
        mask = row.get("active_mask")
        if isinstance(mask, list) and self.science_labels and len(mask) == len(self.science_labels):
            return tuple(label for label, flag in zip(self.science_labels, mask) if bool(flag))
        return None

    @staticmethod
    def _pairs(labels: tuple[str, ...]) -> Iterable[tuple[str, str]]:
        values = sorted(str(value) for value in labels)
        for idx, left in enumerate(values):
            for right in values[idx + 1 :]:
                yield (left, right)


def _classify_nuisance(acc: _Accumulator) -> dict[str, Any]:
    if not acc.science_nuisance_ids or not acc.nuisance_ids or not acc.science_ids:
        return {"status": "unavailable", "classification": "unknown"}
    science_count = len(acc.science_ids)
    nuisance_count = len([key for key in acc.nuisance_ids if key != "unavailable"])
    expected = science_count * nuisance_count
    observed_pairs = len(acc.science_nuisance_ids)
    all_once = all(count == 1 for count in acc.science_nuisance_ids.values())
    if nuisance_count > 0 and observed_pairs == expected and all_once and acc.sample_count == expected:
        classification = "full_cross_product"
        status = "confirmed"
    elif science_count == acc.sample_count and all(count == 1 for count in acc.science_ids.values()):
        classification = "one_nuisance_per_science_sample"
        status = "inferred"
    elif observed_pairs < expected:
        classification = "partial"
        status = "inferred"
    else:
        classification = "unknown"
        status = "unavailable"
    return {
        "status": status,
        "classification": classification,
        "unique_science_states": science_count,
        "unique_nuisance_ids": nuisance_count,
        "observed_science_nuisance_pairs": observed_pairs,
        "expected_full_cross_product_pairs": expected,
    }


def _identity(root: Path, manifest: Mapping[str, Any] | None, *, prepared: bool) -> dict[str, Any]:
    manifest = manifest or {}
    resolved = manifest.get("resolved_system_summary", {}) if isinstance(manifest, Mapping) else {}
    prescription = manifest.get("prescription_resolved", {}) if isinstance(manifest, Mapping) else {}
    system = prescription.get("system", {}) if isinstance(prescription, Mapping) else {}
    source = manifest.get("source_dataset", {}) if isinstance(manifest, Mapping) else {}
    detector = system.get("detector") if isinstance(system, Mapping) else None
    return {
        "dataset_or_artifact_id": _status(
            manifest.get("artifact_id") or root.name,
            "confirmed" if manifest or root.exists() else "inferred",
            "manifest.json or path",
        ),
        "path": _status(str(root), "confirmed", "input path"),
        "schema_version": _status(manifest.get("schema_version"), "confirmed" if manifest.get("schema_version") else "unavailable", "manifest.json"),
        "generator": _status(manifest.get("generator") or source.get("generator"), "confirmed" if manifest.get("generator") or source.get("generator") else "unavailable", "manifest.json"),
        "script_version": _status(manifest.get("script_version"), "confirmed" if manifest.get("script_version") else "unavailable", "manifest.json"),
        "renderer_commit": _status(
            (manifest.get("git_info") or manifest.get("git") or {}).get("commit"),
            "confirmed" if (manifest.get("git_info") or manifest.get("git") or {}).get("commit") else "unavailable",
            "manifest git_info",
        ),
        "optical_configuration": _status(
            manifest.get("system_preset") or resolved.get("preset"),
            "confirmed" if manifest.get("system_preset") or resolved.get("preset") else "unavailable",
            "manifest system summary",
        ),
        "detector_configuration": _status(
            detector,
            "confirmed" if detector is not None else "unavailable",
            "prescription_resolved.json",
        ),
        "exposure_configuration": _status(resolved.get("exposure_time_s"), "confirmed" if resolved.get("exposure_time_s") is not None else "unavailable", "manifest resolved_system_summary"),
        "noise_policy": _status(manifest.get("noise_config"), "confirmed" if manifest.get("noise_config") is not None else "unavailable", "manifest noise_config"),
        "dataset_kind": _status("prepared" if prepared else "raw_or_context", "inferred", "artifact shape"),
    }


def _nuisance_labels(root: Path, manifest: Mapping[str, Any], *, prepared: bool) -> tuple[str, ...]:
    if prepared:
        labels = _labels_from_vector_spaces(root / "vector_spaces.json", "registration_nuisance")
        return labels
    cfg = manifest.get("nuisance_config", {}) if isinstance(manifest, Mapping) else {}
    if isinstance(cfg, Mapping):
        return tuple(str(key) for key in cfg.get("keys", []) or [])
    return ()


def _parameter_metadata(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        rows.append(
            {
                "index": idx,
                "label": record.get("label"),
                "base_key": record.get("base_key"),
                "component_index": record.get("component_index"),
                "unit": record.get("units"),
                "nominal_value": record.get("nominal_value"),
                "parameter_sigma": record.get("parameter_sigma"),
                "min_sigma": record.get("min_sigma"),
                "max_sigma": record.get("max_sigma"),
                "min_abs_delta": record.get("min_abs_delta"),
                "max_abs_delta": record.get("max_abs_delta"),
                "group": record.get("group"),
                "noll_index": record.get("noll_index"),
            }
        )
    return rows


def _summarize_accumulator(
    root: Path,
    manifest: Mapping[str, Any],
    acc: _Accumulator,
    parameter_records: list[dict[str, Any]],
    *,
    prepared: bool,
    shard_manifest: Mapping[str, Any] | None,
    plan_row_count: int | None,
) -> dict[str, Any]:
    duplicate_sample_ids = {key: count for key, count in acc.sample_ids.items() if count > 1}
    duplicate_science = sum(1 for count in acc.science_ids.values() if count > 1)
    duplicate_science_nuisance = sum(1 for count in acc.science_nuisance_ids.values() if count > 1)
    missing_ranges: list[list[int]] = []
    if acc.sample_indices:
        observed = set(acc.sample_indices)
        if len(observed) < max(observed) - min(observed) + 1:
            start = None
            prev = None
            for value in range(min(observed), max(observed) + 1):
                if value in observed:
                    if start is not None:
                        missing_ranges.append([start, prev if prev is not None else start])
                        start = None
                    continue
                if start is None:
                    start = value
                prev = value
            if start is not None:
                missing_ranges.append([start, prev if prev is not None else start])
    nuisance_bank = {
        nuisance_id: [list(vector) for vector in sorted(vectors)]
        for nuisance_id, vectors in sorted(acc.nuisance_vectors_by_id.items(), key=lambda item: str(item[0]))
    }
    fixed_bank = bool(nuisance_bank) and all(len(vectors) == 1 for vectors in acc.nuisance_vectors_by_id.values())
    return {
        "identity": _identity(root, manifest, prepared=prepared),
        "size_structure": {
            "sample_count": _status(acc.sample_count, "confirmed", "metadata rows"),
            "unique_science_state_count": _status(len(acc.science_ids), "confirmed" if acc.science_ids else "unavailable", "science vector identity"),
            "unique_nuisance_state_count": _status(len(acc.nuisance_ids), "confirmed" if acc.nuisance_ids else "unavailable", "nuisance_id"),
            "science_dimension": _status(len(acc.science_labels), "confirmed" if acc.science_labels else "unavailable", "parameter/vector metadata"),
            "nuisance_dimension": _status(len(acc.nuisance_labels), "confirmed" if acc.nuisance_labels else "unavailable", "nuisance metadata"),
            "image_shape": _status(_counter_dict(Counter({str(list(key)): value for key, value in acc.image_shapes.items()})), "confirmed" if acc.image_shapes else "unavailable", "sample rows"),
            "shard_count": _status(None if shard_manifest is None else shard_manifest.get("shard_count"), "confirmed" if shard_manifest else "unavailable", "array_shards_manifest.json"),
            "plan_row_count": _status(plan_row_count, "confirmed" if plan_row_count is not None else "unavailable", "plan csv"),
        },
        "families": {
            "family_counts": _counter_dict(acc.families),
            "sample_role_counts": _counter_dict(acc.roles),
            "split_counts": _counter_dict(acc.splits),
        },
        "ordered_vector_metadata": {
            "science_labels": list(acc.science_labels),
            "nuisance_labels": list(acc.nuisance_labels),
            "parameters": _parameter_metadata(parameter_records),
            "fisher_scales": None if acc.sigmas is None else acc.sigmas.tolist(),
        },
        "science_space_coverage": {
            "physical_delta": _vector_summary(acc.physical_vectors, acc.science_labels),
            "fisher_scaled_delta": _vector_summary(acc.fisher_vectors, acc.science_labels),
            "fisher_radius_l2": _summary([float(np.linalg.norm(row)) for row in acc.fisher_vectors]),
        },
        "sparse_behavior": {
            "active_count_distribution": _counter_dict(acc.active_counts),
            "per_parameter_activation_frequency": _counter_dict(acc.active_parameters),
            "pair_occurrence_summary": {
                "unique_pairs_observed": len(acc.pair_counts),
                "total_pair_occurrences": int(sum(acc.pair_counts.values())),
                "min_count": None if not acc.pair_counts else int(min(acc.pair_counts.values())),
                "max_count": None if not acc.pair_counts else int(max(acc.pair_counts.values())),
            },
            "triple_occurrence_summary": {
                "unique_triples_observed": len(acc.triple_counts),
                "total_triple_occurrences": int(sum(acc.triple_counts.values())),
                "min_count": None if not acc.triple_counts else int(min(acc.triple_counts.values())),
                "max_count": None if not acc.triple_counts else int(max(acc.triple_counts.values())),
            },
        },
        "nuisance_coverage": {
            "nuisance_id_distribution": _counter_dict(acc.nuisance_ids),
            "nuisance_vectors_by_id": nuisance_bank,
            "samples_per_nuisance_id": _counter_dict(acc.nuisance_ids),
            "nuisance_vector_summary": _vector_summary(acc.nuisance_vectors, acc.nuisance_labels),
            "fixed_bank": _status(fixed_bank, "confirmed" if nuisance_bank else "unavailable", "nuisance_id to vector mapping"),
            "replication_classification": _classify_nuisance(acc),
        },
        "integrity": {
            "duplicate_sample_ids": duplicate_sample_ids,
            "duplicate_sample_id_count": len(duplicate_sample_ids),
            "duplicate_science_state_identity_count": duplicate_science,
            "duplicate_science_nuisance_identity_count": duplicate_science_nuisance,
            "vector_dimension_mismatches": acc.vector_dimension_mismatches[:100],
            "vector_dimension_mismatch_count": len(acc.vector_dimension_mismatches),
            "nonfinite_records": acc.nonfinite_records[:100],
            "nonfinite_record_count": len(acc.nonfinite_records),
            "missing_render_reference_count": acc.missing_render_refs,
            "missing_metadata_reference_count": acc.missing_metadata_refs,
            "missing_sample_index_ranges": missing_ranges[:20],
            "manifest_count_mismatch": _manifest_count_mismatch(manifest, acc.sample_count, prepared=prepared),
            "plan_count_mismatch": None if plan_row_count is None else plan_row_count != acc.sample_count,
            "image_index_count_mismatch": _image_index_mismatch(shard_manifest, acc.sample_count),
        },
    }


def _manifest_count_mismatch(manifest: Mapping[str, Any], sample_count: int, *, prepared: bool) -> bool | None:
    if prepared:
        expected = (manifest.get("array_storage") or {}).get("sample_count")
        if expected is None:
            expected = (manifest.get("source_dataset") or {}).get("prepared_sample_count")
    else:
        expected = manifest.get("rendered_sample_count")
    return None if expected is None else int(expected) != int(sample_count)


def _image_index_mismatch(shard_manifest: Mapping[str, Any] | None, sample_count: int) -> bool | None:
    if shard_manifest is None:
        return None
    expected = shard_manifest.get("sample_count")
    return None if expected is None else int(expected) != int(sample_count)


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def audit_dataset(root: Path) -> dict[str, Any]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    if isinstance(manifest, dict) and (root / "prescription_resolved.json").exists():
        manifest = dict(manifest)
        manifest["prescription_resolved"] = _read_json(root / "prescription_resolved.json")
    prepared = (root / "index.jsonl").exists() and (root / "array_shards_manifest.json").exists()
    index_path = root / "index.jsonl" if prepared else root / "samples.jsonl"
    if not index_path.exists():
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "path": str(root),
            "status": "unsupported",
            "reason": "No supported samples.jsonl or prepared index.jsonl metadata was found.",
        }
    if prepared:
        labels = _labels_from_vector_spaces(root / "vector_spaces.json", "fisher_scaled_delta")
        nuisance_labels = _nuisance_labels(root, manifest, prepared=True)
        parameter_records: list[dict[str, Any]] = []
        sigmas = _sigmas_from_vector_spaces(root / "vector_spaces.json", len(labels))
        shard_manifest = _read_json(root / "array_shards_manifest.json")
        plan_count = None
    else:
        labels, parameter_records = _labels_from_parameter_space(root / "parameter_space.json")
        nuisance_labels = _nuisance_labels(root, manifest, prepared=False)
        sigmas = _sigmas_from_parameter_records(parameter_records, labels)
        shard_manifest = None
        plan_path = root / "sparse_mixture_plan.csv"
        pair_path = root / "pair_plan.csv"
        plan_count = _count_csv_rows(plan_path) + _count_csv_rows(pair_path)
    acc = _Accumulator(science_labels=labels, nuisance_labels=nuisance_labels, sigmas=sigmas)
    for row_number, row in enumerate(_iter_jsonl(index_path), start=1):
        acc.add_record(row, row_number, root, prepared=prepared)
    payload = _summarize_accumulator(
        root,
        manifest if isinstance(manifest, Mapping) else {},
        acc,
        parameter_records,
        prepared=prepared,
        shard_manifest=shard_manifest if isinstance(shard_manifest, Mapping) else None,
        plan_row_count=plan_count,
    )
    payload["schema_version"] = AUDIT_SCHEMA_VERSION
    payload["status"] = "ok"
    return payload


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_ready(value) for key, value in row.items()})


def write_output_dir(path: Path, audit: Mapping[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _write_json(path / "audit_summary.json", audit)
    _write_json(path / "integrity_summary.json", audit.get("integrity", {}))
    _write_csv(path / "parameter_summary.csv", audit.get("science_space_coverage", {}).get("fisher_scaled_delta", []))
    nuisance_rows = audit.get("nuisance_coverage", {}).get("nuisance_vector_summary", [])
    _write_csv(path / "nuisance_summary.csv", nuisance_rows)
    sparse = audit.get("sparse_behavior", {})
    sparse_rows = [
        {"section": section, "key": key, "value": value}
        for section in ("active_count_distribution", "per_parameter_activation_frequency")
        for key, value in sparse.get(section, {}).items()
    ]
    _write_csv(path / "sparse_summary.csv", sparse_rows)


def print_summary(audit: Mapping[str, Any]) -> None:
    if audit.get("status") != "ok":
        print(f"Unsupported dataset: {audit.get('reason')}")
        return
    identity = audit["identity"]
    size = audit["size_structure"]
    families = audit["families"]
    nuisance = audit["nuisance_coverage"]
    integrity = audit["integrity"]
    print("Dataset Audit")
    print(f"  path: {identity['path']['value']}")
    print(f"  schema: {identity['schema_version']['value']}")
    print(f"  generator: {identity['generator']['value']}")
    print(f"  samples: {size['sample_count']['value']}")
    print(f"  science_dim: {size['science_dimension']['value']}")
    print(f"  nuisance_dim: {size['nuisance_dimension']['value']}")
    print(f"  families: {families['family_counts']}")
    print(f"  splits: {families['split_counts']}")
    print(f"  nuisance classification: {nuisance['replication_classification']['classification']}")
    print(f"  fixed nuisance bank: {nuisance['fixed_bank']['value']}")
    print(
        "  integrity: "
        f"duplicate_sample_ids={integrity['duplicate_sample_id_count']} "
        f"dimension_mismatches={integrity['vector_dimension_mismatch_count']} "
        f"nonfinite_records={integrity['nonfinite_record_count']}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only audit for SHERA ML dataset metadata.")
    parser.add_argument("path", type=Path, help="Raw dataset root, compact context package, or prepared dataset root.")
    parser.add_argument("--output-json", type=Path, default=None, help="Write the full audit payload to this JSON path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Write summary JSON/CSV artifacts to this directory.")
    args = parser.parse_args(argv)

    audit = audit_dataset(args.path)
    print_summary(audit)
    if args.output_json is not None:
        _write_json(args.output_json, audit)
    if args.output_dir is not None:
        write_output_dir(args.output_dir, audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
