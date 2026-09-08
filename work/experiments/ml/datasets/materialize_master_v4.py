from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import tarfile
import tempfile
from datetime import UTC, datetime
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml
from scipy.stats import qmc

from dluxshera.datasets.schema import (
    VectorComponentSpec,
    VectorSpaceSpec,
    json_ready,
    read_json,
    write_json,
    write_jsonl,
)
from dluxshera.datasets.master_v4 import (
    locate_science_global_index,
    render_index_location,
    render_index_to_science_nuisance,
    render_state_id,
    science_nuisance_to_render_index,
)

DATASET_VERSION = "shera_ml_master_v4"
STATE_PLAN_SCHEMA = "shera_v4_science_state_plan/1"
MASTER_SCHEMA = "shera_v4_master_contract/1"
RENDER_CONTRACT_SCHEMA = "shera_v4_compact_render_contract/1"
RENDER_SYSTEM_CONTRACT_SCHEMA = "shera_v4_render_system_contract/1"
SPLIT_INTEGRITY_SCHEMA = "shera_v4_split_integrity_summary/1"
REVIEW_DECISIONS_SCHEMA = "shera_v4_review_decisions/1"
TRANSFER_PACKAGE_SCHEMA = "shera_v4_transfer_package_manifest/1"

DEFAULT_OUTDIR = Path("work/experiments/ml/datasets/materialized/master_v4")
DEFAULT_PREPARED_V3_NUISANCE = Path("Results/ML Training Datasets/preprocessed/PREP-V3-nuisance-v1")
DEFAULT_V3_NUISANCE_RAW = Path("Results/ML Training Datasets/shera_training_dataset_nuisance_pairs_20260511")
DEFAULT_SPARSE_CONTEXT = Path("Results/ML Training Datasets/shera_test_dataset_sparse_nuisance_20260707_context")
DEFAULT_RENDER_SYSTEM_SOURCE = DEFAULT_V3_NUISANCE_RAW / "prescription_resolved.json"

JOINT_SCALE_STRATA = (0.125, 0.25, 0.5, 1.0)
JOINT_COUNTS = {"train": 65536, "validation": 8192, "test": 8192}
RADIAL_COUNTS = {"train": 16384, "validation": 4096, "test": 4096}
RADIAL_BINS = ((0.0, 100.0), (100.0, 250.0), (250.0, 500.0), (500.0, 1000.0), (1000.0, 1500.0), (1500.0, 2000.0))
TRAIN_PREFIXES = (4096, 8192, 16384, 32768, 65536)
FAMILY_ORDER = ("joint_full_v4", "radial_capture_v4")
SPLIT_ORDER = ("train", "validation", "test")
TRANSFER_INCLUDE = (
    "master_v4.yaml",
    "vector_spaces.json",
    "unit_contract.csv",
    "joint_base_envelope.json",
    "nuisance_bank.json",
    "nuisance_bank.csv",
    "render_system_contract.json",
    "render_contract.json",
    "render_scale_summary.json",
    "subset_registry.json",
    "freeze_manifest.json",
    "compatibility/coordinate_compatibility_comparison.json",
    "compatibility/coordinate_compatibility_components.csv",
    "compatibility/nuisance_bank_comparison.json",
    "qa/split_integrity_summary.json",
    "qa/review_decisions.json",
    "qa/joint_geometry_summary.json",
    "qa/joint_squared_radius_contribution.csv",
    "qa/radial_feasibility_summary.json",
    "qa/radial_direction_conditioning.csv",
    "state_plans/joint_full_v4/train.jsonl",
    "state_plans/joint_full_v4/validation.jsonl",
    "state_plans/joint_full_v4/test.jsonl",
    "state_plans/radial_capture_v4/train.jsonl",
    "state_plans/radial_capture_v4/validation.jsonl",
    "state_plans/radial_capture_v4/test.jsonl",
    "RENDER_HANDOFF.md",
)

SCIENCE_UNIT_OVERRIDES = {
    "source.separation_as": ("arcsec", "binary separation on sky"),
    "source.log_flux_total": (
        "log10(detected_photons)",
        "log10 of total detected photons from both source components over the modeled exposure after collecting area and throughput",
    ),
    "source.contrast": ("dimensionless", "broadband A/B flux ratio"),
    "optics.plate_scale_as_per_pix": ("arcsec / pixel", "effective detector plate scale"),
}
NUISANCE_UNITS = {
    "source.x_position_as": ("arcsec", "source registration x offset on sky"),
    "source.y_position_as": ("arcsec", "source registration y offset on sky"),
    "source.position_angle_deg": ("deg", "binary source position angle perturbation"),
}
CANONICAL_NUISANCE_LABELS = tuple(NUISANCE_UNITS)
FISHER_SCALE_REL_TOL = 1.0e-12
FISHER_SCALE_ABS_TOL = 0.0


def canonical_json(value: Any) -> str:
    return json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_populated(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def prepare_output_root(outdir: Path, *, overwrite: bool = False, dry_run: bool = False) -> None:
    if outdir.exists() and not outdir.is_dir():
        raise ValueError(f"Output path exists and is not a directory: {outdir}")
    if _is_populated(outdir) and not overwrite:
        raise FileExistsError(
            f"Output directory is populated: {outdir}. Use --overwrite to replace the generated artifact tree."
        )
    if dry_run:
        return
    if overwrite and _is_populated(outdir):
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)


def stable_seed(*tokens: Any) -> int:
    digest = hashlib.sha256(canonical_json(tokens).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def _canonical_path_value(value: str, *, base_dir: Path) -> dict[str, Any]:
    path = Path(value)
    candidates = [path] if path.is_absolute() else [base_dir / path, path]
    out: dict[str, Any] = {"path_kind": "local_file_reference"}
    for resolved in candidates:
        if resolved.exists() and resolved.is_file():
            out["file_sha256"] = file_hash(resolved)
            return out
    out["unresolved_logical_reference"] = path.as_posix()
    return out


def canonicalize_scientific_paths(value: Any, *, base_dir: Path) -> Any:
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, val in value.items():
            key_text = str(key)
            if key_text.endswith("_path") and isinstance(val, str):
                out[key_text] = _canonical_path_value(val, base_dir=base_dir)
            else:
                out[key_text] = canonicalize_scientific_paths(val, base_dir=base_dir)
        return out
    if isinstance(value, list):
        return [canonicalize_scientific_paths(item, base_dir=base_dir) for item in value]
    return value


def _science_vector_key(row: Mapping[str, Any]) -> str:
    return content_hash(row["ordered_physical_science_vector"])


def _science_row_key(family: str, split: str, row: Mapping[str, Any]) -> tuple[str, str, str]:
    return family, split, str(row["science_state_id"])


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                payload = json.loads(line)
                if isinstance(payload, Mapping):
                    yield dict(payload)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(json_ready(value), sort_keys=True) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def load_parameter_records(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    records = payload.get("parameters", []) if isinstance(payload, Mapping) else []
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path} must contain non-empty parameters.")
    return [dict(record) for record in records]


def _duplicate_values(values: Sequence[str]) -> list[str]:
    counts = Counter(values)
    return sorted(label for label, count in counts.items() if count > 1)


def authoritative_science_components(prepared_vector_spaces_path: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    prepared = read_json(prepared_vector_spaces_path)
    fisher_space = prepared["spaces"]["fisher_scaled_delta"]
    components = [dict(component) for component in fisher_space["components"]]
    labels = [str(component["label"]) for component in components]
    duplicates = _duplicate_values(labels)
    if duplicates:
        raise ValueError(
            f"{prepared_vector_spaces_path} has duplicate fisher_scaled_delta components: {duplicates}"
        )
    scales = np.asarray(
        prepared["transforms"]["fisher_diagonal_scale"]["scales"],
        dtype=np.float64,
    )
    if scales.shape != (len(labels),):
        raise ValueError(
            f"{prepared_vector_spaces_path} fisher scale count {scales.shape[0]} "
            f"does not match component count {len(labels)}."
        )
    if not np.all(np.isfinite(scales)) or np.any(scales == 0.0):
        raise ValueError(f"{prepared_vector_spaces_path} contains non-finite or zero Fisher scales.")
    return components, scales


def reconcile_science_records(
    *,
    prepared_vector_spaces_path: Path,
    raw_parameter_records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], np.ndarray, list[dict[str, Any]]]:
    """Return raw records in the authoritative prepared V3 science order."""
    prepared_components, prepared_scales = authoritative_science_components(prepared_vector_spaces_path)
    canonical_labels = [str(component["label"]) for component in prepared_components]
    raw_labels = [str(record.get("label")) for record in raw_parameter_records]
    duplicates = _duplicate_values(raw_labels)
    if duplicates:
        raise ValueError(f"Raw parameter_space contains duplicate science labels: {duplicates}")
    records_by_label = {str(record["label"]): dict(record) for record in raw_parameter_records}
    missing = [label for label in canonical_labels if label not in records_by_label]
    unexpected = sorted(label for label in records_by_label if label not in set(canonical_labels))
    if missing or unexpected:
        raise ValueError(
            "Raw parameter_space labels do not match authoritative prepared V3 science labels: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if len(records_by_label) != len(canonical_labels):
        raise ValueError(
            f"Raw parameter_space component count {len(records_by_label)} does not match "
            f"prepared component count {len(canonical_labels)}."
        )
    ordered_records = [records_by_label[label] for label in canonical_labels]
    compatibility_rows = fisher_scale_compatibility_rows(
        prepared_components=prepared_components,
        prepared_scales=prepared_scales,
        ordered_records=ordered_records,
    )
    mismatches = [row for row in compatibility_rows if row["compatibility_status"] != "PASS"]
    if mismatches:
        raise ValueError(
            "Prepared V3 Fisher scales disagree with raw parameter_space by label: "
            f"{mismatches[:5]}"
        )
    return ordered_records, prepared_scales, compatibility_rows


def fisher_scale_compatibility_rows(
    *,
    prepared_components: Sequence[Mapping[str, Any]],
    prepared_scales: Sequence[float],
    ordered_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if len(prepared_components) != len(ordered_records):
        raise ValueError(
            f"Prepared component count {len(prepared_components)} does not match "
            f"raw ordered record count {len(ordered_records)}."
        )
    for idx, (component, prepared_scale, record) in enumerate(
        zip(prepared_components, prepared_scales, ordered_records)
    ):
        label = str(component["label"])
        raw_label = str(record["label"])
        if raw_label != label:
            raise ValueError(
                f"Canonical science label mismatch at index {idx}: prepared {label!r}, raw {raw_label!r}."
            )
        raw_scale = float(record["parameter_sigma"])
        prepared_value = float(prepared_scale)
        abs_diff = abs(prepared_value - raw_scale)
        frac_diff = 0.0 if raw_scale == 0.0 else abs_diff / abs(raw_scale)
        status = (
            "PASS"
            if math.isclose(
                prepared_value,
                raw_scale,
                rel_tol=FISHER_SCALE_REL_TOL,
                abs_tol=FISHER_SCALE_ABS_TOL,
            )
            else "FAIL"
        )
        rows.append(
            {
                "canonical_index": idx,
                "index": idx,
                "label": label,
                "prepared_fisher_scale": prepared_value,
                "raw_fisher_scale": raw_scale,
                "absolute_difference": abs_diff,
                "fractional_difference": frac_diff,
                "compatibility_status": status,
            }
        )
    return rows


def unit_for_science(label: str) -> tuple[str, str]:
    if label in SCIENCE_UNIT_OVERRIDES:
        return SCIENCE_UNIT_OVERRIDES[label]
    if ".zernike_coeffs_nm[" in label:
        return "nm", "Zernike wavefront coefficient in nanometers"
    raise ValueError(f"No explicit unit contract for science component {label!r}.")


def build_vector_contracts(
    *,
    prepared_vector_spaces_path: Path,
    ordered_parameter_records: Sequence[Mapping[str, Any]],
    prepared_fisher_scales: Sequence[float],
    fisher_scale_compatibility: Sequence[Mapping[str, Any]],
    training_source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    components_physical: list[VectorComponentSpec] = []
    components_fisher: list[VectorComponentSpec] = []
    unit_rows: list[dict[str, Any]] = []
    scales: list[float] = []
    ordered_labels = [str(record["label"]) for record in ordered_parameter_records]
    compatibility_by_label = {
        str(row["label"]): dict(row) for row in fisher_scale_compatibility
    }
    for idx, record in enumerate(ordered_parameter_records):
        label = str(record["label"])
        unit, semantic = unit_for_science(label)
        sigma = float(prepared_fisher_scales[idx])
        scales.append(sigma)
        common = {
            "label": label,
            "index": idx,
            "source_key": record.get("base_key"),
            "component_index": record.get("component_index"),
            "display_label": record.get("display_label"),
            "group": record.get("group"),
            "reference_value": record.get("nominal_value"),
            "metadata": {
                "semantic": semantic,
                "noll_index": record.get("noll_index"),
                "sampling_envelope": {
                    "kind": "historical_sweep_extent",
                    "min_sigma": record.get("min_sigma"),
                    "max_sigma": record.get("max_sigma"),
                    "min_abs_delta": record.get("min_abs_delta"),
                    "max_abs_delta": record.get("max_abs_delta"),
                    "source": "S01/S05 V3 nuisance-pair parameter_space.json",
                },
                "fisher_scale_compatibility": compatibility_by_label[label],
                "physical_validity_constraints": {
                    "status": "not_encoded",
                    "reason": "V3 sweep extrema are sampling envelopes, not physical/model-validity bounds.",
                },
            },
        }
        components_physical.append(
            VectorComponentSpec(
                **common,
                unit=unit,
                scale={"kind": "fisher_diagonal_sigma", "value": sigma, "forward_usage": "delta_divided_by_sigma"},
            )
        )
        components_fisher.append(
            VectorComponentSpec(
                **common,
                unit="dimensionless",
                scale={"kind": "dimensionless_fisher_coordinate", "source_sigma": sigma},
            )
        )
        unit_rows.append({"index": idx, "label": label, "physical_unit": unit, "fisher_unit": "dimensionless", "semantic": semantic})
    nuisance_components = []
    nuisance_sigma_components = []
    for idx, label in enumerate(CANONICAL_NUISANCE_LABELS):
        unit, semantic = NUISANCE_UNITS[label]
        nuisance_components.append(VectorComponentSpec(label=label, index=idx, source_key=label, unit=unit, group="registration", metadata={"semantic": semantic}))
        nuisance_sigma_components.append(VectorComponentSpec(label=label, index=idx, source_key=label, unit="dimensionless", group="registration", metadata={"semantic": f"Fisher/sweep-sigma coordinate for {label}"}))
        unit_rows.append({"index": idx, "label": label, "physical_unit": unit, "fisher_unit": "dimensionless", "semantic": semantic})
    physical = VectorSpaceSpec(
        name="shera_v4_physical_science_state",
        components=tuple(components_physical),
        description="Ordered absolute physical science vector for V4, using S01/S05 training-coordinate ordering.",
        metadata=dict(training_source_identity),
    )
    fisher = VectorSpaceSpec(
        name="shera_v4_fisher_scaled_science_delta",
        components=tuple(components_fisher),
        description="Ordered V4 Fisher-scaled science deltas, dimensionless and tied to S01/S05 training scales.",
        metadata=dict(training_source_identity),
    )
    nuisance = VectorSpaceSpec(
        name="shera_v4_registration_nuisance",
        components=tuple(nuisance_components),
        description="Ordered registration nuisance physical vector.",
        metadata={"source": "V3 nuisance-pair recovered bank"},
    )
    nuisance_sigma = VectorSpaceSpec(
        name="shera_v4_registration_nuisance_sigma",
        components=tuple(nuisance_sigma_components),
        description="Ordered registration nuisance vector in V3 sweep-sigma coordinates.",
        metadata={"source": "V3 nuisance-pair recovered bank"},
    )
    payload = {
        "schema_version": "shera_v4_vector_spaces/1",
        "spaces": {
            "physical_science_state": physical.to_dict(),
            "fisher_scaled_delta": fisher.to_dict(),
            "registration_nuisance": nuisance.to_dict(),
            "registration_nuisance_sigma": nuisance_sigma.to_dict(),
        },
        "transforms": {
            "fisher_diagonal_scale": {
                "type": "DiagonalScaleTransform",
                "source_space": physical.name,
                "destination_space": fisher.name,
                "scales": scales,
                "forward_mode": "delta_divide",
                "scale_source": "PREP-V3-nuisance-v1 vector_spaces.json / source parameter_space.json",
            }
        },
        "unit_contract": unit_rows,
    }
    science_identity = {
        "schema_version": "shera_v4_science_vector_space_identity/1",
        "authoritative_order_source": "prepared_v3_fisher_scaled_delta_components",
        "ordered_labels": ordered_labels,
        "physical_space": physical.to_dict(),
        "fisher_space": fisher.to_dict(),
        "fisher_diagonal_scale": {
            "source_space": physical.name,
            "destination_space": fisher.name,
            "scales": scales,
            "forward_mode": "delta_divide",
            "scale_source": "PREP-V3 nuisance fisher_scaled_delta transform",
        },
    }
    nuisance_identity = {
        "schema_version": "shera_v4_nuisance_vector_space_identity/1",
        "authoritative_order_source": "canonical V4 registration nuisance contract verified against selected V3 nuisance-bank manifest",
        "ordered_labels": list(CANONICAL_NUISANCE_LABELS),
        "physical_space": nuisance.to_dict(),
        "fisher_space": nuisance_sigma.to_dict(),
    }
    payload["science_vector_space_id"] = content_hash(science_identity)
    payload["nuisance_vector_space_id"] = content_hash(nuisance_identity)
    payload["science_vector_space_identity"] = science_identity
    payload["nuisance_vector_space_identity"] = nuisance_identity
    payload["authoritative_prepared_vector_spaces_path"] = str(prepared_vector_spaces_path)
    for row in unit_rows:
        if row["physical_unit"] in (None, "") or row["fisher_unit"] in (None, ""):
            raise ValueError(f"Missing unit in vector contract row {row}.")
    payload["content_hash"] = content_hash({key: value for key, value in payload.items() if key != "content_hash"})
    return payload


def compatibility_comparison(
    *,
    prepared_root: Path,
    v3_raw_root: Path,
    sparse_root: Path,
    ordered_training_records: Sequence[Mapping[str, Any]],
    fisher_scale_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    sparse_records = load_parameter_records(sparse_root / "parameter_space.json")
    sparse_duplicates = _duplicate_values([str(record["label"]) for record in sparse_records])
    if sparse_duplicates:
        raise ValueError(f"Sparse parameter_space contains duplicate science labels: {sparse_duplicates}")
    sparse_by_label = {str(record["label"]): record for record in sparse_records}
    fisher_by_label = {str(row["label"]): dict(row) for row in fisher_scale_rows}
    rows = []
    all_agree = True
    for idx, record in enumerate(ordered_training_records):
        label = str(record["label"])
        sparse = sparse_by_label.get(label)
        fisher_row = fisher_by_label[label]
        row = {
            "canonical_index": idx,
            "index": idx,
            "label": label,
            "nominal_value": float(record["nominal_value"]),
            "fisher_sigma": float(fisher_row["prepared_fisher_scale"]),
            "raw_fisher_sigma": float(fisher_row["raw_fisher_scale"]),
            "historical_max_sigma": float(record["max_sigma"]),
            "historical_max_abs_delta": float(record["max_abs_delta"]),
            "source_artifact": str(v3_raw_root / "parameter_space.json"),
            "prepared_fisher_scale": float(fisher_row["prepared_fisher_scale"]),
            "raw_parameter_space_fisher_scale": float(fisher_row["raw_fisher_scale"]),
            "absolute_difference": float(fisher_row["absolute_difference"]),
            "fractional_difference": float(fisher_row["fractional_difference"]),
            "compatibility_status": fisher_row["compatibility_status"],
            "sparse_100k_agrees": False,
        }
        if sparse is not None:
            checks = {
                "nominal_value": math.isclose(float(record["nominal_value"]), float(sparse["nominal_value"]), rel_tol=0.0, abs_tol=0.0),
                "fisher_sigma": math.isclose(float(record["parameter_sigma"]), float(sparse["parameter_sigma"]), rel_tol=0.0, abs_tol=0.0),
                "historical_max_sigma": math.isclose(float(record["max_sigma"]), float(sparse["max_sigma"]), rel_tol=0.0, abs_tol=0.0),
                "historical_max_abs_delta": math.isclose(float(record["max_abs_delta"]), float(sparse["max_abs_delta"]), rel_tol=0.0, abs_tol=0.0),
            }
            row["sparse_100k_values"] = {
                "nominal_value": sparse.get("nominal_value"),
                "fisher_sigma": sparse.get("parameter_sigma"),
                "historical_max_sigma": sparse.get("max_sigma"),
                "historical_max_abs_delta": sparse.get("max_abs_delta"),
                "source_artifact": str(sparse_root / "parameter_space.json"),
            }
            row["sparse_100k_agrees"] = all(checks.values())
            row["sparse_100k_agreement_checks"] = checks
        all_agree = all_agree and bool(row["sparse_100k_agrees"]) and row["compatibility_status"] == "PASS"
        rows.append(row)
    payload = {
        "schema_version": "shera_v4_coordinate_compatibility/1",
        "authoritative_policy": "Use exact S01/S05 prepared V3 nuisance training vector_spaces and source parameter_space for first V4 campaign.",
        "science_order_policy": "Prepared V3 fisher_scaled_delta component order is canonical; raw parameter_space rows are reconciled by label before vector construction.",
        "authoritative_prepared_vector_spaces": str(prepared_root / "vector_spaces.json"),
        "authoritative_prepared_vector_spaces_sha256": file_hash(prepared_root / "vector_spaces.json"),
        "authoritative_raw_parameter_space": str(v3_raw_root / "parameter_space.json"),
        "authoritative_raw_parameter_space_sha256": file_hash(v3_raw_root / "parameter_space.json"),
        "sparse_100k_parameter_space": str(sparse_root / "parameter_space.json"),
        "sparse_100k_parameter_space_sha256": file_hash(sparse_root / "parameter_space.json"),
        "v3_nuisance_pair_and_sparse_100k_contracts_agree": all_agree,
        "components": rows,
    }
    payload["content_hash"] = content_hash({key: value for key, value in payload.items() if key != "content_hash"})
    return payload


@dataclass(frozen=True)
class NuisanceBank:
    name: str
    source_dataset: str
    labels: tuple[str, ...]
    physical_by_id: dict[str, tuple[float, ...]]
    sigma_by_id: dict[str, tuple[float, ...]]
    provenance: dict[str, Any]

    def to_rows(self, *, vector_space_id: str) -> list[dict[str, Any]]:
        rows = []
        for bank_index, nuisance_id in enumerate(sorted(self.physical_by_id, key=lambda value: int(value) if value.isdigit() else value)):
            physical = self.physical_by_id[nuisance_id]
            fisher = self.sigma_by_id.get(nuisance_id, ())
            stable_id = nuisance_state_id(vector_space_id, physical)
            rows.append(
                {
                    "nuisance_state_id": stable_id,
                    "nuisance_vector_space_id": vector_space_id,
                    "bank_index": bank_index,
                    "source_nuisance_id": nuisance_id,
                    "labels": list(self.labels),
                    "ordered_physical_nuisance_vector": list(physical),
                    "ordered_fisher_scaled_nuisance_vector": list(fisher),
                    "physical_values": list(physical),
                    "fisher_scaled_values": list(fisher),
                }
            )
        return rows

    def to_contract(self, bank_identity: str, *, vector_space_id: str) -> dict[str, Any]:
        states = self.to_rows(vector_space_id=vector_space_id)
        scientific_states = [
            {
                "nuisance_state_id": row["nuisance_state_id"],
                "nuisance_vector_space_id": row["nuisance_vector_space_id"],
                "ordered_physical_nuisance_vector": row["ordered_physical_nuisance_vector"],
                "ordered_fisher_scaled_nuisance_vector": row["ordered_fisher_scaled_nuisance_vector"],
            }
            for row in sorted(states, key=lambda item: str(item["nuisance_state_id"]))
        ]
        scientific_identity = {
            "schema_version": "shera_v4_nuisance_bank_scientific_identity/1",
            "bank_identity": bank_identity,
            "ordered_labels": list(self.labels),
            "states": scientific_states,
        }
        payload = {
            "schema_version": "shera_v4_nuisance_bank/1",
            "bank_identity": bank_identity,
            "nuisance_vector_space_id": vector_space_id,
            "name": self.name,
            "source_dataset": self.source_dataset,
            "ordered_labels": list(self.labels),
            "states": states,
            "provenance": self.provenance,
            "scientific_identity": scientific_identity,
        }
        payload["content_hash"] = content_hash(scientific_identity)
        return payload


def recover_nuisance_bank(root: Path, *, name: str) -> NuisanceBank:
    manifest = read_json(root / "manifest.json")
    manifest_labels = tuple(str(key) for key in manifest["nuisance_config"]["keys"])
    duplicates = _duplicate_values(manifest_labels)
    missing = [label for label in CANONICAL_NUISANCE_LABELS if label not in manifest_labels]
    unexpected = sorted(label for label in manifest_labels if label not in set(CANONICAL_NUISANCE_LABELS))
    if duplicates or missing or unexpected:
        raise ValueError(
            f"{root} nuisance_config keys do not match the canonical V4 nuisance labels: "
            f"duplicates={duplicates}, missing={missing}, unexpected={unexpected}"
        )
    labels = CANONICAL_NUISANCE_LABELS
    physical: dict[str, set[tuple[float, ...]]] = defaultdict(set)
    sigma: dict[str, set[tuple[float, ...]]] = defaultdict(set)
    for row in _read_jsonl(root / "samples.jsonl"):
        nuisance_id = row.get("nuisance_id")
        if nuisance_id in (None, "", "unavailable"):
            continue
        key = str(nuisance_id)
        values = row.get("registration_nuisance_values", {}) or {}
        sigma_values = row.get("registration_nuisance_sigma_values", {}) or {}
        if not isinstance(values, Mapping):
            raise ValueError(f"{root} row for nuisance_id {key} has non-mapping nuisance values.")
        value_missing = [label for label in labels if label not in values]
        value_unexpected = sorted(str(label) for label in values if str(label) not in set(labels))
        if value_missing or value_unexpected:
            raise ValueError(
                f"{root} nuisance_id {key} values do not match canonical nuisance labels: "
                f"missing={value_missing}, unexpected={value_unexpected}"
            )
        physical[key].add(tuple(float(values[label]) for label in labels))
        if not isinstance(sigma_values, Mapping):
            raise ValueError(f"{root} row for nuisance_id {key} has non-mapping nuisance sigma values.")
        sigma_missing = [label for label in labels if label not in sigma_values]
        sigma_unexpected = sorted(str(label) for label in sigma_values if str(label) not in set(labels))
        if sigma_missing or sigma_unexpected:
            raise ValueError(
                f"{root} nuisance_id {key} sigma values do not match canonical nuisance labels: "
                f"missing={sigma_missing}, unexpected={sigma_unexpected}"
            )
        sigma[key].add(tuple(float(sigma_values[label]) for label in labels))
    physical_one = {key: next(iter(values)) for key, values in physical.items() if len(values) == 1}
    sigma_one = {key: next(iter(values)) for key, values in sigma.items() if len(values) == 1}
    bad = {key: len(values) for key, values in physical.items() if len(values) != 1}
    if bad:
        raise ValueError(f"{root} has non-unique nuisance vectors by id: {bad}")
    return NuisanceBank(
        name=name,
        source_dataset=str(root),
        labels=labels,
        physical_by_id=physical_one,
        sigma_by_id=sigma_one,
        provenance={
            "manifest_sha256": file_hash(root / "manifest.json"),
            "samples_sha256": file_hash(root / "samples.jsonl"),
            "source_seed": manifest.get("seed"),
            "source_git_info": manifest.get("git_info"),
            "canonical_order_source": "canonical V4 registration nuisance contract",
            "source_manifest_labels": list(manifest_labels),
        },
    )


def compare_nuisance_banks(v3: NuisanceBank, sparse: NuisanceBank, *, vector_space_id: str) -> dict[str, Any]:
    v3_rows = v3.to_rows(vector_space_id=vector_space_id)
    sparse_rows = sparse.to_rows(vector_space_id=vector_space_id)
    equal = v3.labels == sparse.labels and len(v3_rows) == len(sparse_rows)
    differences = []
    if equal:
        for left, right in zip(v3_rows, sparse_rows):
            physical_equal = np.array_equal(np.asarray(left["physical_values"]), np.asarray(right["physical_values"]))
            sigma_equal = np.array_equal(np.asarray(left["fisher_scaled_values"]), np.asarray(right["fisher_scaled_values"]))
            if not (physical_equal and sigma_equal):
                equal = False
                differences.append(
                    {
                        "bank_index": left["bank_index"],
                        "v3_source_nuisance_id": left["source_nuisance_id"],
                        "sparse_source_nuisance_id": right["source_nuisance_id"],
                        "physical_equal": physical_equal,
                        "fisher_scaled_equal": sigma_equal,
                        "v3_physical_values": left["physical_values"],
                        "sparse_physical_values": right["physical_values"],
                        "v3_fisher_scaled_values": left["fisher_scaled_values"],
                        "sparse_fisher_scaled_values": right["fisher_scaled_values"],
                    }
                )
    else:
        differences.append({"reason": "labels or bank sizes differ"})
    selected_identity = "nuisance_bank_v4_s01_s05_v3_training_10"
    payload = {
        "schema_version": "shera_v4_nuisance_bank_comparison/1",
        "v3_training_bank": v3.to_contract("candidate_v3_training", vector_space_id=vector_space_id),
        "sparse_100k_legacy_bank": sparse.to_contract("candidate_sparse_100k_legacy", vector_space_id=vector_space_id),
        "equal": equal,
        "differences": differences,
        "selected_bank_identity": selected_identity,
        "legacy_bank_identity": "nuisance_bank_legacy_sparse100k_10",
        "selection_rationale": "The first V4 campaign changes science-state geometry while holding the S01/S05 nuisance identities fixed.",
    }
    payload["content_hash"] = content_hash({key: value for key, value in payload.items() if key != "content_hash"})
    return payload


def split_bin_counts(total: int, bins: Sequence[tuple[float, float]]) -> list[int]:
    base, rem = divmod(total, len(bins))
    return [base + (1 if idx < rem else 0) for idx in range(len(bins))]


def component_family(label: str) -> str:
    if label == "optics.plate_scale_as_per_pix":
        return "plate_scale"
    if label.startswith("optics.primary.zernike"):
        return "M1_zernike"
    if label.startswith("optics.secondary.zernike"):
        return "M2_zernike"
    if label.startswith("source."):
        return "source"
    return "other"


def state_id(family: str, split: str, vector_space_id: str, physical_state: Sequence[float], fisher_delta: Sequence[float], sampling_contract_id: str) -> str:
    return "science_" + content_hash(
        {
            "dataset_version": DATASET_VERSION,
            "family": family,
            "split_role": split,
            "science_vector_space_id": vector_space_id,
            "ordered_physical_science_vector": [float(v) for v in physical_state],
            "ordered_fisher_scaled_delta": [float(v) for v in fisher_delta],
            "sampling_family_contract_identity": sampling_contract_id,
        }
    )[:32]


def nuisance_state_id(vector_space_id: str, nuisance_vector: Sequence[float]) -> str:
    return "nuisance_" + content_hash(
        {"nuisance_vector_space_id": vector_space_id, "ordered_nuisance_vector": [float(v) for v in nuisance_vector]}
    )[:32]


def build_render_system_contract(
    *,
    resolved_prescription_path: Path,
    manifest_path: Path | None = None,
    coordinate_source_sha256: str | None = None,
    local_source_path: Path | None = None,
) -> dict[str, Any]:
    prescription = read_json(resolved_prescription_path)
    if not isinstance(prescription, Mapping) or not isinstance(prescription.get("system"), Mapping):
        raise ValueError(f"{resolved_prescription_path} must contain a resolved system subtree.")
    manifest = read_json(manifest_path) if manifest_path is not None and manifest_path.exists() else {}
    system = dict(prescription["system"])
    base_dir = resolved_prescription_path.parent
    scientific_system = canonicalize_scientific_paths(system, base_dir=base_dir)
    noise_policy = manifest.get("noise_config", {"enabled": False, "add_noise": False}) if isinstance(manifest, Mapping) else {"enabled": False, "add_noise": False}
    scientific_identity = {
        "schema_version": "shera_v4_render_system_scientific_identity/1",
        "resolved_system": scientific_system,
        "noise_policy": noise_policy,
    }
    if coordinate_source_sha256 is not None:
        scientific_identity["coordinate_source_sha256"] = coordinate_source_sha256
    summary = {
        "system_preset": system.get("preset"),
        "source_kind": (system.get("source") or {}).get("kind") if isinstance(system.get("source"), Mapping) else None,
        "source_target": (system.get("source") or {}).get("target") if isinstance(system.get("source"), Mapping) else None,
        "wavelength_m": (system.get("source") or {}).get("wavelength_m") if isinstance(system.get("source"), Mapping) else None,
        "bandwidth_m": (system.get("source") or {}).get("bandwidth_m") if isinstance(system.get("source"), Mapping) else None,
        "n_lambda": (system.get("source") or {}).get("n_lambda") if isinstance(system.get("source"), Mapping) else None,
        "exposure_time_s": (system.get("source") or {}).get("exposure_time_s") if isinstance(system.get("source"), Mapping) else None,
        "optics_kind": (system.get("optics") or {}).get("kind") if isinstance(system.get("optics"), Mapping) else None,
        "pupil_npix": (system.get("optics") or {}).get("pupil_npix") if isinstance(system.get("optics"), Mapping) else None,
        "psf_npix": (system.get("optics") or {}).get("psf_npix") if isinstance(system.get("optics"), Mapping) else None,
        "pixel_pitch_m": (system.get("optics") or {}).get("pixel_pitch_m") if isinstance(system.get("optics"), Mapping) else None,
        "oversample": (system.get("optics") or {}).get("oversample") if isinstance(system.get("optics"), Mapping) else None,
        "detector_model": (system.get("detector") or {}).get("model") if isinstance(system.get("detector"), Mapping) else None,
        "detector_layers": (system.get("detector") or {}).get("layers") if isinstance(system.get("detector"), Mapping) else None,
        "noise_policy": noise_policy,
    }
    contract_hash = content_hash(scientific_identity)
    payload = {
        "schema_version": RENDER_SYSTEM_CONTRACT_SCHEMA,
        "source_prescription_logical_identity": resolved_prescription_path.name,
        "source_prescription_sha256": file_hash(resolved_prescription_path),
        "source_commit_provenance": (manifest.get("git_info") or manifest.get("git") or {}) if isinstance(manifest, Mapping) else {},
        "source_artifact_local_path": None if local_source_path is None else str(local_source_path),
        "resolved_system": system,
        "scientific_resolved_system": scientific_system,
        "scientific_identity": scientific_identity,
        "summary": summary,
        "render_system_contract_hash": contract_hash,
        "content_hash": contract_hash,
        "path_policy": "Local paths are provenance. Scientific identity uses resolved system content; file references contribute by content hash when available.",
    }
    return payload


def load_science_plans(outdir: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    plans: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for family in FAMILY_ORDER:
        plans[family] = {}
        for split in SPLIT_ORDER:
            plans[family][split] = list(_read_jsonl(outdir / "state_plans" / family / f"{split}.jsonl"))
    return plans


def validate_split_integrity(plans: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]) -> dict[str, Any]:
    same_family_cross_split: list[dict[str, Any]] = []
    cross_family_cross_split: list[dict[str, Any]] = []
    same_split_cross_family: list[dict[str, Any]] = []
    duplicate_science_ids: list[dict[str, Any]] = []
    duplicate_physical_vectors: list[dict[str, Any]] = []
    train_test_collisions: list[dict[str, Any]] = []

    id_locations: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    vector_locations: dict[str, list[tuple[str, str, int, str]]] = defaultdict(list)
    counts: dict[str, dict[str, int]] = {}
    for family, by_split in plans.items():
        counts[str(family)] = {}
        for split, rows in by_split.items():
            counts[str(family)][str(split)] = len(rows)
            for idx, row in enumerate(rows):
                science_id = str(row["science_state_id"])
                id_locations[science_id].append((str(family), str(split), idx))
                vector_locations[_science_vector_key(row)].append((str(family), str(split), idx, science_id))

    def record_collision(target: list[dict[str, Any]], kind: str, key: str, locations: Sequence[tuple[Any, ...]]) -> None:
        target.append(
            {
                "kind": kind,
                "key": key,
                "locations": [
                    {"family": item[0], "split_role": item[1], "row_index": item[2], "science_state_id": item[3] if len(item) > 3 else None}
                    for item in locations[:20]
                ],
                "location_count": len(locations),
            }
        )

    for science_id, locations in id_locations.items():
        if len(locations) > 1:
            record_collision(duplicate_science_ids, "duplicate_science_id", science_id, locations)
    for vector_key, locations in vector_locations.items():
        if len(locations) > 1:
            record_collision(duplicate_physical_vectors, "duplicate_physical_science_vector", vector_key, locations)
            families = {item[0] for item in locations}
            splits = {item[1] for item in locations}
            for family in families:
                family_splits = {item[1] for item in locations if item[0] == family}
                if len(family_splits) > 1:
                    record_collision(same_family_cross_split, "same_family_cross_split_collision", vector_key, locations)
                    break
            if len(families) > 1 and len(splits) > 1:
                record_collision(cross_family_cross_split, "cross_family_cross_split_collision", vector_key, locations)
            if len(families) > 1:
                for split in splits:
                    if len({item[0] for item in locations if item[1] == split}) > 1:
                        record_collision(same_split_cross_family, "same_split_cross_family_overlap", vector_key, locations)
                        break
            has_train = any(item[1] == "train" for item in locations)
            has_test = any(item[1] == "test" for item in locations)
            if has_train and has_test:
                record_collision(train_test_collisions, "new_v4_train_test_collision", vector_key, locations)

    prohibited = {
        "duplicate_science_id_count": len(duplicate_science_ids),
        "duplicate_physical_vector_count": len(duplicate_physical_vectors),
        "same_family_cross_split_collision_count": len(same_family_cross_split),
        "cross_family_cross_split_collision_count": len(cross_family_cross_split),
        "same_split_cross_family_overlap_count": len(same_split_cross_family),
        "new_v4_train_test_collision_count": len(train_test_collisions),
    }
    payload = {
        "schema_version": SPLIT_INTEGRITY_SCHEMA,
        "status": "PASS" if all(value == 0 for value in prohibited.values()) else "FAIL",
        "family_order": list(FAMILY_ORDER),
        "split_order": list(SPLIT_ORDER),
        "science_counts": counts,
        "prohibited_overlap_counts": prohibited,
        "same_family_cross_split_collisions": same_family_cross_split[:100],
        "cross_family_cross_split_collisions": cross_family_cross_split[:100],
        "same_split_cross_family_overlaps": same_split_cross_family[:100],
        "duplicate_science_ids": duplicate_science_ids[:100],
        "duplicate_physical_vectors": duplicate_physical_vectors[:100],
        "new_v4_train_test_collisions": train_test_collisions[:100],
        "legacy_family_overlap": {
            "status": "not_evaluated",
            "policy": "Recoverable V3/legacy overlaps are provenance notes, not V4 materialization failures.",
            "count": None,
        },
    }
    payload["content_hash"] = content_hash({key: value for key, value in payload.items() if key != "content_hash"})
    if payload["status"] != "PASS":
        raise ValueError(f"V4 split integrity validation failed: {prohibited}")
    return payload


def summarize_matrix(matrix: np.ndarray, labels: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for idx, label in enumerate(labels):
        col = matrix[:, idx]
        rows.append(
            {
                "index": idx,
                "label": label,
                "min": float(np.min(col)),
                "p01": float(np.percentile(col, 1)),
                "p50": float(np.percentile(col, 50)),
                "p99": float(np.percentile(col, 99)),
                "max": float(np.max(col)),
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
            }
        )
    return rows


def radius_summary(values: np.ndarray) -> dict[str, float]:
    return {f"p{p:02d}" if p else "min": float(np.percentile(values, p)) for p in (0, 1, 5, 25, 50, 75, 95, 99)} | {"max": float(np.max(values))}


def contribution_summary(fisher: np.ndarray, labels: Sequence[str]) -> dict[str, Any]:
    squared = fisher**2
    denom = float(np.sum(squared))
    rows = []
    family = Counter()
    for idx, label in enumerate(labels):
        fraction = 0.0 if denom == 0.0 else float(np.sum(squared[:, idx]) / denom)
        rows.append({"index": idx, "label": label, "family": component_family(label), "fraction_of_total_radius_squared": fraction})
        family[component_family(label)] += fraction
    family_rows = [{"family": key, "fraction_of_total_radius_squared": float(value)} for key, value in sorted(family.items())]
    flags = []
    for row in rows:
        if row["fraction_of_total_radius_squared"] > 0.30:
            flags.append({"type": "single_coordinate_domination", **row})
    for row in family_rows:
        if row["fraction_of_total_radius_squared"] > 0.50:
            flags.append({"type": "family_domination", **row})
    return {"by_parameter": rows, "by_family": family_rows, "domination_flags": flags}


def generate_joint(
    *,
    outdir: Path,
    labels: Sequence[str],
    nominal: np.ndarray,
    sigmas: np.ndarray,
    base_halfwidth: np.ndarray,
    vector_space_id: str,
    sampling_contract_id: str,
    train_prefixes: Sequence[int] = TRAIN_PREFIXES,
) -> tuple[dict[str, Any], dict[str, str]]:
    plan_dir = outdir / "state_plans" / "joint_full_v4"
    split_hashes: dict[str, str] = {}
    qa: dict[str, Any] = {"schema_version": "shera_v4_joint_geometry_qa/1", "scale_strata": []}
    all_fisher: list[np.ndarray] = []
    for split, total in JOINT_COUNTS.items():
        per_stratum = total // len(JOINT_SCALE_STRATA)
        strata_samples: list[np.ndarray] = []
        for multiplier in JOINT_SCALE_STRATA:
            seed = stable_seed(DATASET_VERSION, "joint_full_v4", split, multiplier)
            sampler = qmc.Sobol(d=len(labels), scramble=True, seed=seed)
            unit = sampler.random_base2(int(math.log2(per_stratum)))
            z = (2.0 * unit - 1.0) * base_halfwidth * float(multiplier)
            strata_samples.append(z)
        rows = []
        split_fisher = []
        for within_idx in range(per_stratum):
            for stratum_idx, multiplier in enumerate(JOINT_SCALE_STRATA):
                global_idx = len(rows)
                fisher = strata_samples[stratum_idx][within_idx]
                delta = fisher * sigmas
                physical = nominal + delta
                sid = state_id("joint_full_v4", split, vector_space_id, physical, fisher, sampling_contract_id)
                rows.append(
                    {
                        "schema_version": STATE_PLAN_SCHEMA,
                        "dataset_version": DATASET_VERSION,
                        "dataset_family": "joint_full_v4",
                        "split_role": split,
                        "science_state_id": sid,
                        "science_vector_space_id": vector_space_id,
                        "sampling_family_contract_identity": sampling_contract_id,
                        "sampling_method": "scrambled_sobol_multiscale_box",
                        "sampling_seed": stable_seed(DATASET_VERSION, "joint_full_v4", split, multiplier),
                        "scale_stratum": stratum_idx,
                        "scale_multiplier": float(multiplier),
                        "within_stratum_sequence_index": within_idx,
                        "global_sequence_index": global_idx,
                        "ordered_physical_science_vector": physical.tolist(),
                        "ordered_physical_delta_vector": delta.tolist(),
                        "ordered_fisher_scaled_delta": fisher.tolist(),
                        "fisher_radius_l2": float(np.linalg.norm(fisher)),
                    }
                )
                split_fisher.append(fisher)
        path = plan_dir / f"{split}.jsonl"
        write_jsonl(path, rows)
        split_hashes[split] = file_hash(path)
        split_arr = np.asarray(split_fisher)
        all_fisher.append(split_arr)
        for stratum_idx, multiplier in enumerate(JOINT_SCALE_STRATA):
            stratum_arr = split_arr[stratum_idx::len(JOINT_SCALE_STRATA)]
            qa["scale_strata"].append(
                {
                    "split_role": split,
                    "scale_stratum": stratum_idx,
                    "scale_multiplier": float(multiplier),
                    "count": int(stratum_arr.shape[0]),
                    "radius_percentiles": radius_summary(np.linalg.norm(stratum_arr, axis=1)),
                }
            )
        qa.setdefault("split_comparisons", {})[split] = {
            "count": total,
            "radius_percentiles": radius_summary(np.linalg.norm(split_arr, axis=1)),
            "scale_stratum_counts": {str(idx): int(np.sum(split_arr[range(idx, total, len(JOINT_SCALE_STRATA))].shape[0])) for idx in range(len(JOINT_SCALE_STRATA))},
        }
        if split == "train":
            qa["nested_prefix_scale_counts"] = {
                str(prefix): dict(Counter(str(rows[idx]["scale_stratum"]) for idx in range(prefix))) for prefix in train_prefixes
            }
    combined = np.vstack(all_fisher)
    qa["combined"] = {
        "count": int(combined.shape[0]),
        "radius_percentiles": radius_summary(np.linalg.norm(combined, axis=1)),
        "per_coordinate_fisher_distributions": summarize_matrix(combined, labels),
        "per_coordinate_physical_delta_distributions": summarize_matrix(combined * sigmas, labels),
        "squared_radius_contribution": contribution_summary(combined, labels),
        "domination_checks": {
            "plate_scale_fraction": next(row["fraction_of_total_radius_squared"] for row in contribution_summary(combined, labels)["by_parameter"] if row["label"] == "optics.plate_scale_as_per_pix"),
            "contrast_fraction": next(row["fraction_of_total_radius_squared"] for row in contribution_summary(combined, labels)["by_parameter"] if row["label"] == "source.contrast"),
            "flags": contribution_summary(combined, labels)["domination_flags"],
        },
    }
    qa["content_hash"] = content_hash({key: value for key, value in qa.items() if key != "content_hash"})
    write_json(outdir / "qa" / "joint_geometry_summary.json", qa)
    _write_csv(outdir / "qa" / "joint_squared_radius_contribution.csv", qa["combined"]["squared_radius_contribution"]["by_parameter"])
    return qa, split_hashes


def generate_radial(
    *,
    outdir: Path,
    labels: Sequence[str],
    nominal: np.ndarray,
    sigmas: np.ndarray,
    base_halfwidth: np.ndarray,
    vector_space_id: str,
    sampling_contract_id: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    plan_dir = outdir / "state_plans" / "radial_capture_v4"
    split_hashes: dict[str, str] = {}
    qa: dict[str, Any] = {"schema_version": "shera_v4_radial_feasibility_qa/1", "bin_allocations": {}, "bins": []}
    all_direction_by_bin: dict[str, list[np.ndarray]] = defaultdict(list)
    for split, total in RADIAL_COUNTS.items():
        counts = split_bin_counts(total, RADIAL_BINS)
        qa["bin_allocations"][split] = {
            f"{low:g}-{high:g}": count for (low, high), count in zip(RADIAL_BINS, counts)
        }
        rows = []
        split_requested = []
        split_actual = []
        split_feasible = []
        for bin_idx, ((low, high), count) in enumerate(zip(RADIAL_BINS, counts)):
            rng_direction = np.random.default_rng(stable_seed(DATASET_VERSION, "radial_capture_v4", split, bin_idx, "direction"))
            rng_radius = np.random.default_rng(stable_seed(DATASET_VERSION, "radial_capture_v4", split, bin_idx, "radius"))
            accepted = 0
            attempts_since_accept = 0
            total_attempts = 0
            while accepted < count:
                attempts_since_accept += 1
                total_attempts += 1
                direction_sequence_index = total_attempts - 1
                raw = rng_direction.normal(size=len(labels))
                direction = raw / np.linalg.norm(raw)
                requested = float(rng_radius.uniform(low, high))
                limits = np.divide(base_halfwidth, np.abs(direction), out=np.full_like(base_halfwidth, np.inf), where=np.abs(direction) > 0.0)
                feasible = float(np.min(limits))
                if requested > feasible:
                    if total_attempts > max(1000000, count * 10000):
                        raise RuntimeError(f"Unable to fill radial bin {split} {low}-{high}; accepted {accepted}/{count}.")
                    continue
                fisher = requested * direction
                delta = fisher * sigmas
                physical = nominal + delta
                global_idx = len(rows)
                sid = state_id("radial_capture_v4", split, vector_space_id, physical, fisher, sampling_contract_id)
                rows.append(
                    {
                        "schema_version": STATE_PLAN_SCHEMA,
                        "dataset_version": DATASET_VERSION,
                        "dataset_family": "radial_capture_v4",
                        "split_role": split,
                        "science_state_id": sid,
                        "science_vector_space_id": vector_space_id,
                        "sampling_family_contract_identity": sampling_contract_id,
                        "sampling_method": "isotropic_direction_radius_rejection",
                        "sampling_seed": stable_seed(DATASET_VERSION, "radial_capture_v4", split, bin_idx),
                        "global_sequence_index": global_idx,
                        "radial_bin": f"{low:g}-{high:g}",
                        "radial_bin_index": bin_idx,
                        "requested_fisher_radius": requested,
                        "actual_fisher_radius": requested,
                        "feasible_fisher_radius": feasible,
                        "attempt_count": attempts_since_accept,
                        "direction_sequence_index": direction_sequence_index,
                        "ordered_physical_science_vector": physical.tolist(),
                        "ordered_physical_delta_vector": delta.tolist(),
                        "ordered_fisher_scaled_delta": fisher.tolist(),
                        "fisher_radius_l2": float(np.linalg.norm(fisher)),
                    }
                )
                split_requested.append(requested)
                split_actual.append(float(np.linalg.norm(fisher)))
                split_feasible.append(feasible)
                all_direction_by_bin[f"{low:g}-{high:g}"].append(direction)
                accepted += 1
                attempts_since_accept = 0
            qa["bins"].append(
                {
                    "split_role": split,
                    "radial_bin": f"{low:g}-{high:g}",
                    "count": count,
                    "total_attempts": total_attempts,
                    "accepted": count,
                    "rejected": total_attempts - count,
                    "acceptance_rate": float(count / total_attempts),
                    "rejection_rate": float((total_attempts - count) / total_attempts),
                }
            )
        path = plan_dir / f"{split}.jsonl"
        write_jsonl(path, rows)
        split_hashes[split] = file_hash(path)
        qa.setdefault("split_summaries", {})[split] = {
            "requested_radius_percentiles": radius_summary(np.asarray(split_requested)),
            "accepted_radius_percentiles": radius_summary(np.asarray(split_actual)),
            "feasible_radius_percentiles": radius_summary(np.asarray(split_feasible)),
        }
    direction_rows = []
    reference_std = None
    flags = []
    for bin_label, directions in sorted(all_direction_by_bin.items()):
        arr = np.asarray(directions)
        std = np.std(arr, axis=0)
        mean_abs = np.mean(np.abs(arr), axis=0)
        if reference_std is None:
            reference_std = std
        for idx, label in enumerate(labels):
            ratio = float(std[idx] / reference_std[idx]) if reference_std[idx] > 0.0 else 1.0
            row = {
                "radial_bin": bin_label,
                "index": idx,
                "label": label,
                "direction_mean": float(np.mean(arr[:, idx])),
                "direction_std": float(std[idx]),
                "mean_abs_direction_component": float(mean_abs[idx]),
                "std_ratio_vs_lowest_bin": ratio,
            }
            direction_rows.append(row)
            if ratio < 0.5 or ratio > 1.8:
                flags.append({"type": "direction_conditioning", **row})
    qa["direction_conditioning"] = {
        "per_bin_coordinate_distributions": direction_rows,
        "flags": flags,
        "policy": "flag std ratios below 0.5 or above 1.8 relative to the lowest radial bin",
    }
    qa["content_hash"] = content_hash({key: value for key, value in qa.items() if key != "content_hash"})
    write_json(outdir / "qa" / "radial_feasibility_summary.json", qa)
    _write_csv(outdir / "qa" / "radial_direction_conditioning.csv", direction_rows)
    return qa, split_hashes


def render_index_contract(
    *,
    outdir: Path,
    science_counts: Mapping[str, Mapping[str, int]],
    science_plan_hashes: Mapping[str, Mapping[str, str]],
    nuisance_bank_contract: Mapping[str, Any],
    render_system_contract_hash: str,
    science_vector_space_id: str,
) -> dict[str, Any]:
    families = []
    offset = 0
    for family in FAMILY_ORDER:
        for split in SPLIT_ORDER:
            count = int(science_counts[family][split])
            families.append(
                {
                    "family": family,
                    "split_role": split,
                    "science_start_index": offset,
                    "science_stop_index_exclusive": offset + count,
                    "science_count": count,
                    "first_render_index": offset * len(nuisance_bank_contract["states"]),
                    "last_render_index": (offset + count) * len(nuisance_bank_contract["states"]) - 1,
                    "plan_path": f"state_plans/{family}/{split}.jsonl",
                    "plan_sha256": science_plan_hashes[family][split],
                }
            )
            offset += count
    nuisance_count = len(nuisance_bank_contract["states"])
    render_count = offset * nuisance_count
    last_render_index = render_count - 1
    payload = {
        "schema_version": RENDER_CONTRACT_SCHEMA,
        "render_system_contract_hash": render_system_contract_hash,
        "render_model_system_contract_identity": render_system_contract_hash,
        "science_vector_space_id": science_vector_space_id,
        "nuisance_vector_space_id": nuisance_bank_contract["nuisance_vector_space_id"],
        "nuisance_bank_identity": nuisance_bank_contract["bank_identity"],
        "nuisance_bank_scientific_content_hash": nuisance_bank_contract["content_hash"],
        "nuisance_count": nuisance_count,
        "science_count": offset,
        "render_count": render_count,
        "render_index_range": [0, last_render_index],
        "mapping": "render_index = science_global_index * nuisance_count + nuisance_bank_index",
        "inverse_mapping": {
            "science_global_index": "render_index // nuisance_count",
            "nuisance_bank_index": "render_index % nuisance_count",
        },
        "render_state_id_formula": "render_state_id = hash(science_state_id, nuisance_state_id, render_system_contract_hash)",
        "family_order": list(FAMILY_ORDER),
        "split_order": list(SPLIT_ORDER),
        "nuisance_order": [
            {
                "nuisance_bank_index": int(row["bank_index"]),
                "nuisance_state_id": row["nuisance_state_id"],
            }
            for row in nuisance_bank_contract["states"]
        ],
        "families": families,
        "storage_layout_independence": "No filesystem path, shard number, timestamp, or SLURM id participates in scientific or render identity.",
    }
    payload["content_hash"] = content_hash({key: value for key, value in payload.items() if key != "content_hash"})
    write_json(outdir / "render_contract.json", payload)
    return payload


def render_scale_summary(render_count: int, *, bytes_per_observed_fits: int = 210240, image_shape: tuple[int, int] = (160, 160)) -> dict[str, Any]:
    pixels = image_shape[0] * image_shape[1]
    raw_float64 = render_count * pixels * 8
    fits = render_count * bytes_per_observed_fits
    metadata = render_count * 4096
    prepared_float32 = render_count * pixels * 4
    payload = {
        "schema_version": "shera_v4_render_scale_summary/1",
        "render_count": render_count,
        "image_shape": list(image_shape),
        "raw_float64_payload_bytes": raw_float64,
        "expected_fits_payload_bytes": fits,
        "expected_metadata_payload_bytes": metadata,
        "potential_prepared_float32_payload_bytes": prepared_float32,
        "total_estimated_persistent_footprint_bytes": fits + metadata + prepared_float32,
        "measured_historical_bytes_per_fits_sample": bytes_per_observed_fits,
        "prelaunch_gate": [
            "df -h /projects/shera_hpc",
            "du -sh /projects/shera_hpc/data/ml_training/*",
        ],
    }
    payload["content_hash"] = content_hash({key: value for key, value in payload.items() if key != "content_hash"})
    return payload


def review_decisions(joint_qa: Mapping[str, Any], radial_qa: Mapping[str, Any]) -> dict[str, Any]:
    radial_bins = []
    for row in radial_qa.get("bins", []):
        radial_bins.append(
            {
                "split_role": row["split_role"],
                "radial_bin": row["radial_bin"],
                "count": row["count"],
                "total_attempts": row["total_attempts"],
                "accepted": row["accepted"],
                "rejected": row["rejected"],
                "acceptance_rate": row["acceptance_rate"],
                "rejection_rate": row["rejection_rate"],
            }
        )
    joint_domination = joint_qa["combined"]["squared_radius_contribution"]
    max_coordinate = max(joint_domination["by_parameter"], key=lambda item: float(item["fraction_of_total_radius_squared"]))
    payload = {
        "schema_version": REVIEW_DECISIONS_SCHEMA,
        "dataset_version": DATASET_VERSION,
        "joint_geometry": {
            "status": "accepted",
            "finding": "joint_full_v4 uses the accepted anisotropic Fisher envelope and four balanced multiscale strata.",
            "rationale": "M1 contributes a relatively large aggregate squared-radius share because there are 8 M1 coordinates and the accepted M1 envelope is wider than M2.",
            "max_single_coordinate_fraction_of_total_radius_squared": max_coordinate,
            "single_coordinate_domination_status": "accepted_no_unintentional_single_coordinate_domination",
        },
        "radial_geometry": {
            "status": "accepted",
            "finding": "Directions are isotropic proposals accepted only when they remain inside the fixed V4 sampling/feasibility envelope.",
            "high_radius_interpretation": "High-radius bins, especially 1000-1500 and 1500-2000, are isotropic proposals conditioned on feasibility, not unconditioned isotropic shells.",
            "acceptance_rejection_by_bin": radial_bins,
        },
        "boundary_stress_v4": {
            "status": "deferred",
            "campaign": "first_render_campaign",
        },
    }
    payload["content_hash"] = content_hash({key: value for key, value in payload.items() if key != "content_hash"})
    return payload


def _format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    size = float(value)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{value} B"


def write_render_handoff(
    outdir: Path,
    *,
    master: Mapping[str, Any],
    render_system_contract: Mapping[str, Any],
    render_contract: Mapping[str, Any],
    split_integrity: Mapping[str, Any],
    review: Mapping[str, Any],
    scale: Mapping[str, Any],
) -> None:
    summary = render_system_contract["summary"]
    counts = master["science_counts"]
    lines = [
        "# V4 Render Handoff",
        "",
        "LOCAL STATE-PLAN DESIGN COMPLETE",
        "",
        "This is the frozen local V4 state-plan handoff for Gattaca2 render-campaign configuration. It does not render images, transfer files, submit SLURM jobs, or contact Gattaca2.",
        "",
        "The cluster renderer must consume these frozen plans. It must not regenerate science or nuisance sampling.",
        "",
        "## Dataset identity",
        "",
        f"- dataset version: `{master['dataset_version']}`",
        f"- master scientific content hash: `{master['master_scientific_content_hash']}`",
        f"- aggregate vector-space contract hash: `{master['vector_space_hash']}`",
        f"- science vector-space ID: `{master['science_vector_space_id']}`",
        f"- nuisance vector-space ID: `{master['nuisance_vector_space_id']}`",
        f"- nuisance-bank hash: `{master['selected_nuisance_bank_hash']}`",
        f"- render-system-contract hash: `{master['render_system_contract_hash']}`",
        f"- render-contract hash: `{master['render_contract_hash']}`",
        f"- review-decisions hash: `{master['review_decisions_hash']}`",
        f"- transfer package: `{master.get('transfer_package_filename', 'not yet packaged')}`",
        f"- transfer package SHA256: `{master.get('transfer_package_sha256', 'not yet packaged')}`",
        "",
        "## Counts",
        "",
    ]
    for family in FAMILY_ORDER:
        family_total = sum(int(counts[family][split]) for split in SPLIT_ORDER)
        lines.extend(
            [
                f"- {family}: train {counts[family]['train']}, validation {counts[family]['validation']}, test {counts[family]['test']}, total {family_total}",
            ]
        )
    lines.extend(
        [
            f"- total science states: {render_contract['science_count']}",
            f"- nuisance states: {render_contract['nuisance_count']}",
            f"- total renders: {render_contract['render_count']}",
            f"- render-index range: 0 to {render_contract['render_index_range'][1]}",
            "",
            "## Render configuration",
            "",
            f"- system preset: `{summary['system_preset']}`",
            f"- source/target: `{summary['source_kind']}` / `{summary['source_target']}`",
            f"- exposure: `{summary['exposure_time_s']}` s",
            f"- wavelength/bandwidth: `{summary['wavelength_m']}` m / `{summary['bandwidth_m']}` m",
            f"- wavelength samples: `{summary['n_lambda']}`",
            f"- image shape: `{scale['image_shape']}`",
            f"- optics: `{summary['optics_kind']}`, pupil_npix `{summary['pupil_npix']}`, psf_npix `{summary['psf_npix']}`, oversample `{summary['oversample']}`",
            f"- detector identity: `{summary['detector_model']}` with layers `{json.dumps(json_ready(summary['detector_layers']), sort_keys=True)}`",
            f"- noise policy: `{json.dumps(json_ready(summary['noise_policy']), sort_keys=True)}`",
            "",
            "## QA state",
            "",
            f"- split leakage: {split_integrity['status']}",
            "- duplicate science IDs: PASS",
            "- joint geometry: reviewed / accepted",
            "- radial feasibility: reviewed / accepted",
            "- boundary stress: deferred",
            "",
            "## Storage estimate",
            "",
            "These are estimates and must be checked against actual renderer output.",
            "",
            f"- raw image payload: {_format_bytes(scale['raw_float64_payload_bytes'])}",
            f"- expected FITS payload: {_format_bytes(scale['expected_fits_payload_bytes'])}",
            f"- expected metadata: {_format_bytes(scale['expected_metadata_payload_bytes'])}",
            f"- potential prepared float32 payload: {_format_bytes(scale['potential_prepared_float32_payload_bytes'])}",
            f"- estimated total persistent footprint: {_format_bytes(scale['total_estimated_persistent_footprint_bytes'])}",
            "",
            "## Required cluster pre-launch gate",
            "",
            "Render submission must not proceed until Projects capacity is reviewed:",
            "",
            "```bash",
            "df -h /projects/shera_hpc",
            "du -sh /projects/shera_hpc/data/ml_training/*",
            "```",
            "",
            "## Reproduction",
            "",
            "Regenerate the frozen production state-plan artifact from tracked source/configuration:",
            "",
            "```bash",
            "python3 work/experiments/ml/datasets/materialize_master_v4.py \\",
            "  --outdir work/experiments/ml/datasets/materialized/master_v4 \\",
            "  --overwrite",
            "```",
            "",
            "Create the compact transfer package without rematerializing:",
            "",
            "```bash",
            "python3 work/experiments/ml/datasets/materialize_master_v4.py \\",
            "  --outdir work/experiments/ml/datasets/materialized/master_v4 \\",
            "  --package-only",
            "```",
            "",
            "Verify after transfer with `shasum -a 256 <package>` on macOS or `sha256sum <package>` on Linux.",
            "",
            "## Review decisions",
            "",
            f"- joint QA: {review['joint_geometry']['status']} - {review['joint_geometry']['rationale']}",
            f"- radial conditioning: {review['radial_geometry']['status']} - {review['radial_geometry']['high_radius_interpretation']}",
            f"- boundary_stress_v4: {review['boundary_stress_v4']['status']}",
        ]
    )
    (outdir / "RENDER_HANDOFF.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _set_handoff_package_fields(handoff_path: Path, *, filename: str, sha256: str) -> None:
    if not handoff_path.exists():
        return
    lines = []
    for line in handoff_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("- transfer package: `"):
            lines.append(f"- transfer package: `{filename}`")
        elif line.startswith("- transfer package SHA256: `"):
            lines.append(f"- transfer package SHA256: `{sha256}`")
        else:
            lines.append(line)
    handoff_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def package_transfer(outdir: Path, *, package_dir: Path | None = None) -> dict[str, Any]:
    outdir = Path(outdir)
    package_dir = outdir if package_dir is None else Path(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(outdir / "freeze_manifest.json")
    manifest.pop("transfer_package_filename", None)
    manifest.pop("transfer_package_sha256", None)
    write_json(outdir / "freeze_manifest.json", manifest)
    (outdir / "master_v4.yaml").write_text(yaml.safe_dump(json_ready(manifest), sort_keys=True), encoding="utf-8")
    _set_handoff_package_fields(
        outdir / "RENDER_HANDOFF.md",
        filename="not yet packaged",
        sha256="not yet packaged",
    )
    package_name = f"{manifest['dataset_version']}_{manifest['master_scientific_content_hash'][:16]}_stateplans.tar"
    package_path = package_dir / package_name
    include_paths = [Path(item) for item in TRANSFER_INCLUDE]
    for rel in include_paths:
        if not (outdir / rel).exists():
            raise FileNotFoundError(outdir / rel)
    with tempfile.NamedTemporaryFile(dir=package_dir, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with tarfile.open(tmp_path, "w", format=tarfile.PAX_FORMAT) as tar:
            for rel in include_paths:
                full = outdir / rel
                info = tar.gettarinfo(str(full), arcname=(outdir.name + "/" + rel.as_posix()))
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                if info.isfile():
                    with full.open("rb") as handle:
                        tar.addfile(info, handle)
                else:
                    tar.addfile(info)
        tmp_path.replace(package_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    sha = file_hash(package_path)
    payload = {
        "schema_version": TRANSFER_PACKAGE_SCHEMA,
        "package_filename": package_path.name,
        "package_path": str(package_path),
        "package_sha256": sha,
        "contained_master_scientific_hash": manifest["master_scientific_content_hash"],
        "created_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "included_paths": [item.as_posix() for item in include_paths],
        "verification": {
            "macos": f"shasum -a 256 {package_path.name}",
            "linux": f"sha256sum {package_path.name}",
        },
    }
    write_json(outdir / "transfer_package_manifest.json", payload)
    manifest["transfer_package_filename"] = package_path.name
    manifest["transfer_package_sha256"] = sha
    write_json(outdir / "freeze_manifest.json", manifest)
    (outdir / "master_v4.yaml").write_text(yaml.safe_dump(json_ready(manifest), sort_keys=True), encoding="utf-8")
    _set_handoff_package_fields(outdir / "RENDER_HANDOFF.md", filename=package_path.name, sha256=sha)
    return payload


def write_qa_notebook(outdir: Path) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# V4 Production State-Plan QA\n",
                    "\n",
                    "This notebook inspects deterministic state-plan QA artifacts only. It does not render images or submit jobs.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import json\n",
                    "import os\n",
                    "from pathlib import Path\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "\n",
                    "root = Path(os.environ.get('SHERA_V4_QA_ROOT', '.')).resolve()\n",
                    "joint = json.loads((root / 'qa' / 'joint_geometry_summary.json').read_text())\n",
                    "radial = json.loads((root / 'qa' / 'radial_feasibility_summary.json').read_text())\n",
                    "split_integrity = json.loads((root / 'qa' / 'split_integrity_summary.json').read_text())\n",
                    "review = json.loads((root / 'qa' / 'review_decisions.json').read_text())\n",
                    "manifest = json.loads((root / 'freeze_manifest.json').read_text())\n",
                    "render_contract = json.loads((root / 'render_contract.json').read_text())\n",
                    "scale = json.loads((root / 'render_scale_summary.json').read_text())\n",
                    "nuisance = json.loads((root / 'nuisance_bank.json').read_text())\n",
                    "contrib = pd.read_csv(root / 'qa' / 'joint_squared_radius_contribution.csv')\n",
                    "directions = pd.read_csv(root / 'qa' / 'radial_direction_conditioning.csv')\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Frozen Identity\n",
                    "\n",
                    "Frozen hashes, split integrity, render count, storage estimate, nuisance bank, and accepted review decisions.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "display(pd.DataFrame([\n",
                    "    {'field': 'master_scientific_content_hash', 'value': manifest['master_scientific_content_hash']},\n",
                    "    {'field': 'vector_space_hash', 'value': manifest['vector_space_hash']},\n",
                    "    {'field': 'science_vector_space_id', 'value': manifest['science_vector_space_id']},\n",
                    "    {'field': 'nuisance_vector_space_id', 'value': manifest['nuisance_vector_space_id']},\n",
                    "    {'field': 'nuisance_bank_hash', 'value': manifest['selected_nuisance_bank_hash']},\n",
                    "    {'field': 'render_system_contract_hash', 'value': manifest['render_system_contract_hash']},\n",
                    "    {'field': 'render_contract_hash', 'value': manifest['render_contract_hash']},\n",
                    "    {'field': 'review_decisions_hash', 'value': manifest['review_decisions_hash']},\n",
                    "]))\n",
                    "display(pd.DataFrame([{\n",
                    "    'split_integrity': split_integrity['status'],\n",
                    "    'science_count': render_contract['science_count'],\n",
                    "    'nuisance_count': render_contract['nuisance_count'],\n",
                    "    'render_count': render_contract['render_count'],\n",
                    "    'render_index_first': render_contract['render_index_range'][0],\n",
                    "    'render_index_last': render_contract['render_index_range'][1],\n",
                    "    'nuisance_states': len(nuisance['states']),\n",
                    "    'estimated_total_persistent_bytes': scale['total_estimated_persistent_footprint_bytes'],\n",
                    "}]))\n",
                    "display(pd.DataFrame([\n",
                    "    {'decision': 'joint_geometry', 'status': review['joint_geometry']['status']},\n",
                    "    {'decision': 'radial_geometry', 'status': review['radial_geometry']['status']},\n",
                    "    {'decision': 'boundary_stress_v4', 'status': review['boundary_stress_v4']['status']},\n",
                    "]))\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Joint Multiscale QA\n",
                    "\n",
                    "Scale-stratum counts, Fisher-radius distributions, coordinate marginals, and squared-radius contributions.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "pd.DataFrame(joint['scale_strata'])\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "ax = contrib.plot.bar(x='label', y='fraction_of_total_radius_squared', figsize=(12, 4), legend=False)\n",
                    "ax.set_ylabel('fraction of ||z||^2')\n",
                    "ax.set_xlabel('parameter')\n",
                    "plt.xticks(rotation=90)\n",
                    "plt.tight_layout()\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Radial Feasibility QA\n",
                    "\n",
                    "Requested/accepted radii, feasible radius, rejection rates, and direction-coordinate conditioning by radial bin.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "pd.DataFrame(radial['bins'])\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "pd.DataFrame(review['radial_geometry']['acceptance_rejection_by_bin'])\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "directions.pivot_table(index='label', columns='radial_bin', values='std_ratio_vs_lowest_bin').plot.bar(figsize=(12, 4))\n",
                    "plt.ylabel('direction std ratio vs lowest bin')\n",
                    "plt.tight_layout()\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (outdir / "production_qa.ipynb").write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")


def materialize(
    outdir: Path,
    *,
    prepared_v3_nuisance: Path = DEFAULT_PREPARED_V3_NUISANCE,
    v3_nuisance_raw: Path = DEFAULT_V3_NUISANCE_RAW,
    sparse_context: Path = DEFAULT_SPARSE_CONTEXT,
    render_system_source: Path = DEFAULT_RENDER_SYSTEM_SOURCE,
    tiny: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    global JOINT_COUNTS, RADIAL_COUNTS

    prepare_output_root(outdir, overwrite=overwrite, dry_run=dry_run)
    if dry_run:
        return {
            "schema_version": MASTER_SCHEMA,
            "dataset_version": DATASET_VERSION + ("_tiny" if tiny else ""),
            "dry_run": True,
            "outdir": str(outdir),
            "would_overwrite": bool(overwrite and _is_populated(outdir)),
        }
    raw_training_records = load_parameter_records(v3_nuisance_raw / "parameter_space.json")
    training_records, prepared_fisher_scales, fisher_scale_rows = reconcile_science_records(
        prepared_vector_spaces_path=prepared_v3_nuisance / "vector_spaces.json",
        raw_parameter_records=raw_training_records,
    )
    prepared_manifest = read_json(prepared_v3_nuisance / "manifest.json")
    training_source_identity = {
        "authoritative_coordinate_source": "S01/S05 prepared V3 nuisance dataset",
        "authoritative_science_order": "fisher_scaled_delta components in prepared V3 vector_spaces.json",
        "prepared_artifact_id": prepared_manifest.get("artifact_id") or prepared_v3_nuisance.name,
        "prepared_vector_spaces_sha256": file_hash(prepared_v3_nuisance / "vector_spaces.json"),
        "raw_parameter_space_role": "raw records are reconciled by label for nominal values and historical sampling-envelope metadata",
    }
    vector_contract = build_vector_contracts(
        prepared_vector_spaces_path=prepared_v3_nuisance / "vector_spaces.json",
        ordered_parameter_records=training_records,
        prepared_fisher_scales=prepared_fisher_scales,
        fisher_scale_compatibility=fisher_scale_rows,
        training_source_identity=training_source_identity,
    )
    write_json(outdir / "vector_spaces.json", vector_contract)
    _write_csv(outdir / "unit_contract.csv", vector_contract["unit_contract"])
    science_vector_space_id = vector_contract["science_vector_space_id"]
    nuisance_vector_space_id = vector_contract["nuisance_vector_space_id"]
    comparison = compatibility_comparison(
        prepared_root=prepared_v3_nuisance,
        v3_raw_root=v3_nuisance_raw,
        sparse_root=sparse_context,
        ordered_training_records=training_records,
        fisher_scale_rows=fisher_scale_rows,
    )
    write_json(outdir / "compatibility" / "coordinate_compatibility_comparison.json", comparison)
    _write_csv(outdir / "compatibility" / "coordinate_compatibility_components.csv", comparison["components"])
    v3_bank = recover_nuisance_bank(v3_nuisance_raw, name="S01/S05 V3 nuisance-pair bank")
    sparse_bank = recover_nuisance_bank(sparse_context, name="sparse_mixture_legacy_100k bank")
    nuisance_comparison = compare_nuisance_banks(v3_bank, sparse_bank, vector_space_id=nuisance_vector_space_id)
    write_json(outdir / "compatibility" / "nuisance_bank_comparison.json", nuisance_comparison)
    selected_bank = v3_bank.to_contract(nuisance_comparison["selected_bank_identity"], vector_space_id=nuisance_vector_space_id)
    write_json(outdir / "nuisance_bank.json", selected_bank)
    _write_csv(outdir / "nuisance_bank.csv", selected_bank["states"])

    labels = [str(record["label"]) for record in training_records]
    nominal = np.asarray([float(record["nominal_value"]) for record in training_records], dtype=float)
    sigmas = np.asarray(prepared_fisher_scales, dtype=float)
    historical_max_sigma = np.asarray([float(record["max_sigma"]) for record in training_records], dtype=float)
    base_halfwidth = np.minimum(historical_max_sigma, 1000.0)
    envelope = {
        "schema_version": "shera_v4_joint_sampling_envelope/1",
        "policy": "joint_base_halfwidth_i = min(historical_max_sigma_i, 1000)",
        "kind": "sampling_envelope",
        "not_physical_validity_bound": True,
        "components": [
            {
                "index": idx,
                "label": label,
                "historical_max_sigma": float(historical_max_sigma[idx]),
                "joint_base_halfwidth_fisher": float(base_halfwidth[idx]),
                "joint_base_halfwidth_physical_delta": float(base_halfwidth[idx] * sigmas[idx]),
            }
            for idx, label in enumerate(labels)
        ],
    }
    envelope["content_hash"] = content_hash({key: value for key, value in envelope.items() if key != "content_hash"})
    write_json(outdir / "joint_base_envelope.json", envelope)

    render_system_contract = build_render_system_contract(
        resolved_prescription_path=render_system_source,
        manifest_path=v3_nuisance_raw / "manifest.json",
        coordinate_source_sha256=comparison["authoritative_prepared_vector_spaces_sha256"],
        local_source_path=render_system_source,
    )
    write_json(outdir / "render_system_contract.json", render_system_contract)
    render_system_contract_hash = render_system_contract["render_system_contract_hash"]
    joint_contract_id = content_hash({"family": "joint_full_v4", "envelope": envelope["content_hash"], "scale_strata": JOINT_SCALE_STRATA})
    radial_contract_id = content_hash({"family": "radial_capture_v4", "envelope": envelope["content_hash"], "radial_bins": RADIAL_BINS, "policy": "requested radius accepted only if <= direction feasible radius"})

    original_joint_counts = dict(JOINT_COUNTS)
    original_radial_counts = dict(RADIAL_COUNTS)
    if tiny:
        JOINT_COUNTS = {"train": 64, "validation": 16, "test": 16}
        RADIAL_COUNTS = {"train": 48, "validation": 24, "test": 24}
    try:
        joint_qa, joint_hashes = generate_joint(
            outdir=outdir,
            labels=labels,
            nominal=nominal,
            sigmas=sigmas,
            base_halfwidth=base_halfwidth,
            vector_space_id=science_vector_space_id,
            sampling_contract_id=joint_contract_id,
            train_prefixes=(16, 32, 64) if tiny else TRAIN_PREFIXES,
        )
        radial_qa, radial_hashes = generate_radial(
            outdir=outdir,
            labels=labels,
            nominal=nominal,
            sigmas=sigmas,
            base_halfwidth=base_halfwidth,
            vector_space_id=science_vector_space_id,
            sampling_contract_id=radial_contract_id,
        )
    finally:
        JOINT_COUNTS = original_joint_counts
        RADIAL_COUNTS = original_radial_counts
    active_joint_counts = {"train": 64, "validation": 16, "test": 16} if tiny else original_joint_counts
    active_radial_counts = {"train": 48, "validation": 24, "test": 24} if tiny else original_radial_counts
    science_counts = {"joint_full_v4": active_joint_counts, "radial_capture_v4": active_radial_counts}
    split_integrity = validate_split_integrity(load_science_plans(outdir))
    write_json(outdir / "qa" / "split_integrity_summary.json", split_integrity)
    review = review_decisions(joint_qa, radial_qa)
    write_json(outdir / "qa" / "review_decisions.json", review)
    render_contract = render_index_contract(
        outdir=outdir,
        science_counts=science_counts,
        science_plan_hashes={"joint_full_v4": joint_hashes, "radial_capture_v4": radial_hashes},
        nuisance_bank_contract=selected_bank,
        render_system_contract_hash=render_system_contract_hash,
        science_vector_space_id=science_vector_space_id,
    )
    scale = render_scale_summary(render_contract["render_count"])
    write_json(outdir / "render_scale_summary.json", scale)
    write_qa_notebook(outdir)
    subset_registry = {
        "schema_version": "shera_v4_subset_registry/1",
        "train_prefixes": {
            "joint_full_v4": list(TRAIN_PREFIXES if not tiny else (16, 32, 64)),
            "prefix_policy": "interleaved scale strata in ascending stable scale order; each standard prefix has equal stratum counts",
        },
        "families": list(science_counts),
    }
    subset_registry["content_hash"] = content_hash({key: value for key, value in subset_registry.items() if key != "content_hash"})
    write_json(outdir / "subset_registry.json", subset_registry)
    scientific_identity = {
        "schema_version": "shera_v4_master_scientific_identity/1",
        "dataset_version": DATASET_VERSION + ("_tiny" if tiny else ""),
        "coordinate_source_sha256": comparison["authoritative_prepared_vector_spaces_sha256"],
        "science_vector_space_id": science_vector_space_id,
        "nuisance_vector_space_id": nuisance_vector_space_id,
        "nuisance_bank_hash": selected_bank["content_hash"],
        "render_system_contract_hash": render_system_contract_hash,
        "render_contract_hash": render_contract["content_hash"],
        "joint_sampling_contract_id": joint_contract_id,
        "radial_sampling_contract_id": radial_contract_id,
        "joint_plan_hashes": joint_hashes,
        "radial_plan_hashes": radial_hashes,
        "split_integrity_hash": split_integrity["content_hash"],
        "review_decisions_hash": review["content_hash"],
        "joint_qa_hash": joint_qa["content_hash"],
        "radial_qa_hash": radial_qa["content_hash"],
        "subset_registry_hash": subset_registry["content_hash"],
        "render_scale_summary_hash": scale["content_hash"],
        "joint_scale_strata": list(JOINT_SCALE_STRATA),
        "joint_counts": active_joint_counts,
        "radial_bins": [list(item) for item in RADIAL_BINS],
        "radial_counts": active_radial_counts,
        "science_counts": science_counts,
        "nuisance_count": render_contract["nuisance_count"],
        "science_count": render_contract["science_count"],
        "render_count": render_contract["render_count"],
        "boundary_stress_v4": "deferred",
    }
    master = {
        "schema_version": MASTER_SCHEMA,
        "dataset_version": DATASET_VERSION + ("_tiny" if tiny else ""),
        "status": "frozen_after_pre_materialization_qa" if not tiny else "tiny_development_materialization",
        "provenance": {
            "authoritative_coordinate_artifact": str(prepared_v3_nuisance / "vector_spaces.json"),
            "authoritative_coordinate_artifact_sha256": comparison["authoritative_prepared_vector_spaces_sha256"],
            "raw_parameter_space": str(v3_nuisance_raw / "parameter_space.json"),
            "raw_parameter_space_sha256": comparison["authoritative_raw_parameter_space_sha256"],
            "sparse_context_parameter_space": str(sparse_context / "parameter_space.json"),
            "sparse_context_parameter_space_sha256": comparison["sparse_100k_parameter_space_sha256"],
            "render_system_source": str(render_system_source),
            "creation_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "output_root": str(outdir),
        },
        "selected_nuisance_bank_identity": selected_bank["bank_identity"],
        "selected_nuisance_bank_hash": selected_bank["content_hash"],
        "science_vector_space_id": science_vector_space_id,
        "nuisance_vector_space_id": nuisance_vector_space_id,
        "legacy_sparse_nuisance_bank_identity": nuisance_comparison["legacy_bank_identity"],
        "joint_scale_strata": list(JOINT_SCALE_STRATA),
        "joint_counts": active_joint_counts,
        "science_counts": science_counts,
        "radial_bins": [list(item) for item in RADIAL_BINS],
        "radial_counts": active_radial_counts,
        "boundary_stress_v4": "deferred",
        "render_system_contract_hash": render_system_contract_hash,
        "render_contract_hash": render_contract["content_hash"],
        "render_count": render_contract["render_count"],
        "vector_space_hash": vector_contract["content_hash"],
        "joint_sampling_contract_id": joint_contract_id,
        "radial_sampling_contract_id": radial_contract_id,
        "joint_plan_hashes": joint_hashes,
        "radial_plan_hashes": radial_hashes,
        "split_integrity_hash": split_integrity["content_hash"],
        "review_decisions_hash": review["content_hash"],
        "joint_qa_hash": joint_qa["content_hash"],
        "radial_qa_hash": radial_qa["content_hash"],
        "subset_registry_hash": subset_registry["content_hash"],
        "render_scale_summary_hash": scale["content_hash"],
        "scientific_identity": scientific_identity,
    }
    master["master_scientific_content_hash"] = content_hash(scientific_identity)
    write_json(outdir / "freeze_manifest.json", master)
    (outdir / "master_v4.yaml").write_text(yaml.safe_dump(json_ready(master), sort_keys=True), encoding="utf-8")
    write_render_handoff(
        outdir,
        master=master,
        render_system_contract=render_system_contract,
        render_contract=render_contract,
        split_integrity=split_integrity,
        review=review,
        scale=scale,
    )
    return master


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize deterministic V4 ML production science-state plans and QA metadata.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--prepared-v3-nuisance", type=Path, default=DEFAULT_PREPARED_V3_NUISANCE)
    parser.add_argument("--v3-nuisance-raw", type=Path, default=DEFAULT_V3_NUISANCE_RAW)
    parser.add_argument("--sparse-context", type=Path, default=DEFAULT_SPARSE_CONTEXT)
    parser.add_argument("--render-system-source", type=Path, default=DEFAULT_RENDER_SYSTEM_SOURCE)
    parser.add_argument("--tiny", action="store_true", help="Write a tiny deterministic development materialization.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing populated generated artifact tree.")
    parser.add_argument("--dry-run", action="store_true", help="Validate CLI options and output-root policy without writing artifacts.")
    parser.add_argument("--package-transfer", action="store_true", help="Create the compact Gattaca2 transfer package after materialization.")
    parser.add_argument("--package-only", action="store_true", help="Package an existing materialization without regenerating it.")
    parser.add_argument("--package-dir", type=Path, default=None, help="Directory for the transfer package. Defaults to --outdir.")
    args = parser.parse_args(argv)
    if args.package_only:
        package = package_transfer(args.outdir, package_dir=args.package_dir)
        print(json.dumps(json_ready(package), indent=2, sort_keys=True))
        return 0
    master = materialize(
        args.outdir,
        prepared_v3_nuisance=args.prepared_v3_nuisance,
        v3_nuisance_raw=args.v3_nuisance_raw,
        sparse_context=args.sparse_context,
        render_system_source=args.render_system_source,
        tiny=args.tiny,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    if args.package_transfer and not args.dry_run:
        package = package_transfer(args.outdir, package_dir=args.package_dir)
        master = dict(master)
        master["transfer_package"] = package
    print(json.dumps(json_ready(master), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
