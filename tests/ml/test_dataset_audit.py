from __future__ import annotations

import json
from pathlib import Path

import pytest

from work.experiments.ml.datasets.audit_dataset import audit_dataset, main


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(keys) + "\n")
        for row in rows:
            handle.write(",".join(str(row.get(key, "")) for key in keys) + "\n")


def _parameter_space(root: Path) -> None:
    _write_json(
        root / "parameter_space.json",
        {
            "parameters": [
                {
                    "label": "science.a",
                    "base_key": "science.a",
                    "nominal_value": 10.0,
                    "parameter_sigma": 2.0,
                    "min_abs_delta": 2.0,
                    "max_abs_delta": 6.0,
                    "min_sigma": 1.0,
                    "max_sigma": 3.0,
                },
                {
                    "label": "science.b",
                    "base_key": "science.b",
                    "nominal_value": 20.0,
                    "parameter_sigma": 4.0,
                    "min_abs_delta": 4.0,
                    "max_abs_delta": 12.0,
                    "min_sigma": 1.0,
                    "max_sigma": 3.0,
                },
            ]
        },
    )


def _manifest(root: Path, *, count: int = 4, rendered: int | None = None) -> None:
    _write_json(
        root / "manifest.json",
        {
            "schema_version": "ml_training_dataset_v3_manifest/2",
            "generator": "test_generator.py",
            "script_version": "test",
            "rendered_sample_count": count if rendered is None else rendered,
            "render_complete": True,
            "render_target_sample_count": count,
            "resolved_system_summary": {
                "preset": "TEST_OPTICS",
                "exposure_time_s": 0.05,
                "estimated_image_shape": [2, 2],
            },
            "noise_config": {"enabled": False},
            "nuisance_config": {"keys": ["nuis.x", "nuis.y"]},
        },
    )


def _raw_row(
    idx: int,
    *,
    sample_id: str | None = None,
    theta_delta: dict[str, float] | None = None,
    theta_sigma: dict[str, float] | None = None,
    active_labels: list[str] | None = None,
    nuisance_id: int = 0,
    nuisance: tuple[float, float] = (0.0, 0.0),
) -> dict[str, object]:
    return {
        "dataset_family": "sparse_mixture",
        "sample_role": "sparse_random",
        "sample_id": sample_id or f"sample_{idx:06d}",
        "sample_index": idx,
        "split": "test",
        "theta_delta": theta_delta or {},
        "theta_sigma": theta_sigma or {},
        "active_labels": active_labels or [],
        "active_count": len(active_labels or []),
        "active_mask": [1 if label in (active_labels or []) else 0 for label in ("science.a", "science.b")],
        "registration_nuisance_values": {"nuis.x": nuisance[0], "nuis.y": nuisance[1]},
        "nuisance_id": nuisance_id,
        "fits_path": f"images/sample_{idx:06d}.fits",
        "metadata_path": f"images/sample_{idx:06d}.json",
        "image_shape": [2, 2],
    }


def test_raw_context_audit_recovers_sparse_counts_ranges_and_one_nuisance_policy(tmp_path: Path) -> None:
    _manifest(tmp_path, count=3)
    _parameter_space(tmp_path)
    rows = [
        _raw_row(0, theta_delta={"science.a": -2.0}, theta_sigma={"science.a": -1.0}, active_labels=["science.a"], nuisance_id=1, nuisance=(0.1, 0.2)),
        _raw_row(1, theta_delta={"science.a": 4.0, "science.b": 8.0}, theta_sigma={"science.a": 2.0, "science.b": 2.0}, active_labels=["science.a", "science.b"], nuisance_id=2, nuisance=(0.3, 0.4)),
        _raw_row(2, theta_delta={"science.b": 12.0}, theta_sigma={"science.b": 3.0}, active_labels=["science.b"], nuisance_id=1, nuisance=(0.1, 0.2)),
    ]
    _write_jsonl(tmp_path / "samples.jsonl", rows)
    _write_csv(tmp_path / "sparse_mixture_plan.csv", rows)
    _write_csv(tmp_path / "pair_plan.csv", [])

    audit = audit_dataset(tmp_path)

    assert audit["status"] == "ok"
    assert audit["size_structure"]["sample_count"]["value"] == 3
    assert audit["size_structure"]["science_dimension"]["value"] == 2
    assert audit["families"]["split_counts"] == {"test": 3}
    assert audit["sparse_behavior"]["active_count_distribution"] == {"1": 2, "2": 1}
    assert audit["sparse_behavior"]["per_parameter_activation_frequency"] == {
        "science.a": 2,
        "science.b": 2,
    }
    fisher = {row["label"]: row for row in audit["science_space_coverage"]["fisher_scaled_delta"]}
    assert fisher["science.a"]["min"] == pytest.approx(-1.0)
    assert fisher["science.a"]["max"] == pytest.approx(2.0)
    assert fisher["science.b"]["max"] == pytest.approx(3.0)
    assert audit["nuisance_coverage"]["fixed_bank"]["value"] is True
    assert audit["nuisance_coverage"]["replication_classification"]["classification"] == "one_nuisance_per_science_sample"


def test_prepared_audit_classifies_full_cross_product_and_counts_shards(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "vector_spaces.json",
        {
            "spaces": {
                "fisher_scaled_delta": {
                    "components": [
                        {"label": "science.a", "index": 0},
                        {"label": "science.b", "index": 1},
                    ]
                },
                "registration_nuisance": {
                    "components": [
                        {"label": "nuis.x", "index": 0},
                    ]
                },
            },
            "transforms": {"fisher_diagonal_scale": {"scales": [2.0, 4.0]}},
        },
    )
    _write_json(
        tmp_path / "array_shards_manifest.json",
        {"schema_version": "array_shard_store/1", "sample_count": 4, "sample_shape": [2, 2], "shard_count": 2},
    )
    _write_json(
        tmp_path / "manifest.json",
        {
            "schema_version": "shera_prepared_dataset/1",
            "artifact_id": "PREP-TEST",
            "array_storage": {"sample_count": 4, "sample_shape": [2, 2], "shard_count": 2},
        },
    )
    rows = []
    idx = 0
    for z in ([0.0, 0.0], [1.0, 0.0]):
        for nuisance_id, nuisance in [("n0", [0.0]), ("n1", [1.0])]:
            rows.append(
                {
                    "sample_id": f"s{idx}",
                    "sample_index": idx,
                    "dataset_family": "joint_full_v4",
                    "physical_delta": [2.0 * z[0], 4.0 * z[1]],
                    "fisher_scaled_delta": z,
                    "nuisance_id": nuisance_id,
                    "nuisance_vector": nuisance,
                    "image_shape": [2, 2],
                }
            )
            idx += 1
    _write_jsonl(tmp_path / "index.jsonl", rows)

    audit = audit_dataset(tmp_path)

    assert audit["identity"]["dataset_or_artifact_id"]["value"] == "PREP-TEST"
    assert audit["size_structure"]["shard_count"]["value"] == 2
    assert audit["size_structure"]["unique_science_state_count"]["value"] == 2
    assert audit["size_structure"]["unique_nuisance_state_count"]["value"] == 2
    assert audit["nuisance_coverage"]["replication_classification"]["classification"] == "full_cross_product"


def test_missing_optional_metadata_degrades_and_unsupported_root_is_reported(tmp_path: Path) -> None:
    minimal = tmp_path / "minimal"
    _write_jsonl(minimal / "samples.jsonl", [{"sample_id": "s0", "sample_index": 0}])
    audit = audit_dataset(minimal)
    assert audit["status"] == "ok"
    assert audit["size_structure"]["science_dimension"]["status"] == "unavailable"
    assert audit["integrity"]["missing_render_reference_count"] == 1

    unsupported = tmp_path / "unsupported"
    unsupported.mkdir()
    assert audit_dataset(unsupported)["status"] == "unsupported"


def test_integrity_flags_duplicate_ids_dimension_mismatch_nonfinite_and_count_mismatch(tmp_path: Path) -> None:
    _manifest(tmp_path, count=3, rendered=99)
    _parameter_space(tmp_path)
    rows = [
        _raw_row(0, sample_id="dup", theta_delta={"science.a": 1.0}, active_labels=["science.a"], nuisance_id=1, nuisance=(0.0, 0.0)),
        _raw_row(1, sample_id="dup", theta_delta={"science.a": float("nan")}, active_labels=["science.a"], nuisance_id=1, nuisance=(0.0, 0.0)),
        {
            **_raw_row(3, theta_delta={"science.b": 1.0}, active_labels=["science.b"], nuisance_id=2, nuisance=(1.0, 0.0)),
            "active_mask": [1, 0, 1],
        },
    ]
    _write_jsonl(tmp_path / "samples.jsonl", rows)
    _write_csv(tmp_path / "sparse_mixture_plan.csv", rows[:2])

    audit = audit_dataset(tmp_path)

    assert audit["integrity"]["duplicate_sample_id_count"] == 1
    assert audit["integrity"]["nonfinite_record_count"] >= 1
    assert audit["integrity"]["manifest_count_mismatch"] is True
    assert audit["integrity"]["plan_count_mismatch"] is True
    assert audit["integrity"]["missing_sample_index_ranges"] == [[2, 2]]


def test_prepared_dimension_mismatch_and_partial_classification(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "vector_spaces.json",
        {
            "spaces": {
                "fisher_scaled_delta": {"components": [{"label": "science.a", "index": 0}, {"label": "science.b", "index": 1}]},
                "registration_nuisance": {"components": [{"label": "nuis.x", "index": 0}]},
            },
            "transforms": {"fisher_diagonal_scale": {"scales": [1.0, 1.0]}},
        },
    )
    _write_json(tmp_path / "array_shards_manifest.json", {"sample_count": 3, "sample_shape": [2, 2], "shard_count": 1})
    _write_json(tmp_path / "manifest.json", {"schema_version": "shera_prepared_dataset/1", "array_storage": {"sample_count": 3}})
    _write_jsonl(
        tmp_path / "index.jsonl",
        [
            {"sample_id": "s0", "physical_delta": [0.0, 0.0], "fisher_scaled_delta": [0.0, 0.0], "nuisance_id": "n0", "nuisance_vector": [0.0]},
            {"sample_id": "s1", "physical_delta": [1.0, 0.0], "fisher_scaled_delta": [1.0, 0.0], "nuisance_id": "n0", "nuisance_vector": [0.0]},
            {"sample_id": "s2", "physical_delta": [0.0], "fisher_scaled_delta": [0.0], "nuisance_id": "n1", "nuisance_vector": [1.0]},
        ],
    )

    audit = audit_dataset(tmp_path)

    assert audit["integrity"]["vector_dimension_mismatch_count"] >= 2
    assert audit["nuisance_coverage"]["replication_classification"]["classification"] == "partial"


def test_cli_writes_machine_readable_outputs(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _manifest(root, count=1)
    _parameter_space(root)
    _write_jsonl(root / "samples.jsonl", [_raw_row(0, active_labels=["science.a"])])
    output_json = tmp_path / "audit.json"
    output_dir = tmp_path / "audit_dir"

    assert main([str(root), "--output-json", str(output_json), "--output-dir", str(output_dir)]) == 0

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "dluxshera_dataset_audit/1"
    assert (output_dir / "audit_summary.json").exists()
    assert (output_dir / "parameter_summary.csv").exists()


def test_verify_files_distinguishes_recorded_paths_from_present_files(tmp_path: Path) -> None:
    _manifest(tmp_path, count=2)
    _parameter_space(tmp_path)
    rows = [
        _raw_row(0, active_labels=["science.a"]),
        _raw_row(1, active_labels=["science.b"]),
    ]
    _write_jsonl(tmp_path / "samples.jsonl", rows)
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "sample_000000.fits").write_bytes(b"fits")
    (tmp_path / "images" / "sample_000000.json").write_text("{}", encoding="utf-8")

    without_check = audit_dataset(tmp_path)
    with_check = audit_dataset(tmp_path, verify_files=True)

    assert without_check["file_verification"]["enabled"] is False
    assert without_check["integrity"]["render_path_recorded_count"] == 2
    assert without_check["integrity"]["render_file_present_count"] == 0
    assert with_check["file_verification"]["enabled"] is True
    assert with_check["integrity"]["render_path_recorded_count"] == 2
    assert with_check["integrity"]["render_file_present_count"] == 1
    assert with_check["integrity"]["render_file_missing_count"] == 1
    assert with_check["integrity"]["metadata_file_present_count"] == 1
    assert with_check["integrity"]["metadata_file_missing_count"] == 1


def test_active_count_labels_and_mask_mismatches_are_reported(tmp_path: Path) -> None:
    _manifest(tmp_path, count=1)
    _parameter_space(tmp_path)
    row = _raw_row(0, active_labels=["science.a"])
    row["active_count"] = 2
    row["active_mask"] = [0, 1]
    _write_jsonl(tmp_path / "samples.jsonl", [row])

    audit = audit_dataset(tmp_path)

    assert audit["sparse_behavior"]["active_consistency_mismatch_count"] >= 2
    reasons = {item["reason"] for item in audit["sparse_behavior"]["active_consistency_mismatches"]}
    assert "active_labels_active_mask_disagree" in reasons
    assert "active_count_disagrees" in reasons


def test_unavailable_nuisance_id_is_missing_metadata_not_real_state(tmp_path: Path) -> None:
    _manifest(tmp_path, count=2)
    _parameter_space(tmp_path)
    rows = [
        _raw_row(0, active_labels=["science.a"], nuisance_id=1, nuisance=(0.1, 0.2)),
        {**_raw_row(1, active_labels=["science.b"], nuisance_id=0, nuisance=(0.0, 0.0)), "nuisance_id": "unavailable"},
    ]
    _write_jsonl(tmp_path / "samples.jsonl", rows)

    audit = audit_dataset(tmp_path)

    assert audit["size_structure"]["unique_nuisance_state_count"]["value"] == 1
    assert audit["nuisance_coverage"]["known_nuisance_id_distribution"] == {"1": 1}
    assert audit["nuisance_coverage"]["missing_nuisance_metadata_count"] == 1
    assert "unavailable" not in audit["nuisance_coverage"]["known_nuisance_id_distribution"]


def test_parameter_metadata_separates_sampling_envelope_from_physical_constraints(tmp_path: Path) -> None:
    _manifest(tmp_path, count=1)
    _parameter_space(tmp_path)
    _write_jsonl(tmp_path / "samples.jsonl", [_raw_row(0, active_labels=["science.a"])])

    audit = audit_dataset(tmp_path)
    first = audit["ordered_vector_metadata"]["parameters"][0]

    assert first["sampling_envelope"]["kind"] == "historical_sweep_extent"
    assert first["sampling_envelope"]["max_sigma"] == 3.0
    assert first["physical_validity_constraints"]["status"] == "unavailable"


def test_v4_compact_state_plan_audit_reports_render_contract_and_split_status(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "freeze_manifest.json",
        {
            "schema_version": "shera_v4_master_contract/1",
            "dataset_version": "shera_ml_master_v4",
            "master_scientific_content_hash": "master",
            "science_vector_space_id": "science-space",
            "nuisance_vector_space_id": "nuisance-space",
        },
    )
    _write_json(
        tmp_path / "vector_spaces.json",
        {
            "spaces": {
                "fisher_scaled_delta": {"components": [{"label": "science.a", "index": 0}]}
            },
            "transforms": {"fisher_diagonal_scale": {"scales": [2.0]}},
        },
    )
    _write_json(
        tmp_path / "nuisance_bank.json",
        {
            "ordered_labels": ["nuis.x"],
            "states": [
                {"nuisance_state_id": "n0", "ordered_physical_nuisance_vector": [0.0]},
                {"nuisance_state_id": "n1", "ordered_physical_nuisance_vector": [1.0]},
            ],
        },
    )
    _write_json(
        tmp_path / "render_contract.json",
        {"render_count": 4, "render_index_range": [0, 3]},
    )
    _write_json(tmp_path / "qa" / "split_integrity_summary.json", {"status": "PASS"})
    _write_jsonl(
        tmp_path / "state_plans" / "joint_full_v4" / "train.jsonl",
        [
            {
                "science_state_id": "science-0",
                "dataset_family": "joint_full_v4",
                "split_role": "train",
                "global_sequence_index": 0,
                "ordered_physical_science_vector": [10.0],
                "ordered_fisher_scaled_delta": [0.0],
            },
            {
                "science_state_id": "science-1",
                "dataset_family": "joint_full_v4",
                "split_role": "train",
                "global_sequence_index": 1,
                "ordered_physical_science_vector": [12.0],
                "ordered_fisher_scaled_delta": [1.0],
            },
        ],
    )

    audit = audit_dataset(tmp_path)

    assert audit["status"] == "ok"
    assert audit["identity"]["dataset_kind"]["value"] == "v4_state_plan"
    assert audit["size_structure"]["sample_count"]["value"] == 2
    assert audit["size_structure"]["unique_nuisance_state_count"]["value"] == 2
    assert audit["nuisance_coverage"]["replication_classification"]["classification"] == "compact_full_cross_product"
    assert audit["v4_state_plan"]["render_count"] == 4
    assert audit["v4_state_plan"]["split_integrity_status"] == "PASS"
