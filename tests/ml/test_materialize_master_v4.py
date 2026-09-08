from __future__ import annotations

import json
from pathlib import Path

import pytest

from dluxshera.datasets.schema import VectorSpaceSpec

from work.experiments.ml.datasets import materialize_master_v4 as v4
from work.experiments.ml.datasets.materialize_master_v4 import (
    build_render_system_contract,
    content_hash,
    locate_science_global_index,
    materialize,
    nuisance_state_id,
    reconcile_science_records,
    render_index_contract,
    render_index_location,
    render_index_to_science_nuisance,
    render_state_id,
    science_nuisance_to_render_index,
    state_id,
    validate_split_integrity,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _parameter_records() -> list[dict[str, object]]:
    labels = [
        ("source.separation_as", "source", None, 1000.0, 0.1, 10.0),
        ("source.contrast", "source", None, 200.0, 0.2, 3.0),
        ("optics.plate_scale_as_per_pix", "optics", None, 20000.0, 0.01, 0.12),
        ("optics.primary.zernike_coeffs_nm[0]", "optics", 0, 1000.0, 0.5, 0.0),
    ]
    return [
        {
            "label": label,
            "base_key": label.split("[")[0],
            "display_label": label,
            "group": group,
            "component_index": component,
            "noll_index": 4 if component == 0 else None,
            "nominal_value": nominal,
            "parameter_sigma": sigma,
            "min_sigma": 1.0,
            "max_sigma": max_sigma,
            "min_abs_delta": sigma,
            "max_abs_delta": sigma * max_sigma,
            "sweep_config": {"min_sigma": 1.0, "max_sigma": max_sigma},
            "sweep_source_key": label.split("[")[0],
            "units": None,
        }
        for label, group, component, max_sigma, sigma, nominal in labels
    ]


def _historical_roots(tmp_path: Path) -> tuple[Path, Path, Path]:
    raw = tmp_path / "v3_raw"
    sparse = tmp_path / "sparse"
    prep = tmp_path / "prep"
    records = _parameter_records()
    for root in (raw, sparse):
        _write_json(root / "parameter_space.json", {"parameters": records})
        _write_json(
            root / "manifest.json",
            {
                "schema_version": "ml_training_dataset_v3_manifest/2",
                "seed": 0,
                "git_info": {"commit": "abc"},
                "nuisance_config": {
                    "keys": [
                        "source.x_position_as",
                        "source.y_position_as",
                        "source.position_angle_deg",
                    ]
                },
            },
        )
        rows = []
        for idx in range(2):
            rows.append(
                {
                    "sample_id": f"s{idx}",
                    "nuisance_id": idx + 1,
                    "registration_nuisance_values": {
                        "source.x_position_as": 0.1 * (idx + 1),
                        "source.y_position_as": -0.2 * (idx + 1),
                        "source.position_angle_deg": 1.5 * (idx + 1),
                    },
                    "registration_nuisance_sigma_values": {
                        "source.x_position_as": 10.0 * (idx + 1),
                        "source.y_position_as": -20.0 * (idx + 1),
                        "source.position_angle_deg": 30.0 * (idx + 1),
                    },
                }
            )
        _write_jsonl(root / "samples.jsonl", rows)
    _write_json(
        prep / "manifest.json",
        {
            "schema_version": "shera_prepared_dataset/1",
            "artifact_id": "PREP-TEST",
            "source_dataset": {"parameter_space_sha256": "unused"},
        },
    )
    _write_json(
        prep / "vector_spaces.json",
        {
            "schema_version": "shera_v3_vector_spaces/1",
            "spaces": {
                "fisher_scaled_delta": {
                    "components": [
                        {"label": record["label"], "index": idx}
                        for idx, record in enumerate(records)
                    ]
                }
            },
            "transforms": {
                "fisher_diagonal_scale": {
                    "scales": [record["parameter_sigma"] for record in records]
                }
            },
        },
    )
    return prep, raw, sparse


def _rewrite_parameter_space(root: Path, records: list[dict[str, object]]) -> None:
    _write_json(root / "parameter_space.json", {"parameters": records})


def _rewrite_nuisance_key_order(root: Path, labels: list[str]) -> None:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["nuisance_config"]["keys"] = labels
    _write_json(root / "manifest.json", manifest)


def _science_rows(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family in ("joint_full_v4", "radial_capture_v4"):
        for split in ("train", "validation", "test"):
            path = root / "state_plans" / family / f"{split}.jsonl"
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    rows.append(json.loads(line))
    return rows


def _science_fingerprint(root: Path) -> dict[str, object]:
    manifest = json.loads((root / "freeze_manifest.json").read_text(encoding="utf-8"))
    vector_spaces = json.loads((root / "vector_spaces.json").read_text(encoding="utf-8"))
    envelope = json.loads((root / "joint_base_envelope.json").read_text(encoding="utf-8"))
    rows = _science_rows(root)
    return {
        "master_scientific_content_hash": manifest["master_scientific_content_hash"],
        "science_vector_space_id": manifest["science_vector_space_id"],
        "labels": vector_spaces["science_vector_space_identity"]["ordered_labels"],
        "nominal": [
            component["reference_value"]
            for component in vector_spaces["spaces"]["physical_science_state"]["components"]
        ],
        "fisher_scales": vector_spaces["transforms"]["fisher_diagonal_scale"]["scales"],
        "joint_envelope": envelope["components"],
        "physical": [row["ordered_physical_science_vector"] for row in rows],
        "fisher": [row["ordered_fisher_scaled_delta"] for row in rows],
        "science_state_ids": [row["science_state_id"] for row in rows],
        "joint_plan_hashes": manifest["joint_plan_hashes"],
        "radial_plan_hashes": manifest["radial_plan_hashes"],
    }


def _nuisance_fingerprint(root: Path) -> dict[str, object]:
    manifest = json.loads((root / "freeze_manifest.json").read_text(encoding="utf-8"))
    bank = json.loads((root / "nuisance_bank.json").read_text(encoding="utf-8"))
    return {
        "master_scientific_content_hash": manifest["master_scientific_content_hash"],
        "nuisance_vector_space_id": manifest["nuisance_vector_space_id"],
        "nuisance_bank_hash": manifest["selected_nuisance_bank_hash"],
        "labels": bank["ordered_labels"],
        "states": [
            {
                "nuisance_state_id": row["nuisance_state_id"],
                "ordered_physical_nuisance_vector": row["ordered_physical_nuisance_vector"],
            }
            for row in bank["states"]
        ],
    }


def _write_render_system(root: Path, *, wavelength: float = 5.5e-7, detector_layers: list[dict[str, object]] | None = None, dp_name: str = "dp.npy") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / dp_name).write_bytes(b"stable pupil")
    payload = {
        "experiment": {"name": "test"},
        "system": {
            "preset": "SHERA_FLIGHT_3P_SIMPLE",
            "source": {
                "kind": "alpha_cen",
                "target": "ALPHA_CEN",
                "wavelength_m": wavelength,
                "bandwidth_m": 4.1e-8,
                "n_lambda": 3,
                "exposure_time_s": 0.05,
            },
            "optics": {
                "kind": "three_plane",
                "pupil_npix": 256,
                "psf_npix": 160,
                "dp_path": dp_name,
                "pixel_pitch_m": 4.6e-6,
                "oversample": 3,
                "primary_noll_indices": [4, 5],
                "secondary_noll_indices": [4, 5],
            },
            "detector": {
                "model": "HWK4123",
                "layers": detector_layers if detector_layers is not None else [{"kind": "Downsample", "kernel_size": 3}],
            },
        },
    }
    path = root / "prescription_resolved.json"
    _write_json(path, payload)
    _write_json(root / "manifest.json", {"noise_config": {"enabled": False}, "git_info": {"commit": "abc"}})
    return path


def test_state_and_render_ids_hash_content_not_paths() -> None:
    sid_a = state_id("joint_full_v4", "train", "vs1", [1.0, 2.0], [0.1, 0.2], "contract")
    sid_b = state_id("joint_full_v4", "train", "vs1", [1.0, 2.0], [0.1, 0.2], "contract")
    sid_changed = state_id("joint_full_v4", "train", "vs1", [1.0, 2.1], [0.1, 0.2], "contract")
    nid_a = nuisance_state_id("nvs", [0.0, 1.0])
    nid_b = nuisance_state_id("nvs", [0.0, 2.0])

    assert sid_a == sid_b
    assert sid_changed != sid_a
    assert render_state_id(sid_a, nid_a, "render") == render_state_id(sid_b, nid_a, "render")
    assert render_state_id(sid_a, nid_b, "render") != render_state_id(sid_a, nid_a, "render")


def test_nuisance_state_ids_are_vector_content_identities() -> None:
    vector_space_id = "nuisance-space"
    nid = nuisance_state_id(vector_space_id, [0.1, -0.2, 3.0])

    assert nuisance_state_id(vector_space_id, [0.1, -0.2, 3.0]) == nid
    assert nuisance_state_id(vector_space_id, [0.1, -0.2, 3.1]) != nid

    states_a = sorted(
        [
            {"nuisance_state_id": nuisance_state_id(vector_space_id, [1.0]), "ordered_physical_nuisance_vector": [1.0]},
            {"nuisance_state_id": nuisance_state_id(vector_space_id, [2.0]), "ordered_physical_nuisance_vector": [2.0]},
        ],
        key=lambda row: row["nuisance_state_id"],
    )
    states_b = list(reversed(states_a))
    identity_a = content_hash({"states": sorted(states_a, key=lambda row: row["nuisance_state_id"])})
    identity_b = content_hash({"states": sorted(states_b, key=lambda row: row["nuisance_state_id"])})
    assert identity_a == identity_b


def test_compact_render_index_round_trip() -> None:
    for science_index in (0, 1, 99):
        for nuisance_index in range(10):
            render_index = science_nuisance_to_render_index(science_index, nuisance_index, 10)
            assert render_index_to_science_nuisance(render_index, 10) == (science_index, nuisance_index)


def test_render_index_contract_boundaries_and_locations(tmp_path: Path) -> None:
    nuisance_bank = {
        "bank_identity": "bank",
        "content_hash": "bankhash",
        "nuisance_vector_space_id": "nuisance-space",
        "states": [
            {"bank_index": 0, "nuisance_state_id": "n0"},
            {"bank_index": 1, "nuisance_state_id": "n1"},
        ],
    }
    contract = render_index_contract(
        outdir=tmp_path,
        science_counts={
            "joint_full_v4": {"train": 3, "validation": 2, "test": 1},
            "radial_capture_v4": {"train": 2, "validation": 1, "test": 1},
        },
        science_plan_hashes={
            "joint_full_v4": {"train": "a", "validation": "b", "test": "c"},
            "radial_capture_v4": {"train": "d", "validation": "e", "test": "f"},
        },
        nuisance_bank_contract=nuisance_bank,
        render_system_contract_hash="render-system",
        science_vector_space_id="science-space",
    )

    assert contract["render_count"] == 20
    assert contract["render_index_range"] == [0, 19]
    assert science_nuisance_to_render_index(0, 0, 2) == 0
    assert science_nuisance_to_render_index(9, 1, 2) == 19
    assert locate_science_global_index(contract, 0) == {
        "family": "joint_full_v4",
        "split_role": "train",
        "science_global_index": 0,
        "science_plan_row_index": 0,
    }
    assert render_index_location(contract, 6)["split_role"] == "validation"
    assert render_index_location(contract, 12)["family"] == "radial_capture_v4"
    assert len({science_nuisance_to_render_index(s, n, 2) for s in range(10) for n in range(2)}) == 20


def test_overwrite_policy_new_empty_populated_and_explicit_replace(tmp_path: Path) -> None:
    prep, raw, sparse = _historical_roots(tmp_path)
    out_new = tmp_path / "out_new"
    out_empty = tmp_path / "out_empty"
    out_empty.mkdir()

    materialize(out_new, prepared_v3_nuisance=prep, v3_nuisance_raw=raw, sparse_context=sparse, render_system_source=_write_render_system(tmp_path / "rs1"), tiny=True)
    materialize(out_empty, prepared_v3_nuisance=prep, v3_nuisance_raw=raw, sparse_context=sparse, render_system_source=_write_render_system(tmp_path / "rs2"), tiny=True)

    with pytest.raises(FileExistsError):
        materialize(out_new, prepared_v3_nuisance=prep, v3_nuisance_raw=raw, sparse_context=sparse, render_system_source=_write_render_system(tmp_path / "rs3"), tiny=True)

    sentinel = out_new / "sentinel.txt"
    sentinel.write_text("old", encoding="utf-8")
    materialize(out_new, prepared_v3_nuisance=prep, v3_nuisance_raw=raw, sparse_context=sparse, render_system_source=_write_render_system(tmp_path / "rs4"), tiny=True, overwrite=True)
    assert not sentinel.exists()
    assert (out_new / "freeze_manifest.json").exists()


def test_split_integrity_fails_on_cross_split_physical_collision() -> None:
    base = {
        "science_state_id": "science-a",
        "ordered_physical_science_vector": [1.0, 2.0],
    }
    plans = {
        "joint_full_v4": {
            "train": [base],
            "validation": [{**base, "science_state_id": "science-b"}],
            "test": [],
        },
        "radial_capture_v4": {"train": [], "validation": [], "test": []},
    }

    with pytest.raises(ValueError):
        validate_split_integrity(plans)


def test_render_system_hash_tracks_system_content_not_local_path(tmp_path: Path) -> None:
    src_a = _write_render_system(tmp_path / "a", dp_name="dp_a.npy")
    src_b = _write_render_system(tmp_path / "b", dp_name="different_name.npy")
    same_a = build_render_system_contract(resolved_prescription_path=src_a, manifest_path=src_a.parent / "manifest.json")
    same_b = build_render_system_contract(resolved_prescription_path=src_b, manifest_path=src_b.parent / "manifest.json")
    changed_wavelength_path = _write_render_system(tmp_path / "c", wavelength=6.0e-7)
    changed_detector_path = _write_render_system(tmp_path / "d", detector_layers=[{"kind": "Downsample", "kernel_size": 5}])
    changed_wavelength = build_render_system_contract(resolved_prescription_path=changed_wavelength_path, manifest_path=changed_wavelength_path.parent / "manifest.json")
    changed_detector = build_render_system_contract(resolved_prescription_path=changed_detector_path, manifest_path=changed_detector_path.parent / "manifest.json")

    assert same_a["render_system_contract_hash"] == same_b["render_system_contract_hash"]
    assert changed_wavelength["render_system_contract_hash"] != same_a["render_system_contract_hash"]
    assert changed_detector["render_system_contract_hash"] != same_a["render_system_contract_hash"]


def test_tiny_materialization_is_deterministic_and_preserves_units_and_prefix_mixture(tmp_path: Path) -> None:
    prep, raw, sparse = _historical_roots(tmp_path)
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    render_system = _write_render_system(tmp_path / "render_system")

    manifest_a = materialize(out_a, prepared_v3_nuisance=prep, v3_nuisance_raw=raw, sparse_context=sparse, render_system_source=render_system, tiny=True)
    manifest_b = materialize(out_b, prepared_v3_nuisance=prep, v3_nuisance_raw=raw, sparse_context=sparse, render_system_source=render_system, tiny=True)

    assert manifest_a["master_scientific_content_hash"] == manifest_b["master_scientific_content_hash"]
    vector_spaces = json.loads((out_a / "vector_spaces.json").read_text(encoding="utf-8"))
    units = {row["label"]: row["physical_unit"] for row in vector_spaces["unit_contract"]}
    assert units["source.separation_as"] == "arcsec"
    assert units["source.contrast"] == "dimensionless"
    assert units["optics.plate_scale_as_per_pix"] == "arcsec / pixel"
    assert units["optics.primary.zernike_coeffs_nm[0]"] == "nm"
    assert all(row["physical_unit"] for row in vector_spaces["unit_contract"])
    physical_space = VectorSpaceSpec.from_dict(vector_spaces["spaces"]["physical_science_state"])
    round_tripped = VectorSpaceSpec.from_dict(physical_space.to_dict())
    assert round_tripped.labels == physical_space.labels
    assert [component.unit for component in round_tripped.components] == [
        component.unit for component in physical_space.components
    ]

    qa = json.loads((out_a / "qa" / "joint_geometry_summary.json").read_text(encoding="utf-8"))
    assert qa["nested_prefix_scale_counts"]["16"] == {"0": 4, "1": 4, "2": 4, "3": 4}
    assert json.loads((out_a / "qa" / "split_integrity_summary.json").read_text(encoding="utf-8"))["status"] == "PASS"
    assert json.loads((out_a / "qa" / "review_decisions.json").read_text(encoding="utf-8"))["boundary_stress_v4"]["status"] == "deferred"
    render_contract = json.loads((out_a / "render_contract.json").read_text(encoding="utf-8"))
    assert render_contract["mapping"] == "render_index = science_global_index * nuisance_count + nuisance_bank_index"
    assert render_contract["render_count"] == (64 + 16 + 16 + 48 + 24 + 24) * 2
    assert render_contract["render_index_range"] == [0, render_contract["render_count"] - 1]


def test_science_ordering_is_prepared_v3_canonical_under_shuffled_raw_rows(tmp_path: Path) -> None:
    prep, raw, sparse = _historical_roots(tmp_path)
    out_normal = tmp_path / "out_normal"
    out_shuffled = tmp_path / "out_shuffled"
    render_system = _write_render_system(tmp_path / "render_system")

    records = _parameter_records()
    materialize(
        out_normal,
        prepared_v3_nuisance=prep,
        v3_nuisance_raw=raw,
        sparse_context=sparse,
        render_system_source=render_system,
        tiny=True,
    )
    _rewrite_parameter_space(raw, list(reversed(records)))
    materialize(
        out_shuffled,
        prepared_v3_nuisance=prep,
        v3_nuisance_raw=raw,
        sparse_context=sparse,
        render_system_source=render_system,
        tiny=True,
    )

    assert _science_fingerprint(out_shuffled) == _science_fingerprint(out_normal)


def test_science_reconciliation_rejects_missing_and_duplicate_labels(tmp_path: Path) -> None:
    prep, raw, _ = _historical_roots(tmp_path)
    records = _parameter_records()

    with pytest.raises(ValueError, match="missing"):
        reconcile_science_records(
            prepared_vector_spaces_path=prep / "vector_spaces.json",
            raw_parameter_records=records[:-1],
        )

    duplicate = records + [dict(records[0])]
    with pytest.raises(ValueError, match="duplicate"):
        reconcile_science_records(
            prepared_vector_spaces_path=prep / "vector_spaces.json",
            raw_parameter_records=duplicate,
        )

    unexpected = [dict(row) for row in records]
    unexpected[0]["label"] = "science.unexpected"
    with pytest.raises(ValueError, match="unexpected"):
        reconcile_science_records(
            prepared_vector_spaces_path=prep / "vector_spaces.json",
            raw_parameter_records=unexpected,
        )


def test_nuisance_ordering_is_canonical_under_shuffled_source_keys(tmp_path: Path) -> None:
    prep, raw, sparse = _historical_roots(tmp_path)
    out_normal = tmp_path / "out_normal"
    out_shuffled = tmp_path / "out_shuffled"
    render_system = _write_render_system(tmp_path / "render_system")

    materialize(
        out_normal,
        prepared_v3_nuisance=prep,
        v3_nuisance_raw=raw,
        sparse_context=sparse,
        render_system_source=render_system,
        tiny=True,
    )
    shuffled = [
        "source.position_angle_deg",
        "source.y_position_as",
        "source.x_position_as",
    ]
    _rewrite_nuisance_key_order(raw, shuffled)
    _rewrite_nuisance_key_order(sparse, shuffled)
    materialize(
        out_shuffled,
        prepared_v3_nuisance=prep,
        v3_nuisance_raw=raw,
        sparse_context=sparse,
        render_system_source=render_system,
        tiny=True,
    )

    assert _nuisance_fingerprint(out_shuffled) == _nuisance_fingerprint(out_normal)


def test_scientific_identity_changes_track_science_and_nuisance_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prep, raw, sparse = _historical_roots(tmp_path)
    render_system = _write_render_system(tmp_path / "render_system")
    baseline = materialize(
        tmp_path / "baseline",
        prepared_v3_nuisance=prep,
        v3_nuisance_raw=raw,
        sparse_context=sparse,
        render_system_source=render_system,
        tiny=True,
    )

    records = _parameter_records()
    changed_value = [dict(row) for row in records]
    changed_value[0]["nominal_value"] = 11.0
    _rewrite_parameter_space(raw, changed_value)
    changed_value_manifest = materialize(
        tmp_path / "changed_value",
        prepared_v3_nuisance=prep,
        v3_nuisance_raw=raw,
        sparse_context=sparse,
        render_system_source=render_system,
        tiny=True,
    )
    assert changed_value_manifest["master_scientific_content_hash"] != baseline["master_scientific_content_hash"]

    changed_scale = [dict(row) for row in records]
    changed_scale[0]["parameter_sigma"] = 0.125
    _rewrite_parameter_space(raw, changed_scale)
    prepared = json.loads((prep / "vector_spaces.json").read_text(encoding="utf-8"))
    prepared["transforms"]["fisher_diagonal_scale"]["scales"][0] = 0.125
    _write_json(prep / "vector_spaces.json", prepared)
    changed_scale_manifest = materialize(
        tmp_path / "changed_scale",
        prepared_v3_nuisance=prep,
        v3_nuisance_raw=raw,
        sparse_context=sparse,
        render_system_source=render_system,
        tiny=True,
    )
    assert changed_scale_manifest["science_vector_space_id"] != baseline["science_vector_space_id"]

    monkeypatch.setitem(
        v4.SCIENCE_UNIT_OVERRIDES,
        "source.separation_as",
        ("mas", "binary separation on sky"),
    )
    changed_unit_manifest = materialize(
        tmp_path / "changed_unit",
        prepared_v3_nuisance=prep,
        v3_nuisance_raw=raw,
        sparse_context=sparse,
        render_system_source=render_system,
        tiny=True,
    )
    assert changed_unit_manifest["science_vector_space_id"] != changed_scale_manifest["science_vector_space_id"]


def test_nuisance_identity_changes_track_values_and_units(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prep, raw, sparse = _historical_roots(tmp_path)
    render_system = _write_render_system(tmp_path / "render_system")
    baseline = materialize(
        tmp_path / "baseline",
        prepared_v3_nuisance=prep,
        v3_nuisance_raw=raw,
        sparse_context=sparse,
        render_system_source=render_system,
        tiny=True,
    )

    rows = [
        {
            **row,
            "registration_nuisance_values": {
                **row["registration_nuisance_values"],
                "source.x_position_as": row["registration_nuisance_values"]["source.x_position_as"] + 1.0,
            },
        }
        for row in [
            json.loads(line)
            for line in (raw / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        ]
    ]
    _write_jsonl(raw / "samples.jsonl", rows)
    changed_value = materialize(
        tmp_path / "changed_nuisance_value",
        prepared_v3_nuisance=prep,
        v3_nuisance_raw=raw,
        sparse_context=sparse,
        render_system_source=render_system,
        tiny=True,
    )
    assert changed_value["selected_nuisance_bank_hash"] != baseline["selected_nuisance_bank_hash"]

    monkeypatch.setitem(
        v4.NUISANCE_UNITS,
        "source.x_position_as",
        ("mas", "source registration x offset on sky"),
    )
    changed_unit = materialize(
        tmp_path / "changed_nuisance_unit",
        prepared_v3_nuisance=prep,
        v3_nuisance_raw=raw,
        sparse_context=sparse,
        render_system_source=render_system,
        tiny=True,
    )
    assert changed_unit["nuisance_vector_space_id"] != changed_value["nuisance_vector_space_id"]


def test_vector_contract_identity_ignores_local_path_timestamp_and_output_root(tmp_path: Path) -> None:
    prep, raw, sparse = _historical_roots(tmp_path)
    render_system = _write_render_system(tmp_path / "render_system")

    first = materialize(
        tmp_path / "path_a",
        prepared_v3_nuisance=prep,
        v3_nuisance_raw=raw,
        sparse_context=sparse,
        render_system_source=render_system,
        tiny=True,
    )
    second = materialize(
        tmp_path / "path_b",
        prepared_v3_nuisance=prep,
        v3_nuisance_raw=raw,
        sparse_context=sparse,
        render_system_source=render_system,
        tiny=True,
    )

    assert first["master_scientific_content_hash"] == second["master_scientific_content_hash"]
    assert first["science_vector_space_id"] == second["science_vector_space_id"]
    assert first["nuisance_vector_space_id"] == second["nuisance_vector_space_id"]
