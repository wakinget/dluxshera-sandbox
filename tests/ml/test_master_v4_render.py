from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dluxshera.datasets.master_v4 import (
    FrozenMasterV4,
    FrozenV4Identities,
    array_task_range,
    build_slurm_array_plan,
    content_hash,
    file_sha256,
    render_index_location,
    render_output_paths,
    render_state_id,
)
from dluxshera.datasets.schema import VectorComponentSpec, VectorSpaceSpec
from dluxshera.params.spec import ParamField, ParamSpec
from dluxshera.params.store import ParameterStore
from work.experiments.ml.datasets import render_master_v4
from work.experiments.ml.datasets.hpc import submit_master_v4_render


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _labels() -> tuple[str, ...]:
    return (
        "source.x_position_as",
        "source.y_position_as",
        "source.position_angle_deg",
        "source.contrast",
        "optics.primary.zernike_coeffs_nm[1]",
        "optics.secondary.zernike_coeffs_nm[2]",
    )


def _make_fixture(root: Path) -> tuple[Path, FrozenV4Identities]:
    plan_root = root / "plan"
    labels = _labels()
    science_space = VectorSpaceSpec(
        name="shera_v4_physical_science_state",
        components=tuple(
            VectorComponentSpec(
                label=label,
                index=idx,
                source_key=label.split("[", 1)[0],
                component_index=1 if "primary" in label else 2 if "secondary" in label else None,
                reference_value=0.0,
            )
            for idx, label in enumerate(labels)
        ),
    )
    fisher_space = VectorSpaceSpec.from_labels("shera_v4_fisher_scaled_science_delta", labels)
    nuisance_labels = (
        "source.x_position_as",
        "source.y_position_as",
        "source.position_angle_deg",
    )
    nuisance_space = VectorSpaceSpec.from_labels("shera_v4_registration_nuisance", nuisance_labels)
    vector_spaces = {
        "schema_version": "shera_v4_vector_spaces/1",
        "spaces": {
            "physical_science_state": science_space.to_dict(),
            "fisher_scaled_science_delta": fisher_space.to_dict(),
            "registration_nuisance": nuisance_space.to_dict(),
        },
    }
    vector_spaces["content_hash"] = content_hash(vector_spaces)
    _write_json(plan_root / "vector_spaces.json", vector_spaces)

    counts = {
        "joint_full_v4": {"train": 2, "validation": 1, "test": 1},
        "radial_capture_v4": {"train": 1, "validation": 1, "test": 1},
    }
    plan_hashes: dict[str, dict[str, str]] = {}
    science_index = 0
    for family, splits in counts.items():
        plan_hashes[family] = {}
        for split, count in splits.items():
            rows = []
            for local in range(count):
                physical = [float(science_index + idx) for idx in range(len(labels))]
                fisher = [float(idx) / 10.0 for idx in range(len(labels))]
                row = {
                    "schema_version": "shera_v4_science_state_plan/1",
                    "dataset_version": "shera_ml_master_v4",
                    "dataset_family": family,
                    "split_role": split,
                    "science_state_id": f"science_{science_index:03d}",
                    "science_vector_space_id": "science-space",
                    "global_sequence_index": local,
                    "ordered_physical_science_vector": physical,
                    "ordered_fisher_scaled_delta": fisher,
                    "fisher_radius_l2": float(np.linalg.norm(fisher)),
                }
                if family == "joint_full_v4":
                    row["scale_stratum"] = local % 4
                    row["scale_multiplier"] = 0.125
                else:
                    row["radial_bin"] = "1000-1500" if split == "validation" else "0-100"
                    row["requested_fisher_radius"] = 1000.0
                    row["actual_fisher_radius"] = 1000.0
                    row["direction_sequence_index"] = local
                rows.append(row)
                science_index += 1
            path = plan_root / "state_plans" / family / f"{split}.jsonl"
            _write_jsonl(path, rows)
            plan_hashes[family][split] = file_sha256(path)

    nuisance_states = []
    for bank_index, vector in enumerate(([0.1, -0.2, 3.0], [1.0, 2.0, -4.0])):
        nuisance_states.append(
            {
                "bank_index": bank_index,
                "nuisance_state_id": f"nuisance_{bank_index}",
                "nuisance_vector_space_id": "nuisance-space",
                "labels": list(nuisance_labels),
                "ordered_physical_nuisance_vector": vector,
                "ordered_fisher_scaled_nuisance_vector": vector,
            }
        )
    nuisance_identity = {
        "schema_version": "shera_v4_nuisance_bank_scientific_identity/1",
        "bank_identity": "fixture-bank",
        "ordered_labels": list(nuisance_labels),
        "states": [
            {
                "nuisance_state_id": row["nuisance_state_id"],
                "nuisance_vector_space_id": row["nuisance_vector_space_id"],
                "ordered_physical_nuisance_vector": row["ordered_physical_nuisance_vector"],
                "ordered_fisher_scaled_nuisance_vector": row["ordered_fisher_scaled_nuisance_vector"],
            }
            for row in nuisance_states
        ],
    }
    nuisance_bank = {
        "schema_version": "shera_v4_nuisance_bank/1",
        "bank_identity": "fixture-bank",
        "nuisance_vector_space_id": "nuisance-space",
        "ordered_labels": list(nuisance_labels),
        "states": nuisance_states,
        "scientific_identity": nuisance_identity,
        "content_hash": content_hash(nuisance_identity),
    }
    _write_json(plan_root / "nuisance_bank.json", nuisance_bank)
    render_system = {
        "schema_version": "shera_v4_render_system_contract/1",
        "resolved_system": {
            "source": {"kind": "alpha_cen"},
            "optics": {"kind": "three_plane"},
            "detector": {"model": "HWK4123"},
        },
        "content_hash": "render-system-hash",
        "render_system_contract_hash": "render-system-hash",
    }
    _write_json(plan_root / "render_system_contract.json", render_system)

    families = []
    offset = 0
    for family in ("joint_full_v4", "radial_capture_v4"):
        for split in ("train", "validation", "test"):
            count = counts[family][split]
            families.append(
                {
                    "family": family,
                    "split_role": split,
                    "science_start_index": offset,
                    "science_stop_index_exclusive": offset + count,
                    "science_count": count,
                    "first_render_index": offset * 2,
                    "last_render_index": (offset + count) * 2 - 1,
                    "plan_path": f"state_plans/{family}/{split}.jsonl",
                    "plan_sha256": plan_hashes[family][split],
                }
            )
            offset += count
    render_contract = {
        "schema_version": "shera_v4_compact_render_contract/1",
        "render_system_contract_hash": "render-system-hash",
        "science_vector_space_id": "science-space",
        "nuisance_vector_space_id": "nuisance-space",
        "nuisance_bank_scientific_content_hash": nuisance_bank["content_hash"],
        "nuisance_count": 2,
        "science_count": offset,
        "render_count": offset * 2,
        "render_index_range": [0, offset * 2 - 1],
        "families": families,
    }
    render_contract["content_hash"] = content_hash(render_contract)
    _write_json(plan_root / "render_contract.json", render_contract)
    scientific_identity = {
        "dataset_version": "shera_ml_master_v4",
        "science_vector_space_id": "science-space",
        "nuisance_vector_space_id": "nuisance-space",
        "nuisance_bank_hash": nuisance_bank["content_hash"],
        "render_system_contract_hash": "render-system-hash",
        "render_contract_hash": render_contract["content_hash"],
    }
    manifest = {
        "schema_version": "shera_v4_master_contract/1",
        "dataset_version": "shera_ml_master_v4",
        "science_vector_space_id": "science-space",
        "nuisance_vector_space_id": "nuisance-space",
        "selected_nuisance_bank_hash": nuisance_bank["content_hash"],
        "render_system_contract_hash": "render-system-hash",
        "render_contract_hash": render_contract["content_hash"],
        "render_count": render_contract["render_count"],
        "scientific_identity": scientific_identity,
        "master_scientific_content_hash": content_hash(scientific_identity),
    }
    _write_json(plan_root / "freeze_manifest.json", manifest)
    expected = FrozenV4Identities(
        master_scientific_content_hash=manifest["master_scientific_content_hash"],
        science_vector_space_id="science-space",
        nuisance_vector_space_id="nuisance-space",
        nuisance_bank_hash=nuisance_bank["content_hash"],
        render_system_contract_hash="render-system-hash",
        render_contract_hash=render_contract["content_hash"],
    )
    return plan_root, expected


def _fake_renderer_spec() -> tuple[ParamSpec, ParameterStore, object]:
    spec = ParamSpec(
        [
            ParamField("source.x_position_as", "source", "primitive", default=0.0),
            ParamField("source.y_position_as", "source", "primitive", default=0.0),
            ParamField("source.position_angle_deg", "source", "primitive", default=0.0),
            ParamField("source.contrast", "source", "primitive", default=1.0),
            ParamField("optics.primary.zernike_coeffs_nm", "optics", "primitive", default=np.zeros(3), shape=(3,)),
            ParamField("optics.secondary.zernike_coeffs_nm", "optics", "primitive", default=np.zeros(4), shape=(4,)),
        ]
    )
    store = ParameterStore.from_spec_defaults(spec)

    class FakeBinder:
        def strip_structural(self, applied_store):
            return applied_store

        def model(self, applied_store):
            value = float(np.asarray(applied_store.get("source.x_position_as")))
            value += float(np.asarray(applied_store.get("optics.primary.zernike_coeffs_nm"))[1])
            return np.full((2, 2), value, dtype=np.float64)

    return spec, store, FakeBinder()


def test_contract_validation_and_render_index_resolution(tmp_path: Path) -> None:
    plan_root, expected = _make_fixture(tmp_path)
    plan = FrozenMasterV4(plan_root, expected=expected)

    assert plan.render_count == 14
    assert render_index_location(plan.render_contract, 0)["science_global_index"] == 0
    assert render_index_location(plan.render_contract, 13)["family"] == "radial_capture_v4"
    state = plan.resolve_indices([3])[0]
    assert state.science_global_index == 1
    assert state.nuisance_bank_index == 1
    assert state.science_labels == _labels()
    assert state.render_state_id == render_state_id("science_001", "nuisance_1", "render-system-hash")


def test_contract_validation_failures(tmp_path: Path) -> None:
    plan_root, expected = _make_fixture(tmp_path)
    with pytest.raises(ValueError, match="master_scientific_content_hash"):
        FrozenMasterV4(
            plan_root,
            expected=FrozenV4Identities(
                master_scientific_content_hash="wrong",
                science_vector_space_id=expected.science_vector_space_id,
                nuisance_vector_space_id=expected.nuisance_vector_space_id,
                nuisance_bank_hash=expected.nuisance_bank_hash,
                render_system_contract_hash=expected.render_system_contract_hash,
                render_contract_hash=expected.render_contract_hash,
            ),
        )

    contract = json.loads((plan_root / "render_contract.json").read_text(encoding="utf-8"))
    contract["families"][0]["plan_sha256"] = "wrong"
    _write_json(plan_root / "render_contract.json", contract)
    with pytest.raises(ValueError, match="State-plan hash mismatch"):
        FrozenMasterV4(plan_root, expected=None)

    plan_root, expected = _make_fixture(tmp_path / "vector")
    with pytest.raises(ValueError, match="science_vector_space_id"):
        FrozenMasterV4(
            plan_root,
            expected=FrozenV4Identities(
                master_scientific_content_hash=expected.master_scientific_content_hash,
                science_vector_space_id="wrong",
                nuisance_vector_space_id=expected.nuisance_vector_space_id,
                nuisance_bank_hash=expected.nuisance_bank_hash,
                render_system_contract_hash=expected.render_system_contract_hash,
                render_contract_hash=expected.render_contract_hash,
            ),
        )

    plan = FrozenMasterV4(plan_root, expected=expected)
    with pytest.raises(ValueError, match="outside"):
        plan.resolve_indices([14])


def test_output_paths_are_deterministic(tmp_path: Path) -> None:
    fits_path, json_path = render_output_paths(
        output_root=tmp_path,
        dataset_family="joint_full_v4",
        split_role="train",
        render_index=1234,
    )

    assert fits_path.as_posix().endswith("images/joint_full_v4/train/shard_0001/render_0001234.fits")
    assert json_path.name == "render_0001234.json"


def test_cli_renders_fits_json_and_resume_behaviour(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan_root, _ = _make_fixture(tmp_path)
    output_root = tmp_path / "out"
    monkeypatch.setattr(render_master_v4, "_build_renderer_system", lambda plan: _fake_renderer_spec())
    args = render_master_v4.build_parser().parse_args(
        [
            "--plan-root",
            str(plan_root),
            "--output-root",
            str(output_root),
            "--render-index",
            "1",
            "--allow-nonproduction-contract",
        ]
    )

    summary = render_master_v4.run(args)

    assert summary["rendered"] == 1
    fits_path, json_path = render_output_paths(
        output_root=output_root,
        dataset_family="joint_full_v4",
        split_role="train",
        render_index=1,
    )
    assert fits_path.exists()
    sidecar = json.loads(json_path.read_text(encoding="utf-8"))
    assert sidecar["render_index"] == 1
    assert sidecar["nuisance_bank_index"] == 1
    assert sidecar["image_shape"] == [2, 2]
    assert sidecar["dtype"] == "float64"

    summary = render_master_v4.run(args)
    assert summary["skipped_valid"] == 1

    json_path.unlink()
    with pytest.raises(FileExistsError, match="partial output"):
        render_master_v4.run(args)
    args.overwrite_invalid = True
    summary = render_master_v4.run(args)
    assert summary["rendered"] == 1

    sidecar = json.loads(json_path.read_text(encoding="utf-8"))
    sidecar["render_state_id"] = "wrong"
    _write_json(json_path, sidecar)
    args.overwrite_invalid = False
    with pytest.raises(FileExistsError, match="sidecar identity"):
        render_master_v4.run(args)

    fits_path.unlink()
    sidecar["render_state_id"] = "still-wrong"
    _write_json(json_path, sidecar)
    with pytest.raises(FileExistsError, match="partial output"):
        render_master_v4.run(args)


def test_verify_only_checks_existing_fits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan_root, _ = _make_fixture(tmp_path)
    output_root = tmp_path / "out"
    monkeypatch.setattr(render_master_v4, "_build_renderer_system", lambda plan: _fake_renderer_spec())
    args = render_master_v4.build_parser().parse_args(
        [
            "--plan-root",
            str(plan_root),
            "--output-root",
            str(output_root),
            "--render-index",
            "0",
            "--allow-nonproduction-contract",
        ]
    )
    render_master_v4.run(args)
    verify = render_master_v4.build_parser().parse_args(
        [
            "--plan-root",
            str(plan_root),
            "--output-root",
            str(output_root),
            "--render-index",
            "0",
            "--verify-only",
            "--allow-nonproduction-contract",
        ]
    )

    summary = render_master_v4.run(verify)

    assert summary["skipped_valid"] == 1


def test_slurm_array_chunking() -> None:
    plan = build_slurm_array_plan(
        render_count=14,
        renders_per_task=4,
        concurrency=3,
        nuisance_count=2,
    )

    assert plan.task_count == 4
    assert plan.final_task_size == 2
    assert plan.array_expression == "0-3%3"
    assert array_task_range(task_id=0, render_count=14, renders_per_task=4) == (0, 4)
    assert array_task_range(task_id=2, render_count=14, renders_per_task=4) == (8, 12)
    assert array_task_range(task_id=3, render_count=14, renders_per_task=4) == (12, 14)
    assert build_slurm_array_plan(render_count=12, renders_per_task=4).final_task_size == 4
    with pytest.raises(ValueError):
        build_slurm_array_plan(render_count=14, renders_per_task=0)


def test_submit_helper_dry_run_prints_array_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan_root, _ = _make_fixture(tmp_path)
    monkeypatch.setattr(submit_master_v4_render, "_repo_sha", lambda repo_root: "abc123")

    rc = submit_master_v4_render.main(
        [
            "--plan-root",
            str(plan_root),
            "--output-root",
            str(tmp_path / "out"),
            "--scratch-root",
            str(tmp_path / "scratch"),
            "--repo-root",
            str(tmp_path),
            "--renders-per-task",
            "4",
            "--concurrency",
            "3",
            "--allow-nonproduction-contract",
            "--dry-run",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total_renders"] == 14
    assert payload["number_of_array_tasks"] == 4
    assert payload["final_partial_task_size"] == 2
    assert payload["array_expression"] == "0-3%3"
    assert "--array=0-3%3" in payload["exact_sbatch_command"]
    assert payload["environment"]["V4_RENDER_COUNT"] == "14"
