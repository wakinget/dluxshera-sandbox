from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from dluxshera.datasets.master_v4 import FrozenMasterV4, render_output_paths
from dluxshera.datasets.prepared_v4 import (
    prepare_shera_v4_dataset,
    validate_prepared_v4_dataset_identity,
)
from dluxshera.datasets.schema import read_json, write_json
from dluxshera.ml import load_sample_catalog
from tests.ml.test_master_v4_render import _make_fixture


def _write_v4_renders(plan_root: Path, source_root: Path) -> None:
    master = FrozenMasterV4(plan_root, expected=None)
    states = master.resolve_indices(range(master.render_count))
    for state in states:
        fits_path, sidecar_path = render_output_paths(
            output_root=source_root,
            dataset_family=state.dataset_family,
            split_role=state.split_role,
            render_index=state.render_index,
        )
        fits_path.parent.mkdir(parents=True, exist_ok=True)
        image = np.full((3, 3), float(state.render_index + 1), dtype=np.float32)
        fits.writeto(fits_path, image, overwrite=True)
        write_json(
            sidecar_path,
            {
                "render_index": state.render_index,
                "dataset_family": state.dataset_family,
                "split_role": state.split_role,
                "science_state_id": state.science_state_id,
                "nuisance_state_id": state.nuisance_state_id,
                "render_state_id": state.render_state_id,
                "render_system_contract_hash": master.manifest["render_system_contract_hash"],
            },
        )


def test_prepared_v4_identity_is_portable_and_detects_index_and_shard_tamper(tmp_path: Path) -> None:
    plan_root, expected = _make_fixture(tmp_path / "fixture")
    source_root = tmp_path / "source_a"
    _write_v4_renders(plan_root, source_root)
    out = tmp_path / "prepared_a"
    summary = prepare_shera_v4_dataset(
        source_root=source_root,
        plan_root=plan_root,
        outdir=out,
        dtype="float32",
        max_samples_per_shard=4,
        validation_samples=2,
        source_audit="full",
        expected=expected,
    )
    assert summary.sample_count == 14
    baseline = read_json(out / "manifest.json")["content_identity"]["sha256"]
    catalog = load_sample_catalog(out)
    assert catalog.artifact_id == "PREP-V4-v1"
    assert catalog.sample_count == 14
    assert set(catalog.dataset_families) == {"joint_full_v4", "radial_capture_v4"}

    relocated = tmp_path / "elsewhere" / "prepared_b"
    shutil.copytree(out, relocated)
    assert validate_prepared_v4_dataset_identity(relocated)["index"]["row_count"] == 14
    assert read_json(relocated / "manifest.json")["content_identity"]["sha256"] == baseline
    assert load_sample_catalog(relocated).prepared_dataset_hash == baseline

    rows = (relocated / "index.jsonl").read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["sample_id"] = "tampered"
    rows[0] = json.dumps(first, sort_keys=True)
    (relocated / "index.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="content_tree"):
        validate_prepared_v4_dataset_identity(relocated)

    shard_tamper = tmp_path / "shard_tamper"
    shutil.copytree(out, shard_tamper)
    shard_path = shard_tamper / "shards" / "shard_00000.npy"
    arr = np.load(shard_path)
    arr[0, 0, 0] += 1.0
    np.save(shard_path, arr)
    with pytest.raises(ValueError, match="shard hash"):
        validate_prepared_v4_dataset_identity(shard_tamper, deep=True)


def test_prepared_v4_source_sidecar_audit_and_resume_guards(tmp_path: Path) -> None:
    plan_root, expected = _make_fixture(tmp_path / "fixture")
    source_root = tmp_path / "source"
    _write_v4_renders(plan_root, source_root)
    out = tmp_path / "prepared"
    prepare_shera_v4_dataset(
        source_root=source_root,
        plan_root=plan_root,
        outdir=out,
        dtype="float32",
        max_samples_per_shard=4,
        validation_samples=1,
        source_audit="full",
        expected=expected,
    )
    for name in ("manifest.json", "index.jsonl", "vector_spaces.json", "array_shards_manifest.json"):
        (out / name).unlink()
    shutil.rmtree(out / "validation")
    resumed = prepare_shera_v4_dataset(
        source_root=source_root,
        plan_root=plan_root,
        outdir=out,
        dtype="float32",
        max_samples_per_shard=4,
        validation_samples=1,
        source_audit="full",
        resume=True,
        expected=expected,
    )
    assert resumed.sample_count == 14
    assert (out / "preparation_state.json").exists()

    with pytest.raises(ValueError, match="incompatible"):
        prepare_shera_v4_dataset(
            source_root=source_root,
            plan_root=plan_root,
            outdir=out,
            dtype="float64",
            max_samples_per_shard=4,
            resume=True,
            expected=expected,
        )

    bad_source = tmp_path / "bad_source"
    shutil.copytree(source_root, bad_source)
    sidecar = next(bad_source.rglob("*.json"))
    payload = read_json(sidecar)
    payload["render_state_id"] = "wrong"
    write_json(sidecar, payload)
    with pytest.raises(ValueError, match="sidecar"):
        prepare_shera_v4_dataset(
            source_root=bad_source,
            plan_root=plan_root,
            outdir=tmp_path / "bad_prepared",
            dtype="float32",
            max_samples_per_shard=4,
            source_audit="full",
            expected=expected,
        )


def test_prepared_v4_resume_regenerates_stale_same_shape_shard(tmp_path: Path) -> None:
    plan_root, expected = _make_fixture(tmp_path / "fixture")
    source_root = tmp_path / "source"
    _write_v4_renders(plan_root, source_root)
    out = tmp_path / "prepared"
    prepare_shera_v4_dataset(
        source_root=source_root,
        plan_root=plan_root,
        outdir=out,
        dtype="float32",
        max_samples_per_shard=4,
        validation_samples=1,
        source_audit="full",
        expected=expected,
    )
    shard_path = out / "shards" / "shard_00000.npy"
    original_hash = read_json(out / "preparation_state.json")["completed_shard_hashes"]["shard_00000"]
    stale = np.load(shard_path)
    stale[:] = 999.0
    np.save(shard_path, stale)
    for name in ("manifest.json", "index.jsonl", "vector_spaces.json", "array_shards_manifest.json"):
        (out / name).unlink()
    shutil.rmtree(out / "validation")

    prepare_shera_v4_dataset(
        source_root=source_root,
        plan_root=plan_root,
        outdir=out,
        dtype="float32",
        max_samples_per_shard=4,
        validation_samples=1,
        source_audit="full",
        resume=True,
        expected=expected,
    )
    assert read_json(out / "preparation_state.json")["completed_shard_hashes"]["shard_00000"] == original_hash
    assert validate_prepared_v4_dataset_identity(out, deep=True)


def test_prepared_v4_resume_rejects_incompatible_preparation_state(tmp_path: Path) -> None:
    plan_root, expected = _make_fixture(tmp_path / "fixture")
    source_root = tmp_path / "source"
    _write_v4_renders(plan_root, source_root)
    out = tmp_path / "prepared"
    prepare_shera_v4_dataset(
        source_root=source_root,
        plan_root=plan_root,
        outdir=out,
        dtype="float32",
        max_samples_per_shard=4,
        validation_samples=1,
        source_audit="full",
        expected=expected,
    )
    for name in ("manifest.json", "index.jsonl", "vector_spaces.json", "array_shards_manifest.json"):
        (out / name).unlink()
    shutil.rmtree(out / "validation")
    state = read_json(out / "preparation_state.json")
    state["request_identity"]["array_storage"]["storage_dtype"] = "float64"
    write_json(out / "preparation_state.json", state)
    with pytest.raises(ValueError, match="preparation_state.json is incompatible"):
        prepare_shera_v4_dataset(
            source_root=source_root,
            plan_root=plan_root,
            outdir=out,
            dtype="float32",
            max_samples_per_shard=4,
            validation_samples=1,
            resume=True,
            expected=expected,
        )


def test_prepared_v4_sidecar_requires_mandatory_identity_fields(tmp_path: Path) -> None:
    plan_root, expected = _make_fixture(tmp_path / "fixture")
    source_root = tmp_path / "source"
    _write_v4_renders(plan_root, source_root)
    sidecar = next(source_root.rglob("*.json"))
    payload = read_json(sidecar)
    payload.pop("science_state_id")
    write_json(sidecar, payload)
    with pytest.raises(ValueError, match="missing mandatory identity field"):
        prepare_shera_v4_dataset(
            source_root=source_root,
            plan_root=plan_root,
            outdir=tmp_path / "prepared",
            dtype="float32",
            max_samples_per_shard=4,
            validation_samples=1,
            source_audit="full",
            expected=expected,
        )
