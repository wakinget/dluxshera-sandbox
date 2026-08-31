from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from dluxshera.datasets import ArrayShardReader
from dluxshera.datasets.schema import read_json, read_jsonl
from dluxshera.datasets.shera import (
    _portable_path,
    build_shera_v3_vector_spaces,
    prepare_shera_v3_dataset,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _synthetic_v3_dataset(root: Path) -> list[np.ndarray]:
    _write_json(
        root / "manifest.json",
        {
            "schema_version": "ml_training_dataset_v3_manifest/2",
            "generator": "test",
            "rendered_sample_count": 3,
            "render_complete": True,
            "next_sample_index": 3,
            "render_target_sample_count": 3,
            "dataset_family_counts": {"pair_grid": 2, "sparse_mixture": 1},
            "nuisance_config": {"keys": ["source.x_position_as", "source.y_position_as"]},
            "parameter_space_summary": {"count": 2},
        },
    )
    _write_json(
        root / "parameter_space.json",
        {
            "parameters": [
                {
                    "label": "source.contrast",
                    "base_key": "source.contrast",
                    "component_index": None,
                    "nominal_value": 1.0,
                    "parameter_sigma": 0.5,
                    "sweep_source_key": "source.contrast",
                    "sweep_config": {"min_sigma": 1.0, "max_sigma": 3.0},
                    "min_abs_delta": 0.5,
                    "max_abs_delta": 1.5,
                    "units": None,
                    "display_label": "contrast",
                    "group": "source",
                    "noll_index": None,
                },
                {
                    "label": "optics.primary.zernike_coeffs_nm[0]",
                    "base_key": "optics.primary.zernike_coeffs_nm",
                    "component_index": 0,
                    "nominal_value": 0.0,
                    "parameter_sigma": 2.0,
                    "sweep_source_key": "optics.primary.zernike_coeffs_nm",
                    "sweep_config": {"min_sigma": 1.0, "max_sigma": 3.0},
                    "min_abs_delta": 2.0,
                    "max_abs_delta": 6.0,
                    "units": "nm",
                    "display_label": "z0",
                    "group": "optics",
                    "noll_index": 4,
                },
            ]
        },
    )
    images = [
        np.arange(4, dtype=np.float64).reshape(2, 2),
        np.arange(4, dtype=np.float64).reshape(2, 2) + 10.0,
        np.arange(4, dtype=np.float64).reshape(2, 2) + 20.0,
    ]
    rows: list[dict[str, object]] = []
    for idx, image in enumerate(images):
        sample_id = f"sample_{idx:06d}"
        path = root / "images" / f"{sample_id}.fits"
        path.parent.mkdir(parents=True, exist_ok=True)
        fits.PrimaryHDU(data=image).writeto(path)
        row: dict[str, object] = {
            "dataset_family": "pair_grid" if idx < 2 else "sparse_mixture",
            "sample_role": "pair_grid" if idx < 2 else "sparse_random",
            "sample_id": sample_id,
            "sample_index": idx,
            "fits_path": f"images/{sample_id}.fits",
            "metadata_path": f"images/{sample_id}.json",
            "theta_delta": {"source.contrast": 0.25 * idx},
            "theta_sigma": {"source.contrast": 0.5 * idx},
            "registration_nuisance_values": {"source.x_position_as": 0.01 * idx},
            "registration_nuisance_sigma_values": {"source.x_position_as": 0.1 * idx},
            "skipped_nuisance_keys": ["source.y_position_as"] if idx == 1 else [],
            "nuisance_id": idx % 2,
            "seed": 100 + idx,
        }
        if idx < 2:
            row.update(
                {
                    "pair_id": "pair_000_001",
                    "pair_label_i": "source.contrast",
                    "pair_label_j": "optics.primary.zernike_coeffs_nm[0]",
                    "grid_i_index": idx,
                    "grid_j_index": 1,
                    "grid_i_sigma": float(idx),
                    "grid_j_sigma": 1.0,
                    "delta_i": 0.25 * idx,
                    "delta_j": 0.0,
                    "delta_units": "parameter_units",
                    "controlled_labels": ["source.contrast"],
                }
            )
        else:
            row.update(
                {
                    "active_count": 1,
                    "active_labels": ["source.contrast"],
                    "active_mask": [1, 0],
                    "split": "test",
                }
            )
        rows.append(row)
    _write_jsonl(root / "samples.jsonl", rows)
    return images


def test_build_shera_v3_vector_spaces_and_fisher_transform(tmp_path: Path) -> None:
    _synthetic_v3_dataset(tmp_path)
    parameter_space = read_json(tmp_path / "parameter_space.json")["parameters"]
    physical, fisher, nuisance, nuisance_sigma, transform = build_shera_v3_vector_spaces(
        parameter_space,
        nuisance_labels=("source.x_position_as",),
    )
    assert physical.labels == (
        "source.contrast",
        "optics.primary.zernike_coeffs_nm[0]",
    )
    assert fisher.dimension == 2
    assert nuisance is not None
    assert nuisance.labels == ("source.x_position_as",)
    assert nuisance_sigma is not None
    assert nuisance_sigma.labels == ("source.x_position_as",)
    np.testing.assert_array_equal(transform.forward([1.0, 4.0]), np.array([2.0, 2.0]))


def test_build_shera_v3_vector_spaces_requires_positive_fisher_sigma(
    tmp_path: Path,
) -> None:
    _synthetic_v3_dataset(tmp_path)
    parameter_space = read_json(tmp_path / "parameter_space.json")["parameters"]
    parameter_space[0]["parameter_sigma"] = 0.0
    with pytest.raises(ValueError, match="finite and > 0"):
        build_shera_v3_vector_spaces(parameter_space)

    parameter_space[0]["parameter_sigma"] = -1.0
    with pytest.raises(ValueError, match="finite and > 0"):
        build_shera_v3_vector_spaces(parameter_space)


def test_portable_path_resolves_under_base_and_sibling_directories(
    tmp_path: Path,
) -> None:
    source = tmp_path / "Results" / "source_v3"
    prepared = tmp_path / "Results" / "prepared_v3"
    source.mkdir(parents=True)
    prepared.mkdir(parents=True)
    artifact = source / "manifest.json"
    artifact.write_text("{}", encoding="utf-8")

    under_source = _portable_path(artifact, base=source)
    assert under_source == "manifest.json"
    assert (source / under_source).resolve() == artifact.resolve()

    sibling = _portable_path(source, base=prepared)
    assert sibling == "../source_v3"
    assert (prepared / sibling).resolve() == source.resolve()


def test_portable_path_falls_back_to_absolute_when_relpath_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "manifest.json"
    artifact.write_text("{}", encoding="utf-8")

    def fail_relpath(path: Path, *, start: Path) -> str:
        raise ValueError("different drives")

    import dluxshera.datasets.shera as shera_module

    monkeypatch.setattr(shera_module.os.path, "relpath", fail_relpath)
    assert _portable_path(artifact, base=tmp_path / "prepared") == str(artifact.resolve())


def test_prepare_shera_v3_dataset_writes_self_describing_outputs(tmp_path: Path) -> None:
    source = tmp_path / "source"
    expected_images = _synthetic_v3_dataset(source)
    outdir = tmp_path / "prepared"

    summary = prepare_shera_v3_dataset(
        source_root=source,
        outdir=outdir,
        dtype="float32",
        max_samples_per_shard=2,
        validation_samples=2,
        seed=7,
    )

    assert summary.sample_count == 3
    assert summary.total_source_sample_count == 3
    assert summary.sample_shape == (2, 2)
    assert summary.storage_dtype == "float32"
    assert summary.shard_count == 2
    manifest = read_json(outdir / "manifest.json")
    assert manifest["schema_version"] == "shera_prepared_dataset/1"
    assert manifest["canonical_policy"]["mutates_source_dataset"] is False
    assert manifest["index_format"]["parquet_required"] is False
    assert manifest["source_dataset"]["samples_sha256"]
    assert manifest["source_dataset"]["total_source_sample_count"] == 3
    assert manifest["source_dataset"]["prepared_sample_count"] == 3
    assert manifest["source_dataset"]["selection_policy"]["description"] == "all source samples"
    assert (outdir / "vector_spaces.json").exists()
    assert (outdir / "validation" / "precision_summary.json").exists()

    rows = list(read_jsonl(outdir / "index.jsonl"))
    assert len(rows) == 3
    assert rows[0]["source_fits_path"] == "images/sample_000000.fits"
    assert rows[0]["shard_id"] == "shard_00000"
    assert rows[2]["shard_id"] == "shard_00001"
    assert rows[1]["physical_delta"] == [0.25, 0.0]
    assert rows[1]["fisher_scaled_delta"] == [0.5, 0.0]
    assert rows[1]["nuisance_vector"] == [0.01, 0.0]
    assert rows[1]["nuisance_sigma_vector"] == [0.1, 0.0]
    assert rows[1]["registration_nuisance_sigma_values"] == {"source.x_position_as": 0.1}
    assert rows[1]["skipped_nuisance_keys"] == ["source.y_position_as"]
    assert rows[1]["seed"] == 101
    assert rows[1]["pair_label_i"] == "source.contrast"
    assert rows[1]["pair_label_j"] == "optics.primary.zernike_coeffs_nm[0]"
    assert rows[1]["grid_i_sigma"] == 1.0
    assert rows[1]["delta_units"] == "parameter_units"
    assert rows[2]["active_count"] == 1
    assert rows[2]["theta_sigma"] == {"source.contrast": 1.0}
    assert rows[2]["split"] == "test"
    assert rows[1]["source_dtype"] in {"float64", ">f8"}
    assert rows[1]["storage_dtype"] == "float32"
    vector_spaces = read_json(outdir / "vector_spaces.json")
    assert (
        vector_spaces["spaces"]["registration_nuisance_sigma"]["name"]
        == "shera_v3_registration_nuisance_sigma"
    )

    with ArrayShardReader(outdir, cache_size=1) as reader:
        assert reader.open_shard_count == 0
        assert reader[0].dtype == np.float32
        np.testing.assert_allclose(reader[0], expected_images[0].astype(np.float32))
        np.testing.assert_allclose(reader[2], expected_images[2].astype(np.float32))
        assert reader.open_shard_count == 1

    precision_rows = list(read_jsonl(outdir / "validation" / "precision_samples.jsonl"))
    assert len(precision_rows) == 2
    assert all(row["readback_dtype"] == "float32" for row in precision_rows)
    assert all(row["readback_matches_expected_cast"] for row in precision_rows)
    assert all(row["source_sample_id_matches_index"] for row in precision_rows)


def test_prepare_shera_v3_dataset_dry_run_does_not_write(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _synthetic_v3_dataset(source)
    outdir = tmp_path / "prepared"
    summary = prepare_shera_v3_dataset(
        source_root=source,
        outdir=outdir,
        dtype="float64",
        max_samples_per_shard=2,
        validation_samples=4,
        dry_run=True,
    )
    assert summary.dry_run is True
    assert summary.shard_count == 2
    assert summary.source_probe_policy == "first_selected_source_sample"
    assert not outdir.exists()


def test_prepare_shera_v3_dataset_fits_read_counts_are_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    _synthetic_v3_dataset(source)
    from dluxshera.datasets import shera

    real_read = shera._read_fits_array
    count = {"n": 0}

    def counted(path: Path) -> np.ndarray:
        count["n"] += 1
        return real_read(path)

    monkeypatch.setattr(shera, "_read_fits_array", counted)
    dry = prepare_shera_v3_dataset(
        source_root=source,
        outdir=tmp_path / "dry",
        dry_run=True,
        validation_samples=2,
    )
    assert dry.sample_count == 3
    assert count["n"] == 1

    count["n"] = 0
    prepare_shera_v3_dataset(
        source_root=source,
        outdir=tmp_path / "prepared",
        dtype="float32",
        validation_samples=2,
        seed=0,
    )
    assert count["n"] == 5


def test_prepare_shera_v3_dataset_rejects_inconsistent_source_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _synthetic_v3_dataset(source)
    manifest = read_json(source / "manifest.json")
    manifest["rendered_sample_count"] = 4
    _write_json(source / "manifest.json", manifest)
    with pytest.raises(ValueError, match="rendered_sample_count"):
        prepare_shera_v3_dataset(source_root=source, outdir=tmp_path / "prepared")


def test_prepare_shera_v3_dataset_rejects_incomplete_source_by_default(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _synthetic_v3_dataset(source)
    manifest = read_json(source / "manifest.json")
    manifest["render_complete"] = False
    _write_json(source / "manifest.json", manifest)
    with pytest.raises(ValueError, match="render_complete=false"):
        prepare_shera_v3_dataset(source_root=source, outdir=tmp_path / "prepared")
    summary = prepare_shera_v3_dataset(
        source_root=source,
        outdir=tmp_path / "partial",
        allow_incomplete_source=True,
        validation_samples=1,
    )
    assert summary.sample_count == 3
    prepared_manifest = read_json(tmp_path / "partial" / "manifest.json")
    assert prepared_manifest["source_dataset"]["completeness"]["allow_incomplete_source"]


def test_prepare_shera_v3_dataset_rejects_duplicate_or_bad_sample_index(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _synthetic_v3_dataset(source)
    rows = list(read_jsonl(source / "samples.jsonl"))
    rows[1]["sample_id"] = rows[0]["sample_id"]
    _write_jsonl(source / "samples.jsonl", rows)
    with pytest.raises(ValueError, match="duplicate sample_id"):
        prepare_shera_v3_dataset(source_root=source, outdir=tmp_path / "dup")

    source2 = tmp_path / "source2"
    _synthetic_v3_dataset(source2)
    rows = list(read_jsonl(source2 / "samples.jsonl"))
    rows[1]["sample_index"] = 7
    _write_jsonl(source2 / "samples.jsonl", rows)
    with pytest.raises(ValueError, match="sample_index"):
        prepare_shera_v3_dataset(source_root=source2, outdir=tmp_path / "bad_index")


def test_prepare_shera_v3_dataset_prefix_selection(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _synthetic_v3_dataset(source)
    outdir = tmp_path / "prepared"
    summary = prepare_shera_v3_dataset(
        source_root=source,
        outdir=outdir,
        max_samples=2,
        validation_samples=1,
    )
    assert summary.total_source_sample_count == 3
    assert summary.sample_count == 2
    rows = list(read_jsonl(outdir / "index.jsonl"))
    assert [row["source_sample_id"] for row in rows] == ["sample_000000", "sample_000001"]
    manifest = read_json(outdir / "manifest.json")
    assert manifest["source_dataset"]["selection_policy"]["requested_max_samples"] == 2
    assert manifest["source_dataset"]["prepared_sample_count"] == 2


def test_prepare_shera_v3_dataset_rejects_unknown_theta_delta_key(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _synthetic_v3_dataset(source)
    rows = list(read_jsonl(source / "samples.jsonl"))
    rows[0]["theta_delta"] = {"not.in.parameter_space": 1.0}
    _write_jsonl(source / "samples.jsonl", rows)

    with pytest.raises(ValueError, match="theta_delta contains keys"):
        prepare_shera_v3_dataset(source_root=source, outdir=tmp_path / "prepared")
