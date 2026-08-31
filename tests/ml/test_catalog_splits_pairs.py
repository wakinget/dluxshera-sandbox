from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from dluxshera.datasets import ArrayShardStore
from dluxshera.datasets.schema import read_json, write_json
from dluxshera.ml import (
    PairPolicy,
    PairSampler,
    generate_frozen_pair_manifest,
    generate_split_registry,
    load_pair_manifest,
    load_sample_catalog,
    load_split_registry,
    write_pair_manifest,
    write_split_registry,
)


def _science_hash(theta: np.ndarray) -> str:
    payload = json.dumps([float(v) for v in theta], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_vector_spaces(root: Path, sigmas: np.ndarray) -> None:
    components = [
        {
            "label": "source.contrast",
            "index": 0,
            "unit": None,
            "scale": {"kind": "fisher_diagonal_sigma", "value": float(sigmas[0])},
            "metadata": {},
        },
        {
            "label": "optics.primary.zernike_coeffs_nm[0]",
            "index": 1,
            "unit": "nm",
            "scale": {"kind": "fisher_diagonal_sigma", "value": float(sigmas[1])},
            "metadata": {},
        },
    ]
    nuisance = [
        {"label": "source.x_position_as", "index": 0, "metadata": {}},
        {"label": "source.y_position_as", "index": 1, "metadata": {}},
    ]
    write_json(
        root / "vector_spaces.json",
        {
            "schema_version": "shera_v3_vector_spaces/1",
            "spaces": {
                "physical_delta": {"name": "physical", "components": components},
                "fisher_scaled_delta": {"name": "fisher", "components": components},
                "registration_nuisance": {"name": "nuisance", "components": nuisance},
                "registration_nuisance_sigma": {"name": "nuisance_sigma", "components": nuisance},
            },
            "transforms": {
                "fisher_diagonal_scale": {
                    "type": "DiagonalScaleTransform",
                    "scales": sigmas.tolist(),
                    "forward_mode": "divide",
                }
            },
        },
    )


def _write_prepared_fixture(root: Path) -> Path:
    sigmas = np.asarray([0.5, 2.0], dtype=np.float32)
    z_states = [
        np.asarray([-1.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.asarray([1.0, 0.0], dtype=np.float32),
        np.asarray([0.0, -1.0], dtype=np.float32),
        np.asarray([0.0, 1.0], dtype=np.float32),
        np.asarray([1.0, 1.0], dtype=np.float32),
    ]
    nuisance_vectors = [
        np.asarray([0.0, 0.0], dtype=np.float32),
        np.asarray([0.1, 0.0], dtype=np.float32),
        np.asarray([0.0, 0.2], dtype=np.float32),
    ]
    yy, xx = np.mgrid[0:8, 0:8].astype(np.float32)
    images: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    sample_index = 0
    for science_index, z in enumerate(z_states):
        theta = z * sigmas
        science_group = _science_hash(theta)
        for nuisance_id, nuisance in enumerate(nuisance_vectors):
            image = 100.0 + 3.0 * z[0] * xx + 2.0 * z[1] * yy + nuisance_id
            images.append(image.astype(np.float32))
            metadata.append(
                {
                    "sample_id": f"sample_{sample_index:04d}",
                    "sample_index": sample_index,
                    "dataset_family": "pair_grid",
                    "sample_role": "pair_grid",
                    "pair_id": "pair_main",
                    "grid_i_index": science_index,
                    "grid_j_index": 0,
                    "nuisance_id": nuisance_id,
                    "physical_delta": theta.tolist(),
                    "fisher_scaled_delta": z.tolist(),
                    "nuisance_vector": nuisance.tolist(),
                    "nuisance_sigma_vector": nuisance.tolist(),
                    "group_ids": {
                        "physical_delta_sha256": science_group,
                        "nuisance": str(nuisance_id),
                    },
                }
            )
            sample_index += 1
    _write_vector_spaces(root, sigmas)
    shard_manifest = ArrayShardStore(
        root,
        storage_dtype="float32",
        max_samples_per_shard=5,
        manifest_name="array_shards_manifest.json",
        index_name="index.jsonl",
    ).write(images, sample_metadata=metadata)
    write_json(
        root / "manifest.json",
        {
            "schema_version": "shera_prepared_dataset/1",
            "source_dataset": {
                "samples_sha256": "synthetic",
                "prepared_sample_count": len(images),
            },
            "array_storage": {
                "sample_shape": [8, 8],
                "sample_count": len(images),
                "storage_dtype": "float32",
                "shard_count": shard_manifest["shard_count"],
            },
            "index_format": {"path": "index.jsonl", "format": "jsonl"},
        },
    )
    return root


def _split(catalog):
    return generate_split_registry(
        catalog,
        seed=7,
        science_fractions={"train": 0.34, "validation": 0.33, "test": 0.33},
        nuisance_fractions={"train": 0.34, "validation": 0.33, "test": 0.33},
    )


def test_catalog_streams_prepared_index_into_compact_arrays(tmp_path: Path) -> None:
    catalog = load_sample_catalog(_write_prepared_fixture(tmp_path))
    assert catalog.sample_count == 18
    assert catalog.science_dim == 2
    assert catalog.nuisance_dim == 2
    assert catalog.science_group_count == 6
    assert catalog.nuisance_group_count == 3
    assert catalog.parameter_labels == (
        "source.contrast",
        "optics.primary.zernike_coeffs_nm[0]",
    )
    assert not hasattr(catalog, "rows")
    assert catalog.sample_id_to_index["sample_0000"] == 0
    np.testing.assert_allclose(catalog.fisher_scaled_deltas[3], [-0.0, 0.0], atol=1.0e-7)
    np.testing.assert_allclose(catalog.physical_from_z([[1.0, -1.0]]), [[0.5, -2.0]])


def test_split_registry_is_deterministic_serializes_and_rejects_mismatch(tmp_path: Path) -> None:
    catalog = load_sample_catalog(_write_prepared_fixture(tmp_path / "prepared"))
    first = _split(catalog)
    second = _split(catalog)
    assert first.science_assignments == second.science_assignments
    assert first.nuisance_assignments == second.nuisance_assignments
    assert first.counts["science_groups"] == {"test": 2, "train": 2, "validation": 2}
    assert first.counts["nuisance_groups"] == {"test": 1, "train": 1, "validation": 1}

    for group_id in set(catalog.science_group_ids):
        sample_splits = {
            first.science_split(str(catalog.science_group_ids[idx]))
            for idx in np.flatnonzero(catalog.science_group_ids == group_id)
        }
        assert len(sample_splits) == 1

    path = tmp_path / "split.json"
    write_split_registry(path, first)
    loaded = load_split_registry(path, catalog=catalog)
    assert loaded.to_dict() == first.to_dict()

    wrong_payload = read_json(path)
    wrong_payload["prepared_dataset"]["prepared_dataset_hash"] = "wrong"
    wrong_path = tmp_path / "wrong_split.json"
    write_json(wrong_path, wrong_payload)
    with pytest.raises(ValueError, match="different prepared dataset"):
        load_split_registry(wrong_path, catalog=catalog)


def test_split_registry_supports_explicit_nuisance_assignments(tmp_path: Path) -> None:
    catalog = load_sample_catalog(_write_prepared_fixture(tmp_path))
    assignments = {"0": "train", "1": "validation", "2": "test"}
    registry = generate_split_registry(
        catalog,
        seed=4,
        science_fractions={"train": 1.0},
        nuisance_fractions={"train": 1.0},
        explicit_nuisance_assignments=assignments,
    )
    assert registry.nuisance_assignments == assignments
    with pytest.raises(ValueError, match="cover exactly"):
        generate_split_registry(
            catalog,
            seed=4,
            science_fractions={"train": 1.0},
            explicit_nuisance_assignments={"0": "train"},
        )


def test_pair_sampler_same_nuisance_targets_reverse_and_distance(tmp_path: Path) -> None:
    catalog = load_sample_catalog(_write_prepared_fixture(tmp_path))
    registry = _split(catalog)
    policy = PairPolicy(
        family_weights={"same_nuisance_different_science": 1.0},
        same_pair_id=True,
        min_fisher_distance=0.5,
        max_fisher_distance=3.0,
    )
    sampler = PairSampler(catalog, registry, policy)
    record = sampler.sample_pair(np.random.default_rng(1), science_split="train", nuisance_split="train")
    assert record.nuisance_a_id == record.nuisance_b_id
    assert record.science_a_id != record.science_b_id
    a = catalog.sample_id_to_index[record.sample_a_id]
    b = catalog.sample_id_to_index[record.sample_b_id]
    np.testing.assert_allclose(
        record.target_delta_z,
        catalog.fisher_scaled_deltas[b] - catalog.fisher_scaled_deltas[a],
    )
    np.testing.assert_allclose(
        record.target_delta_theta,
        catalog.physical_deltas[b] - catalog.physical_deltas[a],
    )
    assert 0.5 <= record.fisher_distance_l2 <= 3.0

    reverse = sampler.make_pair_record(b, a, family=record.pair_family, split="train", eval_slice=None)
    np.testing.assert_allclose(reverse.target_delta_z, -np.asarray(record.target_delta_z))
    assert reverse.sample_a_id == record.sample_b_id
    assert reverse.sample_b_id == record.sample_a_id


def test_pair_sampler_other_family_semantics_and_split_boundaries(tmp_path: Path) -> None:
    catalog = load_sample_catalog(_write_prepared_fixture(tmp_path))
    registry = generate_split_registry(
        catalog,
        seed=3,
        science_fractions={"train": 1.0},
        nuisance_fractions={"train": 1.0},
    )
    same_science = PairSampler(
        catalog,
        registry,
        PairPolicy(
            family_weights={"same_science_different_nuisance": 1.0},
            same_pair_id=True,
            min_fisher_distance=0.0,
            max_fisher_distance=0.0,
        ),
    ).sample_pair(np.random.default_rng(2))
    assert same_science.science_a_id == same_science.science_b_id
    assert same_science.nuisance_a_id != same_science.nuisance_b_id
    np.testing.assert_allclose(same_science.target_delta_z, [0.0, 0.0])

    different_both = PairSampler(
        catalog,
        registry,
        PairPolicy(
            family_weights={"different_science_different_nuisance": 1.0},
            same_pair_id=True,
            min_fisher_distance=0.5,
            max_fisher_distance=3.0,
        ),
    ).sample_pair(np.random.default_rng(3))
    assert different_both.science_a_id != different_both.science_b_id
    assert different_both.nuisance_a_id != different_both.nuisance_b_id


def test_frozen_eval_manifest_is_deterministic_and_validates(tmp_path: Path) -> None:
    catalog = load_sample_catalog(_write_prepared_fixture(tmp_path / "prepared"))
    registry = _split(catalog)
    policy = PairPolicy(
        family_weights={"same_nuisance_different_science": 1.0},
        same_pair_id=True,
        min_fisher_distance=0.5,
        max_fisher_distance=3.0,
        include_reverse=True,
    )
    first = generate_frozen_pair_manifest(
        catalog,
        registry,
        policy=policy,
        split="validation",
        seed=5,
        pairs_per_slice=2,
    )
    second = generate_frozen_pair_manifest(
        catalog,
        registry,
        policy=policy,
        split="validation",
        seed=5,
        pairs_per_slice=2,
    )
    assert [r.to_dict() for r in first.records] == [r.to_dict() for r in second.records]
    assert first.manifest["pair_count"] == len(first.records)
    assert set(first.manifest["eval_slice_counts"]) == {
        "heldout_science_heldout_nuisance",
        "heldout_science_seen_nuisance",
    }
    assert all(record.sample_a_id != record.sample_b_id for record in first.records)
    assert all(record.sample_a_index >= 0 and record.sample_b_index >= 0 for record in first.records)

    outdir = tmp_path / "pairs"
    write_pair_manifest(outdir, first)
    loaded = load_pair_manifest(outdir, catalog=catalog, split_registry=registry)
    assert loaded.summary()["pair_count"] == len(first.records)
