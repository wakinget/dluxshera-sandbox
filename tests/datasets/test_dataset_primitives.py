from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import dluxshera.datasets.arrays as arrays_module
from dluxshera.datasets import (
    ArrayShardReader,
    ArrayShardStore,
    CompositeTransform,
    DiagonalScaleTransform,
    LinearTransform,
    VectorComponentSpec,
    VectorSpaceSpec,
    assign_grouped_split,
    compare_arrays,
)
from dluxshera.datasets.schema import read_jsonl


def test_array_shards_round_trip_multiple_shards_and_cache(tmp_path: Path) -> None:
    samples = [np.full((2, 3), idx, dtype=np.float64) for idx in range(5)]
    metadata = ({"sample_id": f"s{idx}", "sample_index": idx} for idx in range(5))
    manifest = ArrayShardStore(
        tmp_path,
        storage_dtype=np.float64,
        max_samples_per_shard=2,
    ).write(samples, sample_metadata=metadata)

    assert manifest["shard_count"] == 3
    rows = list(read_jsonl(tmp_path / "array_index.jsonl"))
    assert [(row["shard_id"], row["shard_offset"]) for row in rows] == [
        ("shard_00000", 0),
        ("shard_00000", 1),
        ("shard_00001", 0),
        ("shard_00001", 1),
        ("shard_00002", 0),
    ]

    with ArrayShardReader(tmp_path, cache_size=1) as reader:
        np.testing.assert_array_equal(reader[0], samples[0])
        np.testing.assert_array_equal(reader[1], samples[1])
        np.testing.assert_array_equal(reader[2], samples[2])
        assert reader.open_shard_count == 1
        np.testing.assert_array_equal(reader[4], samples[4])
        assert reader.open_shard_count == 1


def test_array_shard_reader_default_samples_survive_eviction_and_close(tmp_path: Path) -> None:
    samples = [np.full((2, 2), idx, dtype=np.float64) for idx in range(4)]
    ArrayShardStore(tmp_path, max_samples_per_shard=2).write(samples)

    reader = ArrayShardReader(tmp_path, cache_size=1)
    retained = reader[0]
    np.testing.assert_array_equal(reader[2], samples[2])
    assert reader.open_shard_count == 1
    np.testing.assert_array_equal(retained, samples[0])
    reader.close()
    assert reader.open_shard_count == 0
    np.testing.assert_array_equal(retained, samples[0])


def test_array_shard_reader_boundaries_use_manifest_mapping(tmp_path: Path) -> None:
    samples = [np.full((1,), idx, dtype=np.int16) for idx in range(7)]
    ArrayShardStore(tmp_path, storage_dtype=np.int16, max_samples_per_shard=3).write(samples)
    with ArrayShardReader(tmp_path, cache_size=2) as reader:
        for idx in (0, 2, 3, 5, 6):
            np.testing.assert_array_equal(reader[idx], samples[idx])


def test_array_shards_float32_conversion_is_explicit(tmp_path: Path) -> None:
    samples = [np.array([[1.0, 1.0 + 1.0e-8]], dtype=np.float64)]
    manifest = ArrayShardStore(tmp_path, storage_dtype="float32").write(samples)
    assert manifest["source_dtypes"] == ["float64"]
    assert manifest["storage_dtype"] == "float32"
    with ArrayShardReader(tmp_path) as reader:
        assert reader[0].dtype == np.float32
        np.testing.assert_allclose(reader[0], samples[0].astype(np.float32))


def test_array_shards_reject_invalid_shape(tmp_path: Path) -> None:
    samples = [np.zeros((2, 2)), np.zeros((2, 3))]
    with pytest.raises(ValueError, match="shape"):
        ArrayShardStore(tmp_path).write(samples)


def test_array_shard_reader_reports_missing_and_corrupt_shards(tmp_path: Path) -> None:
    ArrayShardStore(tmp_path, max_samples_per_shard=1).write([np.zeros((2, 2))])
    shard_path = tmp_path / "shards" / "shard_00000.npy"
    shard_path.unlink()
    with ArrayShardReader(tmp_path) as reader:
        with pytest.raises(FileNotFoundError):
            reader[0]

    other = tmp_path / "corrupt"
    ArrayShardStore(other, max_samples_per_shard=1).write([np.zeros((2, 2))])
    (other / "shards" / "shard_00000.npy").write_text("not a npy", encoding="utf-8")
    with ArrayShardReader(other) as reader:
        with pytest.raises(ValueError, match="Could not load"):
            reader[0]


def test_array_shard_store_refuses_stale_shards_and_keeps_index_atomic(tmp_path: Path) -> None:
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    (shards_dir / "shard_00000.npy").write_bytes(b"stale")
    with pytest.raises(FileExistsError, match="non-empty shard directory"):
        ArrayShardStore(tmp_path).write([np.zeros((2, 2))])

    failing = tmp_path / "failing"
    with pytest.raises(ValueError, match="shape"):
        ArrayShardStore(failing, max_samples_per_shard=1).write(
            [np.zeros((2, 2)), np.zeros((3, 2))]
        )
    assert not (failing / "array_index.jsonl").exists()
    assert not (failing / "array_index.jsonl.tmp").exists()
    assert list((failing / "shards").glob("*.npy")) == []


def test_array_shard_store_finalization_failure_cleans_final_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_write_json(path: Path, payload: object) -> None:
        raise OSError("simulated manifest failure")

    monkeypatch.setattr(arrays_module, "write_json", fail_write_json)
    with pytest.raises(OSError, match="simulated manifest failure"):
        ArrayShardStore(tmp_path, max_samples_per_shard=1).write(
            [np.zeros((2, 2)), np.ones((2, 2))]
        )

    assert not (tmp_path / "array_shards_manifest.json").exists()
    assert not (tmp_path / "array_index.jsonl").exists()
    assert not (tmp_path / "array_index.jsonl.tmp").exists()
    assert list((tmp_path / "shards").glob("*.npy")) == []
    assert list((tmp_path / "shards").glob("*.tmp")) == []


def test_compare_arrays_exact_perturbed_zeros_and_nonfinite() -> None:
    exact = compare_arrays(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    assert exact.max_abs_error == 0.0
    assert exact.relative_l2_error == 0.0

    perturbed = compare_arrays(
        np.array([0.0, 2.0]),
        np.array([1.0e-9, 2.25]),
        relative_denominator_floor=1.0,
    )
    assert perturbed.max_abs_error == pytest.approx(0.25)
    assert perturbed.max_relative_error == pytest.approx(0.125)

    nonfinite = compare_arrays(
        np.array([np.nan, 1.0, np.inf]),
        np.array([0.0, 2.0, np.inf]),
    )
    assert nonfinite.finite_pair_count == 1
    assert nonfinite.nonfinite_reference_count == 2

    mismatch = compare_arrays(np.zeros((2, 2)), np.zeros((3, 2)))
    assert mismatch.shape_match is False
    assert mismatch.max_abs_error is None


def test_compare_arrays_rejects_complex_inputs() -> None:
    with pytest.raises(TypeError, match="complex-valued arrays"):
        compare_arrays(
            np.array([1.0 + 2.0j]),
            np.array([1.0 + 0.0j]),
        )


def test_vector_space_ordering_serialization_and_validation() -> None:
    space = VectorSpaceSpec(
        name="state",
        components=(
            VectorComponentSpec(label="a", index=0, metadata={"source": "unit"}),
            VectorComponentSpec(label="b", index=1),
        ),
    )
    assert space.labels == ("a", "b")
    assert VectorSpaceSpec.from_dict(space.to_dict()) == space
    np.testing.assert_array_equal(space.validate_vector([1.0, 2.0]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="Duplicate"):
        VectorSpaceSpec(
            name="bad",
            components=(VectorComponentSpec("a"), VectorComponentSpec("a")),
        )
    with pytest.raises(ValueError, match="dimension"):
        space.validate_vector([1.0])


def test_coordinate_transforms_forward_inverse_and_composition() -> None:
    physical = VectorSpaceSpec.from_labels("physical", ["x", "y"])
    scaled = VectorSpaceSpec.from_labels("scaled", ["x", "y"])
    basis = VectorSpaceSpec.from_labels("basis", ["u", "v"])
    diag = DiagonalScaleTransform(
        source_space=physical,
        destination_space=scaled,
        name="scale",
        scales=(2.0, 4.0),
    )
    linear = LinearTransform(
        source_space=scaled,
        destination_space=basis,
        name="rotate",
        matrix=np.array([[0.0, 1.0], [1.0, 0.0]]),
    )
    vector = np.array([4.0, 8.0])
    np.testing.assert_array_equal(diag.forward(vector), np.array([2.0, 2.0]))
    np.testing.assert_array_equal(diag.inverse([2.0, 2.0]), vector)
    np.testing.assert_array_equal(linear.forward([2.0, 3.0]), np.array([3.0, 2.0]))
    np.testing.assert_array_equal(linear.inverse([3.0, 2.0]), np.array([2.0, 3.0]))

    composite = CompositeTransform(
        source_space=physical,
        destination_space=basis,
        name="composite",
        transforms=(diag, linear),
    )
    np.testing.assert_array_equal(composite.forward(vector), np.array([2.0, 2.0]))
    np.testing.assert_array_equal(composite.inverse([2.0, 2.0]), vector)

    with pytest.raises(ValueError, match="scales"):
        DiagonalScaleTransform(
            source_space=physical,
            destination_space=scaled,
            name="bad",
            scales=(1.0,),
        )


def test_grouped_split_is_deterministic_and_keeps_groups_together() -> None:
    records = [
        {"sample_id": "a0", "group": "a"},
        {"sample_id": "a1", "group": "a"},
        {"sample_id": "b0", "group": "b"},
        {"sample_id": "c0", "group": "c"},
        {"sample_id": "d0", "group": "d"},
    ]
    first = assign_grouped_split(
        records,
        group_keys="group",
        fractions={"train": 0.5, "validation": 0.25, "test": 0.25},
        seed=1,
    )
    second = assign_grouped_split(
        records,
        group_keys="group",
        fractions={"train": 0.5, "validation": 0.25, "test": 0.25},
        seed=1,
    )
    different = assign_grouped_split(
        records,
        group_keys="group",
        fractions={"train": 0.5, "validation": 0.25, "test": 0.25},
        seed=2,
    )
    assert first.record_assignments == second.record_assignments
    assert first.record_assignments[0] == first.record_assignments[1]
    assert len(first.group_assignments) == 4
    assert first.policy["rounding"] == "largest_remainder"
    assert first.record_assignments != different.record_assignments


def test_grouped_split_small_group_counts() -> None:
    result = assign_grouped_split(
        [{"g": "only"}],
        group_keys="g",
        fractions={"train": 0.8, "test": 0.2},
        seed=0,
    )
    assert result.record_assignments in (("train",), ("test",))


def test_grouped_split_missing_group_key_errors() -> None:
    with pytest.raises(KeyError, match="missing requested group key"):
        assign_grouped_split(
            [{"group": "a"}, {"bad": "b"}],
            group_keys="group",
            fractions={"train": 1.0},
        )


def test_grouped_split_callable_requires_policy_name_and_records_it() -> None:
    records = [{"a": "x", "b": 1}, {"a": "x", "b": 2}]
    with pytest.raises(ValueError, match="policy_name"):
        assign_grouped_split(
            records,
            group_keys=lambda row: row["a"],
            fractions={"train": 1.0},
        )
    result = assign_grouped_split(
        records,
        group_keys=lambda row: row["a"],
        fractions={"train": 1.0},
        policy_name="by_a",
    )
    assert result.record_assignments == ("train", "train")
    assert result.policy["group_keys"] == "callable:by_a"
    assert result.policy["fraction_basis"] == "groups"
