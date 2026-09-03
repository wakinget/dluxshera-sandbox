from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dluxshera.ml import NoiseConfig, PairPolicy, PairSampler, generate_split_registry, load_sample_catalog
from dluxshera.ml.torch_data import DynamicPairDataset
from tests.ml.test_catalog_splits_pairs import _write_prepared_fixture


def _catalog_registry_sampler(tmp_path: Path, *, include_reverse: bool) -> tuple:
    catalog = load_sample_catalog(_write_prepared_fixture(tmp_path / "prepared"))
    registry = generate_split_registry(
        catalog,
        seed=3,
        science_fractions={"train": 1.0},
        nuisance_fractions={"train": 1.0},
    )
    policy = PairPolicy(
        family_weights={"same_nuisance_different_science": 1.0},
        same_pair_id=True,
        min_fisher_distance=0.5,
        max_fisher_distance=3.0,
        include_reverse=include_reverse,
        max_sampling_attempts=4000,
    )
    return catalog, registry, PairSampler(catalog, registry, policy)


def _dataset(
    tmp_path: Path,
    *,
    include_reverse: bool,
    pairs_per_epoch: int,
    noise_config: NoiseConfig | None = None,
) -> DynamicPairDataset:
    catalog, _, sampler = _catalog_registry_sampler(tmp_path, include_reverse=include_reverse)
    return DynamicPairDataset(
        catalog=catalog,
        sampler=sampler,
        pairs_per_epoch=pairs_per_epoch,
        seed=101,
        noise_config=NoiseConfig(enabled=False) if noise_config is None else noise_config,
    )


def _pair_ids(item: dict) -> tuple[str, str]:
    return str(item["sample_a_id"]), str(item["sample_b_id"])


def _loader_pair_ids(dataset: DynamicPairDataset, *, epoch: int, num_workers: int) -> list[tuple[str, str]]:
    dataset.set_epoch(epoch)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=False,
    )
    pairs: list[tuple[str, str]] = []
    for batch in loader:
        pairs.extend(zip((str(v) for v in batch["sample_a_id"]), (str(v) for v in batch["sample_b_id"])))
    return pairs


def test_dynamic_pairs_reverse_disabled_keeps_ordered_length_and_odd_lengths(tmp_path: Path) -> None:
    catalog, _, sampler = _catalog_registry_sampler(tmp_path, include_reverse=False)
    dataset = DynamicPairDataset(
        catalog=catalog,
        sampler=sampler,
        pairs_per_epoch=7,
        seed=101,
        noise_config=NoiseConfig(enabled=False),
    )
    assert len(dataset) == 7

    expected0 = sampler.sample_pair(np.random.default_rng(101))
    expected1 = sampler.sample_pair(np.random.default_rng(102))
    assert _pair_ids(dataset[0]) == (expected0.sample_a_id, expected0.sample_b_id)
    assert _pair_ids(dataset[1]) == (expected1.sample_a_id, expected1.sample_b_id)


def test_dynamic_pairs_reverse_enabled_requires_even_ordered_length(tmp_path: Path) -> None:
    catalog, _, sampler = _catalog_registry_sampler(tmp_path, include_reverse=True)
    with pytest.raises(ValueError, match="include_reverse=True requires pairs_per_epoch to be even"):
        DynamicPairDataset(catalog=catalog, sampler=sampler, pairs_per_epoch=7, seed=101)


def test_dynamic_pairs_reverse_enabled_adjacent_records_are_opposites(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, include_reverse=True, pairs_per_epoch=8)
    assert len(dataset) == 8

    for index in range(0, len(dataset), 2):
        forward = dataset[index]
        reverse = dataset[index + 1]
        assert _pair_ids(reverse) == tuple(reversed(_pair_ids(forward)))
        np.testing.assert_allclose(
            reverse["target_delta_z"].numpy(),
            -forward["target_delta_z"].numpy(),
        )
        np.testing.assert_allclose(
            reverse["target_delta_theta"].numpy(),
            -forward["target_delta_theta"].numpy(),
        )
        np.testing.assert_allclose(
            reverse["nuisance_delta"].numpy(),
            -forward["nuisance_delta"].numpy(),
        )
        assert float(reverse["fisher_distance_l2"]) == float(forward["fisher_distance_l2"])
        assert int(reverse["changed_science_dimensions"]) == int(
            forward["changed_science_dimensions"]
        )


def test_dynamic_pairs_reverse_enabled_is_deterministic_by_base_pair_index(
    tmp_path: Path,
) -> None:
    catalog, _, sampler = _catalog_registry_sampler(tmp_path, include_reverse=True)
    dataset = DynamicPairDataset(
        catalog=catalog,
        sampler=sampler,
        pairs_per_epoch=8,
        seed=101,
        noise_config=NoiseConfig(enabled=False),
    )
    dataset.set_epoch(2)

    first = dataset[2]
    again = dataset[2]
    reverse = dataset[3]
    assert _pair_ids(first) == _pair_ids(again)
    assert str(first["pair_record_id"]) == str(again["pair_record_id"])
    assert _pair_ids(reverse) == tuple(reversed(_pair_ids(first)))

    expected_base = sampler.sample_pair(
        np.random.default_rng(101 + 2 * 1_000_003 + 1),
        science_split="train",
        nuisance_split="train",
        split="train",
        eval_slice=None,
    )
    assert _pair_ids(first) == (expected_base.sample_a_id, expected_base.sample_b_id)

    epoch2_index0 = _pair_ids(dataset[0])
    dataset.set_epoch(3)
    epoch3 = _pair_ids(dataset[0])
    expected_epoch3 = sampler.sample_pair(
        np.random.default_rng(101 + 3 * 1_000_003),
        science_split="train",
        nuisance_split="train",
        split="train",
        eval_slice=None,
    )
    assert epoch3 == (expected_epoch3.sample_a_id, expected_epoch3.sample_b_id)
    assert epoch3 == _pair_ids(dataset[0])
    assert epoch3 != epoch2_index0


def test_dynamic_pairs_reverse_enabled_swaps_images_when_noise_disabled(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, include_reverse=True, pairs_per_epoch=2)
    forward = dataset[0]
    reverse = dataset[1]
    torch.testing.assert_close(reverse["image_a"], forward["image_b"])
    torch.testing.assert_close(reverse["image_b"], forward["image_a"])


def test_observation_noise_remains_attached_to_b_role_under_reversal(tmp_path: Path) -> None:
    clean = _dataset(tmp_path / "clean", include_reverse=True, pairs_per_epoch=2)
    noisy = _dataset(
        tmp_path / "noisy",
        include_reverse=True,
        pairs_per_epoch=2,
        noise_config=NoiseConfig(
            enabled=True,
            apply_to="observation",
            photon_noise=False,
            read_noise=True,
            read_noise_sigma=0.25,
            seed=13,
            training_dynamic=True,
        ),
    )

    clean_forward = clean[0]
    clean_reverse = clean[1]
    noisy_forward = noisy[0]
    noisy_reverse = noisy[1]

    torch.testing.assert_close(noisy_forward["image_a"], clean_forward["image_a"])
    torch.testing.assert_close(noisy_reverse["image_a"], clean_reverse["image_a"])
    assert not torch.equal(noisy_forward["image_b"], clean_forward["image_b"])
    assert not torch.equal(noisy_reverse["image_b"], clean_reverse["image_b"])
    assert not torch.equal(noisy_forward["image_b"], noisy_reverse["image_a"])
    assert not torch.equal(noisy_reverse["image_b"], noisy_forward["image_a"])


def test_multi_worker_dynamic_pair_stream_observes_parent_epoch_state(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, include_reverse=True, pairs_per_epoch=8)
    epoch0_first = _loader_pair_ids(dataset, epoch=0, num_workers=2)
    epoch0_again = _loader_pair_ids(dataset, epoch=0, num_workers=2)
    epoch1 = _loader_pair_ids(dataset, epoch=1, num_workers=2)

    assert epoch0_first == epoch0_again
    assert epoch1 != epoch0_first
    for index in range(0, len(epoch0_first), 2):
        assert epoch0_first[index + 1] == tuple(reversed(epoch0_first[index]))
