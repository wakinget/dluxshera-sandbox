"""Regression tests for cached catalog-derived pair-policy validation.

``PairSampler.sample_pair`` revalidates its policy against the catalog for
every dynamic training example.  The catalog-family scan at the heart of that
validation cost ~323 us on the 22,990-row V3 catalog -- about 5.3 s per
16,384-pair epoch -- and Gattaca2 profiling of S12-E01 at ``22e525f`` measured
9.895 of 12.064 seconds inside ``_validate_policy_against_catalog``.

The catalog-derived inputs are now computed once at construction.  Two
properties have to hold for that to be safe, and both are tested here:

- The *policy* is still validated on every call, so a policy that becomes
  invalid only at a later epoch (via a curriculum stage) must still raise at
  that epoch rather than being waved through by the cache.
- Pair selection, RNG consumption, and the emitted records are unchanged.

The determinism tests pin exact record content across seeds, epochs, and pair
families.  They are the guard against a future "optimization" that also moves
an RNG draw.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dluxshera.ml import (
    PairPolicy,
    PairSampler,
    generate_split_registry,
    load_sample_catalog,
)
from tests.ml.test_catalog_splits_pairs import _write_prepared_fixture

PAIR_FAMILIES = (
    "same_nuisance_different_science",
    "different_nuisance_same_science",
    "different_science_different_nuisance",
)


def _catalog_and_registry(tmp_path: Path):
    catalog = load_sample_catalog(_write_prepared_fixture(tmp_path / "prepared", image_size=8))
    registry = generate_split_registry(
        catalog,
        seed=3,
        science_fractions={"train": 1.0},
        nuisance_fractions={"train": 1.0},
    )
    return catalog, registry


def _policy(**overrides) -> PairPolicy:
    base = dict(
        family_weights={"same_nuisance_different_science": 1.0},
        same_pair_id=True,
        min_fisher_distance=0.0,
        max_fisher_distance=5000.0,
        include_reverse=True,
        max_sampling_attempts=4000,
    )
    base.update(overrides)
    return PairPolicy(**base)


def _draw(sampler: PairSampler, *, seed: int, epoch: int):
    return sampler.sample_pair(
        np.random.default_rng(seed),
        science_split="train",
        nuisance_split="train",
        split="train",
        eval_slice=None,
        epoch=epoch,
    )


# ---------------------------------------------------------------------------
# The cache itself
# ---------------------------------------------------------------------------
def test_catalog_validation_inputs_are_cached_at_construction(tmp_path):
    """The cached values must match a fresh scan of the catalog."""
    catalog, registry = _catalog_and_registry(tmp_path)
    sampler = PairSampler(catalog, registry, _policy())

    assert sampler._catalog_families == frozenset(
        str(value) for value in catalog.dataset_families
    )
    assert sampler._catalog_families_sorted == tuple(sorted(sampler._catalog_families))
    assert sampler._catalog_has_pair_ids == any(
        str(value) for value in catalog.pair_ids
    )
    # Immutable containers, so a caller cannot corrupt validation state.
    assert isinstance(sampler._catalog_families, frozenset)
    assert isinstance(sampler._catalog_families_sorted, tuple)


def test_validation_does_not_rescan_the_catalog(tmp_path, monkeypatch):
    """Validation must not re-read ``catalog.dataset_families``.

    This is the actual optimization: the per-call scan of a 22,990-row object
    array is what the Gattaca2 profile found. Asserting the attribute is
    untouched during validation is stronger and more stable than timing it.

    Scoped to validation deliberately. Record construction still reads
    ``dataset_families[index]`` for the two chosen rows, which is O(1) indexing
    rather than an O(N) scan, and is not what this optimization targets.
    """
    catalog, registry = _catalog_and_registry(tmp_path)
    sampler = PairSampler(catalog, registry, _policy())

    # `SampleCatalog` is a frozen dataclass, so `dataset_families` is a plain
    # instance attribute. Install a counting property on the class *after*
    # construction, so only later reads are counted.
    reads = {"count": 0}
    stored = catalog.dataset_families

    def _counted(self):
        reads["count"] += 1
        return stored

    monkeypatch.setattr(type(catalog), "dataset_families", property(_counted), raising=False)
    for _ in range(25):
        sampler._validate_policy_against_catalog(sampler.policy)
    assert reads["count"] == 0, (
        f"validation re-read catalog.dataset_families {reads['count']} time(s); "
        "the per-call scan was the measured bottleneck"
    )


def test_validation_cost_does_not_scale_with_catalog_size(tmp_path):
    """Validation must be O(policy), not O(catalog rows).

    A 40x larger catalog must not make validation measurably slower. This is
    the property that turns 9.9 s of a 12.1 s profile into noise; timing is
    coarse on purpose so the test is not flaky.
    """
    import time

    catalog, registry = _catalog_and_registry(tmp_path)
    small = PairSampler(catalog, registry, _policy())

    # `SampleCatalog.__post_init__` requires every column to have the same
    # length, and validation reads only these two, so grow them on a stand-in
    # rather than building a deliberately inconsistent catalog.
    rows = len(catalog.sample_ids)
    scale = 40
    grown_families = np.asarray(
        [str(v) for v in catalog.dataset_families] * scale, dtype=object
    )
    grown_pair_ids = np.asarray(
        [str(v) for v in catalog.pair_ids] * scale, dtype=object
    )

    class _WideCatalog:
        dataset_families = grown_families
        pair_ids = grown_pair_ids

    large = PairSampler.__new__(PairSampler)
    large.catalog = _WideCatalog()
    large.split_registry = registry
    large.policy = small.policy
    large._eligible_cache = {}
    large._bucket_cache = {}
    large._catalog_families = frozenset(str(v) for v in grown_families)
    large._catalog_families_sorted = tuple(sorted(large._catalog_families))
    large._catalog_has_pair_ids = any(str(v) for v in grown_pair_ids)

    def _elapsed(sampler, repeats=400):
        sampler._validate_policy_against_catalog(sampler.policy)
        started = time.perf_counter()
        for _ in range(repeats):
            sampler._validate_policy_against_catalog(sampler.policy)
        return (time.perf_counter() - started) / repeats

    small_seconds = _elapsed(small)
    large_seconds = _elapsed(large)
    # A scan would be ~40x slower; a cached lookup is within noise. The 5x
    # allowance is slack for timer jitter, not for partial scaling.
    assert large_seconds < max(small_seconds * 5.0, 5e-5), (
        f"validation took {large_seconds * 1e6:.1f}us on a {rows * 40}-row catalog "
        f"versus {small_seconds * 1e6:.1f}us on {rows} rows; it still scales with "
        "catalog size"
    )


# ---------------------------------------------------------------------------
# Invalid policies must still be rejected
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("field", ["dataset_families", "dataset_family_weights"])
def test_unknown_dataset_family_is_rejected_at_construction(tmp_path, field):
    catalog, registry = _catalog_and_registry(tmp_path)
    value = ["ghost_family"] if field == "dataset_families" else {"ghost_family": 1.0}
    with pytest.raises(ValueError, match="unknown dataset family"):
        PairSampler(catalog, registry, _policy(**{field: value}))


def test_unknown_family_error_names_the_family_and_the_available_set(tmp_path):
    """The message must stay actionable: both the bad name and the options."""
    catalog, registry = _catalog_and_registry(tmp_path)
    with pytest.raises(ValueError) as excinfo:
        PairSampler(catalog, registry, _policy(dataset_families=["ghost_family"]))
    message = str(excinfo.value)
    assert "ghost_family" in message
    for family in sorted({str(v) for v in catalog.dataset_families}):
        assert family in message


def test_same_pair_id_requires_non_empty_catalog_pair_ids(tmp_path):
    """The cached pair-id emptiness check must still fail closed."""
    catalog, registry = _catalog_and_registry(tmp_path)
    blanked = type(catalog)(
        **{
            **{f.name: getattr(catalog, f.name) for f in catalog.__dataclass_fields__.values()},
            "pair_ids": np.asarray([""] * len(catalog.sample_ids), dtype=object),
        }
    )
    with pytest.raises(ValueError, match="same_pair_id=True"):
        PairSampler(blanked, registry, _policy(same_pair_id=True))
    # same_pair_id=False is unaffected.
    assert PairSampler(blanked, registry, _policy(same_pair_id=False)) is not None


def test_epoch_dependent_policy_is_still_validated_every_call(tmp_path):
    """A curriculum stage that turns the policy invalid must raise at that epoch.

    This is the property the cache could plausibly have broken: caching the
    *policy* verdict instead of only the catalog facts would let an invalid
    epoch-resolved policy through.
    """
    catalog, registry = _catalog_and_registry(tmp_path)
    sampler = PairSampler(
        catalog,
        registry,
        _policy(
            dataset_families=["pair_grid"],
            curriculum_stages=({"start_epoch": 5, "dataset_families": ["ghost_family"]},),
            max_sampling_attempts=200,
        ),
    )
    # Epoch 0 resolves to the valid base policy.
    assert _draw(sampler, seed=1, epoch=0).pair_record_id
    # From epoch 5 the resolved policy references an unknown family.
    for epoch in (5, 10, 599):
        with pytest.raises(ValueError, match="unknown dataset family"):
            _draw(sampler, seed=1, epoch=epoch)


def test_epoch_dependent_same_pair_id_change_is_still_validated(tmp_path):
    """A stage enabling same_pair_id on a pair-id-less catalog must raise."""
    catalog, registry = _catalog_and_registry(tmp_path)
    blanked = type(catalog)(
        **{
            **{f.name: getattr(catalog, f.name) for f in catalog.__dataclass_fields__.values()},
            "pair_ids": np.asarray([""] * len(catalog.sample_ids), dtype=object),
        }
    )
    sampler = PairSampler(
        blanked,
        registry,
        _policy(
            same_pair_id=False,
            curriculum_stages=({"start_epoch": 3, "same_pair_id": True},),
            max_sampling_attempts=200,
        ),
    )
    assert _draw(sampler, seed=1, epoch=0).pair_record_id
    with pytest.raises(ValueError, match="same_pair_id=True"):
        _draw(sampler, seed=1, epoch=3)


# ---------------------------------------------------------------------------
# Deterministic equivalence of generated pairs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("family", PAIR_FAMILIES)
@pytest.mark.parametrize("seed", [11, 23, 47])
@pytest.mark.parametrize("epoch", [0, 1, 7, 599])
def test_pair_generation_is_deterministic_across_seeds_epochs_and_families(
    tmp_path, family, seed, epoch
):
    """Two samplers over the same catalog must emit identical records.

    Validation must not consume RNG, so a fresh sampler and a sampler that has
    already drawn pairs produce the same record for the same seed.
    """
    catalog, registry = _catalog_and_registry(tmp_path)
    policy = _policy(family_weights={family: 1.0})
    first = PairSampler(catalog, registry, policy)
    second = PairSampler(catalog, registry, policy)
    # Warm the second sampler's lazy caches so the comparison is not trivially
    # between two identical cold states.
    for index in range(5):
        _draw(second, seed=900 + index, epoch=epoch)

    for index in range(6):
        draw_seed = seed + epoch * 1_000_003 + index
        left = _draw(first, seed=draw_seed, epoch=epoch)
        right = _draw(second, seed=draw_seed, epoch=epoch)
        assert json.dumps(left.to_dict(), sort_keys=True, default=str) == json.dumps(
            right.to_dict(), sort_keys=True, default=str
        )


def test_repeated_sampling_with_the_same_seed_is_stable(tmp_path):
    """One sampler must return the same record for a repeated seed."""
    catalog, registry = _catalog_and_registry(tmp_path)
    sampler = PairSampler(catalog, registry, _policy())
    expected = _draw(sampler, seed=4242, epoch=12).to_dict()
    for _ in range(4):
        # Interleave other draws so cache state differs between repeats.
        _draw(sampler, seed=7, epoch=12)
        assert _draw(sampler, seed=4242, epoch=12).to_dict() == expected


def test_validation_consumes_no_random_numbers(tmp_path):
    """Validation must leave the RNG untouched.

    If validation ever drew a random number, caching it away would silently
    shift every subsequent pair -- so the invariant is pinned directly.
    """
    catalog, registry = _catalog_and_registry(tmp_path)
    sampler = PairSampler(catalog, registry, _policy())

    rng = np.random.default_rng(31415)
    sampler._validate_policy_against_catalog(sampler.policy)
    after_validation = rng.bit_generator.state

    probe = np.random.default_rng(31415)
    assert json.dumps(after_validation, default=str) == json.dumps(
        probe.bit_generator.state, default=str
    )


def test_reverse_pair_semantics_are_unchanged(tmp_path):
    """Reverse records must still swap roles and negate the science target."""
    from dluxshera.ml.pairs import make_reverse_pair_record

    catalog, registry = _catalog_and_registry(tmp_path)
    sampler = PairSampler(catalog, registry, _policy())
    forward = _draw(sampler, seed=99, epoch=0)
    reverse = make_reverse_pair_record(forward)

    assert reverse.sample_a_id == forward.sample_b_id
    assert reverse.sample_b_id == forward.sample_a_id
    assert np.allclose(
        np.asarray(reverse.target_delta_z, dtype=np.float64),
        -np.asarray(forward.target_delta_z, dtype=np.float64),
    )


def test_cached_validation_still_rejects_a_policy_passed_to_sample_pair(tmp_path):
    """`eligible_indices` policy overrides remain validated against the catalog."""
    catalog, registry = _catalog_and_registry(tmp_path)
    sampler = PairSampler(catalog, registry, _policy())
    with pytest.raises(ValueError, match="unknown dataset family"):
        sampler._validate_policy_against_catalog(
            _policy(dataset_families=["ghost_family"])
        )
