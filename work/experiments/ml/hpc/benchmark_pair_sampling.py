#!/usr/bin/env python3
"""Isolate pair-sampling cost, before and after the validation cache.

Gattaca2 profiling of S12-E01 at ``22e525f`` measured 9.895 of 12.064 seconds
inside ``PairSampler._validate_policy_against_catalog``, almost entirely in the
catalog-family set comprehension.  That validation ran once per dynamic
training example even though the catalog-family names never change during
sampling.

This benchmark measures the validation call directly and ``sample_pair``
end-to-end, at a configurable catalog row count so the result can be read at
realistic V3/V4 dimensions rather than at unit-test fixture size.  The
"before" figure is reproduced by re-running the original per-call scan against
the same catalog columns, so both numbers come from one process and one
machine state.

It writes nothing and reads no production artifacts unless ``--prepared-root``
is given, in which case the real catalog's columns are used.

Run from the repository root::

    PYTHONPATH=src:. python work/experiments/ml/hpc/benchmark_pair_sampling.py
    PYTHONPATH=src:. python work/experiments/ml/hpc/benchmark_pair_sampling.py \\
      --prepared-root "Results/ML Training Datasets/preprocessed/PREP-V3-nominal-v1"
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

#: S12's prescribed ordered pairs per epoch and epoch count, used only to
#: project the measured per-call cost onto a campaign.
S12_PAIRS_PER_EPOCH = 16384
S12_EPOCHS = 600


def _time(fn, *, repeats: int) -> float:
    """Return warm mean seconds per call."""
    fn()
    started = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - started) / repeats


def _baseline_validate(families: np.ndarray, pair_ids: np.ndarray, policy_families: set[str]):
    """Reproduce the pre-optimization per-call scan.

    Copied deliberately rather than imported: the point is to measure the code
    that was replaced, against the same columns, in the same process.
    """
    def run() -> None:
        catalog_families = {str(v) for v in families}
        unknown = sorted(policy_families - catalog_families)
        if unknown:
            raise ValueError(f"unknown {unknown}; available={sorted(catalog_families)}")
        if not any(str(v) for v in pair_ids):
            raise ValueError("no pair ids")

    return run


def _synthetic_columns(rows: int, *, families: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Build catalog-shaped family and pair-id columns at ``rows`` length."""
    names = [f"family_{index}" for index in range(max(int(families), 1))]
    family_column = np.asarray(
        [names[index % len(names)] for index in range(int(rows))], dtype=object
    )
    pair_column = np.asarray([f"pair_{index}" for index in range(int(rows))], dtype=object)
    return family_column, pair_column


def benchmark_validation(rows: int, *, repeats: int, families: int) -> dict:
    """Measure validation only: cached lookup versus the original scan."""
    family_column, pair_column = _synthetic_columns(rows, families=families)
    policy_families = {str(family_column[0])}

    cached_families = frozenset(str(v) for v in family_column)
    cached_has_pair_ids = any(str(v) for v in pair_column)

    def optimized() -> None:
        unknown = sorted(policy_families - cached_families)
        if unknown:
            raise ValueError("unknown")
        if not cached_has_pair_ids:
            raise ValueError("no pair ids")

    before = _time(_baseline_validate(family_column, pair_column, policy_families),
                   repeats=repeats)
    after = _time(optimized, repeats=repeats)
    return {
        "catalog_rows": int(rows),
        "distinct_families": int(families),
        "before_seconds_per_call": before,
        "after_seconds_per_call": after,
        "speedup": (before / after) if after > 0 else None,
        "before_seconds_per_epoch": before * S12_PAIRS_PER_EPOCH,
        "after_seconds_per_epoch": after * S12_PAIRS_PER_EPOCH,
        "epoch_seconds_saved": (before - after) * S12_PAIRS_PER_EPOCH,
        "campaign_hours_saved": (before - after) * S12_PAIRS_PER_EPOCH * S12_EPOCHS / 3600.0,
        "note": (
            "Validation only. The campaign projection is CPU data-loading time "
            "that overlaps GPU compute; it is not a wall-clock guarantee."
        ),
    }


def benchmark_sample_pair(*, repeats: int, prepared_root: str | None) -> dict:
    """Measure end-to-end ``sample_pair`` on a real or fixture catalog."""
    from dluxshera.ml import (
        PairPolicy,
        PairSampler,
        generate_split_registry,
        load_sample_catalog,
    )

    if prepared_root:
        catalog = load_sample_catalog(prepared_root)
        source = str(prepared_root)
    else:
        import tempfile
        from pathlib import Path

        from tests.ml.test_catalog_splits_pairs import _write_prepared_fixture

        temporary = tempfile.TemporaryDirectory()
        catalog = load_sample_catalog(
            _write_prepared_fixture(Path(temporary.name) / "prepared", image_size=8)
        )
        source = "synthetic test fixture"

    registry = generate_split_registry(
        catalog,
        seed=3,
        science_fractions={"train": 1.0},
        nuisance_fractions={"train": 1.0},
    )
    policy = PairPolicy(
        family_weights={"same_nuisance_different_science": 1.0},
        min_fisher_distance=0.0,
        max_fisher_distance=5000.0,
        include_reverse=True,
        max_sampling_attempts=4000,
    )
    sampler = PairSampler(catalog, registry, policy)

    counter = {"index": 0}

    def draw() -> None:
        counter["index"] += 1
        sampler.sample_pair(
            np.random.default_rng(counter["index"]),
            science_split="train",
            nuisance_split="train",
            split="train",
            eval_slice=None,
            epoch=0,
        )

    seconds = _time(draw, repeats=repeats)
    validation = _time(
        lambda: sampler._validate_policy_against_catalog(sampler.policy),
        repeats=max(repeats * 10, 1000),
    )
    return {
        "catalog_source": source,
        "catalog_rows": int(len(catalog.sample_ids)),
        "sample_pair_seconds": seconds,
        "pairs_per_second": (1.0 / seconds) if seconds > 0 else None,
        "validation_seconds_per_call": validation,
        "validation_share_of_sample_pair": (validation / seconds) if seconds > 0 else None,
        "projected_epoch_seconds": seconds * S12_PAIRS_PER_EPOCH,
        "note": (
            "sample_pair includes family choice, bucket lookup, candidate "
            "rejection and record construction. Shard reads, noise and scaling "
            "are not included."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=[1000, 22990, 100000],
        help="Catalog row counts for the validation benchmark. The default "
        "middle value is the real PREP-V3-nominal-v1 size.",
    )
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument(
        "--distinct-families",
        type=int,
        default=1,
        help="Distinct dataset-family names. V3/V4 use one.",
    )
    parser.add_argument(
        "--prepared-root",
        default=None,
        help="Optional real prepared dataset for the end-to-end measurement. "
        "Read-only. Omit to use the synthetic test fixture.",
    )
    parser.add_argument(
        "--skip-sample-pair",
        action="store_true",
        default=False,
        help="Measure validation only; skip the end-to-end sampling benchmark.",
    )
    args = parser.parse_args(argv)

    payload: dict = {
        "validation": [
            benchmark_validation(rows, repeats=args.repeats, families=args.distinct_families)
            for rows in args.rows
        ]
    }
    if not args.skip_sample_pair:
        payload["sample_pair"] = benchmark_sample_pair(
            repeats=max(args.repeats // 4, 10), prepared_root=args.prepared_root
        )

    print(f"{'rows':>9} {'before us':>11} {'after us':>10} {'speedup':>9} "
          f"{'s/epoch saved':>14} {'h/campaign':>11}")
    for row in payload["validation"]:
        print(f"{row['catalog_rows']:9d} "
              f"{row['before_seconds_per_call'] * 1e6:11.2f} "
              f"{row['after_seconds_per_call'] * 1e6:10.3f} "
              f"{(row['speedup'] or 0):8.1f}x "
              f"{row['epoch_seconds_saved']:14.2f} "
              f"{row['campaign_hours_saved']:11.2f}")

    if "sample_pair" in payload:
        end_to_end = payload["sample_pair"]
        print(f"\nsample_pair on {end_to_end['catalog_source']} "
              f"({end_to_end['catalog_rows']} rows):")
        print(f"  {end_to_end['sample_pair_seconds'] * 1e6:.1f} us/pair "
              f"({(end_to_end['pairs_per_second'] or 0):.0f} pairs/s)")
        print(f"  validation now {end_to_end['validation_seconds_per_call'] * 1e6:.3f} us/call "
              f"= {(end_to_end['validation_share_of_sample_pair'] or 0) * 100:.2f}% of sample_pair")
        print(f"  projected {end_to_end['projected_epoch_seconds']:.1f} s per "
              f"{S12_PAIRS_PER_EPOCH}-pair epoch (sampling only)")

    print("\n" + json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
