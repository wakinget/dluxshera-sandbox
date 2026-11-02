#!/usr/bin/env python3
"""Characterize prepared-image count statistics and SHERA noise branch selection.

Read-only.  This opens a prepared dataset's shard memory maps, computes
intensity statistics, and reports which branch of
:func:`dluxshera.utils.noise.apply_observation_noise` each sampled image would
select.  It writes nothing, changes no threshold, and never evaluates a locked
test manifest.

Why this exists
---------------
The SHERA photon-noise path chooses between an exact Poisson draw and a bright
Gaussian approximation using the **minimum** pixel of the whole image::

    if jnp.min(image) > bright_threshold:   # Gaussian approximation
    else:                                   # exact Poisson

With ``bright_threshold=100`` and S12's 50 ms exposures, faint PSF outskirts are
expected to hold the minimum well below 100, so essentially every image should
take the exact Poisson branch.  That is an expectation, not a measured V4
result; this tool measures it.

Detector assumption
-------------------
Prepared-image values are treated as expected counts with a unity conversion,
``1 DN = 1 electron``.  That is S12's accepted modeling assumption.  No gain
calibration or detector conversion stage is introduced here.

Usage
-----
Sampling is strided across the catalog rather than taking a leading block, so a
shard-ordered dataset does not yield a one-shard answer::

    python work/experiments/ml/hpc/analyze_image_statistics.py \
      --prepared-root /path/to/PREP-V4-v1 --samples 512

Add ``--validation-manifest`` to additionally report the frozen validation
records, which is the comparison that matters for the noisy-validation recipe.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

#: S12's configured bright-image threshold.  Reported, never modified.
S12_BRIGHT_THRESHOLD = 100.0

#: Percentiles reported per sampled image set.
PERCENTILES = (0.0, 0.1, 1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 99.9, 100.0)


def _summarize(images: list[np.ndarray], *, bright_threshold: float, label: str) -> dict:
    """Summarize count statistics and branch selection for sampled images.

    Parameters
    ----------
    images:
        Sampled count-space arrays.  Not modified.
    bright_threshold:
        Threshold compared against each image's minimum pixel.
    label:
        Short name for the sampled population.

    Returns
    -------
    dict
        JSON-ready statistics, including the measured Poisson/Gaussian split.
    """
    if not images:
        return {"label": label, "images": 0, "status": "no images sampled"}

    minima = np.array([float(image.min()) for image in images])
    maxima = np.array([float(image.max()) for image in images])
    # Per-image fraction above the threshold, then aggregated, so a few large
    # images cannot dominate the reported fraction.
    above = np.array(
        [float(np.mean(np.asarray(image) > bright_threshold)) for image in images]
    )
    # The actual branch predicate: strictly greater, on the whole-image minimum.
    gaussian = minima > float(bright_threshold)
    stacked = np.concatenate([np.asarray(image, dtype=np.float64).ravel() for image in images])
    shapes = sorted({tuple(int(dim) for dim in image.shape) for image in images})
    dtypes = sorted({str(np.asarray(image).dtype) for image in images})
    return {
        "label": label,
        "images": len(images),
        "image_shapes": [list(shape) for shape in shapes],
        "image_dtypes": dtypes,
        "bright_threshold": float(bright_threshold),
        "branch_predicate": "min(image) > bright_threshold selects the Gaussian approximation",
        "count_units": "1 DN = 1 electron (accepted S12 unity conversion)",
        "per_image_min": {
            "min": float(minima.min()),
            "median": float(np.median(minima)),
            "max": float(minima.max()),
        },
        "per_image_max": {
            "min": float(maxima.min()),
            "median": float(np.median(maxima)),
            "max": float(maxima.max()),
        },
        "pooled_pixel_percentiles": {
            f"p{percentile:g}": float(np.percentile(stacked, percentile))
            for percentile in PERCENTILES
        },
        "pooled_pixel_mean": float(stacked.mean()),
        "pooled_fraction_pixels_above_threshold": float(
            np.mean(stacked > float(bright_threshold))
        ),
        "per_image_fraction_pixels_above_threshold": {
            "min": float(above.min()),
            "median": float(np.median(above)),
            "max": float(above.max()),
        },
        "fraction_images_every_pixel_above_threshold": float(np.mean(gaussian)),
        "images_selecting_gaussian_branch": int(gaussian.sum()),
        "images_selecting_poisson_branch": int((~gaussian).sum()),
        "fraction_selecting_poisson_branch": float(np.mean(~gaussian)),
        "poisson_branch_exclusive": bool(not gaussian.any()),
        "negative_pixels_present": bool(stacked.min() < 0.0),
        "note": (
            "Branch selection is measured from the prescribed predicate on "
            "sampled clean images. No threshold, noise utility, or artifact "
            "was modified."
        ),
    }


def _sample_indices(total: int, requested: int) -> list[int]:
    """Return evenly strided indices so sampling is not shard-local."""
    if requested <= 0 or total <= 0:
        return []
    if requested >= total:
        return list(range(total))
    return [int(round(i * (total - 1) / (requested - 1))) for i in range(requested)] \
        if requested > 1 else [0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument(
        "--samples",
        type=int,
        default=256,
        help="Number of catalog samples to inspect, strided across the catalog.",
    )
    parser.add_argument(
        "--bright-threshold",
        type=float,
        default=S12_BRIGHT_THRESHOLD,
        help="Threshold to report against. Reporting only; nothing is changed.",
    )
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=None,
        help="Optional frozen validation manifest. Its records are summarized "
        "separately so training and validation populations can be compared.",
    )
    parser.add_argument("--split-registry", type=Path, default=None,
                        help="Required with --validation-manifest.")
    parser.add_argument("--shard-cache-size", type=int, default=4)
    args = parser.parse_args(argv)

    from dluxshera.ml import load_sample_catalog

    catalog = load_sample_catalog(args.prepared_root)
    reader = catalog.image_reader(cache_size=int(args.shard_cache_size))
    indices = _sample_indices(int(catalog.sample_count), int(args.samples))
    summary = {
        "prepared_dataset": {
            "path": str(args.prepared_root),
            "artifact_id": catalog.artifact_id,
            "prepared_dataset_hash": catalog.prepared_dataset_hash,
            "sample_count": int(catalog.sample_count),
            "sample_shape": [int(dim) for dim in catalog.sample_shape],
        },
        "sampling": {
            "requested": int(args.samples),
            "inspected": len(indices),
            "strategy": "evenly strided across the catalog index",
        },
        "catalog_samples": _summarize(
            [reader.get(index) for index in indices],
            bright_threshold=float(args.bright_threshold),
            label="catalog (strided sample)",
        ),
    }

    if args.validation_manifest is not None:
        if args.split_registry is None:
            parser.error("--validation-manifest requires --split-registry.")
        from dluxshera.ml import load_pair_manifest, load_split_registry

        manifest = load_pair_manifest(
            args.validation_manifest,
            catalog=catalog,
            split_registry=load_split_registry(args.split_registry),
        )
        records = list(manifest.records)
        picked = _sample_indices(len(records), int(args.samples))
        # Both roles are summarized: image A stays clean in S12, but the
        # observation-side B images are the ones the noise branch sees.
        summary["frozen_validation_observation_images"] = _summarize(
            [reader.get(int(records[i].sample_b_index)) for i in picked],
            bright_threshold=float(args.bright_threshold),
            label="frozen validation, observation (B) role",
        )
        summary["frozen_validation"] = {
            "path": str(args.validation_manifest),
            "artifact_id": manifest.artifact_id,
            "records": len(records),
            "records_inspected": len(picked),
        }

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
