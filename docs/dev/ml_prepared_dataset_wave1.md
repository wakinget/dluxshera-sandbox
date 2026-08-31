# SHERA Prepared Dataset Infrastructure

Status: Wave 1 implementation note

This document describes the reusable dataset infrastructure added for SHERA ML
experimentation.  The higher-level ML roadmap remains
`docs/dev/shera_ml_inverse_model_design.md`.

## Canonical vs prepared data

Generated FITS files and their V3 metadata remain the canonical scientific
dataset.  Prepared datasets are reproducible working artifacts optimized for
local array I/O.  The preparation workflow does not mutate the source dataset
and records source manifest hashes and source-relative paths for provenance.
Source `manifest.json`, `parameter_space.json`, and `samples.jsonl` are hashed.
FITS files are not hash-scanned in Wave 1 to avoid turning preparation into an
extra full-data I/O pass.  Source paths are recorded relative to the prepared
dataset or source root when a correct relative path exists; otherwise the
resolved absolute path is used.

The current prepared layout is:

```text
prepared_dataset/
  manifest.json
  vector_spaces.json
  index.jsonl
  array_shards_manifest.json
  shards/
    shard_00000.npy
    shard_00001.npy
  validation/
    precision_summary.json
    precision_samples.jsonl
  provenance/
    source_manifest.json
    source_parameter_space.json
    source_samples.json
```

`index.jsonl` is sample-centric: one row is one rendered source sample.  It
contains universal array-location fields plus V3 metadata such as family, role,
pair id, nuisance id, physical delta vector, Fisher-scaled delta vector, and
optional nuisance vectors.  V3-specific plan fields such as pair labels, grid
coordinates, sigma offsets, skipped nuisance keys, per-sample seed, sparse
active masks, `theta_sigma`, and source split labels are preserved in the index
when present.

## Array shards

`dluxshera.datasets.ArrayShardStore` writes fixed-shape samples into deterministic
`.npy` shards named `shard_00000.npy`, `shard_00001.npy`, and so on.  The writer
validates that every sample has the same shape, casts to an explicit storage
dtype, writes only one shard-sized buffer at a time, and finalizes each shard via
a temporary file replacement.

Shard sizing is controlled primarily by `target_shard_bytes`, with an optional
`max_samples_per_shard` cap for tests and small smoke runs.  The default target
is 128 MiB; it is a workflow default, not part of the storage contract.

`dluxshera.datasets.ArrayShardReader` random-accesses samples through memory
mapped `.npy` shards.  It uses a bounded LRU cache and closes evicted memmaps so
training or analysis code does not accumulate an unbounded number of open shard
handles.  `reader[index]` and `reader.get(index)` return independent sample
copies by default, so returned arrays remain valid after shard eviction or
`reader.close()`.  `reader.get(index, copy=False)` provides an explicit
short-lived view into the cached memmap for callers that knowingly want
zero-copy access.

The store refuses pre-existing final manifest/index files and refuses non-empty
`shards/` directories.  It removes stale `.tmp` files, writes the JSONL index to
a temporary path, and finalizes the index only after shard writing succeeds.  If
final index/manifest creation fails, the store removes the final index or
manifest from the current invocation plus any shards it created.  Failed writes
do not leave a final index or manifest; callers should remove or overwrite the
output directory explicitly before retrying.

## Dtype and fidelity policy

Source FITS images are commonly `float64`.  Prepared arrays may be stored as
`float32` for ML efficiency or `float64` for lossless working copies.  The
storage dtype is always explicit in `manifest.json`, `array_shards_manifest.json`,
and `index.jsonl`; source dtypes are recorded at shard and row level.

For lossy conversions such as `float64` to `float32`, the V3 preparation
workflow runs deterministic representative precision checks against the actual
prepared shard readback path without loading the complete prepared index into
memory.  It streams the source and prepared indexes in lockstep, retaining only
the requested validation rows, and writes:

- `validation/precision_summary.json`
- `validation/precision_samples.jsonl`

The Wave 1 policy is informational metrics only.  It compares canonical FITS
data to `ArrayShardReader` output and reports max absolute error, RMS error,
relative L2 error, robust pixel-relative metrics, finite counts, sum
differences, readback dtype, and whether readback exactly matches
`source.astype(storage_dtype)` for the validated sample.  It does not invent a
hard scientific acceptance threshold.

## Vector spaces and transforms

`VectorSpaceSpec` and `VectorComponentSpec` describe ordered named vectors
without SHERA-specific assumptions.  Components can record labels, source keys,
component indices, display labels, units, groups, reference values, scale
metadata, and structured metadata.

`DiagonalScaleTransform`, `LinearTransform`, and `CompositeTransform` provide
minimal coordinate transforms with explicit source/destination spaces and
dimension validation.  The V3 adapter uses a diagonal transform to build
Fisher-scaled deltas:

```text
z_i = delta_i / parameter_sigma_i
```

Future eigenbasis or whitening work should build on these generic transforms
without changing the prepared array storage contract.

Registration nuisance metadata has two optional vector spaces:

- `shera_v3_registration_nuisance`: physical registration offsets from
  `registration_nuisance_values`.
- `shera_v3_registration_nuisance_sigma`: V3 generator sigma-coordinate nuisance
  draws from `registration_nuisance_sigma_values`.

These sigma coordinates are normalized nuisance draw offsets, not an eigenbasis
or whitening transform.

## Grouped splitting

`assign_grouped_split` assigns records to named partitions while keeping all rows
with the same group id in the same partition.  Missing named group fields raise
by default; callers can explicitly opt into missing-as-`None` compatibility.
Callable group policies require a stable `policy_name` so provenance is
meaningful.  Fractions apply to the number of groups, not necessarily the number
of individual records when groups have unequal sizes.  The helper normalizes
requested fractions, orders groups by a stable seed-derived hash, and rounds
group counts using largest remainders.

This helper does not define the authoritative SHERA train/validation/test policy.
It provides the deterministic mechanism needed to express future policies without
leakage across physical states, nuisance replicates, Sobol blocks, repeated
seeds, or other related families.

## V3 preparation

Use the thin CLI wrapper:

```bash
python3 examples/scripts/prepare_ml_dataset.py \
  --source-root Results/path/to/v3_dataset \
  --outdir Results/path/to/prepared_dataset \
  --dtype float32 \
  --target-shard-bytes 134217728 \
  --max-samples 1024 \
  --validation-samples 32 \
  --seed 0
```

`--max-samples N` prepares the first `N` source rows as an explicit prefix smoke
selection.  The prepared manifest records both total source sample count and
prepared sample count, with selection policy `prefix`.

Add `--dry-run` to inspect source row count, selected prepared count,
first-sample shape/dtype probe, intended storage dtype, expected shard count,
and validation sample count without writing outputs.  Dry-run streams
`samples.jsonl` for metadata validation and opens only the first selected FITS
file, so shape/dtype are marked as a provisional first-sample probe rather than
all-file validation.  Existing non-empty output directories are refused unless
`--overwrite` is provided.

The adapter consumes `manifest.json`, `parameter_space.json`, and `samples.jsonl`.
It reads source FITS paths from sample metadata, not filename inference.  Missing
optional V3 fields are preserved as absent/null metadata rather than promoted to
generic requirements.

Before touching FITS, the adapter makes one streaming pass over the full
`samples.jsonl` to validate row count, duplicate sample ids, sample index
ordering, nuisance label coverage, and `theta_delta` keys against
`parameter_space.json`.  It retains only bounded summary/provenance state and
the first selected row needed for the FITS probe; selected source rows are
streamed again for array writing and metadata-index generation.  `--max-samples
N` keeps deterministic first-`N` prefix semantics while still validating the
full source metadata.  When source manifest fields are present,
`rendered_sample_count` must match the JSONL row count, `next_sample_index` must
match `rendered_sample_count`, and `render_complete=false` is refused unless
`--allow-incomplete-source` is provided.  Opt-in incomplete-source preparation is
recorded prominently in source provenance.

## Parquet decision

Wave 1 does not require Parquet and adds no `pyarrow`, `zarr`, or `h5py`
dependency.  JSONL is used for streamable row metadata and preserves numeric
vectors as JSON arrays.  Optional columnar export can be added later behind an
extra dependency if training throughput demands it.

## Deferred work

Wave 1 intentionally does not include PyTorch/JAX dataset classes, dynamic pair
sampling, Siamese or JEPA models, W&B integration, V4/Sobol generation,
eigenvalue-weighted losses, noise augmentation, or ADORA inference wiring.
