# ML training dataset generator V3

`work/experiments/generate_training_dataset_v3.py` is the first plan-first V3
training-dataset generator for dLuxShera.  It keeps the V2 prescription pattern
(`system` plus `experiment`) and reuses V2-style Fisher-sigma sweep ranges, but
expands the dataset plan beyond one-parameter-at-a-time samples.

## How V3 differs from V2

- **V2** builds a one-parameter sensitivity atlas: one nominal image followed by
  mirrored nonzero offsets for each scalar parameter/component.
- **V3** builds explicit plan artifacts before rendering:
  - `pair_grid`: all unordered 2D grids over the scalarized parameter space,
  - optional registration-nuisance replicates for pair-grid samples,
  - `sparse_mixture`: a separate sparse random mixture dataset for held-out
    evaluation.
- Dry runs write `manifest.json`, `parameter_space.json`, `pair_plan.csv`,
  `sparse_mixture_plan.csv`, and quicklook summaries without writing FITS
  images.

## Quick usage

From the repository root:

```bash
python work/experiments/generate_training_dataset_v3.py \
  --prescription work/experiments/ml_dataset_v3_template.yaml \
  --dry-run
```

Render only the first two planned samples for a smoke test:

```bash
python work/experiments/generate_training_dataset_v3.py \
  --prescription work/experiments/ml_dataset_v3_template.yaml \
  --run-name ml_training_v3_smoke \
  --max-samples 2
```

The CLI also supports `--system-preset`, `--experiment-preset`, `--outdir`, and
`--run-name` overrides.

## Reusing V2 sweeps

The V3 template intentionally keeps the V2 fields:

- `experiment.sweep_keys` defines the eligible base parameter keys.
- Vector-valued keys such as `optics.primary.zernike_coeffs_nm` are scalarized
  into labels like `optics.primary.zernike_coeffs_nm[0]`.
- `experiment.sweeps` remains the canonical source for Fisher-sigma amplitude
  ranges.  For each scalarized label, V3 uses the base-key sweep entry (falling
  back to `sweeps.default`) and maps sigma offsets to physical deltas via the
  Fisher diagonal.

For pair grids, the default V3 template uses
`experiment.datasets.pair_grid.level_mode: symmetric_grid_from_sweeps` with
`grid_size: 11`.  In this mode, the V2 sweep range controls the maximum absolute
sigma amplitude and `grid_size` controls the 2D grid density.  The preserved
`n_magnitudes` values remain visible in the prescription and manifest for
provenance and backward compatibility.

## Registration nuisance replicates

The `experiment.datasets.nuisance_replicates` block controls optional X/Y/PA
registration context offsets for pair-grid samples.  These are recorded as
`registration_nuisance_values`, not as observation noise.  With the default
`collision_policy: skip_if_key_is_controlled_axis`, V3 does not add a hidden
registration offset to a key that is already one of the pair-grid controlled
axes.

## Sparse mixture plan

The `experiment.datasets.sparse_mixture` block creates a separate plan with
random active scalar labels, active masks, nominal/delta/applied theta metadata,
registration context values, and deterministic per-sample seeds.  It is intended
for held-out testing rather than for pair-grid training coverage.

## PSF/image size override

The current system resolver and detector builder use `system.optics.psf_npix` as
the PSF/image-size field.  The V3 template sets:

```yaml
system:
  optics:
    psf_npix: 160
```

The resolved value is written to `prescription_resolved.json` and summarized in
`manifest.json` as `resolved_system_summary.optics_psf_npix`.  Smoke-rendered
FITS files should therefore have dimensions controlled by this field after the
configured detector layers are applied.

## Why small grids are recommended first

Pair grids grow quickly.  With 23 scalarized parameters, an upper-triangle
all-pairs plan has 253 unordered pairs.  An 11x11 grid with one nominal and
three random nuisance replicates yields more than 120,000 pair-grid rows before
including sparse-mixture samples.  Start with dry runs and small `--max-samples`
smoke renders before attempting production-scale emission.
