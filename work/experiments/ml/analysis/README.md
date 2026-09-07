# dLuxShera ML Campaign Analysis

This directory contains the local campaign tracker for ML inverse-model
training runs. It is experimental analysis infrastructure, not a public library
API and not a service. The intended workflow is:

```text
LS6 training jobs
  -> durable compact results under LS6 $WORK
  -> explicit local rsync
  -> Results/hpc_imports/ml/<site>/
  -> reusable loader
  -> Jupyter campaign tracker
```

Imported results are data and remain outside git. The repository top-level
`.gitignore` ignores `Results/`.

## Compact Result Policy

By default, `sync_results.py` transfers only compact analysis artifacts when
they exist:

- `run_manifest.json`
- `run_config_resolved.json`
- `history.csv`
- `metrics.json`
- `evaluation_predictions.npz`
- `source_snapshot.txt`

It does not transfer checkpoints, SLURM logs, prepared datasets, training
caches, or materialized image shards by default. Checkpoints and logs require
explicit CLI switches.

## Dry Sync

Preview the effective LS6 command without copying files:

```bash
python work/experiments/ml/analysis/sync_results.py \
  --site tacc_ls6 \
  --user <tacc-user> \
  --dry-run
```

The default remote root is:

```text
/work/11689/<user>/ls6/dLuxShera-Results/ml/
```

The default local mirror is:

```text
Results/hpc_imports/ml/ls6/
```

Both can be overridden:

```bash
python work/experiments/ml/analysis/sync_results.py \
  --host <host> \
  --user <user> \
  --remote-root /work/11689/<user>/ls6/dLuxShera-Results/ml/ \
  --local-root Results/hpc_imports/ml/ls6 \
  --dry-run
```

To synchronize a study subdirectory only:

```bash
python work/experiments/ml/analysis/sync_results.py \
  --site tacc_ls6 \
  --user <tacc-user> \
  --study S05
```

Study-scoped syncs preserve the local hierarchy. The command above reads from
the remote `.../dLuxShera-Results/ml/S05/` tree and writes into
`Results/hpc_imports/ml/ls6/S05/`, not directly into the generic `ls6/` root.

To include checkpoints only when they are explicitly needed:

```bash
python work/experiments/ml/analysis/sync_results.py \
  --site tacc_ls6 \
  --user <tacc-user> \
  --include-checkpoints
```

After a successful non-dry-run sync, the helper writes
`Results/hpc_imports/ml/ls6/sync_manifest.json` with non-secret provenance:
site, host, remote root, local root, UTC sync time, and artifact policy.

## Discovery

`campaign_analysis.load_campaign(...)` discovers runs recursively by locating
`run_manifest.json`; it does not use a hard-coded run registry or fixed
study/experiment nesting. Each run directory is inspected for optional compact
artifacts and classified as:

- `complete`: all standard compact analysis artifacts are present;
- `partial`: manifest plus at least one companion artifact is present;
- `incomplete`: only the manifest is present.

Duplicate run IDs are handled deterministically. Complete results are preferred
over partial results, durable/imported paths are preferred over scratch-like
paths, and newest compatible artifacts are used only as a final tie-breaker. If
duplicate run IDs disagree on scientific identity or provenance, loading fails
with a clear error rather than silently merging data.

Future studies should appear automatically after their compact run directories
are synchronized into the local mirror and the notebook is rerun.

## Notebook

Launch Jupyter from the repository root and open:

```text
work/experiments/ml/analysis/ml_campaign_tracker.ipynb
```

The notebook does not open SSH connections or run `rsync`. Refreshing the
campaign state is intentionally explicit:

1. run `sync_results.py` outside the notebook;
2. rerun the notebook or use Run All;
3. newly synchronized runs appear automatically.

## Tables

The loader returns a `CampaignData` object with:

- `runs`: one row per selected run;
- `history`: one row per run and epoch;
- `parameters`: one row per run and science parameter;
- `slices`: one row per run and evaluation slice;
- `distance_bins`: one row per run and Fisher-distance bin;
- `sync_manifests`: local sync provenance rows;
- `warnings`: metric/provenance warnings surfaced during loading.

Raw prediction arrays are loaded lazily with
`campaign.load_predictions(run_id)`.

## Initializer Metrics

The network is evaluated as an initializer/correction for ADORA, not as the
final science estimator. Headline metrics are Fisher-scaled and compare the
learned correction against doing nothing.

For truth `t = y_true_z` and prediction `p = y_pred_z`:

```text
residual = p - t
baseline residual = -t
baseline_mse = mean(t**2)
model_mse = mean((p - t)**2)
baseline_rmse = sqrt(baseline_mse)
model_rmse = sqrt(model_mse)
mse_skill = 1 - model_mse / baseline_mse
rmse_reduction = 1 - model_rmse / baseline_rmse
```

`mse_skill = 0` means the network is no better than zero correction.
`mse_skill > 0` means the correction improves on doing nothing.
`mse_skill = 1` is perfect correction. `mse_skill < 0` means the correction
makes the Fisher-scaled error worse.

Physical-unit RMSE is reported only per parameter where units remain
meaningful. It is not used as a cross-parameter headline score because the
science vector mixes arcseconds, dimensionless source parameters, plate scale,
and nanometres.

## Geometry Diagnostics

For each prediction pair, the loader can compute:

- true correction norm;
- predicted correction norm;
- residual norm;
- correction cosine alignment;
- correction norm ratio;
- relative residual norm.

`relative_residual_norm < 1` means the learned correction moved the state
closer than zero correction in Fisher-scaled Euclidean distance. Ratios and
cosines with zero denominators are reported as `NaN`.

## Provenance and Comparability

The run table keeps source and evaluation identities visible, including source
commit, source archive ID, prepared dataset hash, split identity, validation
manifest identity, test manifest identity, and `test_evaluated`.

Cross-source comparisons are not rejected solely because source commits differ.
The important campaign comparability check is whether the scientific/evaluation
contract is shared, especially the prepared dataset, split registry, and frozen
validation manifest identity. The notebook displays these identities and flags
runs where `test_evaluated` is true so tuning/development analyses do not
accidentally treat test evaluation as ordinary validation evidence.
