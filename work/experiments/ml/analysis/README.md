# dLuxShera ML Campaign Analysis

This directory contains reusable analysis code and the lightweight local
campaign tracker for ML inverse-model training runs. It is experimental
analysis infrastructure, not a public library API and not a service. The local
workflow has three layers:

```text
work/experiments/ml/analysis/
  reusable committed loaders, tables, plotting helpers, sync helper, and tracker

Results/hpc_imports/ml/<site>/<study>/
  authoritative local mirror of synchronized compact HPC artifacts

Results/ml_analysis/<study>/
  local study-specific notebooks, figures, tables, and exploratory outputs
```

Imported results and study-specific analysis outputs are local/generated data
and remain outside git. The repository top-level `.gitignore` ignores
`Results/`.

Generally useful calculations, table shaping, and plotting should be promoted
into the committed Python helpers in this directory rather than duplicated
across notebooks.

## Studies and review status (local audit: 2026-10-30)

This is the stable ML study landing page. The [campaign notebook](ml_campaign_tracker.ipynb)
remains the interactive, data-driven tracker. Artifact completeness in that
notebook does not mean scientific review completion or current cluster status.
No new Slurm scan was performed; last observed submissions are historical only.

| Study | Local data evidence | Scientific review |
| --- | --- | --- |
| [S01](../s01/README.md) | 9 result-bearing runs in local retrospective | [Local S01/S05 retrospective](../../../../Results/ml_analysis/early_campaigns/s01_s05_early_ml_campaign_retrospective.ipynb); no tracked standalone review |
| [S05](../s05/README.md) | 4 result-bearing runs in local retrospective | Same local retrospective |
| [S06](../s06/README.md) | Verified production archive: 9/9 complete, validation only; [history reconciled](s06/reconciliation.md) | [Completed bridge review](s06/review.md), [support/provenance](s06/support.md) |
| [S07](../s07/README.md) | Verified completed export: 10/10 operationally complete (6 at budget, 4 legitimately early-stopped), validation only; [history reconciled](s07/reconciliation.md) | [Completed V4 joint-state review](s07/review.md) |
| [S08](../s08/README.md) | Local LS6 artifacts present; live status unverified | No integrated review |
| [S09](../s09/README.md) | 9 complete runs in local summary | [Local capture analysis](../../../../Results/ml_analysis/S09/s09_capture_range_analysis.ipynb); no tracked standalone review |
| [S10](../s10/README.md) | Partial local gattaca2 cohort; live status unverified | No integrated review |
| [S11](../s11/README.md) | No local import found; live status unverified | No integrated review |
| [S12](../s12/README.md) | No local import found; live status unverified | No integrated review |

### Canonical locations

- Maintained reusable loaders, plots, sync code and interactive tracker: this
  tracked directory. `work/` is not blanket-ignored; materializations are ignored.
- Tracked study-specific scientific reviews: `analysis/<study>/review.md`;
  S06 and S07 are authoritative here. Keep numeric findings there, not in
  duplicate ledgers.
- Local imported run artifacts: `Results/hpc_imports/ml/<site>/<study>/`.
  Generated notebooks, tables and figures: `Results/ml_analysis/<study>/`.
  These paths are ignored by `.gitignore`; no Results files are currently tracked.
- S06 optical source: original `Results/ml_eigenmodes/` sweep remains authoritative;
  [support](s06/support.md) labels its local analysis copy and records hashes.
- Existing `early_campaigns/` and S09 results stay in place. Local links require
  imported artifacts and will not resolve in a clean clone without those imports.

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

This notebook is the high-level cross-study tracker. It is intended to answer:
what has run, what is complete, what is comparable, and roughly how well is it
doing? It stays focused on sync/provenance, completion state, evaluation
contracts, epoch/runtime summaries, global Fisher metrics, and compact
science-priority physical metrics such as binary-separation RMSE in
milliarcseconds when available.

The tracker does not open SSH connections or run `rsync`. Refreshing the
campaign state is intentionally explicit:

1. run `sync_results.py` outside the notebook;
2. rerun the notebook or use Run All;
3. newly synchronized runs appear automatically.

Tracked scientific reviews belong beside this index (for example [S06](s06/review.md)).
Exploratory scientific interpretation belongs in local study notebooks under
`Results/ml_analysis/<study>/`. For example, an S09 capture-range analysis can
live at:

```text
Results/ml_analysis/S09/s09_capture_range_analysis.ipynb
```

That notebook should read directly from
`Results/hpc_imports/ml/<site>/S09/`; do not copy synchronized artifacts into
`Results/ml_analysis/`.

Retrospectives that intentionally span several historical studies can use a
named local subdirectory such as:

```text
Results/ml_analysis/early_campaigns/
```

These notebooks should follow the same rule: read synchronized artifacts from
`Results/hpc_imports/ml/` and keep campaign-specific interpretation local.

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

### Parameter column order

`metrics.json` is written with `sort_keys=True`, so its
`fisher_per_parameter_rmse` key order is **alphabetical and is not the column
order** of `y_true_z`/`y_pred_z`. Using those keys positionally mislabels every
column and silently corrupts any per-parameter or residual-structure conclusion;
global RMSE, skill, slice and distance-bin metrics are unaffected. The loader
recovers the true order by matching each archived per-parameter RMSE to the
column-wise RMSE recomputed from the saved arrays, and returns positional `z[i]`
labels rather than guessing when that match is ambiguous. For the V3/V4 science
vector the recovered order is:

```text
[separation_as, log_flux_total, contrast, plate_scale, M1 Z4-Z11, M2 Z4-Z11]
```

Do not reintroduce positional use of the sorted metric keys. See
[S07 reconciliation](s07/reconciliation.md) for how this was found.

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

## Campaign analysis workflow

### 1. Import an identified collection

Run from the repository root in the configured ML Python environment. Always
choose the actual source root and a generation-specific destination. The default
work root can contain older generations; it is not automatically the latest
production source. Preview (prints the rsync command; does not contact the host):

```bash
python work/experiments/ml/analysis/sync_results.py \
  --user dmckeith --host ls6.tacc.utexas.edu \
  --remote-root /scratch/11689/dmckeith/dLuxShera-ML/runs/ \
  --study S07 --local-root Results/hpc_imports/ml/ls6/production_YYYYMMDD \
  --dry-run
```

Remove `--dry-run` to transfer. `--study S07` appends S07 to both roots.
For another cluster use explicit `--host`, `--remote-root`, and `--local-root`;
the CLI's only site preset is `tacc_ls6`, so document the actual cluster in the
review rather than treating the preset as cluster verification.

Imports now stage into an empty temporary tree, check all file collisions, then
copy. A differing existing file or recorded source identity is rejected before
copying any run files; select a new snapshot destination. Identical re-imports
and missing-file additions from the same source are allowed. Imports to one
destination must be serialized. This is conservative snapshot preservation,
not an automatic latest-run merge. There is no overwrite switch. A legacy
collection without a manifest still gets file-collision protection; establish
its identity separately. Interrupted transfers do not update the sync manifest.
Dry-run does not validate remote contents or predict content collisions.

The sync manifest records user/host/root, timestamp, artifact policy, and a
per-file SHA256 inventory. Run manifests retain run IDs, revisions, dataset,
split and evaluation identities. Preserve these files together. Importing a
local tar archive requires checking member paths/types, retaining its SHA256,
and recording archive provenance; see [S06 reconciliation](s06/reconciliation.md).
Keep historical generations outside active recursive discovery, for example
`Results/hpc_imports/ml_history/`. Never merge renamed `S06B-*` into production.
A stale import is not always a different experiment generation: S07's older copy
was an in-flight snapshot of the *same* runs at the same revision, superseded
once they finished ([reconciliation](s07/reconciliation.md)). Distinguish the
cases from per-run history length and manifest stopping state before deciding.

### 2. Validate and analyze with existing interfaces

Use a specific collection root, not a directory containing conflicting historical
generations. `load_campaign` reads saved arrays to build tables; it does not run
inference or evaluate test data. Loading can use substantial memory for large
campaigns. `load_predictions` provides detailed arrays on demand.

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path('work/experiments/ml/analysis').resolve()))
from campaign_analysis import (
    load_campaign, evaluation_contract_summary, experiment_summary_table,
    matched_seed_delta_table,
)

campaign = load_campaign([Path('Results/hpc_imports/ml/ls6/S06')])
print(campaign.runs[['run_id', 'status', 'best_fisher_rmse']])
print(campaign.warnings)
print(evaluation_contract_summary(campaign.runs))
summary = experiment_summary_table(campaign.runs)
deltas = matched_seed_delta_table(campaign.runs, 'S06-E01', 'S06-E02')
out = Path('Results/ml_analysis/S06/reproduced')
out.mkdir(parents=True, exist_ok=True)
summary.to_csv(out / 'experiment_summary.csv', index=False)
deltas.to_csv(out / 'matched_seed_deltas.csv', index=False)
```

Inspect expected run counts and histories separately from `status`: `complete`
means compact artifacts present, not budget reached or scientific review done.
Review source revisions/dirty flags, prepared hash, split and validation identity,
parameter labels/order, evaluation slices and `test_evaluated`. Confirm matching
pair IDs/truth and baselines before paired comparisons. Inspect missing files
and metric warnings; do not bypass duplicate identity failures. A manifest's
test identity can be recorded without the test being evaluated.

`compute_initializer_metrics` and `compute_prediction_geometry` retain the
zero-correction baseline and scaled-z definitions above. `parameters`, `slices`,
and `distance_bins` supply common detailed tables; `residual_error_budget_table`
and `distance_bin_delta_table` provide existing aggregations. Physical-unit
metrics remain per-parameter, never a mixed-unit headline RMSE.

### 3. Visualize, review and track

Continue the example using the reusable plotting layer:

```python
import matplotlib.pyplot as plt
import campaign_plots as plots
fig, ax = plots.plot_validation_rmse_vs_epoch(campaign.history)
fig.savefig(out / 'validation_history.png', dpi=150, bbox_inches='tight')
plt.close(fig)
fig, ax = plots.plot_experiment_metric_summary(campaign.runs, 'best_fisher_rmse')
fig.savefig(out / 'experiment_rmse.png', dpi=150, bbox_inches='tight')
plt.close(fig)
```

Use parameter/slice/distance plots from `campaign_plots.py` where appropriate.
Keep genuinely study-specific diagnostics next to the study or in its local
notebook, reading the same imported arrays. The S09 local notebook already
uses shared loaders, summaries and plots; curriculum exposure schedules and
policy interpretation appropriately remain study-specific. S06's handoff script
duplicates run traversal, metrics, binning and plots, so remains noncanonical
archive input. Its mirror residual covariance/PCA diagnostics are study-specific,
not a reason to retain its hardcoded duplicate pipeline. No shared numerical
behavior or metric definitions were changed in this cleanup.

Write `analysis/sXX/review.md` using the [review template](review_template.md),
link reproducible outputs under `Results/ml_analysis/SXX/`, then update the
concise studies index here. Rerun the existing tracker only after validating
imports. Its root search works from the repository or a descendant directory;
its default root discovers all active imported studies. No heavy notebook cells
need execution merely to edit a review. Keep durable conclusions in the review,
with the developer ledger linking here rather than duplicating numeric tables.

Module ownership: `sync_results.py` imports and records source identity;
`campaign_analysis.py` loads, computes and aggregates; `campaign_plots.py` plots;
`ml_campaign_tracker.ipynb` explores the shared tables; per-study reviews carry
scientific interpretation. `work/experiments/ml/sXX/` remains the prescription;
`Results/ml_eigenmodes/` remains the original optical sensitivity store.
