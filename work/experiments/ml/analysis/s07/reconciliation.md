# S07 provenance reconciliation — 2026-10-30

The [scientific review](review.md) is reproducible from the completed export,
installed at `Results/hpc_imports/ml/ls6/S07/`.

The discrepancy was simpler than S06's: the pre-existing local import was an
**in-flight snapshot of the same ten runs at the same source revision**, taken
while four of them were still training. It was not a different experiment
generation, a different evaluation population, or a renamed cohort.

| Population | Evidence | Disposition |
| --- | --- | --- |
| Earlier import, synced 2026-09-29T01:23:55Z from `/scratch/11689/dmckeith/dLuxShera-ML/runs` | Same ten run IDs, same revision `5fa65027f1fe4c7495f9be292303d25fce82fe4e`, same evaluation contract. Six runs byte-identical to the completed export; four captured mid-training | Superseded in-flight snapshot, preserved byte-for-byte at `Results/hpc_imports/ml_history/ls6/S07_inflight_20260929/` |
| Completed export `s07_complete_20261030T174759Z` | Ten runs, six at the 600-epoch budget and four legitimately early-stopped; 50 artifacts hash-verified | Preferred scientific reference; now the active S07 import |

## The four differing runs

Each differed only because the older copy was taken before training finished:

| Run | Older history rows / last epoch | Completed history rows / last epoch |
| --- | ---: | ---: |
| S07-E01-R002 | 316 / 315 | 600 / 599 |
| S07-E02-R001 | 360 / 359 | 600 / 599 |
| S07-E02-R003 | 358 / 357 | 591 / 590 |
| S07-E06-R001 | 360 / 359 | 600 / 599 |

The remaining six (E01-R001, E01-R003, E02-R002, E03-R001, E04-R001, E05-R001)
were byte-identical across all five artifacts, which is consistent with those
runs having already finished before the September sync. The four stale copies
also recorded `git.source_commit = null` where the completed export carries the
revision; `git.commit` agreed in both.

## Verification performed

- The archive `s07_complete_20261030T174759Z.tar.gz` hashes to
  `5937e7e6a739fdab2214745ea9126511e91c085f82511c81932277585f392aea`. All 51
  members are regular files with no absolute or parent-relative paths.
- All 50 compact artifacts named in `s07_export_manifest.json` were hashed and
  matched its SHA-256 inventory exactly, then matched again against the installed
  import's own `files_sha256` — zero mismatches in both directions.
- The ten expected run IDs are present, with no extras and nothing missing.
- Best-checkpoint and final-epoch states were read separately; the export
  manifest's `status` field (`FULL` / `EARLY_STOP`) agrees with each run
  manifest's `early_stopping` block.
- RMSE and MSE skill recomputed from the saved scaled-z arrays agree with
  `metrics.json` to **0.0 absolute difference** in all ten runs.
- All ten share the PREP-V4 hash
  `2b6defd272a3ecf7c28bae15d396e29236644d8e1ff44251815314f54bb98e7f`, split hash
  `951f9d38c3a33a2c5d24e7c63df7938b24a44d8c9945d4f3be7d934d7fa5467a`, and
  validation manifest SHA256
  `27608006dd9d76e3eb59a4eca9bbfafe04d7c50880b4d8ae949fbf577f534505`
  (`S07-VALIDATION-PAIRS-v1`).
- Taking E01-R001 as reference, all ten runs have identical `pair_record_id`
  arrays in identical order, and `y_true_z` and `fisher_distance_l2` max absolute
  differences of 0.0. Only predictions differ.
- All ten manifests record `test_evaluated: false`. No test evaluation was
  performed.

## Import method and limitations

The completed export was installed through the maintained
`sync_results.import_staged_results`, which stages, checks every file collision
and refuses to overwrite differing content. Because the older in-flight copies
would have collided, they were first moved to the historical namespace; nothing
was overwritten in place. The active `S07/sync_manifest.json` records the archive
path and SHA256, the import method, the export manifest path and production time,
the source root, and a per-file SHA256 inventory. The original parent sync
manifest is preserved unchanged at
`Results/ml_analysis/S07/reconciliation/older_inflight_sync_manifest.json`, with a
copy retained alongside the historical population.

**Limitations.** The remote source root is taken from the export manifest's
`source_root` and the older sync manifest; no remote scan or independent remote
hash was obtained. All ten runs record `git.dirty = true` and the working-tree
diff is not in the compact archive, so the recorded revision identifies the
launch commit but does not fully pin the executed code.

The historical population is outside the notebook's recursive
`Results/hpc_imports/ml/` discovery, which prevents run-ID collisions. Do not
load the active and historical roots together or disable strict duplicate checks.

## Local audit products

Under `Results/ml_analysis/S07/reconciliation/`:

- `inventory.json`: both generations' run identities, history lengths, last
  epochs, git identity, and SHA256 for every per-run file.
- `older_inflight_sync_manifest.json`: the original parent metadata, unchanged.

The supplied export tree and archive are retained unchanged at
`Results/ml_analysis/S07/s07_complete_20261030T174759Z/` and
`s07_complete_20261030T174759Z.tar.gz` as noncanonical input; the active copies
live in the import store.

## Parameter-order correction

While validating the saved arrays, the maintained loader was found to mislabel
every parameter column. `metrics.json` is written with `sort_keys=True`, so its
`fisher_per_parameter_rmse` key order is alphabetical, not the array column
order. The true order, recovered by matching archived per-parameter RMSE values
to column-wise RMSE from the saved arrays and consistent with S06's
independently documented order, is

```text
[separation_as, log_flux_total, contrast, plate_scale, M1 Z4-Z11, M2 Z4-Z11]
```

`campaign_analysis._parameter_labels` now performs that recovery and returns
positional `z[i]` labels rather than guessing when the match is ambiguous. This
affects per-parameter and residual-structure readings for S06, S07 and S09;
global RMSE, skill, slice and distance-bin metrics are unaffected because they do
not depend on column labels.
