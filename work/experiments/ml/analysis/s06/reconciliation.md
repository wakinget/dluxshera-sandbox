# S06 provenance reconciliation — 2026-10-30

The [scientific review](review.md) is now reproducible from the verified
production archive, installed at `Results/hpc_imports/ml/ls6/S06/`.
The discrepancy was an earlier-generation work-tree mirror sharing production
run IDs. It was not a change of evaluation population. Stale synchronization
explains why that generation remained in the active local path; the precise
remote transfer history cannot be reconstructed from the available metadata.

| Population | Evidence | Disposition |
| --- | --- | --- |
| Earlier `/work/11689/dmckeith/ls6/dLuxShera-Results/ml/S06` mirror | Eight runs, E01/E02 R001–R003, E03-R001, E04-R001; last epoch indices 266–399; revision `5687100cb148590980dfa343bd24ee9aff635ddc`, source archive `4546f4caccd653220708a56612869d22b02c9a24f4f42d8f48c01b41adc39e05` | Superseded historical population, preserved byte-for-byte under `Results/hpc_imports/ml_history/ls6/S06_work_mirror_20261030/` |
| `S06-bridge_20260907`, renamed `S06B-*` | User reports seven retained runs; no matching local population found in repository imports or Downloads | Historical, separate; not merged. Retrieve original bridge directory/manifests if its provenance needs verification |
| Production `/scratch/11689/dmckeith/dLuxShera-ML/runs/S06` export | Nine runs E01–E03 × three seeds; eight last epochs 499, E01-R003 last epoch 434, configured early stopping; revision `5fa65027f1fe4c7495f9be292303d25fce82fe4e`, dirty tree | Preferred scientific reference; verified local archive, now active S06 import |

Archive found at `/Users/dmckeith/Downloads/s06_s09_review_20261008.tar.gz`;
SHA256 `6c5c9db0819759bc7c11c43a8d198a7fabbe34521aa517b594efb206cf15d190`.
Only regular-file `runs/S06/` members were extracted; S09 imports were unchanged.
Remote origin is supplied by the user; no remote status scan or independent
remote archive hash was obtained. The archive's dirty working-tree diff remains
unavailable. E02/E03 have `git.source_commit = null` but `git.commit` records
the revision above; use the existing loader's source-commit fallback.

Both populations share PREP-V3 hash
`4cdc325fbf8d4a0e07195ab075bea6f5035dfc01c9990cac03ee1f59c131e5e6`,
split hash `29f0e95c3819cbeb5ce00aafb593445510723ea5fc20e2e7f3e585c1b9615314`,
and validation SHA256
`68ccd41a35d286c8b060f291eef6c788a6b0d97c9660868f74e01b2b4feae499`.
The production saved arrays share truth vectors and ordered pair IDs across all
nine runs; RMSE recomputed from saved scaled-z arrays agrees with metrics and
handoff CSVs. Mean RMSEs are 49.502709, 48.089540, 48.890410. All production
manifests record `test_evaluated: false`; no test evaluation was performed.
The old E01-R001 RMSE was 56.919606 versus production 48.561461.

The old parent sync manifest names the work root and timestamp
2026-09-07T22:17:56Z, which predates some run creation timestamps. Treat it as
inherited collection metadata, not proof of the last per-run transfer time.
Individual manifests/history/metrics establish the generation distinction.

Local audit products in `Results/ml_analysis/S06/reconciliation/`:

- `inventory.json`: all 17 run identities, histories, metrics, dataset/split/
  validation identities, SHA256 for every imported per-run file, old and new paths.
- `older_parent_sync_manifest.json`: original parent metadata, unchanged.
- `archive_extracted/runs/S06/`: verified extraction; active copies are in the
  import store. `production_summary.csv`: maintained-loader aggregation.

The active `S06/sync_manifest.json` records archive identity, import method,
source root, import time and per-file hashes. History is outside the notebook's
recursive `Results/hpc_imports/ml/` discovery to avoid run-ID collisions. Do not
load active and historical roots together or disable strict duplicate checks.

The scaled-z to physical Fisher basis transform and optical-model identity
remain independent open scientific questions; this reconciliation does not
resolve or use that transform.
