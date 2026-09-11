# ML HPC Orchestration

This directory contains the shared execution layer for tracked ML studies.  A
study-specific `study.yaml` remains the scientific source of truth; these
helpers only bind that prescription to site-specific scheduler and environment
details.

## Structure

- `preflight_ml_gpu.py`: generic CUDA and study-contract preflight for any
  tracked ML study.
- `run_study_training.sh`: generic batch-body wrapper that validates the CUDA
  environment, runs preflight, persists compact study artifacts when
  `ML_PERSIST_ARTIFACT_ROOT` is set, trains via `train_from_study.py`, and
  copies compact run outputs when `ML_PERSIST_DIR` is set.
- `submit_study_run.py`: dry-run capable Slurm submit helper.  It uses tracked
  site profiles, exports the CLI-selected study/run identity into the submitted
  batch environment, creates Slurm log directories on the submit host, and
  parses real `sbatch --parsable` outputs from TACC and Gattaca2 Edge.
- `sites/gattaca2/train_ml.sbatch`: Gattaca2 Slurm profile wrapper.
- `sites/tacc_ls6/train_ml.sbatch`: TACC Lonestar6 Slurm profile wrapper.

The old `work/experiments/ml/s01/hpc/` commands remain compatibility entry
points for S01 and delegate to this generic layer where practical.

## Run Environment

`study.yaml` is the scientific source of truth.  The submit helper chooses the
site profile and constructs the submitted batch environment from explicit CLI
arguments.  In particular, `--study`, `--experiment-id`, and `--run-id` become
`ML_STUDY_PATH`, `ML_EXPERIMENT_ID`, and `ML_RUN_ID` inside the batch job; stale
parent-shell values for those variables are stripped before submission.

Pass the remaining batch-body paths explicitly for real submissions:

```bash
--repo-root <repo-root-on-cluster>
--conda-sh <scratch>/software/miniforge3/etc/profile.d/conda.sh
--conda-prefix <scratch>/conda/envs/<cuda-pytorch-env>
--prepared-root <scratch>/data/PREP-V3-nuisance-v1
--split-registry <scratch>/artifacts/S01/split/SPLIT-ML-v1.json
--validation-manifest <scratch>/artifacts/S01/validation_pairs/S01-VALIDATION-PAIRS-v1
--test-manifest <scratch>/artifacts/S01/test_pairs/S01-TEST-PAIRS-v1
--run-dir <scratch>/runs/S05/S05-E01/S05-E01-R001
--persist-dir <persistent>/S05/S05-E01/S05-E01-R001
--persist-artifact-root <persistent>/S01/artifacts
--source-commit <exact-source-commit>
```

`--repo-root` is exported as `ML_REPO_ROOT` and is used by the site wrapper
before the generic runner is invoked, so Slurm may start the job from a
submission directory outside the repository.  The generic runner then changes
to that repository root before running Python entry points.

For archive-deployed source trees with no `.git`, set
`--source-commit` and, when available, `--source-archive-id`; these populate
both `ML_SOURCE_*` and `DLUXSHERA_SOURCE_*` provenance fields in the submitted
environment.  Local Git metadata may be recorded when present, but the run path
does not require `git rev-parse HEAD` to succeed on the cluster.

`ML_PERSIST_DIR` is for transient run products such as manifests, metrics,
predictions, and checkpoints.  `ML_PERSIST_ARTIFACT_ROOT` is for compact
study-defining artifacts that should survive scratch cleanup:

```text
<artifact-root>/
  split/SPLIT-ML-v1.json
  validation_pairs/S01-VALIDATION-PAIRS-v1/
  test_pairs/S01-TEST-PAIRS-v1/
```

The preflight path validates those artifacts before copying them.  Re-copying
the same artifact is idempotent; a destination with the same scientific name but
different identity is rejected.  Artifact publication copies into a unique
temporary sibling and then publishes atomically, so concurrent same-identity
jobs can share one persistent artifact root without treating a benign race as a
failure.  The prepared shard store is intentionally not copied into this
artifact tree.

## Submit Behavior

The helper passes explicit `--output=<logroot>/%x-%j.out` and
`--error=<logroot>/%x-%j.err` arguments to `sbatch`.  The log root defaults to
`work/experiments/ml/hpc/logs` resolved on the submit host and can be changed
with `--log-root`.  The parent directories are created before `sbatch` is
called because Slurm opens log files before the batch script body runs.

The parser accepts:

- `3418708`
- `576430;edge`
- TACC wrapper output with banner/status lines followed by a valid parsable ID

The stored canonical job ID is the numeric prefix.  Malformed output, including
cluster-qualified strings with extra fields, fails instead of guessing.

## Site Notes

Gattaca2 keeps account `shera_hpc`, side-local scratch conventions, and
externally selectable GPU scheduler arguments. The tracked Gattaca2 profile
uses 24 hours, which is intentionally longer than the 8-hour LS6 profile. If a
specific Gattaca2 GPU partition has a different walltime limit, adjust the
launch-time scheduler option/profile for that submission; do not change
scientific artifact IDs or prepared-data locks to encode a scheduler limit.

Lonestar6 uses account `JPL-PUB`, partition `gpu-a100-small`, 1 node, 1 task,
8 CPUs per task, and 8 hours.  Do not request `--mem` or a normal GPU `--gres`
on this partition.  The LS6 wrapper unloads TACC's default Python module when
present, clears Python path variables, sets `PYTHONNOUSERSITE=1`, and then
activates the requested Conda environment.  Prefer passing LS6's Conda setup
script and environment prefix explicitly:

```bash
--conda-sh "$SCRATCH/software/miniforge3/etc/profile.d/conda.sh"
--conda-prefix "$SCRATCH/conda/envs/dluxshera-ml-py311"
```

The runner sources `ML_CONDA_SH` when supplied, verifies that `conda` is
available, and then activates either `ML_CONDA_ENV` or `ML_CONDA_PREFIX`.

## Dry-Run Examples

S05-E01 on LS6:

```bash
python work/experiments/ml/hpc/submit_study_run.py \
  --site tacc_ls6 \
  --study work/experiments/ml/s05/study.yaml \
  --experiment-id S05-E01 \
  --run-id S05-E01-R001 \
  --repo-root <repo-root-on-ls6> \
  --conda-sh "$SCRATCH/software/miniforge3/etc/profile.d/conda.sh" \
  --conda-prefix "$SCRATCH/conda/envs/dluxshera-ml-py311" \
  --prepared-root <scratch>/data/PREP-V3-nuisance-v1 \
  --split-registry <scratch>/artifacts/S01/split/SPLIT-ML-v1.json \
  --validation-manifest <scratch>/artifacts/S01/validation_pairs/S01-VALIDATION-PAIRS-v1 \
  --test-manifest <scratch>/artifacts/S01/test_pairs/S01-TEST-PAIRS-v1 \
  --run-dir <scratch>/runs/S05/S05-E01/S05-E01-R001 \
  --persist-dir <persistent>/S05/S05-E01/S05-E01-R001 \
  --persist-artifact-root <persistent>/S01/artifacts \
  --source-commit <exact-source-commit> \
  --dry-run
```

S05-E02 on LS6:

```bash
python work/experiments/ml/hpc/submit_study_run.py \
  --site tacc_ls6 \
  --study work/experiments/ml/s05/study.yaml \
  --experiment-id S05-E02 \
  --run-id S05-E02-R001 \
  --repo-root <repo-root-on-ls6> \
  --conda-sh "$SCRATCH/software/miniforge3/etc/profile.d/conda.sh" \
  --conda-prefix "$SCRATCH/conda/envs/dluxshera-ml-py311" \
  --prepared-root <scratch>/data/PREP-V3-nuisance-v1 \
  --split-registry <scratch>/artifacts/S01/split/SPLIT-ML-v1.json \
  --validation-manifest <scratch>/artifacts/S01/validation_pairs/S01-VALIDATION-PAIRS-v1 \
  --test-manifest <scratch>/artifacts/S01/test_pairs/S01-TEST-PAIRS-v1 \
  --run-dir <scratch>/runs/S05/S05-E02/S05-E02-R001 \
  --persist-dir <persistent>/S05/S05-E02/S05-E02-R001 \
  --persist-artifact-root <persistent>/S01/artifacts \
  --source-commit <exact-source-commit> \
  --dry-run
```

S01 compatibility on Gattaca2 Edge still uses the legacy wrapper.  Set the
existing S01 variables, including `S01_PROJECT_ARTIFACT_ROOT` if the default
`$S01_PROJECT_RESULTS_ROOT/S01/artifacts` is not desired:

```bash
export S01_SCRATCH_SIDE=edge
export S01_SLURM_CLUSTER=edge
export S01_CONDA_ENV=<cuda-pytorch-env>
export S01_GPU_SBATCH_ARGS="--partition=<gpu_partition> --gres=<gpu_resource>"
work/experiments/ml/s01/hpc/submit_s01_e01.sh
```

## S06-S09 Human Runbook

Do not execute cluster preparation or submission commands from a local
development task. The sequence below is the implemented command surface for a
human operator.

### Local / Before Cluster

```bash
PYTHONPATH=src pytest -q \
  tests/ml/test_prepared_v4.py \
  tests/ml/test_catalog_splits_pairs.py \
  tests/ml/test_study_prescriptions.py \
  tests/ml/test_s06_s09_studies.py \
  tests/ml/test_models_training.py \
  tests/ml/test_dynamic_pair_dataset.py \
  tests/ml/test_s05_study_and_hpc.py
```

```bash
PYTHONPATH=src python3 work/experiments/ml/materialize_study_artifacts.py audit-study \
  --study work/experiments/ml/s06/study.yaml \
  --study work/experiments/ml/s07/study.yaml \
  --study work/experiments/ml/s08/study.yaml \
  --study work/experiments/ml/s09/study.yaml
```

```bash
PYTHONPATH=src python3 work/experiments/ml/materialize_study_artifacts.py expand \
  --study work/experiments/ml/s09/study.yaml \
  --output-dir /tmp/s09_run_plan \
  --overwrite
```

### Gattaca2 Preparation

Set site roots explicitly. `V4_PLAN_ROOT` must be the materialized V4
state-plan directory containing `freeze_manifest.json` and
`render_contract.json`, for example the transferred
`render_v4/stateplans/master_v4` directory used by the renderer.

```bash
export REPO_ROOT=<repo-root-on-gattaca2>
export GATTACA_SCRATCH=<scratch>
export V4_SOURCE_ROOT=<gattaca2-projects>/shera_ml_master_v4
export V4_PLAN_ROOT=<materialized-v4-state-plan-root>
```

Audit raw V4 metadata/files:

```bash
PYTHONPATH=src python3 work/experiments/ml/datasets/audit_dataset.py \
  "$V4_SOURCE_ROOT" \
  --verify-files \
  --output-json "$GATTACA_SCRATCH/audits/v4_raw_audit.json"
```

Dry-run prepared V4:

```bash
PYTHONPATH=src python3 examples/scripts/prepare_ml_dataset.py \
  --dataset-kind v4 \
  --source-root "$V4_SOURCE_ROOT" \
  --v4-plan-root "$V4_PLAN_ROOT" \
  --outdir "$GATTACA_SCRATCH/prepared/PREP-V4-v1" \
  --dtype float32 \
  --v4-source-audit sample \
  --dry-run
```

Production prepare V4:

```bash
PYTHONPATH=src python3 examples/scripts/prepare_ml_dataset.py \
  --dataset-kind v4 \
  --source-root "$V4_SOURCE_ROOT" \
  --v4-plan-root "$V4_PLAN_ROOT" \
  --outdir "$GATTACA_SCRATCH/prepared/PREP-V4-v1" \
  --dtype float32 \
  --v4-source-audit sample
```

Resume preparation:

```bash
PYTHONPATH=src python3 examples/scripts/prepare_ml_dataset.py \
  --dataset-kind v4 \
  --source-root "$V4_SOURCE_ROOT" \
  --v4-plan-root "$V4_PLAN_ROOT" \
  --outdir "$GATTACA_SCRATCH/prepared/PREP-V4-v1" \
  --dtype float32 \
  --v4-source-audit sample \
  --resume
```

Deep prepared-data audit:

```bash
PYTHONPATH=src python3 - <<'PY'
from pathlib import Path
from dluxshera.datasets.prepared_v4 import validate_prepared_v4_dataset_identity
validate_prepared_v4_dataset_identity(Path("<prepared-root>"), deep=True)
print("PREPARED_V4_DEEP_AUDIT: PASS")
PY
```

Materialize compact artifacts:

```bash
PYTHONPATH=src python3 work/experiments/ml/materialize_study_artifacts.py make-split \
  --prepared-root "$GATTACA_SCRATCH/prepared/PREP-V4-v1" \
  --out "$GATTACA_SCRATCH/artifacts/v4/SPLIT-V4-ROLE-PRESERVING-v1.json" \
  --artifact-id SPLIT-V4-ROLE-PRESERVING-v1 \
  --role-preserving-v4
```

```bash
PYTHONPATH=src python3 work/experiments/ml/materialize_study_artifacts.py make-split \
  --prepared-root "$GATTACA_SCRATCH/prepared/PREP-V4-v1" \
  --out "$GATTACA_SCRATCH/artifacts/v4/SPLIT-V4-S08-NUISANCE-HOLDOUT-v1.json" \
  --artifact-id SPLIT-V4-S08-NUISANCE-HOLDOUT-v1 \
  --role-preserving-v4 \
  --nuisance-holdout-indices 8,9
```

```bash
PYTHONPATH=src python3 work/experiments/ml/materialize_study_artifacts.py make-scaler \
  --prepared-root "$GATTACA_SCRATCH/prepared/PREP-V4-v1" \
  --split-registry "$GATTACA_SCRATCH/artifacts/v4/SPLIT-V4-ROLE-PRESERVING-v1.json" \
  --out "$GATTACA_SCRATCH/artifacts/v4/SCALER-V4-GLOBAL-MAX-ABS-v1.json" \
  --artifact-id SCALER-V4-GLOBAL-MAX-ABS-v1 \
  --dataset-family joint_full_v4 \
  --dataset-family radial_capture_v4 \
  --mode global_max_abs
```

```bash
PYTHONPATH=src python3 work/experiments/ml/materialize_study_artifacts.py make-pairs \
  --study work/experiments/ml/s08/study.yaml \
  --prepared-root "$GATTACA_SCRATCH/prepared/PREP-V4-v1" \
  --split-registry "$GATTACA_SCRATCH/artifacts/v4/SPLIT-V4-ROLE-PRESERVING-v1.json" \
  --split-profile nuisance_holdout="$GATTACA_SCRATCH/artifacts/v4/SPLIT-V4-S08-NUISANCE-HOLDOUT-v1.json" \
  --output-root "$GATTACA_SCRATCH/artifacts" \
  --artifact all
```

```bash
PYTHONPATH=src python3 work/experiments/ml/materialize_study_artifacts.py make-pairs \
  --study work/experiments/ml/s09/study.yaml \
  --prepared-root "$GATTACA_SCRATCH/prepared/PREP-V4-v1" \
  --split-registry "$GATTACA_SCRATCH/artifacts/v4/SPLIT-V4-ROLE-PRESERVING-v1.json" \
  --output-root "$GATTACA_SCRATCH/artifacts" \
  --artifact all
```

S08 standard profile lock for E01/E02/E03:

```bash
PYTHONPATH=src python3 work/experiments/ml/materialize_study_artifacts.py make-lock \
  --study work/experiments/ml/s08/study.yaml \
  --artifact-profile standard \
  --prepared-root "$GATTACA_SCRATCH/prepared/PREP-V4-v1" \
  --split-registry "$GATTACA_SCRATCH/artifacts/v4/SPLIT-V4-ROLE-PRESERVING-v1.json" \
  --scaler "$GATTACA_SCRATCH/artifacts/v4/SCALER-V4-GLOBAL-MAX-ABS-v1.json" \
  --pair-manifest validation_c="$GATTACA_SCRATCH/artifacts/S08/validation_c_pairs/S08-C-VALIDATION-PAIRS-v1" \
  --pair-manifest validation_abc="$GATTACA_SCRATCH/artifacts/S08/validation_abc_pairs/S08-ABC-VALIDATION-PAIRS-v1" \
  --pair-manifest test="$GATTACA_SCRATCH/artifacts/S08/test_pairs/S08-TEST-PAIRS-v1" \
  --out "$GATTACA_SCRATCH/artifacts/S08/S08-STANDARD-ARTIFACT-LOCK-v1.json"
```

S08 holdout profile lock for E04:

```bash
PYTHONPATH=src python3 work/experiments/ml/materialize_study_artifacts.py make-lock \
  --study work/experiments/ml/s08/study.yaml \
  --artifact-profile nuisance_holdout \
  --prepared-root "$GATTACA_SCRATCH/prepared/PREP-V4-v1" \
  --split-registry "$GATTACA_SCRATCH/artifacts/v4/SPLIT-V4-S08-NUISANCE-HOLDOUT-v1.json" \
  --scaler "$GATTACA_SCRATCH/artifacts/v4/SCALER-V4-GLOBAL-MAX-ABS-v1.json" \
  --pair-manifest validation_abc_seen="$GATTACA_SCRATCH/artifacts/S08/validation_abc_seen_pairs/S08-ABC-SEEN-VALIDATION-PAIRS-v1" \
  --pair-manifest test_seen="$GATTACA_SCRATCH/artifacts/S08/test_seen_pairs/S08-SEEN-TEST-PAIRS-v1" \
  --pair-manifest unseen_nuisance_audit="$GATTACA_SCRATCH/artifacts/S08/unseen_nuisance_audit_pairs/S08-UNSEEN-NUISANCE-AUDIT-PAIRS-v1" \
  --out "$GATTACA_SCRATCH/artifacts/S08/S08-HOLDOUT-ARTIFACT-LOCK-v1.json"
```

S09 lock:

```bash
PYTHONPATH=src python3 work/experiments/ml/materialize_study_artifacts.py make-lock \
  --study work/experiments/ml/s09/study.yaml \
  --prepared-root "$GATTACA_SCRATCH/prepared/PREP-V4-v1" \
  --split-registry "$GATTACA_SCRATCH/artifacts/v4/SPLIT-V4-ROLE-PRESERVING-v1.json" \
  --scaler "$GATTACA_SCRATCH/artifacts/v4/SCALER-V4-GLOBAL-MAX-ABS-v1.json" \
  --pair-manifest validation="$GATTACA_SCRATCH/artifacts/S09/validation_pairs/S09-VALIDATION-PAIRS-v1" \
  --pair-manifest test="$GATTACA_SCRATCH/artifacts/S09/test_pairs/S09-TEST-PAIRS-v1" \
  --out "$GATTACA_SCRATCH/artifacts/S09/S09-ARTIFACT-LOCK-v1.json"
```

```bash
PYTHONPATH=src python3 work/experiments/ml/materialize_study_artifacts.py validate-lock \
  --study work/experiments/ml/s09/study.yaml \
  --prepared-root <scratch>/prepared/PREP-V4-v1 \
  --split-registry <scratch>/artifacts/v4/SPLIT-V4-ROLE-PRESERVING-v1.json \
  --scaler <scratch>/artifacts/v4/SCALER-V4-GLOBAL-MAX-ABS-v1.json \
  --validation-manifest <scratch>/artifacts/S09/validation_pairs/S09-VALIDATION-PAIRS-v1 \
  --test-manifest <scratch>/artifacts/S09/test_pairs/S09-TEST-PAIRS-v1 \
  --artifact-lock <scratch>/artifacts/S09/S09-ARTIFACT-LOCK-v1.json
```

Large staged product: prepared shards and `index.jsonl`. Compact staged
products: manifests, scaler, split registries, pair manifests, and artifact
locks. Prepared identity is root-independent and should survive moving from
Gattaca2 to Lonestar6.

### Lonestar6

Set site roots:

```bash
export REPO_ROOT=<repo-root-on-ls6>
export LS6_SCRATCH=<ls6-scratch>
```

GPU preflight:

```bash
PYTHONPATH=src python3 work/experiments/ml/hpc/preflight_ml_gpu.py \
  --study work/experiments/ml/s09/study.yaml \
  --experiment-id S09-E01 \
  --run-id S09-E01-R001 \
  --prepared-root <ls6-scratch>/prepared/PREP-V4-v1 \
  --split-registry <ls6-scratch>/artifacts/v4/SPLIT-V4-ROLE-PRESERVING-v1.json \
  --scaler <ls6-scratch>/artifacts/v4/SCALER-V4-GLOBAL-MAX-ABS-v1.json \
  --validation-manifest <ls6-scratch>/artifacts/S09/validation_pairs/S09-VALIDATION-PAIRS-v1 \
  --test-manifest <ls6-scratch>/artifacts/S09/test_pairs/S09-TEST-PAIRS-v1 \
  --artifact-lock <ls6-scratch>/artifacts/S09/S09-ARTIFACT-LOCK-v1.json \
  --device cuda:0
```

S08-E04 holdout preflight must include both the seen primary validation
manifest and the unseen-nuisance audit manifest:

```bash
PYTHONPATH=src python3 work/experiments/ml/hpc/preflight_ml_gpu.py \
  --study work/experiments/ml/s08/study.yaml \
  --experiment-id S08-E04 \
  --run-id S08-E04-R001 \
  --prepared-root "$LS6_SCRATCH/prepared/PREP-V4-v1" \
  --split-registry "$LS6_SCRATCH/artifacts/v4/SPLIT-V4-S08-NUISANCE-HOLDOUT-v1.json" \
  --scaler "$LS6_SCRATCH/artifacts/v4/SCALER-V4-GLOBAL-MAX-ABS-v1.json" \
  --validation-manifest "$LS6_SCRATCH/artifacts/S08/validation_abc_seen_pairs/S08-ABC-SEEN-VALIDATION-PAIRS-v1" \
  --test-manifest "$LS6_SCRATCH/artifacts/S08/test_seen_pairs/S08-SEEN-TEST-PAIRS-v1" \
  --audit-manifest unseen_nuisance_audit="$LS6_SCRATCH/artifacts/S08/unseen_nuisance_audit_pairs/S08-UNSEEN-NUISANCE-AUDIT-PAIRS-v1" \
  --artifact-lock "$LS6_SCRATCH/artifacts/S08/S08-HOLDOUT-ARTIFACT-LOCK-v1.json" \
  --device cuda:0
```

One V3 S06 smoke:

```bash
python work/experiments/ml/hpc/submit_study_run.py \
  --site tacc_ls6 \
  --study work/experiments/ml/s06/study.yaml \
  --experiment-id S06-E01 \
  --run-id S06-E01-R001 \
  --repo-root <repo-root-on-ls6> \
  --conda-prefix <cuda-pytorch-env> \
  --prepared-root <ls6-scratch>/prepared/PREP-V3-nuisance-v1 \
  --split-registry <ls6-scratch>/artifacts/S01/split/SPLIT-ML-v1.json \
  --validation-manifest <ls6-scratch>/artifacts/S01/validation_pairs/S01-VALIDATION-PAIRS-v1 \
  --test-manifest <ls6-scratch>/artifacts/S01/test_pairs/S01-TEST-PAIRS-v1 \
  --run-dir <ls6-scratch>/runs/S06/S06-E01/S06-E01-R001 \
  --dry-run
```

One V4 real-data smoke preview and full plan preview:

```bash
python work/experiments/ml/hpc/submit_study_run.py \
  --site tacc_ls6 \
  --study work/experiments/ml/s09/study.yaml \
  --experiment-id S09-E01 \
  --run-id S09-E01-R001 \
  --repo-root <repo-root-on-ls6> \
  --conda-prefix <cuda-pytorch-env> \
  --prepared-root <ls6-scratch>/prepared/PREP-V4-v1 \
  --split-registry <ls6-scratch>/artifacts/v4/SPLIT-V4-ROLE-PRESERVING-v1.json \
  --scaler <ls6-scratch>/artifacts/v4/SCALER-V4-GLOBAL-MAX-ABS-v1.json \
  --validation-manifest <ls6-scratch>/artifacts/S09/validation_pairs/S09-VALIDATION-PAIRS-v1 \
  --test-manifest <ls6-scratch>/artifacts/S09/test_pairs/S09-TEST-PAIRS-v1 \
  --artifact-lock <ls6-scratch>/artifacts/S09/S09-ARTIFACT-LOCK-v1.json \
  --run-dir <ls6-scratch>/runs/S09/S09-E01/S09-E01-R001 \
  --dry-run
```

```bash
python work/experiments/ml/hpc/submit_study_run.py \
  --site tacc_ls6 \
  --study work/experiments/ml/s06/study.yaml \
  --repo-root <repo-root-on-ls6> \
  --prepared-root <ls6-scratch>/prepared/PREP-V3-nuisance-v1 \
  --split-registry <ls6-scratch>/artifacts/S01/split/SPLIT-ML-v1.json \
  --validation-manifest <ls6-scratch>/artifacts/S01/validation_pairs/S01-VALIDATION-PAIRS-v1 \
  --test-manifest <ls6-scratch>/artifacts/S01/test_pairs/S01-TEST-PAIRS-v1 \
  --run-dir <ls6-scratch>/runs \
  --plan-preview
```

For S07-S09, use the same `--plan-preview` command with the V4 prepared root,
split registry, scaler, manifests, and artifact lock. S08-E04 uses the holdout
split, seen-validation manifest, unseen audit manifest, and holdout artifact
lock; S08-E01/E02/E03 use the standard split and standard lock.

Explicit multi-run submission uses `--submit-plan`; preview never submits:

```bash
python work/experiments/ml/hpc/submit_study_run.py \
  --site tacc_ls6 \
  --study work/experiments/ml/s09/study.yaml \
  --repo-root "$REPO_ROOT" \
  --conda-prefix <cuda-pytorch-env> \
  --prepared-root "$LS6_SCRATCH/prepared/PREP-V4-v1" \
  --split-registry "$LS6_SCRATCH/artifacts/v4/SPLIT-V4-ROLE-PRESERVING-v1.json" \
  --scaler "$LS6_SCRATCH/artifacts/v4/SCALER-V4-GLOBAL-MAX-ABS-v1.json" \
  --validation-manifest "$LS6_SCRATCH/artifacts/S09/validation_pairs/S09-VALIDATION-PAIRS-v1" \
  --test-manifest "$LS6_SCRATCH/artifacts/S09/test_pairs/S09-TEST-PAIRS-v1" \
  --artifact-lock "$LS6_SCRATCH/artifacts/S09/S09-ARTIFACT-LOCK-v1.json" \
  --run-dir "$LS6_SCRATCH/runs" \
  --launch-packet "$LS6_SCRATCH/launches/S09" \
  --submit-plan
```

To submit one selected row, use the same command plus
`--experiment-id <ID> --run-id <ID-RNNN> --submit-plan`. To retry or resume one
row, use the same selected-row command with either `--overwrite` or
`--resume-checkpoint <run-dir>/checkpoint_last.pt`.

## S10-S12 Gattaca2 Preview Surface

S10-S12 use the same generic materialization, preflight, preview, and
submission machinery as S06-S09. Do not run `--submit-plan` until the preview
and preflight outputs have been inspected by the launch operator.

Repository audit:

```bash
PYTHONPATH=src python3 work/experiments/ml/materialize_study_artifacts.py audit-study \
  --study work/experiments/ml/s10/study.yaml \
  --study work/experiments/ml/s11/study.yaml \
  --study work/experiments/ml/s12/study.yaml
```

Expected output: S10 = 9, S11 = 6, S12 = 6, total = 21, and
`all_evaluate_test_false: true`.

Materialize the S10-v1 nominal physical-theta science FIM source by recomputing
the full nominal science FIM with the V3/V4 Fisher-scale loss convention. This
writes the science-only physical FIM with `coordinate_space: physical_theta`,
derives its diagonal Fisher sigmas, compares those sigmas to the authoritative
PREP-V4 scales, and records the transformed Fisher-scaled-z sanity summary:

```bash
PYTHONPATH=src python3 work/experiments/ml/materialize_study_artifacts.py make-s10-fim-source \
  --prepared-root "$GATTACA_SCRATCH/prepared/PREP-V4-v1" \
  --out "$GATTACA_SCRATCH/artifacts/S10/s10_science_fim_source.json"
```

Materialize S10 eigenbasis from the coordinate-declared science FIM source:

```bash
PYTHONPATH=src python3 work/experiments/ml/materialize_study_artifacts.py make-eigenbasis \
  --prepared-root "$GATTACA_SCRATCH/prepared/PREP-V4-v1" \
  --matrix-json "$GATTACA_SCRATCH/artifacts/S10/s10_science_fim_source.json" \
  --artifact-id S10-V4-SCIENCE-FIM-EIGENBASIS-v1 \
  --out "$GATTACA_SCRATCH/artifacts/S10/S10-V4-SCIENCE-FIM-EIGENBASIS-v1.json"
```

Materialize pair manifests for each new study:

```bash
PYTHONPATH=src python3 work/experiments/ml/materialize_study_artifacts.py make-pairs \
  --study work/experiments/ml/s10/study.yaml \
  --prepared-root "$GATTACA_SCRATCH/prepared/PREP-V4-v1" \
  --split-registry "$GATTACA_SCRATCH/artifacts/v4/SPLIT-V4-ROLE-PRESERVING-v1.json" \
  --output-root "$GATTACA_SCRATCH/artifacts" \
  --artifact all
```

Repeat the same `make-pairs` command with `s11/study.yaml` and
`s12/study.yaml`.

Freeze the S12 noisy-validation recipe after the clean S12 validation pairs are
materialized:

```bash
PYTHONPATH=src python3 work/experiments/ml/materialize_study_artifacts.py make-noisy-eval \
  --study work/experiments/ml/s12/study.yaml \
  --experiment-id S12-E01 \
  --prepared-root "$GATTACA_SCRATCH/prepared/PREP-V4-v1" \
  --split-registry "$GATTACA_SCRATCH/artifacts/v4/SPLIT-V4-ROLE-PRESERVING-v1.json" \
  --pair-manifest "$GATTACA_SCRATCH/artifacts/S12/validation_pairs/S12-VALIDATION-PAIRS-v1" \
  --artifact-id S12-PHOTON-NOISE-VALIDATION-v1 \
  --out "$GATTACA_SCRATCH/artifacts/S12/S12-PHOTON-NOISE-VALIDATION-v1.json"
```

S10 plan preview on Gattaca2:

```bash
python work/experiments/ml/hpc/submit_study_run.py \
  --site gattaca2 \
  --study work/experiments/ml/s10/study.yaml \
  --repo-root "$REPO_ROOT" \
  --conda-prefix <cuda-pytorch-env> \
  --prepared-root "$GATTACA_SCRATCH/prepared/PREP-V4-v1" \
  --split-registry "$GATTACA_SCRATCH/artifacts/v4/SPLIT-V4-ROLE-PRESERVING-v1.json" \
  --scaler "$GATTACA_SCRATCH/artifacts/v4/SCALER-V4-GLOBAL-MAX-ABS-v1.json" \
  --validation-manifest "$GATTACA_SCRATCH/artifacts/S10/validation_pairs/S10-VALIDATION-PAIRS-v1" \
  --test-manifest "$GATTACA_SCRATCH/artifacts/S10/test_pairs/S10-TEST-PAIRS-v1" \
  --artifact-lock "$GATTACA_SCRATCH/artifacts/S10/S10-ARTIFACT-LOCK-v1.json" \
  --eigenbasis-artifact "$GATTACA_SCRATCH/artifacts/S10/S10-V4-SCIENCE-FIM-EIGENBASIS-v1.json" \
  --run-dir "$GATTACA_SCRATCH/runs" \
  --extra-sbatch-arg=--partition=<gpu_partition> \
  --extra-sbatch-arg=--gres=<gpu_resource> \
  --plan-preview
```

For S11 and S12 previews, use their study paths and their own validation/test
manifests and locks. S11 does not need `--eigenbasis-artifact`; S12 must include
`--noisy-eval-artifact "$GATTACA_SCRATCH/artifacts/S12/S12-PHOTON-NOISE-VALIDATION-v1.json"`.
S12 is the SHERA photon-noise observation condition: photon noise is enabled,
read noise and dark current remain disabled, and noise is applied in count
space before `IntensityScaler`.
