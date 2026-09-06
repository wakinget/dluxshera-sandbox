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
externally selectable GPU scheduler arguments.

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
