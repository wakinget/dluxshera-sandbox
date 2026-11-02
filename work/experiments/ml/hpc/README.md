# ML HPC Orchestration

S12 optimization follow-up: [performance review and Gattaca2 CPU qualification](../../../../docs/dev/s12_pipeline_performance.md).
Two opt-in noise strategies (`batched` JAX and deterministic per-record `numpy`)
are under local validation; `reference` remains the default. That document holds
the step-by-step qualification procedure, the experiment-aware validator, the
complete 1,024-record equivalence check, the NumPy statistical validation, and
the first-real-run GPU verification. Historical incident and recovery notes
below are retained.

Diagnostics in this directory:

| Script | Purpose |
|---|---|
| `preflight_ml_gpu.py` | CUDA and frozen-artifact preflight plus a bounded data-loader smoke test. Supports `--noise-implementation reference\|batched\|numpy`. |
| `diagnose_ml_dataloader.py` | Supervised throughput, clean baseline, stage profile for all three strategies, and exact reference/batched equivalence. Does not read the locked test manifest. |
| `qualify_s12_pipeline.py` | Reduces the qualification JSON reports to one PASS / REMEASURE / FAIL decision. Reruns nothing. Exit 0 / 2 / 1 respectively, so a shell can branch on it. |
| `analyze_image_statistics.py` | Read-only count statistics and SHERA Poisson/Gaussian branch selection for a prepared dataset. |
| `benchmark_s12_fixture.py` | Local synthetic-fixture throughput comparison across clean/reference/batched/numpy. Engineering regression only, never capacity certification. |
| `benchmark_pair_sampling.py` | Isolates pair-policy validation and `sample_pair` cost at configurable catalog sizes, reporting before/after the catalog-validation cache. Read-only. |
| `submit_study_run.py` | Launch preview and submission. Supports `--noise-implementation`. |

Selecting `numpy` affects **training noise only**: the frozen noisy-validation
recipe stays JAX-backed, so no frozen artifact needs regenerating. `numpy` is a
different RNG stream, not a bitwise match to the reference, so
`--verify-equivalence` rejects it; its validation is statistical, in
`tests/ml/test_numpy_noise.py`. It also does not make a run JAX-free: JAX is
still imported, frozen validation still samples through it, and the fork-safe
worker start method is still required.

A successful diagnostic exit does not mean its throughput measurement is usable
— a run can succeed while reporting `sustained_measurement_valid: false`, which
NumPy triggers routinely by draining the requested batches faster than the
20-second validity window. Always finish with `qualify_s12_pipeline.py`, which
separates that case (`REMEASURE`, exit 2) from a real failure (`FAIL`, exit 1).

The Poisson lookup-table research prototype lives in
`work/experiments/ml/research/` and is deliberately outside the production
import graph; it was evaluated and rejected for S12.

This directory contains the shared execution layer for tracked ML studies.  A
study-specific `study.yaml` remains the scientific source of truth; these
helpers only bind that prescription to site-specific scheduler and environment
details.

## Structure

- `preflight_ml_gpu.py`: generic CUDA and study-contract preflight for any
  tracked ML study.  Also runs a bounded data-loader smoke test when noise
  augmentation is enabled; see
  [Data-Loader Safety and Fail-Fast Checks](#data-loader-safety-and-fail-fast-checks).
- `diagnose_ml_dataloader.py`: standalone data-pipeline diagnostic.  Runs the
  real dataset and noise path with a finite timeout and no training.
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

## Data-Loader Safety and Fail-Fast Checks

### Why JAX-backed noise and forked workers deadlock

JAX initialises a multithreaded backend and is unsafe to fork from.  Importing
`dluxshera.ml` is enough to trigger that initialisation: `utils/noise.py` imports
`components.detectors`, which imports `dLux`, which executes a module-level
`jnp.zeros(...)`.  So the training process is fork-unsafe from import time, well
before any noise is applied.

A study whose noise model executes JAX inside `Dataset.__getitem__`
(`noise_model: shera_observation`) therefore cannot use `fork`-started
DataLoader workers.  Each worker inherits the parent's initialised backend and
deadlocks on its first JAX call: no error, no progress, and the job runs to its
wall-time limit.  This destroyed six 72-hour S12 allocations; see
`docs/dev/lessons_learned.md`.

### Supported worker configurations

The start method is chosen in code, not in `study.yaml`, so frozen study and
artifact identities are unaffected.

| Noise model | `num_workers` | Start method | Status |
|---|---|---|---|
| disabled or `legacy_numpy` | any | PyTorch default | supported; unchanged from S01-S11 |
| `shera_observation` | 0 | n/a, in-process | supported fallback; benchmark on target hardware |
| `shera_observation` | > 0 | `forkserver` (automatic) | **recommended** |
| `shera_observation` | > 0 | `spawn` | supported; equivalent throughput, slower pool startup |
| `shera_observation` | > 0 | `fork` | **rejected** during configuration |

The unsafe combination **fails closed**: requesting `fork` with JAX-backed noise
and `num_workers > 0` raises `UnsafeWorkerStartMethodError` at setup rather than
warning and then stalling. Reproducing the deadlock deliberately requires the
diagnostic's explicit
`--i-understand-this-will-deadlock-reproduce-fork-hang` flag, which is not
reachable from study configuration.

Override with `training.dataloader_multiprocessing_context` only when
diagnosing. `persistent_workers` stays `False`: the dynamic pair sampler is
epoch-dependent, and `set_epoch()` reaches workers only because the dataset is
re-pickled each epoch.

### Bounded waiting

Two independent bounds apply, and they cover different failures:

- **Per-batch** — `DataLoader(timeout=...)`, defaulting to
  `DEFAULT_PER_BATCH_TIMEOUT_SECONDS` (600 s) whenever workers are active.
  Override with `training.dataloader_timeout_seconds`; `0` disables it. This
  applies to *every* fetch including the first, so it does cover a worker that
  wedges before producing anything. PyTorch rejects a nonzero timeout with
  `num_workers=0`, so it is omitted there.
- **Whole-probe** — preflight and the diagnostic run each probe in a supervised child process
  and kills that process *and its worker descendants* on timeout. This is
  necessary because a genuinely deadlocked worker cannot be interrupted from
  inside the process that owns it; an in-process watchdog thread would leave
  orphaned workers behind.

Performance note: `forkserver` re-execs a clean interpreter per worker, so the
first batch costs a few seconds of startup that `fork` does not. That cost is
paid once per epoch and is negligible against epoch wall time.

### Why CUDA availability does not establish pipeline readiness

The original preflight validated CUDA and every artifact hash and still passed
six jobs that could never produce a batch.  CUDA checks the training device;
they say nothing about whether worker processes can hand back data.  Note also
that PyTorch using CUDA does not imply JAX does — in the S12 runs JAX silently
selected its CPU backend.  Both devices are now recorded in
`run_config_resolved.json` under `runtime.dataloader`.

### Required smoke test before long jobs

Preflight runs a bounded data-loader smoke test automatically whenever noise
augmentation is enabled.  It answers "can this pipeline produce batches at all",
not "how fast is it".

Control it through the submit helper with `--dataloader-smoke` /
`--no-dataloader-smoke`, `--dataloader-smoke-batches`, and
`--dataloader-smoke-timeout`, which export `ML_DATALOADER_SMOKE`,
`ML_DATALOADER_SMOKE_BATCHES`, and `ML_DATALOADER_SMOKE_TIMEOUT` into the batch
environment.  Those three keys are allowlisted in `SUBMITTED_ENV_KEYS`; setting
them only in the parent shell does **not** work, because the submitted
environment strips parent-shell `ML_*` variables before applying explicit
values.

For a standalone check on a compute node, with no GPU required:

```bash
PYTHONPATH=src python3 work/experiments/ml/hpc/diagnose_ml_dataloader.py \
  --study work/experiments/ml/s12/study.yaml \
  --experiment-id S12-E01 \
  --run-id S12-E01-R001 \
  --prepared-root "$GATTACA_SCRATCH/prepared/PREP-V4-v1" \
  --split-registry "$GATTACA_SCRATCH/artifacts/v4/SPLIT-V4-ROLE-PRESERVING-v1.json" \
  --scaler "$GATTACA_SCRATCH/artifacts/v4/SCALER-V4-GLOBAL-MAX-ABS-v1.json" \
  --validation-manifest "$GATTACA_SCRATCH/artifacts/S12/validation_pairs/S12-VALIDATION-PAIRS-v1" \
  --noisy-eval-artifact "$GATTACA_SCRATCH/artifacts/S12/S12-PHOTON-NOISE-VALIDATION-v1.json" \
  --artifact-lock "$GATTACA_SCRATCH/artifacts/S12/S12-ARTIFACT-LOCK-v1.json" \
  --batches 60 --timeout-seconds 1800
```

Useful variations: `--num-workers 0` for the in-process fallback,
`--loader validation` for the frozen noisy manifest, and `--with-training-step`
to add a bounded GPU forward/backward check that includes the second-view
noise-consistency term when the experiment defines one.  The diagnostic never
reads the locked test manifest and never writes to run directories.

To demonstrate the deadlock rather than avoid it, combine
`--multiprocessing-context fork` with
`--i-understand-this-will-deadlock-reproduce-fork-hang`.  The probe is expected
to time out; the supervisor kills it and its workers, so nothing is left
running.

#### Sustained throughput

`--batches 60` is the right scale for a capacity decision.  Workers prefetch, so
a short run measures queue drain rather than sustained loading and overstates
throughput by 30-60%.  The report discards the prefetch window
(`warmup_batches_discarded`) and then certifies the result only when it has at
least 10 post-warmup batches spanning at least 20 s:

- `sustained_pairs_per_second` — the number to plan with.
- `sustained_measurement_valid` — **treat throughput as unusable when `false`**.
- `sustained_measurement_note` — says what was insufficient.
- `capacity_projection` — projected epoch minutes and campaign hours from the
  study's own `pairs_per_epoch` and `epochs`, marked `not_defensible` when the
  measurement did not qualify.

### Recognising a job stuck before its first epoch

Training prints a `training start ...` line with the resolved device, JAX
backend, worker count, start method, and per-batch timeout, then a
`first training batch ...` line.

- `training start` present, no `first training batch`, and no `epoch 001/...`
  line → stalled before producing any data.  With a nonzero per-batch timeout
  this now self-terminates with `DataLoader timed out after N seconds` instead
  of running to the wall-clock limit.
- Neither line present → the stall is earlier, in artifact loading or scaler
  fitting.

### Stalled worker vs slow but healthy loading

Both look like "no progress" in the log.  Distinguish by worker CPU:

```bash
squeue --me
srun --jobid <job> --pty top -b -n 1 | head -20
```

- Worker processes present at ~0% CPU, state `S`, and a `first batch` line that
  never arrives → **deadlock**.  Expect the JAX fork RuntimeWarning in stderr.
  Fix the start method; do not wait.
- Workers busy near 100% CPU with batches arriving slowly → **throughput**.
  Compare `sustained_pairs_per_second` from the diagnostic against
  `pairs_per_epoch` and the epoch budget, then request more wall time, split the
  campaign across allocations with `--resume-checkpoint`, or revisit
  `pairs_per_epoch` (which changes the scientific prescription).  Note that for
  JAX-backed noise, raising `num_workers` measured as **no** improvement
  locally — verify on the cluster before relying on it.

The bounded smoke test reports the same distinction in its failure message
rather than blocking indefinitely.

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

Preflight's whole-probe deadline includes zero-worker loading and teardown.
The low-level `smoke_test_dataloader` has only a cooperative overall budget:
a blocked zero-worker fetch cannot observe it. Production zero-worker training
has no batch timeout; use the supervised diagnostic to qualify that fallback.
Default S12 uses four safe workers and a 600-second per-batch timeout.
`sustained_batches` counts post-warmup batches (formerly `sustained_samples`);
`sustained_pairs` counts actual pairs, including a partial final batch.
Local E01 measurements were 41.1 pairs/s with zero workers and 49.4 with four;
worker count alone does not predict cluster speedup.
