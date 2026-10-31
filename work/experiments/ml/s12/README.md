# S12 V4 Photon-Noise Observation Robustness

S12 reuses the S10-E01 clean reference cohort (`S10-E01-R001..R003`) rather
than retraining a separate baseline. S12-E01 trains with dynamic SHERA photon
observation noise and evaluates against a fixed-seed noisy validation
manifest. S12-E02 adds the same dynamic noise plus a modest (weight 0.05)
independent-realization prediction-consistency loss.

All runs keep `evaluate_test: false`. See
`docs/dev/notes/ml_s10_s12_campaign_plan.md` for the full launch prescription.

## Runtime constraint: JAX-backed noise cannot use forked workers

S12 is the first study to set `noise_model: shera_observation`, which executes
JAX inside `Dataset.__getitem__`. JAX is unsafe to fork from, so
`fork`-started DataLoader workers deadlock on their first noise call: no error,
no progress, and the job runs to its wall-time limit.

The training runtime now selects a fork-safe start method (`forkserver`)
automatically whenever JAX-backed noise is enabled. `study.yaml` is unchanged,
so `training_identity` and `S12-ARTIFACT-LOCK-v1` remain valid. See
[the HPC runbook](../hpc/README.md#data-loader-safety-and-fail-fast-checks) for
the mechanism, supported worker configurations, and how to tell a stalled
worker from slow loading.

## Incident status: six runs lost, no usable checkpoints

Six S12 runs under source revision `0f20f63ada40677d12068bfc89d90d6be1bdd942`
each reached the 72-hour limit without completing epoch 0. Representative:
`S12-E01-R001` (job 19830195, timed out 2026-10-20) and `S12-E02-R001`
(job 19830198, timed out 2026-10-26). They wrote `run_manifest.json` and
`run_config_resolved.json` but produced no `history.csv` and no checkpoint, so
**recovery is a restart from epoch 0, not a resume.**

## Recovery procedure (human operator, Gattaca2)

Do not run these from an automated coding task. Do not use `--overwrite`
against the historical run directories without explicit authorization.

### 0. Set and validate the recovery environment

On Gattaca2, set these before running the recovery commands. Choose the
corrected committed revision and deploy its isolated source checkout first.
All commands using relative repository paths must run from that isolated S12
checkout, not the historical `~/dluxshera-sandbox` checkout used by S10/S11.

```bash
export GATTACA_SCRATCH=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-ML

: "${S12_SOURCE_COMMIT:?Set the corrected source commit}"
: "${CUDA_CONDA_PREFIX:?Set the CUDA environment}"
: "${GPU_PARTITION:?Confirm the GPU partition}"
: "${GPU_GRES:?Confirm the GPU resource request}"
: "${REPO_ROOT:?Set the isolated S12 source checkout}"
cd "$REPO_ROOT" || exit 1

export S12_INCIDENT_ROOT="$GATTACA_SCRATCH/incidents/S12"
export S12_RESTART_ROOT="$GATTACA_SCRATCH/runs_s12_restart_${S12_SOURCE_COMMIT:0:12}"
export S12_PERSIST_ROOT="$GATTACA_SCRATCH/persistent_s12_restart_${S12_SOURCE_COMMIT:0:12}"

mkdir -p "$S12_INCIDENT_ROOT" "$S12_RESTART_ROOT" "$S12_PERSIST_ROOT"
```

Keep these values in the operator shell through launch preview/submission.
The nonempty checks prevent an empty path argument, particularly
`--persist-dir`, from resolving to an unintended working directory.

### 1. Preserve the failed-run evidence, and use a fresh output root

The original run directories are non-empty: they hold `run_manifest.json` and
`run_config_resolved.json` but no checkpoint. Training refuses to reuse a
non-empty directory unless given `--resume-checkpoint` (impossible — there is no
checkpoint) or `--overwrite`, which `shutil.rmtree`s the directory and would
destroy the evidence. **Do not use `--overwrite`.**

Instead, give the restart a **separate physical output root** while keeping the
same six canonical logical run IDs. The run path is composed as
`<run-dir>/<STUDY>/<EXPERIMENT>/<RUN>`, so changing only the root is sufficient
and requires no change to run IDs or study configuration:

```bash
# Original, failed; leave contents and permissions unchanged.
export S12_FAILED_ROOT="$GATTACA_SCRATCH/runs"
```

Also archive the Slurm logs, which live outside the run directories:

```bash
export S12_FAILED_ARCHIVE="$S12_INCIDENT_ROOT/failed_runs_0f20f63a"
mkdir -p "$S12_FAILED_ARCHIVE/logs"
cp -p work/experiments/ml/hpc/logs/S12-E0*-R00*.{out,err} "$S12_FAILED_ARCHIVE/logs/" 2>/dev/null || true
tar -cpf "$S12_FAILED_ARCHIVE/failed_S12.tar" -C "$S12_FAILED_ROOT" S12
```

#### Keeping the analysis mirror unambiguous

`campaign_analysis` discovers runs by recursively globbing `run_manifest.json`
and groups them by `run_id`. If both the failed and restarted trees were
mirrored, each canonical ID would appear twice with different
`git.source_commit` values, and because `source_commit` is part of the run
identity tuple this is treated as a **conflict**: a `RuntimeWarning` in normal
mode and a hard `ValueError` under `--strict`. (Verified directly; non-strict
mode does pick the complete restart run, but the warning is noise and strict
mode fails outright.)

After all six restarted runs have passed the completion audit, mirror **only**
their compact finalized analysis artifacts. Run this transfer on the **local
machine**. Define the remote restart path there using the exact value recorded
on Gattaca2; remote shell exports are not inherited locally. Also define the
local canonical mirror destination and the confirmed SSH destination.

```bash
# Set this to the operator-confirmed Gattaca2 SSH destination (user@host).
# Do not use sync_results.py: its only supported site/host is LS6.
: "${GATTACA_SSH:?Set the confirmed Gattaca2 SSH destination}"
: "${S12_RESTART_ROOT:?Set the remote S12 restart root recorded on Gattaca2}"
: "${LOCAL_RESULTS_ROOT:?Set the canonical local analysis mirror}"
mkdir -p "$LOCAL_RESULTS_ROOT/S12"
# Run only after all six restart runs have passed the completion audit.
rsync -av --prune-empty-dirs \
  --include='*/' \
  --include='run_manifest.json' \
  --include='run_config_resolved.json' \
  --include='history.csv' \
  --include='metrics.json' \
  --include='evaluation_predictions.npz' \
  --include='source_snapshot.txt' \
  --exclude='*' \
  "$GATTACA_SSH:$S12_RESTART_ROOT/S12/" "$LOCAL_RESULTS_ROOT/S12/"
```

These rules exclude checkpoints and other training products. The source is
only the fresh restart tree; never sync the historical failed-run tree into
the canonical analysis mirror.

Keep `$S12_FAILED_ARCHIVE` outside any synced results root. It is incident
evidence, not a campaign result. If you ever do need both visible at once, use
non-strict analysis and expect the conflict warning.

### 2. Record source provenance

Deploy into a new versioned source directory, for example
`$GATTACA_SCRATCH/source/s12-<corrected-sha>`, and set `REPO_ROOT` to it.
Do not replace or update the historical source tree: pending S10/S11 jobs may
still start from it. Keep this source tree fixed through all S12 allocations.
Capture the revision explicitly, since archive
deployments have no `.git`:

```bash
--source-commit <corrected-source-commit>
--source-archive-id <archive-id-if-any>
```

If the deployed tree has uncommitted modifications, record that fact and the
diff alongside the launch packet. `run_manifest.json` carries a `git.dirty`
flag, but the diff itself is not captured.

### 3. Confirm artifact identities are unchanged

These keep their existing identities; nothing in this fix alters them:
`PREP-V4-v1`, `SPLIT-V4-ROLE-PRESERVING-v1`, `SCALER-V4-GLOBAL-MAX-ABS-v1`,
`S12-VALIDATION-PAIRS-v1`, `S12-PHOTON-NOISE-VALIDATION-v1`,
`S12-ARTIFACT-LOCK-v1`.

```bash
PYTHONPATH=src python3 work/experiments/ml/materialize_study_artifacts.py validate-lock \
  --study work/experiments/ml/s12/study.yaml \
  --prepared-root "$GATTACA_SCRATCH/prepared/PREP-V4-v1" \
  --split-registry "$GATTACA_SCRATCH/artifacts/v4/SPLIT-V4-ROLE-PRESERVING-v1.json" \
  --scaler "$GATTACA_SCRATCH/artifacts/v4/SCALER-V4-GLOBAL-MAX-ABS-v1.json" \
  --validation-manifest "$GATTACA_SCRATCH/artifacts/S12/validation_pairs/S12-VALIDATION-PAIRS-v1" \
  --test-manifest "$GATTACA_SCRATCH/artifacts/S12/test_pairs/S12-TEST-PAIRS-v1" \
  --artifact-lock "$GATTACA_SCRATCH/artifacts/S12/S12-ARTIFACT-LOCK-v1.json" \
  --noisy-eval-artifact "$GATTACA_SCRATCH/artifacts/S12/S12-PHOTON-NOISE-VALIDATION-v1.json"
```

### 4. Compute-node data-loader smoke test (no GPU needed)

Run the `diagnose_ml_dataloader.py` command from
[the HPC runbook](../hpc/README.md#required-smoke-test-before-long-jobs) on a
compute node, for **both** `S12-E01` and `S12-E02`. E02 generates a second
independent noise realization per sample and is the throughput-critical case.

Use `--batches 60` for a capacity decision. The diagnostic discards the worker
prefetch window and then requires at least 10 post-warmup batches spanning at
least 20 s before it will certify a throughput number. Check
`loader_smoke.result.sustained_measurement_valid`:

**Do not plan capacity against a measurement where `sustained_measurement_valid`
is `false`.** Short runs overstate throughput by 30-60% because they drain a
queue filled during worker startup.

The report includes a `capacity_projection` block computed from the study's own
`pairs_per_epoch` and `epochs`, with `status: "not_defensible"` when the
underlying measurement did not meet the thresholds.

**Gate:** if `projected_campaign_hours` plus GPU-step time does not fit the
Gattaca2 wall-clock limit with margin, do not launch.

**E02 very likely does not fit as a single job.** Local sustained measurements
(Apple Silicon, 160x160, `forkserver`, all `sustained_measurement_valid: true`):

| variant | workers | pairs/s | 600 epochs |
|---|---|---|---|
| E01 | 4 | 49.4 | ~55 h |
| E01 | 8 | 47.4 | ~58 h |
| E02 | 4 | 24.4 | **~112 h** |

Two consequences worth internalising before the cluster run:

- **Raising `num_workers` does not help.** 8 workers measured no better than 4.
  Earlier guidance suggesting otherwise was based on prefetch-contaminated
  numbers and is withdrawn.
- E01 at ~55 h fits 72 h but with modest margin; E02 at ~112 h does not.

These are local numbers; the cluster may differ materially, so measure before
concluding. If E02 does not fit on the cluster either, the options in order of
preference are: a longer-walltime partition; **checkpoint-and-resume across two
or three sequential allocations** (E02 writes `checkpoint_last.pt` every epoch
and `--resume-checkpoint` is supported, with the same run directory and logical ID; see the reproducibility limits below); or escalating
`docs/dev/notes/s12_batched_noise_proposal.md` for scientific review. Launching
E01 while E02 is still under review is reasonable.

### 5. Bounded GPU training smoke test

Add `--with-training-step --device cuda:0 --batches 8` to the diagnostic. For
S12-E02 this runs the **actual** objective, including the second-view
noise-consistency term, not a science-loss-only stand-in. Confirm in the
output that:

- `training_step_smoke.result.exercised_second_view_objective` is `true` for
  E02 (and `false` for E01, which has no consistency term).
- `mean_noise_consistency_loss` is finite and non-null for E02.
- `objective_completeness` describes what was actually exercised.

The step probe runs under the same process-level supervision as the loader
probe, so a wedged GPU step is killed rather than left running.

Then, optionally, shorten the feedback loop with a single bounded run using a
small `epochs` override into a **scratch** run directory under
`$S12_RESTART_ROOT/../smoke`. Never point a smoke run at a canonical run
directory.

### 6. Stage the six production submissions

Only after steps 4 and 5 pass. Preview first; `--plan-preview` never submits.

```bash
python work/experiments/ml/hpc/submit_study_run.py \
  --site gattaca2 \
  --study work/experiments/ml/s12/study.yaml \
  --repo-root "$REPO_ROOT" \
  --conda-prefix "$CUDA_CONDA_PREFIX" \
  --prepared-root "$GATTACA_SCRATCH/prepared/PREP-V4-v1" \
  --split-registry "$GATTACA_SCRATCH/artifacts/v4/SPLIT-V4-ROLE-PRESERVING-v1.json" \
  --scaler "$GATTACA_SCRATCH/artifacts/v4/SCALER-V4-GLOBAL-MAX-ABS-v1.json" \
  --validation-manifest "$GATTACA_SCRATCH/artifacts/S12/validation_pairs/S12-VALIDATION-PAIRS-v1" \
  --test-manifest "$GATTACA_SCRATCH/artifacts/S12/test_pairs/S12-TEST-PAIRS-v1" \
  --artifact-lock "$GATTACA_SCRATCH/artifacts/S12/S12-ARTIFACT-LOCK-v1.json" \
  --noisy-eval-artifact "$GATTACA_SCRATCH/artifacts/S12/S12-PHOTON-NOISE-VALIDATION-v1.json" \
  --run-dir "$S12_RESTART_ROOT" \
  --persist-dir "$S12_PERSIST_ROOT" \
  --launch-packet "$S12_RESTART_ROOT/launches" \
  --source-commit "$S12_SOURCE_COMMIT" \
  --dataloader-smoke --dataloader-smoke-batches 4 --dataloader-smoke-timeout 900 \
  --extra-sbatch-arg="--partition=$GPU_PARTITION" \
  --extra-sbatch-arg="--gres=$GPU_GRES" \
  --extra-sbatch-arg=--time=3-00:00:00 \
  --extra-sbatch-arg=--cpus-per-task=8 \
  --extra-sbatch-arg=--mem=64G \
  --plan-preview
```

Note `--run-dir "$S12_RESTART_ROOT"` — the fresh root from step 1, **not** the
original `$GATTACA_SCRATCH/runs`. Verify in the preview output that `ML_RUN_DIR`
points under the restart root and that `ML_OVERWRITE` is absent.

The `--dataloader-smoke*` flags are exported into the batch environment and make
preflight prove the pipeline can produce batches before training starts. These
keys are allowlisted in `SUBMITTED_ENV_KEYS`; without that they would be
silently stripped along with other parent-shell `ML_*` variables.

Replace `--plan-preview` with `--submit-plan` once the preview is inspected.

Use the canonical run identities `S12-E01-R001..R003` and
`S12-E02-R001..R003` so results stay unambiguous. The restart supersedes the
failed attempts under the same logical names while living in a different
physical root, which is what keeps the originals intact. Avoid inventing
`-R004`-style IDs, which would imply six extra scientific replicates that do
not exist.

The effective request is one node/task, eight CPUs, 64 GB RAM, 72 hours,
and the operator-selected GPU partition/GRES. Set GRES to the site's confirmed
single A100 80 GB resource spelling; verify GPU type and count before submission.

### Sequential allocations and reproducibility

Resume only from `checkpoint_last.pt` in the same nested restart run directory,
using `train_from_study.py --resume-checkpoint <that-path>` with the same study,
experiment, run ID, source, and artifact arguments. Do not overwrite or create
new replicate IDs. Arrange sequential jobs; concurrent writers are unsafe.
Checkpoints restore model, optimizer, scheduler and early-stopping state and
validate training identity. Epoch-dependent pairs and noise remain keyed by
epoch/seed. Global Python/NumPy/Torch RNG states are not checkpointed, so resume
is not guaranteed bitwise equivalent to uninterrupted execution. A killed
partial epoch is replayed from the last completed checkpoint. Preserve and
record allocation boundaries and validate resume on a scratch run first.
