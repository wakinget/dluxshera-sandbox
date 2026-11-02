# S12 pipeline optimization review (2026-11-01)

Status: two opt-in candidates, local validation only. Production remains on the
reference implementation until the V4 compute-node gates below pass. No study
prescription or frozen artifact was changed. No locked test evaluation was run.

Three noise execution strategies are now supported, selected explicitly:

| Strategy | Realization | Default | Frozen validation |
|---|---|---|---|
| `reference` | prescribed JAX stream | **yes** | yes |
| `batched` | bitwise identical to reference | no | yes |
| `numpy` | same probability model, different RNG stream | no | **no — maps to `reference`** |

## What is established versus hypothesized

Keeping these apart matters: several numbers below are local-only and a few are
cluster facts.

**Established facts**

- The original S12 failure was a JAX-in-forked-DataLoader-worker deadlock; six
  72-hour allocations produced no epoch 0. Fixed in `2e44b68`.
- Reference data loading succeeds on Gattaca2 CPU node `cn029` for both
  experiments, after artifact validation.
- Measured Gattaca2 reference throughput: **E01 21.64 pairs/s (126.2 h
  projected), E02 21.28 pairs/s (128.3 h)**, loading only.
- Batched JAX is locally faster and bitwise exact against the reference on the
  tested JAX 0.4.30 CPU backend; see the fixture and resident tables below.
- Exact-equivalence tests cover sides, seeds, batch partitions, E02 views,
  epochs, reverse pairs, every dataset field, partial batches, forkserver and
  spawn workers, and every fallback configuration.
- Deterministic per-record NumPy sampling is 4.73× faster than batched JAX in
  isolated noise generation, and the per-record seed contract costs ~4% against
  an unconstrained shared-RNG ceiling. It reproduces Poisson moments, Fano
  factor, zero probability, tails, and goodness-of-fit at λ from 0.1 to 9,000.
  See [the NumPy investigation](#numpy-photon-noise-investigation).

**Hypotheses still requiring Gattaca2 V4 access**

- Whether the Gaussian branch is ever selected on V4 (local V3 says no; see
  below).
- Whether random shard I/O, not noise generation, is the remaining bottleneck —
  the clean baseline in step 4 of the qualification isolates this. Local
  fixtures cannot answer it: their shards are small and warm.
- Whether batched or NumPy throughput on V4 clears the 70 pairs/s engineering
  target, and by how much NumPy's local 14–18× loader advantage shrinks once
  real GPFS random reads are in the measurement.
- Whether offline preprocessing would improve GPU utilization enough to justify
  its storage and orchestration cost.

**Settled by this pass, and not favorable**

- Precomputed Poisson lookup tables are not a viable S12 optimization. See
  [the lookup investigation](#research-precomputed-poisson-lookup-tables).

## Detector assumption

S12 treats prepared-image values as expected counts with a **unity conversion,
1 DN = 1 electron**. This is an accepted modeling assumption for the present
study, and it is what makes `jr.poisson(key, image)` the correct sampler: the
Poisson rate is the pixel value itself with no intervening gain. No gain
calibration, detector conversion stage, or detector-model complexity was
introduced. `detector_model` is `null` in both S12 experiments, and read noise
and dark current are disabled, so no `DetectorSpec` is consulted at all.

## Noise-regime characterization: Poisson versus Gaussian

The shared observation-noise utility selects its branch on the **whole-image
minimum**, not the maximum and not per pixel
(`src/dluxshera/utils/noise.py:147`):

```python
if jnp.min(image) > bright_threshold:   # Gaussian approximation
else:                                   # exact Poisson
```

S12's `bright_threshold` is 100.0 in both the training and validation noise
blocks. Because the predicate is the minimum, an image is only treated as
bright if *every* pixel exceeds 100 — a single faint outskirt pixel sends the
whole image to the exact Poisson branch. Reading "fraction of pixels above 100"
as if it were the branch condition gets this backwards, which is why
`analyze_image_statistics.py` reports both quantities separately.

`work/experiments/ml/hpc/analyze_image_statistics.py` is a read-only diagnostic
for this. Measured locally on 512 strided samples of the real
`PREP-V3-nominal-v1` catalog (160×160 float32, 22,990 samples):

| Quantity | Value |
|---|---:|
| Images selecting exact Poisson | **512 / 512** |
| Images selecting Gaussian | 0 |
| Per-image minimum: min / median / max | 1.19 / 2.16 / 3.56 |
| Per-image maximum: median | 6,673 |
| Pooled pixel median | 61.7 |
| Pooled fraction of pixels above 100 | 0.369 |
| Fraction of images with every pixel above 100 | 0.000 |
| Negative pixels present | no |

So on V3 the Gaussian branch is **never** selected, and it is not close: the
largest per-image minimum observed is 3.56 against a threshold of 100. The
expectation for V4's 50 ms exposures is the same, since the mechanism is the
faint PSF outskirts rather than anything V3-specific. **This is a V3
measurement, not a V4 result**; step 3 of the qualification runs the identical
command on PREP-V4-v1 and the frozen validation observation images.

If V4 confirms Poisson-exclusive selection, a Poisson-only execution path
becomes *conceivable* — but note the branch check is a single `np.min` per
image, already negligible against the draw itself, so specializing it would buy
essentially nothing. The threshold and the shared utility were not changed.

## Findings and limits

The local checkout contains V3 prepared images, but PREP-V4-v1 and the S12 frozen
artifacts were not found. Consequently this report does **not** establish the
root cause of the 21 pairs/s Gattaca2 result or certify a 72-hour campaign.

A cProfile run of 64 ordered 160×160 fixture pairs took 0.706 s: 0.678 s
(96%) was inside `apply_pair_noise`, 0.648 s inside SHERA observation noise,
0.502 s in Poisson and 0.103 s in normal sampling. These are nested cumulative
measurements, not additive stage totals. Approximately 0.081 s was compilation;
this first profile did not fully warm both branches. Pair sampling, shard reads,
scaling and tensor construction together accounted for the remaining 0.028 s.
Fixture shards are small and warm; this does not characterize GPFS random I/O.

A separate resident-image benchmark used the first 32 pairs of the real local
PREP-V3-nominal-v1 catalog, 160×160 float32, JAX 0.4.30 CPU, eight warm repetitions:

| One noise view / batch of 32 | Reference | Batched |
|---|---:|---:|
| First call, seconds | 1.735 | 0.985 |
| Mean warm batch, seconds | 1.300 | 0.192 |
| Exact output equality | yes | yes |

All those observations selected Poisson. The 6.76× isolated improvement includes
key construction, dispatch and host conversion; it is not a measured dispatch-only
speedup. Other local work was active, so these are exploratory measurements.

A later, quieter repetition of the same measurement through the **shipped
functions** on the same 32 resident V3 images gave reference 0.7134 s, batched
0.1481 s, and NumPy per-record 0.0313 s per batch. The reference/batched ratio
there is 4.8× rather than 6.76×, which is the spread to expect from an
unreserved machine; treat all of these as exploratory single-machine numbers and
the cluster measurement as authoritative.

An alternative bottleneck was identified but not changed: every `sample_pair`
validates family names by scanning the catalog. On the actual 22,990-row V3
catalog this costs 0.524 ms/pair (100 repetitions). It may matter more on V4.
The sampler already caches eligible indices and buckets. Forward/reverse pairs
repeat sampling and image reads; this patch deliberately retains their sequence.
No evidence yet justifies a storage layout change, larger shard cache, or eight
workers. The clean baseline and stage profiler below distinguish those costs.
Cold OS cache eviction was not attempted; reruns characterize warmed caches only.

## Implementation and scientific contract

`DynamicPairDataset` and `PairManifestDataset` implement PyTorch batch fetching
through `__getitems__`. Individual `__getitem__` calls remain the reference.
Workers resolve the same records and offsets, read count-space images, batch
noise, then call the existing scaler and tensor/metadata assembly. Default
collation remains unchanged. E02 runs the same operation again with offset
1,000,000,007; reverse ordering happens before the observation B role is selected.

`noise_implementation="batched"` and `"numpy"` are opt-in; the default is
`reference`. `DLUXSHERA_NOISE_IMPLEMENTATION` supplies the runtime default. The
submission CLI accepts `--noise-implementation`, explicitly exports it, and
training records it in runtime provenance (including checkpoints). It is absent
from scientific noise/training identities. Existing noiseless datasets take
their original path.

Under `numpy`, `__getitem__` delegates to `__getitems__` so a single-item fetch
cannot produce a different realization from a batch fetch. With `reference` and
`batched` that delegation is unnecessary because the two are bitwise identical;
with `numpy` its absence would make the realization depend on whether PyTorch
chose the batch-fetch path, which is exactly the kind of hidden dependence the
per-record seeding exists to eliminate.

Runtime provenance records the requested strategy, the **effective** training
strategy (a requested `numpy` degrades to `reference` for configurations it does
not cover), the validation strategy, the BitGenerator, and the NumPy and JAX
versions. Resume rejects any mismatch, so a campaign never mixes RNG methods.

The batched CPU JIT vmaps the original split-key Poisson draw, using unchanged
`make_subkey` and `pair_noise_side_seeds`. Each image is independently checked
against the strict minimum-pixel bright threshold after clipping. Bright images
use the original eager Gaussian path to avoid fused-arithmetic rounding changes.
Read/dark/disabled/photon-disabled/legacy configurations use the reference path.
This is intentional exact fallback, not an approximate replacement noise model.

Mixed-brightness groups are padded to the requested batch width to avoid a new
compilation for every possible bright-image count. Padding results are discarded.
Different image shapes and full/partial batch widths specialize separately. The
compiled callable is created once per process; each new epoch's worker pool
still pays initialization/compilation costs. No persistent workers were enabled.
JAX's in-process executable cache follows shape diversity; ordinary S12 has one
image shape and fixed batches. There is no persistent disk compilation cache.

At 32×160×160, each float32 image stack is 3.125 MiB. Clean A/B, output views,
stacked noise inputs, JAX temporaries and prefetched tensor batches coexist; E02
requires more. This is bounded by batch/prefetch size rather than dataset size.
Reported RSS is process high-water RSS (bytes on macOS, KiB on Linux), not summed
live worker memory. Cluster aggregate memory still needs measurement.

The fork rejection, cleared forkserver preload, nonpersistent workers, mmap
serialization and per-batch/process deadlines remain in place. Compilation must
finish within the existing deadline; do not remove supervision to obtain a pass.

## Reproducible checks

`diagnose_ml_dataloader.py` now accepts:

- `--noise-implementation reference|batched|numpy` for full loader comparison.
  On `--loader validation` a `numpy` selection resolves to `reference`, matching
  training's frozen-validation isolation; the report's
  `identity.noise_implementation` shows what was actually used.
- `--clean-baseline` to disable augmentation after validating original artifacts.
- `--profile` for a separately warmed, single-process cProfile breakdown and
  resident-image timing of **all three** strategies. This happens **after** the
  uninstrumented throughput measurement. Cumulative profile times overlap and
  must not be summed.
- `--numpy-shared-rng-ceiling` to add the shared-RNG microbenchmark to the
  profile, reported with `production_candidate: false` and a `NONPRODUCTION`
  suffix. It is a performance ceiling, not a selectable strategy.
- `--verify-equivalence` to compare every tensor and metadata field exactly on
  the requested ordered records, including validation records. No files are
  rewritten. This is a **bitwise** check, so it rejects `numpy` with an
  explanatory error rather than reporting an expected non-exact result; use the
  statistical tests in `tests/ml/test_numpy_noise.py` for that pathway.

Existing first-batch, actual pair counts, warmup, sustained duration and validity
gates remain. Use `--warmup-batches 8` even with zero workers to remove compilation
from steady-state timing. If the span is below 20 seconds the report now supplies
`recommended_batches_for_valid_measurement`, computed from the observed rate with
a 25% margin — re-run with that value. A short span means the loader was fast,
not that it is unusable, which matters because NumPy routinely triggers it.
Use the same experiment, epoch, workers, shapes and artifact paths for comparisons.
The profiler's first in-process call can compile even after loader workers warmed;
resident timing follows that profile and is not an independent cold compile test.

- `--equivalence-full-manifest` compares every record in the dataset rather than
  `--batches` worth, which is what certifies the complete 1,024-record frozen
  manifest; `--equivalence-batch-size N` (repeatable) repeats the comparison
  under another partition width.

The equivalence comparison is **count-driven, not `zip`-driven**. It requires
exact expected and actual record counts, complete coverage of every requested
index, no missing or duplicated records, identical field sets, matching record
identity per position, and bitwise-equal tensors (shape, dtype, and values) and
metadata. A truncated, padded, reordered, or non-list optimized batch fails
explicitly; previously a short result was silently compared as a prefix and
reported exact. Each partition reports expected/compared counts, batch count,
final-batch width and partiality, and any missing/duplicated/mismatched records.
Partial sampling is labelled as such, so it cannot be read as complete-manifest
coverage. The command exits nonzero unless the result is exact.

This comparison is **in-process** and says nothing about multiprocessing. Worker
equivalence under forkserver and spawn is covered separately by
`tests/ml/test_batched_noise.py::test_dataset_exact`.

`qualify_s12_pipeline.py` reduces the reports written by those commands to one
decision, rerunning nothing. It exists because a zero exit status from the
diagnostic does not mean the measurement it produced is usable: a run can
succeed while reporting `sustained_measurement_valid: false`. It distinguishes
`PASS` (exit 0), `REMEASURE` (exit 2 — correct implementation, measurement
window too short, with the larger `--batches` value to use), and `FAIL` (exit 1
— a failed check, or a missing, malformed, or stale report). Operational
throughput bands are reported separately from scientific correctness, so a slow
but correct implementation passes while still being prevented from claiming the
72-hour campaign fits.

`analyze_image_statistics.py` reports count statistics and SHERA noise-branch
selection for a prepared dataset and, optionally, the frozen validation
observation images. It is read-only, strides its sample across the catalog so a
shard-ordered dataset does not yield a shard-local answer, and modifies no
threshold or artifact.

`benchmark_s12_fixture.py` is a reproducible 160×160 synthetic-fixture comparison
of clean/reference/batched/numpy E01-like and E02-like datasets with zero/four
workers. It requires repository tests on PYTHONPATH and writes only temporary
shards. It is useful for regression engineering, never V4 capacity certification.

Tests cover exact mixed bright/Poisson/clipped outputs, sides, seeds, batch
partitions, E02 views, epochs, reverse pairs, every dataset field, partial batches,
forkserver/spawn, manifest replay, fallback configurations and separate moment
checks for Poisson/Gaussian noise. They additionally cover the skipped-preflight
path (noiseless study, explicit disable, both implementations, environment and
explicit precedence), truncated/duplicated/corrupted equivalence results, the
noise-branch predicate, and the lookup prototype's conclusions.

`tests/ml/test_numpy_noise.py` covers the NumPy strategy on its own terms:
Poisson moments/Fano/P(0) at eight λ from 0.1 to 9,000, chi-square
goodness-of-fit and tail probabilities at five λ, a KS comparison against direct
JAX samples, spatial lag correlations, side-offset and E02-view independence,
seed distinctness over 6,000 logical draws, reproducibility, independence from
batch width (five widths) and batch composition, real multiprocessing across
three worker/start-method combinations, single-versus-batch-fetch agreement,
epoch freshness, reverse-pair semantics, clean A side, negative-value policy,
the bright-image branch including threshold equality, bitwise-exact fallback for
four unsupported configurations, provenance content, resume rejection in all
four switch directions, and the frozen-validation isolation verified bitwise
against the JAX reference.

Exact equality for `batched` is established only on the tested JAX/backend
combination, and NumPy determinism only on the tested NumPy build; recheck both
on the cluster. Actual frozen S12 recipe verification remains an operator gate,
not a claim based on synthetic manifests.

## Gattaca2 CPU qualification procedure

Run this only after review, commit, and push. The operator supplies two
site-specific values: the reviewed full commit SHA, and the GPU
partition/GRES spellings in step 7. Nothing here writes to
`~/dluxshera-sandbox`'s working tree, the previous isolated checkout
`s12-2e44b6878861`, the failed S12 run outputs, S10/S11 jobs, or any frozen
artifact.

**Required qualification** is steps 0–6: Git deployment (login node, step 0),
environment setup, artifact identity, image statistics, the eight throughput
conditions, frozen-validation equivalence, then cluster regression tests and the
qualification gate. Steps 1–6 run inside **one two-hour CPU allocation**; step 0
needs none.

**Step 7** is the production decision, launch preview, and the single E01
submission. **Step 8 is optional investigation** — stage profiling, the
shared-RNG ceiling, alternative worker counts — and should be run only if step 6
leaves a question open. Do not profile by default: it costs compute-node time
and answers nothing the primary conditions have not already answered.

The two-hour budget replaces the earlier one hour, which did not reliably cover
reference-JAX loading at ~21 pairs/s plus two complete 1,024-record validation
comparisons. If the compute partition caps wall time below two hours, split
steps 2–5 and step 6 across two allocations; the reports persist in
`$S12_QUAL_DIR`, so the gate can run in a later session.

### Step 0 — Fetch and verify the reviewed revision (login node)

Git preparation happens on the **login node**, before any allocation, so no
allocated time is spent on it. The reviewed commit is not assumed to be present:
a freshly pushed revision will not be in the cluster's object database until
fetched.

```bash
# Fresh Gattaca2 login shell. No allocation yet; this is all cheap Git work.
set -o pipefail
export GATTACA_SCRATCH=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-ML
export HIST_REPO="$HOME/dluxshera-sandbox"
export S12_SOURCE_COMMIT=REPLACE_WITH_REVIEWED_FULL_SHA

# Fetch the branch without touching the historical checkout's working tree.
# `fetch` only updates refs and objects; no checkout, no reset, no merge.
git -C "$HIST_REPO" fetch origin feature/ml-inverse-model

# The reviewed SHA must now exist as a commit object. Do NOT substitute
# origin/feature/ml-inverse-model: the branch tip may have advanced past the
# revision that was actually reviewed.
git -C "$HIST_REPO" cat-file -e "${S12_SOURCE_COMMIT}^{commit}" || {
  echo "FATAL: $S12_SOURCE_COMMIT not found after fetch. Confirm it was pushed." >&2
  exit 1; }
export S12_SOURCE_COMMIT="$(git -C "$HIST_REPO" rev-parse "${S12_SOURCE_COMMIT}^{commit}")"
echo "reviewed commit: $S12_SOURCE_COMMIT"

# Report whether the reviewed commit is the branch tip, without requiring it.
git -C "$HIST_REPO" rev-parse origin/feature/ml-inverse-model
git -C "$HIST_REPO" merge-base --is-ancestor \
  "$S12_SOURCE_COMMIT" origin/feature/ml-inverse-model \
  && echo "reviewed commit is on origin/feature/ml-inverse-model" \
  || echo "WARNING: reviewed commit is not an ancestor of the remote branch tip"
```

Create the isolated detached worktree. The target directory name carries the
revision, so each qualification gets its own source tree and none can overwrite
another. An existing target is **verified, not replaced**:

```bash
export REPO_ROOT="$GATTACA_SCRATCH/source/s12-${S12_SOURCE_COMMIT:0:12}"

if [ -d "$REPO_ROOT" ]; then
  # Reuse only if it is already at the reviewed revision. Never overwrite.
  actual="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo MISSING)"
  if [ "$actual" != "$S12_SOURCE_COMMIT" ]; then
    echo "FATAL: $REPO_ROOT exists at $actual, not $S12_SOURCE_COMMIT." >&2
    echo "Remove it deliberately or choose another path; not overwriting." >&2
    exit 1
  fi
  echo "reusing existing worktree at the reviewed revision"
else
  mkdir -p "$GATTACA_SCRATCH/source"
  git -C "$HIST_REPO" worktree add --detach "$REPO_ROOT" "$S12_SOURCE_COMMIT"
fi

# Verify source identity before spending any allocated time on it.
test "$(git -C "$REPO_ROOT" rev-parse HEAD)" = "$S12_SOURCE_COMMIT" \
  || { echo "FATAL: worktree HEAD mismatch" >&2; exit 1; }
git -C "$REPO_ROOT" status --porcelain | head   # must be empty: a clean checkout
test -z "$(git -C "$REPO_ROOT" status --porcelain)" \
  || { echo "FATAL: worktree is dirty; qualify only a pristine checkout" >&2; exit 1; }
# The S12 prescription must match the reviewed revision exactly.
git -C "$REPO_ROOT" diff --quiet "$S12_SOURCE_COMMIT" -- work/experiments/ml/s12/study.yaml \
  && echo "study.yaml matches the reviewed commit"
echo "deployed: $REPO_ROOT @ $S12_SOURCE_COMMIT"
```

The historical checkout at `$HIST_REPO` keeps whatever branch and working tree
it had; `fetch` and `worktree add` do not alter it. The prior S12 worktree
`s12-2e44b6878861` is untouched.

### Step 1 — Allocate and initialize the environment

Every Python command from here runs **inside** the allocation: the diagnostics
import JAX and read real V4 batches, neither of which belongs on a login node.
Two hours covers the required steps with margin for slow reference-JAX loading
and two full 1,024-record validation comparisons; the earlier one-hour budget
did not. `exit` at the end releases the node.

```bash
# Still on the login node: carry the verified values into the allocation.
srun --account=shera_hpc --partition=compute \
  --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --mem=64G --time=02:00:00 --pty bash -l
```

```bash
# Inside the allocation. A fresh login shell does not inherit the exports
# above, so redefine everything rather than assuming it is present.
set -o pipefail
echo "node=$(hostname) job=${SLURM_JOB_ID:-NONE}"
test -n "${SLURM_JOB_ID:-}" \
  || { echo "FATAL: not inside an allocation; do not run diagnostics here" >&2; exit 1; }

export GATTACA_SCRATCH=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-ML
export S12_SOURCE_COMMIT=REPLACE_WITH_REVIEWED_FULL_SHA
export REPO_ROOT="$GATTACA_SCRATCH/source/s12-${S12_SOURCE_COMMIT:0:12}"
test "$(git -C "$REPO_ROOT" rev-parse HEAD)" = "$S12_SOURCE_COMMIT" \
  || { echo "FATAL: $REPO_ROOT is not at the reviewed revision; rerun step 0" >&2; exit 1; }

export CUDA_CONDA_PREFIX=/gpfs/scratch/shera_hpc/dmckeith/conda/envs/dluxshera-ml-py311
# Source the site profile script directly. `conda info --base` requires conda
# to already be on PATH, which it is not in a fresh non-interactive shell.
source /cm/shared/apps/miniforge/etc/profile.d/conda.sh
conda activate "$CUDA_CONDA_PREFIX"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# Stamped into every diagnostic report so the qualification gate can reject a
# stale report produced by a different revision.
export DLUXSHERA_SOURCE_COMMIT="$S12_SOURCE_COMMIT"
python -c 'import jax, numpy, torch; print("jax", jax.__version__, jax.default_backend(),
 "numpy", numpy.__version__, "torch", torch.__version__)'
```

Separate paths for source, run outputs, logs, launch packets, persistent
artifacts, and qualification reports. All are outside `$REPO_ROOT`, so the
versioned source checkout stays pristine and a later `git status` on it is
meaningful. All carry the revision, so none can collide with the original failed
attempts:

```bash
export S12_TAG="s12-${S12_SOURCE_COMMIT:0:12}"
export S12_RESTART_ROOT="$GATTACA_SCRATCH/runs-$S12_TAG"      # training outputs
export S12_LOG_ROOT="$GATTACA_SCRATCH/logs-$S12_TAG"          # Slurm stdout/stderr
export S12_LAUNCH_ROOT="$GATTACA_SCRATCH/launches-$S12_TAG"   # launch packets
export S12_PERSIST_ROOT="$GATTACA_SCRATCH/persist-$S12_TAG"   # persisted artifacts
export S12_QUAL_DIR="$GATTACA_SCRATCH/qualification-$S12_TAG" # CPU reports
mkdir -p "$S12_RESTART_ROOT" "$S12_LOG_ROOT" "$S12_LAUNCH_ROOT" \
         "$S12_PERSIST_ROOT" "$S12_QUAL_DIR"

# Refuse to reuse the original failed run root.
case "$S12_RESTART_ROOT" in
  */runs-s12-2e44b6878861|*/runs) echo "FATAL: would reuse a prior run root" >&2; exit 1;;
esac

common=(--study work/experiments/ml/s12/study.yaml
 --prepared-root "$GATTACA_SCRATCH/prepared/PREP-V4-v1"
 --split-registry "$GATTACA_SCRATCH/artifacts/v4/SPLIT-V4-ROLE-PRESERVING-v1.json"
 --scaler "$GATTACA_SCRATCH/artifacts/v4/SCALER-V4-GLOBAL-MAX-ABS-v1.json"
 --validation-manifest "$GATTACA_SCRATCH/artifacts/S12/validation_pairs/S12-VALIDATION-PAIRS-v1"
 --artifact-lock "$GATTACA_SCRATCH/artifacts/S12/S12-ARTIFACT-LOCK-v1.json"
 --noisy-eval-artifact "$GATTACA_SCRATCH/artifacts/S12/S12-PHOTON-NOISE-VALIDATION-v1.json")
```

### Step 2 — Scientific validation (experiment-aware)

This gate is **fail-closed**: a validation failure stops the procedure before
any throughput measurement or submission. Do not replace the exit-status checks
with a message-and-continue idiom.

```bash
for exp in S12-E01 S12-E02; do
 if ! python work/experiments/ml/hpc/diagnose_ml_dataloader.py "${common[@]}" \
   --experiment-id "$exp" --batches 1 --num-workers 0 --timeout-seconds 600 \
   > "$S12_QUAL_DIR/$exp-identity.json"; then
  echo "FATAL: artifact validation failed for $exp; see" \
   "$S12_QUAL_DIR/$exp-identity.json" >&2
  return 1 2>/dev/null || exit 1
 fi
done

# Exit status alone is not sufficient: require identity.status == ok explicitly.
# Step 6's qualification gate rechecks this; doing it here stops the procedure
# before spending ~an hour of allocated time on throughput measurement.
python - <<'EOF' || { echo "FATAL: identity gate not satisfied" >&2; return 1 2>/dev/null || exit 1; }
import json, os, pathlib, sys
root = pathlib.Path(os.environ["S12_QUAL_DIR"])
failed = []
for exp in ("S12-E01", "S12-E02"):
    identity = json.loads((root / f"{exp}-identity.json").read_text()).get("identity", {})
    status = identity.get("status")
    print(f"{exp}: identity.status={status}")
    if status != "ok":
        failed.append((exp, status))
if failed:
    print(f"FATAL: experiment-aware artifact validation failed: {failed}", file=sys.stderr)
    sys.exit(1)
print("artifact validation ok for both experiments")
EOF
```

Every later step assumes this gate passed. If you are running the blocks
individually rather than as one script, confirm `artifact validation ok for both
experiments` appeared before continuing.

The diagnostic's supervised `describe` calls the **experiment-aware**
`load_study_contract_artifacts(... experiment_id=..., config=...)` before any
batch is fetched, covering the prepared V4 dataset, split registry, frozen
intensity scaler, frozen validation manifest, frozen noisy-validation recipe,
and the S12 artifact lock. Both experiments must report `identity.status=ok`.
Do **not** use the generic `validate-lock` CLI: its known experiment-context
limitation is uncorrected and untested for S12. The locked test manifest is
never loaded.

### Step 3 — Image statistics and noise-branch verification

```bash
python work/experiments/ml/hpc/analyze_image_statistics.py \
 --prepared-root "$GATTACA_SCRATCH/prepared/PREP-V4-v1" --samples 512 \
 --validation-manifest "$GATTACA_SCRATCH/artifacts/S12/validation_pairs/S12-VALIDATION-PAIRS-v1" \
 --split-registry "$GATTACA_SCRATCH/artifacts/v4/SPLIT-V4-ROLE-PRESERVING-v1.json" \
 > "$S12_QUAL_DIR/v4-image-statistics.json"
```

Read-only; writes nothing to the prepared dataset. Record
`fraction_selecting_poisson_branch` and `poisson_branch_exclusive` for both the
catalog sample and the frozen validation observation images. This answers
production question 3 directly.

### Step 4 — Throughput comparison (eight conditions)

Four conditions per experiment: clean baseline, reference JAX, batched JAX, and
deterministically seeded NumPy. All use the prescribed four workers, the same
real V4 dataset, the same warmup policy, and the same measurement methodology.

```bash
for exp in S12-E01 S12-E02; do
 for impl in reference batched numpy; do
  python work/experiments/ml/hpc/diagnose_ml_dataloader.py "${common[@]}" \
   --experiment-id "$exp" --noise-implementation "$impl" --num-workers 4 \
   --batches 80 --warmup-batches 8 --timeout-seconds 1800 \
   > "$S12_QUAL_DIR/$exp-$impl.json"
 done
 python work/experiments/ml/hpc/diagnose_ml_dataloader.py "${common[@]}" \
  --experiment-id "$exp" --clean-baseline --num-workers 4 --batches 80 \
  --warmup-batches 8 --timeout-seconds 1800 \
  > "$S12_QUAL_DIR/$exp-clean.json"
done
```

No profiling here: `--profile` and `--numpy-shared-rng-ceiling` are step 8,
conditional on step 6 showing an unresolved bottleneck. They cost compute-node
time and answer nothing these eight conditions do not.

Keep the primary four-worker configuration; the study's `num_workers: 4` is part
of `training_identity` and must not change. Start at 80 batches. If
`sustained_measurement_valid` is false, read
`recommended_batches_for_valid_measurement` from the same report and re-run only
that condition with that `--batches` value — a short span means the loader was
**fast**, not that it is unusable, and NumPy is expected to trigger this. Step 6
reports exactly which conditions need this and with what value, so you do not
have to inspect the JSON by hand. Do not extend a benchmark past what the
20-second validity gate requires.

`--warmup-batches 8` removes compilation from steady-state timing. First-batch
latency in each report is the repeated per-epoch worker-startup cost; it recurs
600 times and is additional to steady-state loading. Do not compare an invalid
short span against a valid sustained run.

The `numpy` condition keeps the frozen validation loader on JAX automatically:
`--noise-implementation numpy --loader validation` resolves to `reference`, and
the report's `identity.noise_implementation` shows the strategy actually used.

### Step 5 — Complete frozen-validation equivalence (all 1,024 records)

```bash
for exp in S12-E01 S12-E02; do
 python work/experiments/ml/hpc/diagnose_ml_dataloader.py "${common[@]}" \
  --experiment-id "$exp" --loader validation --noise-implementation batched \
  --num-workers 0 --batches 4 --verify-equivalence \
  --equivalence-full-manifest --equivalence-batch-size 17 \
  --timeout-seconds 3600 \
  > "$S12_QUAL_DIR/$exp-validation-exact.json"
 python - "$S12_QUAL_DIR/$exp-validation-exact.json" <<'EOF'
import json, sys
d = json.loads(open(sys.argv[1]).read())["equivalence"]
assert d["status"] == "ok", d
r = d["result"]
print(f"{sys.argv[1]}: exact={r['exact']} records={r['records']}/"
      f"{r['dataset_records_available']} complete={r['complete_manifest_compared']}")
assert r["records"] == 1024 and r["complete_manifest_compared"], r
for part in r["partitions"]:
    print(f"  batch_size={part['batch_size']} batches={part['batches']} "
          f"compared={part['compared_records']} final={part['final_batch_records']} "
          f"partial={part['final_batch_is_partial']} exact={part['exact']} "
          f"missing={part['missing_indices']} dup={part['duplicated_indices']} "
          f"mismatched={part['mismatched_field_count']}")
    assert part["exact"] and part["compared_records"] == 1024
EOF
done
```

This must report `records=1024`, `complete_manifest_compared=true`, and
`exact=true` for **both** partitions: 32 batches of 32 divide the manifest
evenly, so only the indivisible width 17 (61 batches, final batch of 7)
exercises a partial final batch. The comparison requires exact record counts,
complete index coverage, no missing or duplicated records, identical field sets,
and bitwise-equal tensors and metadata; a short or reordered optimized batch
fails rather than comparing a prefix. The exit code is nonzero unless the result
is exact.

This is **in-process** equivalence on identical records. It is deliberately
separate from multiprocessing regression testing: forkserver and spawn worker
equivalence is covered by `tests/ml/test_batched_noise.py::test_dataset_exact`,
not by this command. Nothing is regenerated and the locked test set is not read.

### Step 6 — Cluster regression tests and the qualification gate

The NumPy determinism claim is scoped to a NumPy version and build, and the
batched bitwise-equality claim to a JAX build, so both are re-established on the
cluster rather than inherited from local runs:

```bash
python -m pytest tests/ml/test_numpy_noise.py tests/ml/test_batched_noise.py \
  tests/ml/test_qualify_s12_pipeline.py -q
```

Then reduce every report from steps 2–5 to one decision. This exists because a
zero exit status from the diagnostic does **not** mean its measurement is
usable: a run can succeed while reporting `sustained_measurement_valid: false`,
which happens routinely for NumPy because it drains the requested batches before
the 20-second window elapses.

```bash
python work/experiments/ml/hpc/qualify_s12_pipeline.py \
  --report-dir "$S12_QUAL_DIR" \
  --expected-source-commit "$S12_SOURCE_COMMIT" \
  --json-out "$S12_QUAL_DIR/qualification.json"
status=$?
case "$status" in
  0) echo "QUALIFICATION PASSED — proceed to step 7" ;;
  2) echo "REMEASURE — rerun the named conditions with the larger --batches, then rerun this gate" ;;
  *) echo "QUALIFICATION FAILED — do not submit; resolve the listed failures" >&2 ;;
esac
```

| Exit | Status | Meaning | Action |
|---:|---|---|---|
| 0 | `PASS` | Every required check passed and every measurement is valid | Proceed to step 7 |
| 2 | `REMEASURE` | No check failed; a measurement spans too little time | Rerun only the named conditions with the `--batches` value given, then rerun the gate |
| 1 | `FAIL` | A validation failed, or a report is missing/malformed/stale | Do not submit; resolve the listed failures |

What the gate checks, per experiment: artifact identity is `ok` and for the right
experiment; all four throughput conditions exist, ran the **effective**
implementation claimed, reported `loader_smoke.status=ok`, a finite positive
sustained rate over a nonzero sampled-pair count, and a valid sustained
measurement; frozen validation is exact over all 1,024 records across both
partitions with no missing, duplicated, or mismatched records, and ran a
JAX-backed strategy; and no report carries a source commit other than the
reviewed one. A stale report left in a reused directory therefore cannot
silently qualify a new revision.

A throughput below 70 pairs/s does **not** fail a scientifically correct
implementation. It is reported as an operational band — encouraging (≥70),
marginal (40–70), or inadequate for a single 72-hour allocation without further
evidence (<40) — and it prevents the gate from claiming feasibility, which is a
separate judgement from correctness.

The gate also reports `best/clean`: how close the fastest noisy strategy gets to
no-noise loading. Near 1.0 means noise generation is no longer the bottleneck,
shard I/O is, and no further sampler work would help — that is the finding that
decides whether step 8 is worth running.

Remaining gates the script cannot read from the reports, to check by eye:

| Gate | Where |
|---|---|
| Memory | node RSS within the 64 GB allocation across four workers |
| Branch | `fraction_selecting_poisson_branch` from step 3 on real V4 images |
| Regression | the pytest invocation above passed on the cluster build |

### Step 7 — Production decision and launch preview

Compare against the historical reference of 21.64 pairs/s (E01) and 21.28 (E02).

**Choosing a strategy for the first E01 run — decide from the V4 numbers, not
from local fixtures.** The gate's `recommended_action` names the fastest
strategy that clears *both* experiments, ranking by each strategy's worst
experiment, because the campaign has to fit E02 as well as E01. Use that as the
starting point and apply judgement:

- If NumPy is substantially faster on V4 and all scientific checks passed, it is
  the provisional preferred strategy: simplest implementation, no JAX backend
  constraint, no per-worker-pool compilation.
- If `best/clean` is near 1.0, shard I/O or pair sampling dominates and the
  choice barely matters. Prefer `batched`, which keeps training bitwise
  continuous with the prescribed JAX realization at no real throughput cost.
- Prefer `batched` regardless if review wants bitwise continuity in training as
  well as validation. That is a scientific-review preference, not a performance
  one.

Do not hard-code a preference from the local fixture measurements: they overstate
NumPy's loader advantage because warm synthetic shards make noise the entire
cost. Once chosen, keep the strategy fixed across all six runs; resume rejects a
mid-campaign switch, and mixing strategies would mix RNG methods within one
campaign.

Data loading overlaps GPU computation, so do not add both elapsed times.
Validation, checkpoints, and non-overlapped compute are additional. A CPU
projection cannot by itself certify 72 hours.

Keep the compute shell open for the preview; `exit` releases the allocation.

Set `S12_IMPL` to the strategy selected above, and keep it fixed for the whole
campaign.

```bash
export GPU_PARTITION=REPLACE_WITH_SITE_GPU_PARTITION
export GPU_GRES=REPLACE_WITH_SINGLE_A100_GRES
export S12_IMPL=REPLACE_WITH_batched_OR_numpy

# Every output path is explicit and outside $REPO_ROOT, so the versioned source
# checkout stays pristine. --run-dir nests <root>/S12/<EXPERIMENT>/<RUN>;
# --persist-dir nests the same way; --log-root holds Slurm %x-%j.{out,err}.
submit=(--site gattaca2 "${common[@]}" --repo-root "$REPO_ROOT"
 --conda-prefix "$CUDA_CONDA_PREFIX"
 --test-manifest "$GATTACA_SCRATCH/artifacts/S12/test_pairs/S12-TEST-PAIRS-v1"
 --run-dir "$S12_RESTART_ROOT"
 --log-root "$S12_LOG_ROOT"
 --persist-dir "$S12_PERSIST_ROOT"
 --launch-packet "$S12_LAUNCH_ROOT"
 --source-commit "$S12_SOURCE_COMMIT" --noise-implementation "$S12_IMPL"
 --dataloader-smoke --dataloader-smoke-batches 4 --dataloader-smoke-timeout 900
 --extra-sbatch-arg="--partition=$GPU_PARTITION"
 --extra-sbatch-arg="--gres=$GPU_GRES" --extra-sbatch-arg=--time=3-00:00:00
 --extra-sbatch-arg=--cpus-per-task=8 --extra-sbatch-arg=--mem=64G)
python work/experiments/ml/hpc/submit_study_run.py "${submit[@]}" --plan-preview \
  > "$S12_QUAL_DIR/launch-preview.json"
```

Verify the preview before submitting anything. These assertions check what the
helper actually resolved, rather than trusting that the flags were accepted:

```bash
python - <<'EOF' || { echo "FATAL: launch preview did not verify" >&2; return 1 2>/dev/null || exit 1; }
import json, os, sys
env = os.environ
payload = json.loads(open(f"{env['S12_QUAL_DIR']}/launch-preview.json").read())
expected_runs = {f"S12-E{e:02d}-R{r:03d}" for e in (1, 2) for r in (1, 2, 3)}
problems = []

if payload["submission_performed"] is not False:
    problems.append("preview reports a submission was performed")
if payload["run_count"] != 6:
    problems.append(f"run_count={payload['run_count']}, expected 6")
actual_runs = {c["run_id"] for c in payload["commands"]}
if actual_runs != expected_runs:
    problems.append(f"run IDs {sorted(actual_runs)} != {sorted(expected_runs)}")

for entry in payload["commands"]:
    run, command = entry["run_id"], " ".join(entry["command"])
    submitted, logs = entry["submitted_environment"], entry["log_paths"]
    if entry["missing_required_environment"]:
        problems.append(f"{run}: missing env {entry['missing_required_environment']}")
    # Scheduler resources.
    for option in (f"--partition={env['GPU_PARTITION']}", f"--gres={env['GPU_GRES']}",
                   "--time=3-00:00:00", "--cpus-per-task=8", "--mem=64G"):
        if option not in command:
            problems.append(f"{run}: {option} absent from sbatch command")
    # Output paths: each kind under its own root, none inside the source tree.
    for label, value, root in (
        ("stdout", logs["output"], env["S12_LOG_ROOT"]),
        ("stderr", logs["error"], env["S12_LOG_ROOT"]),
        ("run dir", submitted.get("ML_RUN_DIR", ""), env["S12_RESTART_ROOT"]),
        ("persist dir", submitted.get("ML_PERSIST_DIR", ""), env["S12_PERSIST_ROOT"]),
    ):
        if not value or os.path.realpath(root) not in os.path.realpath(value):
            problems.append(f"{run}: {label} {value!r} is not under {root}")
        if value and os.path.realpath(env["REPO_ROOT"]) in os.path.realpath(value):
            problems.append(f"{run}: {label} {value!r} is inside the source checkout")
    # Run identity and strategy.
    if submitted.get("ML_SOURCE_COMMIT") != env["S12_SOURCE_COMMIT"]:
        problems.append(f"{run}: ML_SOURCE_COMMIT={submitted.get('ML_SOURCE_COMMIT')!r}")
    if submitted.get("DLUXSHERA_NOISE_IMPLEMENTATION") != env["S12_IMPL"]:
        problems.append(
            f"{run}: strategy={submitted.get('DLUXSHERA_NOISE_IMPLEMENTATION')!r}, "
            f"expected {env['S12_IMPL']!r}")
    if submitted.get("ML_CONDA_PREFIX") != env["CUDA_CONDA_PREFIX"]:
        problems.append(f"{run}: conda prefix {submitted.get('ML_CONDA_PREFIX')!r}")
    # An overwrite flag would delete the only record of the failed attempts.
    if "ML_OVERWRITE" in submitted:
        problems.append(f"{run}: ML_OVERWRITE is set; refusing")
    # Nothing may point at the original failed run root.
    if "2e44b6878861" in json.dumps(submitted):
        problems.append(f"{run}: references the prior S12 checkout/run root")

if problems:
    print("LAUNCH PREVIEW PROBLEMS:", file=sys.stderr)
    for item in problems:
        print(f"  - {item}", file=sys.stderr)
    sys.exit(1)
print(f"preview ok: {payload['run_count']} runs, no submission, paths isolated")
EOF

# Print the first run's command, environment, and artifacts for eye inspection.
python - <<'EOF'
import json, os
preview = json.loads(open(f"{os.environ['S12_QUAL_DIR']}/launch-preview.json").read())
first = next(c for c in preview["commands"] if c["run_id"] == "S12-E01-R001")
print("command:", " ".join(first["command"]))
print("log_paths:", json.dumps(first["log_paths"], indent=1, sort_keys=True))
print("environment:", json.dumps(first["submitted_environment"], indent=1, sort_keys=True))
print("artifacts:", json.dumps(first["resolved_artifacts"], indent=1, sort_keys=True))
EOF
```

The assertions cover the six run IDs, scheduler resources, the four output path
families, run identity, the selected strategy, and the absence of
`ML_OVERWRITE`. Confirm the remaining items by eye from the printed block: the
prepared V4 and frozen-artifact paths, and the resolved artifact IDs.

Note the sbatch command carries `--time` twice: the Gattaca2 profile default
(`24:00:00`) followed by the `--extra-sbatch-arg` override (`3-00:00:00`). Slurm
honours the last occurrence, so the effective wall time is 72 hours. The
assertion checks the override is present rather than that it appears alone.

```bash
# Only after the gate passed and the preview checks out: ONE run.
python work/experiments/ml/hpc/submit_study_run.py "${submit[@]}" \
 --experiment S12-E01 --run S12-E01-R001 --submit-plan
```

The first real E01 run is the GPU verification; a separate A100 smoke allocation
is unnecessary. Watch the Slurm output/error paths under `$S12_LOG_ROOT`, the
launch manifest in `$S12_LAUNCH_ROOT`, and
`$S12_RESTART_ROOT/S12/S12-E01/S12-E01-R001/{history.csv,checkpoint_last.pt}`.
Require CUDA availability, successful noisy batch loading, a completed first
epoch, finite training and validation losses, `history.csv` and checkpoint
creation, and this provenance before submitting the remaining five:

```bash
python - <<'EOF'
import os, torch
run = os.path.expandvars("$S12_RESTART_ROOT/S12/S12-E01/S12-E01-R001")
ckpt = torch.load(f"{run}/checkpoint_last.pt", weights_only=True)
runtime = ckpt["runtime_provenance"]
print({k: runtime.get(k) for k in (
    "noise_implementation", "validation_noise_implementation",
    "jax_backend", "jax_version", "numpy_version")})
print("rng:", runtime.get("numpy_noise_rng", {}).get("bit_generator"))
assert runtime["noise_implementation"] == os.environ["S12_IMPL"]
# Frozen validation must remain JAX-backed whichever training strategy is used.
assert runtime["validation_noise_implementation"] in ("reference", "batched")
assert "noise_implementation" not in ckpt["training_identity"]
print("runtime provenance ok")
EOF
```

With `numpy` selected, `validation_noise_implementation` must read `reference` —
that is the frozen-recipe isolation, and seeing `numpy` there would mean the
frozen validation realization had changed.

The full first-run checklist, in order, before the remaining five are submitted:

| # | Check | Where |
|---:|---|---|
| 1 | Bounded preflight completed its batches | Slurm stdout under `$S12_LOG_ROOT` |
| 2 | CUDA available; first batch delivered | Slurm stdout (`training start`, first-batch line) |
| 3 | Epoch 1 completed | `history.csv` has a row for epoch 0 |
| 4 | Training and validation losses finite | `history.csv` |
| 5 | `history.csv` and `checkpoint_last.pt` exist | `$S12_RESTART_ROOT/S12/S12-E01/S12-E01-R001/` |
| 6 | Runtime provenance correct | the assertion block above |
| 7 | Observed epoch wall time recorded | Slurm stdout epoch timing |

Then decide on the remaining five from **measured end-to-end epoch duration**,
not from the CPU loader projection. These are different quantities and must not
be added or conflated:

- *CPU data-loader capacity* — what step 4 measures. Overlaps GPU compute.
- *GPU training step time* — only the real run shows this.
- *Validation and checkpoint overhead* — per-epoch, additional, not overlapped.
- *Repeated worker startup* — first-batch latency recurs 600 times, since
  persistent workers stay disabled so `set_epoch` reaches the workers.
- *Multi-allocation checkpoint/resume* — if 600 epochs do not fit 72 hours, this
  is the sanctioned fallback and preserves the prescribed epoch pair and noise
  keys. Resume requires the same strategy, which is why it must not change
  mid-campaign.

Preserve the incident archive. Do not alter the S12 prescription to fit the
allocation.

### Step 8 — Optional investigation (only if step 6 leaves a question open)

Skip this by default. Run it only when step 6 shows an unresolved bottleneck —
in practice when `best/clean` is well below 1.0 (noise still dominates and you
want the stage breakdown) or when a strategy underperforms its local ratio by an
unexplained margin. It is diagnostic, not qualification, and it costs
compute-node time.

```bash
# Stage breakdown plus resident per-implementation noise timing. The
# shared-RNG ceiling is labelled NON-PRODUCTION: its realization depends on
# batch composition, so it bounds achievable speed and is not selectable.
for exp in S12-E01 S12-E02; do
 for impl in batched numpy; do
  python work/experiments/ml/hpc/diagnose_ml_dataloader.py "${common[@]}" \
   --experiment-id "$exp" --noise-implementation "$impl" --num-workers 4 \
   --batches 80 --warmup-batches 8 --profile --numpy-shared-rng-ceiling \
   --timeout-seconds 1800 \
   > "$S12_QUAL_DIR/optional-$exp-$impl-profile.json"
 done
done

python - <<'EOF'
import json, pathlib, os
root = pathlib.Path(os.environ["S12_QUAL_DIR"])
for path in sorted(root.glob("optional-*-profile.json")):
    resident = json.loads(path.read_text()).get("loader_smoke", {}).get(
        "result", {}).get("resident_noise_one_view", {})
    if not resident:
        continue
    print(f"\n{path.stem}: isolated noise, seconds per batch")
    for name, row in resident.items():
        flag = "" if row.get("production_candidate") else "  [NON-PRODUCTION]"
        print(f"  {name:42} {row['warm_batch_seconds']:.4f}"
              f"  ({row.get('images_per_second') or 0:7.1f} img/s){flag}")
EOF
```

Profile reports are written with an `optional-` prefix so the qualification gate
in step 6 does not consider them. Eight workers were not part of the
prescription and `num_workers: 4` is in `training_identity`, so an alternative
worker count is an investigation only — it cannot be adopted without changing
the scientific identity.

## NumPy photon-noise investigation

**Outcome: adopted as a third opt-in strategy, `--noise-implementation numpy`.**
Reference remains the default. The decisive result is that the deterministic
per-record seed contract is essentially free, so NumPy keeps almost all of the
isolated speed its unmatched microbenchmark suggested.

### Why NumPy is admissible here at all

The dynamic training-noise generator does not need to be differentiable. The
optical modeling ecosystem is JAX-based, but the CNN training pathway is
PyTorch, and the noise stage operates on already-prepared count-space images
with no automatic differentiation downstream. So the sampler's RNG algorithm is
an execution choice, not a modeling choice — provided the sampled *distribution*
is unchanged. The NumPy strategy therefore returns `float32` NumPy arrays
straight into the existing scaler and tensor interface; no JAX conversion is
added at the end of the noise stage.

### Deterministic per-record seed design

The NumPy sampler reuses the existing logical seed schedule verbatim: the same
`pair_noise_side_seeds()` derivation from base noise seed, pair record identity,
epoch-dependent dynamic offset, side-specific offset (`+17` model / `+31`
observation), and E02's `+1,000,000,007` independent-view offset. Only the
stream generated *from* that integer differs.

Each image is drawn from its own generator, `np.random.Generator(PCG64(seed))`,
constructed from that derived integer. No global or per-process RNG state is
consulted, which is what makes the realization a function of the logical record
identity alone. Measured consequences, all covered by tests:

| Property | Status |
|---|---|
| Same logical pair + epoch reproduces the same image | yes |
| Independent of batch size (widths 1, 2, 3, 5, 8) | yes |
| Independent of batch composition and ordering | yes |
| Independent of worker count (0, 2, 4) | yes |
| Independent of start method (forkserver, spawn) | yes |
| Fresh realization per epoch | yes |
| E02's two views independently generated | yes |
| Forward/reverse observation-side convention preserved | yes |

`PCG64` is named explicitly in `dluxshera.ml.noise.NUMPY_BIT_GENERATOR` rather
than relying on `np.random.default_rng`, whose default algorithm is a
documented-but-changeable NumPy implementation detail. Recorded provenance is
`PCG64`, NumPy 2.3.5, seeded through `SeedSequence`.

That `SeedSequence` hashing step matters: the schedule's side offsets differ by
only 14, and a naive linear seeding would give near-identical streams. Measured
correlation between the `+17` and `+31` side streams at λ=5 over 200×200 pixels
is ρ = 0.0014, against a 1/√N floor of 0.005 — indistinguishable from
independent. E02 view-pair correlation over 200 record/epoch combinations gives
mean ρ = +0.0004, σ = 0.0062, max |ρ| = 0.020, against an arbitrary-seed floor
of mean +0.0003, σ = 0.0067, max 0.018. The E02 views are as independent as two
unrelated seeds.

### Statistical validation

NumPy need not reproduce JAX's samples, so it is validated as a distribution.
At 640×640 = 409,600 draws per level, tolerances sized from each statistic's
sampling distribution:

| λ | mean | variance | Fano | P(0) measured | P(0) true | χ² p |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.1005 | 0.1005 | 1.005 | 0.9044 | 0.9048 | 0.849 |
| 0.37 | 0.3701 | 0.3676 | 0.994 | 0.6898 | 0.6907 | 0.091 |
| 0.50 | 0.5002 | 0.4973 | 0.995 | 0.6057 | 0.6065 | 0.018 |
| 1.00 | 1.0017 | 1.0000 | 1.000 | 0.3671 | 0.3679 | 0.350 |
| 5.00 | 5.0009 | 5.0077 | 1.002 | 0.0071 | 0.0067 | 0.049 |
| 20.0 | 19.991 | 20.104 | 1.005 | — | — | 0.628 |
| 300.0 | 299.96 | 301.35 | 1.004 | — | — | 0.362 |
| 9000.0 | 8999.7 | 9040.9 | 1.005 | — | — | 0.251 |

Contrast this with the lookup table's behaviour at the same low λ, where
`nearest` collapses λ<0.5 to a deterministic zero and `mixture` inflates the
Fano factor to 1.5. NumPy's Fano factor is 1 throughout, including the faint
regime that is S12's actual subject. Tests additionally check selected tail
probabilities at λ+k√λ, spatial lag correlations (|ρ| < 6/√N at lags (0,1),
(1,0), (8,0), (0,8)), and a two-sample KS test of NumPy against direct JAX
samples at λ=3 — which does not reject, while the realizations themselves
differ, exactly as intended.

### Strategy A versus Strategy B

Strategy A is the deterministic per-image generator. Strategy B is one RNG over
the whole stacked batch, measured as a performance ceiling only: its stream
depends on batch composition, ordering, and generator state, so it cannot hold
the reproducibility contract.

| Strategy | s / 32-image batch | images/s | vs batched JAX | Reproducible per record |
|---|---:|---:|---:|---|
| A — per-image `PCG64` (**production candidate**) | 0.0313 | 1022.8 | 4.73× | yes |
| B — one shared RNG over the stack (non-production) | 0.0301 | 1064.5 | 4.93× | no |

**Strategy A captures 96% of Strategy B's throughput.** The reproducibility
contract costs ~4%, so there is no trade to make and Strategy B was not
integrated. The diagnostic can still print it under
`--numpy-shared-rng-ceiling`, labelled `NON-PRODUCTION`.

The stage decomposition explains why. Per 32-image batch, measured on resident
real V3 images:

| Stage | s / batch | Share |
|---|---:|---:|
| Poisson draw | 0.02943 | 94% |
| Input clipping (`np.maximum` ×32) | 0.00052 | 1.7% |
| Scaler transform ×32 | 0.00027 | 0.9% |
| Output `float32` cast ×32 | 0.00023 | 0.7% |
| RNG initialization (`PCG64` ×32) | 0.00014 | 0.4% |
| Bright-branch selection (`np.min` ×32) | 0.00009 | 0.3% |
| Seed derivation (SHA-256 ×32) | 0.00006 | 0.2% |
| Tensor assembly ×32 | 0.00004 | 0.1% |

Seed derivation plus RNG construction is 0.0002 s — under 1% of the batch. The
draw itself dominates, which is why 32 small vectorized calls cost almost the
same as one large one.

### Noise model preserved

The NumPy sampler reimplements the SHERA branch decision exactly as
`src/dluxshera/utils/noise.py` has it: the Gaussian approximation is selected
only when the **whole-image minimum** exceeds `bright_threshold`, so a single
faint outskirt pixel sends the image to exact Poisson. Tests cover the bright
branch, a mixed image with one faint pixel, and equality at the threshold
(strictly greater, so it stays on Poisson). The shared JAX implementation and
its bright-image behaviour were not removed or modified.

Unsupported configurations — disabled noise, `legacy_numpy`, read noise, dark
current — fall back to the reference path and are asserted **bitwise equal** to
it. The fallback is exact rather than an approximation in a different detector
model. `numpy_supports()` is the predicate, and
`effective_noise_implementation()` reports what will actually run so provenance
cannot claim `numpy` for a configuration that silently used reference.

### Frozen validation isolation

**The frozen S12 noisy-validation recipe is unchanged and stays JAX-backed.**
`resolve_validation_noise_implementation()` maps a `numpy` training selection to
`reference` for the validation loader; `reference` and `batched` pass through
because they are bitwise identical. `VALIDATION_NOISE_IMPLEMENTATIONS` omits
`numpy` deliberately. This is why the training optimization can be evaluated
without regenerating `S12-PHOTON-NOISE-VALIDATION-v1`.

Training and validation therefore use the **same intended photon-noise
probability model** but may use **different RNG algorithms and different
realized noise samples**. That asymmetry is deliberate and recorded in runtime
provenance as separate `noise_implementation` and
`validation_noise_implementation` fields. The diagnostic mirrors it:
`--noise-implementation numpy --loader validation` resolves to `reference`.

### JAX overhead under NumPy training

Selecting `numpy` removes JAX from the **dynamic training-noise** pathway only.
It does not make a run JAX-free, and the distinction matters when reading the
first-batch latencies:

| Cost | Under `numpy` training |
|---|---|
| JAX import and backend initialization | **still paid** — `dluxshera` imports JAX via dLux at module import |
| Training-noise XLA compilation per worker pool | removed |
| Training-noise JAX dispatch per batch | removed |
| Frozen validation-noise JAX sampling | **still paid** — validation stays on the JAX reference |
| Validation XLA compilation | not applicable: the reference path is eager, not JIT |
| Fork-safe worker start method | **still required** — `noise_requires_safe_workers` keys off the noise *model*, which is `shera_observation` for validation regardless of strategy |

So the fork-deadlock constraint and the forkserver/spawn requirement are
unchanged under `numpy`, because the validation loader still runs JAX in its
workers. Do not read "NumPy training" as "no JAX runtime cost"; read it as "no
JAX in the per-batch training-noise draw".

### Caveats and limits

- Not bitwise comparable to the JAX reference, by design. `--verify-equivalence`
  rejects `numpy` with an explanatory error rather than reporting an expected
  "not exact" — a bitwise check against a different stream says nothing about
  correctness.
- Determinism is claimed **within this NumPy version and build**. Bitwise
  stability across NumPy versions or hardware platforms is not claimed and was
  not tested. `PCG64` is a stable named algorithm, which is the basis for
  expecting portability, not evidence of it.
- The derived seed space is 2³², and the schedule draws ~9.83 million seeds per
  side per campaign, so ~11,250 birthday collisions are expected. This is a
  property of the **existing** schedule, shared with the JAX reference, not
  something NumPy introduces. A collision means two logical records share a
  realization; at 1.1 × 10⁻³ of records it does not affect the training
  distribution, and changing it would alter the prescribed noise identity.
- Resume rejects any switch between strategies, so one campaign never mixes RNG
  methods. Historical checkpoints without the field resume only as `reference`.
- Selection is explicit. There is no automatic backend choice based on
  performance.

### Local loader throughput, and why it overstates the cluster gain

Full loader numbers for all three strategies are in
[Sustained local fixture results](#sustained-local-fixture-results); NumPy
reaches 6,235.59 pairs/s (E01-like) and 4,226.01 (E02-like) against batched
JAX's 445.60 and 235.94.

That is 14.0× and 17.9× on these fixtures — **far above the 4.73× isolated noise
ratio, and that discrepancy is the warning, not the headline**. The synthetic
shards are small and warm, so fixture I/O is negligible and the measurement is
dominated by the noise kernel. Real V4 random GPFS reads are not, so the
loader-level gain on the cluster will be smaller than this and bounded above by
the clean baseline. **Do not extrapolate these numbers to Gattaca2.** Step 4 of
the qualification measures all four conditions on real V4 data, and the clean
baseline is what reveals how much of the remaining cost is not photon noise at
all.

For scale, the clean fixture baseline at 8,000 batches reached 28,656 pairs/s
(E01-like) and 18,007 (E02-like) — both still below the 20-second span gate and
therefore **invalid for capacity**, so read them only as a rough fixture
ceiling. NumPy sits at roughly 22–23% of it, meaning the noise draw is still the
dominant per-batch cost even at 6,235 pairs/s on warm fixtures. On V4 that
ordering may invert, which is exactly what the cluster clean baseline settles.

NumPy needed 6,000 batches to satisfy the 20-second sustained gate at these
speeds — the loader is fast, not unusable, which is the case the gate's new
`recommended_batches_for_valid_measurement` field exists to signal.

## Research: precomputed Poisson lookup tables

**Conclusion: not viable for S12. Do not pursue.** The prototype lives at
`work/experiments/ml/research/poisson_lookup_prototype.py`, is imported by
nothing in `src/dluxshera`, and must stay that way. A test asserts that
separation. It was not integrated into the DataLoader and the S12 experiment is
unchanged.

The idea was to replace a live Poisson draw with an indexed read from a table
`T[k, u, v]` of precomputed draws at expected-count level `k`, with `(u, v)`
randomized so a pixel's outcome varies across epochs. Both proposed indexing
strategies were implemented and measured: **Strategy A** draws an independent
`(u, v)` per output pixel; **Strategy B** draws one `(dx, dy)` per image (or per
count level) and indexes `((x + dx) % W, (y + dy) % H)`.

### The result that settles it: the baseline was wrong

Before comparing lookup against batched JAX, note what the same measurement
says about plain NumPy. Resident real V3 images, 32×160×160 float32, warm:

| Sampler | s / 32-image batch | images/s | ratio vs batched JAX |
|---|---:|---:|---:|
| JAX reference, per image | 0.7290 | 43.9 | 0.19× |
| **JAX batched (production)** | **0.1367** | **234.0** | 1.00× |
| NumPy `rng.poisson` (vectorized, shared RNG) | 0.0302 | 1060.1 | **4.53×** |
| Lookup A, per-pixel index, 1.11 GiB table | 0.0218 | 1467.6 | 6.3× |
| Lookup B, offsets, 1.11 GiB table | 0.0137 | 2342.4 | 10.0× |

**Read these ratios as a comparison between particular benchmark pathways, not
as an end-to-end speedup available by swapping one function.** They are not
matched measurements: the JAX rows include per-record key derivation and data
assembly, while the NumPy row is one vectorized RNG call over a pre-stacked
array with no per-record seeding at all. The lookup rows additionally exclude
table construction, which the timing table below measures separately at 17.0 s.
Replacing the sampler changes only the noise stage; the loader also pays shard
I/O, pair sampling, scaling, and tensor assembly, so a 4.5× isolated noise
ratio does not imply 4.5× loader throughput. The
[NumPy investigation](#numpy-photon-noise-investigation) re-measures this with
the real per-record seed contract and reports loader throughput separately.

Within those limits the ordering is still informative: lookup is only 1.4–2.2×
faster than *NumPy*, so its apparent 6–10× advantage is largely an artifact of
measuring against a JAX baseline. The headline comparison is therefore not
"lookup versus JAX" but "lookup versus NumPy", and in that comparison a sub-2×
gain does not pay for a gigabyte-scale table, a new noise identity, and the loss
of exact reproducibility.

Cost decomposition at a 128 MiB table: index generation alone is 0.0047 s/batch
and the gather alone 0.0056 s/batch, so neither half dominates — the lookup is
memory-system bound, which is also why the larger full-coverage table is slower
than the small one despite doing identical work.

### Distribution fidelity: the integer level grid is fatal

400,000 draws per cell, 256×256 pool, 1,024 integer levels. `nearest` rounds
λ to the closest stored integer; `mixture` randomizes between the bracketing
integers with probability equal to the fractional part.

| λ | policy | mean rel. err | var rel. err | TV distance | P(0) measured | P(0) true |
|---:|---|---:|---:|---:|---:|---:|
| 0.10 | nearest | **−1.000** | **−1.000** | 0.0952 | **1.0000** | 0.9048 |
| 0.10 | mixture | −0.005 | **+0.896** | 0.0538 | 0.9371 | 0.9048 |
| 0.37 | nearest | **−1.000** | **−1.000** | 0.3093 | **1.0000** | 0.6907 |
| 0.37 | mixture | +0.004 | **+0.645** | 0.1192 | 0.7656 | 0.6907 |
| 1.50 | nearest | **+0.333** | +0.334 | 0.1520 | 0.1354 | 0.2231 |
| 1.50 | mixture | +0.003 | **+0.170** | 0.0435 | 0.2506 | 0.2231 |
| 12.37 | nearest | −0.030 | −0.021 | 0.0430 | — | — |
| 12.37 | mixture | −0.001 | +0.021 | 0.0081 | — | — |
| 20.0 | nearest | −0.001 | +0.012 | 0.0077 | — | — |
| 300.0 | nearest | −0.000 | −0.009 | 0.0147 | — | — |

The low-count regime is where S12 actually lives: the V3 statistics above show
per-image minima of 1.2–3.6 and a pooled pixel median of 61.7, so a large
fraction of every image sits at single-digit counts. There, nearest-integer
lookup is not approximate, it is **wrong**:

| λ | → level | mean (want) | variance | P(0) measured (want) |
|---:|---:|---|---:|---|
| 0.10 | 0 | 0.0000 (0.1000) | 0.0000 | 1.0000 (0.9048) |
| 0.37 | 0 | 0.0000 (0.3700) | 0.0000 | 1.0000 (0.6907) |
| 0.80 | 1 | 0.9979 (0.8000) | 1.0004 | 0.3701 (0.4493) |
| 1.20 | 1 | 0.9979 (1.2000) | 1.0004 | 0.3701 (0.3012) |
| 1.50 | 2 | 2.0032 (1.5000) | 2.0081 | 0.1356 (0.2231) |
| 3.56 | 4 | 3.9953 (3.5600) | 3.9877 | 0.0200 (0.0284) |

Every λ below 0.5 collapses to a deterministic zero — the photon noise simply
disappears from the faintest pixels. `mixture` repairs the mean but is
**overdispersed by construction**: a mixture of Poisson(k) and Poisson(k+1) with
weight f has variance λ + f(1−f), so at λ=0.5 the variance is inflated 50% and
the Fano factor is 1.5, not 1. For a study whose entire subject is photon-noise
robustness, getting the noise variance wrong in the faint regime is
disqualifying. Finer quantization fixes fidelity but multiplies the level count,
and therefore the memory, by the inverse resolution.

### Independence and finite-table bias

Correlation is *not* the problem. Spatial lag correlations for both strategies
sit at the finite-sample noise floor set by exact NumPy draws (|ρ| ≲ 0.005 at
lags (0,1), (1,0), (8,0), against a NumPy floor of ~0.0005–0.005), and
cross-realization correlation between two independently requested draws of the
same expected image is ≤0.0013. Randomized indexing does remove the obvious
fixed spatial pattern.

Correlation measured across *differing* expected images cannot, however, detect
a shared index: the expected values differ, so the outputs differ regardless.
A duplicate-input probe isolates the indexing, and it finds a real defect in one
variant. With `offset_per_level=True` the translations are drawn once per count
level and reused for every image in the call, so **four identical expected
images in one batch receive four identical draws** — requested "independent"
realizations are literally the same sample. The default per-image offset and
per-pixel indexing do not share this defect. `correlation_report` now reports
`duplicate_input_independence`, and
`tests/ml/test_poisson_lookup_research.py::test_duplicate_input_probe_exposes_the_shared_offset_defect`
pins both behaviours. This is recorded as a limitation of a rejected prototype,
not fixed.

Finite-table *bias* is the real constraint, and it is a function of pool size.
At λ=20 with 400k draws, the χ² test against Poisson detects the empirical
pool's deviation until the pool is ~10⁶:

| Pool per level | TV distance | χ² p-value |
|---:|---:|---:|
| 4,096 (64²) | 0.0274 | 0 |
| 16,384 (128²) | 0.0137 | 4.6e−217 |
| 65,536 (256²) | 0.0054 | 1.2e−36 |
| 262,144 (512²) | 0.0045 | 9.4e−05 |
| 1,048,576 (1024²) | 0.0025 | **0.55** |

So ~10⁶ draws per level is the threshold at which a single level's marginal
becomes statistically indistinguishable from Poisson. That is the figure the
proposal anticipated — and it is still not independence. Within one 160×160
image, Strategy A reuses 17.2% of its table coordinates (25,600 pixels drawn
with replacement from a 65,536-entry slice); Strategy B reuses none within an
image but reuses whole translated regions across images. Across a full S12 run
the draw count is 600 × 16,384 × 25,600 ≈ **2.5e11**, so every entry at a
commonly occurring count level is revisited on the order of 10⁵–10⁶ times. No
amount of index randomization makes 2.5e11 draws from a 10⁶-entry pool
independent.

### Memory

Table size is `levels × W × H × itemsize`. The binding constraint is not the
pool but the **count range**: real V3 images reach ~10,600 expected counts, so
integer coverage needs ~10,600 levels, and pixels above the top stored level
receive draws from the wrong level entirely. At 1,024 levels and a 256² pool,
9.0% of real V3 pixels were silently clipped — concentrated in the bright core,
where the signal is.

| Spatial | Levels | dtype | Size | ×4 workers |
|---|---:|---|---:|---:|
| 64² | 10,593 | uint16 | 0.08 GiB | 0.32 GiB |
| 256² | 1,024 | uint16 | 0.13 GiB | 0.50 GiB |
| 256² | 10,593 | uint16 | **1.29 GiB** | 5.17 GiB |
| 1024² | 1,024 | uint16 | 2.00 GiB | 8.00 GiB |
| 1024² | 10,593 | uint16 | **20.69 GiB** | 82.76 GiB |
| 1024² | 1,024 | float32 | 4.00 GiB | 16.00 GiB |

Satisfying both constraints at once — ~10⁶ draws per level *and* full count
coverage — is the 1024²×10,593 cell: **20.7 GiB**, or 82.8 GiB if copied per
worker, against a 64 GB allocation. S12 workers start via forkserver/spawn, so
a table built per worker *is* duplicated per worker. A read-only `np.memmap`
avoids duplication, but then Strategy A's random per-pixel indexing becomes
page-fault bound on GPFS — the opposite of the intended optimization. Building
the 1.11 GiB full-coverage table measured in the timing table took **17.0 s**,
repaid on every non-persistent worker pool start, i.e. 600 times per run.
uint16 is safe to 8σ only up to ~65,000 counts; Poisson is unbounded above, so
dtype safety is a stated margin, not a guarantee.

### The one conditionally exact construction, and why it still loses

Poisson is infinitely divisible and closed under addition of independent
variates, so `Poisson(a) + Poisson(b) = Poisson(a+b)`. Storing one pool per
dyadic rate `2^j·δ` and summing independent draws for each set bit of
`round(λ/δ)` yields a result that is Poisson at the quantized rate —
categorically different from `nearest` and `mixture`, which are approximations
by construction.

**The exactness is conditional, and the condition is idealized.** It holds for a
single draw at λ on the δ lattice, *assuming the summed pool entries are
independent draws*. A finite table cannot supply unlimited independent draws:
entries are revisited as the campaign proceeds (the reuse arithmetic below
applies here too), so the independence assumption underpinning the exactness
degrades with reuse. λ off the lattice is quantized, and λ above the
representable maximum is clipped. So "exact" below means *exact under the stated
idealization*, never "equivalent to unlimited independent Poisson draws". That
distinction is why this construction would still require scientific review
rather than being treated as a drop-in equivalent, and it is pinned by
`tests/ml/test_poisson_lookup_research.py::test_dyadic_exactness_is_conditional_not_unqualified`.

Verified at a 1024² pool per rate, λ on the lattice:

| λ | mean rel. err | var rel. err | TV | χ² p | gathers |
|---:|---:|---:|---:|---:|---:|
| 0.25 | +0.0026 | +0.0025 | 0.0005 | 0.628 | 1 |
| 1.00 | −0.0012 | −0.0024 | 0.0008 | 0.783 | 1 |
| 5.00 | +0.0001 | +0.0002 | 0.0015 | 0.053 | 2 |
| 12.25 | +0.0002 | −0.0003 | 0.0016 | 0.309 | 3 |
| 20.0 | −0.0001 | +0.0004 | 0.0021 | 0.304 | 2 |
| 300.0 | −0.0000 | +0.0003 | 0.0035 | 0.715 | 4 |

Fidelity is a non-issue and memory collapses to 24–34 MiB, because the rate
count is logarithmic in the range rather than linear. But the cost is one gather
per set bit, and real images span the whole range, so every pixel pays nearly
the full bit count:

| δ | rates | table | gathers/call | s / 32-image batch |
|---:|---:|---:|---:|---:|
| 0.25 | 17 | 34.0 MiB | 16 | 0.1359 |
| 1.00 | 15 | 30.0 MiB | 14 | 0.1277 |
| 4.00 | 13 | 26.0 MiB | 12 | 0.1067 |
| — | — | NumPy `rng.poisson` | — | **0.0302** |

The exact construction is **3.5–4.5× slower than simply calling
`rng.poisson`**, and roughly par with batched JAX. The method that is
statistically sound is not faster, and the methods that are faster are not
statistically sound.

### Decision

| Approach | Distribution fidelity | Independence | Memory | s / 32 batch |
|---|---|---|---:|---:|
| Per-image JAX (default) | Reference (prescribed) | Reference | Low | 0.7290 |
| **Batched JAX (opt-in)** | **Exact on tested backend** | **Preserved** | Bounded by batch | **0.1367** |
| **NumPy per-record (opt-in)** | **Exact Poisson, different stream** | **Full; per-record seeded** | Low | **0.0313** |
| Lookup A, per-pixel index | Approximate; fails λ<0.5 | 17% intra-image reuse; 10⁵ campaign reuse | 1.3–20.7 GiB | 0.0218 |
| Lookup B, random offsets | Approximate; fails λ<0.5 | No intra-image reuse; per-level offsets share one draw across a batch | 1.3–20.7 GiB | 0.0137 |
| Dyadic superposition | Exact at lattice λ, under an idealization | Independent summands *in the ideal*; degrades with table reuse | 24–34 MiB | 0.1067–0.1359 |

Against the stated criteria, lookup fails at least four: the performance
advantage over the real CPU alternative is under 2×, memory for a statistically
and numerically adequate table is 20.7 GiB, the distributional discrepancy in
the low-count regime is unacceptable rather than small, and it cannot be
bitwise reproducible against the prescribed JAX realization — adopting it would
require a new `noise_model` identity and a regenerated frozen validation recipe,
not a drop-in substitution. The prototype is internally seed-reproducible, which
is a necessary condition it meets and not a sufficient one.

**The measured lead here was vectorized NumPy Poisson**, not lookup: ~4.5× over
batched JAX, exactly Poisson, trivial memory, no table. That lead has since been
followed up and integrated as the opt-in `numpy` strategy; see
[the NumPy investigation](#numpy-photon-noise-investigation) for the
production-compatible per-record measurement, the statistical validation, and
the frozen-validation isolation that made it adoptable without regenerating the
noisy-validation recipe. The lookup table remains rejected: its advantage over
NumPy is under 2×, and unlike NumPy it cannot reproduce the Poisson distribution
in the faint regime at any affordable table size.

Reproduce (writes nothing; default table budget is 512 MiB):

```bash
PYTHONPATH=src:. python work/experiments/ml/research/poisson_lookup_prototype.py \
  --mode all --table-side 256 --table-levels 1024 \
  --prepared-root "Results/ML Training Datasets/preprocessed/PREP-V3-nominal-v1"
# Larger tables require raising the budget deliberately:
PYTHONPATH=src:. python work/experiments/ml/research/poisson_lookup_prototype.py \
  --mode performance --table-side 1024 --table-levels 1024 --memory-budget-mib 2100
```

## Fallback if V4 throughput remains insufficient

| Option | Scientific meaning | Operational cost |
|---|---|---|
| On-the-fly batching | Exact tested samples/seeds | CPU during GPU allocation; smallest change |
| Deterministic NumPy sampling | Same Poisson model, different RNG stream; frozen validation unchanged | Implemented and opt-in; needs the V4 measurement in step 4 |
| Epoch/chunk precompute | Can preserve all keys/order | ~1.68 GB per 16,384-example single noisy view; ~1 TB/run total writes; E02 doubles; transactional chunk/epoch metadata required |
| Bounded asynchronous producer | Can preserve all keys/order | Hides latency only if CPU keeps up; shutdown/backpressure/recovery complexity |
| Chunked CPU preprocessing | Can preserve all keys/order | CPU nodes are more available than A100s on Gattaca2, but adds scheduler orchestration, chunk-level transactional metadata, and GPFS write throughput as a new bottleneck |
| Reusable noise library | Changes epoch-dependent distribution | Less storage, requires a new scientific prescription; not an S12 optimization |
| Lookup-table noise | Approximate; fails the faint-count regime | Rejected above; not a candidate |
| Checkpoint/resume | Preserves prescribed epoch pair/noise keys | Multiple allocations; replay partial epoch; global RNG state is not restored, so not bitwise uninterrupted training |

Ordering if batching leaves insufficient margin: **try NumPy sampling first**,
since it is already implemented, measured 4.7× faster in isolated generation,
and leaves the frozen validation recipe untouched. Then
**checkpoint/resume** — it needs no new code, preserves the prescribed epoch
pair and noise keys, and trades only bitwise-uninterrupted training for multiple
allocations. CPU-side
preprocessing is the next option and is genuinely attractive on availability
grounds, since CPU nodes are easier to obtain than A100s; but it converts a
compute problem into a ~1 TB/run storage-and-orchestration problem with its own
recovery semantics, so it needs a measured need rather than a projected one. No
offline preprocessing system was implemented in this pass. Do not create a
multi-terabyte offline campaign without measured need.

## Local validation and review record

Commands used the PyCharm-configured Python 3.11.9 environment (`.venv/bin/python`):

```text
python -m pytest tests/ml -q
  429 passed, 1 failed (271.25 s)   # release-polish pass
python -m pytest tests/ml/test_qualify_s12_pipeline.py -q
  48 passed (0.38 s)
python -m pytest tests/ml/test_numpy_noise.py tests/ml/test_batched_noise.py -q
  103 passed (135.05 s)
python -m pytest tests/ml/test_poisson_lookup_research.py -q
  16 passed (1.27 s)
python devtools/check_no_src_imports.py
python devtools/check_no_examples_imports.py
  both passed
```

Earlier checkpoints in this work recorded 283+1, 294+1, 323+1, and then 381+1;
the 429 count adds the 48 qualification-gate and launch-preview tests.

The sole full-suite failure is
`test_train_from_study_validates_scaler_and_locked_artifacts_before_training`:
its split edit did not raise the expected ValueError. It also fails on an
untouched `git archive HEAD` extraction of 2e44b68 (0.98 s), independently
confirming it is pre-existing and unrelated to this patch. No attempt was made
to fix it in this task.
JAX 0.4.30 emits a deprecation warning for explicit CPU selection via the JIT
backend argument; execution and exact equality pass. Multiprocessing tests
required permission to use local forkserver sockets and PyTorch shared memory;
initial sandbox permission failures were not pipeline failures.

`git diff --check`, new documentation link resolution, and `bash -n` on the
operator command blocks pass. Submission preview reports six runs and no
submission. S12 YAML is byte-identical to 2e44b68. Study expansion remains
S10 = 9 runs, S11 = 6, S12 = 6, all with `evaluate_test: false`. The existing
ML suite covers study matrices, frozen contracts, fork rejection, supervised
timeout cleanup, serialization and epoch updates. No production output or
locked-test evaluation was created by this work; test fixtures and benchmark
output live in temporary locations.

Changed files (new files marked “new”):

- `src/dluxshera/ml/noise.py`: batched photon draws and exact fallbacks;
  deterministic per-record NumPy sampler (`apply_pair_noise_numpy`), its
  coverage predicate (`numpy_supports`), effective-strategy reporting
  (`effective_noise_implementation`), and RNG provenance
  (`numpy_noise_provenance`, `NUMPY_BIT_GENERATOR`).
- `src/dluxshera/ml/torch_data.py`: worker batch-fetch boundary; shared
  `resolve_noise_implementation` so the strategy can be resolved without a
  dataset; `numpy` added to `NOISE_IMPLEMENTATIONS`; frozen-validation isolation
  via `VALIDATION_NOISE_IMPLEMENTATIONS` and
  `resolve_validation_noise_implementation`; single-item fetch delegates to the
  batch path under `numpy` so both agree.
- `src/dluxshera/ml/training.py`: runtime implementation provenance, now
  including the validation strategy, effective strategies, NumPy/JAX versions
  and BitGenerator; datasets receive the already-resolved implementation so
  provenance cannot diverge; the validation dataset receives the JAX-backed
  strategy; the sustained gate recommends a larger sample instead of failing a
  fast loader.
- `src/dluxshera/ml/hpc.py`: explicit submission environment allowlist.
- `work/experiments/ml/hpc/preflight_ml_gpu.py`: skipped-path regression fix,
  `--noise-implementation` (now including `numpy`), effective/validation
  strategy and RNG provenance in the report.
- `work/experiments/ml/hpc/submit_study_run.py`: opt-in submission argument,
  extended to `numpy`; the launch preview now reports the resolved submitted
  environment and log paths, so `--run-dir`, `--persist-dir`, `--log-root` and
  the selected strategy are verifiable before submission rather than only after.
- `work/experiments/ml/hpc/qualify_s12_pipeline.py` (new): the executable
  qualification gate. Consumes the diagnostic reports and returns
  PASS / REMEASURE / FAIL as exit 0 / 2 / 1.
- `work/experiments/ml/hpc/diagnose_ml_dataloader.py`: baseline/profile modes,
  the counted coverage-verified equivalence comparison, `numpy` selection,
  three-way resident noise timing, the labelled non-production shared-RNG
  ceiling, frozen-validation isolation via `resolved_noise_implementation`, and
  explicit rejection of a bitwise check against the NumPy stream.
- `work/experiments/ml/hpc/analyze_image_statistics.py` (new): read-only
  count-statistics and noise-branch diagnostic.
- `work/experiments/ml/hpc/benchmark_s12_fixture.py` (new): bounded local
  comparison, including the `numpy` strategy and NumPy version provenance.
- `work/experiments/ml/research/poisson_lookup_prototype.py` (new): lookup-table
  research prototype, outside the production import graph; conditional-exactness
  and shared-offset caveats stated, duplicate-input independence probe added.
- `tests/ml/test_batched_noise.py` (new): exact, preflight, equivalence-coverage,
  and noise-regime regression checks.
- `tests/ml/test_numpy_noise.py` (new): NumPy distribution fidelity, tails,
  goodness-of-fit, spatial and view independence, per-record determinism across
  batch widths/composition/workers/start methods, dataset semantics, frozen
  validation isolation, provenance, resume rejection, and the sustained-gate
  recommendation.
- `tests/ml/test_poisson_lookup_research.py` (new): bounded checks pinning the
  lookup investigation's conclusions, the shared-offset defect, and the
  conditional nature of the dyadic construction's exactness.
- `tests/ml/test_qualify_s12_pipeline.py` (new): fixture-based gate tests —
  complete pass, missing/corrupted/non-object reports, wrong experiment or
  implementation, effective-versus-requested strategy, clean-baseline flag
  mismatch, stale and absent source provenance, invalid sustained measurement
  with and without a recommendation, scientific failure dominating
  remeasurement, loader failure, non-finite and zero throughput, zero sampled
  pairs, recomputed projection, truncated and single-partition frozen
  validation, missing/duplicated/mismatched records, NumPy-backed validation
  rejection, wrong loader, feasibility bands, an operationally marginal pass,
  worst-experiment ranking, and launch-preview path isolation.
- `docs/dev/s12_pipeline_performance.md` (new): this review and operator procedure.
- `docs/dev/notes/s12_batched_noise_proposal.md`: preserves and labels historical evidence.
- `docs/dev/lessons_learned.md`: follow-up lesson.
- `docs/README.md`: report link.
- `work/experiments/ml/hpc/README.md`: current procedure pointer.
- `work/experiments/ml/research/README.md` (new): research-prototype scope and
  corrected caveats.
- `work/experiments/ml/s12/README.md`: current procedure pointer.

No changes have been committed, deployed, or submitted; no Slurm job was queued
and no frozen artifact was touched. Suggested commit:
`Add opt-in batched and NumPy S12 photon-noise strategies`.

Remaining acceptance work: actual V4 stage profiles, frozen S12 recipe replay on
the cluster JAX build, repeated valid sustained CPU measurements for all four
conditions and aggregate worker memory, the NumPy statistical suite on the
cluster NumPy build, then the first real E01 epoch/checkpoint. Neither E01 nor
E02 is yet certified ready for a full six-run relaunch. The candidates are ready
for code review and a short cluster qualification, not an unconditional capacity
claim.

## Sustained local fixture results

Apple Silicon, Python 3.11.9, JAX 0.4.30 CPU; four forkserver workers,
160×160 float32, batch 32, 400 requested batches, eight discarded. All four
measurements pass the existing 10-batch/20-second gate. Each measures 392 batches
and 12,544 actual pairs. These are small synthetic prepared shards with a mix of
bright and ordinary images, not the 20-dimensional V4 catalog or its filesystem.
Some reference measurement overlapped regression checks; the machine was not
reserved exclusively. Treat variability and portability as unquantified.

| Fixture condition | Strategy | Batches | First batch s | Sustained s | Pairs/s | 600-epoch loader hours* | Self peak MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| E01-like | reference | 392 | 2.81 | 101.72 | 123.32 | 22.14 | 322.6 |
| E01-like | batched | 392 | 3.33 | 28.15 | 445.60 | 6.13 | 323.8 |
| E01-like | numpy | 5,992 | 1.47 | 30.75 | 6,235.59 | 0.44 | 320.6 |
| E02-like | reference | 392 | 3.37 | 193.22 | 64.92 | 42.06 | 325.1 |
| E02-like | batched | 392 | 2.99 | 53.17 | 235.94 | 11.57 | 323.2 |
| E02-like | numpy | 5,992 | 1.45 | 45.37 | 4,226.01 | 0.65 | 329.2 |
| E01-like | clean (invalid span) | 7,992 | — | 8.92 | 28,655.82 | — | 324.3 |
| E02-like | clean (invalid span) | 7,992 | — | 14.20 | 18,006.84 | — | 321.5 |

NumPy required 5,992 post-warmup batches to span the 20-second gate, versus 392
for the JAX strategies; comparing its rate against a 392-batch JAX rate is still
valid because both spans exceed the gate. Its first-batch latency is roughly
half the JAX strategies', since the **training** loader compiles no XLA kernel
per worker pool.

This does not eliminate JAX from the run. See
[JAX overhead under NumPy training](#jax-overhead-under-numpy-training): frozen
validation still uses the JAX reference sampler, and JAX is imported either way.

The two clean rows are **invalid for capacity**: even at 8,000 batches their
spans are 8.9 s and 14.2 s, below the 20-second gate. They are listed only as a
rough fixture ceiling showing that NumPy reaches 22–23% of no-noise loading on
warm synthetic shards, i.e. the noise draw still dominates locally. Do not quote
them as throughput.

*Arithmetic extrapolation of fixture throughput only; **not a V4 campaign
estimate**. Excludes repeated startup, validation, GPU work and checkpoints.
Peak RSS is the supervising probe process, not aggregate worker memory.

Reproduce from the repository root (no production artifacts written):

```bash
PYTHONPATH=src:. python work/experiments/ml/hpc/benchmark_s12_fixture.py \
 --workers 4 --implementations reference batched --batches 400
# numpy and clean need a larger sample for a valid sustained span; each report
# supplies recommended_batches_for_valid_measurement.
PYTHONPATH=src:. python work/experiments/ml/hpc/benchmark_s12_fixture.py \
 --workers 4 --implementations numpy --batches 6000
PYTHONPATH=src:. python work/experiments/ml/hpc/benchmark_s12_fixture.py \
 --workers 4 --implementations clean --batches 8000
```

Exploratory 120-batch zero-worker results were 85.29/204.14 pairs/s for
reference/batched E01-like and 48.67/109.86 for E02-like. The zero-worker
E01 batched span was only 18.81 seconds and **invalid for capacity**; those runs
also included compilation in the zero-worker timing. Four-worker exploratory
results were 128.35/454.46 and 61.87/211.66 respectively; both batched spans were
too short, motivating the 400-batch run above. Clean fixture loading took only
0.15–0.86 seconds in all configurations and is likewise invalid for capacity;
it demonstrates negligible warm fixture I/O, not fast production random I/O.
Eight workers were not tested: four already exposed the noise-kernel improvement,
and there is no evidence yet that eight helps on the eight-CPU cluster allocation.

Historical cluster reference remains E01 21.64 pairs/s (126.2 hours), E02 21.28
(128.3 hours). **No optimized cluster measurement exists in this patch.**

## Final correction policy

Batched SHERA execution requires JAX's selected backend to be CPU and fails
explicitly otherwise. The NumPy strategy has no backend constraint, because it
never calls JAX. No global JAX backend environment setting is changed. Runtime
provenance records the actual JAX backend, the JAX and NumPy versions, the
requested and effective implementations, the validation implementation, and the
NumPy BitGenerator. Resume must preserve implementation: explicit mismatches are
rejected; historical checkpoints lacking the field may resume only as reference.
There is no migration override. Scientific training identity remains unchanged,
and `noise_implementation` is deliberately absent from it.

### Three-epoch lifecycle measurement

Local 160×160, four forkserver workers, batched, 20 batches per epoch:

- E01-like first-batch seconds: 3.25, 3.20, 2.85.

- E02-like first-batch seconds: 3.12, 3.67, 3.29.

These latencies include startup, compilation and the first batch, not isolated compiler timings. Budget roughly 600 times this latency in addition to steady-state loading (about half an hour locally); measure again on Gattaca2. Short-epoch steady-state rates fail the 20-second validity gate and must not replace the longer benchmarks. Persistent workers remain disabled. Reproduce with `benchmark_s12_fixture.py --workers 4 --implementations batched --batches 20 --epochs 3`.

The final integration test uses the submission environment helper, actual
preflight loader, and one CPU training epoch with two forkserver workers on
32×32 images and E02's 0.05 second-view objective. It checks complete preflight
batches, history/checkpoint creation, runtime implementation provenance and its
absence from scientific identity. Backend guard and resume tests cover
CPU support, mocked GPU rejection, same-implementation resume, both mismatch
directions, and historical checkpoints. Default/invalid selection tests are included.

### Release corrections in this pass

**Skipped preflight (release-blocking).** `preflight_ml_gpu.py` read
`dataset.noise_implementation` in the early-return path taken when the smoke
test is skipped, before any dataset was constructed. That raised
`UnboundLocalError` for noiseless studies and for `--no-dataloader-smoke` — so
every S10/S11 preflight would have failed outright, not just S12's. The strategy
is now resolved from an explicit override or the environment through a shared
`resolve_noise_implementation` helper in `torch_data`, which needs no dataset,
and the skip report carries correct provenance. The normal noisy pathway is
unchanged except that it now passes the resolved value into the dataset instead
of letting the dataset re-read the environment independently; `training.py` does
the same, so execution and recorded provenance cannot diverge. Preflight also
gained `--noise-implementation` for operator selection. Covered by eight new
tests: noiseless default, explicit disable, both implementations on the skipped
path, explicit and environment precedence, successful noisy preflight for both
implementations, and rejection of an unknown value.

**Equivalence counting (release-blocking).** The comparison used `zip()` over
optimized and reference results, which silently compares only the shorter
sequence — a truncated or dropped record read as "exact". It is now
count-driven with explicit coverage accounting, as described under
[Reproducible checks](#reproducible-checks). Five new tests cover a
deliberately truncated batch, a duplicated record at the correct count, a
corrupted field value, independent per-partition comparison with a partial final
batch, partial-versus-complete labelling, and restoration of the dataset's
original implementation afterwards.

Final validation: `python -m pytest tests/ml -q` completed with **429 passes**
and the same previously baseline-reproduced split-validation failure (271.25 s),
independently re-confirmed on a clean `git archive HEAD` extraction of 2e44b68.
Import hygiene, report links and intra-document anchors, `bash -n` on all 18
operator command blocks, `git diff --check` (staged and unstaged), and 9/6/6
study expansion passed. The step-7 launch-preview assertion block was executed
against a real six-run preview: it passes on a correct preview and fails on a
strategy mismatch. S10/S11/S12 retain `evaluate_test: false`; S12 YAML
matches 2e44b68 exactly. Submission environment export was verified for all
three strategies. The patch is staged for review, with no commit, deployment or
job submission performed.
