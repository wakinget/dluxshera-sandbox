# S05 Architecture / Representation Study

S05 asks whether controlled changes to the pairwise correction network's
comparator and capacity improve held-out science-state correction performance
relative to the canonical S01 baseline while keeping the benchmark data,
pair distribution, optimizer, training budget, image scaling, no-noise
condition, and frozen evaluation contract fixed.

This is not a broad hyperparameter sweep. Wave 1 changed only the comparator
or a coordinated capacity bracket and has completed its first seed-11
production pass.

## Fixed Benchmark Contract

S05 Wave 1 reuses the S01 benchmark artifacts by identity:

- prepared dataset: `PREP-V3-v1`,
  `4cdc325fbf8d4a0e07195ab075bea6f5035dfc01c9990cac03ee1f59c131e5e6`
- split registry: `SPLIT-ML-v1`,
  `29f0e95c3819cbeb5ce00aafb593445510723ea5fc20e2e7f3e585c1b9615314`
- validation pairs: `S01-VALIDATION-PAIRS-v1`,
  `68ccd41a35d286c8b060f291eef6c788a6b0d97c9660868f74e01b2b4feae499`,
  2048 ordered pairs
- test pairs: `S01-TEST-PAIRS-v1`,
  `375451064bd363a6afb33c6f3f1bdff7e92efe1384c1513b3491b42318c87b82`,
  4096 ordered pairs

The pair policy is unchanged from S01: same nuisance, different science state,
same original V3 pair-grid ID, Fisher distance in `[0, 5000]`, explicit reverse
ordered-pair augmentation, and `max_sampling_attempts: 4000`.

The frozen test artifact is locked and identity-checked but `evaluate_test:
false` remains the ordinary Wave 1 setting.  Do not use the test set for
architecture selection.

## Wave 1 Matrix

| experiment | purpose | channels | embedding | encoder hidden | head hidden | comparator | parameters |
|---|---|---:|---:|---:|---:|---|---:|
| `S05-E01-R001` | S05 reference baseline, scientifically matching S01-E01 seed 11 | `[16, 32, 64, 128]` | 128 | 256 | 256 | `concat_diff` | 767220 |
| `S05-E02-R001` | Difference-only comparator | `[16, 32, 64, 128]` | 128 | 256 | 256 | `difference` | 701684 |
| `S05-E03-R001` | Smaller coordinated capacity bracket | `[8, 16, 32, 64]` | 64 | 128 | 128 | `concat_diff` | 193540 |
| `S05-E04-R001` | Larger coordinated capacity bracket | `[32, 64, 128, 256]` | 256 | 512 | 512 | `concat_diff` | 3055060 |

All four use `normalization: batch`, `adaptive_pool_shape: [4, 4]`, AdamW,
learning rate `0.0005`, weight decay `0.0001`, batch size `32`, 8192 ordered
pairs per epoch, 100 maximum epochs, and the S01 early-stopping policy.

Parameter counts were generated with `dluxshera.ml.models.count_parameters`
for 20 science outputs.  They are trainable parameter counts, not memory or
throughput measurements.

## Wave 1 Completion Snapshot 2026-09-08

All ordinary S05 Wave 1 runs completed successfully on TACC Lonestar6:

| run | LS6 job | architecture | scheduler status | elapsed | best epoch | best validation RMSE | final validation RMSE | MSE skill |
|---|---:|---|---|---:|---:|---:|---:|---:|
| `S05-E01-R001` | 3419396 | baseline `concat_diff` | COMPLETED | 00:39:57 | 99 | 72.3262 | not recorded here | 0.916862 |
| `S05-E02-R001` | 3419397 | difference-only comparator | COMPLETED | 00:31:20 | 99 | 89.2171 | not recorded here | 0.873496 |
| `S05-E03-R001` | 3419398 | smaller `concat_diff` model | COMPLETED | 00:30:14 | 99 | 89.3951 | not recorded here | 0.872991 |
| `S05-E04-R001` | 3419399 | larger `concat_diff` model | COMPLETED | 00:30:04 | 93 | 68.4548 | 70.5641 | 0.925524 |

All meaningful persistent S05 production run directories contain
`run_manifest.json`, `run_config_resolved.json`, `history.csv`, `metrics.json`,
and `evaluation_predictions.npz`. The cancelled `S05-E01-LS6-SMOKE` job
3419329 was an infrastructure event before execution, not a failed scientific
result.

Ordinary S05 development runs used `evaluate_test: false`, so these are
validation results. The locked test artifact was not used for model selection.

`S05-E01` reproduced the `S01-E01` seed-11 baseline closely: S05 best
validation RMSE 72.3262 versus S01 seed-11 best validation RMSE 72.4681 under
the same benchmark contract. This provides a useful internal comparability
check.

The difference-only comparator (`S05-E02`) and smaller capacity bracket
(`S05-E03`) were substantially worse than the baseline. The larger
`concat_diff` model (`S05-E04`, approximately 3.055M parameters) is the best
Wave 1 architecture result, improving best validation RMSE by 5.3527% relative
to `S05-E01`.

Treat `S05-E04` as the provisional architecture winner, not a final
architecture. It has only one production seed, used the old 5e-4 / 100-epoch
training prescription, and has not yet been combined with the `S01-E03`
fixed-1e-3 longer-training prescription.

The highest-value controlled bridge experiment is therefore:

1. test the `S05-E04` architecture with the `S01-E03` optimizer/training
   prescription on the existing frozen V3 benchmark at seed 11;
2. if clearly promising, repeat at seeds 23 and 47;
3. keep the frozen test set locked during model selection.

## Evaluation Discipline

All Wave 1 variants use seed `11` so the first-pass comparison isolates the
architecture prescription under one common deterministic training seed and pair
stream. Any promising candidate should be confirmed later across multiple
seeds, and the next specific candidate is the untested `S05-E04` architecture
plus `S01-E03` optimizer/training prescription.

`S05-E01` exists so the architecture study is self-describing. Its completed
seed-11 result closely reproduces the S01 seed-11 baseline under the same
scientific/evaluation contract, but S05 remains its own study and does not
substitute S01 metrics for S05 run artifacts.

## Site-Aware Launch

Use the generic ML HPC layer under `work/experiments/ml/hpc/`; it separates the
tracked study prescription from site-specific Slurm and environment setup.
Populate paths through explicit launcher arguments rather than hard-coding
usernames.

Example dry-run command construction:

```bash
python work/experiments/ml/hpc/submit_study_run.py \
  --site tacc_ls6 \
  --study work/experiments/ml/s05/study.yaml \
  --experiment-id S05-E02 \
  --run-id S05-E02-R001 \
  --repo-root <repo-root-on-cluster> \
  --conda-env <cuda-pytorch-env> \
  --prepared-root <scratch>/data/PREP-V3-nuisance-v1 \
  --split-registry <scratch>/artifacts/S01/split/SPLIT-ML-v1.json \
  --validation-manifest <scratch>/artifacts/S01/validation_pairs/S01-VALIDATION-PAIRS-v1 \
  --test-manifest <scratch>/artifacts/S01/test_pairs/S01-TEST-PAIRS-v1 \
  --run-dir <scratch>/runs/S05/S05-E02/S05-E02-R001 \
  --persist-dir <persistent>/S05/S05-E02/S05-E02-R001 \
  --persist-artifact-root <persistent>/S01/artifacts \
  --dry-run
```

The actual job wrapper expects:

```bash
export ML_REPO_ROOT=<repo-root-on-cluster>
export ML_CONDA_SH=<conda.sh>
export ML_CONDA_ENV=<cuda-pytorch-env>
export ML_STUDY_PATH=work/experiments/ml/s05/study.yaml
export ML_EXPERIMENT_ID=S05-E01
export ML_RUN_ID=S05-E01-R001
export ML_PREPARED_ROOT=<scratch>/data/PREP-V3-nuisance-v1
export ML_SPLIT_REGISTRY=<scratch>/artifacts/S01/split/SPLIT-ML-v1.json
export ML_VALIDATION_MANIFEST=<scratch>/artifacts/S01/validation_pairs/S01-VALIDATION-PAIRS-v1
export ML_TEST_MANIFEST=<scratch>/artifacts/S01/test_pairs/S01-TEST-PAIRS-v1
export ML_RUN_DIR=<scratch>/runs/S05/S05-E01/S05-E01-R001
export ML_PERSIST_DIR=<persistent>/S05/S05-E01/S05-E01-R001
export ML_PERSIST_ARTIFACT_ROOT=<persistent>/S01/artifacts
export DLUXSHERA_SOURCE_COMMIT=<exact-source-commit>
```

For Lonestar6, submit through the `tacc_ls6` profile or
`work/experiments/ml/hpc/sites/tacc_ls6/train_ml.sbatch`.  That profile uses
`gpu-a100-small`, 1 node, 1 task, 8 CPUs per task, and 8 hours, with no `--mem`
and no normal GPU `--gres`.
