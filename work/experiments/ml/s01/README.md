# S01 Pairwise Correction Learnability

This directory is the tracked orchestration layer for the first production ML
study.  It deliberately stays small: `study.yaml` contains the scientific
prescription, `hpc/` contains S01 compatibility launch wrappers, and
`../hpc/` contains the generic site-aware ML execution layer.
Generated configs, frozen materializations, SLURM logs, checkpoints, and metrics
belong on scratch and are ignored under this study directory.

The canonical prepared artifact ID is `PREP-V3-v1`.  The common directory name
`PREP-V3-nuisance-v1` is the prepared dataset instance/root, not the artifact
ID recorded in catalog, split, pair-manifest, or study-contract identity.  The
canonical prepared dataset hash is
`4cdc325fbf8d4a0e07195ab075bea6f5035dfc01c9990cac03ee1f59c131e5e6`.

## Status Snapshot 2026-09-06

The current S01 production launch used exact source snapshot
`5d397eca9e206180785ce4b0d1593e19878c79b7` on Lonestar6.  The tracked
`replicas.yaml` file was added after launch to document the already-submitted
derived prescriptions; it does not change the LS6 source snapshot used for
those submissions.

Lonestar6 artifact transfer and strict GPU preflight passed with the canonical
prepared dataset, split registry, frozen validation-pair, and frozen test-pair
identities recorded in `study.yaml`.  The validated environment was Python
3.11.16, PyTorch `2.11.0+cu128`, torch CUDA runtime 12.8, and an NVIDIA
A100-PCIE-40GB GPU.

A real-data training smoke test completed one epoch with `pairs_per_epoch:
256`, `batch_size: 32`, `num_workers: 4`, and `cuda:0`.  The observed epoch
time was approximately 41.4 seconds.  It wrote `checkpoint_best.pt`,
`checkpoint_last.pt`, `evaluation_predictions.npz`, `history.csv`,
`metrics.json`, `run_config_resolved.json`, and `run_manifest.json`.  The smoke
loss is not an S01 scientific result; the run only validated infrastructure.

Three otherwise-identical S01-E01 production seeds were accepted by Slurm on
TACC Lonestar6 and were pending for priority/resources at this snapshot:

| run | seed | LS6 job |
|---|---:|---:|
| `S01-E01-R001` | 11 | 3418678 |
| `S01-E01-R002` | 23 | 3418707 |
| `S01-E01-R003` | 47 | 3418708 |

R002 and R003 were created on LS6 as derived copies of the canonical S01 study
with exactly `run_id` and `seed` changed; a comparison guard verified no other
field changed.  Completion and scientific results are not recorded here unless
future result artifacts prove them.

The next controlled model-development study is S05 under
`work/experiments/ml/s05/`.

## Scientific Prescription

`S01-E01-R001` asks whether the shared CNN can learn Fisher-scaled corrections
between held-out V3 science states while generalizing across a held-out
registration-nuisance realization.  No observation noise is enabled.

The reusable study-level pair policy is `s01_clean_same_pair_grid_v1`:

```yaml
family_weights:
  same_nuisance_different_science: 1.0
same_pair_id: true
min_fisher_distance: 0.0
max_fisher_distance: 5000.0
include_reverse: true
max_sampling_attempts: 4000
```

`batch_size` is the number of ordered pair examples per optimizer step.  With
`batch_size: 32` and `include_reverse: true`, the current adjacent index
semantics normally produce 16 sampled base pairs plus their 16 reverse
directions.  Each ordered example contains `image_a`, `image_b`, and
`target_delta_z[20]`; the shared encoder processes the A and B batches with the
same weights.

Image scaling uses `global_max_abs`: one scalar is fit from at most 512 training
images, then every A/B image is divided by that same scalar.  This preserves
relative image amplitude and flux information.  It is not per-image
normalization, and `max_samples: 512` does not limit training to 512 images.

`num_workers: 4` means four PyTorch DataLoader worker processes prepare/read
batches.  `shard_cache_size: 4` means each reader keeps up to four prepared
`.npy` shard memory maps in its LRU cache.  Workers may each own a separate
reader/cache.  Persistent workers are intentionally not enabled because
`DynamicPairDataset.set_epoch()` mutates epoch state in the parent dataset.

## Frozen Evaluation

Validation is materialized once and used for checkpoint selection, early
stopping, and model-development decisions.  Test is materialized and locked, but
`evaluate_test: false` for `S01-E01-R001`.

Validation recipe:

- seed `1101`
- 512 base pairs per slice
- slices: `validation/train` and `validation/validation`
- `include_reverse: true`, giving 1024 ordered examples per slice

Test recipe:

- seed `1102`
- 1024 base pairs per slice
- slices: `test/train` and `test/test`
- `include_reverse: true`, giving 2048 ordered examples per slice

Pair manifests include a stable `content_identity.sha256` over the prepared
dataset hash, split content identity, pair policy, slice recipe, seed, and
ordered pair records.  Generation timestamps are not part of the stable
identity.

S01 pins the frozen pair manifests by content:

- validation: `68ccd41a35d286c8b060f291eef6c788a6b0d97c9660868f74e01b2b4feae499`
- test: `375451064bd363a6afb33c6f3f1bdff7e92efe1384c1513b3491b42318c87b82`

The S01 split registry is pinned by both artifact ID and content hash:

```yaml
split_registry:
  artifact_id: SPLIT-ML-v1
  content_sha256: 29f0e95c3819cbeb5ce00aafb593445510723ea5fc20e2e7f3e585c1b9615314
```

A registry with the same artifact ID but different science/nuisance assignments
does not satisfy the S01 contract.

## Distance Bins And Early Stopping

S01 diagnostics use configurable Fisher-distance edges:

```yaml
evaluation:
  fisher_distance_bin_edges: [0, 100, 250, 500, 1000, 2000, 5000]
```

Bins are `[lo, hi)` except the final bin, which includes exactly the upper edge.
Samples outside the configured range are counted separately.  Serialized metrics
store distance diagnostics as:

```json
{
  "bin_edges": [0, 100, 250, 500, 1000, 2000, 5000],
  "below_range_count": 0,
  "above_range_count": 0,
  "outside_range_count": 0,
  "bins": {
    "0-100": {"sample_count": 0}
  }
}
```

Empty configured bins remain explicit with `sample_count: 0`.

Early stopping monitors `validation_loss`, where the training loop computes
loss as squared Fisher overall RMSE.  `checkpoint_best.pt` still updates on any
strictly lower validation loss.  The patience counter resets only when the loss
improves by at least `min_delta_relative`, so tiny fluctuations do not keep a
run alive indefinitely.  Checkpoints include the early-stopping reference loss,
bad-epoch count, best loss/epoch, and completed epoch count for resume.

Resume means continuing the same logical run directory.  The run directory owns
`checkpoint_best.pt`, `checkpoint_last.pt`, `history.csv`, `metrics.json`,
`evaluation_predictions.npz`, and `run_manifest.json`.  On resume, existing
history is loaded, checked against the checkpoint epoch, and appended.  If the
resumed segment does not produce a new best validation loss, the prior
`checkpoint_best.pt`, `metrics.json`, and `evaluation_predictions.npz` remain
the authoritative best artifacts.  Resume rejects changed prepared dataset,
split-registry content, pair-policy, or frozen validation-manifest identity.

## Gattaca2 Scratch Layout

Use side-local scratch for active data and runs:

```text
/scratch-<side>/shera_hpc/$USER/dLuxShera-ML/
  data/PREP-V3-nuisance-v1/
  artifacts/S01/split/SPLIT-ML-v1.json
  artifacts/S01/validation_pairs/S01-VALIDATION-PAIRS-v1/
  artifacts/S01/test_pairs/S01-TEST-PAIRS-v1/
  runs/S01/S01-E01/S01-E01-R001/
```

Small final products can be copied to:

```text
/projects/shera_hpc/$USER/dLuxShera-Results/ml/S01/S01-E01/S01-E01-R001/
```

The launch script copies compact run products such as `run_manifest.json`,
`run_config_resolved.json`, `history.csv`, `metrics.json`,
`evaluation_predictions.npz`, and checkpoints to `ML_PERSIST_DIR`.  Large
prepared datasets and transient caches should not be copied to `/projects`.

The split registry and frozen validation/test pair manifests are small
scientific study artifacts, not transient cache.  Keep active working copies on
scratch, and also retain durable copies under:

```text
/projects/shera_hpc/$USER/dLuxShera-Results/ml/S01/artifacts/
  split/SPLIT-ML-v1.json
  validation_pairs/S01-VALIDATION-PAIRS-v1/
  test_pairs/S01-TEST-PAIRS-v1/
```

Do not copy the prepared `PREP-V3-nuisance-v1` shard store into that results
artifact tree when a canonical prepared-data copy already exists elsewhere
under `/projects`.

The S01 compatibility batch wrapper maps `S01_PROJECT_ARTIFACT_ROOT` to the
generic `ML_PERSIST_ARTIFACT_ROOT`; if unset, it defaults to
`$S01_PROJECT_RESULTS_ROOT/S01/artifacts`.  The generic preflight persists the
split and frozen validation/test artifacts only after the study contract passes,
and it rejects an existing destination with the same artifact name but a
different identity.  The compatibility wrapper treats the `S01_*` namespace as
authoritative: before delegating to the generic runner it resets generic
`ML_STUDY_PATH`, `ML_EXPERIMENT_ID`, `ML_RUN_ID`, data paths, run paths,
artifact paths, and source-provenance variables from S01 values and defaults.
Stale generic `ML_*` values from an earlier S05 submission are not preserved.
For Conda setup, explicit `S01_CONDA_SH` wins; if it is unset and
`/cm/shared/apps/miniforge/etc/profile.d/conda.sh` exists, the S01
compatibility layer maps that historical Gattaca2 setup script to
`ML_CONDA_SH`.  Otherwise the generic runner emits its normal Conda
initialization error.

## Lonestar6 Execution Notes

Lonestar6 single-A100 production jobs use the generic ML HPC wrapper under
`work/experiments/ml/hpc/` with the `tacc_ls6` site profile:

- account `JPL-PUB`
- partition `gpu-a100-small`
- 1 node, 1 task, 8 CPUs per task
- wall time `08:00:00`
- no `--mem`
- no normal GPU `--gres`

The `gpu-a100-small` partition exposes virtual-node memory bookkeeping that
rejected a normal `--mem=64G` request.  TACC's `sbatch --parsable` wrapper can
print a banner before the numeric job ID; submit helpers must parse the final
pure-integer line.

The validated launch isolation sequence unloads `python3/3.9.7` when present,
unsets `PYTHONPATH` and `PYTHONHOME`, sets `PYTHONNOUSERSITE=1`, and then
activates the dedicated Conda environment.  Source trees transferred by
`git archive` lack `.git`, so set `S01_SOURCE_COMMIT` in the compatibility
environment or use the generic submit helper's `--source-commit` option to
preserve exact source provenance.

## Operational Sequence

1. Commit the code and this prescription from a clean worktree.
2. On Gattaca2, choose the side-local filesystem:

```bash
export S01_SCRATCH_SIDE=jpl   # or edge
export S01_SCRATCH_ROOT=/scratch-${S01_SCRATCH_SIDE}/shera_hpc/$USER/dLuxShera-ML
```

3. Ensure the active environment has CUDA-enabled PyTorch:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```

4. Stage the prepared dataset to the chosen scratch side:

```bash
mkdir -p "$S01_SCRATCH_ROOT/data"
rsync -aH --info=progress2 --partial \
  /projects/shera_hpc/$USER/dLuxShera-ML/data/PREP-V3-nuisance-v1/ \
  "$S01_SCRATCH_ROOT/data/PREP-V3-nuisance-v1/"
PYTHONPATH=src python examples/scripts/build_ml_pairwise_artifacts.py inspect-catalog \
  --prepared-root "$S01_SCRATCH_ROOT/data/PREP-V3-nuisance-v1"
```

5. Place or generate `SPLIT-ML-v1.json` under
`$S01_SCRATCH_ROOT/artifacts/S01/split/`, then verify its content hash matches
`study.yaml`.

6. Materialize frozen validation and test artifacts:

```bash
PYTHONPATH=src python work/experiments/ml/materialize_study_artifacts.py \
  --study work/experiments/ml/s01/study.yaml \
  --prepared-root "$S01_SCRATCH_ROOT/data/PREP-V3-nuisance-v1" \
  --split-registry "$S01_SCRATCH_ROOT/artifacts/S01/split/SPLIT-ML-v1.json" \
  --output-root "$S01_SCRATCH_ROOT/artifacts"
```

After materialization, copy the compact study-defining artifacts to the durable
`/projects` artifact tree shown above, or set `S01_PROJECT_ARTIFACT_ROOT` before
batch submission.  The batch wrapper performs this copy from the scratch
artifacts it preflights.

7. Discover the site-valid GPU SLURM syntax before submission:

```bash
sinfo -M "${S01_SLURM_CLUSTER:-}" -o "%P %G %c %m %l" 2>/dev/null || sinfo -o "%P %G %c %m %l"
scontrol show partition
```

8. Set the CUDA PyTorch environment and verified GPU scheduling arguments:

```bash
export S01_CONDA_ENV=<cuda-pytorch-env>
export S01_GPU_SBATCH_ARGS="--partition=<gpu_partition> --gres=<gpu_resource>"
# for Edge submission, also set:
export S01_SLURM_CLUSTER=edge
```

9. Run preflight directly on an allocated GPU node or let the batch wrapper run
it before training:

```bash
PYTHONPATH=src python work/experiments/ml/s01/hpc/preflight_s01_gpu.py \
  --study work/experiments/ml/s01/study.yaml \
  --prepared-root "$S01_SCRATCH_ROOT/data/PREP-V3-nuisance-v1" \
  --split-registry "$S01_SCRATCH_ROOT/artifacts/S01/split/SPLIT-ML-v1.json" \
  --validation-manifest "$S01_SCRATCH_ROOT/artifacts/S01/validation_pairs/S01-VALIDATION-PAIRS-v1" \
  --test-manifest "$S01_SCRATCH_ROOT/artifacts/S01/test_pairs/S01-TEST-PAIRS-v1" \
  --device cuda:0
```

`preflight_s01_gpu.py` and `train_from_study.py` use the same study-contract
validation layer.  Direct production training therefore validates the prepared
catalog identity, split artifact ID and content hash, frozen validation recipe,
optional test recipe, and resolved experiment pair policy before training
starts.

10. Submit the single-GPU run:

```bash
work/experiments/ml/s01/hpc/submit_s01_e01.sh
```

The notebook remains an exploratory and post-analysis surface.  Production
training should come from the tracked prescription and saved artifacts.
