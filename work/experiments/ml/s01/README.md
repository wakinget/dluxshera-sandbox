# S01 Pairwise Correction Learnability

This directory is the tracked orchestration layer for the first production ML
study.  It deliberately stays small: `study.yaml` contains the scientific
prescription, and `hpc/` contains the Gattaca2 preflight and launch wrappers.
Generated configs, frozen materializations, SLURM logs, checkpoints, and metrics
belong on scratch and are ignored under this study directory.

The canonical prepared artifact ID is `PREP-V3-v1`.  The common directory name
`PREP-V3-nuisance-v1` is the prepared dataset instance/root, not the artifact
ID recorded in catalog, split, pair-manifest, or study-contract identity.

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

- validation: `25140389d15dc4ebef35fd9cc7f9b7c368ad83c04ffdfbf96237b00188de9b55`
- test: `113993487bc0e432d0ef8d68a1f63fe5d7f20988eece6661f43774cf640d1680`

The S01 split registry is pinned by both artifact ID and content hash:

```yaml
split_registry:
  artifact_id: SPLIT-ML-v1
  content_sha256: a640e2555cb2bb55ffd3a8855ff2c587ef7c65c9990535f224ff0ed80241e35e
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

The launch script copies `run_manifest.json`, `run_config_resolved.json`,
`history.csv`, `metrics.json`, `evaluation_predictions.npz`, and
`checkpoint_best.pt`.  Large prepared datasets and transient caches should not
be copied to `/projects`.

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
`/projects` artifact tree shown above.  The batch wrapper also performs this
copy from the scratch artifacts it preflights.

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
