# Proposal: batched SHERA photon-noise generation for S12 throughput

**Status: proposal only. Not implemented, not enabled.** This exists because
S12-E02's measured data-loading cost is close to the Gattaca2 wall-clock budget.
It requires scientific review before any use, because it changes *where* noise is
applied even though it does not change *what* noise is applied.

## Problem

S12 applies SHERA observation noise per sample inside `Dataset.__getitem__` via
`jax.random.poisson`. Each call pays JAX dispatch and host-transfer overhead for
one image. E02 doubles this because it generates a second independent
realization per sample.

Local measurements (Apple Silicon, 18 cores, JAX 0.4.30 CPU backend, 160x160
images, batch size 32, `forkserver`). All rows below are **sustained
post-warmup** measurements taken with `smoke_test_dataloader`, discarding the
prefetch window and reporting only samples that pass its
`sustained_measurement_valid` gate (>=10 batches spanning >=20 s):

| Variant | workers | pairs/s | batches / span | epoch @ 16384 | 600 epochs |
|---|---|---|---|---|---|
| E01 | 0 (in-process) | 41.1 | 40 / 31 s | 6.6 min | ~66 h |
| E01 | 4 | 49.4 | 52 / 34 s | 5.5 min | ~55 h |
| E01 | 8 | 47.4 | 44 / 30 s | 5.8 min | ~58 h |
| E02 | 4 | 24.4 | 52 / 68 s | 11.2 min | **~112 h** |

**E02 projects ~112 h of data loading alone against a 72 h wall-clock limit,
before any GPU-step time.** E01 at ~55 h is inside the limit but leaves modest
margin.

Three findings that shape the options:

- **Worker scaling is essentially flat.** 4 workers give only 1.2x over
  in-process, and 8 workers are no better than 4 (47.4 vs 49.4 pairs/s, i.e.
  within noise). Limiting XLA CPU threads per worker
  (`--xla_cpu_multi_thread_eigen=false`) changed nothing, so this is not simple
  thread oversubscription. **Raising `num_workers` is therefore not a viable
  mitigation on this hardware** — a conclusion that only became visible once
  prefetch artifacts were excluded from the measurement.
- Per-call JAX dispatch appears to be the binding constraint, which is exactly
  what batching addresses.
- These are **local Mac** numbers. Gattaca2 compute-node measurements are
  required for any real decision; a different CPU, memory bandwidth, and shard
  filesystem could move them substantially in either direction. In particular,
  worker scaling may behave differently on a 8-CPU Slurm allocation than on an
  18-core laptop.

Measurement caveat worth recording: earlier short-run numbers for the same
configurations read 30-60% higher (E01 at 4 workers appeared to be 58.6 and even
130 pairs/s) because they included the startup-filled prefetch queue. Any
capacity claim must come from a run whose report says
`sustained_measurement_valid: true`.

## Proposed change

Replace the per-sample `jr.poisson` call with a `jax.vmap`-ed, `jit`-compiled
batched call over a stack of images and a stack of per-sample PRNG keys.

```python
batched_poisson = jax.jit(jax.vmap(lambda key, x: jr.poisson(key, x).astype(x.dtype)))
```

## Measured evidence

Batch of 32, 160x160, same backend:

| Path | Time per batch |
|---|---|
| per-item loop (current) | 717 ms |
| `vmap` + `jit` batched | 132 ms |

**~5.4x faster.** Critically, the outputs are **bit-for-bit identical** to the
per-item loop when given the same per-sample keys — verified with
`np.array_equal` across the full batch. The noise draw is therefore unchanged;
only the dispatch granularity differs.

Also measured and worth recording: `jit` alone, without `vmap`, gives no
speedup (22.8 ms vs 22.9 ms per image). The win is entirely from batching
dispatch, not from compilation. Likewise, constraining XLA intra-op threads had
no measurable effect, so this is not a thread-contention problem.

## Why this still needs review

Bit-identical output is necessary but not sufficient. Open questions:

1. **Where does noise get applied?** Per-sample noise currently happens inside
   `__getitem__`, which is what makes `noise_config_identity`'s
   `physical_ordering` claim ("noise is applied to count-space image arrays
   before `IntensityScaler.transform`") literally true per sample. A batched
   path must either run inside a batched dataset stage or at collate time, and
   must still precede intensity scaling. Scaling is currently applied per
   sample immediately after noise, so this reordering needs explicit care.
2. **Seed schedule provenance.** The current seeds come from
   `pair_noise_side_seeds(...)` keyed on `(pair_record_id,
   dynamic_seed_offset)`. A batched path must assemble exactly those keys in
   exactly that order. `S12-PHOTON-NOISE-VALIDATION-v1` freezes the per-record
   seed schedule and `noisy_eval.py` validates it per record, so any
   mismatch must fail loudly rather than silently reseed.
3. **Frozen validation must remain reproducible.** The frozen noisy validation
   recipe is content-hashed. Confirm a batched path reproduces it exactly,
   including under different batch sizes and worker counts — batch size must
   not be able to change the noise.
4. **Ragged final batch.** `jit` recompiles per shape; a short trailing batch
   triggers a second compilation. Confirm this does not alter results and is
   not a performance cliff.
5. **Provenance identity.** Decide whether the implementation change warrants a
   new `noise_model` identity string. Output being bit-identical argues for
   keeping `SHERA-PHOTON-NOISE-OBSERVATION-v1`; the changed application site
   argues for recording the implementation in run provenance regardless.

## Cheaper alternatives to consider first

These carry no scientific risk and should be measured on the cluster first:

- **Request a longer partition or more wall time.** Now the most promising
  no-risk option, since worker scaling does not help.
- **Raise `num_workers` from 4 to 8.** Free in principle — the Gattaca2 profile
  already allocates 8 CPUs — but measured locally at *no* benefit. Worth
  re-measuring on the cluster rather than assumed. Note this lives in
  `training`, so changing it in `study.yaml` would alter `training_identity`.
- **Checkpoint-and-resume across allocations.** E02 already writes
  `checkpoint_last.pt` every epoch and supports `--resume-checkpoint`, so a
  ~112 h campaign could run as two or three sequential jobs with reproducibility limits (global RNG state is not checkpointed). This is probably the best option if a longer partition is
  unavailable, and it needs no new code.
- **Reduce `pairs_per_epoch`.** Effective but *not* cost-free: `pairs_per_epoch`
  is part of the scientific prescription and `training_identity`, so this
  changes the experiment rather than its execution.

## Recommendation

Measure E01 and E02 on a Gattaca2 compute node with
`sustained_measurement_valid: true` before deciding anything.

Then, in order of preference: a longer partition, or checkpoint-and-resume
across sequential allocations. Adopt this proposal only if neither is available,
subject to the resume reproducibility limits in the S12 runbook.

If it is needed, implement behind an explicit, default-off configuration flag,
with tests asserting bit-identical output against the per-item path and exact
reproduction of `S12-PHOTON-NOISE-VALIDATION-v1` across batch sizes and worker
counts.

A note on sequencing: the deadlock fix is independent of this proposal and
should not wait on it. E01 fits the allocation; E02 is the case at risk.

Runtime report terminology: `sustained_batches` counts post-warmup batches;
`sustained_pairs` counts the actual pairs in those batches. The old
`sustained_samples` name counted batches and has been removed. Capacity must
still pass the existing batch-count and elapsed-time thresholds.
