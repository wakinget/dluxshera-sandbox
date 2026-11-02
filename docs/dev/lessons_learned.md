# Lessons Learned

- 2025-12-18 — Zodiax dotted-key filtering vs params containers:
  - **Symptom:** When ModelParams/SheraThreePlaneParams store external parameter names with dots (e.g., `"m1_aperture.coefficients"`), passing those dotted strings to `zdx.filter_value_and_grad` makes Zodiax treat them as structural paths, triggering missing-attribute errors during pure-mode optimization.
  - **Failed attempt:** Supplying tuple paths (e.g., `( "params", "m1_aperture.coefficients" )`) to avoid dot splitting causes `hasattr`/attribute lookups on tuples inside Zodiax's filter helpers, resulting in `TypeError: attribute name must be string`.
  - **Resolution:** For gradients over params containers, bypass Zodiax filtering and call `jax.value_and_grad` directly on the params dictionary, letting `eqx.filter_jit` mask nondifferentiable leaves. Reserve `zdx.filter_value_and_grad` for model-object gradients where dotted paths intentionally traverse the model tree.

- 2026-09-06 — TACC Lonestar6 ML job bring-up:
  - **Python module leakage:** TACC's default `python3/3.9.7` module can leak Python-3.9-era paths into a Conda environment. For isolated ML jobs, unload it when present, unset `PYTHONPATH` and `PYTHONHOME`, set `PYTHONNOUSERSITE=1`, then activate the dedicated Conda environment.
  - **A100 partition memory semantics:** The `gpu-a100-small` virtual-node partition exposes `RealMemory=1 MB` bookkeeping; a normal `--mem=64G` request was rejected. Do not request `--mem` for that partition.
  - **GPU resource flags:** Do not assume ordinary GPU `--gres` semantics on `gpu-a100-small`; the validated single-A100 production request used the partition without an additional GPU GRES request.
  - **Submission parsing:** TACC's `sbatch --parsable` wrapper can print a welcome/validation banner before the numeric job ID, while Gattaca2 Edge can return `jobid;cluster` such as `576430;edge`. Parse the final valid Slurm parsable ID and store the numeric prefix instead of assuming stdout is one pure-integer line.
  - **Slurm log directories:** Slurm opens `--output`/`--error` paths before the batch script body runs. Create log directories on the submit host and pass explicit log paths to `sbatch`; `mkdir -p` inside the batch script is too late for missing parent directories.
  - **Archive deployments:** Source trees transferred with `git archive` have no `.git`; preserve explicit source commit/archive provenance in the job environment and run artifacts.

- 2026-10-30 — S12 photon-noise training consumed six 72-hour allocations without completing epoch 0:
  - **Symptom:** All six S12 runs (source `0f20f63a`) hit the Slurm wall-time limit with `run_manifest.json` and `run_config_resolved.json` written but no `history.csv` and no checkpoint. stderr carried JAX's `os.fork() ... will likely lead to a deadlock` RuntimeWarning. CUDA-enabled PyTorch on an A100 was reported while JAX selected its CPU backend.
  - **Root cause:** S12 is the first study using `noise_model: shera_observation`, which calls `jax.random.poisson` inside `Dataset.__getitem__`. With `num_workers: 4` and PyTorch's Linux default `fork` start method, each worker inherited an already-initialised multithreaded JAX backend and deadlocked on its first JAX call. Reproduced 8/8 times locally; children reach `import jax.random` and never return from the poisson call.
  - **JAX initialises at import, not first use:** `dluxshera/utils/noise.py` imports `components.detectors`, which pulls in `dLux`, whose `dLux/utils/propagation.py` runs a module-level `jnp.zeros(...)`. The training process is therefore unsafe to fork from the moment `dluxshera.ml` is imported — long before any noise is applied. No lazy-import refactor of our own code avoids this; the start method must change.
  - **Why CI missed it — the most transferable lesson:** the deadlock is image-size dependent. Every existing multi-worker test used 8×8 fixtures, which complete under `fork`; 32×32 and larger hang. A passing multi-worker test at toy resolution proves nothing about a production-resolution pipeline. `_write_prepared_fixture` now takes `image_size`, and `tests/ml/test_s12_dataloader_runtime.py` asserts at 32×32.
  - **`forkserver` preload is a trap:** the default `multiprocessing` forkserver preload is `["__main__"]`. Because our entry points import the JAX chain at module level, preloading re-initialises JAX inside the forkserver process and restores the deadlock. Verified: with preload set, the pool hangs; with `set_forkserver_preload([])`, it passes. Always clear the preload.
  - **Memory maps pickle by value:** `ArrayShardReader` caches `np.memmap` shards, and a 205 MB shard round-trips as 205 MB of pickle. Under `forkserver`/`spawn` the dataset is re-pickled every epoch, so the datasets now implement `__getstate__` to drop the warm reader; each worker reopens its own maps.
  - **`persistent_workers=False` is load-bearing:** `DynamicPairDataset.set_epoch()` reaches workers only because the dataset is re-pickled per epoch. Enabling persistent workers would silently freeze the sampling epoch at its first value.
  - **Resolution:** select a fork-safe start method in code whenever JAX-backed noise is enabled (`resolve_dataloader_context`), **fail closed** rather than warn when `fork` is requested anyway, keep `study.yaml` byte-identical so frozen artifact identities stay valid, and make the preflight pull real batches.
  - **`DataLoader(timeout=)` does cover a stalled first batch.** An earlier draft of this fix claimed it did not, and used an in-process watchdog thread instead. Measured directly: with `num_workers=2` and a worker that never returns an item, `DataLoader(timeout=8)` raises `RuntimeError: DataLoader timed out after 8 seconds`. The timeout applies to every queue get, including the first. Production training now sets a 600 s per-batch timeout.
  - **A watchdog thread cannot bound a deadlock.** It detects the stall but cannot terminate the wedged worker, so orphaned workers accumulate at 0% CPU. Diagnostics now run each probe in a child process and kill the whole process tree (including grandchild DataLoader workers) on timeout. Verified: reproducing the real deadlock under supervision terminates in 30 s and leaks zero workers.
  - **Prefetch makes short benchmarks lie.** Workers prefetch `num_workers * prefetch_factor` batches during startup, so a short run measures queue drain, not loading. Successive measurements of the same S12-E01 configuration read 130, then 58.6, then 49.4 pairs/s as more of the prefetch window was excluded — a 2.6x spread. Throughput numbers now discard the warmup window and are certified only with >=10 post-warmup batches spanning >=20 s. The corrected numbers also reversed a mitigation: 8 workers measured no faster than 4, so "raise `num_workers`" was withdrawn as advice.
  - **Operational lesson:** CUDA availability and valid artifact hashes do not establish pipeline readiness. The original preflight validated both and still let six jobs stall before epoch 0. Any long HPC job must prove it can produce a batch, under a bounded timeout, before entering a multi-day allocation.
  - **Recovery directories collide with the refusal to reuse non-empty output.** The failed runs left manifests but no checkpoint, so neither `--resume-checkpoint` nor plain reuse works, and `--overwrite` would `rmtree` the only evidence of the incident. Restart into a **separate physical output root** while keeping the canonical logical run IDs; the run path is `<run-dir>/<STUDY>/<EXPERIMENT>/<RUN>`, so changing the root alone suffices. Mirror only the restart root into the analysis tree: `campaign_analysis` groups by `run_id` and treats differing `git.source_commit` as a conflicting identity, which warns in normal mode and raises under `--strict`.
  - **Submitted-environment allowlist:** `prepare_sbatch_submission` strips parent-shell `ML_*` variables and rebuilds the environment from `SUBMITTED_ENV_KEYS`. A new `ML_*` control that is documented but not added to that tuple is silently dropped, and the batch job runs with defaults while the operator believes otherwise.

- 2026-10-31 — S12 batching follow-up: isolated noise speedups do not certify
  full campaign throughput. Keep mixed-brightness decisions per image and preserve
  eager Gaussian arithmetic when exact rounding matters. See the
  [performance review](s12_pipeline_performance.md) for evidence and limitations.
  - **Report provenance from the source that produced it, not from a downstream
    object.** The skipped-preflight path reported
    `dataset.noise_implementation` on a branch where no dataset is ever built,
    raising `UnboundLocalError` for every noiseless study. A value needed on
    both the taken and skipped branch must be resolved before the branch.
  - **`zip()` is the wrong tool for verifying equivalence.** Comparing an
    optimized result against a reference with `zip` stops at the shorter
    sequence, so a truncated or dropped record reports as exact. Equivalence
    checks must assert counts and index coverage *before* comparing values.
  - **Pick the baseline before claiming a speedup.** The Poisson lookup-table
    investigation measured 6–10× over batched JAX, but vectorized
    `numpy.random.Generator.poisson` is itself ~4.5× faster than batched JAX for
    the same work, so the real lookup advantage was under 2× — not enough to pay
    for a gigabyte-scale table. An impressive ratio against a slow baseline is
    not evidence of a fast method.
  - **Quantizing a discrete distribution's parameter is not an approximation in
    the low-count regime; it is a different distribution.** Nearest-integer
    Poisson lookup maps every λ < 0.5 to a deterministic zero, deleting the
    photon noise from exactly the faint pixels a photon-noise study is about.

- 2026-11-01 — S12 NumPy photon-noise follow-up. See the
  [performance review](s12_pipeline_performance.md#numpy-photon-noise-investigation).
  - **Check what the reproducibility contract actually costs before trading it
    away.** The fast NumPy microbenchmark used one shared RNG over a stacked
    batch, whose stream depends on batch composition — unusable for a
    per-record determinism contract. Re-measured with deterministic per-image
    `PCG64` generators, seed derivation plus RNG construction came to 0.0002 s of
    a 0.0313 s batch: under 1%, capturing 96% of the shared-RNG ceiling. The
    contract was nearly free, and assuming otherwise would have cost either the
    speedup or the reproducibility for no reason.
  - **An unmatched benchmark ratio is not a speedup.** The 4.5× NumPy-over-JAX
    figure compared a bare vectorized RNG call against a JAX path that also did
    per-record key derivation and data assembly. On warm synthetic fixtures the
    *loader* ratio came out at 14–18× instead, because small warm shards make
    noise generation the whole cost; on real GPFS reads it will be smaller.
    State which pathways a ratio compares, and measure isolated stages and
    end-to-end delivery separately.
  - **A different RNG stream needs a different kind of test.** Bitwise
    equivalence is the right check for `batched`, which reproduces the
    reference exactly, and a meaningless one for `numpy`, where "not exact" is
    the expected result. The diagnostic now rejects the bitwise check for
    `numpy` rather than reporting a failure that says nothing; correctness there
    is established by moments, Fano factor, zero probability, tails, and
    goodness-of-fit at the λ the study actually occupies.
  - **Hash the seed before using it.** The noise schedule separates sides by
    `+17`/`+31`, which a linear seeding would turn into near-identical streams.
    NumPy routes integers through `SeedSequence`, which hashes them, so the
    measured side-stream correlation sits at the 1/√N floor. A per-record
    seeding scheme must be checked for exactly this, not assumed.
  - **A validity gate should distinguish "too slow to measure" from "too fast
    to measure".** The 20-second sustained-span gate failed NumPy because 400
    batches drained in 1.8 s, which reads as a failure and is the opposite. The
    gate now computes and reports the batch count that would satisfy it.
  - **Correlation across differing inputs cannot detect a shared random
    index.** The lookup prototype's `offset_per_level` variant draws one
    translation per count level and reuses it batch-wide, so identical expected
    images get identical "independent" draws — invisible to correlation between
    different images, obvious the moment duplicate inputs are fed in.
