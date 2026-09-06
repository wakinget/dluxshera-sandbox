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
