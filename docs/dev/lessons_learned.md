# Lessons Learned

- 2026-07-07 - Gattaca2 Edge launches: avoid `/scratch` aliases for cross-side environments:
  - **Symptom:** Edge-side full-fidelity campaign shards failed immediately
    with `EnvironmentNameNotFound: Could not find conda environment:
    dluxshera-py311`. The jobs completed in only a few seconds and showed
    negligible MaxRSS, around 5 MB.
  - **Root cause:** `/scratch` is side-dependent on Gattaca2. On the
    default/JPL side it resolves to `/scratch-jpl`; on Edge it resolves to
    `/scratch-edge`. The two scratch filesystems are independent and are not
    mirrored, so a Conda environment created on JPL scratch is not found when an
    Edge job interprets the same `/scratch/...` path as Edge scratch.
  - **Fix:** Source shared Miniforge and activate the environment by explicit,
    side-correct prefix:

    ```bash
    source /cm/shared/apps/miniforge/etc/profile.d/conda.sh
    conda activate /scratch-jpl/shera_hpc/dmckeith/conda/envs/dluxshera-py311
    ```

    Print `CONDA_PREFIX`, `which python`, `sys.executable`, and import checks
    for `jax` and `dluxshera` at the start of Edge sbatch jobs.
  - **Follow-up:** Run one-job smoke tests before full Edge launches. Keep
    generated submit scripts and common sbatch templates side-explicit.
    Consider creating a separate `/scratch-edge/...` environment only if we
    want Edge-local environments long term.

- 2026-07-06 - HPC Git hygiene and full-fidelity benchmark launch debugging:
  - **Symptom:** Full-fidelity runtime benchmark launch debugging mixed heavy
    campaign/model plan construction, head-node import smoke tests, Slurm shell
    initialization failures, and broad `work/` cleanup/staging commands. Even a
    dry run could build enough artifacts to stress the head node, importing
    full-fidelity/JAX scripts triggered CPU backend thread creation and aborted
    with `pthread_create` failures, and broad recursive Git operations over old
    campaign trees were slow on the shared filesystem.
  - **Resolution:** Treat full-fidelity dry runs and plan-generation paths as
    compute-node work when they build campaign or model artifacts. On the head
    node, prefer syntax-only checks such as
    `python -m py_compile examples/scripts/run_obs_subblock_study.py`; run
    import and runtime checks for JAX/full-fidelity scripts on an interactive or
    batch compute node. In Slurm scripts, initialize shells and Conda before
    enabling strict unset-variable handling: start with `set -eo pipefail`,
    temporarily `set +u` around `source ~/.bashrc`,
    `eval "$(conda shell.bash hook)"`, and `conda activate ...`, then re-enable
    `set -u`.
  - **Git hygiene:** Commit narrow source fixes immediately during debugging
    loops, such as the runtime-profiler import fix in
    `examples/scripts/run_obs_subblock_study.py`, before continuing launch
    iteration. Keep active benchmark configs and launchers in explicit,
    reviewable locations under `work/`, and keep one-off sbatch launchers out
    of the repo root by placing them under `work/slurm/...` or a
    campaign-specific directory. Do not commit generated results, exports,
    tarballs, backups, logs, or broad historical scratch directories.
  - **Shared-filesystem caution:** Avoid broad recursive commands such as
    `git add work/...` or `find work ...` over historical campaign output
    trees; Git may need to walk or hash large artifact sets. Stage explicit
    files or small directories only after confirming they are small text
    artifacts. Prefer repo-root `.gitignore` for project-wide generated and
    scratch patterns. Use `.git/info/exclude` only for personal or local clutter
    that should not affect collaborators.
  - **Recommended pattern:** Patch one issue, run only head-node-safe checks,
    commit the narrow fix, relaunch with a new run suffix, archive bulky
    generated scratch artifacts outside the repo or ignore them, and avoid
    recursive Git operations over old campaign trees.

- 2026-05-22 - Keep packaging metadata in one canonical place:
  - **Symptom:** `pyproject.toml`, `requirements.txt`, and setup docs diverged,
    which made environment setup inconsistent across local and agent workflows.
  - **Resolution:** make `pyproject.toml` canonical, keep requirements files as
    compatibility shims, and document editable install (`-e ".[dev]"`) as the
    default setup path.

- 2026-05-22 - `/usr/bin/time` portability is not enough; capability probing is required:
  - **Symptom:** On BSD/macOS hosts, `/usr/bin/time` exists but rejects GNU
    `-v`, causing wrapper subprocess failures even when child commands are valid.
  - **Resolution:** Probe GNU timing support before enabling `-v`, then fall
    back to portable timing or disabled external timing while preserving child
    return-code semantics and recording requested/effective mode in diagnostics.

- 2026-05-22 - Observation summary units are not optimizer loss units:
  - **Symptom:** A recovered-reference solve may need `subblock_reduce: mean`,
    but consuming that optimizer-normalized Schur summary as observation
    information undercounts a multi-frame subblock.
  - **Resolution:** Default Schur export records summed-likelihood information
    accounting, and real-summary forecast/update consumers require that scale
    metadata unless a legacy/debug override is explicitly recorded.

- 2025-12-18 — Zodiax dotted-key filtering vs params containers:
  - **Symptom:** When ModelParams/SheraThreePlaneParams store external parameter names with dots (e.g., `"m1_aperture.coefficients"`), passing those dotted strings to `zdx.filter_value_and_grad` makes Zodiax treat them as structural paths, triggering missing-attribute errors during pure-mode optimization.
  - **Failed attempt:** Supplying tuple paths (e.g., `( "params", "m1_aperture.coefficients" )`) to avoid dot splitting causes `hasattr`/attribute lookups on tuples inside Zodiax's filter helpers, resulting in `TypeError: attribute name must be string`.
  - **Resolution:** For gradients over params containers, bypass Zodiax filtering and call `jax.value_and_grad` directly on the params dictionary, letting `eqx.filter_jit` mask nondifferentiable leaves. Reserve `zdx.filter_value_and_grad` for model-object gradients where dotted paths intentionally traverse the model tree.
