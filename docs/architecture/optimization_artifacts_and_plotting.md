# Optimization artifacts, logging, and plotting strategy (v0)

Status: Draft (strategy-level; implementation details intentionally deferred)

## Goals

We want a refactor-era workflow where:

- Core inference operates in a **parameterization-agnostic θ-space** (primitive vs eigenmodes; whitened vs un-whitened).
- Logging and saved artifacts are **reproducible**, **scalable**, and friendly to both:
  - one-off debugging runs, and
  - large ensembles / time-series inference campaigns.
- Plotting utilities are **robust and reusable**, staying mostly agnostic to SHERA-specific semantics while still supporting SHERA-style diagnostic figures via optional “recipes”.
- The codebase remains flexible to different **optimization engines** (SGD/Adam/RMSProp/preconditioned GD/quasi-Newton) and future **sampling engines** (HMC/NUTS), with clean boundaries between:
  - objective building,
  - inference engines,
  - artifact I/O,
  - plotting.

## Core organizing concept: Trace vs Signals

### Trace (raw optimizer output)
Trace is the minimal, stable record of what the optimizer directly produces:

- Required:
  - `loss[t]` (scalar time series)
  - `theta[t, :]` (time series in optimizer θ-space)
- Optional generic diagnostics (kept minimal by default):
  - `grad_norm[t]`, `step_norm[t]`, learning-rate channel(s), accept/reject flags, etc.

Design principle:
- Trace does NOT require parameter names. θ may represent primitive parameters or eigenmodes.
- Trace is designed to remain stable even as parameterizations and packing evolve.

### Signals (derived, plot-oriented time series)
Signals are derived time series intended for plotting/comparison:

- Examples:
  - `binary.x_position_as` and `binary.y_position_as` plotted together
  - photometric diagnostics such as `raw_flux_error_ppm` derived from `(log_flux_total, contrast)`
  - Zernikes/eigenmodes plotted as components OR summarized as RMS/norm

Signals may depend on:
- a decoder (binder/spec/theta-map) to interpret θ,
- truth values (for residuals),
- domain-specific transforms.

Design principle:
- Plotting core should consume Signals (named arrays + units + optional truth), not optimizer internals.

## Plotting strategy

### Keep plotting core generic
Plotting helpers should:
- accept named arrays (`values[t]` or `values[t, k]`), units, optional truth
- support optional axis/figure injection, saving, and headless operation
- avoid embedding SHERA-specific derived computations in the plotting core

### Vector-valued histories are first-class
Vector-valued params (Zernikes, eigenmodes, star-wise flux vectors) should support selectable modes:
- “components”: plot each component as a line
- “summary”: plot RMS / norm / mean-abs as a single curve
- “hybrid”: RMS + selected top-N components (optional)

### Support diagnostic panel grouping
Legacy plotting grouped multiple quantities into one panel (e.g., x/y together, star A/B flux together).
Future grid plotting should support panel specifications:
- each panel may include one or more signals
- titles may include “final value” / “final residual” summaries for quick scanning

### Units and scaling
When possible, scaling and units should be automatic using refactor-era naming conventions and helper utilities.
Manual overrides remain supported.

## Run artifacts and logging strategy

### Per-run artifact layout: arrays vs metadata split
Use:
- NPZ for dense numeric arrays
- JSON for metadata (human-readable, stable)
- JSONL for sparse, per-step scalar logs (optional)

Recommended run directory contents:

Always-save (small, scalable):
- `trace.npz`
  - arrays: `loss`, `theta`, optional generic diagnostic channels
- `meta.json`
  - lightweight identity + θ interpretation metadata + config identifiers
- `summary.json`
  - tiny, aggregation-friendly scalars (final loss, runtime, convergence flags, etc.)

Optional diagnostics (opt-in):
- `diag_steps.jsonl`
  - sparse per-step scalar logs (log every N steps)
- `grads.npz`
  - gradients only when explicitly enabled; prefer sparse logging
- `signals.npz`
  - cached derived time series used for plotting (optional)
- `curvature.npz` / `precond.npz`
  - curvature/preconditioning artifacts for advanced optimizers (optional)

Design principle:
- Default outputs should remain small enough to support large ensembles/time-series runs.

### Experiment-level structure (multi-run / time-series)
An experiment directory contains:
- `manifest.json` describing sweep/time-series settings and dataset identifiers
- `runs/<run_id>/...` per-run artifacts (as above)
- optional aggregated table (CSV/Parquet) built from `summary.json` (+ selected `meta.json` keys)

Design principle:
- Ensemble analysis should depend primarily on `summary.json` and minimal metadata, not heavy diagnostics.

## Optimization run artifacts (v0) — schema + IndexMap

This project standardizes a small, stable set of run artifacts to support refactored optimization, plotting, and later per-parameter / block learning-rate shaping. The guiding principle is to keep the **Trace minimal and generic**, and record interpretation in **metadata** (plus optional derived artifacts).

### Design principles

- **Trace is minimal and parameterization-agnostic**:
  - Required: `loss[t]` and `theta[t, :]`.
  - No decoded/physical parameter values in Trace by default.
- **Interpretation lives in metadata**:
  - `meta.json` carries all information needed to interpret `theta`, `lr_vec`, curvature/preconditioning outputs, etc.
- **Diagnostics are opt-in**:
  - Do not save full gradient histories by default.
  - Save only small generic scalars in Trace; larger arrays go into optional files.

### Run directory layout (recommended)

    runs/<run_id>/
      trace.npz              # always
      meta.json              # always
      summary.json           # always

      # optional:
      signals.npz
      diag_steps.jsonl
      grads.npz
      curvature.npz
      precond.npz

### `trace.npz` (always saved)

**Required**
- `loss`: shape `(T,)`
- `theta`: shape `(T, D)`

**Recommended optional (generic)**
- `grad_norm`: `(T,)` (scalar norm only; no full grad vectors by default)
- `step_norm`: `(T,)`
- `base_lr`: `(T,)` (or constant repeated)
- `accepted`: `(T,)` (reserved for accept/reject methods)

### `meta.json` (always saved)

`meta.json` is the primary, human-readable record of: (a) how to interpret `theta`, and (b) what optimizer/preconditioning configuration produced the trace.

Recommended top-level fields:
- `artifact_schema`: e.g. `"dluxshera-run-v0"`
- `run_id`, `created_at` (ISO-8601)
- `git`: `{commit, dirty}` (recommended)
- `theta`: details about θ-space and its interpretation (see below)
- `objective`: objective/loss identity (name + data identifiers)
- `optimizer`: algorithm identity + step count + schedule
- `environment`: optional runtime details (jax/jaxlib/platform)

### IndexMap (stored in metadata, not in Trace)

**IndexMap** is the serialized description of the θ layout used by packing/unpacking. It is *not a separate mapping system*; it is the exported, recorded view of the packing order. It exists to make arrays like `theta[t, :]`, `lr_vec[:]`, and curvature/preconditioning vectors interpretable without guessing ordering.

**Key decision (v0):** IndexMap is stored in `meta.json`, not inside `trace.npz`.

IndexMap is represented as an ordered list of packed segments (“entries”), each defining a θ slice and its semantic meaning:
- `name`: dotted key / leaf identifier (e.g., `"binary.x_position_as"`, `"primary.zernike_coeffs_nm"`)
- `start`, `stop`: θ slice indices (stop-exclusive)
- `shape`: original leaf shape (e.g., `[27]`)
- `block`: optional conceptual grouping label (e.g., `"astrometry"`, `"primary.zernikes"`)

A stable hash (e.g., of ordered `(name, shape)` pairs) may be stored as `layout_hash` to quickly sanity-check trace ↔ decoder/spec compatibility.

    ```json
    {
      "theta": {
        "dim": 123,
        "theta_space": "primitive|eigen",
        "theta_map_id": "optional",
        "theta_map_hash": "optional",
        "index_map": {
          "layout_hash": "optional",
          "entries": [
            {"name": "binary.x_position_as", "start": 0, "stop": 1, "shape": [1], "block": "astrometry"},
            {"name": "binary.y_position_as", "start": 1, "stop": 2, "shape": [1], "block": "astrometry"},
            {"name": "primary.zernike_coeffs_nm", "start": 2, "stop": 29, "shape": [27], "block": "primary.zernikes"}
          ]
        }
      }
    }
    ```

Notes:
- “Blocks” can be as fine as per-parameter leaves (one entry per parameter), but the schema supports coarser groupings later without changing Trace.
- If numeric grouping is ever needed for fast plotting, a future optional addition is to store a `block_id_by_index` integer vector in an NPZ file; this is not required in v0.

### Optional artifacts (v0)

- `precond.npz`: optional preconditioning outputs (e.g., `lr_vec[D]`, `precond[D]`).
- `curvature.npz`: optional curvature summaries (e.g., `curv_diag[D]`).
- `grads.npz`: optional sparse debugging gradients (not default).
- `diag_steps.jsonl`: optional sparse per-step scalar logs.
- `signals.npz`: optional cached derived time series for plotting (to be specified next).

## Learning rates and curvature-based preconditioning (expanded)

This section records the strategy for moving beyond `run_simple_gd` (single global LR) toward a reusable “advanced GD” routine that supports per-parameter learning rates derived from curvature estimates (e.g., diagonal Fisher), while remaining compatible with refactor-era θ-space parameterization (primitive vs eigenmodes, whitened vs un-whitened).

### Motivation
- A single global learning rate is often sufficient when θ has been well-conditioned (e.g., a whitened eigenmode basis).
- For un-whitened eigenmodes and for primitive parameter bases, curvature can vary by orders of magnitude across θ components, making a single LR:
  - unstable for stiff directions (overshoot),
  - painfully slow for weakly constrained directions.
- The legacy codebase addressed this using an “lr_model” concept: compute curvature (FIM) and convert it into per-parameter learning rates (or a learning-rate model).

Refactor-era goal:
- Preserve the simplicity and reproducibility of “learning rates from curvature,”
- While staying agnostic to the specifics of the model (the optimizer should operate on θ),
- And keeping outputs aligned with our Trace / Meta / Summary / Signals philosophy.

### Key idea: per-index learning-rate vector in θ-space
The advanced optimizer will operate on a flat θ vector and use a per-index learning-rate vector:

- Let θ ∈ R^D
- Let g = ∂loss/∂θ ∈ R^D
- Define lr_vec ∈ R^D (per-index learning rates)

Then the update can be written generically as:

- θ_{t+1} = θ_t - (lr_vec ⊙ g_t)

This is deliberately compatible with:
- primitive θ (physical parameters),
- eigenmode θ (mode amplitudes),
- any future θ-map, as long as we can map “indices” to “meaning” for metadata/diagnostics.

### Blocks: an optional organizational layer (not a limitation)
We may optionally define “blocks” of θ indices (slices or index lists) that correspond to conceptual parameter families:
- astrometry (x/y, separation/PA, platescale)
- photometry (log_flux_total, contrast)
- wavefront blocks (primary.zernikes, secondary.zernikes)
- eigenmodes (optionally subdivided by families)

Blocks are not required to compute lr_vec (we can compute per-index LRs directly), but they are useful for:
- interpretability (human-readable mapping from θ indices to model meaning),
- plotting/diagnostics (grouped summaries),
- stability guardrails (clipping rules per family),
- future extension when the parameter set grows substantially.

Blocks are tightly related to packing/unpacking and θ-mapping: the θ layout is defined there, so those utilities are the natural place to generate an IndexMap used by the optimizer and stored in metadata.

### Curvature sources for learning-rate construction
We aim to derive lr_vec from a curvature proxy. We explicitly distinguish “what the optimizer needs” (a nonnegative vector of curvature magnitudes) from “what we call it” (Fisher, Hessian, Gauss–Newton, empirical Fisher).

Candidate curvature definitions (diagonal-only preferred):

1) Diagonal Fisher Information Matrix (FIM) (idealized target)
- FIM is often defined as: F(θ) = E[ (∂/∂θ log p(y|θ)) (∂/∂θ log p(y|θ))^T ]
- For our purposes, we typically only need diag(F), a D-vector.

2) Empirical Fisher / gradient-statistics (practical, robust)
- Approximate curvature using squared gradients:
  - diag(F) ≈ E[g^2]
- Implementations may use:
  - per-observation gradients (if loss can be decomposed),
  - or an exponential moving average (EMA) of g^2 across iterations:
    - v_t = β v_{t-1} + (1-β) g_t^2
- This is closely related to RMSProp/Adam-style preconditioning, but can be made explicit, reproducible, and savable.

3) Hutchinson diagonal estimator of a curvature operator (general scalar-loss option)
- If we only have a scalar loss L(θ), we can estimate diag(H) where H is the Hessian:
  - H = ∂²L/∂θ²
- Hutchinson estimator can approximate diag(H) without forming H:
  - diag(H) ≈ mean_k [ (H v_k) ⊙ v_k ] for random Rademacher vectors v_k ∈ {±1}^D
- Requires Hessian-vector products (HVPs), computable via autodiff (jvp-of-grad patterns).
- This works without changing the loss API, making it attractive for an initial v0.

4) Gauss–Newton diagonal (if residual structure is available or added later)
- If L(θ) ≈ 0.5 ||r(θ)||^2 with weights, then GN curvature is:
  - F ≈ J^T W J where J = ∂r/∂θ
- Diagonal:
  - diag(F)_j = Σ_i W_i (J_{i,j})^2
- This can be preferable when the loss has a strong least-squares interpretation, but may require exposing residuals or per-observation structure.

We have not committed to one curvature definition yet. The strategy is to keep the interface flexible so we can start with a low-friction option (e.g., Hutchinson-diag(H) once at init), and later evolve toward empirical Fisher or GN-diagonal variants as needed.

### Learning-rate construction from diagonal curvature
Once we have a nonnegative diagonal curvature vector curv_diag (length D), we define a per-index scaling. The default “physics-inspired” form is:

- precond_i = 1 / sqrt(curv_diag_i + eps)

Then:

- lr_vec_i = base_lr * precond_i

Where:
- base_lr is a global scale controlling overall step size
- eps prevents blow-ups when curvature is near zero

Stability guardrails (strongly recommended):
- Clip lr_vec to avoid extreme steps:
  - lr_vec_i = clip(lr_vec_i, lr_min, lr_max)
- Optionally clip the preconditioner instead of lr_vec.
- Consider a “floor” on curvature:
  - curv_diag_i = max(curv_diag_i, curv_floor)

Optional enhancements (not required initially but worth recording):
- Block-level scaling: base_lr can be replaced by base_lr[group(i)].
- Schedules: base_lr can be time-varying (warmup, decay).
- Blending: combine multiple curvature estimates:
  - curv = α * curv_hutch + (1-α) * curv_ema
- Refresh cadence: recompute Hutchinson curvature every K steps and smooth with EMA.

### Relationship to Optax optimizers (SGD / RMSProp / Adam)
- optax.sgd with a single LR corresponds to a uniform lr_vec.
- Momentum adds a velocity state that can accelerate progress in consistent directions.
- RMSProp and Adam maintain EMAs of g^2 (and g), which function like an adaptive diagonal preconditioner.
- Our strategy does not forbid using Adam/RMSProp; rather, it clarifies:
  - why they help (diagonal scaling),
  - and why we might prefer explicit curvature-derived lr_vec for:
    - reproducibility,
    - diagnostic interpretability,
    - storing and reusing the learned/estimated preconditioner,
    - alignment with legacy FIM logic.

We may still want to benchmark against Adam as a sanity check (“does adaptive diagonal scaling fix conditioning?”) even if our preferred long-term path is explicit curvature-based lr_vec.

### Loss function structure and Fisher compatibility
Current state:
- Loss functions typically return a scalar NLL: loss(theta) -> scalar.

Potential evolution paths (optional):
- Allow returning (loss, aux) where aux can include:
  - per-observation loss terms (to compute empirical Fisher by averaging per-term gradients),
  - residual vectors (to enable GN-diagonal).
- These additions should remain optional to avoid forcing all objectives to change at once.

### Metadata to record for curvature-based learning rates
To make runs reproducible and interpretable, meta.json should store enough information to reconstruct how lr_vec was produced.

Recommended metadata fields (draft; subject to refinement):
- optimizer identity:
  - optimizer name (e.g., "preconditioned_gd")
  - base_lr and schedule (if any)
  - number of iterations, convergence criteria
- curvature/preconditioning:
  - method: ("hutchinson_hessian_diag" | "ema_grad2" | "empirical_fisher" | "gn_diag" | ...)
  - eps, clipping bounds, curvature floors
  - if Hutchinson: number of probes, RNG seed, refresh cadence
  - if EMA: beta, initialization strategy
  - whether curvature is fixed-at-init or updated during run
- θ interpretation:
  - theta_space: ("primitive" | "eigen")
  - theta-map identifier/hash (if eigen)
  - ParamSpec identifier/hash
- θ layout / IndexMap summary (optional but useful):
  - mapping of θ slices to conceptual groups ("blocks")
  - this is especially helpful for debugging and future extensibility

Optional artifacts (opt-in):
- Save lr_vec and/or curv_diag arrays:
  - `precond.npz` and/or `curvature.npz`
This can be invaluable for diagnosing “why did this run behave badly?” without bloating default run outputs.

### Output expectations for the advanced GD routine
The advanced GD routine should follow our Trace/Meta strategy:
- Output a Trace containing at minimum:
  - loss[t]
  - theta[t, :]
  - plus minimal optional generic diagnostics (grad_norm, step_norm)
- Write artifacts using our run directory conventions:
  - trace.npz
  - meta.json
  - summary.json
  - optional diagnostics based on configuration flags

Design principle:
- The optimizer engine produces Trace; plotting derives Signals later via decoding and optional domain-specific transforms.
- This keeps the optimizer reusable across model variants, parameterizations, and future inference workflows.


## Advanced optimizer strategy (beyond simple GD)

### Keep `run_simple_gd` intentionally simple
- Single learning rate (baseline)
- Useful especially for whitened eigenmode bases or as a smoke-test optimizer

### Add an “advanced” optimizer tier
Motivation:
- Primitive θ-space and un-whitened eigenmode θ-space can be ill-conditioned under a single LR.

Proposed approach:
- Keep optimizer state in θ-space, but allow an optional per-index scaling:
  - `theta_{t+1} = theta_t - base_lr * p(theta_t) ⊙ grad(theta_t)`
- This supports per-parameter learning-rate behavior without requiring parameter names in Trace.

Optional organizational layer:
- θ index maps / “blocks” may be used for interpretability and guardrails (clipping/scaling per family),
  without preventing per-index scaling.

### Diagonal curvature / Fisher candidates
For a scalar NLL loss, practical diag-curvature approaches include:

- Hutchinson estimator:
  - estimate diagonal of the Hessian (or curvature operator) via random probes and Hessian-vector products
  - attractive as an initialization-time (or periodic) curvature estimate

- Empirical Fisher / gradient-statistics:
  - use squared gradients (often with EMA) as an adaptive diagonal preconditioner
  - closely related to RMSProp/Adam-style scaling, but can be made explicit/reproducible and saved

Loss structure may evolve later to expose per-observation terms/residual vectors if desired, but is not required to begin.

## Objective interfaces and sampler flexibility

### Objective builders
We aim to keep clean boundaries so multiple inference engines can plug in:

- Optimizers require:
  - `loss(theta) -> scalar` (optionally `(loss, aux)`)

- Samplers (HMC/NUTS) require:
  - `log_prob(theta) -> scalar` (often `-loss(theta)` up to a constant)
  - gradient w.r.t θ

Design principles:
- Prefer unconstrained θ-space or provide consistent transforms/bijectors.
- Store enough metadata to interpret θ and to support future sampler mass-matrix initialization.

### Relation to samplers (NUTS/HMC)
- Optimizers produce point estimates; samplers produce distributions (samples).
- Main conceptual bridge is geometry:
  - HMC/NUTS uses a mass matrix that plays a role similar to a preconditioner.
  - Diagonal curvature/Fisher estimates can inform mass-matrix initialization and diagnostics.

## Metadata: minimum useful categories (not finalized)

Metadata should be lightweight but sufficient for reproducibility:

- Run identity: id, timestamp, optional git hash
- θ interpretation:
  - θ-space type (primitive/eigen)
  - theta-map identifier/hash
  - ParamSpec identifier/hash
  - optional human-readable θ layout / index map summary
- Inference engine config:
  - optimizer name + key hyperparameters
  - if preconditioning used: method, eps, clipping, probe counts/seeds, EMA β, refresh cadence
- Optional environment/provenance: package versions, platform notes

Design principle:
- Keep metadata schema versioned and evolvable (avoid over-specifying v0).

## Signals strategy (v0) — where they live, and what belongs in Transforms vs Signals

Signals are **plot-oriented, run-context derived time series** computed from optimization outputs (Trace + metadata), typically by decoding `theta[t, :]` into physically meaningful parameters and then applying domain-specific post-processing (residuals, unit scaling, grouping). The core idea is to keep optimization artifacts stable and generic, while allowing diagnostic/plotting needs to evolve without changing the optimizer.

### Separation of concerns

#### TransformRegistry / DerivedResolver (parameter semantics)
Use registered Transforms for **truth-independent, reusable derived quantities** that belong to the model’s parameter vocabulary. A Transform should be:
- Deterministic given the Store (and resolver context)
- Meaningful without access to “true values”
- Useful as a first-class value (for loss wiring, reporting, diagnostics, or interpretation)

Transforms are *not* for plotting conventions (ppm, µas) or residuals.

**Rule of thumb:** if it’s meaningful on a single inferred state without truth, it’s a Transform candidate.

#### Signals (analysis / plotting products)
Signals are **run-level diagnostics** derived from Trace and optional truth values. A Signal may:
- Require truth values (residual/error definitions)
- Apply unit conversions and presentation scalings (µas, ppm)
- Group multiple quantities for plotting (panels with multiple overlays)
- Include vector-valued histories and their summaries (components vs RMS/norm)

Signals are allowed to be opinionated and evolve as diagnostic tastes change.

**Rule of thumb:** if it depends on truth, or is primarily a plotting/unit convention, it belongs in Signals.

### Where Signals are defined
Signals should be defined in the optimization/plotting layer (not in the parameter/transform layer). A signal builder typically takes:
- `Trace` (`theta`, `loss`, optional scalar diagnostics)
- `meta.json` (theta-space interpretation + IndexMap)
- a decoder (binder/spec/theta-map) to map `theta` → parameter values
- optional `truth` values for residuals

The output is a set of named numeric arrays (optionally cached as `signals.npz`) plus lightweight signal metadata (names, units, shapes, grouping).

### Conventions adopted for v0

#### X/Y residuals: separate signals, grouped by plot panel
For binary astrometry, compute residuals and scale to micro-arcseconds as separate Signals:
- `binary.x_error_uas`: `(T,)`
- `binary.y_error_uas`: `(T,)`

These are plotted together by defining a plotting “panel recipe” that overlays both signals on the same axis (rather than encoding `(T,2)` as a single signal). This keeps signals composable and metadata simple.

#### Plate scale and other relative errors: Signals (ppm scaling)
Quantities like plate scale error in ppm are run diagnostics and remain Signals:
- `system.plate_scale_error_ppm`: `(T,)` computed as `1e6 * (ps_est - ps_true) / ps_true` (equivalently `1e6 * ps_est / ps_true - 1e6`)

#### Raw fluxes: Transform for values; Signals for errors
Flux-related products are split into:
- Transform: `binary.raw_fluxes` (truth-independent)
  - computed deterministically from inferred primitives (e.g., `log_flux_total` and contrast) using the source model’s conversion logic
  - stored/available as a first-class derived quantity for any downstream use
- Signal: `binary.raw_flux_error_ppm` (truth-dependent)
  - computed from time series of `binary.raw_fluxes[t, :]` relative to truth and scaled to ppm:
    `1e6 * (raw_fluxes_est - raw_fluxes_true) / raw_fluxes_true`

This keeps “values” in parameter semantics and “errors/scalings” in Signals.

#### Zernikes and eigenmodes
- Zernike residuals in nm are Signals (truth-dependent) but typically require no additional unit scaling beyond subtraction.
- Vector-valued signals should support plotting modes:
  - “components” (plot each coefficient), and
  - “summary” (e.g., RMS/norm time series)
- For eigenmode runs, eigenmode residuals can be plotted in eigen space, but the preferred diagnostic path is typically:
  - map eigen θ back to primitive parameter space, then compute the standard Signals above (x/y/separation/plate scale/fluxes/zernikes).

(Next: define an introductory standard library of Signals + plotting panel recipes, and decide naming/unit conventions for saved `signals.npz`.)

## Run I/O strategy (v0) — artifact saving/loading API and minimal schemas

This project’s primary workflow is to run an optimization and immediately inspect plots and summaries. A secondary (but important) workflow is to occasionally **reload an optimized point** in a separate script to perform deeper diagnostics (e.g., gradient/FIM analysis around the best-fit parameters). The v0 I/O strategy prioritizes a **small number of stable artifacts** and a **simple, functions-first API**, with optional convenience wrappers only if/when they reduce repeated boilerplate.

### Location in codebase

Implement the run I/O utilities in:

    dluxshera/inference/run_artifacts.py

Plotting and analysis code should import I/O helpers from this module (not vice-versa), so artifact layout and loading logic remains centralized.

---

## Artifacts recap (v0)

A run directory contains the always-saved core artifacts plus optional diagnostics and caches:

    runs/<run_id>/
      trace.npz              # always
      meta.json              # always
      summary.json           # always

      # optional:
      checkpoint_best.npz
      checkpoint_final.npz
      signals.npz
      diag_steps.jsonl
      precond.npz
      curvature.npz
      grads.npz

Key principle: **Trace is minimal and generic**, while `meta.json` contains the information needed to interpret `theta` (including IndexMap). Signals and diagnostics are optional.

---

## Functions-first I/O API (v0)

The primary interface is a small set of functions (not a mandatory new class):

- `save_run(run_dir, trace, meta, summary, *, signals=None, precond=None, curvature=None, checkpoint_best=None, checkpoint_final=None, diag_steps=None, grads=None)`
- `load_trace(run_dir)` → returns trace arrays (at minimum `theta`, `loss`)
- `load_meta(run_dir)` / `load_summary(run_dir)`
- `load_checkpoint(run_dir, which="best"|"final")` → returns checkpoint payload (`theta`, `step`, `loss`, etc.)

This supports the most common workflow (inspect results immediately) while enabling the diagnostic workflow (reload best-fit state later).

### Checkpoints: enabling gradient/FIM analysis workflows

To support “load the optimized point later and analyze gradients,” v0 adds optional checkpoints:

- `checkpoint_best.npz`: best step `theta_best`, `best_step`, `best_loss` (and any minimal extras)
- `checkpoint_final.npz`: final step `theta_final`, `final_step`, `final_loss`

These allow downstream scripts to rehydrate the best/final θ state without re-running the optimizer and without requiring large trace loads.

---

## Minimal content expectations (v0)

### `meta.json` (interpretation + reproducibility)

`meta.json` should answer: **“How do I interpret θ?”** The goal is not perfect reproducibility on day one, but enough structure to decode past runs without guessing.

Minimum recommended fields (v0):
- `artifact_schema`: `"dluxshera-run-v0"`
- `run_id`, `created_at` (ISO-8601)
- `git`: `{commit, dirty}` (recommended)
- `theta`:
  - `dim`
  - `theta_space`: `"primitive"` or `"eigen"`
  - `theta_map_hash` (recommended if `theta_space == "eigen"`)
  - `index_map.entries` (IndexMap slices: `name/start/stop/shape/block`)
- `optimizer`:
  - `name`, `num_steps`, `base_lr`
  - optional `preconditioning` block if enabled (method, eps, lr_clip, refresh cadence)
- `objective`: basic identity (e.g., loss name, optional data identifiers)

Everything beyond this can remain optional/iterative during early implementation.

### `summary.json` (quick scanning and sweep aggregation)

`summary.json` should answer: **“What happened?”** It is intended for sweep tables and quick triage without loading large arrays.

Minimum recommended fields (v0):
- identity: `run_id`, `created_at`, `git.commit`
- `status`: `"ok"` or `"failed"` (and optional `message`)
- step counts: `num_steps_completed`
- loss summary: `loss_init`, `loss_final`, `loss_best`, `best_step`
- runtime: `runtime_total_s` (approximate is acceptable)
- artifact presence flags:
  - `has_signals`, `has_precond`, `has_curvature`, `has_checkpoint_best`, `has_checkpoint_final`

This schema may evolve, but these fields provide a stable baseline for run indexing and comparison.

---

## Signals caching: one file, self-contained

Signals are optional derived time series used mainly for plotting. If saved, `signals.npz` should be **self-contained** (no separate `signals_meta.json` file).

Preferred v0 approach:
- Store numeric arrays keyed by fully-qualified signal names that encode unit conventions in the name, e.g.:
  - `binary.x_error_uas`
  - `system.plate_scale_error_ppm`
  - `primary.zernike_error_nm`

This keeps the signals cache lightweight and easy to manage. A strong invalidation story is not required in v0; the decoding context is expected to be clear from run metadata and signal naming.

(Optionally, a future enhancement is embedding a single JSON string key inside `signals.npz` for richer signal metadata without creating additional files.)

---

## Optional convenience wrapper (future)

A thin `RunArtifacts` helper may be introduced later if it reduces repeated boilerplate across scripts. If added, it must remain lightweight and non-invasive (a directory handle with lazy-loading helpers), and the functions-first API should remain the canonical interface.

Avoid early “smart” behavior in v0:
- automatic binder reconstruction
- invalidation/caching frameworks
- run databases

The objective is to keep the system understandable while still making common workflows easy.

---


## Open questions / decisions to revisit
- Exact metadata schema v0 (which keys, how much provenance)
- Whether to standardize a “signals cache” format (`signals.npz`) or keep it ad-hoc at first
- Whether and how to expose per-observation loss components for empirical Fisher calculations
- How much θ layout/index-map detail to store (human-readable vs hashed identifiers)
- How to manage optional heavy diagnostics in ensemble/time-series runs

## Related docs
- `docs/architecture/inference_and_loss.md`
- `docs/architecture/eigenmodes.md`
- `docs/architecture/params_and_store.md`
- `docs/architecture/binder_and_graph.md`
