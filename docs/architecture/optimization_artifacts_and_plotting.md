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
