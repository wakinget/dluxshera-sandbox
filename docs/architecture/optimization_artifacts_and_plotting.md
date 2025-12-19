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
