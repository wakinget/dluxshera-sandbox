# dLuxShera Working Plan & Notes (dev-facing)
_Last updated: 2026-02-10_

This is a living, dev-facing document summarizing the goals, architecture, decisions, tasks, and gotchas for dLuxShera as it moves through V1.0 and beyond. It replaces the refactor-era index while keeping the running plan in one place.

This Working Plan is the near/medium-term map for developers. For the theme-level, longer-horizon roadmap see `docs/dev/roadmap.md`. For concept-level architecture detail (ParamSpec/Store, Binder-based execution, loss/optimization, eigenmodes), use the `docs/architecture/*.md` set referenced below; this doc points to them rather than duplicating their content.

## How to use this doc
- **Sections 1–12:** Current architecture focus areas, gotchas, and open questions (developer-facing summaries with links to canonical architecture docs).
- **Sections 13–15:** Backward-compatibility notes, references, and binder namespace ergonomics.
- **Sections 16–18:** Tasks, priorities, and policy analysis (what’s done vs active P0/P1 work).
- **Sections 19–21:** Changelog and analysis/historical mappings (marked with status lines).
- **Section 22:** Merge strategy and near-term focus for V1.0.
- **Sections 23–25:** Documentation housekeeping, implementation follow-through notes, and the parking lot/backlog.
- **Historical context:** For narrative history and ADR-style rationale, see `docs/archive/REFACTOR_HISTORY.md` and `docs/architecture/adr/0001-core-architecture-foundations.md`.

## Progress refresh (2026-02)

- Roadmap priorities are largely stable this cycle; no major theme reprioritization is needed.
- Experiment workflows have advanced: prescribed Monte Carlo now has a maintained recipe entry point plus templates in `examples/recipes/prescription_templates/`.
- Experiment metadata tracking improved: experiment-level notes and per-run notes now propagate into manifest/aggregate outputs.
- Near-term focus remains optimizer robustness, regression depth, and doc/tutorial cleanup rather than major architecture rewrites.

---

## 1) Context & Problem Statement

- **Entanglement:** Primitives (e.g., `m1.focal_length`, `system.plane_separation`) and derived values (e.g., `imaging.psf_pixel_scale`) are computed in multiple places → unclear source of truth.
- **Bugs:** After optimization, `psf_pixel_scale` can be missing/incorrect in extraction; init at exact zero can lead to zero gradients.
- **Scaling pain:** Adding parameters/nodes couples logic into optics code; docs/tests lack a single schema for units/bounds/priors.

**Goal:** Cleanly separate *what parameters exist*, *where values live*, *how things are derived*, and *how the system executes*—while keeping a stable facade for users and examples.

**Target outcome (Partially met):**
- ✅ Consistent `psf_pixel_scale` (and other deriveds) regardless of whether they are optimized directly or computed from primitives.
- ⚠️ Clear primitives↔derived boundary and testable pure transforms (global registry implemented; system-scoped resolver still pending).
- ⚠️ Structured execution graph (legacy SystemGraph scaffold removed; binder-only execution is current).
- ⚠️ Minimal churn to current examples; future models (e.g., four-plane) to slot in (four-plane support missing).

---

## 2) Architecture (High-Level)

The refactor-era architecture cleanly separates **what exists** (ParamSpec), **what values are in play** (ParameterStore), **how deriveds are computed** (DerivedResolver/transform registry), and **how execution is wired** (Binder). The public model façade for Shera systems remains the binder-based PSF generator; legacy helpers wrap this internally.

- **Why this shape:** Legacy flows intertwined parameter definitions, derived computations, and execution; the new layering keeps ParamSpec/Store declarative and Binder as the sole runtime surface. Derived transforms stay pure and testable.
- **Current state:** ParamSpec/Store are in daily use with strict-by-default validation; the transform registry is scoped by system_id; Binder is the supported forward path for both two- and three-plane optics. The legacy SystemGraph scaffold has been removed.
- **Details:** For full diagrams and API notes see `docs/architecture/binder_and_graph.md` (Binder) and `docs/architecture/inference_and_loss.md` (loss stack, packing/unpacking). Eigenmode-specific context lives in `docs/architecture/eigenmodes.md`. Broader rationale sits in `docs/architecture/adr/0001-core-architecture-foundations.md`.

---

## 3) Repository Layout (Actual vs Proposed)

High-level snapshot (illustrative; run `python devtools/print_tree.py` or `python devtools/generate_context_snapshot.py` for the authoritative view):

```
dLuxShera/
├─ docs/
│  ├─ architecture/{binder_and_graph.md,eigenmodes.md,inference_and_loss.md,optimization_artifacts_and_plotting.md,...}
│  ├─ dev/working_plan.md   ← this document
│  └─ tutorials/{modeling_overview.md,canonical_astrometry_demo.md}
├─ src/dluxshera/
│  ├─ builders/{optics.py,source.py}
│  ├─ components/{optics.py,detector.py,source.py}
│  ├─ systems/{three_plane.py,two_plane.py}
│  ├─ inference/{losses.py,prior.py,numpyro_bridge.py,optimization.py}
│  ├─ params/{spec.py,store.py,transform_registry.py,packing.py,transforms.py,shera_threeplane_transforms.py}
│  ├─ optics/{config.py,builder.py,optical_systems.py}
│  ├─ plot/plotting.py
│  └─ utils/utils.py
├─ examples/{recipes/canonical_astrometry.py,recipes/twoplane_astrometry.py,runners/run_canonical_astrometry.py,runners/run_twoplane_astrometry.py}
├─ devtools/{print_tree.py,generate_context_snapshot.py}
└─ tests/
```

Use the devtools scripts above for current trees and ParamSpec/transform context snapshots; the ASCII sketch is intentionally non-authoritative.

---

## 4) ParamSpec (Schema & Metadata)

ParamSpec declares the parameter vocabulary (metadata only) for Shera systems. Forward specs mirror the truth-level binary astrometry vocabulary and optional bases (e.g., Zernike arrays sized by Noll index config); inference subspecs are views built from the forward spec and keep ordering stable for packing/unpacking.

- **Why:** This isolates schema from values and makes derived vs primitive intent explicit without coupling to runtime data.
- **Gotchas:** Derived and primitive kinds coexist in one spec; inference views reuse the same underlying definition. `ParamSpec.subset(...)` remains include-only; `ParamSpec.without(...)` complements it for drop-based ergonomics while preserving ordering and strict unknown-key handling.

For details on builders, helper APIs, and inference packing, see `docs/architecture/inference_and_loss.md`.

---

## 5) ParameterStore (Values)

ParameterStore is the immutable `{key → value}` holder registered as a JAX pytree. Validation is strict-by-default (reject derived keys unless explicitly allowed); helpers exist to refresh/strip deriveds and check consistency against a spec. Forward flows build primitive-only stores and then refresh deriveds; inference uses subspec views for packing without constructing separate stores.

- **Why:** Freezing values and being explicit about deriveds reduces stale data and keeps θ overlays predictable.
- **Gotchas:** Overrides of deriveds are possible only with opt-in validation flags; use refresh helpers when primitives change. Config dataclasses follow the same immutability pattern via `.replace(...)`.

See `docs/architecture/inference_and_loss.md` for canonical flows, and `docs/architecture/optimization_artifacts_and_plotting.md` for how stores feed logging/trace artifacts.

---

## 6) Transform Registry / DerivedResolver

Scoped `DerivedResolver` instances own per-system `TransformRegistry` objects so derived computations stay pure and system-aware. Shera three-plane transforms cover focal length, plate scale, and log flux; two-plane coverage is additive as variants land. Registration is lazy via `ensure_registered(system_id)` so callers do not need explicit side-effect imports.

- **Why:** Decouples derived math from call sites and keeps transforms testable.
- **Gotchas:** Overrides are only for opt-in debug flows; the target stance is primitives-first unless a transform is invertible.

Details and registry diagrams live in `docs/architecture/binder_and_graph.md` (transform registry section) and `docs/architecture/inference_and_loss.md` (derived refresh and loss wiring).

---

## 7) Integrating dLux `ThreePlaneOpticalSystem`

Shera binders own configs/specs/stores, expose `.model(store_delta)` as the PSF generator, and evaluate via the cached telescope + optics/source builders. Binders stay mostly immutable (`.with_store(...)`) to keep JAX friendliness.

- **Why:** Keeps execution encapsulated while letting θ overlays be the only source of dynamism. Structural caching lives in the optics builders; graph caching hooks remain future work.
- **Gotchas:** Derived values must be refreshed before binding; structural changes require a binder rebuild.

Loss wiring, Binder NLL helpers, and canonical demo usage are summarized in `docs/architecture/inference_and_loss.md` and exercised in `docs/tutorials/canonical_astrometry_demo.md`. Binder intent and design trade-offs live in `docs/architecture/binder_and_graph.md`.

---

## 8) Parameter Profiles & IO (Planned)

- Not yet implemented. Profiles (lab/instrument defaults), YAML/JSON loading, and serialization helpers remain to be built once primitives-only policy is finalized.

---

## 9) Docs & examples (Phase 1 shipped)

- Canonical binder-based astrometry recipe lives in `examples/recipes/canonical_astrometry.py` with both pure-θ and eigenmode GD flows; the two-plane companion is `examples/recipes/twoplane_astrometry.py`. Execute-first runners live in `examples/runners/run_canonical_astrometry.py` and `examples/runners/run_twoplane_astrometry.py`.
  - The demo showcases the refactor-era plotting helpers: PSF visualisation via `plot_psf_single` / `plot_psf_comparison` and parameter trajectories via `plot_parameter_history_grid`. Plotting utilities follow the IO policy (return fig/axes; caller decides to save/show) and save figures only when requested to keep tests headless.
- Current doc stack:
  - Concept orientation: `docs/tutorials/modeling_overview.md`
  - Architecture: `docs/architecture/{binder_and_graph.md,eigenmodes.md,inference_and_loss.md,params_and_store.md,optimization_artifacts_and_plotting.md}`
  - Tutorials: `docs/tutorials/canonical_astrometry_demo.md`
  - Dev-facing: this plan (`docs/dev/working_plan.md`)
  - Dev-facing: `docs/dev/style_guide.md` (Style guide)

---

## 10) Testing Philosophy

- Existing tests cover: ParamSpec/store validation and packing, transform resolution (including cycle guards), optics builder/binder smoke paths, optimization loss wrapper, eigenmode utilities, and the canonical astrometry demo in fast mode.
- Missing: Four-plane variant tests and serialization/profile coverage.

---

## 11) Gotchas & Decisions

- **Primitives-only store:** Default validation is strict; long-term policy on allowing derived overrides (debug flows) is still pending.
- **Plate-scale policy:** Whether to always recompute vs allow override is still undecided.
- **Structural caching:** Three-plane builder now caches structural builds keyed by a deterministic hash and exposes a cache clear helper (env flag available to disable caching).
- **Scopes:** Per-system scoping added via `DerivedResolver`; ergonomics for additional variants will matter as new systems arrive.
- **Zodiax dotted-key trap:** Model-parameter containers can carry external names with dots (e.g., `m1_aperture.coefficients`). Passing those names to `zdx.filter_value_and_grad` makes Zodiax interpret them as traversal paths, yielding missing-attribute errors; tuple paths also fail because Zodiax's internal `hasattr` expects strings. When taking gradients over params containers, call `jax.value_and_grad` directly on the params dict and let `eqx.filter_jit` mask non-differentiable leaves; reserve Zodiax filtering for model-object gradients where dotted paths intentionally traverse the model tree.

---

## 12) Open Questions

- Final policy for accepting derived keys in ParameterStore (`validate` default vs production enforcement).
- Whether to expose alias setters for invertible deriveds (e.g., pixel scale) or force primitive updates only.
- Canonical plate-scale handling in binder (always derived? allow override?).
- Structural hash definition for three-plane optics (which primitives are structural?).

---

## 13) Notes on Backward Compatibility

---

### Attribute Access

- **Baseline test status:** Running `pytest` requires adding `src/` to `PYTHONPATH` (or installing the package) so Binder-related tests (`tests/test_binder_smoke.py`, `tests/test_binder_shared_behaviour.py`, and Binder-backed loss/optimization tests) import `dluxshera` correctly.
- **Binder mutability/shape:** `SheraThreePlaneBinder` instances are `dataclasses` with `frozen=False` and `slots=False`; they do not present as `equinox.Module` or `zodiax.Base`, so mutation is guarded only by convention rather than framework-level immutability.
- **Store surface area:** `binder.base_forward_store` is a `ParameterStore` exposing `get/keys/items/values/as_dict/replace/validate_against` and similar mapping semantics.
- **Derived placement:** With the default forward spec + refreshed forward store (via `tests.helpers.make_forward_store`), derived values such as `system.plate_scale_as_per_pix` are present directly in the base forward store prior to any evaluation.

- `SheraThreePlane_Model` remains the public entry point; new plumbing should remain internal to avoid churn in existing scripts.
- Legacy optics helpers still carry pre-refactor pathways; the refactor must avoid breaking current examples until replacements land.

---

## 14) Prior Art / References

- dLux core APIs for `ThreePlaneOpticalSystem` and PSF generation.
- Prior optimization scripts in `examples/` (still legacy-style; to be updated after binder workflows stabilize).

---

## 15) Binder namespace ergonomics (Task 1A–1E status)

- Status: binder read ergonomics are now Binder-first and contract-driven. Supported access patterns include `binder.get(...)`, `binder.ns("prefix")`, contract-driven semantic leaves (runtime-first, store fallback), runtime component access (`binder.source/optics/detector`), and runtime-leaf fallback for unique component leaves. `binder.cfg` remains explicit provenance, not part of the fallback chain. Pretty-printing now foregrounds source/optics/detector.

---

## 16) Tasks & Priorities

Legend: ✅ Implemented · ⚠️ Partial · ⏳ Not implemented

**Completed to date (highlights)**  
- ✅ ParamSpec ergonomics (`subset` include-only, `without` for drops) with regression tests.  
- ✅ ParameterStore strict-by-default validation, derived refresh/strip helpers, and shallow serialization.  
- ✅ Inference packing/unpacking (θ ↔ store delta) with tests.  
- ✅ Scoped transform registry + Shera plate-scale/log-flux transforms.  
- ✅ ThreePlaneBuilder structural hash + cache/clear helper.  
- ✅ Binder-first loss wiring (Binder NLL helpers using `gaussian_image_nll`) and binder namespace UX (Task 1A–1E).  
- ✅ SystemGraph single-node scaffold (removed from mainline; retained in archive history).  
- ✅ Binder NLL stationary-point regression landed; follow-on scenarios pending (multi-wavelength/multi-PSF).  
- ✅ Prescribed Monte Carlo workflow promoted into examples with maintained templates and updated naming (`overrides.csv` semantics, notes propagation).
- ✅ Aggregation metadata improvements landed (`run_note` plus experiment-level notes in manifests/results summaries).

**P0 — Current focus**  
- ✅ **Optimization artifacts & logging**: Phase A scaffold (`run_artifacts.py`) is in place and Phase B wiring now emits required artifacts from `run_simple_gd` and binder-backed `run_image_gd` when opt-in flags are provided. Integration smoke tests cover end-to-end writes and metadata (trace/meta/summary + optional checkpoints).  
- ⚠️ **Optimizer control (learning-rate shaping)**: θ-space preconditioning is now available via `PreconditioningConfig` + `compute_precond_vectors` and is wired into `run_image_gd` with optional artifact capture (curvature/precond). The current implementation uses an `ema_grad2` diagonal curvature estimate and does not yet expose FIM-based methods or auto-derivation inside `run_simple_gd`.  
  - TODO: Add FIM-derived preconditioning in `PreconditioningConfig` (e.g., `method='fim_diag'`) so the same configuration both computes and applies the preconditioner, rather than relying on external `lr_vec` scripts.  
  - TODO: Extend `run_simple_gd` (or a successor) to optionally compute/apply preconditioning for non-Shera loss functions, keeping parity with artifact logging.
- ⚠️ **Loss regression hardening**: Keep the landed Binder NLL stationary-point regression; add coverage for multi-wavelength / multi-PSF scenarios as new demos land and surface any remaining edge cases.

**P1 — Next up**  
- ⏳ **Profiles/IO and serialization**: YAML/JSON profiles and richer serialization once primitives-only policy remains stable.  
- ⚠️ **Documentation and example polish**: canonical and prescribed Monte Carlo docs improved, but README quickstart and tutorial cross-links are still incomplete.
- ⏳ **Expanded transform coverage**: Broaden registry coverage for additional systems (two-/future four-plane) as specs land.

**P2 — Variants & ergonomics**  
- ⏳ Four-plane variant (specs, transforms, builder, resolver tests).  
- ⏳ Ergonomic shims (`ModelParams`), deprecation path for legacy APIs, upstream PR prep.

---

## System / Binder / Builder Reorganization Plan (Phased)

This section captures the agreed-upon plan for reorganizing the `dluxshera` codebase around a
clear separation between **components**, **builders**, and **systems**, with the Binder acting
as the authoritative system object that owns a persistent cached telescope.

This plan is intended to be executed incrementally. The working plan should be updated as each
phase is completed (checkboxes, notes, or links to commits), so this document remains the
authoritative record of architectural intent.


---

### Architectural Goals (Summary)

We are reorganizing the codebase to:

- Improve discoverability: make it obvious where two-plane and three-plane systems are defined.
- Reduce cross-file scatter of plane-specific logic.
- Clarify layering and ownership:
  - **components/** → dLux-compatible classes we own (Optics / Sources / Detectors)
  - **builders/** → assembly logic, runtime bindings, structural hashing
  - **systems/** → Binder + Config + presets + forward spec builders
- Evolve the Binder to hold a **persistent cached `dl.Telescope`**, with:
  - per-component structural vs runtime parameter policies,
  - a clear conflict rule: *if any component says “structural”, the parameter is structural*.


---

### Phase 0 — Architecture Freeze (Design Only)

**Goal:** Lock the mental model before touching code.

Status:
- Three-layer architecture agreed (components / builders / systems).
- Binder semantics agreed (cached telescope, runtime vs structural update logic).
- Legacy builders will live under `legacy/`.
- Forward spec builders move out of `params/spec.py` into `systems/`.
- `components/detectors.py` will be plural.
- No backward-compatibility shims required during refactor.

Deliverables:
- This working plan section (kept up to date).
- No code changes yet.

Documentation:
- This section is the authoritative design reference.


---

### Phase 1 — Create New Package Skeleton (No Behavior Changes)

**Goal:** Introduce the new directory structure without changing semantics.

Status:
- ✅ Complete (topology-only updates; broken imports are acceptable at this phase).

Actions:
- Create new packages:
  - `src/dluxshera/systems/`
  - `src/dluxshera/builders/`
  - `src/dluxshera/components/`
- Create placeholder files:
  - `systems/base.py`
  - `systems/two_plane.py`
  - `systems/three_plane.py`
  - `builders/optics.py`, `builders/source.py`, `builders/detector.py`
  - `components/optics.py`, `components/sources.py`, `components/detectors.py`
- Move `optical_systems.py` → `components/optics.py` (verbatim).
- Move legacy builder logic into `legacy/builders.py`.

Rules:
- No refactoring of logic.
- Imports may temporarily break.
- Focus is topology, not correctness.

Tests:
- Not required to pass during this phase.

Documentation:
- Update this working plan with notes once skeleton is in place (topology-only, imports may be broken).


---

### Phase 2 — Systems Layer Extraction (Binder + Config + Presets)

**Goal:** Make `systems/` the authoritative home of plane-specific logic.

**Status:** ✅ Complete — BaseConfig/BaseSheraBinder now live in `systems/base.py`, with two-plane/three-plane configs, binders, and presets co-located in `systems/two_plane.py` and `systems/three_plane.py`. Imports now flow through `systems/` (with legacy re-exports in `optics/config.py`). 

Actions:
- Move `BaseBinder` into `systems/base.py`.
- Move:
  - `SheraTwoPlaneBinder` → `systems/two_plane.py`
  - `SheraThreePlaneBinder` → `systems/three_plane.py`
- Move configs out of optics:
  - `SheraTwoPlaneConfig` → `systems/two_plane.py`
  - `SheraThreePlaneConfig` → `systems/three_plane.py`
- Move named presets (testbed / flight) alongside their systems.

Outcome:
- Opening `systems/two_plane.py` or `systems/three_plane.py` should give a complete picture
  of that system’s configuration and Binder.

Tests:
- Fix imports and ensure basic scripts can import the new system modules.
- Run:
    PYTHONPATH=src pytest -q

Documentation:
- Update `code_structure.md` and/or `binder_and_graph.md` if references to legacy `core/` remain.


---

### Phase 3 — Builder Consolidation and Ownership

**Goal:** Make `builders/` the single source of truth for component assembly logic.

Status: ✅ Complete — optics, source, and detector assembly are consolidated under `src/dluxshera/builders/` with binders delegating to these builders.

Actions:
- Move optics builder implementation into `builders/optics.py`:
  - build functions for two-plane and three-plane optics
  - runtime bindings
  - structural hashing / structural subset helpers
  - optional builder-level caches
- Move `build_alpha_cen_source` into `builders/source.py`.
- Add trivial `builders/detector.py` for symmetry.
- Remove assembly logic from binders and components.

Outcome:
- Binders *call* builders.
- Components define objects, not construction.
- Runtime binding logic lives with the builder.

Tests:
- Update system binders to use builders.
- Run:
    PYTHONPATH=src pytest -q

Documentation:
- Update this working plan to mark Phase 3 complete.


---

### Phase 4 — Spec Ownership Realignment

**Goal:** Align spec construction with system ownership.

Status: ✅ Complete

Actions:
- Keep in `params/spec.py`:
  - `ParamField`
  - `ParamSpec`
  - `build_inference_spec_basic`
  - `make_inference_subspec`
- Move forward spec builders out of `params/spec.py`:
  - two-plane forward spec builder → `systems/two_plane.py`
  - three-plane forward spec builder → `systems/three_plane.py`
- Optionally normalize naming (e.g. `build_forward_spec_from_config`).

Notes:
- Inference spec remains shared across systems for now.
- A `systems/spec_utils.py` may be introduced later if duplication emerges.
- Forward spec builders now live in `systems/` with system-specific ownership.
- `params/spec.py` now only contains spec types and generic inference helpers.

Tests:
- Update imports in scripts and binders.
- Run:
    PYTHONPATH=src pytest -q

Test notes:
- `PYTHONPATH=src pytest -q` reported failures in inference/optics/params/devtools tests
  (e.g., legacy model validation, signal plotting, plate scale runtime checks, and
  ParameterStore/refresh_derived expectations). These are tracked for follow-up in
  later phases.

Documentation:
- Update any docs referencing `params/spec.py` forward builders.


---

### Phase 5 — Params Consolidation

**Goal:** Reduce cognitive overhead in the params subsystem.

Status: ✅ Complete

Actions:
- Merge `StoreNamespace` into `params/store.py`.
- Create `params/transform_registry.py`:
  - move registry classes, resolver logic, dependency/cycle checks here.
- Move all actual transform functions into `params/transforms.py`
  (including contents of `shera_threeplane_transforms.py`).

Notes:
- `StoreNamespace` now lives alongside `ParameterStore` in `params/store.py`.
- Transform registry/resolver plumbing moved to `params/transform_registry.py`;
  concrete transforms live in `params/transforms.py`.

Tests:
- Verify derived parameter resolution still works.
- Run:
    PYTHONPATH=src pytest -q

Documentation:
- Update `params_and_store.md` if needed.


---

### Phase 6 — Binder Behavior Evolution (Intentional Behavior Change)

**Goal:** Implement the new Binder semantics cleanly, now that structure is stable.

**Status:** ✅ Complete (Binder now caches telescopes, applies runtime updates per component, and enforces explicit rebuild policy for structural changes.)

Actions:
- Finalize cached telescope behavior.
- Centralize structural vs runtime detection logic.
- Apply runtime bindings per component (source / optics / detector).
- Enforce rebuild policy explicitly.
- Remove redundant optics-only caching inside Binder if telescope is cached.

This is the phase where behavior changes are expected and intentional.

Tests:
- Add or update tests covering:
  - runtime updates
  - structural rebuild detection
  - error behavior when rebuild is disallowed
- Run:
    PYTHONPATH=src pytest -q


---

### Phase 7 — Validation, Tests, and Documentation Alignment

**Goal:** Ensure the new architecture is legible and stable.

Actions:
- Update:
  - `binder_and_graph.md`
  - `code_structure.md`
  - any architecture diagrams or notes
- Ensure examples and scripts use the new system modules.
- Use tests and example scripts as validation (not backward compatibility).

Status:
- ✅ Complete.
- ✅ Overall reorganization effort complete (components/builders/systems layering in place).

Post-Refactor Notes:
- **Key outcomes:** Binders are the canonical system surface; builders own structural caching and runtime bindings; ParamSpec/Store/transform registry provide schema + value + derived plumbing with strict validation defaults.
- **Known limitations:** Preconditioning remains heuristic (EMA of squared gradients); multi-PSF/multi-wavelength regression coverage is still partial; profiles/IO are still pending.
- **Suggested follow-ups:** add FIM-based preconditioning options, expand transform coverage for additional system variants, and formalize profile/serialization workflows.


---

### Test Execution Notes

All tests should be run from the repository root using:

    PYTHONPATH=src pytest -q

This should be noted whenever new tests are added or test behavior changes during the refactor.





---

## 17) Recommended Next 3–5 Tasks (to reach end-to-end flow)

1. **Add SystemGraph + DLuxSystemNode scaffold (P0) — DONE**
   - **Outcome:** Added `graph/` package with `DLuxSystemNode` + `SystemGraph`, regression-tested against the legacy three-plane forward path.
   - **Follow-ups:** This scaffold has been removed; any future graph work should re-evaluate caching/multi-node needs before reintroducing it.

2. **Scoped DerivedResolver with system IDs (P0) — DONE**
   - **Outcome:** Added `params/transform_registry.py` with system-scoped resolver/decorator, defaulting to the Shera three-plane system; tests cover isolation across system_ids and existing Shera transforms continue to resolve via the default registry.
   - **Follow-ups:** Extend coverage for future system variants (two-/four-plane) once their specs land and align ergonomics with ParameterStore primitives-only enforcement.

3. **ParameterStore enforcement + serialization (P0) — DONE**
   - **Outcome:** ParameterStore validation defaults to strict (rejects derived keys), with opt-in override for debug/override flows plus helpers to strip/refresh/check derived values. Shallow serialization (`from_dict`, `from_spec_defaults`, `as_dict`) is in place; YAML/JSON profile IO remains deferred to the profiles/IO task.
   - **Follow-ups:** Add optional YAML/JSON helpers alongside the profile/IO workstream if still desired.

4. **Structural hash/cache for ThreePlaneBuilder (P1) — DONE**
   - **Outcome:** Structural subset documented in `optics/config.py`; deterministic hash helper and cache/clear APIs added to `optics/builder.py` (env flag `DLUXSHERA_THREEPLANE_CACHE_DISABLED`). Tests cover cache hit/miss, non-structural reuse, and hash stability.
   - **Follow-ups:** Consider exposing cache stats and integrating hash/caching if a future graph layer is reintroduced.

5. **Canonical astrometry demo + docs (P1)**
   - **Status:** ✅ Added `examples/recipes/canonical_astrometry.py` (with runner at `examples/runners/run_canonical_astrometry.py`) using ParamSpec + ParameterStore + DerivedResolver to build truth/variant stores and a SheraThreePlaneBinder forward model, plus Optax GD with prior penalties. README updated with run command; smoke test exercises `main(fast=True)`.
   - **Two-plane companion:** Added `examples/recipes/twoplane_astrometry.py` and `examples/runners/run_twoplane_astrometry.py` as lighter-weight analogues that exercise the SheraTwoPlaneConfig/Binder stack; both demos serve as reference examples for upcoming docs/tutorials.

---

## 18) ParamSpec + ParameterStore policy (plate_scale/log_flux) — analysis & options

**Current behavior (“split personality”)**
- `build_inference_spec_basic()` marks `system.plate_scale_as_per_pix` and `binary.log_flux_total` as **primitive knobs** for optimisation.【F:src/dluxshera/params/spec.py†L95-L167】【F:src/dluxshera/params/spec.py†L207-L245】
- `build_forward_spec_from_config()` (in the system modules) mirrors geometry/throughput primitives from `SheraThreePlaneConfig` and declares `system.plate_scale_as_per_pix` and `binary.log_flux_total` as **derived** with registered transforms (geometric plate scale and collecting-area × band × throughput flux).【F:src/dluxshera/systems/three_plane.py†L456-L656】
- The transform registry is **store-wins**: if a key is present in the `ParameterStore`, the transform is skipped; otherwise dependencies are resolved recursively.【F:src/dluxshera/params/transform_registry.py†L1-L200】 Tests exercise this by computing plate scale/log flux from a forward-model store seeded with primitives only.【F:tests/optics/test_shera_threeplane_transforms.py†L1-L120】
- `ParameterStore.from_spec_defaults()` skips derived fields, so a forward-model store built from defaults contains only primitives unless the caller injects derived values. Validation now rejects derived keys by default (unless `allow_derived=True`) and provides `refresh_derived`/`strip_derived` helpers for deterministic recomputation.【F:src/dluxshera/params/store.py†L72-L167】【F:src/dluxshera/params/store.py†L202-L251】

**Practical interactions & risks**
- Forward-model workflows: build spec → seed store from defaults → run transforms to fill plate scale/log flux; these derived numbers are often copied into an inference store and then treated as primitives. If a caller modifies primitives (e.g., focal lengths) without recomputing, a stale derived could persist because validation allows it and the resolver will return the stored value.
- Override behavior is currently used in tests and is convenient for debugging (e.g., dropping a plate scale directly into the store to avoid recomputing). Removing it abruptly would break that ergonomics.

**Design options (targeting plate_scale/log_flux but generalisable)**
- **Option A — Formalize current split (ForwardModelSpec derived, InferenceSpec primitive):**
  - Keep forward spec transform-driven for geometry/flux; document that inference spec treats the same keys as primitives. Add `ParameterStore.validate_against(spec, allow_derived=False)` defaulting to strict mode; add an `override=True` flag or `allow_derived=True` path for expert flows. Provide a `refresh_derived(spec, store, system_id)` helper so callers can recompute deriveds before serialization/copying. Pros: minimal churn, preserves geometry-based truth generation; keeps override ergonomics opt-in. Cons: mental overhead that key “kind” depends on spec; requires discipline to call refresh when primitives change.
- **Option B — Align specs on primitives for mainline runs:**
  - Make ForwardModelSpec treat plate_scale/log_flux as primitives too; keep transforms registered but reserve them for specialised “physics-mode” specs (new builders) that explicitly mark those keys as derived. Default validation enforces primitives-only; transform-driven specs opt into derived resolution. Pros: reduces spec-dependent semantics for common keys; simplifies training/ops flows. Cons: truth-generation loses automatic geometry derivation unless callers pick the physics spec; more boilerplate when wanting geometric values.
- **Option C — Hybrid calibration stance:**
  - Keep transforms for structural/geometry-only quantities (e.g., focal_length, maybe plate_scale) but treat user-facing calibration knobs (log_flux, effective plate scale) as primitives in both specs. Use transforms in forward spec only for upstream structural nodes; log_flux becomes a primitive there. Pros: avoids stale brightness overrides; focuses derivations on geometry. Cons: still split semantics for plate scale unless it is also made primitive; may underuse existing flux transform.

**Recommendation & incremental path**
- Prefer **Option A** short term, with strict-by-default validation and explicit override/debug mode to bound risk while keeping current workflows working. Rationale: users expect inference on effective knobs, but forward-model truth generation benefits from existing geometry/flux transforms; the cost of extra keys (geom/eff/final) feels high relative to spec-as-mode clarity.
- Near-term steps (P0/P1):
  1. Implement `validate_against(..., allow_derived=False)` default and `refresh_derived` helper to recompute deriveds for a given spec/system_id before copying/serialization.
  2. Add docstrings/comments in spec builders explaining the primitive/derived split and pointing to the override flag.
  3. Add tests for stale-override prevention (strict validation rejects derived keys when flag is false) and for refresh correctness.
- Progress update: strict `ParameterStore.validate_against` now rejects derived keys by default, and helper utilities `strip_derived`, `refresh_derived`, and `check_consistency` are available to manage override/debug flows and recompute derived values deterministically.
- Deferred/experimental: add optional “physics-mode” inference/forward spec builders that mark plate_scale/log_flux as derived; keep override flag for manual injection; revisit whether log_flux should migrate to primitive in forward spec if calibration use-cases dominate.
- Diagnostics & UX (Option C — Phase 1): plotting utilities have been normalised to a refactor-era IO policy. PSF visualisation (`plot_psf_single`, `plot_psf_comparison`), parameter history panels (`plot_parameter_history`, `plot_parameter_history_grid`), colourbar alignment (`merge_cbar`), OPD and sweep helpers now return figures/axes, avoid implicit `plt.show()`, and support explicit `save_path` for headless/CI usage. Open follow-ups: dedicated FIM/eigen visualisation helpers (spectra + loadings), richer `ParameterStore` inspection/pretty-printing, and a simple logging/trace container that integrates with plotting and canonical demos.

---

## 19) Changelog of Decisions

See `docs/architecture/adr/0001-core-architecture-foundations.md` for a curated, ADR-style summary of the major choices referenced here.

- Transform registry is now system-scoped (defaulting to Shera three-plane) with lazy registration.  
- ParameterStore validation is strict-by-default (primitives-first) with opt-in derived overrides and refresh helpers.  
- Binder-first loss wiring is the canonical inference path; binders execute via the cached telescope/optics builders.  
- ThreePlaneBuilder structural hashing + caching shipped; graph-level caching/derived hooks remain future work.

---

## 20) Legacy Shera two-plane stack → refactor-era mapping (analysis)
Status: analysis + historical mapping; two-plane refactor implementation now landed (see follow-ups below). Implementation overview lives alongside the three-plane stack in `docs/architecture/binder_and_graph.md`.

For a concise mapping of legacy APIs to the current architecture, see `docs/archive/LEGACY_APIS_AND_MIGRATION.md`.

**Current two-plane parameter vocabulary and behavior (legacy `SheraTwoPlaneParams`/`SheraTwoPlane_Model`)**
- Point designs expose primary/secondary diameters, PSF pixel scale (primitive, arcsec/pix), bandpass width, and log flux; operational knobs include pupil/PSF sampling, binary astrometry (x/y offsets, separation in mas, PA in deg, contrast), central wavelength, number of wavelengths, and a single Zernike basis (Noll indices + amplitudes) applied to the primary pupil mask. Noise fields include calibrated/uncalibrated 1/f power-law/amplitude pairs. No explicit plate-scale derivation occurs; pixel scale is passed straight into the optics builder.【F:src/dluxshera/inference/optimization.py†L1224-L1312】
- Binary astrometry mirrors the three-plane vocabulary (x/y offsets, separation, PA, contrast, log_flux) and is forwarded to the `AlphaCen` source; no secondary-mirror parameters appear. Flux is handled as a stored log_flux scalar (no transform), and plate scale is treated as a primitive `psf_pixel_scale` handed directly to the optics and detector sampling (oversample=1).
- The optics path is Toliman-like: a two-plane `SheraTwoPlaneOptics` fed by wavefront/PSF sampling, oversample, pixel scale, aperture diameters, strut geometry, diffractive pupil (dp_design_wavel), and optional Zernike basis; primary Zernikes are normalized to nm before setting coefficients. Detector is a simple downsample layer; PSF sampling equals the provided oversample (hard-coded to 1 in the model).【F:src/dluxshera/optics/optical_systems.py†L95-L190】

**SheraTwoPlaneOptics vs TolimanThreePlaneSystem**
- SheraTwoPlaneOptics is an `AngularOpticalSystem` with only aperture + diffractive pupil layers, optional Zernike basis on the primary, and propagator knobs for PSF sampling, oversample, and pixel scale. It uses primary/secondary diameters and strut geometry to build a single pupil; no secondary mirror surface/aberrations or Fresnel relay are present. Aberrations are strictly Zernike-based (no 1/f WFE), and the diffractive pupil is loaded from a numpy mask and converted to an aberrated layer. This mirrors Toliman-style two-plane optics but with Shera-specific defaults (diameters 0.09/0.025 m, four struts at -45°, 550 nm design wavelength).【F:src/dluxshera/optics/optical_systems.py†L95-L190】

**Two-plane vs three-plane comparison**
- Shared: binary astrometry/flux knobs, central wavelength + bandwidth + wavelength sampling, pupil/PSF grid sizes, primary Zernike basis (nm-scaled), and calibrated/uncalibrated 1/f knobs (though the three-plane applies them to both mirrors). Both models hand binaries to `AlphaCen` with the same argument set and normalize Zernikes on the primary.
- Three-plane-only: explicit mirror focal lengths, plane separation, detector pixel size, derived plate scale (EFL from two-mirror relay plus pixel size), and full secondary mirror aperture with its own Zernike basis and 1/f layers. The optics builder constructs a Fresnel relay (`SheraThreePlaneOptics`) and adds separate calibrated/uncalibrated WFE layers for both mirrors. Detector sampling uses `oversample=1` but plate scale comes from geometry unless overridden.【F:src/dluxshera/inference/optimization.py†L1224-L1312】
- Two-plane-only: primitive PSF pixel scale passed directly into `SheraTwoPlaneOptics`; no secondary mirror geometry/aberrations; no plane separation/focal lengths; 1/f maps inserted but only after the single aperture layer. The pipeline is strictly pupil → focal plane without Fresnel relay.

**Mapping plan to refactor-era concepts**
- *Optics naming and placement*: Rename the two-plane optics class to `SheraTwoPlaneOptics` and the three-plane class to `SheraThreePlaneOptics` (both in `optics/optical_systems.py`) to align with Shera family naming.
- *Config + forward spec*: Introduce `SheraTwoPlaneConfig` alongside the three-plane config, sharing binary vocabulary, wavelength/bandwidth sampling, and primary Zernike basis fields. Two-plane-specific primitives: `psf_pixel_scale` (primitive, arcsec/pix), primary aperture geometry (p1/p2 diameters, strut geometry, diffractive pupil design wavelength), sampling (`pupil_npix`, `psf_npix`, `oversample`). Exclude three-plane-only fields (focal lengths, plane separation, detector pixel size, secondary basis/1/f). Forward spec should mirror the binary vocabulary used by the three-plane builder (unit-aware `binary.x_position_as`, `binary.y_position_as`, `binary.separation_as`, `binary.position_angle_deg`, `binary.contrast`), include `optics.primary.zernike_coeffs_nm` when a basis is configured (default zeros), omit any secondary terms, treat `psf_pixel_scale` as primitive (no transform), and derive `binary.log_flux_total` via the same transform family as the three-plane system.
- *Inference spec sharing*: Provide a shared “Shera astrometry inference spec” builder that covers the common binary vocabulary, primary Zernike coefficients, and plate scale as a primitive knob. Secondary-specific keys (secondary Zernikes) should be included only for three-plane runs; callers can drop them via `ParamSpec.without(...)` for two-plane cases. From inference’s perspective, both systems remain `dl.Telescope`-like forward models differing mainly by the presence of secondary aberration knobs.
- *Feature parity scope (v1 two-plane refactor)*: Match the three-plane binary vocabulary; support a primary Zernike basis; exclude secondary mirror and secondary Zernikes; defer 1/f WFE to parity with the current three-plane refactor scope; reuse the three-plane log_flux transform semantics.

**Two-plane refactor implementation status**
- ✅ Renamed the legacy two-plane optics class to `SheraTwoPlaneOptics`, eliminating the wrapper and preserving Toliman-like pupil→focal behaviour (no new 1/f WFE added).
- ✅ Added `SheraTwoPlaneConfig` capturing two-plane structural knobs (pupil/PSF sampling, bandpass, aperture geometry/struts/DP hooks, primitive plate scale, optional primary Zernike basis; no secondary/relay geometry).
- ✅ Built `build_shera_twoplane_forward_spec_from_config`, mirroring the three-plane forward vocabulary with binary primitives, optional primary Zernikes, primitive plate scale, and derived log-flux via the shared transform set (no secondary terms).
- ✅ Updated the inference-spec builder so two- and three-plane runs share the same baseline astrometry/flux/plate-scale keys, with secondary Zernikes omitted when `include_secondary=False`.

**Follow-up implementation tasks (next steps)**
- ✅ Wired a `SheraTwoPlaneBinder` path plus smoke tests to validate parity with the legacy two-plane model. Binder mirrors the three-plane API (forward-style base store with deriveds refreshed, `.model(store_delta)` public entry point) and uses the same structural hash/cache pattern, now including plate scale as a structural knob sourced from the effective store. Optics and source both consume the merged store (base + delta).
- Loss/optimisation stack now dispatches binders based on cfg type inside `make_binder_image_nll_fn`, so downstream helpers (`run_shera_image_gd`, `run_shera_image_gd_eigen`, FIM helpers) accept two- or three-plane configs without special casing.
- Graph templates have been removed; consider factoring shared binder helpers or a base binder class if/when the systems converge further.
- ✅ Added a minimal two-plane astrometry demo mirroring the canonical three-plane example (`examples/recipes/twoplane_astrometry.py` and `examples/runners/run_twoplane_astrometry.py` using `SheraTwoPlaneConfig` + `SheraTwoPlaneBinder`).
- Evaluate whether shared binder behaviour (two- vs three-plane) should live in a common base class once both paths exist.

---

## 21) Task 10 — Legacy SystemGraph design options (deprecated)
Status: archived; SystemGraph is no longer part of the runtime. The legacy
design is preserved only in historical documentation. New work should focus
on binder + optics caching and shared binder helpers if further consolidation
is needed.

---

## 22) Merge Strategy and V1.0 Milestones

This section captures our strategy for (a) deciding when to merge the refactor work into the main dLuxShera repo, and (b) when to consider the refactor “done” and treat the current architecture as V1.0. There are currently no external users of the main repo; migration concerns are therefore purely for my own workflow and notebooks.

- Historical rationale for the refactor lives in `docs/archive/REFACTOR_HISTORY.md` and `docs/architecture/adr/0001-core-architecture-foundations.md`; this section is about the current merge/V1.0 strategy.
- V1.0 user-facing docs should describe the current architecture as the default without surfacing “refactor” or “legacy” language.

---

### 22.1 Goals

- Present a clean, “this is how dLuxShera works” story to future users and collaborators.
- Avoid user-facing mentions of “refactor” or “legacy” once V1.0 is in place.
- Use the current sandbox / refactor branch to harden the architecture and demos before merging into main.
- Treat “merge to main” and “V1.0” as related but distinct milestones.

---

### 22.2 Milestone A – Merge Refactor Branch into Main

**Intent:** Switch main dLuxShera over to the new ParamSpec / ParameterStore / Binder stack as the canonical implementation. This is the point where I personally prefer the new stack for any real Shera work.

**Criteria for merge:**

- **Code & tests**
  - ParamSpec / ParameterStore / transforms / DerivedResolver are wired together and passing tests.
  - Optics builders (2- and 3-plane) use the new patterns and have basic test coverage.
  - Binder is the main way to instantiate and run models; legacy SystemGraph tests have been removed.
  - Canonical three-plane astrometry demo runs end-to-end and has at least a smoke test.
  - Test suite passes on my main development environment.

- **Practical usability (for me)**
  - I can:
    - Build a Shera model via the Binder and run a forward model.
    - Run a basic inference loop and/or FIM/eigenmode computation without touching old APIs.
  - For any new analysis or notebook, it is natural to reach for the new stack first.

- **Housekeeping**
  - Legacy code is either removed or clearly quarantined (e.g., in a legacy module or with “deprecated” notes).
  - Main branch is updated so that the new stack is the default entry point for Shera modeling.

**Outcome:** Once these criteria are met, the sandbox/refactor work is merged into main. From this point forward, ongoing work (demos, priors, plotting, new optics variants) happens directly on main and is treated as normal feature work rather than blocking “the refactor.”

---

### 22.3 Milestone B – V1.0 Architecture & Documentation

**Intent:** Stabilize the architecture and present dLuxShera as if this design has always existed. All user-facing docs should describe the current system as “V1.0” without mentioning “refactor,” “old stack,” or “new stack.”

**Criteria for V1.0:**

- **API & naming stability**
  - Core concepts and names are settled (e.g., Binder class names, optics system names, ParamSpec / ParameterStore terminology).
  - No further renames of the fundamental building blocks are anticipated without a major version bump.

- **User-facing docs (V1.0 perspective)**
  - **README**:
    - Describes what dLuxShera is and how to install it.
    - Provides a short “hello world” example: create a config, build a Binder, run a forward model and show a PSF.
    - Links to the canonical astrometry demo and concept docs.
  - **Quickstart / Canonical Demo doc**:
    - Walks through the canonical three-plane astrometry workflow step-by-step (config → Binder → simulate data → loss/inference → plotting).
  - **Concept docs** (short, focused):
    - Parameters & Stores: ParamSpec, ParameterStore, transforms.
    - Binder execution: Binder as the user-facing “model object,” with cached telescope evaluation and no graph layer.
    - Optical Systems: three-plane Shera optics as the baseline, two-plane optics as a simplified variant.
  - **examples index**:
    - Lists the canonical three-plane demo, the two-plane demo, and any specialty examples (FIM, eigenmodes, priors) with one-line descriptions.
  - **Status update:** The core architecture concept docs (params_and_store, binder_and_graph, inference_and_loss, eigenmodes) now carry a V1.0 narrative with no user-facing “refactor” or “legacy” language.

- **examples**
  - Three-plane canonical astrometry demo is polished and matches the V1.0 docs.
  - Two-plane demo is available and documented as the simplified alternative (even if lighter-weight than the three-plane example).

- **Dev-facing docs**
  - Working Plan and any architecture notes live under `docs/dev/` (or similar).
  - These can still reference “refactor,” planning tasks, legacy notes, etc., but are not exposed as primary user docs.

**Outcome:** When these conditions are met, the library is considered to have reached “V1.0” in spirit, even if version numbers are adjusted later. Any subsequent work (e.g., advanced priors, HMC, four-plane optics, additional plotting utilities) is treated as incremental feature development on top of a stable base.

---

### 22.4 Near-Term Focus

- Deliver the **optimization artifacts/logging pipeline** described in `docs/architecture/optimization_artifacts_and_plotting.md` and wire it into the canonical and two-plane demos plus `work/scratch/refactored_astrometry_retrieval.py`; keep plotting helpers aligned with the run-directory layout.
- Advance **optimizer control** by adding per-parameter/block learning-rate shaping (FIM/curvature-derived) to the gradient-descent helpers while keeping compatibility with the new artifacts/logging story.
  - Decision: for the refactor path, blocks are implemented via θ-space `lr_vec` and IndexMap rather than `optax.multi_transform` over the ParamStore tree.
- Keep the **doc stack coherent**: use this Working Plan as a “map of maps,” point to `docs/architecture/*.md` for details, and ensure README/tutorials stay in sync as the artifacts/logging work lands. Merge readiness (Milestone A) follows once these pieces are stable.

## 23) Documentation roadmap for dLuxShera
Status: docs housekeeping (dev-facing)

- Canonical long-range roadmap: `docs/dev/roadmap.md`. Treat this as the theme-level plan; keep this Working Plan focused on near/medium-term execution and dev notes.
- Concept/architecture sources of truth: `docs/architecture/{binder_and_graph.md,eigenmodes.md,inference_and_loss.md,optimization_artifacts_and_plotting.md,params_and_store.md}`. Use these for detailed design rather than duplicating content here.
- Tutorials and modeling overview: `docs/tutorials/modeling_overview.md` and `docs/tutorials/canonical_astrometry_demo.md` (plus `examples/README.md`, `examples/recipes/canonical_astrometry.py`, `examples/recipes/twoplane_astrometry.py`, `examples/runners/run_canonical_astrometry.py`, and `examples/runners/run_twoplane_astrometry.py` for runnable flows).
- Dev-facing planning: this file (`docs/dev/working_plan.md`) and any future dev notes under `docs/dev/`. Keep cross-links back to the architecture docs for specifics.
- Time-domain design contract (Phase 1): `docs/dev/obs_subblock_generator_design.md` captures the observation sub-block generator interface and artifact layout before implementation.
- Observation sub-block helper separation (Phase 3): explicit trace generation now has a separate recipe (`examples/recipes/observation_subblock_trace.py`) and builder utility (`src/dluxshera/utils/obs_subblock_trace_builders.py`), while rendering remains in `examples/recipes/observation_subblock.py`.
- Navigation helpers: `devtools/generate_context_snapshot.py` and `devtools/print_tree.py` remain the authoritative way to browse the live tree and ParamSpec/transform snapshots.

Near-term doc housekeeping:
- Keep this Working Plan as a “map of maps” that points to the architecture/tutorial docs and the roadmap, rather than re-explaining them.
- Ensure architecture docs and tutorials stay the canonical detail; keep this file focused on status, priorities, and where to look next.

## 24) Binder/SystemGraph shared implementation follow-through
Status: implemented; historical context

- Base implementations landed for binders; the SystemGraph scaffold has been removed from the codebase. Optics builders remain system-specific.
- Implementation follow-up: caching and derived-resolution hooks remain future work if a graph layer is ever reintroduced; see `docs/architecture/binder_and_graph.md` for current binder intent.

## 25) Parking Lot

- Two/Four-plane optics variant design and transforms.
- Extended inference methods (HMC, priors, eigenspace optimization) after core stack stabilizes.
- Ergonomic shims (`ModelParams`) and deprecation strategy for legacy APIs.
- High-level model design / capabilities documentation describing what the Shera-style model does (optical/astrometric forward model, main outputs, supported questions) and its key assumptions/approximations, written for proposal and systems-engineering consumers rather than just implementers.
- Model–error-budget interface and parameter dependency mapping: lightweight docs/figures that show how model outputs and sensitivities map onto specific error-budget terms, and how primitives vs. derived parameters (ParamSpec → Store → transforms) relate to those terms for traceability.

### 25.1 Detector roadmap: pixel grid offsets (dx/dy) and calibration-driven detector layers

**Overview**

We currently use a minimal detector path (single Downsample layer with `kernel_size = cfg.oversample`). The next detector-model expansion is a per-pixel offset layer that applies measured pixel-center shifts (`dx`, `dy`) before final detector sampling. This gives us a clear way to include detector metrology in the forward model while keeping detector calibration/product handling separate from low-level layer mechanics.

**Decisions (recorded)**

- **Units:** `dx`/`dy` are expressed in detector pixel units on the final (post-downsample) detector grid.
- **Oversampled operation:** the pixel-offset layer runs on the oversampled image and scales offsets internally by `oversample` (so supplied calibration maps stay in detector-pixel units).
- **Sign convention:** (`dx`, `dy`) represent where the *actual* pixel center sits relative to the ideal grid in detector coordinates. Positive `dx` means the pixel center is to the right (sample at larger `x`); positive `dy` means the center is at larger `y`.
- **Responsibility split:** the layer is intentionally “dumb” (apply offsets only). Calibration ingestion/selection/synthesis belongs to a separate provider component.

**Calibration products & provider concept**

- Calibration products may be partial maps in the near term (e.g., 100×100 or 200×200 regions) and may evolve to full-frame maps later.
- Near-term strategy: provide ROI-local `dx`/`dy` arrays directly to the detector layer path.
- Future strategy: add explicit detector-coordinate anchoring (e.g., ROI origin/global index mapping) so the same provider API can serve arbitrary subarrays from global products.
- Synthetic offset generation (for testing/what-if studies) should plug into the same provider interface used by measured maps or stitched mosaics.

**Repository organization (planned)**

- `src/dluxshera/layers/detector_layers.py`: custom detector layers (pixel offsets first; later fill-factor, diffusion, and related effects).
- `src/dluxshera/components/detectors.py`: named detector model/spec definitions and calibration metadata hookups.
- `src/dluxshera/builders/detector.py`: builder wiring that selects a detector model, resolves calibration products, and assembles the `LayeredDetector` pipeline.
- Placeholder naming is acceptable for now (e.g., `ApplyPixelOffsets`, name TBD); avoid locking in final class names until implementation.

**Config naming guidance (non-binding)**

- Prefer a model-selector key such as `cfg.detector_model` for camera/detector choice.
- Optionally add a separate calibration selector (e.g., `cfg.detector_calibration_id`) for product/version choice.
- Keep detector noise parameters as detector-model metadata for likelihood/simulation usage for now; do not force them into forward-model layers yet.

## 26) Implementation Plan — Optimization Artifacts + Signals + I/O (v0)

### 26.1 Current state (survey)

- **Optimization + packing surfaces:** θ-space loops live in `src/dluxshera/inference/optimization.py` (e.g., `run_simple_gd`, binder-aware `run_image_gd`, and Fisher helpers). Packing/unpacking utilities live in `src/dluxshera/params/packing.py`; binder NLL builders and theta mapping hooks are in `src/dluxshera/inference/losses.py` and `src/dluxshera/inference/inference.py`. IndexMap export exists via `run_artifacts.build_index_map(...)`; packing order is aligned with `ParamSpec.subset(...)`.
- **Transforms/DerivedResolver:** Transform registration and recursive resolution live in `src/dluxshera/params/transform_registry.py`; Shera-specific transforms (plate scale, log flux, raw fluxes) are in `src/dluxshera/params/shera_threeplane_transforms.py`.
- **Plotting:** Refactor-era plotting helpers (PSF and parameter histories) are in `src/dluxshera/plot/plotting.py` with headless-friendly IO (return fig/axes, optional `save_path`). Signal builders and panel recipes for intro diagnostics live in `src/dluxshera/inference/{signals.py,plotting.py}` and feed optional run artifacts/plots.
- **Recipes/runners:** Canonical/binder-based runs are in `examples/recipes/canonical_astrometry.py`, `examples/recipes/twoplane_astrometry.py`, `examples/runners/run_canonical_astrometry.py`, `examples/runners/run_twoplane_astrometry.py`, and `work/scratch/refactored_astrometry_retrieval.py`; artifact writing is opt-in and disabled by default.
- **Docs:** Strategy and schema for artifacts/signals/preconditioning live in `docs/architecture/optimization_artifacts_and_plotting.md` (source of truth). Working plan now tracks phased implementation here; `src/dluxshera/inference/run_artifacts.py` and regression tests cover the core I/O scaffold.

### 26.2 Phased plan (aligned to architecture doc and decisions)

**Phase A — Run artifact I/O scaffold (module only) — DONE**
- Deliverables:
  - Add `src/dluxshera/inference/run_artifacts.py` with functions-first API: `save_run(run_dir, trace, meta, summary, *, signals=None, grads=None, curvature=None, precond=None, checkpoints=None, diag_steps=None)` plus `load_trace`, `load_meta`, `load_summary`, `load_checkpoint(which="best"|"final")`.
  - Helper to build and serialize an IndexMap (ordered entries of `name/start/stop/shape/block`) from a `ParamSpec` subset and reference store/θ for shape validation; store it only in `meta.json`.
  - Enforce required artifact layout (always write `trace.npz`, `meta.json`, `summary.json`), keep gradients off by default, and allow optional artifacts (signals, diag_steps.jsonl, grads.npz, curvature.npz, precond.npz, checkpoints).
- Acceptance criteria:
  - Round-trip save/load for trace/meta/summary works on synthetic data; IndexMap slices align with provided θ dimensionality; optional artifacts are skipped cleanly when not provided.
  - `signals.npz` remains self-contained (no sidecar metadata) and optional.
  - No gradient history is emitted unless explicitly passed.
- Tests to add/run:
  - New fast unit test (e.g., `tests/inference/test_run_artifacts_io.py`) covering save/load round-trip, IndexMap validation, and optional artifact skipping.
  - Command: `PYTHONPATH=src pytest tests/inference/test_run_artifacts_io.py -q`.
- Docs/touchpoints:
  - Link `docs/architecture/optimization_artifacts_and_plotting.md` to the new module/API.
  - Update this working plan status after landing.
- Dependencies:
  - Uses existing packing utilities for IndexMap; no optimizer changes yet.

**Phase B — Integrate artifact writing into optimization loops — DONE**
- Deliverables:
  - ✅ Wrap `run_simple_gd` (and binder helpers such as `run_image_gd`) with optional artifact emission: create `runs/<run_id>/`, write `trace/meta/summary` at end-of-run, and support opt-in checkpoints (`checkpoint_best.npz`, `checkpoint_final.npz`).
  - ✅ Record optimizer/binder/spec identifiers and IndexMap in `meta.json`; keep trace minimal (`loss`, `theta`, optional `grad_norm/step_norm/base_lr/accepted`).
  - ✅ CLI/demo wiring: opt-in kwargs for canonical/two-plane demos and `work/scratch/refactored_astrometry_retrieval.py` allow artifact writing without slowing default runs.
- Acceptance criteria:
  - ✅ Tiny smoke optimization produces a run directory with required artifacts and no gradients by default; checkpoints saved when enabled and shapes align with θ.
  - ✅ Summary includes minimal scalars (final loss, step count, elapsed time if available).
  - ✅ Legacy behaviours preserved when artifact writing is disabled.
- Tests to add/run:
  - ✅ `tests/inference/test_run_artifacts_integration.py` covers quadratic and binder-backed smoke runs, asserting required files/keys exist.
  - Command: `PYTHONPATH=src pytest tests/inference/test_run_artifacts_integration.py -q`.
- Docs/touchpoints:
  - ✅ `docs/architecture/optimization_artifacts_and_plotting.md` references the integration points and schema.
- Dependencies:
  - Requires Phase A helpers; IndexMap generation must be wired via packing/infer_keys used by the optimizer.

**Phase C — Signals builders + panel recipes (plotting integration) — DONE**
Now that artifact emission (Phase B) is wired, this phase focuses on decoding traces into signals and lightweight plotting/recipes.
- Deliverables:
  - Add `src/dluxshera/inference/signals.py` to build derived time-series signals from trace + decoder/binder + optional truth: x/y astrometry residuals (µas), separation residual (µas), plate-scale error (ppm), raw flux error ppm (via new `binary.raw_fluxes` transform), zernike residuals (nm) with RMS summariser. Truth-absent cases fill NaNs but keep shapes stable.
  - Add `src/dluxshera/inference/plotting.py` to provide intro panel recipes (astrometry overlay, separation, plate scale, raw flux A/B overlay, zernike RMS + optional components) saved under `<run_dir>/plots/` headlessly.
  - Allow caching signals to optional `signals.npz` via Phase A API (binder-backed runner wiring), and optionally emit plots alongside other artifacts.
- Acceptance criteria:
  - Signal builders accept trace + meta (IndexMap) + binder/spec + truth and return named arrays with consistent shapes; optional truth fills NaNs without shape churn.
  - Raw fluxes computed via a registered Transform (truth-independent) and used for ppm residuals when truth is supplied.
  - Panel helpers can render x/y overlay and flux A/B overlay headlessly and write deterministic PNGs.
  - Binder-backed runner can opt-in to writing `signals.npz` and plots when artifacts are enabled.
- Tests to add/run:
  - Unit tests for signal shape/content on synthetic trace (no binder) plus raw_flux transform correctness; ensure ppm scaling and zernike RMS summaries are correct.
  - Smoke plot test that writes PNGs headlessly.
  - Command: `PYTHONPATH=src pytest -q tests/inference/test_signals.py tests/inference/test_plotting_smoke.py`.
- Docs/touchpoints:
  - Document signal names/units in `docs/architecture/optimization_artifacts_and_plotting.md` and mark status as Phase A/B implemented.
  - Update examples/runners to optionally cache signals and produce plots (headless-save only).
- Dependencies:
  - Relies on Phase A/B artifacts + IndexMap; needs TransformRegistry hook for `binary.raw_fluxes`.

**Phase D — Preconditioning artifacts (lr_vec, curvature) — IN PROGRESS**
- Deliverables:
  - Extend optimizer utilities to optionally compute/store per-index lr_vec and curvature/preconditioner vectors; save to `precond.npz` and/or `curvature.npz` (lr_vec in `precond.npz` per decision).
  - Capture preconditioning config in `meta.json` (method, eps, clipping bounds, refresh cadence) and keep gradients history off by default.
  - Validate checkpoints include any persisted optimizer state needed for restart.
- Acceptance criteria:
  - When enabled, `precond.npz` contains lr_vec (and optional preconditioner) with shape matching θ and aligned with IndexMap; absence when disabled is clean.
  - `meta.json` records optimizer/preconditioning identity and parameters; summary notes whether preconditioning was active.
  - Core GD path remains backward-compatible when preconditioning is off.
- Status: 
  - v0 path uses `ema_grad2` at θ₀ to derive `curv_diag`, `precond`, and `lr_vec`; artifacts are emitted via `precond.npz` / `curvature.npz` with metadata recorded under `optimizer.preconditioning`. Covered by `tests/inference/test_precond_artifacts.py`.
  - v0 implementation now includes a concrete FIM → diag(FIM) → `lr_vec` path, exercised in `work/scratch/refactored_astrometry_retrieval.py` via `fim_theta(loss_fn, theta_true)` and `run_image_gd(..., lr_vec=...)`. This uses θ-space vector LRs rather than the legacy tree-based `lr_model`.
- Tests to add/run:
  - Shape/metadata validation tests (e.g., `tests/inference/test_precond_artifacts.py`) using synthetic curvature vectors; ensure saved arrays reload and align with θ dim.
  - Command: `PYTHONPATH=src pytest tests/inference/test_precond_artifacts.py -q`.
- Docs/touchpoints:
  - Expand `docs/architecture/optimization_artifacts_and_plotting.md` preconditioning section with the concrete file layout and metadata fields.
  - Note optimizer flag names in examples/working plan.
- Dependencies:
  - Builds atop Phase B artifact plumbing; optional hooks from Phase C (signals) not required.

**Phase E — Polish + documentation consistency**
- Deliverables:
  - Sweep docs/tutorials/examples to ensure run_artifacts usage, signals caching, and preconditioning flags are documented consistently; add brief troubleshooting notes for missing optional files.
  - Update `docs/dev/working_plan.md` status per phase completion and record any follow-up tasks.
  - ✅ Added sweep summary CSV tooling (`dluxshera.inference.sweeps`, `examples/scripts/summarize_runs.py`) and checkpoint gradient diagnostics (`dluxshera.inference.diagnostics`, `examples/scripts/analyze_checkpoint_gradients.py`).
- Acceptance criteria:
  - Architecture docs reference the implemented module paths and schema; examples README shows how to enable/inspect run directories.
  - No stale references to legacy logging; working plan reflects completed phases vs. upcoming.
- Tests to add/run:
  - Rely on existing unit/integration coverage; rerun smoke demo tests if they exercise artifact flags.
  - Command: `PYTHONPATH=src pytest -q` (or a narrowed subset if runtime becomes heavy).
- Dependencies:
  - All prior phases.

### 26.3 Open questions / blockers

- **Run directory identity:** adopt a deterministic `run_id` strategy (timestamp vs. UUID vs. caller-provided) and whether to embed git hash automatically or gate on availability.
- **Truth availability for signals:** for demos/tests, define how truth is surfaced to signal builders (pass through optimizer API vs. loaded alongside data) to avoid coupling to specific demos.
- **Checkpoint contents:** decide minimal checkpoint schema (θ only vs. θ + optimizer state) while keeping restart support lightweight for optax-based loops.


## 27) Known Issues (lightweight tracker)

Use this section as a quick in-doc ledger for active issues that are worth tracking between formal GitHub issue triage passes.

- **Config and Store Documentation is lacking** a complete list of parameters. Users should have easy-to-find documentation for all parameters present in the configs and in the stores, which parameters are treated as structural, etc.
- **Source parameters are treated as structural.** The `wavelength_m`, `bandwidth_m`, and `n_lambda` source settings are currently carried in the config object and are treated as structural in the structural parameter set. Because the AlphaCen source is built at model evaluation time and is not cached, the `forward_store` can carry these instead, which simplifies the call signature to `build_alpha_cen_source()`. If we eventually move to a more complicated source model, we may end up caching the source object and then re-defining a structural subset anyway, in which case we might end up where we started. Until then, though, I think it makes sense to consolidate these source settings into the forward_store with the other source settings ("binary.x_position_as", "binary.y_position_as", etc.) to reduce some mental overhead.
- **`imaging.throughput` might not be modelled** We set a value in the store, but it's unclear if anything in the model actually applies the throughput.
- **Profiles/IO consistency across workflows is incomplete.** Prescription/override flows are strong for experiment runners, but a unified YAML/JSON profile experience across all entry points is still pending.
