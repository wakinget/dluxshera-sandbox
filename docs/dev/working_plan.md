# dLuxShera Working Plan & Notes (dev-facing)
_Last updated: 2025-12-19_

This is a living, dev-facing document summarizing the goals, architecture, decisions, tasks, and gotchas for dLuxShera as it moves through V1.0 and beyond. It replaces the refactor-era index while keeping the running plan in one place.

This Working Plan is the near/medium-term map for developers. For the theme-level, longer-horizon roadmap see `docs/architecture/roadmap.md`. For concept-level architecture detail (ParamSpec/Store, Binder/SystemGraph, loss/optimization, eigenmodes), use the `docs/architecture/*.md` set referenced below; this doc points to them rather than duplicating their content.

## How to use this doc
- **Sections 1–12:** Current architecture focus areas, gotchas, and open questions (developer-facing summaries with links to canonical architecture docs).
- **Sections 13–15:** Backward-compatibility notes, references, and binder namespace ergonomics.
- **Sections 16–18:** Tasks, priorities, and policy analysis (what’s done vs active P0/P1 work).
- **Sections 19–21:** Changelog and analysis/historical mappings (marked with status lines).
- **Section 22:** Merge strategy and near-term focus for V1.0.
- **Sections 23–25:** Documentation housekeeping, implementation follow-through notes, and the parking lot/backlog.
- **Historical context:** For narrative history and ADR-style rationale, see `docs/archive/REFACTOR_HISTORY.md` and `docs/architecture/adr/0001-core-architecture-foundations.md`.

---

## 1) Context & Problem Statement

- **Entanglement:** Primitives (e.g., `m1.focal_length`, `system.plane_separation`) and derived values (e.g., `imaging.psf_pixel_scale`) are computed in multiple places → unclear source of truth.
- **Bugs:** After optimization, `psf_pixel_scale` can be missing/incorrect in extraction; init at exact zero can lead to zero gradients.
- **Scaling pain:** Adding parameters/nodes couples logic into optics code; docs/tests lack a single schema for units/bounds/priors.

**Goal:** Cleanly separate *what parameters exist*, *where values live*, *how things are derived*, and *how the system executes*—while keeping a stable facade for users and examples.

**Target outcome (Partially met):**
- ✅ Consistent `psf_pixel_scale` (and other deriveds) regardless of whether they are optimized directly or computed from primitives.
- ⚠️ Clear primitives↔derived boundary and testable pure transforms (global registry implemented; system-scoped resolver still pending).
- ⚠️ Structured execution graph (minimal SystemGraph scaffold exists; still single-node and internal-only).
- ⚠️ Minimal churn to current examples; future models (e.g., four-plane) to slot in (four-plane support missing).

---

## 2) Architecture (High-Level)

The refactor-era architecture cleanly separates **what exists** (ParamSpec), **what values are in play** (ParameterStore), **how deriveds are computed** (DerivedResolver/transform registry), and **how execution is wired** (Binder + SystemGraph). The public model façade for Shera systems remains the binder-based PSF generator; legacy helpers wrap this internally.

- **Why this shape:** Legacy flows intertwined parameter definitions, derived computations, and execution; the new layering keeps ParamSpec/Store declarative and Binder/SystemGraph as the sole runtime surface. Derived transforms stay pure and testable.
- **Current state:** ParamSpec/Store are in daily use with strict-by-default validation; the transform registry is scoped by system_id; Binder/SystemGraph are the supported forward path for both two- and three-plane optics, with SystemGraph still intentionally minimal (single node, caching hooks planned).
- **Details:** For full diagrams and API notes see `docs/architecture/binder_and_graph.md` (Binder/SystemGraph) and `docs/architecture/inference_and_loss.md` (loss stack, packing/unpacking). Eigenmode-specific context lives in `docs/architecture/eigenmodes.md`. Broader rationale sits in `docs/architecture/adr/0001-core-architecture-foundations.md`.

---

## 3) Repository Layout (Actual vs Proposed)

High-level snapshot (illustrative; run `python devtools/print_tree.py` or `python devtools/generate_context_snapshot.py` for the authoritative view):

```
dLuxShera/
├─ docs/
│  ├─ architecture/{binder_and_graph.md,eigenmodes.md,inference_and_loss.md,optimization_artifacts_and_plotting.md,roadmap.md,...}
│  ├─ dev/working_plan.md   ← this document
│  └─ tutorials/{modeling_overview.md,canonical_astrometry_demo.md}
├─ src/dluxshera/
│  ├─ core/{binder.py,modeling.py,universe.py}
│  ├─ graph/system_graph.py
│  ├─ inference/{losses.py,prior.py,numpyro_bridge.py,optimization.py}
│  ├─ params/{spec.py,store.py,registry.py,packing.py,transforms.py,shera_threeplane_transforms.py}
│  ├─ optics/{config.py,builder.py,optical_systems.py}
│  ├─ plot/plotting.py
│  └─ utils/utils.py
├─ examples/scripts/{run_canonical_astrometry_demo.py,run_twoplane_astrometry_demo.py}
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

Shera binders own configs/specs/stores, expose `.model(store_delta)` as the PSF generator, and can run through a minimal SystemGraph (single node today) or a direct call into the optics builder. Binders stay mostly immutable (`.with_store(...)`) to keep JAX friendliness; SystemGraph is eager but intentionally lightweight.

- **Why:** Keeps execution encapsulated while letting θ overlays be the only source of dynamism. Structural caching lives in the optics builders; graph caching hooks remain future work.
- **Gotchas:** Derived values must be refreshed before binding; SystemGraph is single-node by design right now.

Loss wiring, Binder NLL helpers, and canonical demo usage are summarized in `docs/architecture/inference_and_loss.md` and exercised in `docs/tutorials/canonical_astrometry_demo.md`. SystemGraph/Binder intent and design trade-offs live in `docs/architecture/binder_and_graph.md`.

---

## 8) Parameter Profiles & IO (Planned)

- Not yet implemented. Profiles (lab/instrument defaults), YAML/JSON loading, and serialization helpers remain to be built once primitives-only policy is finalized.

---

## 9) Docs & examples (Phase 1 shipped)

- Canonical binder/SystemGraph astrometry demo lives in `examples/scripts/run_canonical_astrometry_demo.py` with both pure-θ and eigenmode GD flows; the two-plane companion is `examples/scripts/run_twoplane_astrometry_demo.py`.
  - The demo showcases the refactor-era plotting helpers: PSF visualisation via `plot_psf_single` / `plot_psf_comparison` and parameter trajectories via `plot_parameter_history_grid`. Plotting utilities follow the IO policy (return fig/axes; caller decides to save/show) and save figures only when requested to keep tests headless.
- Current doc stack:
  - Concept orientation: `docs/tutorials/modeling_overview.md`
  - Architecture: `docs/architecture/{binder_and_graph.md,eigenmodes.md,inference_and_loss.md,params_and_store.md,optimization_artifacts_and_plotting.md}`
  - Tutorials: `docs/tutorials/canonical_astrometry_demo.md`
  - Dev-facing: this plan (`docs/dev/working_plan.md`)

---

## 10) Testing Philosophy

- Existing tests cover: ParamSpec/store validation and packing, transform resolution (including cycle guards), optics builder/binder smoke paths, optimization loss wrapper, eigenmode utilities, SystemGraph smoke/regression via new graph tests, and the canonical astrometry demo in fast mode.
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

- **Baseline test status:** Running `pytest` requires adding `src/` to `PYTHONPATH` (or installing the package) so Binder-related tests (`tests/test_binder_smoke.py`, `tests/test_binder_shared_behaviour.py`, graph smoke, and Binder-backed loss/optimization tests) import `dluxshera` correctly.
- **Binder mutability/shape:** `SheraThreePlaneBinder` instances are `dataclasses` with `frozen=False` and `slots=False`; they do not present as `equinox.Module` or `zodiax.Base`, so mutation is guarded only by convention rather than framework-level immutability.
- **Store surface area:** `binder.base_forward_store` is a `ParameterStore` exposing `get/keys/items/values/as_dict/replace/validate_against` and similar mapping semantics.
- **Derived placement:** With the default forward spec + refreshed forward store (via `tests.helpers.make_forward_store`), derived values such as `system.plate_scale_as_per_pix` are present directly in the base forward store prior to any evaluation.

- `SheraThreePlane_Model` remains the public entry point; new plumbing should remain internal to avoid churn in existing scripts.
- Legacy files (`modeling.py`, optics helpers) still carry pre-refactor pathways; the refactor must avoid breaking current examples until replacements land.

---

## 14) Prior Art / References

- dLux core APIs for `ThreePlaneOpticalSystem` and PSF generation.
- Prior optimization scripts in `examples/` (still legacy-style; to be updated after SystemGraph lands).

---

## 15) Binder namespace ergonomics (Task 1A–1E status)

- Task 1A–1E: Completed (binder.get, StoreNamespace proxy, Binder.ns explicit access, cfg-field attribute raising, store-prefix attribute raising).
- Supported access patterns so far: `binder.get(...)`, `binder.ns("prefix")`, cfg forwarding such as `binder.psf_npix`, and store prefix raising such as `binder.system.plate_scale_as_per_pix` / `binder.binary.x_position_as`.

---

## 16) Tasks & Priorities (Updated)

Legend: ✅ Implemented · ⚠️ Partial · ⏳ Not implemented

**Completed to date (highlights)**  
- ✅ ParamSpec ergonomics (`subset` include-only, `without` for drops) with regression tests.  
- ✅ ParameterStore strict-by-default validation, derived refresh/strip helpers, and shallow serialization.  
- ✅ Inference packing/unpacking (θ ↔ store delta) with tests.  
- ✅ Scoped transform registry + Shera plate-scale/log-flux transforms.  
- ✅ ThreePlaneBuilder structural hash + cache/clear helper.  
- ✅ Binder-first loss wiring (Binder NLL helpers using `gaussian_image_nll`), binder/SystemGraph parity, and binder namespace UX (Task 1A–1E).  
- ✅ SystemGraph single-node scaffold owned by binders.  
- ✅ Binder NLL stationary-point regression landed; follow-on scenarios pending (multi-wavelength/multi-PSF).  

**P0 — Current focus**  
- ✅ **Optimization artifacts & logging**: Phase A scaffold (`run_artifacts.py`) is in place and Phase B wiring now emits required artifacts from `run_simple_gd` and binder-backed `run_image_gd` when opt-in flags are provided. Integration smoke tests cover end-to-end writes and metadata (trace/meta/summary + optional checkpoints).  
- ⏳ **Optimizer control (learning-rate shaping)**: Extend `run_simple_gd` (or successor) with per-parameter/block learning rates derived from FIM/curvature estimates; ensure compatibility with the logging/artifacts pipeline above.  
- ⚠️ **Loss regression hardening**: Keep the landed Binder NLL stationary-point regression; add coverage for multi-wavelength / multi-PSF scenarios as new demos land and surface any remaining edge cases.

**P1 — Next up**  
- ⏳ **Profiles/IO and serialization**: YAML/JSON profiles and richer serialization once primitives-only policy remains stable.  
- ⏳ **Documentation and example polish**: README quickstart, additional tutorials, and aligning examples/README.md with the new optimization artifact flow.  
- ⏳ **Expanded transform coverage**: Broaden registry coverage for additional systems (two-/future four-plane) as specs land.

**P2 — Variants & ergonomics**  
- ⏳ Four-plane variant (specs, transforms, builder, resolver tests).  
- ⏳ Ergonomic shims (`ModelParams`), deprecation path for legacy APIs, upstream PR prep.

**Near-term hygiene**  
- ⚠️ Plate-scale policy decision and SystemGraph caching/multi-node hooks remain open.  
- ⚠️ Sweep remaining scripts/tests for legacy unit-less astrometry aliases once canonical unit-aware keys settle.

---

## 17) Recommended Next 3–5 Tasks (to reach end-to-end flow)

1. **Add SystemGraph + DLuxSystemNode scaffold (P0) — DONE**
   - **Outcome:** Added `graph/` package with `DLuxSystemNode` + `SystemGraph`, tied into binder loss via optional flag, and regression-tested against the legacy three-plane forward path.
   - **Follow-ups:** Add caching/structural hashing, multi-node support, and derived-resolution enforcement before relying on it in production.

2. **Scoped DerivedResolver with system IDs (P0) — DONE**
   - **Outcome:** Added `params/registry.py` with system-scoped resolver/decorator, defaulting to the Shera three-plane system; tests cover isolation across system_ids and existing Shera transforms continue to resolve via the default registry.
   - **Follow-ups:** Extend coverage for future system variants (two-/four-plane) once their specs land and align ergonomics with ParameterStore primitives-only enforcement.

3. **ParameterStore enforcement + serialization (P0) — DONE**
   - **Outcome:** ParameterStore validation defaults to strict (rejects derived keys), with opt-in override for debug/override flows plus helpers to strip/refresh/check derived values. Shallow serialization (`from_dict`, `from_spec_defaults`, `as_dict`) is in place; YAML/JSON profile IO remains deferred to the profiles/IO task.
   - **Follow-ups:** Add optional YAML/JSON helpers alongside the profile/IO workstream if still desired.

4. **Structural hash/cache for ThreePlaneBuilder (P1) — DONE**
   - **Outcome:** Structural subset documented in `optics/config.py`; deterministic hash helper and cache/clear APIs added to `optics/builder.py` (env flag `DLUXSHERA_THREEPLANE_CACHE_DISABLED`). Tests cover cache hit/miss, non-structural reuse, and hash stability.
   - **Follow-ups:** Consider exposing cache stats and integrating hash/caching at the SystemGraph layer once multi-node support lands.

5. **Canonical astrometry demo + docs (P1)**
   - **Status:** ✅ Added `examples/scripts/run_canonical_astrometry_demo.py` using ParamSpec + ParameterStore + DerivedResolver to build truth/variant stores, SheraThreePlaneBinder/SystemGraph forward model, and Optax GD with prior penalties. README updated with run command; smoke test exercises `main(fast=True)`.
   - **Two-plane companion:** Added `examples/scripts/run_twoplane_astrometry_demo.py` as a lighter-weight analogue that exercises the SheraTwoPlaneConfig/Binder stack; both demos serve as reference examples for upcoming docs/tutorials.

---

## 18) ParamSpec + ParameterStore policy (plate_scale/log_flux) — analysis & options

**Current behavior (“split personality”)**
- `build_inference_spec_basic()` marks `system.plate_scale_as_per_pix` and `binary.log_flux_total` as **primitive knobs** for optimisation.【F:src/dluxshera/params/spec.py†L95-L167】【F:src/dluxshera/params/spec.py†L207-L245】
- `build_forward_model_spec_from_config()` mirrors geometry/throughput primitives from `SheraThreePlaneConfig` and declares `system.plate_scale_as_per_pix` and `binary.log_flux_total` as **derived** with registered transforms (geometric plate scale and collecting-area × band × throughput flux).【F:src/dluxshera/params/spec.py†L337-L458】
- The transform registry is **store-wins**: if a key is present in the `ParameterStore`, the transform is skipped; otherwise dependencies are resolved recursively.【F:src/dluxshera/params/registry.py†L117-L186】 Tests exercise this by computing plate scale/log flux from a forward-model store seeded with primitives only.【F:tests/test_shera_threeplane_transforms.py†L1-L71】
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
- Binder-first loss wiring is the canonical inference path; SystemGraph is the default internal executor (single node for now).  
- ThreePlaneBuilder structural hashing + caching shipped; graph-level caching/derived hooks remain future work.

---

## 20) Legacy Shera two-plane stack → refactor-era mapping (analysis)
Status: analysis + historical mapping; two-plane refactor implementation now landed (see follow-ups below). Implementation overview lives alongside the three-plane stack in `docs/architecture/binder_and_graph.md`.

For a concise mapping of legacy APIs to the current architecture, see `docs/archive/LEGACY_APIS_AND_MIGRATION.md`.

**Current two-plane parameter vocabulary and behavior (legacy `SheraTwoPlaneParams`/`SheraTwoPlane_Model`)**
- Point designs expose primary/secondary diameters, PSF pixel scale (primitive, arcsec/pix), bandpass width, and log flux; operational knobs include pupil/PSF sampling, binary astrometry (x/y offsets, separation in mas, PA in deg, contrast), central wavelength, number of wavelengths, and a single Zernike basis (Noll indices + amplitudes) applied to the primary pupil mask. Noise fields include calibrated/uncalibrated 1/f power-law/amplitude pairs. No explicit plate-scale derivation occurs; pixel scale is passed straight into the optics builder.【F:src/dluxshera/inference/optimization.py†L1224-L1312】【F:src/dluxshera/core/modeling.py†L330-L419】
- Binary astrometry mirrors the three-plane vocabulary (x/y offsets, separation, PA, contrast, log_flux) and is forwarded to the `AlphaCen` source; no secondary-mirror parameters appear. Flux is handled as a stored log_flux scalar (no transform), and plate scale is treated as a primitive `psf_pixel_scale` handed directly to the optics and detector sampling (oversample=1).【F:src/dluxshera/core/modeling.py†L330-L419】
- The optics path is Toliman-like: a two-plane `JNEXTOpticalSystem` fed by wavefront/PSF sampling, oversample, pixel scale, aperture diameters, strut geometry, diffractive pupil (dp_design_wavel), and optional Zernike basis; primary Zernikes are normalized to nm before setting coefficients. Detector is a simple downsample layer; PSF sampling equals the provided oversample (hard-coded to 1 in the model).【F:src/dluxshera/core/modeling.py†L330-L384】【F:src/dluxshera/optics/optical_systems.py†L95-L190】

**JNEXTOpticalSystem (legacy two-plane optics) vs TolimanThreePlaneSystem**
- JNEXT is an `AngularOpticalSystem` with only aperture + diffractive pupil layers, optional Zernike basis on the primary, and propagator knobs for PSF sampling, oversample, and pixel scale. It uses primary/secondary diameters and strut geometry to build a single pupil; no secondary mirror surface/aberrations or Fresnel relay are present. Aberrations are strictly Zernike-based (no 1/f WFE), and the diffractive pupil is loaded from a numpy mask and converted to an aberrated layer. This mirrors Toliman-style two-plane optics but with Shera-specific defaults (diameters 0.09/0.025 m, four struts at -45°, 550 nm design wavelength).【F:src/dluxshera/optics/optical_systems.py†L95-L190】

**Two-plane vs three-plane comparison**
- Shared: binary astrometry/flux knobs, central wavelength + bandwidth + wavelength sampling, pupil/PSF grid sizes, primary Zernike basis (nm-scaled), and calibrated/uncalibrated 1/f knobs (though the three-plane applies them to both mirrors). Both models hand binaries to `AlphaCen` with the same argument set and normalize Zernikes on the primary.【F:src/dluxshera/core/modeling.py†L120-L221】【F:src/dluxshera/core/modeling.py†L330-L419】
- Three-plane-only: explicit mirror focal lengths, plane separation, detector pixel size, derived plate scale (EFL from two-mirror relay plus pixel size), and full secondary mirror aperture with its own Zernike basis and 1/f layers. The optics builder constructs a Fresnel relay (`SheraThreePlaneSystem`) and adds separate calibrated/uncalibrated WFE layers for both mirrors. Detector sampling uses `oversample=1` but plate scale comes from geometry unless overridden.【F:src/dluxshera/core/modeling.py†L120-L221】【F:src/dluxshera/inference/optimization.py†L1224-L1312】
- Two-plane-only: primitive PSF pixel scale passed directly into `JNEXTOpticalSystem`; no secondary mirror geometry/aberrations; no plane separation/focal lengths; 1/f maps inserted but only after the single aperture layer. The pipeline is strictly pupil → focal plane without Fresnel relay.

**Mapping plan to refactor-era concepts**
- *Optics naming and placement*: Rebrand/alias `JNEXTO​pticalSystem` as `SheraTwoPlaneOptics` (class in `optics/optical_systems.py`) to align with the Shera family naming. Keep two- and three-plane optics in the same module for now; harmonizing three-plane naming to `SheraThreePlaneOptics` is a deferred cleanup.
- *Config + forward spec*: Introduce `SheraTwoPlaneConfig` alongside the three-plane config, sharing binary vocabulary, wavelength/bandwidth sampling, and primary Zernike basis fields. Two-plane-specific primitives: `psf_pixel_scale` (primitive, arcsec/pix), primary aperture geometry (p1/p2 diameters, strut geometry, diffractive pupil design wavelength), sampling (`pupil_npix`, `psf_npix`, `oversample`). Exclude three-plane-only fields (focal lengths, plane separation, detector pixel size, secondary basis/1/f). Forward spec should mirror the binary vocabulary used by the three-plane builder (unit-aware `binary.x_position_as`, `binary.y_position_as`, `binary.separation_as`, `binary.position_angle_deg`, `binary.contrast`), include `primary.zernike_coeffs_nm` when a basis is configured (default zeros), omit any secondary terms, treat `psf_pixel_scale` as primitive (no transform), and derive `binary.log_flux_total` via the same transform family as the three-plane system.
- *Inference spec sharing*: Provide a shared “Shera astrometry inference spec” builder that covers the common binary vocabulary, primary Zernike coefficients, and plate scale as a primitive knob. Secondary-specific keys (secondary Zernikes) should be included only for three-plane runs; callers can drop them via `ParamSpec.without(...)` for two-plane cases. From inference’s perspective, both systems remain `dl.Telescope`-like forward models differing mainly by the presence of secondary aberration knobs.
- *Feature parity scope (v1 two-plane refactor)*: Match the three-plane binary vocabulary; support a primary Zernike basis; exclude secondary mirror and secondary Zernikes; defer 1/f WFE to parity with the current three-plane refactor scope; reuse the three-plane log_flux transform semantics.

**Two-plane refactor implementation status**
- ✅ Introduced `SheraTwoPlaneOptics` as a Shera-branded wrapper around the legacy two-plane system (keeps Toliman-like pupil→focal behaviour; no new 1/f WFE added).
- ✅ Added `SheraTwoPlaneConfig` capturing two-plane structural knobs (pupil/PSF sampling, bandpass, aperture geometry/struts/DP hooks, primitive plate scale, optional primary Zernike basis; no secondary/relay geometry).
- ✅ Built `build_shera_twoplane_forward_spec_from_config`, mirroring the three-plane forward vocabulary with binary primitives, optional primary Zernikes, primitive plate scale, and derived log-flux via the shared transform set (no secondary terms).
- ✅ Updated the inference-spec builder so two- and three-plane runs share the same baseline astrometry/flux/plate-scale keys, with secondary Zernikes omitted when `include_secondary=False`.

**Follow-up implementation tasks (next steps)**
- ✅ Wired a `SheraTwoPlaneBinder`/SystemGraph path plus smoke tests to validate parity with the legacy two-plane model. Binder mirrors the three-plane API (forward-style base store with deriveds refreshed, `.model(store_delta)` public entry point, optional SystemGraph) and uses the same structural hash/cache pattern, now including plate scale as a structural knob sourced from the effective store. Optics and source both consume the merged store (base + delta) in graph and non-graph modes.
- Loss/optimisation stack now dispatches binders based on cfg type inside `make_binder_image_nll_fn`, so downstream helpers (`run_shera_image_gd`, `run_shera_image_gd_eigen`, FIM helpers) accept two- or three-plane configs without special casing.
- Graph templates remain per-variant for clarity (single-node cfg/spec/store → optics/source/detector → telescope.model). Consider factoring shared helpers or a base binder class if/when the systems converge further.
- ✅ Added a minimal two-plane astrometry demo mirroring the canonical three-plane example (`examples/scripts/run_twoplane_astrometry_demo.py` using `SheraTwoPlaneConfig` + `SheraTwoPlaneBinder`).
- Evaluate whether shared binder behaviour (two- vs three-plane) should live in a common base class once both paths exist.

---

## 21) Task 10 — Shared Binder / SystemGraph design options (analysis only)
Status: analysis-only; shared base implementations now exist (`core/binder.py`, `graph/system_graph.py`) with design intent captured in `docs/architecture/binder_and_graph.md`.

**Scope:** Planning-only comparison of the new SheraThreePlaneBinder/SystemGraph vs SheraTwoPlaneBinder/SystemGraph. No refactor performed; binders remain standalone classes.

### Binder comparison (inventory)

- **Constructor shape & stored attributes:**
  - Both binders accept `(cfg, forward_spec, base_forward_store, use_system_graph=True)`, validate the base store against the forward spec with deriveds allowed, and eagerly build a `LayeredDetector` downsample layer tied to `cfg.oversample`. Both optionally construct a SystemGraph instance and cache it on `_graph`; both hold `_detector` for reuse across calls.【F:src/dluxshera/core/binder.py†L22-L83】【F:src/dluxshera/core/binder.py†L150-L208】
  - Differences: `cfg`/optics builders are system-specific (`SheraThreePlaneConfig` + `build_shera_threeplane_optics` vs `SheraTwoPlaneConfig` + `build_shera_twoplane_optics`). SystemGraph factories differ accordingly (`build_shera_system_graph` vs `build_shera_twoplane_system_graph`).
- **model(store_delta) flow:**
  - Both expose `.model(store_delta=None)` as the single entry point: merge `store_delta` into `base_forward_store` via `_merge_store`, then either call `_graph.evaluate(..., outputs=("psf",))` when `use_system_graph` is enabled or fall back to `_direct_model` building optics/source/detector and calling `dl.Telescope.model()`.【F:src/dluxshera/core/binder.py†L61-L115】【F:src/dluxshera/core/binder.py†L188-L244】
  - Both support `.with_store(new_base_store)` to clone with a new base store while preserving cfg/spec/use_system_graph. No structural hash/caching hooks are defined at the binder layer (delegated to the optics builder caches).
- **Store handling:**
  - `_merge_store` logic is identical in spirit (validate overlay allowing missing/derived, disallow extra, then `.replace` on the base store) with only minor ordering differences of keyword arguments. Derived parameters are assumed pre-populated in the base forward store; binders do not recompute deriveds themselves.
- **System-specific responsibilities:**
  - Optics builder selection (`build_shera_threeplane_optics` vs `build_shera_twoplane_optics`).
  - Config types and implied structural knobs (three-plane Fresnel vs two-plane Toliman-style pixel-scale primitive). Structural-hash policies live inside the optics builders, not the binder.

### Binder behaviour categorization

- **Clearly shared & safe to factor:** parameter ownership (cfg/spec/base_forward_store), store-delta merge semantics, `.model(store_delta)` signature/flow, `use_system_graph` toggle with graph/direct parity, detector reuse, and `.with_store` immutability pattern.
- **Potentially shareable with care:** structural cache integration hooks (if binders begin surfacing structural hashes), and validation nuances for derived keys (ensuring both binders keep the same strictness knobs). These could be template methods on a base class or shared helper functions.
- **System-specific:** optics/detector construction details and config typing; any structural hash key selection remains tied to the respective optics builder (three-plane Fresnel geometry vs two-plane pixel-scale primitive) and should stay per-binder.

### SystemGraph comparison

- **Shape:** Both graphs are minimal single-node executors that validate `base_forward_store`, merge an optional `store_delta`, build system-specific optics, reuse a shared detector, construct the Alpha Cen source, instantiate a `dl.Telescope`, and return `psf` (or a dict when multiple outputs are requested). Each exposes `evaluate`/`forward`/`run` aliases and is built via a small factory (`build_shera_system_graph` or `build_shera_twoplane_system_graph`).【F:src/dluxshera/graph/system_graph.py†L1-L86】【F:src/dluxshera/graph/system_graph.py†L102-L157】
- **Common skeleton:** cfg + forward_spec + base_forward_store → `_merge_store(store_delta)` → system-specific `build_shera_*plane_optics` → `build_alpha_cen_source` → `dl.Telescope(...).model()` → psf/dict.
- **Differences:** Only the optics builder and cfg/detector typing vary; no additional nodes or transforms differentiate the graphs today.

### Design options (binder layer)

- **Option 1 — Shared base class (e.g., `BaseSheraBinder`):** Move the shared mechanics (init storing cfg/spec/base_store, detector construction, `_merge_store`, `.model` flow with graph/direct toggle, `.with_store`) into a base class with abstract hooks for `build_optics(eff_store)` and `build_graph(detector)`. Pros: eliminates duplication, centralizes immutability/validation semantics, eases future variants. Cons: introduces inheritance into a JAX-facing type (consider `frozen`/static fields), may obscure system identity unless type annotations remain explicit.
- **Option 2 — Composition/helpers (no inheritance):** Extract helper functions (e.g., `merge_store(base, delta, spec)`, `maybe_build_graph(cfg, spec, base, detector, builder_fn)`, `direct_model(cfg, builder_fn, detector, store)`) and let each binder compose them. Pros: avoids inheritance and keeps type clarity; lower risk to JIT/static-arg behaviour. Cons: some duplication remains; harder to enforce consistent immutability policies.
- **Option 3 — Meta-binder façade:** Introduce a `SheraBinder` factory that inspects cfg type and delegates to the appropriate binder (optionally leveraging Options 1 or 2 internally). Pros: single public entry point once multiple variants exist. Cons: can blur system identity for users and complicate typing; should not break direct construction of specific binder classes.
- **Recommendation:** Start with **Option 1 (base class)** or **Option 2 (helpers)**—leaning toward a small base class because the shared surface is already large and behaviourally identical. Preserve concrete `SheraTwoPlaneBinder`/`SheraThreePlaneBinder` types to maintain system clarity and backward compatibility. Meta-binder façade can remain a later opt-in convenience once more variants emerge.

### Design options (SystemGraph layer)

- **Option A — Shared skeleton with pluggable nodes:** Factor a `build_shera_system_graph(cfg, spec, base_store, detector, optics_fn)` helper that assembles the common merge→optics→source→telescope→psf path, with optics_fn capturing system-specific pieces. Pros: captures the evident template, reduces duplication, aligns with a base binder. Cons: marginal benefit while graphs stay single-node; must keep type clarity for cfg/optics.
- **Option B — Separate graphs + shared subgraphs:** Keep per-system graph classes but pull out reusable merge/source/telescope helpers. Pros: preserves explicit system identity and leaves room for divergent node shapes later. Cons: smaller deduplication win.
- **Option C — Status quo with documentation:** Accept duplication for now and document the parallel structure; revisit if/when graphs become multi-node or need caching/derived-resolution hooks. Pros: zero refactor risk; avoids premature abstraction. Cons: ongoing duplication and risk of drift.
- **Recommendation:** Tentatively **Option A** if/when graph complexity grows alongside a binder base class; until multi-node/caching features land, **Option C** is acceptable with clear comments noting the shared shape.

### Constraints/guardrails for any future refactor

- **JAX-friendliness:** Keep binders/graphs mostly static (cfg/spec/detector/graph cached); dynamic data should flow through `store_delta`/theta overlays. Avoid hidden state mutation during `model()` to remain JIT-safe.
- **Immutability:** Preserve the current “mostly immutable” contract and `.with_store` cloning pattern; no in-place mutations inside `model`/`evaluate`.
- **System identity clarity:** Even with shared bases/templates, it must stay obvious whether an instance is two-plane vs three-plane (distinct types or explicit mode flag). Backward compatibility for explicit `SheraThreePlaneBinder(...)` construction is required.
- **Backward compatibility:** Existing call sites and tests constructing the concrete binders should continue to work; any façade should be additive.

### Follow-on plan (future work, not yet implemented)

- Prototype a lightweight `BaseSheraBinder` encapsulating shared mechanics; migrate both binders with minimal surface changes and retain concrete subclasses for clarity.
- If/when graphs expand, introduce a shared graph builder helper mirroring the current merge→optics→source→telescope→psf skeleton, keeping system-specific optics functions injectable.
- Re-run binder/graph smoke tests for both systems after refactors; update docs to emphasize immutability/JAX constraints and system identity.

### Implementation follow-up (Task 10 landed)

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

**Intent:** Switch main dLuxShera over to the new ParamSpec / ParameterStore / Binder / SystemGraph stack as the canonical implementation. This is the point where I personally prefer the new stack for any real Shera work.

**Criteria for merge:**

- **Code & tests**
  - ParamSpec / ParameterStore / transforms / DerivedResolver are wired together and passing tests.
  - Optics builders (2- and 3-plane) use the new patterns and have basic test coverage.
  - Binder is the main way to instantiate and run models; SystemGraph is exercised in tests and at least one demo.
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
    - Binders & SystemGraphs: Binder as the user-facing “model object,” SystemGraph as underlying wiring.
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
- Keep the **doc stack coherent**: use this Working Plan as a “map of maps,” point to `docs/architecture/*.md` for details, and ensure README/tutorials stay in sync as the artifacts/logging work lands. Merge readiness (Milestone A) follows once these pieces are stable.

## 23) Documentation roadmap for dLuxShera
Status: docs housekeeping (dev-facing)

- Canonical long-range roadmap: `docs/architecture/roadmap.md`. Treat this as the theme-level plan; keep this Working Plan focused on near/medium-term execution and dev notes.
- Concept/architecture sources of truth: `docs/architecture/{binder_and_graph.md,eigenmodes.md,inference_and_loss.md,optimization_artifacts_and_plotting.md,params_and_store.md}`. Use these for detailed design rather than duplicating content here.
- Tutorials and modeling overview: `docs/tutorials/modeling_overview.md` and `docs/tutorials/canonical_astrometry_demo.md` (plus `examples/README.md` and `examples/scripts/run_canonical_astrometry_demo.py` / `run_twoplane_astrometry_demo.py` for runnable flows).
- Dev-facing planning: this file (`docs/dev/working_plan.md`) and any future dev notes under `docs/dev/`. Keep cross-links back to the architecture docs for specifics.
- Navigation helpers: `devtools/generate_context_snapshot.py` and `devtools/print_tree.py` remain the authoritative way to browse the live tree and ParamSpec/transform snapshots.

Near-term doc housekeeping:
- Keep this Working Plan as a “map of maps” that points to the architecture/tutorial docs and the roadmap, rather than re-explaining them.
- Ensure architecture docs and tutorials stay the canonical detail; keep this file focused on status, priorities, and where to look next.

## 24) Binder/SystemGraph shared implementation follow-through
Status: implemented; historical context

- Base implementations landed for binders and SystemGraphs (see `core/binder.py`, `graph/system_graph.py`). Direct binder/graph paths still share detectors and preserve immutability; optics builders remain system-specific.
- Implementation follow-up: caching and derived-resolution hooks remain future work as SystemGraph grows beyond single-node; see `docs/architecture/binder_and_graph.md` for the design intent and next hooks to add.

## 25) Parking Lot

- Two/Four-plane optics variant design and transforms.
- Extended inference methods (HMC, priors, eigenspace optimization) after core stack stabilizes.
- Ergonomic shims (`ModelParams`) and deprecation strategy for legacy APIs.
- High-level model design / capabilities documentation describing what the Shera-style model does (optical/astrometric forward model, main outputs, supported questions) and its key assumptions/approximations, written for proposal and systems-engineering consumers rather than just implementers.
- Model–error-budget interface and parameter dependency mapping: lightweight docs/figures that show how model outputs and sensitivities map onto specific error-budget terms, and how primitives vs. derived parameters (ParamSpec → Store → transforms) relate to those terms for traceability.

## 26) Implementation Plan — Optimization Artifacts + Signals + I/O (v0)

### 26.1 Current state (survey)

- **Optimization + packing surfaces:** θ-space loops live in `src/dluxshera/inference/optimization.py` (e.g., `run_simple_gd`, binder-aware `run_image_gd`, and Fisher helpers). Packing/unpacking utilities live in `src/dluxshera/params/packing.py`; binder NLL builders and theta mapping hooks are in `src/dluxshera/inference/losses.py` and `src/dluxshera/inference/inference.py`. IndexMap export exists via `run_artifacts.build_index_map(...)`; packing order is aligned with `ParamSpec.subset(...)`.
- **Transforms/DerivedResolver:** Transform registration and recursive resolution live in `src/dluxshera/params/registry.py`; Shera-specific transforms (plate scale, log flux, raw fluxes) are in `src/dluxshera/params/shera_threeplane_transforms.py`.
- **Plotting:** Refactor-era plotting helpers (PSF and parameter histories) are in `src/dluxshera/plot/plotting.py` with headless-friendly IO (return fig/axes, optional `save_path`). Signal builders and panel recipes for intro diagnostics live in `src/dluxshera/inference/{signals.py,plotting.py}` and feed optional run artifacts/plots.
- **Scripts/demos:** Canonical/binder-based runs are in `examples/scripts/run_canonical_astrometry_demo.py`, `examples/scripts/run_twoplane_astrometry_demo.py`, and `work/scratch/refactored_astrometry_retrieval.py`; artifact writing is opt-in and disabled by default.
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
- Status: v0 path uses `ema_grad2` at θ₀ to derive `curv_diag`, `precond`, and `lr_vec`; artifacts are emitted via `precond.npz` / `curvature.npz` with metadata recorded under `optimizer.preconditioning`. Covered by `tests/inference/test_precond_artifacts.py`.
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
