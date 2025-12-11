# dLuxShera Refactor — Working Plan & Notes
_Last updated: 2026-02-18 00:00_

This is a living document summarizing the goals, architecture, decisions, tasks, and gotchas for the dLuxShera parameterization
refactor. It’s designed so either of us can get back up to speed quickly and not lose the important details.

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

**Layers (current vs target)**
1. **ParamSpec** — declarative schema & metadata (no numbers). ✅ Exists with `ParamField`/`ParamSpec` plus inference/forward builders; primitives and derived fields share the same spec.
2. **ParameterStore** — immutable values map. ✅ Implemented as frozen mapping + pytree with strict-by-default validation (rejects derived keys unless explicitly allowed) and helpers for refreshing/stripping deriveds.
3. **DerivedResolver (Scoped Transform Registry)** — computes derived values as pure functions. ⚠️ Registry exists and resolves recursive deps; currently global (not system-id scoped) and limited to three transforms.
4. **SystemGraph** — executable DAG of nodes. ⚠️ Minimal single-node scaffold exists (`DLuxSystemNode` + `SystemGraph`) wrapping the existing three-plane builder; still single-node/no caching.
5. **Facade** — `SheraThreePlane_Model` wrapper. ✅ Still primary entry point; internally uses partial refactor helpers.

ASCII sketch (target)
```
ParamSpec  ──(validates)──▶ ParameterStore (primitives)
     │                               │
     │                          (inputs)
     ▼                               ▼
Scoped Transform Registry ──▶ DerivedResolver (system_id-aware) ──▶ derived dict
     │                                                             │
     └─────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                         SystemGraph.forward()
                            │       ▲
        [Builder] ThreePlaneOpticalSystem  [Binder] (set numeric arrays, derived sampling)
                            │
                            ▼
                  .model(wavelengths, weights)  →  PSF
```

---

## 3) Repository Layout (Actual vs Proposed)

```
dLuxShera/
├─ pyproject.toml
├─ README.md
├─ dLuxShera_Refactor_Working_Plan.md
├─ src/dluxshera/
│  ├─ __init__.py
│  ├─ core/{builder.py, binder.py, modeling.py, universe.py}
│  ├─ params/{spec.py, store.py, transforms.py, packing.py, shera_threeplane_transforms.py}
│  ├─ optics/{config.py, builder.py, optical_systems.py}
│  ├─ inference/{inference.py, optimization.py}
│  ├─ utils/utils.py
│  └─ plot/plotting.py
├─ tests/
└─ Examples/
```

**Gaps vs proposal:** No `graph/`, `io/`, `viz/`, `docs/` tree; params registry/serialize modules absent; four-plane/variant code not present.

---

## 4) ParamSpec (Schema & Metadata)

- `ParamField`/`ParamSpec` implemented with docstrings, defaults, bounds, dtype/shape, and builders for forward/inference subsets.
- Forward spec now mirrors the full truth-level binary vocabulary with unit-aware keys (`binary.x_position_as`, `binary.y_position_as`, `binary.separation_as`, `binary.position_angle_deg`, `binary.contrast`) and conditionally includes Zernike coefficient arrays whose lengths are tied to configured Noll indices, defaulting to zero vectors when a basis exists.
- Derived and primitive fields coexist in single spec; separation by kind is not enforced in store validation.
- Cross-referencing between docs/tests and spec remains manual.

---

## 5) ParameterStore (Values)

- Frozen mapping `{key → value}` registered as JAX pytree; supports `.replace`, iteration, and `.validate` against a `ParamSpec`.
- Validation is strict by default (rejects derived keys); override/debug mode opt-in via `allow_derived=True`. Helpers `strip_derived`, `refresh_derived`, and `check_consistency` keep derived values fresh or flag stale overrides.
- Canonical forward-store flow is primitives-only defaults → explicit primitive overrides → `refresh_derived(...)` to populate plate scale / log flux and other deriveds; derived keys are omitted from `from_spec_defaults` by design.
- Shallow serialization helpers exist (`from_dict`, `from_spec_defaults`, `as_dict`); YAML/JSON IO still planned in the profiles/IO workstream.

---

## 6) Transform Registry / DerivedResolver

- Scoped `DerivedResolver` now wraps per-system `TransformRegistry` instances with system-id aware registration and resolution helpers (defaulting to the Shera three-plane system for backward compatibility).
- Three Shera transforms registered under `shera_threeplane`: focal length from focal ratio, plate scale from focal length, and log flux for brightness; analytic/legacy-consistency tests exist (now exercised via a primitives-only forward-store refresh pattern).
- Future work: expand transform coverage for additional systems (two-plane/four-plane) once those variants land.

**Policy on setting deriveds**
- Current code allows overriding deriveds if validation disabled. Target policy remains: disallow unless explicitly invertible.

---

## 7) Integrating dLux `ThreePlaneOpticalSystem`

- **Builder:** Frozen config dataclass and named point designs exist; builder constructs legacy `SheraThreePlaneSystem`, optionally injects Zernike coefficients from store, and now caches structural builds via `structural_hash_from_config` + `clear_threeplane_optics_cache` (opt-out env flag `DLUXSHERA_THREEPLANE_CACHE_DISABLED`).
- **Binder:** Merges stores and forwards through static optics; canonical loss wrapper implemented and tested. Derived resolution step is not enforced; plate-scale binding policy unresolved.
- **Graph layer:** Minimal `DLuxSystemNode` + `SystemGraph` wrap the three-plane builder; still single-node with no caching/derived resolution enforcement.

---

## 8) Parameter Profiles & IO (Planned)

- Not yet implemented. Profiles (lab/instrument defaults), YAML/JSON loading, and serialization helpers remain to be built once primitives-only policy is finalized.

---

## 9) Docs & Examples (Planned)

- Canonical binder/SystemGraph astrometry demo now lives in `Examples/scripts/run_canonical_astrometry_demo.py` with both pure-θ and eigenmode gradient descent flows. README/MkDocs pages and additional notebooks remain to be authored.

---

## 10) Testing Philosophy

- Existing tests cover: ParamSpec/store validation and packing, transform resolution (including cycle guards), optics builder/binder smoke paths, optimization loss wrapper, eigenmode utilities, SystemGraph smoke/regression via new graph tests, and the canonical astrometry demo in fast mode.
- Missing: Four-plane variant tests and serialization/profile coverage.

---

## 11) Gotchas & Decisions

- **Primitives-only store:** Decision still pending; current implementation can accept deriveds when validation is disabled.
- **Plate-scale policy:** Whether to always recompute vs allow override is still undecided.
- **Structural caching:** Three-plane builder now caches structural builds keyed by a deterministic hash and exposes a cache clear helper (env flag available to disable caching).
- **Scopes:** Per-system scoping added via `DerivedResolver`; ergonomics for additional variants will matter as new systems arrive.

---

## 12) Open Questions

- Final policy for accepting derived keys in ParameterStore (`validate` default vs production enforcement).
- Whether to expose alias setters for invertible deriveds (e.g., pixel scale) or force primitive updates only.
- Canonical plate-scale handling in binder (always derived? allow override?).
- Structural hash definition for three-plane optics (which primitives are structural?).

---

## 13) Notes on Backward Compatibility

- `SheraThreePlane_Model` remains the public entry point; new plumbing should remain internal to avoid churn in existing scripts.
- Legacy files (`modeling.py`, optics helpers) still carry pre-refactor pathways; the refactor must avoid breaking current examples until replacements land.

---

## 14) Prior Art / References

- dLux core APIs for `ThreePlaneOpticalSystem` and PSF generation.
- Prior optimization scripts in `Examples/` (still legacy-style; to be updated after SystemGraph lands).

---

## 15) Tasks & Priorities (Updated)

Legend: ✅ Implemented · ⚠️ Partial · ⏳ Not implemented

**P0 — Stabilize primitives/derived boundary & binder**
- ⚠️ **ParamSpec core keys & docs**: Spec exists with metadata; forward spec now includes unit-aware binary astrometry and Noll-index-tied Zernike coeffs with zero defaults; ensure docstrings cross-reference tests/examples once SystemGraph lands.
- ✅ **ParameterStore policy**: Primitives-only defaults enforced; canonical flow documented (`from_spec_defaults` → primitive overrides → `refresh_derived`); serialization still to follow.
- ✅ **Inference parameter packing**: `pack_params`/`unpack_params` with tests.
- ✅ **Transforms registry + psf_pixel_scale (three-plane)**: Global registry with three transforms and consistency tests.
- ✅ **ThreePlaneBuilder (structural hash/cache)**: Structural subset documented, deterministic hash added, cache + clear helper in builder with opt-out env flag.
- ⚠️ **ThreePlaneBinder (phase/sampling bind)**: Binder exists; clarify plate-scale policy and enforce derived resolution.
- ⚠️ **Canonical loss wiring**: New binder-based loss implemented; migrate examples once SystemGraph exists.
- ✅ **DLuxSystemNode / SystemGraph**: Minimal single-node scaffold wraps the three-plane builder; next steps are caching, multi-node wiring, and derived resolution enforcement.

**P1 — Docs, demos, and scope-aware transforms**
- ✅ **Scoped DerivedResolver**: System-ID scoping (three-plane/four-plane) and transform coverage expansion.
- ✅ **Canonical astrometry demo**: Script/notebook generates truth + synthetic data, runs binder/SystemGraph loss with Optax, and now includes a mirrored eigenmode gradient-descent segment using EigenThetaMap.
- ✅ **Eigenmode parameterization**: FIM/eigen utilities integrated into the inference API with an eigen-space GD helper, demo walkthrough, and regression tests.
- ⏳ **Docs & examples**: README quickstart, MkDocs pages, notebooks, updated Examples using new stack.
- ⏳ **Profile/IO helpers**: YAML/JSON profiles, serialization, and loading; depends on store policy.

**P2 — Variants & ergonomics**
- ⏳ **Four-plane variant**: Transforms, builder, resolver selection tests.
- ⏳ **Ergonomics**: `ModelParams` shim, deprecation warnings, upstream PR prep.

**Next sprint follow-ups**
- ⚠️ Plate-scale policy decision and SystemGraph follow-ups (caching/multi-node) remain. Extend docs/README coverage and add serialization/profile helpers to support the new demo + eigenmode pathways.
- ⚠️ Sweep remaining scripts/tests for legacy unit-less astrometry aliases once the canonical unit-aware binary keys have settled.

**Newly noted tasks**
- ✅ Structural hash/caching for three-plane builder (cache + clear helper landed).
- ✅ Enforce primitives-only store in production mode (strict validation default, refresh/strip helpers).
- ⏳ Add serialization (`params/serialize.py`) and transform registry module (`params/registry.py`).
- ⏳ Extend SystemGraph with caching/derived resolution hooks once resolver is scoped.

---

## 16) Recommended Next 3–5 Tasks (to reach end-to-end flow)

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
   - **Status:** ✅ Added `Examples/scripts/run_canonical_astrometry_demo.py` using ParamSpec + ParameterStore + DerivedResolver to build truth/variant stores, SheraThreePlaneBinder/SystemGraph forward model, and Optax GD with prior penalties. README updated with run command; smoke test exercises `main(fast=True)`.

---

## 17) ParamSpec + ParameterStore policy (plate_scale/log_flux) — analysis & options

**Current behavior (“split personality”)**
- `build_inference_spec_basic()` marks `system.plate_scale_as_per_pix` and `binary.log_flux_total` as **primitive knobs** for optimisation.【F:src/dluxshera/params/spec.py†L95-L167】【F:src/dluxshera/params/spec.py†L207-L245】
- `build_forward_model_spec_from_config()` mirrors geometry/throughput primitives from `SheraThreePlaneConfig` and declares `system.plate_scale_as_per_pix` and `binary.log_flux_total` as **derived** with registered transforms (geometric plate scale and collecting-area × band × throughput flux).【F:src/dluxshera/params/spec.py†L337-L458】
- The transform registry is **store-wins**: if a key is present in the `ParameterStore`, the transform is skipped; otherwise dependencies are resolved recursively.【F:src/dluxshera/params/registry.py†L117-L186】 Tests exercise this by computing plate scale/log flux from a forward-model store seeded with primitives only.【F:tests/test_shera_threeplane_transforms.py†L1-L71】
- `ParameterStore.from_spec_defaults()` skips derived fields, so a forward-model store built from defaults contains only primitives unless the caller injects derived values. Validation today checks only key set equality and can allow extras; it does **not** enforce “primitives-only”.【F:src/dluxshera/params/store.py†L72-L167】【F:src/dluxshera/params/store.py†L202-L251】

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

---

## 18) Changelog of Decisions

- Transform registry implemented globally; slated for system-scoped refactor.
- ParameterStore currently permissive; decision pending to enforce primitives-only by default.
- Canonical binder-based loss added to inference stack; SystemGraph integration planned.
- Structural caching not yet started; acknowledged as necessary for performance.

---

## 19) Parking Lot

- Two/Four-plane optics variant design and transforms.
- Extended inference methods (HMC, priors, eigenspace optimization) after core stack stabilizes.
- Ergonomic shims (`ModelParams`) and deprecation strategy for legacy APIs.
- High-level model design / capabilities documentation describing what the Shera-style model does (optical/astrometric forward model, main outputs, supported questions) and its key assumptions/approximations, written for proposal and systems-engineering consumers rather than just implementers.
- Model–error-budget interface and parameter dependency mapping: lightweight docs/figures that show how model outputs and sensitivities map onto specific error-budget terms, and how primitives vs. derived parameters (ParamSpec → Store → transforms) relate to those terms for traceability.

---
