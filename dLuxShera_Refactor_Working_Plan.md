# dLuxShera Refactor — Working Plan & Notes
_Last updated: 2025-11-13 01:13_

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
2. **ParameterStore** — immutable values map. ✅ Implemented as frozen mapping + pytree; currently allows any keys (primitives/derived) depending on validation toggle.
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
- Derived and primitive fields coexist in single spec; separation by kind is not enforced in store validation.
- Cross-referencing between docs/tests and spec remains manual.

---

## 5) ParameterStore (Values)

- Frozen mapping `{key → value}` registered as JAX pytree; supports `.replace`, iteration, and `.validate` against a `ParamSpec`.
- Validation toggle allows derived keys; policy choice (primitives-only) still outstanding.
- No serialization helpers; no `refresh` convenience API.

---

## 6) Transform Registry / DerivedResolver

- Scoped `DerivedResolver` now wraps per-system `TransformRegistry` instances with system-id aware registration and resolution helpers (defaulting to the Shera three-plane system for backward compatibility).
- Three Shera transforms registered under `shera_threeplane`: focal length from focal ratio, plate scale from focal length, and log flux for brightness; analytic/legacy-consistency tests exist.
- Future work: expand transform coverage for additional systems (two-plane/four-plane) once those variants land.

**Policy on setting deriveds**
- Current code allows overriding deriveds if validation disabled. Target policy remains: disallow unless explicitly invertible.

---

## 7) Integrating dLux `ThreePlaneOpticalSystem`

- **Builder:** Frozen config dataclass and named point designs exist; builder constructs legacy `SheraThreePlaneSystem` and optionally injects Zernike coefficients from store. No structural hash/cache.
- **Binder:** Merges stores and forwards through static optics; canonical loss wrapper implemented and tested. Derived resolution step is not enforced; plate-scale binding policy unresolved.
- **Graph layer:** Minimal `DLuxSystemNode` + `SystemGraph` wrap the three-plane builder; still single-node with no caching/derived resolution enforcement.

---

## 8) Parameter Profiles & IO (Planned)

- Not yet implemented. Profiles (lab/instrument defaults), YAML/JSON loading, and serialization helpers remain to be built once primitives-only policy is finalized.

---

## 9) Docs & Examples (Planned)

- README, MkDocs structure, runnable notebooks, and scripted demos are not yet built. Existing examples rely on legacy flows.

---

## 10) Testing Philosophy

- Existing tests cover: ParamSpec/store validation and packing, transform resolution (including cycle guards), optics builder/binder smoke paths, optimization loss wrapper, eigenmode utilities, and SystemGraph smoke/regression via new graph tests.
- Missing: Demo workflows, four-plane variant tests, and serialization/profile coverage.

---

## 11) Gotchas & Decisions

- **Primitives-only store:** Decision still pending; current implementation can accept deriveds when validation is disabled.
- **Plate-scale policy:** Whether to always recompute vs allow override is still undecided.
- **Structural caching:** Builder currently rebuilds every call; caching keyed by structural primitives is planned.
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
- ⚠️ **ParamSpec core keys & docs**: Spec exists with metadata; ensure docstrings cross-reference tests/examples once SystemGraph lands.
- ⚠️ **ParameterStore policy**: Primitives-only enforcement + default validation mode; add optional `refresh` helper and serialization later.
- ✅ **Inference parameter packing**: `pack_params`/`unpack_params` with tests.
- ✅ **Transforms registry + psf_pixel_scale (three-plane)**: Global registry with three transforms and consistency tests.
- ⚠️ **ThreePlaneBuilder (structural hash/cache)**: Build path exists; add structural subset definition and caching policy.
- ⚠️ **ThreePlaneBinder (phase/sampling bind)**: Binder exists; clarify plate-scale policy and enforce derived resolution.
- ⚠️ **Canonical loss wiring**: New binder-based loss implemented; migrate examples once SystemGraph exists.
- ✅ **DLuxSystemNode / SystemGraph**: Minimal single-node scaffold wraps the three-plane builder; next steps are caching, multi-node wiring, and derived resolution enforcement.

**P1 — Docs, demos, and scope-aware transforms**
- ✅ **Scoped DerivedResolver**: System-ID scoping (three-plane/four-plane) and transform coverage expansion.
- ⏳ **Canonical astrometry demo**: Script/notebook to generate truth, synth data, run Optax with new loss.
- ⚠️ **Eigenmode parameterization**: FIM/eigen utilities exist; need optimizer/loss integration and docs.
- ⏳ **Docs & examples**: README quickstart, MkDocs pages, notebooks, updated Examples using new stack.
- ⏳ **Profile/IO helpers**: YAML/JSON profiles, serialization, and loading; depends on store policy.

**P2 — Variants & ergonomics**
- ⏳ **Four-plane variant**: Transforms, builder, resolver selection tests.
- ⏳ **Ergonomics**: `ModelParams` shim, deprecation warnings, upstream PR prep.

**Next sprint follow-ups**
- ⚠️ Canonical loss is in place; remaining sprint items: plate-scale policy decision, SystemGraph follow-ups (caching/multi-node), demo script.

**Newly noted tasks**
- ⏳ Structural hash/caching for three-plane builder.
- ⏳ Enforce primitives-only store in production mode.
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

3. **ParameterStore enforcement + serialization (P0)**
   - **Goal:** Enforce primitives-only store in validation by default; add `refresh` helper and serialization (`to_dict`/`from_dict` and YAML/JSON hooks).
   - **Files:** `src/dluxshera/params/{store.py, serialize.py}`, tests in `tests/params/test_store.py`.
   - **Dependencies:** Scoped resolver changes (deriveds recomputed); may affect binder validation.
   - **Risks/Ambiguities:** Handling legacy flows that may pass deriveds; migration path via explicit flag.
   - **Tests:** Validation rejecting derived keys, round-trip serialization, refresh behavior with transforms.

4. **Structural hash/cache for ThreePlaneBuilder (P1)**
   - **Goal:** Define structural subset and cache optics builds keyed by hash; avoid rebuilds during optimization.
   - **Files:** `src/dluxshera/optics/builder.py`, maybe `optics/config.py`, tests in `tests/optics/test_builder.py`.
   - **Dependencies:** ParameterStore enforcement to identify structural keys cleanly; SystemGraph can reuse cache.
   - **Risks/Ambiguities:** Hash stability across JAX types; invalidation when structural params change.
   - **Tests:** Cache hit/miss behavior, correctness after structural override, hash determinism.

5. **Canonical astrometry demo + docs (P1)**
   - **Goal:** Provide runnable script/notebook demonstrating truth generation, synthetic data, and Optax run using new binder/graph.
   - **Files:** `Examples/` (new script/notebook), `README.md`, possibly `docs/` scaffold if added.
   - **Dependencies:** SystemGraph and scoped resolver in place; canonical loss stable.
   - **Risks/Ambiguities:** Dataset location, runtime expectations; balancing simplicity with realism.
   - **Tests:** CI smoke run of demo (short), docstring/unit test ensuring example executes without errors.

---

## 17) Changelog of Decisions

- Transform registry implemented globally; slated for system-scoped refactor.
- ParameterStore currently permissive; decision pending to enforce primitives-only by default.
- Canonical binder-based loss added to inference stack; SystemGraph integration planned.
- Structural caching not yet started; acknowledged as necessary for performance.

---

## 18) Parking Lot

- Four-plane optics variant design and transforms.
- Extended inference methods (HMC, priors, eigenspace optimization) after core stack stabilizes.
- Ergonomic shims (`ModelParams`) and deprecation strategy for legacy APIs.

---
