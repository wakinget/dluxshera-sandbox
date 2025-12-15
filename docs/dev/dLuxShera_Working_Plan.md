# dLuxShera Working Plan & Notes (dev-facing)
_Last updated: 2026-02-18 00:00_

This is a living, dev-facing document summarizing the goals, architecture, decisions, tasks, and gotchas for dLuxShera as it moves through V1.0 and beyond. It replaces the refactor-era index while keeping the running plan in one place.

## How to use this doc
- **Sections 1–17:** Current architecture focus areas, gotchas, and open questions.
- **Section 18:** Merge strategy and V1.0 milestones (active roadmap).
- **Section 23:** Parking lot / future ideas and backlog.
- **Historical context:** For narrative history and ADR-style rationale, see `docs/archive/REFACTOR_HISTORY.md` and `docs/archive/ARCHITECTURE_DECISIONS.md`.

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

For narrative context and the rationale behind these layers, see `docs/archive/ARCHITECTURE_DECISIONS.md`.

---

## 3) Repository Layout (Actual vs Proposed)

```
dLuxShera/
├─ pyproject.toml
├─ README.md
├─ docs/
│  ├─ dev/dLuxShera_Working_Plan.md
│  └─ archive/
├─ src/dluxshera/
│  ├─ __init__.py
│  ├─ core/{builder.py, binder.py, modeling.py, universe.py}
│  ├─ params/{spec.py, store.py, transforms.py, packing.py, shera_threeplane_transforms.py}
│  ├─ optics/{config.py, builder.py, optical_systems.py}
│  ├─ inference/{inference.py, optimization.py}
│  ├─ utils/utils.py
│  └─ plot/plotting.py
├─ tests/
└─ examples/
```

**Gaps vs proposal:** No `graph/`, `io/`, `viz/`, `docs/` tree; params registry/serialize modules absent; four-plane/variant code not present.

---

## 4) ParamSpec (Schema & Metadata)

- `ParamField`/`ParamSpec` implemented with docstrings, defaults, bounds, dtype/shape, and builders for forward/inference subsets.
- Forward spec now mirrors the full truth-level binary vocabulary with unit-aware keys (`binary.x_position_as`, `binary.y_position_as`, `binary.separation_as`, `binary.position_angle_deg`, `binary.contrast`) and conditionally includes Zernike coefficient arrays whose lengths are tied to configured Noll indices, defaulting to zero vectors when a basis exists.
- Derived and primitive fields coexist in single spec; separation by kind is not enforced in store validation.
- Cross-referencing between docs/tests and spec remains manual.
- **Include vs exclude helpers:** `ParamSpec.subset(keys)` stays strictly include-only, preserving caller-provided ordering and raising on unknown keys. A complementary `ParamSpec.without(keys)` now drops whole fields by key (including grouped/vector fields such as `primary.zernike_coeffs`) while preserving the original ordering of everything else; it raises on unknown keys for consistency with `subset`. Tests cover no-op, full-drop, complement equivalence, and vector-field removal cases.

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

- **Builder:** Frozen config dataclass and named point designs exist; builder constructs legacy `SheraThreePlaneSystem`, optionally injects Zernike coefficients from a forward store, and now caches structural builds via `structural_hash_from_config` + `clear_threeplane_optics_cache` (opt-out env flag `DLUXSHERA_THREEPLANE_CACHE_DISABLED`).
- **Binder (canonical model object):** `SheraThreePlaneBinder` now owns a forward-style `ParamSpec`, a base forward store with deriveds refreshed, and constructs its `SystemGraph` eagerly. `.model(store_delta)` is the sole public entry point; `.forward` has been removed. Bindings are **mostly immutable (Option A)**—use `.with_store(...)` to derive new binders rather than mutating in-place. Binder treats SystemGraph as an internal detail (Binder → Graph → builders) and merges store deltas against the stored forward base.
  - Option A (implemented): static binder captured in JAX closures; dynamic behaviour comes from `store_delta` / θ overlays. Example usage:
    ```python
    binder = SheraThreePlaneBinder(cfg, forward_spec, base_store, use_system_graph=True)
    psf = binder.model()
    psf_delta = binder.model(delta_store)
    binder2 = binder.with_store(new_store)
    ```
  - Option B (documented only): mutable binder updating cfg/base store in-place was rejected due to JIT/static-arg friendliness and legacy immutability expectations.
- **Graph layer:** `SystemGraph` is now constructed eagerly and owned by Binder when `use_system_graph=True`; execution reuses the optics builder (and its structural-hash cache) and Alpha Cen source inside a lightweight `evaluate(outputs=("psf",))` call. Graph remains single-node but mirrors binder output for smoke/regression tests.

### Loss stack (Binder-centric clarity)

- **Flow:** `theta` → `ParameterStore` delta (restricted to `infer_keys`) → `binder.model(store_delta)` → PSF image → `gaussian_image_nll(image, data, var)` → scalar loss. This keeps Binder as the sole model API and makes the Gaussian NLL math transparent.
- **Primitives:** A glass-box `gaussian_image_nll` helper (JAX-friendly, sum/mean/None reductions) now lives in `inference/losses.py` and is used by both the legacy `gaussian_loss` wrapper and Binder-based constructors.
- **Binder loss constructor:** `make_binder_image_nll_fn` returns `(loss_fn, theta0)` where `loss_fn(theta)` unpacks `theta` into a store delta, evaluates `binder.model(...)`, and applies the chosen noise model. A pre-built binder can be passed in or constructed internally from `(cfg, forward_spec, base_forward_store)`.
- **Demo wiring:** `examples/scripts/run_canonical_astrometry_demo.py` now reads linearly: build the Binder image NLL, then add a Gaussian prior penalty for a MAP objective. Comments spell out the `theta → store → binder.model → image → NLL` path.

### Eigenmode-based optimisation (clarified)

- **Pure-θ vs eigen-θ:** Both optimisation paths now share the same Binder-based image NLL. Pure-θ gradient descent calls the loss directly. Eigen-θ runs define `loss_z(z) = loss_theta(EigenThetaMap.theta_from_z(z))`, so the only difference is a linear change of coordinates (plus optional truncation/whitening).
- **One-time setup:** `run_shera_image_gd_eigen` builds the Binder loss via `make_binder_image_nll_fn`, computes a θ-space curvature/FIM once at `theta_ref`, constructs `EigenThetaMap` (supports truncation/whitening), and JITs the z-space loss/grad. Per-iteration cost is therefore one Binder-based loss/grad plus cheap linear transforms; Binder/SystemGraph construction and pack/unpack scaffolding stay outside the hot loop.
- **Documentation/API:** `EigenThetaMap` now carries a detailed docstring (column eigenvectors, whitening meaning, shapes) and exposes `theta_from_z` / `z_from_theta` aliases alongside the legacy names for clarity. It is JAX-friendly (uses `jax.numpy` throughout) and safe inside jitted regions.
- **Demo story:** The canonical astrometry script comments now state that the eigen run reuses the Binder NLL, explain curvature estimation, and describe whitening (unit-curvature scaling) and truncation (top-k modes) options. Aside from the one-time eigendecomposition, per-step cost should mirror the pure-θ path.

### Prior handling (landscape + new abstraction)

- **Legacy/observed patterns:**
  - `inference/optimization.py::construct_priors_from_dict` builds NumPyro distributions (Normal/Uniform/LogNormal) directly from a `{param: {mean, sigma, dist}}` mapping; tightly coupled to NumPyro and not used elsewhere in the refactor flow.
  - The canonical astrometry demo defines a `priors` dict of sigmas keyed by `ParamKey`, manually jitters the truth store with NumPy noise, and hand-computes a quadratic MAP penalty keyed by the same sigmas (supports both scalars and Zernike coefficient vectors). The notebook `examples/notebooks/Shera_Eigen_Inference_Example.ipynb` mirrors this pattern, including NumPyro helper usage and prior-drawn perturbations.
- **New backend-agnostic layer:** `inference/prior.py` introduces `PriorField` (currently Normal-only mean/sigma) and `PriorSpec` (key→field mapping) with helpers for: (a) `from_sigmas(center_store, sigmas)` to seed priors at reference values, (b) `quadratic_penalty(store, center_store)` for MAP-style sums over `(value-mean)^2/(2*sigma^2)`, and (c) `sample_near(center_store, rng_key)` to jitter an initial store from the priors. These are pure JAX operations with no PPL dependency.
- **NumPyro bridge stub:** `inference/numpyro_bridge.py` documents the intended adapter surface (`numpyro_priors_from_spec`) but intentionally raises `NotImplementedError` to avoid hard-coding backend logic yet.
- **Current demo usage:** The canonical demo now builds a `PriorSpec` from the `priors` sigma dict, uses `sample_near(...)` for its initialisation jitter, and wraps the MAP penalty via `quadratic_penalty(...)`, keeping binder/SystemGraph wiring unchanged.
- **Integration plan:** Future work will thread `PriorSpec` through binder-aware loss constructors (e.g., MAP penalties layered alongside data NLL) and PPL adapters while keeping the core abstraction backend-agnostic and ParamKey-addressable.

---

## 8) Parameter Profiles & IO (Planned)

- Not yet implemented. Profiles (lab/instrument defaults), YAML/JSON loading, and serialization helpers remain to be built once primitives-only policy is finalized.

---

## 9) Docs & examples (Phase 1 shipped)

- Canonical binder/SystemGraph astrometry demo now lives in `examples/scripts/run_canonical_astrometry_demo.py` with both pure-θ and eigenmode gradient descent flows. README/MkDocs pages and additional notebooks remain to be authored.
  - The demo now showcases the refactor-era plotting helpers: PSF visualisation via `plot_psf_single` / `plot_psf_comparison` and parameter trajectories via `plot_parameter_history_grid`. Plotting utilities follow the IO policy (return fig/axes; caller decides to save/show), and the demo saves figures when a destination directory is provided (keeping smoke tests headless). Future follow-ons could add eigenmode-specific diagnostics (eigenvalue spectra, mode loadings) and prior visualisation once the pattern stabilises.
- Phase 1 documentation skeleton exists under `docs/`: `docs/modeling_overview.md` (conceptual entry), `docs/tutorials/canonical_astrometry_demo.md` (walkthrough), architecture stubs, and dev notes (this working plan in `docs/dev/dLuxShera_Working_Plan.md`). Next steps include fleshing out architecture details and adding the forthcoming two-plane tutorial.

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
- Prior optimization scripts in `examples/` (still legacy-style; to be updated after SystemGraph lands).

---

## 15) Tasks & Priorities (Updated)

Legend: ✅ Implemented · ⚠️ Partial · ⏳ Not implemented

**P0 — Stabilize primitives/derived boundary & binder**
- ⚠️ **ParamSpec core keys & docs**: Spec exists with metadata; forward spec now includes unit-aware binary astrometry and Noll-index-tied Zernike coeffs with zero defaults; ensure docstrings cross-reference tests/examples once SystemGraph lands.
- ✅ **ParamSpec subset ergonomics (include vs exclude)**: Confirmed `subset(keys)` remains include-only; added `ParamSpec.without(keys)` complement with ordering preservation and unknown-key errors plus regression tests in `tests/test_params_spec.py`.
- ✅ **ParameterStore policy**: Primitives-only defaults enforced; canonical flow documented (`from_spec_defaults` → primitive overrides → `refresh_derived`); serialization still to follow.
- ✅ **Inference parameter packing**: `pack_params`/`unpack_params` with tests.
- ✅ **Transforms registry + psf_pixel_scale (three-plane)**: Global registry with three transforms and consistency tests.
- ✅ **ThreePlaneBuilder (structural hash/cache)**: Structural subset documented, deterministic hash added, cache + clear helper in builder with opt-out env flag.
- ✅ **ThreePlaneBinder (phase/sampling bind)**: Binder is the canonical, mostly immutable model; owns an eager SystemGraph, uses forward-style base stores with deriveds refreshed, and exposes `.model(store_delta)` as the public API.
- ✅ **Loss function clarity / canonical loss wiring**: Binder-based image NLL helpers now route through `binder.model(store_delta)` with explicit θ→store mappings, reuse the glass-box `gaussian_image_nll` kernel, and surface a clear `(loss_fn, theta0)` API. The astrometry demo mirrors this flow and layers a Gaussian prior for MAP loss.
- ✅ **DLuxSystemNode / SystemGraph**: Minimal single-node scaffold wraps the three-plane builder; eager construction now lives under Binder ownership (caching/derived resolution hooks still future work).

**P1 — Docs, demos, and scope-aware transforms**
- ✅ **Scoped DerivedResolver**: System-ID scoping (three-plane/four-plane) and transform coverage expansion.
- ✅ **Canonical astrometry demo**: Script/notebook generates truth + synthetic data, runs binder/SystemGraph loss with Optax, and now includes a mirrored eigenmode gradient-descent segment using EigenThetaMap.
- ✅ **Eigenmode parameterization**: FIM/eigen utilities integrated into the inference API with an eigen-space GD helper, demo walkthrough, and regression tests.
- ⏳ **Docs & examples**: README quickstart, MkDocs pages, notebooks, updated examples using new stack.
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

- **Future improvements:** integrate a declarative PriorSpec/NumPyro-friendly prior layer into the loss stack, and add alt-noise kernels (e.g., Poisson) following the same glass-box helper pattern.

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
   - **Status:** ✅ Added `examples/scripts/run_canonical_astrometry_demo.py` using ParamSpec + ParameterStore + DerivedResolver to build truth/variant stores, SheraThreePlaneBinder/SystemGraph forward model, and Optax GD with prior penalties. README updated with run command; smoke test exercises `main(fast=True)`.
   - **Two-plane companion:** Added `examples/scripts/run_twoplane_astrometry_demo.py` as a lighter-weight analogue that exercises the SheraTwoPlaneConfig/Binder stack; both demos serve as reference examples for upcoming docs/tutorials.

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
- Diagnostics & UX (Option C — Phase 1): plotting utilities have been normalised to a refactor-era IO policy. PSF visualisation (`plot_psf_single`, `plot_psf_comparison`), parameter history panels (`plot_parameter_history`, `plot_parameter_history_grid`), colourbar alignment (`merge_cbar`), OPD and sweep helpers now return figures/axes, avoid implicit `plt.show()`, and support explicit `save_path` for headless/CI usage. Open follow-ups: dedicated FIM/eigen visualisation helpers (spectra + loadings), richer `ParameterStore` inspection/pretty-printing, and a simple logging/trace container that integrates with plotting and canonical demos.

---

## 18) Changelog of Decisions

See `docs/archive/ARCHITECTURE_DECISIONS.md` for a curated, ADR-style summary of the major choices referenced here.

- Transform registry implemented globally; slated for system-scoped refactor.
- ParameterStore currently permissive; decision pending to enforce primitives-only by default.
- Canonical binder-based loss added to inference stack; SystemGraph integration planned.
- Structural caching not yet started; acknowledged as necessary for performance.

---

## 19) Legacy Shera two-plane stack → refactor-era mapping (analysis)

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
- *Config + forward spec*: Introduce `SheraTwoPlaneConfig` alongside the three-plane config, sharing binary vocabulary, wavelength/bandwidth sampling, and primary Zernike basis fields. Two-plane-specific primitives: `psf_pixel_scale` (primitive, arcsec/pix), primary aperture geometry (p1/p2 diameters, strut geometry, diffractive pupil design wavelength), sampling (`pupil_npix`, `psf_npix`, `oversample`). Exclude three-plane-only fields (focal lengths, plane separation, detector pixel size, secondary basis/1/f). Forward spec should mirror the binary vocabulary used by the three-plane builder (unit-aware `binary.x_position_as`, `binary.y_position_as`, `binary.separation_as`, `binary.position_angle_deg`, `binary.contrast`), include `primary.zernike_coeffs` when a basis is configured (default zeros), omit any secondary terms, treat `psf_pixel_scale` as primitive (no transform), and derive `binary.log_flux_total` via the same transform family as the three-plane system.
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

## 18) Merge Strategy and V1.0 Milestones

This section captures our strategy for (a) deciding when to merge the refactor work into the main dLuxShera repo, and (b) when to consider the refactor “done” and treat the current architecture as V1.0. There are currently no external users of the main repo; migration concerns are therefore purely for my own workflow and notebooks.

- Historical rationale for the refactor lives in `docs/archive/REFACTOR_HISTORY.md` and `docs/archive/ARCHITECTURE_DECISIONS.md`; this section is about the current merge/V1.0 strategy.
- V1.0 user-facing docs should describe the current architecture as the default without surfacing “refactor” or “legacy” language.

---

### 18.1 Goals

- Present a clean, “this is how dLuxShera works” story to future users and collaborators.
- Avoid user-facing mentions of “refactor” or “legacy” once V1.0 is in place.
- Use the current sandbox / refactor branch to harden the architecture and demos before merging into main.
- Treat “merge to main” and “V1.0” as related but distinct milestones.

---

### 18.2 Milestone A – Merge Refactor Branch into Main

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

### 18.3 Milestone B – V1.0 Architecture & Documentation

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

### 18.4 Near-Term Focus

- Finish tightening the **basic doc structure** (initial architecture concept pass done):
  - Draft or refine the V1.0-style README.
  - Solidify the canonical three-plane demo doc and script.
  - Expand concept docs as needed beyond the initial Parameters/Stores, Binders/SystemGraphs, inference, and eigenmodes coverage.
- Once those are in place and tests remain green, proceed with **Milestone A (merge into main)**.
- After merge, iterate toward **Milestone B (V1.0)** by:
  - Polishing the two-plane demo.
  - Filling in concept docs.
  - Migrating the Working Plan and dev notes into `docs/dev/`.

- Implemented `BaseSheraBinder` in `core/binder.py`, with SheraThreePlaneBinder and SheraTwoPlaneBinder now inheriting shared merge/model/with_store semantics while preserving public signatures and system-specific optics/graph hooks.
- Added `BaseSheraSystemGraph` skeleton in `graph/system_graph.py`, adopted by the three- and two-plane SystemGraphs to centralize store merge + evaluation flow; factories remain unchanged.
- Binder/graph smoke and regression tests cover both systems plus shared output/merge behaviour; public APIs and numerical paths are unchanged.

## 23) Binder/SystemGraph shared implementation follow-through

- Summary: Base implementations landed for binders and SystemGraphs (see `core/binder.py`, `graph/system_graph.py`, and new tests). Direct binder/graph paths still share detectors and preserve immutability; optics builders remain system-specific.
- Remaining follow-ups: consider further consolidation if more variants arrive (e.g., four-plane), and revisit caching/derived-resolution hooks once multi-node graphs emerge.


## 22) Documentation roadmap for dLuxShera

This roadmap sketches the documentation “stack” we want for dLuxShera, how the pieces fit together, and where they live in the repo. It’s meant to guide incremental work rather than be implemented all at once.

High-level goals
----------------

- Provide multiple entry points for different audiences:
  - New users who just want to run a demo.
  - Collaborators who need the conceptual model design.
  - Developers who need to understand ParamSpec/ParameterStore, Binder, SystemGraph, and inference.
- Separate conceptual model design from implementation details:
  - Concept docs explain “how the model thinks about the world”.
  - Architecture/API docs explain “how this is encoded in code”.
- Keep dev-facing planning material (Working Plan, design notes) clearly separated from user-facing docs, but cross-linked.
- Make it easy to navigate: each doc points “up” to context and “down” to details.

Proposed doc families
---------------------

We’ll organise docs into a few coherent families:

1. Top-level orientation  
2. Conceptual model design  
3. Architecture & system design  
4. Tutorials & workflows  
5. Developer & planning docs (including the Working Plan)  
6. API & reference (later / optional automation)

Each family is described below with suggested filenames and content boundaries.

1) Top-level orientation
------------------------

Doc: README.md (root)

Audience: Anyone landing on the repo.

Purpose:
- Briefly answer:
  - What is dLuxShera?
  - What problem does it solve (Shera/TOLIMAN astrometric modeling)?
  - How does it relate to dLux/dLuxToliman and JAX?
- Provide a “choose your own path” section linking into docs:
  - Run the canonical astrometry demo
  - Understand the modeling pipeline
  - Modify or extend the code

Status: Exists; will be updated once the docs tree is in place.

2) Conceptual model design
--------------------------

Doc: docs/modeling_overview.md (new)

Audience: Model users and colleagues (including non-Python folks) who need the “big picture”.

Purpose:
- Provide a high-level overview of the modeling pipeline:
  cfg → forward_spec → forward_store (+derived) → Binder/SystemGraph → PSF → loss(θ) → optimisation → eigenmodes → priors.
- Explain:
  - Configs and point designs  
  - Forward vs inference spaces  
  - ParameterStore: primitives vs derived  
  - Binder as the canonical generative model object  
  - Pure-θ vs eigen-θ optimisation
  - Where priors fit (MAP or PPL)
- Include simple diagrams or flow charts.

3) Architecture & system design
-------------------------------

These documents zoom in one level to describe how subsystems work in code.

A. docs/architecture/params_and_store.md  
   Covers:
   - ParamField & ParamSpec (defaults, primitive/derived flags, forward vs inference)  
   - ParameterStore behavior (from_spec_defaults, replace, validation)  
   - Derived transforms & policy  
   - subset(keys) vs without(keys)  
   - Zernike basis handling tied to config

B. docs/architecture/binder_and_graph.md  
   Covers:
   - SheraThreePlaneBinder as the primary model object  
   - SystemGraph as the internal DAG  
   - Structural hash + optics builder caching  
   - How binder.model(store_delta) works step-by-step  
   - Relationship to legacy SheraThreePlane_Model

C. docs/architecture/inference_and_loss.md  
   Covers:
   - InferenceSpec + infer_keys  
   - θ packing/unpacking  
   - gaussian_image_nll (primitive)  
   - make_binder_image_nll_fn (Binder bridge)  
   - MAP losses and PriorSpec  
   - run_shera_image_gd_basic

D. docs/architecture/eigenmodes.md  
   Covers:
   - θ → FIM → EigenThetaMap → z-space  
   - Whitening/truncation  
   - How eigen-based optimisation wraps the same Binder-based θ-loss  
   - Conceptual guarantees and example diagrams

4) Tutorials & workflows
------------------------

Narrative, step-by-step guides for using the system.

Initial tutorial: docs/tutorials/canonical_astrometry_demo.md  
- Walk through the canonical demo script:
  1. Build config, forward spec/store, truth PSF  
  2. Build inference_spec + infer_keys  
  3. Create Binder, build NLL  
  4. Add priors (dict or PriorSpec)  
  5. Run pure-θ GD  
  6. Run eigen-θ GD  
  7. Plot and inspect recovered parameters  

Later tutorials may include:
- Adding new parameters  
- Defining new optical variants  
- Trying different noise models  

5) Developer & planning docs
----------------------------

These are internal docs for maintainers.

A. Move Working Plan into docs/dev/  
   - Source of truth for planning and design decisions  
   - Lives under docs/dev/ to keep root clean  
   - Cross-linked from architecture docs where appropriate

B. docs/dev/code_structure.md  
   - Summarises the repo layout  
   - Links major modules to the relevant architecture docs  
   - Helps onboarding new developers

C. CONTRIBUTING.md (optional)  
   - Test instructions  
   - Coding style  
   - How to add parameters or system variants  

6) API & reference docs (optional / long-term)
----------------------------------------------

Potential later addition under docs/api/:
- Manually curated or auto-generated references for key modules  
- Not needed immediately; can wait until the architecture stabilises further  

Suggested directory layout
--------------------------

README.md  
docs/
  modeling_overview.md  
  architecture/
    params_and_store.md  
    binder_and_graph.md  
    inference_and_loss.md  
    eigenmodes.md  
  tutorials/
    canonical_astrometry_demo.md
  dev/
    dLuxShera_Working_Plan.md
    code_structure.md
  archive/
    REFACTOR_HISTORY.md
    ARCHITECTURE_DECISIONS.md
    LEGACY_APIS_AND_MIGRATION.md
  api/   (optional, future)

Implementation phases
---------------------

Phase 1 — Skeleton & migration  
- Create docs/ tree with empty/skeletal files  
- Move Working Plan into docs/dev/  
- Update README.md to link into docs  

Phase 2 — Core conceptual & architecture docs  
- Fill in modeling_overview.md  
- Fill in params_and_store.md and binder_and_graph.md  
- Draft inference_and_loss.md  

Phase 3 — Tutorials & refinements  
- Write canonical_astrometry_demo.md  
- Flesh out eigenmodes.md  
- Add code_structure.md (optional)

Phase 4 — API/reference (future)  
- Add docs/api/ if desired  
- Consider Sphinx/mkdocs automation later  

This roadmap gives us a structured, navigable documentation ecosystem: a conceptual top layer, a clear architecture layer, tutorial workflows, and well-isolated dev docs—with the Working Plan now appropriately placed under docs/dev/.

- Devtools context snapshot (devtools/generate_context_snapshot.py) now groups ParamSpecs and transforms by system_id, calling out primitive vs derived keys and transform dependencies in the Markdown summary to keep demos aligned with the active registry.

---

## 23) Parking Lot

- Two/Four-plane optics variant design and transforms.
- Extended inference methods (HMC, priors, eigenspace optimization) after core stack stabilizes.
- Ergonomic shims (`ModelParams`) and deprecation strategy for legacy APIs.
- High-level model design / capabilities documentation describing what the Shera-style model does (optical/astrometric forward model, main outputs, supported questions) and its key assumptions/approximations, written for proposal and systems-engineering consumers rather than just implementers.
- Model–error-budget interface and parameter dependency mapping: lightweight docs/figures that show how model outputs and sensitivities map onto specific error-budget terms, and how primitives vs. derived parameters (ParamSpec → Store → transforms) relate to those terms for traceability.

---
