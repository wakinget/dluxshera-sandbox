# Parameters, specs, and stores

## ParamSpecs
`ParamSpec` is the canonical schema for a dLuxShera system. It lists every parameter name, its shape, and whether it is a primitive (supplied by the user) or a derived quantity (computed from primitives). Specs can be grouped by `system_id` so a three-plane Shera setup and a two-plane variant carry clear, versioned vocabularies. Because the spec is the shared contract for configs, docs, and tests, it is the first stop when asking “what parameters exist for this system?”

### Forward specs
A forward spec describes the full generative model vocabulary for a system/config. It includes both primitive keys (user supplied) and derived keys (computed by transforms). System-specific builders like `build_forward_spec_from_config(cfg)` in the two-plane and three-plane implementations create the appropriate forward spec for a given config, including system-dependent shapes and defaults.

### Inference specs
An inference spec is the subset/view of parameters that participate in θ-space. Use `make_inference_subspec(*, base_spec, infer_keys, ...)` to declare which keys are being inferred, or start from the baseline Shera vocabulary via `build_inference_spec_basic(...)`. Inference specs are about packing/unpacking and optimization, not about describing the full forward model. This is why they can be narrower than the forward spec while still being valid for parameter packing.

## ParameterStores
`ParameterStore` instances hold the numeric values for a given `ParamSpec`. Primitive parameters are the source of truth: callers build a store from spec defaults, apply small updates (for example during optimisation), and reuse that immutable snapshot across runs. Derived parameters are normally recomputed from primitives rather than edited directly; helpers keep derived fields refreshed so PSF generation and losses always see a consistent view. Packing and unpacking utilities convert between stores and flat θ-vectors while respecting this primitive-first policy.

### Store basics
- **Create a store:** `ParameterStore.from_spec_defaults(spec)` builds a primitives-only store with default values.
- **Update values:** `store.replace({"system.m1_focal_length_m": ...})` (and keyword updates) returns a new store with updated primitives.
- **Validate:** `store.validate_against(spec, ...)` can enforce expected shapes, dtypes, and derived/primitive expectations.
- **Manage derived values:**
  - `store.refresh_derived(spec, system_id=..., include_derived=True)` recomputes derived fields from primitives.
  - `strip_derived(store, spec)` removes derived keys when you want to force a recompute.
  - `check_consistency(...)` helps debug missing keys, mismatched shapes, or inconsistent derived values.
- **Ergonomics:** `StoreNamespace` enables dotted access when appropriate, and `as_dict()` exports a flat mapping for logging or serialization.

Typical flow (forward spec + derived refresh):
  cfg = ...
  forward_spec = build_forward_spec_from_config(cfg)
  base_forward_store = ParameterStore.from_spec_defaults(forward_spec)
  base_forward_store = base_forward_store.replace({"system.m1_focal_length_m": ...})
  base_forward_store = base_forward_store.refresh_derived(forward_spec, system_id=forward_spec.system_id)

## Transforms and derived parameters
Derived values are computed, reproducible quantities that can always be reconstructed from primitives (and other derived values). They are dependency-tracked and typically not edited directly; the normal path is to update primitives and refresh derived values via the registry.

A few key components make this work:
- **`TransformRegistry`:** maps parameter keys to transform functions and their dependencies.
- **`DerivedResolver`:** holds per-system registries and resolves derived values; it supports lazy registration/import for system-specific transforms.
- **Dependency resolution:** the resolver walks the dependency graph, detects missing inputs, and protects against cycles or recursion depth issues. When a dependency is missing or a cycle is detected, refresh will fail with an informative error.

### `transform_registry` and system IDs
Transform registration is keyed by `system_id`, so Shera can keep separate transform sets for `shera_threeplane` vs `shera_twoplane`. When you call `refresh_derived(spec, system_id=...)`, the resolver uses that system ID to select the correct registry. This keeps derived behavior consistent with the forward spec that declared the keys.

### Concrete examples
- `system.focal_length_m` depends on `system.m1_focal_length_m`, `system.m2_focal_length_m`, `system.m1_m2_separation_m`.
- `system.plate_scale_as_per_pix` depends on `system.focal_length_m`, `system.pixel_pitch_m`.
- `binary.raw_fluxes` depends on `binary.log_flux_total`, `binary.contrast`.

## How to add a new transform (developer checklist)
1) Add the derived key to the relevant **forward spec** (as a derived `ParamField`).
2) Implement the transform function (pure function of dependencies via the context).
3) Register the transform under the correct `system_id` via `register_transform(key, depends_on=[...], system_id="...")` (or the helper used in that module).
4) Ensure the transform module is imported/registered (lazy registration should handle this, but add imports if needed).
5) Add tests:
   - `tests/params/test_params_transforms.py` for dependency + compute correctness.
   - Optional system-level expectations under `tests/optics/` when appropriate.

Template shape (conceptual):
  @register_transform("my.derived_key", depends_on=("a", "b"), system_id="shera_threeplane")
  def transform_my_derived_key(ctx):
      a = ctx["a"]
      b = ctx["b"]
      return ...

## Packing θ-vectors (for inference)
Use `pack_params(spec_subset, store)` to flatten an inference spec subset into θ, and `unpack_params(spec_subset, theta, base_store)` to rebuild a store from θ and a base store. The subset is typically an inference spec or subspec, so packing stays aligned with the optimizer’s view. `IndexMap` / `build_index_map(...)` provide a stable layout and labeling for artifacts, plots, and debugging.

## Where in the code?
- `src/dluxshera/params/spec.py`
  - `ParamField`, `ParamSpec`
  - `make_inference_subspec(...)`
  - `build_inference_spec_basic(...)`
- `src/dluxshera/systems/three_plane.py` and `src/dluxshera/systems/two_plane.py`
  - `build_forward_spec_from_config(cfg)`
- `src/dluxshera/params/store.py`
  - `ParameterStore` (`from_spec_defaults`, `replace`, `refresh_derived`, `validate_against`)
  - store helpers like `strip_derived`, `subset_store`, `check_consistency`
- `src/dluxshera/params/transform_registry.py`
  - `TransformRegistry`, `DerivedResolver`
  - `register_transform(...)`, `resolve_derived(...)`, `ensure_registered(...)`
- `src/dluxshera/params/transforms.py`
  - system-registered transforms (e.g. `optics.focal_length_m`, `optics.plate_scale_as_per_pix`, `source.raw_fluxes`)
- `src/dluxshera/params/packing.py`
  - `pack_params`, `unpack_params`, `build_index_map`

## Worked example / reference implementation
For a runnable, canonical example of building specs, creating a store, and refreshing derived values, see `examples/recipes/canonical_astrometry.py`.

## Gotchas / FAQ
- “I tried to set a derived key directly; why did it change later?” Derived values are recomputed by `refresh_derived`, so direct edits are overwritten.
  More precisely, “store wins” is only true for keys treated as primitives in the active spec passed to `refresh_derived`. If a key is marked derived in that spec, refresh recomputes and overwrites it; to keep an explicit value, re-apply it after refresh. For example, setting only `source.exposure_time_s` updates derived `source.log_flux_total`, setting only `source.log_flux_total` uses that explicit value, and setting both still resolves to explicit `source.log_flux_total` when it is re-applied after refresh.
- “Why does my transform fail?” Check for missing primitives, incorrect dependency lists, wrong `system_id`, or dependency cycles.
- “When do I refresh derived?” Typically right after store updates and before building binders or computing losses.

## See also
- `docs/architecture/inference_and_loss.md`
- `docs/architecture/eigenmodes.md`
- `docs/architecture/optimization_artifacts_and_plotting.md`
