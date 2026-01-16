# Systems and binders

## Terminology

- **System**: The user-facing model object that evaluates the optical system.
- **Binder**: The implementation class in code (historical naming) that binds configuration, parameter spec, and stores into a callable System. In code, Systems live in `dluxshera.systems.*` as `Binder` classes.

> **Quick glossary**
> - `cfg`: System configuration (geometry, wavelengths, etc.).
> - `ParamSpec`: Parameter specification / mapping rules.
> - `ParameterStore`: Dictionary-like container for parameter values and deriveds.
> - `store_delta`: Store overlay containing updates relative to a baseline.
> - `structural keys`: Store keys that change optics structure (sampling, topology, shapes).

## Systems (implemented as binders)

### System-level configs and named designs

System configuration now lives at the system level: you instantiate a `SheraTwoPlaneConfig` or `SheraThreePlaneConfig` once, then pass that config into builders/spec generation and the System (Binder). The config is treated as a stable, mostly immutable definition of the optical design and sampling defaults, so new users should think of it as the canonical source of truth for geometry, wavelengths, and structural settings. For common starting points, the codebase exports named designs like `SHERA_TESTBED_CONFIG` and `SHERA_FLIGHT_CONFIG` (both two-plane and three-plane variants) with a `design_name` set for traceability. The recommended workflow is to start from a named design and use `.replace(...)` (or an equivalent helper) to derive a customized config, keeping the original untouched so caching, hashing, and reproducibility stay predictable.

The System (Binder) is the primary model object in dLuxShera. It combines a configuration, the relevant `ParamSpec`, and a baseline `ParameterStore`, then exposes user-facing methods to evaluate the optical system. Callers supply parameter deltas or θ-vectors, and the System handles packing/unpacking, derived parameter resolution, and interaction with the underlying optics. When you “build a Shera model” in the canonical demos, you are creating a System (Binder) and invoking it to produce PSFs or images.

### Immutability + evaluation contract

Systems are **operationally (mostly) immutable** by design:

- Treat a System (Binder) instance as a **stable, canonical model**: it captures `(cfg, forward_spec, base_forward_store)` and any internal build products (e.g., a cached telescope) that should remain fixed for the lifetime of that System.
- “Changing parameters” during optimization or inference should happen via **inputs** (deltas / θ-overlays), not by mutating the System.
- If you need a new baseline store/config, create a **new System** from the old one.

This is an API- and workflow-level contract (immutability by convention), not necessarily enforced by Python-level freezing. In practice, we avoid in-place mutation because Systems are commonly captured in JAX-jitted closures and treated as static context.

### Baseline vs delta stores

- **Baseline (captured in System):**
  - `cfg`: system configuration (geometry, wavelengths, etc.)
  - `forward_spec`: parameter specification / mapping rules
  - `base_forward_store`: baseline `ParameterStore` with derived parameters refreshed
  - internal build products (e.g., cached telescope) used to evaluate the optical system

- **Delta (passed per call):**
  - `store_delta`: a `ParameterStore`-like overlay containing updates relative to the baseline
  - θ-vectors / packed parameter vectors that the System unpacks into deltas

### Structural vs non-structural changes

We distinguish two categories of values:

1) Structural values (affect shapes / sampling / object topology)
   - Changing these requires rebuilding the optics *structure*.
   - These values must be included in the structural hash used for caching.

2) Non-structural values (affect coefficients only)
   - Changing these should reuse the existing optics structure.
   - Examples: Zernike coefficients, small OPD perturbations, flux parameters, etc.

The structural subset is documented alongside each optics config in `src/dluxshera/optics/config.py`.

### Builders and caching boundaries

This project uses “builders” as *pure factories* that turn:
- config (structural knobs),
- ParamSpec (schema), and
- ParameterStore (values; typically primitives-only + refreshed deriveds)

into *runtime objects* used to evaluate the forward model.

#### What is built where

- Optics builders (`src/dluxshera/builders/optics.py`)
  - Build the dLux optical system (“optics stack”) used by the telescope forward model.
  - Own the **structural hash + caching** policy for optics construction.
  - Key idea: optics are expensive to build; coefficients/parameters are cheap to update.

- Universe/source builders (`src/dluxshera/builders/source.py`)
  - Build astrophysical sources (e.g., Alpha Cen) from the effective store.

- Systems (`src/dluxshera/systems/{three_plane.py,two_plane.py}`)
  - Binder is the public entry point: `.model(store_delta)` merges an overlay store and evaluates.

### Public API expectations

- `.model(store_delta=None, ...)` is the primary entry point for evaluation.
- With `store_delta=None`, `.model()` uses the System's persistent telescope for a fast-path evaluation (no optics rebuild).
- `.with_store(new_store)` returns a **new System** whose baseline store is replaced (and deriveds refreshed).
- `.update_store(new_store)` returns a **new System** with a refreshed base store; it is the preferred way to persist new truth stores or apply structural changes (it will rebuild if the structural hash changes).
- If a similar need arises for config changes, prefer a constructor or `with_cfg(...)`-style helper rather than mutating in-place.

```python
binder = SheraThreePlaneBinder(cfg, forward_spec, base_store)

# baseline evaluation
psf = binder.model()

# dynamic behavior comes from overlays (not mutation)
psf2 = binder.model(delta_store)

# derive a new system when you truly need a new baseline
binder2 = binder.with_store(new_store)
```

### Store-delta helpers (recommended usage)

System evaluation expects **non-structural deltas** (only values that can be updated without rebuilding optics). Two helpers in `params.store` make it easy to prepare these deltas:

- `subset_store(store, infer_keys)` trims a full `ParameterStore` down to just the inference keys packed into θ. This is the recommended way to convert `store_unpack_params(...)` outputs into a delta suitable for `binder.model`.
- `strip_structural(store)` (or `strip_structural(store, structural_keys=...)`) removes structural keys like `system.*` / `band.*` when you want to sanitize an overlay before evaluation. For finer control, pass the system's own structural key set (e.g., `binder._structural_store_keys()`) so runtime-bound keys like `system.plate_scale_as_per_pix` remain allowed.

In short: **unpack θ → subset → model**, and reserve structural updates for explicit rebuilds via `binder.model(..., allow_rebuild=True)` or by creating a fresh system with `binder.update_store(...)`.

### Why we avoid in-place mutation

The System (Binder) is frequently used as “static context” for compiled/JIT code. Mutating a System that has been captured by a JAX closure can lead to confusing behavior (compiled functions may not reflect changes the way users expect, and reproducibility suffers). The “mostly immutable” pattern (static System + dynamic deltas) keeps evaluation predictable and JIT-friendly.

## Relationship diagram

```
    cfg + forward_spec + base_forward_store
                 |
                 v
         System (Binder)
        (merge store_delta)
                 |
                 v
          effective store
        /        |        \
       v         v         v
   optics      source    detector
 (cached)      (build)   (build/reuse)
       \         |         /
        \        |        /
                 v
             dl.Telescope
                 |
                 v
              telescope.model()
```

## See also

- [Parameters and store](params_and_store.md)
- [Inference and loss](inference_and_loss.md)
- [Eigenmodes](eigenmodes.md)
