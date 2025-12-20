# Binder and SystemGraph

## Binders

The Binder is the primary model object in dLuxShera. It combines a configuration, the relevant `ParamSpec`, and a baseline `ParameterStore`, then exposes user-facing methods to evaluate the optical system. Callers supply parameter deltas or θ-vectors, and the Binder handles packing/unpacking, derived parameter resolution, and interaction with the underlying optics. When you “build a Shera model” in the canonical demos, you are creating a Binder and invoking it to produce PSFs or images.

### Immutability and intended usage

Binders are **operationally (mostly) immutable** by design:

- Treat a Binder instance as a **stable, canonical model**: it captures `(cfg, forward_spec, base_forward_store)` and any internal build products (e.g., a `SystemGraph`) that should remain fixed for the lifetime of that Binder.
- “Changing parameters” during optimization or inference should happen via **inputs** (deltas / θ-overlays), not by mutating the Binder.
- If you need a new baseline store/config, create a **new Binder** from the old one.

This is an API- and workflow-level contract (immutability by convention), not necessarily enforced by Python-level freezing. In practice, we avoid in-place mutation because Binders are commonly captured in JAX-jitted closures and treated as static context.

### Stable vs dynamic data

- **Stable (captured in Binder):**
  - `cfg`: system configuration (geometry, wavelengths, etc.)
  - `forward_spec`: parameter specification / mapping rules
  - `base_forward_store`: baseline `ParameterStore` with derived parameters refreshed
  - internal build products (e.g., `SystemGraph`) used to evaluate the optical system

- **Dynamic (passed per call):**
  - `store_delta`: a `ParameterStore`-like overlay containing updates relative to the baseline
  - θ-vectors / packed parameter vectors that the Binder unpacks into deltas

### Public API expectations

- `.model(store_delta=None, ...)` is the primary entry point for evaluation.
- `.with_store(new_store)` returns a **new Binder** whose baseline store is replaced (and deriveds refreshed).
- If a similar need arises for config changes, prefer a constructor or `with_cfg(...)`-style helper rather than mutating in-place.

    ```python
    binder = SheraThreePlaneBinder(cfg, forward_spec, base_store, use_system_graph=True)

    # baseline evaluation
    psf = binder.model()

    # dynamic behavior comes from overlays (not mutation)
    psf2 = binder.model(delta_store)

    # derive a new binder when you truly need a new baseline
    binder2 = binder.with_store(new_store)
    ```

### Why we avoid in-place mutation

The Binder is frequently used as “static context” for compiled/JIT code. Mutating a Binder that has been captured by a JAX closure can lead to confusing behavior (compiled functions may not reflect changes the way users expect, and reproducibility suffers). The “mostly immutable” pattern (static Binder + dynamic deltas) keeps evaluation predictable and JIT-friendly.

### Implementation notes (current expectations)

As of the current refactor:
- Binder is a **plain Python dataclass** (for inspection/debuggability).
- Binder is **not slotted** (keeps flexibility during active refactor and avoids slot-related edge cases).
- Binder is **not** an `equinox.Module` and **not** a `zodiax.Base`—it is a lightweight wrapper around config/spec/store plus evaluation helpers.
- The baseline store is a `ParameterStore` that supports dict-like introspection (`get`, `keys`, `items`, `as_dict`, …) and includes core derived keys (e.g., `system.plate_scale_as_per_pix`) after construction.

## SystemGraphs
Under the hood, a `SystemGraph` represents the computation as a directed acyclic graph. Nodes correspond to optical elements, intermediate wavefronts or images, and detector steps; edges capture data flow between them. The Binder manages this graph so that typical users do not need to manipulate it directly, but the structure is available for advanced workflows such as inspecting intermediate values or enabling caching.

## How they fit together
ParamSpecs and ParameterStores define what can change. The Binder is the object you call to run the model, taking care of parameter bookkeeping and model evaluation. The SystemGraph is the wiring the Binder orchestrates to generate outputs. Canonical demos interact with the Binder interface and treat the SystemGraph as an implementation detail unless deeper introspection is needed.
