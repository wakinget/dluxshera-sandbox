# Parameters, specs, and stores

## ParamSpecs
`ParamSpec` is the canonical schema for a dLuxShera system. It lists every parameter name, its shape, and whether it is a primitive (supplied by the user) or a derived quantity (computed from primitives). Specs can be grouped by `system_id` so a three-plane Shera setup and a two-plane variant carry clear, versioned vocabularies. Because the spec is the shared contract for configs, docs, and tests, it is the first stop when asking “what parameters exist for this system?”

## ParameterStores
`ParameterStore` instances hold the numeric values for a given `ParamSpec`. Primitive parameters are the source of truth: callers build a store from spec defaults, apply small updates (for example during optimisation), and reuse that immutable snapshot across runs. Derived parameters are normally recomputed from primitives rather than edited directly; helpers keep derived fields refreshed so PSF generation and losses always see a consistent view. Packing and unpacking utilities convert between stores and flat θ-vectors while respecting this primitive-first policy.

## Transforms and derived parameters
Derived values come from pure transforms registered for each `system_id`. The `DerivedResolver` selects the right transform set, resolves dependencies, and updates a `ParameterStore` when derived quantities are requested. Typical workflows either read from a store that has already been resolved (as in the canonical astrometry demo) or call the resolver to compute the needed derived values before constructing the Binder. This keeps derived quantities reproducible and tied to the same primitive sources used throughout the modeling overview.
