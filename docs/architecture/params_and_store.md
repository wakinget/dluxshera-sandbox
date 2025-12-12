# Parameters, specs, and stores

## ParamSpec
- Declares the parameter vocabulary (primitives and derived) with metadata, defaults, bounds, and shapes.
- Forward/inference builders construct the subset relevant to the optical model; derived fields live alongside primitives but are marked by kind.
- Serves as the authoritative schema so docs, tests, and configs speak a common language.

## ParameterStore
- Frozen mapping that holds numeric values keyed by `ParamKey`; primitives are authoritative and are validated against a `ParamSpec`.
- Derived parameters are computed, not stored manually: build a primitives-only store then call `refresh_derived` to populate derived values.
- Supports packing/unpacking to θ-vectors for optimisation, with helpers to strip or refresh derived fields for consistency.

## DerivedResolver and transforms
- `DerivedResolver` dispatches pure transform functions keyed by `system_id`, so different optical systems can define their own derived rules.
- Transforms consume primitive fields and produce derived values (e.g., focal length ➜ plate scale), enabling reproducible derived computation.
