# Legacy APIs and migration notes (archival)

This document maps legacy APIs to the current ParamSpec/ParameterStore/Binder architecture. It is intentionally rough and should be updated as migrations progress.

**Note:** The SystemGraph scaffold referenced in this document has been removed from the codebase; binder-only evaluation is the current runtime path.

## High-level mapping
- **Legacy model classes:** `SheraThreePlane_Model` → `Binder` (binder-only evaluation for the three-plane system).
- **Legacy optics builders/configs:** imperative optics builders → `optics` and `config` modules driven by `ParamSpec` definitions and derived transforms.
- **Inference helpers:** legacy inference utilities → `InferenceSpec` and associated binders configured via the parameter store.

## Legacy bridge builders

Some helper functions exist solely to bridge refactor-era config/spec/store inputs to legacy
Shera model classes used by older scripts. These are not part of the canonical Binder-based
forward model pipeline.

- `build_legacy_shera_threeplane_model(cfg, spec, store)`:
  Returns a legacy `SheraThreePlane_Model` for compatibility only.

New development should prefer:
- `SheraThreePlaneBinder(...).model(store_delta)`

and rely on optics builder caching for structure reuse.

## Gotchas and semantic changes
- Derived values are resolved by the registry; avoid recomputing them ad-hoc in optics code.
- Parameter validation is stricter; ensure specs include bounds/units where required.
- Execution is graph-driven; cache assumptions from the legacy stack may not hold once nodes are split or reordered.
