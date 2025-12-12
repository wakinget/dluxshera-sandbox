# Legacy APIs and migration notes (archival)

This document maps legacy APIs to the current ParamSpec/ParameterStore/Binder/SystemGraph architecture. It is intentionally rough and should be updated as migrations progress.

## High-level mapping
- **Legacy model classes:** `SheraThreePlane_Model` → `Binder` wrapping a `SystemGraph` node (`DLuxSystemNode`) for the three-plane system.
- **Legacy optics builders/configs:** imperative optics builders → `optics` and `config` modules driven by `ParamSpec` definitions and derived transforms.
- **Inference helpers:** legacy inference utilities → `InferenceSpec` and associated binders configured via the parameter store.

## Gotchas and semantic changes
- Derived values are resolved by the registry; avoid recomputing them ad-hoc in optics code.
- Parameter validation is stricter; ensure specs include bounds/units where required.
- Execution is graph-driven; cache assumptions from the legacy stack may not hold once nodes are split or reordered.
