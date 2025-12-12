# Refactor history (archival)

This document captures the history of dLuxShera's transition to the current ParamSpec/ParameterStore/Binder/SystemGraph architecture.
It is meant for future maintainers who want a narrative of how the platform evolved.

## Timeline highlights
- **Early prototypes:** Initial Shera models coupled parameter definitions, derived quantities, and execution order inside optics builders and inference scripts.
- **ParamSpec/ParameterStore design:** Introduced declarative parameter specifications and an immutable store to separate metadata from values and enforce validation.
- **DerivedResolver and transform registry:** Added pure-function transforms to derive secondary quantities, gradually moving logic out of optics builders and into a system-scoped registry.
- **Binder/SystemGraph introduction:** Created Binder as the primary model object with a backing SystemGraph to make execution order explicit and testable.
- **Canonical demo creation:** Built the canonical astrometry demo on top of the new stack to validate the architecture end-to-end.

## Before vs. after (conceptual)
- **Before:** Optics builders owned both parameters and execution; derived values were recomputed in multiple places with limited validation and unclear provenance.
- **After:** ParamSpec and ParameterStore define parameters and values, DerivedResolver handles derivations via a registry, and Binder/SystemGraph orchestrates execution with clear, testable flows.
