# Core architecture foundations

- **Date:** 2025-12-19
- **Status:** Accepted

## Context

The architecture relies on a handful of foundational choices to keep parameter validation, derivation, and execution coherent. These decisions consolidate the refactor-era rationale into a single ADR so the rationale remains visible as the system evolves.

## Decision

### ParamSpec/ParameterStore as the primary parameter representation

- **Context:** Parameter definitions, metadata, and values were embedded in optics builders, making validation and reuse difficult.
- **Decision:** Use declarative `ParamSpec` objects plus an immutable `ParameterStore` for values.
- **Rationale:** Separates schema from execution, enables validation, and keeps primitives/deriveds consistent.
- **Tradeoffs:** Adds upfront verbosity; requires discipline to keep specs authoritative instead of ad-hoc dicts.

### DerivedResolver and transform registry keyed by `system_id`

- **Context:** Derived values (e.g., pixel scales) were computed in multiple places with inconsistent defaults.
- **Decision:** Centralize derivations in a resolver fed by a registry of pure transforms, scoped by `system_id` where needed.
- **Rationale:** Single source of truth for deriveds, improved testability, and clearer provenance.
- **Tradeoffs:** Registry management introduces indirection; requires guardrails to avoid duplicate or conflicting transforms.

### Binder as the primary model object, with direct telescope evaluation

- **Context:** Execution order and parameter flow were implicit inside optics code, making debugging and extension difficult.
- **Decision:** Make `Binder` the main model façade that owns a cached telescope and evaluates via the optics/source builders.
- **Rationale:** Execution remains explicit and testable while avoiding extra graph wiring for the current single-node flow.
- **Tradeoffs:** Foregoes graph-level inspection until a multi-node execution layer is reintroduced.

### Aligning with dLux/JAX idioms where possible

- **Context:** The codebase mixes legacy patterns with emerging JAX-friendly practices.
- **Decision:** Prefer dLux-aligned APIs, pytree-friendly data structures, and functional transforms where practical.
- **Rationale:** Improves composability, autodiff performance, and community familiarity.
- **Tradeoffs:** Can increase friction when integrating legacy utilities; requires careful migration to avoid breaking existing demos.

## Consequences

- Parameter schemas and values remain synchronized through explicit specs and immutable stores.
- Derived values follow a single registry-driven resolver, reducing drift across components.
- Binder provides an explicit, testable execution path with room for reuse across systems.
- Adopting dLux/JAX patterns prioritizes composability and performance while requiring mindful migrations from legacy utilities.

## Links

- `docs/archive/REFACTOR_HISTORY.md`
