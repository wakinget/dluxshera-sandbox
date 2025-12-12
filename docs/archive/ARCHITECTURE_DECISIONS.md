# Architecture decisions (ADR-lite)

Short summaries of key architectural choices. See the Working Plan for current workstreams; historical context and rationale live here.

## Choice: ParamSpec/ParameterStore as the primary parameter representation
- **Context:** Parameter definitions, metadata, and values were embedded in optics builders, making validation and reuse difficult.
- **Decision:** Use declarative `ParamSpec` objects plus an immutable `ParameterStore` for values.
- **Rationale:** Separates schema from execution, enables validation, and keeps primitives/deriveds consistent.
- **Tradeoffs:** Adds upfront verbosity; requires discipline to keep specs authoritative instead of ad-hoc dicts.

## Choice: DerivedResolver and transform registry keyed by `system_id`
- **Context:** Derived values (e.g., pixel scales) were computed in multiple places with inconsistent defaults.
- **Decision:** Centralize derivations in a resolver fed by a registry of pure transforms, scoped by `system_id` where needed.
- **Rationale:** Single source of truth for deriveds, improved testability, and clearer provenance.
- **Tradeoffs:** Registry management introduces indirection; requires guardrails to avoid duplicate or conflicting transforms.

## Choice: Binder as the primary model object, backed by a SystemGraph
- **Context:** Execution order and parameter flow were implicit inside optics code, making debugging and extension difficult.
- **Decision:** Make `Binder` the main model façade backed by an explicit `SystemGraph` DAG of nodes.
- **Rationale:** Execution becomes explicit, cacheable, and testable; new systems can reuse nodes and wiring.
- **Tradeoffs:** Additional abstraction layers; SystemGraph tooling must stay lightweight to avoid over-engineering.

## Choice: Aligning with dLux/JAX idioms where possible
- **Context:** The codebase mixes legacy patterns with emerging JAX-friendly practices.
- **Decision:** Prefer dLux-aligned APIs, pytree-friendly data structures, and functional transforms where practical.
- **Rationale:** Improves composability, autodiff performance, and community familiarity.
- **Tradeoffs:** Can increase friction when integrating legacy utilities; requires careful migration to avoid breaking existing demos.
