# dLuxShera Refactor — Working Plan & Notes
_Last updated: 2025-11-13 01:13_

This is a living document summarizing the goals, architecture, decisions, tasks, and gotchas for the dLuxShera parameterization refactor. It’s designed so either of us can get back up to speed quickly and not lose the important details.

---

## 1) Context & Problem Statement

- **Entanglement:** Primitives (e.g., `m1.focal_length`, `system.plane_separation`) and derived values (e.g., `imaging.psf_pixel_scale`) are computed in multiple places → unclear source of truth.
- **Bugs:** After optimization, `psf_pixel_scale` can be missing/incorrect in extraction; init at exact zero can lead to zero gradients.
- **Scaling pain:** Adding parameters/nodes couples logic into optics code; docs/tests lack a single schema for units/bounds/priors.

**Goal:** Cleanly separate *what parameters exist*, *where values live*, *how things are derived*, and *how the system executes*—while keeping a stable facade for users and examples.

**Target outcome (Done):**
- Consistent `psf_pixel_scale` (and other deriveds) regardless of whether they are optimized directly or computed from primitives.
- A clear primitives↔derived boundary and testable pure transforms.
- A structured graph that executes optics cleanly (JAX-friendly).
- Minimal churn to current examples; future models (e.g., four-plane) slot in.

---

## 2) Architecture (High-Level)

**Layers**
1. **ParamSpec** — declarative schema & metadata (no numbers).  
   Defines keys, units, bounds, priors, defaults, kind _(primitive/derived)_.

2. **ParameterStore** — immutable values map for **primitives only** (JAX pytree).  
   Pure `.replace(**overrides)` returns a new store.

3. **DerivedResolver (Scoped Transform Registry)** — computes derived values as pure functions of primitives, **scoped to the active system** (e.g., `three_plane` vs `four_plane`).

4. **SystemGraph** — an executable DAG of nodes; each node is a pure function that takes the store + required deriveds and produces outputs. A single node calls into the dLux **`ThreePlaneOpticalSystem.model()`** for multi-λ PSF generation.

5. **Facade** — `SheraThreePlane_Model` wrapper stays as the public entry point; internally uses the new stack.

### ASCII sketch
```
ParamSpec  ──(validates)──▶ ParameterStore (primitives)
     │                               │
     │                          (inputs)
     ▼                               ▼
Scoped Transform Registry ──▶ DerivedResolver (system_id-aware) ──▶ derived dict
     │                                                             │
     └─────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                         SystemGraph.forward()
                            │       ▲
        [Builder] ThreePlaneOpticalSystem  [Binder] (set numeric arrays, derived sampling)
                            │
                            ▼
                  .model(wavelengths, weights)  →  PSF
```

---

## 3) Repository Layout (Proposed)

```
dLuxShera/
├─ pyproject.toml
├─ README.md
├─ mkdocs.yml
├─ docs/
│  ├─ getting-started/
│  ├─ concepts/
│  │   ├─ param-store.md
│  │   ├─ system-graph.md
│  │   └─ three-plane.md
│  ├─ how-to/
│  │   ├─ run-fit.md
│  │   ├─ eigen-truncation.md
│  │   └─ hmc.md
│  └─ reference/
├─ src/
│  └─ dluxshera/
│     ├─ __init__.py
│     ├─ core/{types.py, utils.py}
│     ├─ params/{spec.py, store.py, transforms.py, registry.py, serialize.py}
│     ├─ graph/{nodes.py, system_graph.py, builders.py}
│     ├─ optics/{propagation/{fresnel.py, aberrations.py}, optical_systems.py, psf.py}
│     ├─ inference/{losses.py, priors.py, eigenbasis.py, optimizers.py, hmc.py}
│     ├─ io/{config.py, datasets.py, save_load.py}
│     ├─ viz/plots.py
│     └─ legacy/{shera_threeplane.py, keymap.py, proxy.py}
├─ examples/{notebooks/, configs/, README.md}
├─ scripts/{run_fit.py, generate_synth_data.py}
└─ tests/
```

**Principles:**
- Small, testable modules; pure functions.
- JAX-friendly (immutable dataclasses, pytrees).
- Docs via `mkdocstrings` and NumPy-style docstrings.

---

## 4) ParamSpec (Schema & Metadata)

Each parameter is a `ParamField` with:
- `key`: e.g., `m1.focal_length`
- `kind`: `"primitive"` or `"derived"`
- `units`, `dtype`, `shape`
- `doc`: human-friendly description; optional LaTeX label
- `bounds`: `(lo, hi)` or validator
- `prior`: optional
- `default`: concrete value, factory, or **`NO_DEFAULT`**
- `transform`: optimization bijector (log/softplus), optional
- `depends_on`: _(optional)_ only for documentation; actual dependency is in the transform registry

**Defaults handling**
- Safe global default → set in `default`.
- Instrument/lab default → put in **profile** YAML (see §8).
- Unknown by design → `NO_DEFAULT` (error if not provided).

---

## 5) ParameterStore (Values, Primitives Only)

- Frozen mapping `{key → value}` for **primitives**.
- JAX pytree; use small arrays (jnp scalars/arrays).  
- Methods:
  - `.get(key)`
  - `.replace(**overrides) -> ParameterStore` (returns new store)
  - `.validate(spec)` for shape/dtype/bounds

**Why primitives-only?** Prevents drift; deriveds are always recomputed via pure transforms.

---

## 6) Scoped Transform Registry (DerivedResolver)

We keep the **meaning** of a derived key stable (e.g., `imaging.psf_pixel_scale`), while the **implementation** depends on `system_id`.

```python
@dataclass(frozen=True)
class TransformSpec:
    key: str
    deps: tuple[str, ...]
    fn: Callable[[dict[str, Any]], Any]  # env -> value (pure)
```

**Three- vs Four-plane example**
```python
register_transform("three_plane",
  TransformSpec(
    key="imaging.psf_pixel_scale",
    deps=("m1.focal_length","system.plane_separation","detector.pixel_pitch"),
    fn=lambda env: compute_psf_scale_three(env["m1.focal_length"],
                                           env["system.plane_separation"],
                                           env["detector.pixel_pitch"])
))

register_transform("four_plane",
  TransformSpec(
    key="imaging.psf_pixel_scale",
    deps=("relay.eff_focal","detector.pixel_pitch"),
    fn=lambda env: compute_psf_scale_four(env["relay.eff_focal"],
                                          env["detector.pixel_pitch"])
))
```

**Resolver**
```python
class DerivedResolver:
    def compute(self, system_id: str, store, keys):
        reg = transforms_registry.get(system_id, {})
        out = {}
        for k in keys:
            ts = reg.get(k)
            if ts is None: raise KeyError(f"No transform for {k} in {system_id}")
            env = {d: store.get(d) for d in ts.deps}
            miss = [d for d,v in env.items() if v is None]
            if miss: raise ValueError(f"Missing primitives {miss} for {k}")
            out[k] = ts.fn(env)
        return out
```

**Policy on setting deriveds**
- **Default:** disallow setting deriveds; update the primitives instead.
- **If invertible & justified:** offer explicit alias setter (opt-in).

---

## 7) Integrating dLux `ThreePlaneOpticalSystem`

Treat dLux as the **engine**; we build/bind around it.

**Split parameters:**
- **Structural:** changing requires **rebuild** (e.g., `detector.n_pix`, geometry choices).
- **Numeric:** changing requires **rebind** only (e.g., Zernikes, small misalignments).

### Builder (cached on structural keys)
```python
class ThreePlaneBuilder:
    def get_or_build(self, store: ParameterStore) -> ThreePlaneOpticalSystem:
        # hash structural subset; cache OS per structure
        # construct ThreePlaneOpticalSystem with structural kwargs
        return os
```

### Binder (pure array updates using store + deriveds)
```python
class ThreePlaneBinder:
    def bind(self, os, store, system_id="three_plane"):
        needed = { "imaging.psf_pixel_scale", "m1.phase_map", "m2.phase_map" }
        derived = resolver.compute(system_id, store, needed)
        # Map primitives/derived -> layer arrays; return new OS/layers
        return os_updated
```

### Graph node
```python
class DLuxSystemNode:
    def __call__(self, store):
        os = builder.get_or_build(store)
        osb = binder.bind(os, store)
        return osb.model(wavelengths=wgrid, weights=ww)  # multi-λ handled by dLux
```

**No upstream (dLux) changes required now.** Helpers like `os.replace(layers=...)` can be added later if desired.

---

## 8) Config, Defaults & Profiles

**Load order (left overrides right):**  
`CLI overrides` → `run config` → `profile` → `spec defaults`.

- **Profiles**: `profiles/toliman_A.yaml`, `profiles/lab_bench.yaml` define instrument/site defaults.
- **NO_DEFAULT** parameters error early with a clear message listing missing keys and which node/transform needs them.
- Example required-but-profiled: `source.wavelengths` may be required in examples but not globally defaulted in the spec.

---

## 9) Backwards Compatibility (Legacy Ergonomics)

- `ParamsProxy` — dict-like view of store + resolver: read derived computes on the fly; write derived disallowed.
- `ModelParams` — wrapper exposing `.replace(...)`, `extract_params()` and `__getitem__/__setitem__` via proxy.
- `LegacyKeyMap` — maps old names → canonical keys, with optional deprecation warnings.

Keep the public facade `SheraThreePlane_Model` calling into the graph. Deprecate legacy names gradually.

---

## 10) Examples & Docs (Minimum Lovable)

- **Notebooks**:  
  `00_env_check` • `01_quickstart_model` • `02_make_synth_dataset` • `03_inference_grad` • `04_eigen_truncation` • `05_plate_scale_vs_primitives` • `06_multi_init_local_bias` • `(07_hmc optional)`
- **Scripts**: `generate_synth_data.py`, `run_fit.py` (toy/real switch).
- **Docs**: Concepts pages for ParamSpec/Store/Graph/Transforms; How‑tos for adding parameters, registering nodes, switching systems; API via mkdocstrings.

---

## 11) Testing & CI

**Unit/props**
- Spec lint: no cycles; all deriveds registered per system; bounds/unit sanity.
- Transforms: finite-diff/analytic checks; unit scaling.
- Store: validation; replace semantics; pytree roundtrip.

**Graph & optics**
- Tiny 16×16 forward PSF; JIT/grad smoke; shape/norm checks.
- Selection tests for scoped transforms (three- vs four-plane).

**Inference**
- Gradient-based fit reduces loss on synthetic data (fast).

**Notebooks (smoke)**
- Run 01 & 03 via nbconvert/papermill with short runtime.

**Performance**
- Builder cache hit on structural hash; binder avoids rebuilds.
- Document first-call JIT latency vs subsequent calls.

---

## 12) Performance & JAX Gotchas

- Immutable dataclasses; explicit pytrees.
- Keep structural choices static inside jitted fns.
- Avoid Python side effects; return arrays only.
- Consistent dtypes (float32/float64).
- Wavelength vectors as arrays (avoid static_argnames if possible).
- Minimize device↔host transfers in loops.

---

## 13) Migration Plan & Milestones

**Phase 0 (now)**
- Builder, binder, graph node; route `SheraThreePlane_Model` through them.
- Seed `ParamSpec` & a few key params.

**Phase 1**
- Introduce `ParameterStore`; examples use it via `ModelParams` shim.
- Add scoped transform registry; centralize `psf_pixel_scale`.

**Phase 2**
- Add `four_plane` builder + transforms.
- `system_id` switch; end-to-end smoke tests.

**Phase 3 (optional)**
- Upstream small dLux ergonomics; deprecate legacy names by default.

---

## 14) Open Questions

- Allow alias setter for invertible deriveds? (default no)
- Need transform predicates beyond `system_id`?
- Minimal structural subset for builder hash?

---

## 15) Tasks & Priorities

**P0 — Core plumbing**
- [ ] ParamSpec core keys
- [ ] ParameterStore (frozen, pytree, validate)
- [ ] Transforms registry + resolver; `psf_pixel_scale` (three-plane)
- [ ] ThreePlaneBuilder (structural hash/cache)
- [ ] ThreePlaneBinder (phase/sampling bind)
- [ ] DLuxSystemNode; wire SheraThreePlane_Model
- [ ] Unit tests: spec/store/transforms/node

**P1 — Docs & examples**
- [ ] README quickstart
- [ ] MkDocs concepts (spec/store/graph/transforms)
- [ ] Notebooks 01 & 03 runnable
- [ ] API docstrings

**P2 — Variant support**
- [ ] Four-plane transforms & builder/binder
- [ ] Resolver selection tests
- [ ] End-to-end smoke test

**P3 — Ergonomics**
- [ ] ModelParams shim + proxy + keymap
- [ ] Deprecation warnings
- [ ] Optional upstream PRs

---

## 16) Conventions

- Keys: `element.group.name` (e.g., `m2.zernike.j4`)
- Units in spec; validated in transforms
- NumPy-style docstrings
- snake_case modules; CamelCase classes

---

## 17) Quickstart (Dev)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
python scripts/run_fit.py --steps 200 --lr 5e-2 --seed 0
mkdocs serve
```

**Git**
```bash
git checkout -b refactor/params-graph
git add -p
git commit -m "refactor: ParamSpec/Store + three-plane transforms + builder/binder"
git push -u origin refactor/params-graph
```

---

## 18) Appendix — Checklists

**Add a new primitive**
- [ ] Spec entry (units/bounds/prior/default or `NO_DEFAULT`)
- [ ] Profile defaults if instrument-specific
- [ ] Use in transforms/nodes
- [ ] Tests + docs

**Add/modify a derived**
- [ ] Register transform per `system_id` (deps + fn)
- [ ] Tests: numerical sanity, missing-deps errors, selection
- [ ] Docs: definition & interpretation

**Add a new system (e.g., four-plane)**
- [ ] New builder/binder
- [ ] Register variant transforms
- [ ] End-to-end smoke test; examples with `system_id`

---

## 19) Decision Log (living table)

> Record key decisions succinctly so future you (or collaborators) can see what changed and why.

| Date       | Decision                                                                 | Rationale                                                                                                                                  | Status   | Owner | Links |
|------------|---------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|----------|-------|-------|
| 2025-11-13 | Use **scoped transform registry** for derived keys (per `system_id`).        | Same user-facing meaning (e.g., `imaging.psf_pixel_scale`) with system-specific math; keeps API stable and internals swappable.            | AGREED   | DMK   |  |
| 2025-11-13 | Adopt **Builder (structural) + Binder (numeric)** around `ThreePlaneOpticalSystem`. | Avoid upstream churn in dLux; rebuild only when structure changes; rebind for numeric updates each step.                                   | AGREED   | DMK   |  |
| 2025-11-13 | Keep `SheraThreePlane_Model` as stable facade; route through `SystemGraph`.   | Zero breakage for users; internal flexibility for graph/refactor; easy example continuity.                                                 | AGREED   | DMK   |  |
| 2025-11-13 | **Disallow writing derived params** directly (alias setter only if invertible). | Prevents drift & ambiguity; primitives remain single source of truth; explicit alias only when reversible mapping is defined.              | AGREED   | DMK   |  |
| 2025-11-13 | `ParameterStore` holds **primitives only**; recompute derived via transforms. | Ensures consistency; simplifies autograd & validation; fewer stale caches.                                                                | AGREED   | DMK   |  |
| 2025-11-13 | Defaults: spec if safe; otherwise **profiles**; else `NO_DEFAULT`.           | Clear precedence; early, actionable errors; instrument/site separation.                                                                   | AGREED   | DMK   |  |
| 2025-11-13 | Preserve legacy ergonomics via **ParamsProxy/ModelParams** shim.             | Smooth migration; deprecations later without blocking users.                                                                               | AGREED   | DMK   |  |
| 2025-11-13 | Add **four-plane** support via new builder/binder + transforms.              | Scale architecture to new systems while reusing graph and API.                                                                             | PLANNED  | DMK   |  |
| 2025-11-21 | Created `refactor/params-graph` branch for parameter refactor work. | Keep refactor changes isolated from stable main; enables incremental PRs. | IN PROGRESS | DMK |  |
| 2025-11-21 | Reorganized project directory into new src/dluxshera/ hierarchy per refactor plan.         | Establish a clean package structure before implementing ParamSpec/ParameterStore; improves clarity and maintainability. | Completed   | DMK    | Moved oneoverf.py to utils/ instead of core/. |
| 2025-11-21 | Introduce ParamField/ParamSpec schema, root tests/ layout, and pyproject/pytest setup for src-based package. | Establishes a declarative parameter schema and testing harness before wiring into SheraThreePlane; makes refactor safer/tested. | COMPLETED | DMK   |       |
| 2025-11-21 | ParameterStore.replace() uses a mapping for updates (not kwargs) for hierarchical keys with dots.      | Param keys are canonical string IDs like "binary.separation_mas"; Python kwargs cannot represent these safely. Mapping-based updates avoid subtle key mismatches. | COMPLETED | DMK   | kwargs allowed only for simple identifier-like keys. |

---

## 20) AI Collaborator — Operating Instructions

When asking an AI assistant to write code or docs for this repo, include these constraints so outputs are plug-and-play:

- **Purity & JAX:** Prefer pure functions, immutable dataclasses, and explicit pytree registration. No global state or side effects inside jitted paths.
- **Structural vs numeric:** Identify which parameters are *structural* (trigger rebuild) vs *numeric* (rebind only). Keep structural choices outside of `jit` or pass as static args.
- **Transforms:** Register derived parameters via the **scoped transform registry**. Do not hardcode derived math inside nodes or optics.
- **Docstrings:** Use **NumPy-style** docstrings and include units, shapes, and parameter semantics; keep public APIs documented for mkdocstrings.
- **Tests first:** For every new primitive/derived/node, provide a tiny unit test (16×16 grids) and a gradient sanity check when applicable.
- **Naming & keys:** Use canonical keys like `element.group.name` (e.g., `m2.zernike.j4`). Avoid alternate spellings; add to `LegacyKeyMap` only when required.
- **Examples:** Provide a runnable snippet (≤30s) and, if heavy, note how to toggle small sizes for CI.
- **Performance:** Avoid unnecessary host-device transfers; batch wavelengths as arrays; note first-call JIT cost.
- **Safety rails:** Never write large data to git; put scratch data under `examples/data/` (git-ignored).

### Prompt templates (copy/paste)

- **Add a new primitive**  
  "Add a primitive `{key}` with units `{units}` and default `{default}`. Update `ParamSpec`, add validation to `store.validate()`, and include a unit test verifying bounds and dtype."

- **Register a derived transform**  
  "Register derived `{derived_key}` for `system_id='{sys}'` with deps `{deps}` and provide `compute_{short}`. Add tests for (1) numerical sanity and (2) missing deps error messaging."

- **Create a SystemGraph node**  
  "Create a node named `{name}` that consumes `{inputs}` and derived keys `{deriveds}`, and returns `{outputs}`. Include a topo-order test and a JIT smoke test."

- **Wire builder/binder**  
  "Implement `ThreePlaneBuilder` with structural hash `{struct_keys}` and `ThreePlaneBinder` mapping store+derived to layer arrays. Add a forward test that calls `.model()` on a 32×32 grid."

- **Write a gradient sanity test**  
  "For parameter `{key}`, compare autodiff gradient vs central finite difference on a minimal 16×16 PSF loss; assert relative error < 5e-2."

---

## 21) Glossary (Canonical Vocabulary)

- **Primitive:** A parameter stored in `ParameterStore` (e.g., `m1.focal_length`). Source of truth for values.
- **Derived:** A parameter computed from primitives via the transform registry (e.g., `imaging.psf_pixel_scale`).
- **Structural:** Primitive whose change requires rebuilding the optical system (e.g., `detector.n_pix`).
- **Numeric:** Primitive that only affects numeric arrays (e.g., Zernike coefficients) and can be rebound.
- **SystemGraph:** DAG orchestrating node execution; calls into dLux via a node.
- **Builder/Binder:** Builder constructs/caches a dLux `OpticalSystem` from structural primitives; Binder injects numeric/derived arrays each iteration.
- **Facade:** Public class (e.g., `SheraThreePlane_Model`) offering a stable API while delegating to the new stack.

---

## 22) PR & Issue Templates

**PR title**: `refactor(params): introduce ParameterStore & psf_pixel_scale transform (three-plane)`

**PR checklist**
- [ ] New/changed params documented in `ParamSpec` with units and defaults/profiles.
- [ ] Transform registered per `system_id` with tests.
- [ ] Builder/binder updated; structural hash tested.
- [ ] Examples/notebooks still run in <2 minutes.
- [ ] Added/updated docstrings and mkdocs pages.
- [ ] Deprecation warnings (if any) included.

**Issue template** (bug)
- **Observed**: …  
- **Expected**: …  
- **Repro**: script/notebook + seed  
- **Suspected area**: spec/store/transforms/graph/optics  
- **Artifacts**: small `.npz` or console logs

---

## 23) Troubleshooting Runbook

- **Zero gradients at init**: Ensure non-zero tiny perturbations in init; verify transform bijectors are well-conditioned (avoid hard clamps inside jit).
- **Shape mismatch**: Confirm consistent `[H, W]` across nodes; check structural hash rebuilds after changing `n_pix`.
- **NaNs in PSF**: Validate units and bounds; check that phase maps are finite and sampling (`psf_pixel_scale`) is > 0.
- **JIT recompiles frequently**: A structural arg is not static; move it outside jit or mark `static_argnames`.
- **Slow first call**: Expected JIT; subsequent calls should be fast. If not, inspect control flow for Python branches.

---

## 24) Code Templates (snippets)

**Transform registration (three-plane)**
```python
register_transform("three_plane",
  TransformSpec(
    key="imaging.psf_pixel_scale",
    deps=("m1.focal_length","system.plane_separation","detector.pixel_pitch"),
    fn=compute_psf_scale_three,
  )
)
```

**System node skeleton**
```python
@dataclass(frozen=True)
class DLuxSystemNode:
    builder: ThreePlaneBuilder
    binder: ThreePlaneBinder
    wavelengths: jnp.ndarray
    weights: jnp.ndarray | None = None

    def __call__(self, store: ParameterStore) -> jnp.ndarray:
        os = self.builder.get_or_build(store)
        os_bound = self.binder.bind(os, store)
        return os_bound.model(wavelengths=self.wavelengths, weights=self.weights)
```

**Gradient test**
```python
def test_grad_psf_pixel_scale():
    store = make_minimal_store()
    os = builder.get_or_build(store)
    def loss(scale):
        st = store.replace(**{"imaging.psf_pixel_scale": scale})  # if using alias setter; else adjust primitives
        psf = os.model(wavelengths=W)
        return jnp.sum((psf - target)**2)
    g = jax.grad(loss)(1.0)
    g_fd = finite_diff(loss, 1.0)
    assert jnp.allclose(g, g_fd, rtol=5e-2, atol=5e-4)
```

---

## 25) Performance Budget & Data Policy

- **PSF size**: default examples use ≤ 64×64; CI tests ≤ 32×32.
- **Runtime**: single example notebook cell ≤ 20s on CPU; entire quickstart ≤ 2 minutes.
- **Data**: commit only tiny `.npz` fixtures (<100 KB). Place larger data in `examples/data/` (git-ignored).

---

## 26) Versioning & Deprecation

- Semantic-ish versioning via `__version__` in `dluxshera`.
- Deprecate legacy names with warnings for one minor release before removal.
- Maintain a `CHANGELOG.md` summarizing user-facing changes.
