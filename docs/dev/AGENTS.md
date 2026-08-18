# AGENTS.md — Developer Guidepost (docs/dev)

This file orients contributors (including coding agents) to **developer-facing**
documentation and “golden path” workflows for the dLuxShera codebase.
It is intentionally brief and points to canonical sources.

## Start here (map + orientation)
- **Docs map:** `docs/README.md` — high-level index of architecture, dev docs,
  tutorials, and archive references.
- **Code layout overview:** `docs/dev/code_structure.md`

## Quick commands (common agent workflows)
From repository root:

```bash
# Install editable + dev extras (canonical path)
python -m pip install -e ".[dev]"

# On systems where `python` is unavailable, use `python3`.
# Compatibility shim still available:
# python -m pip install -r requirements-dev.txt

# Run tests
pytest

# Run a focused subset (examples)
pytest tests/params -q
pytest tests/binder -q
pytest tests/inference -q
```

Import-hygiene checks (if relevant to the task):

```bash
# Repo-specific devtools that enforce import boundaries
python devtools/check_no_src_imports.py
python devtools/check_no_examples_imports.py
```

## Optional: generate a context snapshot (not committed)
If a task benefits from a full “repo index” (tree + symbols + doc summaries),
generate a snapshot locally and attach/use it as task context.

```bash
# Generates devtools/context_snapshot_<timestamp>/
python devtools/generate_context_snapshot.py
```

The output directory typically contains:
- `project_tree.txt` (filesystem tree)
- `project_index.json` (symbol/index metadata)
- `context_snapshot.md` / `.json` (summarized developer context)

Snapshots are **not** committed by default; treat them as disposable artifacts.

## Dev-facing references (this directory)
Use these as your primary working set:
- **Working plan:** `docs/dev/working_plan.md`  
  Living near/medium-term plan with architecture notes, decisions, and priorities.
- **Roadmap:** `docs/dev/roadmap.md`  
  Long-horizon themes and priorities (non-binding).
- **Code structure:** `docs/dev/code_structure.md`  
  Overview of package layout and key entry points.
- **Style guide:** `docs/dev/style_guide.md`  
  Naming, typing, docstrings, immutability, and API conventions.
- **Lessons learned:** `docs/dev/lessons_learned.md`  
  Practical gotchas (tooling pitfalls, common failure modes).
- **Runtime plate scale semantics:** `docs/dev/plate_scale_runtime.md`  
  Guidance on plate-scale behavior with runtime bindings and cached optics.

## Architecture deep dives
For conceptual and design details, consult:
- `docs/architecture/binder_and_graph.md`
- `docs/architecture/params_and_store.md`
- `docs/architecture/inference_and_loss.md`
- `docs/architecture/optimization_artifacts_and_plotting.md`
- ADRs in `docs/architecture/adr/`

## Tutorials and examples
Hands-on references for intended usage:
- `docs/tutorials/modeling_overview.md`
- `docs/tutorials/canonical_astrometry_demo.md`
- `examples/` (scripts + notebooks for runnable reference patterns)

## Repo scope and “where to make changes”
- **Library / API code:** `src/dluxshera/`
- **Tests:** `tests/` (prefer adding/updating tests with any behavior change)
- **Docs:** `docs/` (update when public APIs or expected workflows change)
- **Examples:** `examples/` (should demonstrate intended usage; keep runnable)
- **Legacy / scratch:** `work/` and `src/dluxshera/legacy/`  
  Modify only when explicitly requested or when migrating functionality.

Rule of thumb routing:
- Parameter specs / store / transforms → `src/dluxshera/params/` + `tests/params/`
- Binder / graph semantics → `src/dluxshera/core/` + `tests/binder/`
- Systems / optical configs → `src/dluxshera/systems/`, `src/dluxshera/optics/` + `tests/optics/`
- Inference / optimization / priors / artifacts → `src/dluxshera/inference/` + `tests/inference/`
- Plotting helpers → `src/dluxshera/plot/` + `tests/plotting/`

## Definition of Done (for agent-completed tasks)
- Change is localized to the requested scope (avoid drive-by refactors).
- Tests added/updated where behavior changed, and `pytest` passes.
- Docs updated if user-facing behavior/workflows changed (prefer `working_plan.md`
  for active decisions; tutorials for usage; architecture docs for concepts).
- Import boundaries respected (run the devtools checks when relevant).

## When you add or change developer docs
- Keep new dev docs under `docs/dev/`.
- Link them from `docs/README.md` if broadly relevant.
- Avoid duplicating content that already exists in architecture or tutorial docs;
  prefer adding a short pointer here instead.
