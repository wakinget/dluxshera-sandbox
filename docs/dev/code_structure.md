# Code structure (stub)

## Package layout
- Outline the major packages under `src/dluxshera/` (core, params, optics, inference, plot) and how they relate to examples.

## Key modules
- Highlight primary entry points (binders, configs, transforms) and how to extend them.

### “Builder” modules: naming and intent

We use “builder” for modules that *construct runtime objects* from config/spec/store.

- `src/dluxshera/optics/builder.py`:
  Canonical optics builders + structural hashing + caching.

## Examples and tests
- Example scripts and notebooks live under `examples/` (scripts call into modules inside `dluxshera.demos`). Tests are under `tests/`.
