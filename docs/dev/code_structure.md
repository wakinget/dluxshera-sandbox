# Code structure (current overview)

## Package layout (current)

`src/dluxshera/` is organized around a **components / builders / systems** split:

- `components/`: dLux-compatible classes we own (optics, sources, detectors).
- `builders/`: assembly logic, runtime bindings, and structural hashing/caching (optics, source, detector).
- `systems/`: user-facing Shera binders + configs + forward spec builders.
- `params/`: ParamSpec + ParameterStore + transforms/registry + packing utilities.
- `inference/`: losses, optimization loops, eigenmode helpers, artifacts.
- `plot/`: plotting helpers and signal panels for diagnostics.
- `legacy/`: compatibility wrappers for older Shera model APIs.

Older `core/` modules are retained only as thin re-exports for transition
paths; new code should import binders and builders directly from `systems/`
and `builders/` respectively.

## Key modules and entry points

- **Binders + configs**: `dluxshera.systems.{three_plane,two_plane}` (export configs, presets, and Binder entry points).
- **Forward ParamSpecs**: `build_forward_spec_from_config(...)` in each system module (composes source/optics/detector contracts).
- **Inference ParamSpecs**: `forward_spec.subset(infer_keys)` or `dluxshera.params.spec.build_inference_spec_basic(...)` (subset of the forward spec).
- **Transforms / derived registry**: `dluxshera.params.transforms` and
  `dluxshera.params.transform_registry`.
- **Optimization**: `dluxshera.inference.optimization` (Binder-first NLL, GD).
- **Plotting**: `dluxshera.plot.plotting` for PSF + signal panels.

## Builder modules: naming and intent

We use “builder” for modules that *construct runtime objects* from
`(config, spec, store)` inputs:

- `builders/optics.py`: canonical optics builders + structural hashing + caching.
- `builders/source.py`: Alpha Cen source assembly + runtime bindings.
- `builders/detector.py`: detector layer composition (declarative `detector.layers`), detector contracts, and runtime patching scope.

## Examples and tests

- Example scripts and notebooks live under `examples/` and call into
  the binder-based APIs in `dluxshera.systems` and `dluxshera.inference`.
- Tests live under `tests/` and exercise both binder and legacy bridges.
