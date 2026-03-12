# Detector Pipeline Audit (diagnostic)

## Scope and method

This is a read-only audit of the current detector construction path, calibration map handling, config interfaces, path resolution, and binder/model integration. No runtime behavior was modified.

Inspected files include:

- `src/dluxshera/builders/detector.py`
- `src/dluxshera/components/detectors.py`
- `src/dluxshera/layers/detector_layers.py`
- `src/dluxshera/systems/base.py`
- `src/dluxshera/systems/two_plane.py`
- `src/dluxshera/systems/three_plane.py`
- `examples/recipes/canonical_astrometry.py`
- `examples/recipes/canonical_monte_carlo.py`

---

## 1) Current behavior summary

### What detector object is built today

- `build_detector(cfg)` returns a `SheraDetector`, which subclasses `dLux.LayeredDetector` and adds non-pytree metadata via `.spec`.
  - Builder: `src/dluxshera/builders/detector.py` (`build_detector`).
  - Wrapper class: `src/dluxshera/components/detectors.py` (`class SheraDetector`).
- In practice, binders consume it as a `dl.LayeredDetector` and pass it to `dl.Telescope`.

### Layer pipeline and ordering

- Layers are **declarative**: `system.detector.layers` lists ordered layer configs (downsample, pixel_offsets, pixel_response, jitter supported today).
- Each layer entry is converted via `build_detector_layer(name, layer_cfg, target_shape)`, and `target_shape` is derived from `system.optics.psf_npix`.

### Config keys currently read for detector construction

- Preferred path: nested detector block (`system.detector`), including:
  - `model` (metadata/spec selection)
  - `layers` (ordered detector pipeline)
- The builder normalizes nested detector blocks via `_normalize_detector_cfg`; declarative `system.detector.layers` is required.

### Config declarations in dataclasses

- `SheraThreePlaneConfig` and `SheraTwoPlaneConfig` include `detector_model` and `detector_layers` fields; defaults seed the declarative detector pipeline directly.

---

## 2) Code map (cfg → detector → binder → model evaluation)

### Call graph (current)

1. Binder construction (`SheraTwoPlaneBinder` / `SheraThreePlaneBinder`) calls `BaseSheraBinder.__init__`.
2. `BaseSheraBinder.__init__` calls `self._build_detector()` once at binder init.
3. `BaseSheraBinder._build_detector()` imports and calls `build_detector(self.cfg)`.
4. `build_detector(cfg)` assembles calibrated layer objects and returns `SheraDetector`.
5. `BaseSheraBinder.__init__` then builds a cached `self.telescope = self._build_telescope(..., detector=detector)`.
6. `model()` behavior:
   - `model(store_delta=None)`: uses cached `self.telescope.model()` (no detector rebuild).
   - `model(store_delta=...)`: applies runtime updates via `_apply_runtime_updates`; detector runtime updates are limited to jitter sigma plus any explicit detector bindings (currently none beyond jitter).
   - structural rebuild path (`update_store(..., allow_rebuild=True)`): detector is rebuilt only if detector is marked structural. Today detector structural keys are empty, so detector rebuilds remain rare.

### Key functions and locations

- Detector build and calibration wiring:
  - `src/dluxshera/builders/detector.py`
    - `_resolve_detector_spec`
    - `_resolve_repo_path`
    - `_load_array`
    - `_condition_detector_map`
    - `build_detector`
    - `apply_runtime_bindings`
- Detector classes/specs:
  - `src/dluxshera/components/detectors.py`
    - `DetectorSpec`
    - `GSENSE2020BSI_SPEC`, `HWK4123_SPEC`
    - `SheraDetector(dl.LayeredDetector)`
- Custom offset layer implementation:
  - `src/dluxshera/layers/detector_layers.py`
    - `ApplyPixelOffsets`
- Binder integration:
  - `src/dluxshera/systems/base.py`
    - `BaseSheraBinder.__init__`
    - `_build_detector`
    - `_build_telescope`
    - `_apply_runtime_updates`
    - `_rebuild_telescope`
    - `model`
- Binder concrete classes:
  - `src/dluxshera/systems/two_plane.py` (`SheraTwoPlaneBinder`)
  - `src/dluxshera/systems/three_plane.py` (`SheraThreePlaneBinder`)

---

## 3) Calibration map handling (dx/dy/prf)

### What is supported now

- `dx` and `dy` maps:
  - Loaded from file paths (`ppu_dx_path`, `ppu_dy_path`) if present.
  - Else default to zero maps of shape `(psf_npix, psf_npix)`.
- Pixel response (`prf_path`):
  - Loaded from file if present.
  - Else defaults to ones map of shape `(psf_npix, psf_npix)`.

### Conditioning behavior

`_condition_detector_map(arr, map_name, target_shape)` enforces:

- Must be 2D.
- Must be square.
- If larger than target: center crop.
- If smaller than target: reflect-pad centered.
- If equal: pass through.
- Emits warning when shape conditioning occurs.

This conditioning is used for all three map types in `build_detector`:

- `dx_map`
- `dy_map`
- `pixel_response`

### File formats

`_load_array(path)` supports:

- `.npy`: direct `np.load`
- `.npz`: key preference order:
  - `data`, `arr_0`, `dx`, `dy`, `prf`, `pixel_response`
  - fallback to first array in archive
- Other suffixes raise `ValueError`.

### Array input support (direct arrays)

- Current API supports **paths only** for calibration maps in builder config.
- There is no explicit pathway for `cfg` to provide dx/dy/prf as in-memory arrays directly.

---

## 4) Path resolution findings

### Existing behavior

A repo-relative resolver exists already in `src/dluxshera/builders/detector.py`:

- `_find_repo_root(Path(__file__).resolve())` searches upward for `.git`, `pyproject.toml`, or `setup.cfg`.
- `_REPO_ROOT` is cached module-level.
- `_resolve_repo_path(path)` resolves:
  - absolute paths unchanged
  - relative paths against `_REPO_ROOT`

Therefore calibration paths in detector builder are interpreted as repo-root-relative when not absolute.

### Working-directory assumptions

- Detector builder itself is **not CWD-dependent** for relative calibration paths due to `_resolve_repo_path`.
- Some example scripts define their own repo-root logic independently (e.g., `REPO_ROOT = Path(__file__).resolve().parents[2]`, and helper `_repo_relative_path` in Monte Carlo recipe), but this is not shared detector-path utility.

### Recommendation

- Reuse detector builder’s existing `_resolve_repo_path` behavior as the canonical path semantics for detector calibration inputs.
- If future `system.detector.layers` accepts file-backed params broadly, centralize path resolution in a shared utility (e.g., `utils/paths.py`) to avoid duplicate repo-root logic across modules.

---

## 5) Recommendations for next implementation step (`system.detector.layers`)

## Minimal, lowest-risk insertion point

The best insertion point is **inside `build_detector(cfg)`** in `src/dluxshera/builders/detector.py`, because it already centralizes:

- detector model metadata resolution,
- calibration map loading and conditioning,
- layer object assembly and ordering,
- repo-root-relative path handling.

Add a helper like `build_detector_layers(cfg, target_shape, spec)` (or `build_detector_layer(layer_cfg, ...)`) that returns the ordered layer tuples consumed by `SheraDetector`.

## Existing scaffolding to preserve

Keep/reuse:

- `_condition_detector_map(...)` (already robust for shape coercion + warnings)
- `_load_array(...)` (npy/npz support + key fallback)
- `_resolve_repo_path(...)` (repo-root-relative behavior)
- `ApplyPixelOffsets` implementation and current defaults
- `SheraDetector` wrapper and detector spec logic

## Proposed compatibility strategy

1. Preserve current behavior when `system.detector.layers` is absent:
   - build the exact same 4-layer default pipeline in same order.
2. If `system.detector.layers` present:
   - parse ordered list and instantiate recognized layer types.
   - validate required params per layer; provide clear errors.
3. For map-backed layers (`pixel_offsets`, `pixel_response`):
   - accept explicit map paths (or future arrays), then apply existing conditioning.
   - if map parameter omitted, keep identity defaults (zeros for dx/dy, ones for prf).
4. Keep detector runtime bindings empty initially unless explicit non-structural detector knobs are introduced.

## Suggested config schema direction

Current config is flat (`cfg.<field>`). To minimize risk, support a dual-read strategy temporarily:

- Preferred new nested: `cfg.system.detector.layers` (or equivalent store/config object)
- Backward compatibility fallback: existing flat keys (`ppu_dx_path`, etc.)

This allows gradual migration while avoiding breaking existing examples and binders.

## Risks / gotchas to avoid

- Accidentally changing default layer order (this can alter numerics).
- Losing current identity defaults when calibration files are missing.
- Making path interpretation CWD-dependent.
- Adding detector structural-store keys prematurely without clear rebuild semantics.
- Diverging two-plane vs three-plane behavior unnecessarily; both currently inherit the same detector builder path via `BaseSheraBinder`.

---

## Direct answers to the audit questions

1. **Current pipeline shape:** always built as `SheraDetector` (a `LayeredDetector` subclass) with ordered layers: downsample → pixel_offsets → pixel_response → jitter.
2. **Current config interface:** flat keys on `cfg` via `getattr`; includes detector model, map paths, interpolation, jitter knobs, plus `psf_npix`/`oversample`.
3. **Calibration maps:** file-loaded from `.npy/.npz`; default identity maps when missing; all conditioned through `_condition_detector_map`.
4. **Path handling:** repo-root-relative resolution already exists in detector builder; not CWD-based there.
5. **Best hook for future `system.detector.layers`:** detector builder layer assembly in `build_detector`, preserving existing helpers/defaults and introducing an ordered layer-construction helper.
