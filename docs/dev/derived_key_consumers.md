# Derived keys usage audit: runtime vs reporting

## Scope and method

This report audits derived keys declared in the **forward specs** for:
- `shera_twoplane`
- `shera_threeplane`

and traces where those derived keys are consumed across:
- (A) runtime forward model / loss path (`binder.model`, optimization)
- (B) builders (`build_alpha_cen_source`, optics runtime bindings)
- (C) analysis / reporting / plotting (`signals`, plots, summaries)

It also gives a focused recommendation for `binary.raw_fluxes` (aka “source.raw_fluxes” in earlier wording).

Current source-kind convention:
- `source.kind: binary_target` is the legacy-compatible target-registry path and
  still declares `source.raw_fluxes` for compatibility.
- `source.kind: single_star` is the calibration-friendly dLux `PointSource`
  path. It exposes `source.log_flux_total` as the public flux parameter and does
  not require `source.separation_as`, `source.contrast`, or `source.raw_fluxes`.
- `source.kind: binary` is the generic dLux `BinarySource` path. It exposes
  `source.log_flux_total` as total binary photons and converts internally to
  dLux `mean_flux`.
- Linear dLux `flux` / `mean_flux` values are source-builder details. Public
  inference and reporting should continue to use `source.log_flux_total`.
- New reporting code should prefer the source-aware helper
  `compute_source_flux_diagnostics(...)` instead of assuming
  `source.raw_fluxes` exists for every source kind.
- `examples/scripts/generate_target_grating_portraits.py` includes an optional
  Alpha Cen A-like `single_star` visual smoke (`--include-alpha-cen-a-single-star`)
  that exercises the `PointSource` path without introducing a calibration-star
  registry.

---

## 1) Derived keys declared by system kind

## Two-plane forward spec (`src/dluxshera/systems/two_plane.py`)
- `source.log_flux_total` (derived)
  - transform: `source.log_flux_total`
  - depends on:
    - `source.spectral_flux_density`
    - `source.throughput`
    - `source.exposure_time_s`
    - `optics.m1_diameter_m`
    - `optics.bandwidth_m`
- `source.raw_fluxes` (derived)
  - transform: `source.raw_fluxes`
  - depends on:
    - `source.log_flux_total`
    - `source.contrast`

## Three-plane forward spec (`src/dluxshera/systems/three_plane.py`)
- `optics.plate_scale_as_per_pix` (derived)
  - transform: `optics.plate_scale_as_per_pix`
  - depends on:
    - `optics.focal_length_m`
    - `detector.pixel_pitch_m`
- `source.log_flux_total` (derived)
  - transform: `source.log_flux_total`
  - depends on:
    - `source.spectral_flux_density`
    - `source.throughput`
    - `source.exposure_time_s`
    - `optics.m1_diameter_m`
    - `optics.bandwidth_m`
- `source.raw_fluxes` (derived)
  - transform: `source.raw_fluxes`
  - depends on:
    - `source.log_flux_total`
    - `source.contrast`

---

## 2) Consumption classification: runtime-critical vs report-only

| Derived key | Two-plane | Three-plane | Runtime forward model/loss (A) | Builders (B) | Analysis/reporting/plotting (C) | Classification |
|---|---:|---:|---|---|---|---|
| `optics.plate_scale_as_per_pix` | primitive in forward spec | ✅ derived in forward spec | Used by optics runtime binding path that sets `psf_pixel_scale` | Optics runtime bindings include this key | Used in diagnostics (`plate_scale_error_ppm`) | **Runtime-critical** |
| `source.log_flux_total` | ✅ | ✅ | Consumed by source build; directly sets AlphaCen `log_flux` used for image generation and thus loss | `build_alpha_cen_source` reads this key | Also used in labels/summary contexts | **Runtime-critical** |
| `source.raw_fluxes` | ✅ | ✅ | Not used in binder/model/loss code paths | Not consumed by source/optics builders | Consumed by `build_signals` to produce `source.raw_flux_error_ppm`; consumed by plotting/tests around that signal | **Report-only** |

Notes:
- Inference spec also declares `binary.raw_fluxes` as a derived field for diagnostics-oriented decoding (`refresh_derived(inference_spec)` workflows).
- Runtime modeling path primarily needs `binary.log_flux_total`, `binary.contrast`, and plate scale; it does not require `binary.raw_fluxes` to form the model image.

---

## 3) Focus audit: `source.raw_fluxes` / `binary.raw_fluxes`

## All call sites found

### Transform / declaration layer
- Declared in three-plane forward spec as derived key `binary.raw_fluxes`.
- Declared in inference spec as derived key `binary.raw_fluxes`.
- Computed by transform `binary_raw_fluxes` from `(binary.log_flux_total, binary.contrast)`.

### Runtime path (binder/model/loss)
- **No call sites found** in binder model path or optimization loss construction that require `binary.raw_fluxes`.
- Source builder constructs `AlphaCen` from `log_flux_total` + `contrast`; no read of `binary.raw_fluxes`.

### Reporting / plotting path
- `inference/signals.py` computes component fluxes on demand from
  source-kind-aware public primitives. It does not require `source.raw_fluxes`
  in decoded step mappings.
- `plot/plotting.py` renders panel `raw_flux_error_ppm.png` from that signal.
- `tests/inference/test_signals.py` exercises transform parity and signal
  generation for binary flux diagnostics.

## Minimal replacement recommendation

If the goal is to remove `binary.raw_fluxes` from the **forward spec** only:
1. Keep `binary.raw_fluxes` out of runtime model/loss paths (already true).
2. Compute raw fluxes for reporting from source semantics directly, using one of:
   - `AlphaCen.raw_fluxes` property from a temporary source object built from decoded `(log_flux_total, contrast)`.
   - or a shared utility function implementing the exact AlphaCen mapping.

A low-risk approach is a shared utility, e.g.:
- `compute_raw_fluxes_from_logflux_contrast(log_flux_total, contrast)`
- used by both transform code (if kept) and signals/reporting, so there is one canonical formula.

Parity check:
- Current transform formula matches `AlphaCen.raw_fluxes` algebraically:
  - `flux_B = total / (1 + contrast)`
  - `flux_A = contrast * flux_B`

## Impact of removing from forward spec
- Expected runtime impact: **none** for binder/model/loss, because no runtime consumer currently requires this field.
- Reporting impact: none if signals/truth pathways compute it on demand from `(log_flux_total, contrast)` or via `AlphaCen.raw_fluxes`.

---

## 4) Staleness-risk notes

Derived-key freshness risks in current architecture:

1. `ParameterStore.replace(...)` does **not** auto-refresh derived keys.
   - Any downstream consumer that expects up-to-date derived values must call `refresh_derived(spec)` explicitly.

2. `refresh_derived` mitigates staleness by stripping derived values and recomputing them from primitives.
   - This avoids stale carry-over when explicitly invoked.

3. Binder runtime evaluation (`binder.model(store_delta)`) validates but does not auto-refresh derived fields.
   - If a caller supplies derived keys directly in `store_delta`, those values can override recomputation and may become stale/inconsistent with edited primitives.

4. Signal decoding paths are only fresh when decoder returns refreshed stores.
   - Current optimization helper defaults do call `.refresh_derived(forward_spec)` in decoder construction, which is good.

Practical takeaway:
- Treat runtime-critical derived quantities (`binary.log_flux_total`, effective plate scale in three-plane forward workflows) as needing explicit refresh after primitive edits unless the workflow constructs them directly.
- `binary.raw_fluxes` is best computed “just in time” in reporting from current `(log_flux_total, contrast)` to minimize staleness surface.

---

## Decision summary

- **Can `binary.raw_fluxes` be removed from the forward spec?**
  - **Yes, likely safely**, based on current code paths: runtime model/loss do not consume it.
- **What must be preserved?**
  - Reporting/signal generation of raw-flux error should still derive fluxes from current photometric primitives, ideally via `AlphaCen.raw_fluxes` semantics through a shared utility.
