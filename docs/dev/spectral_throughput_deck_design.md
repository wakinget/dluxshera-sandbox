# Spectral Throughput Deck Integration

## Source Config Mapping

`EffectiveSpectrum` objects are applied to the existing `system.source` mapping.
The config integration helper writes:

- `source.wavelength_m`: center of the deck wavelength range in meters,
- `source.bandwidth_m`: max minus min wavelength in meters,
- `source.n_lambda`: number of spectral samples,
- `source.wavelengths_m`: explicit wavelength grid in meters,
- `source.weights`: normalized weights for `single_star`,
- `source.component_weights`: duplicated normalized rows for binary-like sources,
- `source.spectral_deck_label`: applied spectrum label,
- `source.spectral_deck_provenance`: JSON-friendly diagnostics and assumptions.

Generic `single_star` and `binary` builders consume explicit `wavelengths_m` and
weights directly. `binary_target` and `alpha_cen` keep the current linear
bandpass construction, so the deck bridge also updates center, bandwidth, and
count. The v1 deck grids are linear, making those fields consistent with the
deck wavelengths.

## Flux Semantics

`source.log_flux_total` and `source.contrast` are detected, post-response,
band-integrated quantities. The integration helper preserves them by default.

The spectral deck flux factor is recorded in provenance and diagnostics only. It
is not multiplied into normalized source weights, and applying a spectrum keeps
`weights.sum() == 1.0` for single-star sources and each component-weight row
summing to one for binary-like sources.

This keeps the first mismatch studies focused on chromatic PSF-mixture error,
not total photon-count error.

## Truth and Inference Config Split

`build_spectral_truth_inference_system_configs` takes one base system config and
a `SpectralDeck`, then returns:

- a truth system config patched with `deck.truth`,
- an inference system config patched with `deck.inference`,
- JSON-friendly provenance containing labels, wavelength counts, flux factors,
effective wavelengths, and deck comparison metrics.

No active inference parameters are added. Spectral shape remains nuisance
realism in this integration layer.

## Render-Only Smoke

`examples/scripts/render_spectral_deck_smoke.py` demonstrates the integration by:

1. reading `experiment.spectral_model` from the full-fidelity template,
2. building a synthetic truth/inference spectral deck,
3. patching one fast two-plane system config into truth and inference configs,
4. rendering one PSF/image for each config,
5. writing spectral artifacts, patched configs, `*.npy` images, and
   `render_summary.json`.

The smoke should be interpreted only as a wiring check. It does not run
sub-block inference, Schur summaries, iterative updates, campaigns, optimized
quadrature, or HPC workflows.
