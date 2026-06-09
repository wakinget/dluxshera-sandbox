# Spectral Throughput Deck Integration

## Source Config Mapping

`EffectiveSpectrum` objects are applied to the existing `system.source` mapping.
The config integration helper writes:

- `source.wavelength_m`: center of the deck wavelength range in meters,
- `source.bandwidth_m`: max minus min wavelength in meters,
- `source.n_lambda`: number of spectral samples,
- `source.wavelengths_m`: explicit wavelength grid in meters,
- `source.weights`: normalized weights for `single_star`,
- `source.component_weights`: one normalized row per binary component,
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

For target-aware source decks, `SourceSpectralDeck` stores truth and inference
spectra by component. Single-star decks contain a `star` component. Binary-like
decks contain `primary` and `secondary` components built on the same wavelength
grid so the source config can carry distinct component rows without exposing
spectral shape as an optimizer-visible parameter.

## Render-Only Smoke

`examples/scripts/render_spectral_deck_smoke.py` demonstrates the integration by:

1. reading `experiment.spectral_model` from the full-fidelity template,
2. building a target-aware truth/inference spectral deck from packaged SEDs,
3. patching one fast three-plane simple system config into truth and inference configs,
4. rendering one PSF/image for each config,
5. writing spectral artifacts, patched configs, `*.npy` images, and
   `render_summary.json`.

The smoke should be interpreted only as a wiring check. It does not run
sub-block inference, Schur summaries, iterative updates, campaigns, optimized
quadrature, or HPC workflows.

## Default Response Data

The full-fidelity spectral template now references real response curves by
default:

- `data/filter_response/SHERA Notch Filter V2.csv`
- `data/detector_qe/LTN4323_QE.csv`

The V2 notch filter is the baseline SHERA filter response. The V1 design may be
used by explicitly overriding the filter response path, but it is not the
baseline.

In the current checkout these files are stored as package data under
`src/dluxshera/data/...`. The spectral response resolver accepts template-facing
`data/...` paths and resolves them to the packaged files when running from the
source tree. Absolute paths and paths relative to the current working directory
are also accepted.

The filter files contain metadata rows before the data header. The configured
columns are explicit:

- wavelength column: `Wavelength (nm)`
- filter response column: `R (%)`
- filter response unit: percent reflection
- filter response scale: `0.01`, converting percent reflectance to a
  dimensionless throughput

The detector QE file uses:

- wavelength column: `Wavelength (nm)`
- QE response column: `QE`
- QE response scale: `1.0`

No clipping or special negative-value handling is expected. Response values are
validated to be finite, non-negative, and not above one after scaling.

## Detector QE Proxy

`LTN4323_QE.csv` is used as the current detector QE curve even when the rendered
system config uses the `HWK4123` detector model. This is a near-term proxy:
LTN4323 and HWK4123 are close enough in the relevant specifications for the
first spectral deck and render-smoke studies.

The proxy assumption is recorded in the full-fidelity template, spectral deck
provenance, and render-smoke summary. This task does not add a new LTN4323
detector model or change detector-layer behavior; it only changes the effective
source-level spectral weighting.

## Response Overrides

`examples/scripts/render_spectral_deck_smoke.py` defaults to real response
curves from the template. It also supports:

```bash
python3 examples/scripts/render_spectral_deck_smoke.py \
  --response-mode synthetic-flat
```

and explicit response CSV overrides:

```bash
python3 examples/scripts/render_spectral_deck_smoke.py \
  --filter-response "data/filter_response/SHERA Notch Filter V2.csv" \
  --detector-qe data/detector_qe/LTN4323_QE.csv
```

Because the notch filter is a reflective M2 filter, the deck uses reflected
light from `R (%)`. The transmissive `T (%)` column is not the correct baseline
for the flight-like reflective-filter model.

Real response curves make the render-only smoke more representative of the
future full-fidelity campaign, but the smoke remains only a wiring check. It does
not run inference, Schur summaries, observation updates, campaign execution, or
HPC workflows.


## Target-Aware SEDs

The render-only smoke and source spectral deck use packaged target SED data by
default. SED files are resolved through the existing `target_sed_root()`
package-data helper. The SED loader convention is the existing
source-photometry convention: input files contain wavelength in nm and energy
flux density in `W / m^2 / nm`, which the shared utility converts to photon
spectral flux density per nm on the requested model grid.

Single-star smoke/calibration configs default to Alpha Cen A as a placeholder:
`data/target_seds/alfCenA_SED.dat`. This is a calibration convenience, not a
claim that all single-star calibrators have Alpha Cen A spectra.

Binary target configs resolve component-specific SED files from `source.target`.
The Alpha Cen mapping is:

- primary: `data/target_seds/alfCenA_SED.dat`
- secondary: `data/target_seds/alfCenB_SED.dat`

The target registry also contains explicit A/B mappings for other packaged
binary targets when their SED filenames are unambiguous, including 61 Cyg,
70 Oph, 36 Oph, xi Boo, p Eri, and HR 2667/2668. Generic `binary` configs must
provide explicit component SED paths unless a smoke-only Alpha Cen fallback is
requested. Shared-SED binary rows are now a debug fallback only and are marked in
provenance with `shared_across_binary_components: true`.

For component-specific binary decks, `source.component_weights[0, :]` receives
the primary weights and `source.component_weights[1, :]` receives the secondary
weights. Each row is normalized independently. Component flux factors are
diagnostics/provenance only and do not silently rewrite `source.contrast`, which
remains detected, post-response, and band-integrated.

Synthetic SED fallbacks remain available for debugging:

```bash
python3 examples/scripts/render_spectral_deck_smoke.py --sed-mode synthetic-ramp
python3 examples/scripts/render_spectral_deck_smoke.py --sed-mode flat
```
