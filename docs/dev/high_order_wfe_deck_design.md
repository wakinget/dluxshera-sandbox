# High-Order WFE Deck Design

## Purpose

The high-order WFE deck is the first reusable artifact layer for the future
full-fidelity SHERA algorithm campaign. It generates deterministic primary and
secondary mirror OPD truth maps, extracts the active low-order Zernike state, and
records truth/knowledge artifacts without adding campaign execution or active
high-order inference.

## Baseline OPD Budget

The baseline mirror surface figure budget is about 10 nm RMS surface. Reflection
turns surface height into twice the optical path difference, so the deck baseline
is 20 nm RMS OPD for each mirror map.

Deck-facing units are nanometres. FITS filenames use `_opd_nm.fits`, FITS
headers use `BUNIT = nm`, and JSON manifests state `opd_unit: nm` and
`coefficient_unit: nm`.

## Map Generation

Each mirror receives an independent correlated random OPD map. The v1 generator
creates deterministic complex Fourier coefficients from a seed, scales the
amplitude by radial frequency as `f**(-alpha / 2)`, suppresses the DC term, and
inverse-transforms to a real spatial map. With the default `alpha = 2.5`, the
approximate two-dimensional power spectrum follows `1/f**2.5`.

The generated map is finite, real-valued, and normalized over the pupil mask to
the requested RMS OPD. The default full-truth RMS is 20 nm.

## Mask Policy

The v1 lightweight fallback is a centered circular pupil mask in normalized
coordinates. The deck records `mask_policy: circular_fallback` in provenance.
Callers may provide an explicit boolean mask; in that case provenance records
`mask_policy: explicit`.

A future optical integration task should replace or augment this with the actual
system pupil when that can be accessed without building a large dLux model.

## Zernike Processing

The deck uses Noll indices. Processing order for each mirror is:

1. Generate a correlated raw OPD map.
2. Fit and subtract piston, tip, and tilt (`Z1`, `Z2`, `Z3`) over the mask.
3. Normalize the PTT-removed map to 20 nm RMS OPD over the mask.
4. Fit low-order active WFE coefficients `Z4` through `Z11` over the mask.
5. Reconstruct the fitted `Z4` through `Z11` map.
6. Subtract that reconstruction from the normalized full map.
7. Retain the residual as high-order WFE truth.

Coefficient labels are explicit (`Z4`, ..., `Z11`). Active-state mapping is also
recorded explicitly, for example `optics.primary.zernike_coeffs_nm[0]`
corresponds to `Z4`; array index is not treated as the Noll index.

## Knowledge Model

Low-order knowledge coefficients are generated independently per coefficient:

```text
knowledge_coeff_nm = truth_coeff_nm + error_nm
```

The default coefficient error scale is 2.0 nm per coefficient. The deck supports
other scalar profiles such as 0.1, 0.3, 1.0, and 3.0 nm.

High-order knowledge maps are additive correlated map errors:

```text
high_order_knowledge_nm = high_order_truth_nm + high_order_error_nm
```

The default high-order error map RMS is 0.3 nm OPD over the same mask. Its power
law defaults to the truth alpha, and can be configured independently.

## Artifact Contract

`write_high_order_wfe_deck_artifacts(...)` writes an `optics/`-style directory
containing:

- `high_order_wfe_deck_manifest.json`
- `low_order_zernike_truth.csv`
- `low_order_zernike_knowledge.csv`
- `low_order_zernike_errors.csv`
- `primary_full_truth_opd_nm.fits`
- `primary_high_order_truth_opd_nm.fits`
- `primary_high_order_knowledge_opd_nm.fits`
- `primary_high_order_error_opd_nm.fits`
- `primary_mask.fits`
- `secondary_full_truth_opd_nm.fits`
- `secondary_high_order_truth_opd_nm.fits`
- `secondary_high_order_knowledge_opd_nm.fits`
- `secondary_high_order_error_opd_nm.fits`
- `secondary_mask.fits`

The manifest records schema version, timestamps, units, seeds, mask policy,
Zernike indices, measured RMS values, coefficient mappings, and compact M1/M2
comparisons.

## Deferred Work

This task intentionally does not implement sub-block inference, Schur summaries,
observation-level updates, full campaign execution, iterative campaign logic,
detector pixel-offset or flat-field decks, smear modeling, active high-order WFE
inference, HPC workflows, or a full optical refactor.

The current optics builder has only a partial high-order map hook. Full wiring of
truth/knowledge low-order coefficients plus primary and secondary high-order OPD
maps into renderable system configs is deferred to the next focused task.
