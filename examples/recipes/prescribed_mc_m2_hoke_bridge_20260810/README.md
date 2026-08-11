# Prescribed-MC M2 HO-WFE Bridge

Purpose: run one noiseless canonical prescribed-MC fit using the imported
full-fidelity M2 high-order WFE knowledge-error map at center field.

Source bundle:

`Results/hpc_imports/m2_hoke_canonical_bridge_0p5nm_center/`

The imported bundle came from
`ff_howfe_production_center_cond_m2_hoke_0p5nm_xp0p0_yp0p0_w10x30_draw_000`.
It is treated as immutable source data. The prescription references its NPY
arrays by repo-relative path and does not regenerate truth or error maps.

WFE split:

- Truth/data model: M1 high-order truth map, M2 high-order truth map.
- Inference/reference model: same M1 high-order truth map, same M2 high-order
  truth map plus the saved 0.5 nm M2 high-order knowledge-error residual.
- M1 high-order knowledge error is disabled.
- M1/M2 low-order Z4-Z11 physical truth coefficients come from the imported
  HO-WFE decomposition. The campaign plan's
  `high_order_wfe.provenance.*.low_order_truth_coefficients_nm` values match
  `model_split/high_order_wfe/high_order_wfe_summary.json`.

State semantics:

- Physical source truth is the nominal/campaign truth state, not the draw-000
  prior realization. X/Y truth is stationary at zero.
- PA truth is the nominal ALPHA_CEN target-registry value, `14.508 deg`.
- Separation and contrast use the campaign plan's iterative physical truth
  values because they are the resolved truth for the imported spectral setup.
- `source.log_flux_total` is derived from the canonical `1800 s` exposure
  normalization. The imported campaign frame truth was recorded for `0.05 s`;
  this bridge preserves the draw-000 log-flux offset for initialization rather
  than forcing the absolute 0.05 s log flux into a 1800 s image.
- The draw-000 slow-state offsets from `campaign_plan.json` control
  initialization, not truth. The recipe uses `init.sampling: prior`, then
  explicit init overrides for separation, log flux, contrast, plate scale, and
  M1/M2 Z4-Z11.
- X/Y/PA initialization remains the reproducible canonical prior draw around
  the stationary/nominal truth.
- Truth and inference spectra are deliberately identical; no spectral-deck
  truth/inference split is reproduced in this canonical surrogate.
- Truth and inference detector models are deliberately identical and stationary:
  pixel MTF, diffusion, pixel offsets, pixel response, and static jitter are
  retained. The trajectory-driven line-smear layer is intentionally omitted.
- The only intended truth/inference model mismatch is the imported 0.5 nm M2
  HO-WFE knowledge-error residual.

Known M2 provenance:

- campaign/base truth seed: `20260610`
- realized M2 truth seed: `20262621`
- realized M2 KE seed: `221835431`
- M2 full truth RMS: `20.0 nm`
- M2 high-order truth RMS after low-order removal: `16.243746781693204 nm`
- M2 KE RMS: `0.5 nm`
- M2 truth hash: `23aa6f63709b0341ceedd649f015240064fdbb81264accfd70c885d2a18593de`
- M2 KE hash: `ee501a62681796f1c251ebfe91d0992003abf4751291e4ee85dd1301aeb9b390`
- normalized M2 KE morphology hash:
  `48e71e820ba20e36b33114f3ce299d8460c18b417bc62aaeb8b3748cc355ae01`

Fitted canonical parameters:

`source.x_position_as`, `source.y_position_as`,
`source.position_angle_deg`, `source.log_flux_total`, `source.contrast`,
`source.separation_as`, `optics.plate_scale_as_per_pix`,
`optics.primary.zernike_coeffs_nm[0:8]`, and
`optics.secondary.zernike_coeffs_nm[0:8]` for 23 scalar fitted primitives.
Exposure time is fixed at 1800 s and is not inferred.

Review WFE maps:

```bash
python3 examples/scripts/review_m2_hoke_bridge_wfe_maps.py
```

This writes
`Results/prescribed_mc_m2_hoke_bridge_20260810/wfe_review/wfe_review_summary.json`
and
`Results/prescribed_mc_m2_hoke_bridge_20260810/wfe_review/m2_hoke_bridge_wfe_maps.png`.
It also writes
`Results/prescribed_mc_m2_hoke_bridge_20260810/wfe_review/bridge_state_audit.json`
and
`Results/prescribed_mc_m2_hoke_bridge_20260810/wfe_review/bridge_state_audit.csv`,
which list the 23 fitted scalar truth values, initial values, init-minus-truth
offsets, and imported campaign provenance.

Prescribed-MC optimization plots are disabled in this recipe because the local
process exits during the heavier plot-generation stage on this workstation.

Dry-run:

```bash
python3 examples/recipes/prescribed_monte_carlo.py \
  --prescription examples/recipes/prescribed_mc_m2_hoke_bridge_20260810/prescription.yaml \
  --outdir Results/prescribed_mc_m2_hoke_bridge_20260810 \
  --dry-run
```

Execute one local smoke test:

```bash
python3 examples/recipes/prescribed_monte_carlo.py \
  --prescription examples/recipes/prescribed_mc_m2_hoke_bridge_20260810/prescription.yaml \
  --outdir Results/prescribed_mc_m2_hoke_bridge_20260810 \
  --results-orientation row
```

Noise and loss:

Observation noise is disabled. The Gaussian NLL uses a fixed variance floor of
`0.5` through `data_var = max(expected image counts, 0.5)` so the deterministic
image has a well-defined likelihood without drawing photon/read/dark noise.

Success comparison:

The canonical surrogate should recover a deterministic signed separation bias
near `-0.6 mas`, comparable to the iterative campaign's `-603.8 uas` standard
final-window ensemble result and `-594.2 uas` tail-3 cumulative result. Exact
equality is not expected because this bridge uses a single noiseless canonical
image with global X/Y/PA nuisance terms rather than the iterative campaign's
per-frame registration treatment.
