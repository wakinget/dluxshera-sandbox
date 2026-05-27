# Single-Star Calibration Demo

This recipe runs the first image-backed calibration observation demo. It uses a
centered `source.kind: single_star` target and eliminates frame-local
registration parameters (`source.x_position_as`, `source.y_position_as`) from
each short sub-block Schur summary.

`source.position_angle_deg` is not part of the active solve for single-star
calibration. Optional PA trace jitter can be retained as `inert_diagnostic`
truth provenance (`subblocks.trace_jitter.pa_mode`) but is not solved, plotted
as recovered, or scored as a calibration failure.

The slow observation state includes `source.log_flux_total`,
`optics.plate_scale_as_per_pix`, and configurable M1/M2 Zernike coefficients.
It intentionally excludes binary-only `source.separation_as` and
`source.contrast`.

Photometry is a placeholder: the script derives an Alpha Cen A component flux
from the existing Alpha Cen binary-target photometry path and stores it as
`source.log_flux_total` for the single-star source. This is not a calibration
star registry.

Smoke commands:

```bash
PYTHONPATH=src python examples/scripts/run_single_star_calibration_demo.py \
  --config examples/recipes/single_star_calibration_demo_template/prescription.yaml \
  --results-root Results/single_star_calibration_demo \
  --run-name dryrun_smoke \
  --dry-run

PYTHONPATH=src python examples/scripts/run_single_star_calibration_demo.py \
  --config examples/recipes/single_star_calibration_demo_template/prescription.yaml \
  --results-root Results/single_star_calibration_demo \
  --run-name tiny_1block_3frame_noiseless \
  --n-subblocks 1 \
  --n-frames 3 \
  --noise disabled \
  --zernike-indices 0 \
  --max-workers 1
```

Important outputs include `campaign_plan.json`, `subblock_plan.csv`,
`posterior_by_parameter.csv`, `posterior_history.csv`, and eigenmode CSVs under
each case directory. Forecasts to 1800 sub-blocks are extrapolation diagnostics,
not a replacement for a real 30-minute image-backed run.

Trajectory-backed smoke:

```bash
PYTHONPATH=src python examples/scripts/run_single_star_calibration_demo.py \
  --config examples/recipes/single_star_calibration_demo_template/trajectory_airbus_smoke.yaml \
  --results-root Results/single_star_calibration_demo \
  --dry-run
```

`subblocks.trace_source.mode: trajectory` materializes per-subblock
`frame_truth.csv` and `starting_guess_prediction.csv` files during planning.
Single-star trajectory mode defaults to X/Y only; PA is not required. IID jitter
remains the default when `trace_source` is omitted or set to `iid_jitter`.
