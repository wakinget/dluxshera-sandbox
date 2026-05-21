# Image-backed observation subblock Schur summary workflow

This tutorial walks through the first end-to-end validation path for a real
image-backed observation subblock summary. The goal is to prove that one
prepared subblock can emit a Schur-reduced local likelihood summary over slow
observation-level parameters after eliminating fast registration nuisance
variables.

The workflow is:

```text
trace generation -> subblock rendering -> Schur summary export -> inspection -> observation update
```

This is a hands-on smoke workflow. It is not a full campaign generator, a
trajectory ingestion layer, or the final online observation filter.

## Conceptual map

`Theta` is the observation-level slow/shared parameter vector. In this tutorial
it contains four scalar physical parameters: separation, log total flux,
contrast, and plate scale.

`phi` is the local fast registration state used inside one subblock. For the
first case it is three frame-level values per frame: X, Y, and position angle.

`SubblockSummary` is the exported reduced local quadratic over `Theta`. It is
what the observation-level updater consumes.

The Schur complement removes the fast `phi` block from the local curvature so
the summary can describe the information the subblock contributes about
`Theta`.

`ObservationBeliefState` holds the prior/posterior mean and covariance over
`Theta`. The one-summary update demonstrates that a real image-backed summary
can update this belief object.

For the full design context, see
[observation_level_estimation_design.md](../dev/notes/observation_level_estimation_design.md).

## Recommended first validation case

Use the smallest case that exercises the real image-backed path:

- `n_frames = 3`
- render noise disabled
- registration-iid trace template:
  `examples/recipes/observation_subblock_trace_template/subblock_trace_registration_iid_prescription.yaml`
- four scalar `Theta` keys:
  - `source.separation_as`
  - `source.log_flux_total`
  - `source.contrast`
  - `optics.plate_scale_as_per_pix`
- `phi_ref = truth_when_available`
- Zernikes disabled
- dense dimension `13`, from `n_theta = 4` and `n_phi = 3 frames * 3`
- max dense dimension `40`
- Schur damping `1e-8`

This case is intentionally small. It isolates the Schur summary machinery from
registration optimizer convergence, avoids a large dense Hessian, and validates
shape, sign, and provenance plumbing before scaling.

`schur_summary` mode uses the registration-iid trace template by default when
`--trace-template` is omitted. The explicit path is included in the first
commands below so the truth-model source is easy to find and edit.

## Step 1: Dry-run the Schur summary plan

From the repository root:

```bash
python examples/scripts/run_obs_subblock_study.py \
  --results-root Results/obs_subblock_summary_validation \
  --case-name schur_smoke_four_scalar_tutorial \
  --mode schur_summary \
  --trace-template examples/recipes/observation_subblock_trace_template/subblock_trace_registration_iid_prescription.yaml \
  --n-frames 3 \
  --noise disabled \
  --theta-keys source.separation_as,source.log_flux_total,source.contrast,optics.plate_scale_as_per_pix \
  --phi-ref truth_when_available \
  --max-dense-dim 40 \
  --schur-damping 1e-8 \
  --dry-run
```

Inspect:

- `Results/obs_subblock_summary_validation/schur_smoke_four_scalar_tutorial/study/schur_summary/schur_summary_plan.json`
- `Results/obs_subblock_summary_validation/schur_smoke_four_scalar_tutorial/study/schur_summary/summary.json`
- `Results/obs_subblock_summary_validation/schur_smoke_four_scalar_tutorial/study/schur_summary/frame_truth_preview.json`, if a frame-truth CSV already existed

Expected plan fields:

- `n_theta = 4`
- `n_phi = 9`
- `combined_dim = 13`
- `dense_hessian_allowed = true`
- `phi_ref_mode = truth_when_available`
- `reference_inference_will_run = false`
- `trace_truth`
- `inference_init`
- `preconditioning.preconditioning_actually_used = false`
- `trace_template_source = cli_override` when the explicit tutorial flag is
  used, or `schur_summary_default` when the flag is omitted
- `registration_iid_trace_template_used = true`

On a pure dry run, there may be no generated frame-truth preview yet because no
trace or render stage has executed. The plan should still show the effective
trace template assumptions.

## Step 2: Control trace truth and inference initialization

Trace truth and optimizer initialization are different.

The trace truth is used to render the synthetic cube. Optimizer initialization
is where registration inference would start if a recovered-reference solve
runs. In a `phi_ref=truth_when_available` smoke test, reference inference does
not run, but the plan still records the initialization that would be used.

The study CLI supports narrow smoke-test overrides:

- `--trace-x0-as`
- `--trace-y0-as`
- `--trace-pa0-deg`
- `--trace-jitter-x-sigma-as`
- `--trace-jitter-y-sigma-as`
- `--trace-jitter-pa-sigma-deg`
- `--trace-seed`
- `--init-x-as`
- `--init-y-as`
- `--init-pa-deg`

The jitter flags patch an existing `iid_jitter.sigma` or
`random_walk.sigma_step` effect. The recommended Schur workflow uses the
registration-iid trace template, which includes stochastic X, Y, and PA effects.
Therefore `--trace-jitter-pa-sigma-deg` works with the recommended template. If
a different trace template is passed with `--trace-template`, the PA jitter
override still requires that template to define a compatible stochastic PA
effect.

Example dry run with explicit trace and init values:

```bash
python examples/scripts/run_obs_subblock_study.py \
  --results-root Results/obs_subblock_summary_validation \
  --case-name schur_smoke_configured_trace \
  --mode schur_summary \
  --trace-template examples/recipes/observation_subblock_trace_template/subblock_trace_registration_iid_prescription.yaml \
  --n-frames 3 \
  --noise disabled \
  --theta-keys source.separation_as,source.log_flux_total,source.contrast,optics.plate_scale_as_per_pix \
  --phi-ref truth_when_available \
  --trace-x0-as 0.0 \
  --trace-y0-as 0.0 \
  --trace-pa0-deg 14.508 \
  --init-x-as 0.0 \
  --init-y-as 0.0 \
  --init-pa-deg 14.508 \
  --max-dense-dim 40 \
  --schur-damping 1e-8 \
  --dry-run
```

The CLI flags are convenience controls. The durable source of truth remains the
generated case-local configs:

- `trace_config.json`
- `render_config.json`
- `study/schur_summary/summary_export/inference_config.json`
- copied templates under `study/schur_summary/templates/`

If a requested trace override does not map to the copied trace template, the
script fails rather than silently ignoring it.

The registration-iid template is the best place to edit the default stochastic
truth model for Schur smoke tests. Earlier smoke tests may have used the older
general trace template, which could include deterministic PA behavior and a
plate-scale constant offset. New Schur summary validation runs should use the
registration-iid template so the X/Y/PA truth model is explicit and aligned
with the intended iid registration-jitter demonstration.

## Where to change defaults

Use the smallest control surface that matches the change:

- Trace truth defaults live in
  `examples/recipes/observation_subblock_trace_template/subblock_trace_registration_iid_prescription.yaml`.
  Use narrow `--trace-*` flags for one-off smoke-test nominal and jitter values.
- Inference initialization defaults live in
  `examples/recipes/observation_subblock_inference_template/subblock_inference_prescription.yaml`.
  Use `--init-x-as`, `--init-y-as`, and `--init-pa-deg` for quick registration
  initialization checks.
- Recovered-reference optimizer and preconditioning defaults are template-owned
  unless you pass `--reference-preconditioning-enabled`,
  `--reference-preconditioning-disabled`, or
  `--reference-preconditioning-reference`.
- Detailed recovered-reference diagnostics and plots are template-owned unless
  you pass `--reference-diagnostics-profile none|basic|review|full`.
- Generated case-local configs are the durable run record. The plan and audit
  summarize the effective values and sources, including whether a value came
  from the inference template, a generated config patch, or a CLI override.

## Step 3: Run the Schur summary export

Run the same small case without `--dry-run`:

```bash
python examples/scripts/run_obs_subblock_study.py \
  --results-root Results/obs_subblock_summary_validation \
  --case-name schur_smoke_four_scalar_tutorial \
  --mode schur_summary \
  --trace-template examples/recipes/observation_subblock_trace_template/subblock_trace_registration_iid_prescription.yaml \
  --n-frames 3 \
  --noise disabled \
  --theta-keys source.separation_as,source.log_flux_total,source.contrast,optics.plate_scale_as_per_pix \
  --phi-ref truth_when_available \
  --max-dense-dim 40 \
  --schur-damping 1e-8
```

For production-style 20-frame recovered-reference summaries, keep
``--max-dense-dim 40`` so ``auto`` selects the structured independent-frame path
and add frame masking:

```bash
  --phi-ref recovered \
  --schur-frame-quality-policy mask \
  --schur-frame-chi2-threshold 5.0 \
  --schur-frame-mask-denominator original
```

Structured Schur export supports:

- `structured_independent_frames` for independent temporal models;
- `structured_linear_drift` for hard linear-drift temporal models;
- `structured_residual_prior` for profiled
  `linear_drift_residual_jitter_prior` temporal models.

Robust routing policy across demo workflows:

- independent frame-local models (including the single-star calibration demo
  `source.x_position_as/source.y_position_as` local solve) should request
  `structured_independent_frames` by default;
- hard linear-drift models should request `structured_linear_drift`;
- residual-prior models should request `structured_residual_prior`;
- dense Schur should be treated as validation/debug behavior, not the normal
  default path.

To confirm actual routing and memory behavior, inspect `schur_diagnostics.json`
and `subblock_status.csv` fields such as
`schur_curvature_method_requested`, `schur_curvature_method_effective`,
`structured_curvature_used`, and `dense_global_hessian_materialized`, then
cross-check `subprocess_diagnostics.json` and
`schur_summary_memory_audit.json` for memory attribution.

The residual-prior backend combines structured frame-separable image-data
curvature with analytic temporal-prior curvature in expanded per-frame
coordinates, so 20-frame four-scalar residual-prior runs do not need dense
image-backed Hessian evaluation by default.

For biased-reference correction experiments, keep render truth nominal and bias
only the inference/reference `Theta` value with a repeatable reference override.
The biased value becomes the exported `theta_ref`; the plan, audit, and summary
metadata record that it was applied to inference/reference only. A first small
plate-scale smoke command is:

```bash
python examples/scripts/run_obs_subblock_study.py \
  --results-root Results/obs_subblock_biased_reference \
  --case-name plate_scale_bias_3f_noiseless_dryrun \
  --mode schur_summary \
  --n-frames 3 \
  --noise disabled \
  --theta-keys source.separation_as,source.log_flux_total,source.contrast,optics.plate_scale_as_per_pix \
  --phi-ref recovered \
  --theta-reference-offset optics.plate_scale_as_per_pix=1e-5 \
  --max-dense-dim 40 \
  --schur-damping 1e-8 \
  --dry-run
```

``warn`` is the backward-compatible default and records diagnostics while keeping
all frames. ``mask`` excludes high-chi-squared frames from the structured Schur
accumulation. ``reject`` fails the summary export when frame quality is
unavailable or any frame exceeds the threshold.

To repair an existing case without rerunning recovered-reference optimization,
run the same Schur-summary command with ``--reuse-reference-inference auto`` and
the desired frame-quality flags. This expects the case-local
``study/schur_summary/reference_inference`` artifacts to still be present.

Expected layout:

```text
Results/obs_subblock_summary_validation/schur_smoke_four_scalar_tutorial/
  trace/
  render/
  study/schur_summary/
    schur_summary_plan.json
    schur_summary_audit.json
    frame_truth_preview.json
    subblock_summary.json
    subblock_summary_matrices.npz
    schur_diagnostics.json
    combined_curvature_diagnostics.json
    local_surrogate_validation.csv
    local_surrogate_validation.png
```

Required review artifacts are `schur_summary_plan.json`,
`schur_summary_audit.json`, `subblock_summary.json`,
`subblock_summary_matrices.npz`, and `schur_diagnostics.json`.
`frame_truth_preview.json`, `combined_curvature_diagnostics.json`, and local
surrogate validation outputs are preferred diagnostics for this smoke path.

## Step 4: Inspect the summary

Run the compact summary inspector:

```bash
python examples/scripts/inspect_subblock_summary.py \
  Results/obs_subblock_summary_validation/schur_smoke_four_scalar_tutorial/study/schur_summary/subblock_summary.json \
  --report-json Results/obs_subblock_summary_validation/schur_smoke_four_scalar_tutorial/study/schur_summary/inspection_report.json
```

Check:

- `theta_labels` are the requested four scalar keys.
- `phi_labels` are 3 frames times X/Y/PA.
- `H_pp` diagnostics are finite and full rank.
- Reduced information is symmetric and positive semidefinite within tolerance.
- Reduced score is finite.
- Reduced score should be near zero for a noiseless truth-reference run.
- `theta_ref` reflects the effective short-exposure summary context.
- `source.log_flux_total` reflects the subblock exposure context, not a stale
  long-exposure default preset.

The audit file is usually the best single file to read first because it links
the plan, trace truth summary, preview, Schur diagnostics, local surrogate
validation summary, and observation-prior recommendation.

## Step 5: Review local surrogate validation

`local_surrogate_validation.csv` compares the Schur-reduced quadratic
prediction against fixed-`phi` objective slices around `theta_ref`.

The comparison is intentionally limited:

- Schur-reduced predictions are nuisance-adjusted and profile-like.
- Fixed-`phi` actual objective deltas are not nuisance-adjusted.
- They do not need to match perfectly.
- The first check is sign, curvature direction, and order-of-magnitude behavior
  near `theta_ref`.

For the noiseless truth-reference smoke test, the signs should usually be
consistent for nonzero perturbations. Ratio agreement can be imperfect because
the actual slices hold nuisance registration fixed while the Schur prediction
has eliminated that block.

## Step 6: Run one-summary observation update

Feed the exported real summary into the observation-level belief update demo:

```bash
python examples/scripts/run_observation_belief_update_demo.py \
  --summary-path Results/obs_subblock_summary_validation/schur_smoke_four_scalar_tutorial/study/schur_summary/subblock_summary.json \
  --results-dir Results/observation_belief_from_real_summary \
  --run-name schur_smoke_four_scalar_tutorial
```

Expected outputs:

```text
Results/observation_belief_from_real_summary/schur_smoke_four_scalar_tutorial/
  observation_update_summary.json
  posterior_table.csv
  eigenmode_table.csv
  prior_whitened_eigenmode_table.csv
  cumulative_update_table.csv
  posterior_sigma_vs_n_subblocks.png
  posterior_sigma_over_prior_sigma_vs_n_subblocks.png
  precision_eigenvalue_spectrum.png
  prior_whitened_information_gain_spectrum.png
```

If truth context is available, the run may also write posterior error plots.

## Step 7: Interpret the one-summary result

In the recommended smoke case, `phi_ref=truth_when_available` and real-summary
mode initializes the prior mean from `summary_theta_ref`. That means the
posterior mean may barely move.

That is expected. The mean update is driven by the reduced score `g`. A
truth-reference noiseless run should have a small local score. The uncertainty
update is driven by the reduced information `S`, so posterior sigma can shrink
dramatically even when the mean does not move.

This validates information plumbing and update mechanics. It is not yet a
belief-correction demonstration. To test mean correction, use a later
experiment that intentionally offsets the prior or renders truth away from the
prior.

## Glossary of reference concepts

- Truth trace: simulated frame-level X/Y/PA values used to render the cube.
- Render context: resolved system and trace inputs used by the renderer.
- Optimizer initialization: active-state starting point for registration
  inference.
- `phi_ref`: fast-state point used to linearize the Schur summary.
- `phi_ref=truth_when_available`: use frame truth as the fast reference when
  available; reference inference does not run.
- `phi_ref=init`: use optimizer initialization as the fast reference.
- `phi_ref=recovered`: run registration inference and use the recovered state
  as the fast reference.
- Recovered reference: `phi_ref` obtained from registration inference.
- `preconditioning_reference`: point used to build optimizer preconditioning
  when reference inference runs.
- `theta_ref`: slow observation-level point used to linearize the summary.
- Observation prior mean: belief mean used by the observation update.
- `summary_theta_ref`: prior-mean source that uses the real summary's own
  `theta_ref`.
- Schur damping: small stabilization added when eliminating the nuisance block.
- Dense dimension: packed local dimension `n_theta + n_phi`; the first exporter
  uses a dense Hessian and should stay small.
- Local surrogate validation: small perturbation check comparing reduced
  predictions to fixed-`phi` objective slices.

## Moving toward more realistic runs

Scale deliberately:

1. Four-scalar, 3-frame, noiseless, `phi_ref=truth_when_available`.
2. Four-scalar, 3-frame, noiseless, `phi_ref=recovered`, with preconditioning
   enabled in the inference config.
3. Four-scalar, 5-20 frame subblocks.
4. Three to five independent subblocks aggregated in one observation update.
5. Add shot noise or provided variance sidecars.
6. Add one matched M1/M2 Zernike pair.
7. Move toward larger 20-frame subblock sequences.

The first multi-subblock target should be 3-5 blocks, not 100 or 1800.

## Troubleshooting

- `ConcretizationTypeError`: this should no longer occur for the four scalar
  keys. If it does, check the JAX-safe source photometry update path.
- Dense dimension too large: reduce frame count or disable Zernikes. Prefer
  structured Schur methods for campaign runs. Dense image-backed Schur should
  be used only for explicit small-case validation.
- Posterior log flux starts at a long-exposure value: check
  `observation_update_summary.json["prior_mean_source"]`; real-summary mode
  should use `summary_theta_ref` unless an explicit prior context was supplied.
- Posterior mean does not move: expected when the reduced score is near zero at
  the truth/reference point.
- Posterior sigma shrinks too much: expected in noiseless, high-flux,
  summed-NLL smoke tests. Do not interpret absolute sigmas scientifically yet.
- `reference_inference_will_run = false`: expected for
  `phi_ref=truth_when_available`.
- Preconditioning disabled but no inference ran: harmless for the
  truth-reference run.
- Recovered reference looks poor: enable or inspect preconditioning and
  optimizer diagnostics before trusting `phi_ref=recovered`.

## What this tutorial does not cover

This tutorial does not cover:

- full observation campaign generation,
- trajectory-driven campaigns,
- structured large-scale Schur extraction,
- online Kalman filtering,
- selective refresh,
- final parameter allocation,
- full canonical Zernike observation-level updates,
- mission-scale scientific interpretation.

## Runtime profiling and cacheability diagnostics

Enable profiling:
```bash
PYTHONPATH=src python examples/scripts/run_obs_subblock_study.py --results-root Results/obs_subblock_runtime_profile --case-name schur_profile_3f_noiseless --mode schur_summary --n-frames 3 --noise disabled --theta-keys source.separation_as,source.log_flux_total,source.contrast,optics.plate_scale_as_per_pix --phi-ref truth_when_available --max-dense-dim 40 --schur-damping 1e-8 --profile-runtime
```

Recovered-reference profiling:
```bash
PYTHONPATH=src python examples/scripts/run_obs_subblock_study.py --results-root Results/obs_subblock_runtime_profile --case-name schur_profile_20f_recovered --mode schur_summary --n-frames 20 --noise enabled --theta-keys source.separation_as,source.log_flux_total,source.contrast,optics.plate_scale_as_per_pix --phi-ref recovered --schur-curvature-method auto --max-dense-dim 40 --schur-damping 1e-8 --reference-diagnostics-profile none --profile-runtime
```

`runtime_profile_summary.json` includes stage totals and environment metadata. Treat timings as diagnostics only.

## Runtime profiling and cacheability diagnostics

Enable profiling:
```bash
PYTHONPATH=src python examples/scripts/run_obs_subblock_study.py --results-root Results/obs_subblock_runtime_profile --case-name schur_profile_3f_noiseless --mode schur_summary --n-frames 3 --noise disabled --theta-keys source.separation_as,source.log_flux_total,source.contrast,optics.plate_scale_as_per_pix --phi-ref truth_when_available --max-dense-dim 40 --schur-damping 1e-8 --profile-runtime
```

Recovered-reference profiling:
```bash
PYTHONPATH=src python examples/scripts/run_obs_subblock_study.py --results-root Results/obs_subblock_runtime_profile --case-name schur_profile_20f_recovered --mode schur_summary --n-frames 20 --noise enabled --theta-keys source.separation_as,source.log_flux_total,source.contrast,optics.plate_scale_as_per_pix --phi-ref recovered --schur-curvature-method auto --max-dense-dim 40 --schur-damping 1e-8 --reference-diagnostics-profile none --profile-runtime
```
