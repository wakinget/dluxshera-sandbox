# Observation Sub-Block Inference Template

## Purpose

This template configures the current tested observation sub-block inference
workflow.

The recipe itself is now driven by the configured active state, initialization,
objective, and temporal blocks rather than by hard-coded registration terms.
The bundled template remains the first validated case and jointly fits
per-frame:

- `source.x_position_as`
- `source.y_position_as`
- `source.position_angle_deg`

while keeping all shared model parameters fixed.

Within `experiment.inference.objective`, the recipe now separates:

- `frame_reduce`: pixel-domain reduction within each frame's Gaussian image NLL
- `subblock_reduce`: aggregation across the resulting frame-level terms

## Recommended workflow

1. Generate a trace CSV:

```bash
PYTHONPATH=src python examples/recipes/subblock_trace_generation.py \
  --config examples/recipes/observation_subblock_trace_template/subblock_trace_prescription.yaml
```

2. Render an observation cube:

```bash
PYTHONPATH=src python examples/recipes/observation_subblock.py \
  --config examples/recipes/observation_subblock_template/subblock_generation_prescription.yaml
```

3. Point this template at the rendered cube, then run inference:

```bash
PYTHONPATH=src python examples/recipes/observation_subblock_inference.py \
  --config examples/recipes/observation_subblock_inference_template/subblock_inference_prescription.yaml
```

Only `data.cube` is strictly required when the renderer manifest lives beside
the cube. In that common layout the inference recipe will:

- auto-discover `render/manifest.json`
- infer the truth trace path from that manifest
- write truth-comparison outputs automatically when the trace is available

Set `data.manifest` explicitly only when the render manifest is not a sibling
of the cube. Set `data.truth_trace` explicitly only when you want to override the
manifest-derived truth path.

## Lightweight screening harness

For repeated screening studies, use
`examples/scripts/run_obs_subblock_study.py`. It keeps the existing
trace -> render -> quick-look -> inference recipes visible, but adds explicit
study modes under the same stable case root:

- `full_case`
- `fisher_only`
- `profile_objective`
- `nuisance_absorption`

Example plate-scale screening runs:

```bash
PYTHONPATH=src python examples/scripts/run_obs_subblock_study.py \
  --results-root Results/obs_subblock_screening \
  --case-name plate_scale_case_01 \
  --mode fisher_only \
  --candidate optics.plate_scale_as_per_pix \
  --truth-value 0.006
```

```bash
PYTHONPATH=src python examples/scripts/run_obs_subblock_study.py \
  --results-root Results/obs_subblock_screening \
  --case-name plate_scale_case_01 \
  --mode profile_objective \
  --candidate optics.plate_scale_as_per_pix \
  --scan-values 0.0058,0.0060,0.0062
```

The harness reuses the repeated-study case layout:

```text
Results/<study>/<case>/
  trace/
  render/
  render/quicklook/
  inference/
  study/<mode>/
```

Each study mode writes a small machine-readable `summary.json` under
`study/<mode>/` plus mode-specific artifacts such as Fisher summaries,
profile curves, or nuisance-bias summaries.

For the worked Fisher / Schur shared-parameter screening case, use the
study-specific runner:

```bash
PYTHONPATH=src python examples/scripts/run_candidate_fisher_screen.py \
  --candidate optics.plate_scale_as_per_pix
```

This runs the explicit `fisher_only` matrix for one canonical candidate key at
a time. Supported examples include:

- `optics.plate_scale_as_per_pix`
- `source.log_flux_total`
- `source.contrast`
- `source.separation_as`
- `optics.primary.zernike_coeffs_nm[3]`

The worked matrix remains:

- target: `ALPHA_CEN`
- frame counts: `1, 5, 20, 50`
- noise modes: `noiseless`, `shot_noise_only`

and writes per-case Fisher/Schur summaries plus aggregate CSV/JSON/PNG review
artifacts under a derived study root of the form:

- `Results/<candidate_slug>_fisher_<target_slug>/`

For focused diagnosis of the expensive Fisher stage, narrow the matrix with the
existing subset flags. For example:

```bash
PYTHONPATH=src python examples/scripts/run_candidate_fisher_screen.py \
  --candidate optics.primary.zernike_coeffs_nm[3] \
  --study-root Results/optics_primary_zernike_coeffs_nm_i3_fisher_debug \
  --frame-counts 1,5,20 \
  --noise-modes noiseless
```

## Image-backed Schur summary validation

Use the new `schur_summary` study mode to validate one real image-backed
`SubblockSummary` before attempting any larger observation-level workflow.
For a narrative hands-on walkthrough, see
`docs/tutorials/observation_subblock_schur_summary_workflow.md`.
The recommended Schur validation path uses the registration-iid trace template
at
`examples/recipes/observation_subblock_trace_template/subblock_trace_registration_iid_prescription.yaml`.

The recommended first validation case is deliberately small:

- `n_frames: 3`
- render noise disabled
- registration-iid X/Y/PA truth jitter
- registration-only fast state
- observation-level Theta keys:
  - `source.separation_as`
  - `source.log_flux_total`
  - `source.contrast`
  - `optics.plate_scale_as_per_pix`
- `phi_ref: truth_when_available`
- Zernikes disabled

From repo root:

```bash
PYTHONPATH=src python examples/scripts/run_obs_subblock_study.py \
  --results-root Results/obs_subblock_summary_validation \
  --case-name schur_smoke_four_scalar \
  --mode schur_summary \
  --n-frames 3 \
  --noise disabled \
  --theta-keys source.separation_as,source.log_flux_total,source.contrast,optics.plate_scale_as_per_pix \
  --phi-ref truth_when_available \
  --max-dense-dim 40 \
  --schur-damping 1e-8 \
  --dry-run
```

Review the written plan first:

- `study/schur_summary/schur_summary_plan.json`
- `study/schur_summary/summary.json`

The plan is the fastest review artifact. It summarizes the resolved trace
truth setup, optimizer initialization, `phi_ref`, preconditioning status, and
planned output paths. When a frame-truth CSV already exists, it also links
`study/schur_summary/frame_truth_preview.json`.

Where to change defaults:

- Trace truth defaults live in the registration-iid trace template or narrow
  `--trace-*` CLI overrides.
- Registration initialization defaults live in this inference template or
  narrow `--init-*` CLI overrides.
- Recovered-reference optimizer and preconditioning defaults live in this
  inference template unless targeted `--reference-optimizer-*`,
  `--reference-schedule-*`, or `--reference-preconditioning-*` CLI overrides
  are passed.
- Detailed diagnostics and plot settings live in this inference template unless
  `--reference-diagnostics-profile` is selected.
- Generated case-local configs are the durable record; `schur_summary_plan.json`
  and `schur_summary_audit.json` summarize effective values and sources.

Then run the actual smoke path:

```bash
PYTHONPATH=src python examples/scripts/run_obs_subblock_study.py \
  --results-root Results/obs_subblock_summary_validation \
  --case-name schur_smoke_four_scalar \
  --mode schur_summary \
  --n-frames 3 \
  --noise disabled \
  --theta-keys source.separation_as,source.log_flux_total,source.contrast,optics.plate_scale_as_per_pix \
  --phi-ref truth_when_available \
  --max-dense-dim 40 \
  --schur-damping 1e-8
```

This prepares the tiny case, uses the truth-backed registration reference when
available, computes the dense local curvature over `[Theta, phi]`,
Schur-reduces the fast block, and writes:

- `subblock_summary.json`
- `subblock_summary_matrices.npz`
- `schur_diagnostics.json`
- `combined_curvature_diagnostics.json`
- `local_surrogate_validation.csv`
- `frame_truth_preview.json`
- `schur_summary_audit.json`

Concise smoke-test progression:

1. Run the command with `--dry-run`.
2. Inspect `schur_summary_plan.json`, especially `trace_truth`,
   `inference_init`, `phi_ref_mode`, and `preconditioning`.
3. Run the same command without `--dry-run`.
4. Inspect `subblock_summary.json`.
5. Run `inspect_subblock_summary.py` and inspect `inspection_report.json`.
6. Inspect `local_surrogate_validation.csv` and the audit summary of it.
7. Run the one-summary observation update.
8. Inspect `observation_update_summary.json["prior_mean_source"]` and the
   posterior table.

Reference glossary:

- Truth trace: simulated frame-level values used to render the image cube.
- Optimizer initialization: active-state values where registration inference
  starts if a recovered reference solve runs.
- `phi_ref`: fast-state point used to linearize the Schur summary.
- Recovered reference: a `phi_ref` obtained from registration inference.
- `preconditioning_reference`: point used to build optimizer preconditioning.
- `theta_ref`: slow observation-level point used to linearize the summary.
- Observation prior mean: belief mean used by the observation update; in
  real-summary mode it defaults to the summary `theta_ref`.

The exporter now applies `source.log_flux_total` and `source.contrast` with a
JAX-safe local runtime update path. The previous limitation was not about those
parameters being scientifically unsupported. It came from differentiating
through full `ParameterStore.refresh_derived(...)`, which reached transform
functions that used Python `float(...)` on traced values. The Schur local
objective now follows canonical inference semantics more closely: active Theta
values are authoritative, and only the minimal dependent source photometry term
(`source.raw_fluxes`) is repaired explicitly inside autodiff.

For the smallest possible debug case, the older two-key fallback is still
useful:

```bash
--theta-keys source.separation_as,optics.plate_scale_as_per_pix
```

Inspect the exported summary:

```bash
PYTHONPATH=src python examples/scripts/inspect_subblock_summary.py \
  Results/obs_subblock_summary_validation/schur_smoke_four_scalar/study/schur_summary/subblock_summary.json
```

Optionally write the compact inspection report to JSON:

```bash
PYTHONPATH=src python examples/scripts/inspect_subblock_summary.py \
  Results/obs_subblock_summary_validation/schur_smoke_four_scalar/study/schur_summary/subblock_summary.json \
  --report-json Results/obs_subblock_summary_validation/schur_smoke_four_scalar/study/schur_summary/inspection_report.json
```

Then run a one-summary observation update from the exported artifact:

```bash
PYTHONPATH=src python examples/scripts/run_observation_belief_update_demo.py \
  --summary-path Results/obs_subblock_summary_validation/schur_smoke_four_scalar/study/schur_summary/subblock_summary.json \
  --results-dir Results/observation_belief_from_real_summary \
  --run-name schur_smoke_four_scalar
```

In real-summary mode, the observation update now defaults the prior mean to the
summary's own `theta_ref` context rather than silently falling back to the bare
`SHERA_FLIGHT_3P` preset. That matters for effective render overrides such as
short exposure time, because exposure-dependent derived quantities like
`source.log_flux_total` should inherit the same context that produced the image
data and Schur summary.

Override the default prior context explicitly when needed:

```bash
PYTHONPATH=src python examples/scripts/run_observation_belief_update_demo.py \
  --summary-path Results/obs_subblock_summary_validation/schur_smoke_four_scalar/study/schur_summary/subblock_summary.json \
  --config path/to/prior_context.json \
  --system-preset SHERA_FLIGHT_3P \
  --prior-source auto \
  --results-dir Results/observation_belief_from_real_summary \
  --run-name schur_smoke_four_scalar_explicit_prior
```

For the default real-summary path, inspect
`observation_update_summary.json["prior_mean_source"]`. It should report
`summary_theta_ref` unless an explicit prior config/preset was supplied.

### First-run checklist

- `theta_labels` match the requested scalar Theta keys.
- `phi_labels` match the registration variables and frame count.
- `schur_summary_plan.json` shows the expected `n_theta`, `n_phi`,
  `combined_dim`, `phi_ref_mode`, and dense-dimension guard.
- `subblock_summary.json`, `subblock_summary_matrices.npz`, and
  `schur_diagnostics.json` exist.
- `H_pp` rank and condition number are finite and not unexpectedly singular.
- Schur damping is not obviously dominating the nuisance solve.
- `reduced_information` is symmetric within tolerance and has nonzero
  information for at least one Theta label.
- `reduced_score` is finite.
- If `phi_ref=recovered` is used later, the plan and console output should show
  whether preconditioning is enabled for the reference inference solve.
- `local_surrogate_validation.csv` shows the predicted and fixed-phi actual
  objective deltas with consistent local sign near `theta_ref`.
- `observation_update_summary.json` reports `prior_mean_source=summary_theta_ref`
  for the default real-summary observation update path.
- The one-summary observation update writes posterior artifacts successfully.

The current surrogate validation compares the reduced quadratic against
fixed-phi objective slices only. That is not yet a full nuisance-profiled
validation, but it is still the first useful check of the curvature and score
sign convention.

## Adam hyperparameter sweep

Use `examples/scripts/sweep_obs_subblock_adam.py` to run the small Adam sweep
for the current three-frame registration-only toy problem. This is a focused
workflow for choosing Adam settings for this recipe; it is not a general
optimizer benchmark harness.

Start from one inference prescription that already points at the rendered toy
cube. The active block must remain:

```yaml
active:
  frame_keys:
    - source.x_position_as
    - source.y_position_as
    - source.position_angle_deg
  shared_keys: []
```

The prescription must also make truth available, either through
`experiment.inference.data.truth_trace` or through a renderer `manifest.json`
that the inference recipe can discover from the cube path.

Run the default staged sweep:

```bash
PYTHONPATH=src python examples/scripts/sweep_obs_subblock_adam.py \
  --config examples/recipes/observation_subblock_inference_template/subblock_inference_prescription.yaml \
  --results-dir Results/obs_subblock_adam_sweeps \
  --no-progress
```

The default grid is:

- `optimizer.kind: adam`
- `objective.frame_reduce: mean`
- `objective.subblock_reduce: sum`
- `optimizer.base_lr` in `[0.03, 0.04, 0.05, 0.06]`
- `optimizer.kwargs.b1` in `[0.65, 0.7, 0.75]`
- `optimizer.kwargs.b2` in `[0.999]`
- `optimizer.kwargs.eps: 1.0e-8`

Per-run inference plots are disabled by default to keep the sweep lightweight.
Add `--per-run-plots` when you want the standard inference plots for every grid
point. Use `--dry-run` to check the planned run IDs without running inference.

Outputs are written under one experiment directory:

```text
Results/obs_subblock_adam_sweeps/<experiment>/
  manifest.json
  results.csv
  ranked_summary.csv
  recommendation.json
  recommendation.md
  final_truth_score_vs_base_lr.png
  iter_to_90pct_improvement_vs_base_lr.png
  runs/
    adam_lr.../
      sweep_run_config.json
      manifest.json
      *_recovered_trace.csv
      *_truth_comparison.csv
      truth_score_curve.csv
      normalized_residual_history.csv
```

The ranking uses successful completion only as a gate. Completed runs are then
ordered by:

1. lowest `final_truth_score`
2. lowest `iter_to_90pct_improvement`
3. lowest `settling_iter_tol`
4. lowest `ringing_index`
5. lowest `tail_std_last_k`
6. lowest `max_overshoot_ratio`

`final_truth_score` is a combined normalized RMS over recovered-minus-truth
residuals across all frames and active keys. The fixed first-pass scales are:

- `source.x_position_as`: `1.0e-3` arcsec
- `source.y_position_as`: `1.0e-3` arcsec
- `source.position_angle_deg`: `1.0e-2` deg

`iter_to_90pct_improvement` is the first iteration where the truth score reaches
90 percent of the total improvement between the initial and final truth score.

`settling_iter_tol` and `ringing_index` were added to distinguish accurate,
smooth convergence from accurate but underdamped ring-down. Both operate on the
per-iteration normalized residual history, using the same key scales as
`final_truth_score`.

`settling_iter_tol` is the first iteration where every normalized residual
component stays within `+/-0.10` for the rest of the run. Smaller values mean
the run enters the final tolerance band earlier and does not leave it. If a run
never enters and stays in the band, the metric is set to the final recorded
iteration index.

`ringing_index` counts meaningful sign changes in each normalized residual
component after ignoring samples inside a `+/-0.05` deadband. Each sign change
is weighted by the smaller adjacent absolute amplitude, then summed across all
frame/key components. Smaller values mean less oscillatory ring-down; tiny
late-stage jitter inside the deadband does not contribute.

`tail_std_last_k` is the standard deviation of the truth score over the final
10 recorded samples. `max_overshoot_ratio` is the maximum truth score during the
run divided by the initial truth score.

Find the recommendation in:

- `ranked_summary.csv`, row with `rank == 1`
- `manifest.json`, `recommendation`
- `recommendation.json`
- `recommendation.md`

To extend the grid later without editing code, pass comma-separated values:

```bash
PYTHONPATH=src python examples/scripts/sweep_obs_subblock_adam.py \
  --config path/to/subblock_inference_prescription.yaml \
  --base-lrs 1e-4,3e-4,1e-3 \
  --b1s 0.9,0.8,0.7 \
  --b2s 0.999,0.99 \
  --eps-values 1e-8 \
  --no-progress
```

## Implemented schema

The current recipe accepts:

- `experiment.kind: subblock_inference`
- `experiment.inference.data`
- `experiment.inference.active`
- `experiment.inference.init.frame` and `experiment.inference.init.shared`
- `experiment.inference.priors`
- `experiment.inference.temporal.frame_model`
- `experiment.inference.objective`
- `experiment.inference.optimizer`
- `experiment.inference.diagnostics`

For the current objective schema:

- `objective.frame_reduce` controls pixel aggregation inside each frame
- `objective.subblock_reduce` controls aggregation across frame-level data terms
- legacy `objective.reduce` is still accepted for compatibility and maps to
  `frame_reduce = reduce`, `subblock_reduce = sum`

Optimizer diagnostics can be enabled with:

- `first_step_report`
- `save_first_step_json`
- `save_fim_debug`
- `finite_difference_check`
- `plot_parameter_history_heatmap`
- `plot_parameter_residual_history_heatmap`
- `plot_parameter_history_lines`
- `plot_parameter_residual_history_lines`
- `top_k`

The current tested workflow is:

- `active.frame_keys` must be exactly `source.x_position_as`, `source.y_position_as`, `source.position_angle_deg`
- `active.shared_keys` must be `[]`
- `init.frame.mode` should be `shared_guess` with values under `init.frame.values`
- `temporal.frame_model.kind` must be `independent`
- `objective.kind` must be `nll`
- `objective.noise_model.kind` must be `gaussian`
- `objective.noise_model.variance_model` is typically `data`, optional debug `scalar`,
  or `provided_cube` when a study or helper wants to reuse an explicit variance
  cube written by the renderer
- `objective.noise_model.variance_floor` applies only to `variance_model: data`;
  it defaults to `1.0` as a count-space approximation so zero-count pixels are
  not treated as extreme-confidence pixels

The solve assumes the resolved top-level `system` block is the fixed shared
state for the block solve. `experiment.truth` is not used for shared overrides
in this recipe.

The core implementation is no longer written around a hard-coded x/y/PA theta
vector. Internally it packs:

- frame-varying active state from `active.frame_keys`
- shared active state from `active.shared_keys`
- config-driven init from `init.frame` and `init.shared`
- a block objective composed as data term + prior term + temporal term

Current limitations remain explicit:

- non-empty `priors.frame` / `priors.shared` are not implemented yet
- temporal behavior beyond `frame_model.kind: independent` is not implemented yet
- frame init modes beyond `shared_guess` / `from_system` are not implemented yet

## Output artifacts

The inference recipe writes:

- `manifest.json`
- `<file_prefix>_<timestamp>_recovered_trace.csv`
- optional `<file_prefix>_<timestamp>_truth_comparison.csv` (when truth trace is available)
- diagnostic plots (loss history, trace plots, image-fit panel)

The per-frame CSV artifacts include `frame_nll`, `frame_chi2`,
`frame_reduced_chi2`, and `frame_chi2_dof_pixels`. The manifest records matching
block-level chi-squared summaries for the initial and final models.

The image-fit panel samples representative frames and shows data, model, raw
residual, and variance-scaled Z-score maps. The Z-score uses the same variance
cube used by the Gaussian image NLL objective.

The saved inference manifest includes:

- source `config_path`
- resolved input cube/truth-trace/manifest paths
- whether the render manifest was auto-discovered
- fixed shared `system` config snapshot
- active-key partition, resolved initialization, recovered shared state, objective, optimizer, temporal, and metrics
- artifact paths
