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
- `objective.noise_model.variance_model` must be `data` (or optional debug `scalar`)

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
