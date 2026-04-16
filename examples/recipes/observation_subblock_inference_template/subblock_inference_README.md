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

The saved inference manifest includes:

- source `config_path`
- resolved input cube/truth-trace/manifest paths
- whether the render manifest was auto-discovered
- fixed shared `system` config snapshot
- active-key partition, resolved initialization, recovered shared state, objective, optimizer, temporal, and metrics
- artifact paths
