# Observation Sub-Block Inference Template

## Purpose

This template configures the first observation sub-block inference milestone:
registration-only block inference.

It jointly fits per-frame:

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

Only `inputs.cube` is strictly required when the renderer manifest lives beside
the cube. In that common layout the inference recipe will:

- auto-discover `render/manifest.json`
- infer the truth trace path from that manifest
- write truth-comparison outputs automatically when the trace is available

Set `inputs.manifest` explicitly only when the render manifest is not a sibling
of the cube. Set `inputs.trace` explicitly only when you want to override the
manifest-derived truth path.

## Output artifacts

The inference recipe writes:

- `manifest.json`
- `<file_prefix>_<timestamp>_recovered_trace.csv`
- optional `<file_prefix>_<timestamp>_truth_comparison.csv` (when truth trace is available)
- diagnostic plots (loss history, trace plots, image-fit panel)

The saved inference manifest includes:

- source `config_path`
- resolved input cube/trace/manifest paths
- whether the render manifest was auto-discovered
- fixed shared `system` config snapshot
- `shared_truth` overrides and the shared initialization used for all frames
- optimizer, loss, metrics, and artifact paths
