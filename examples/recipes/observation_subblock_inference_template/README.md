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
PYTHONPATH=src python examples/recipes/observation_subblock_trace.py \
  --config examples/recipes/observation_subblock_trace_template/prescription.yaml
```

2. Render an observation cube:

```bash
PYTHONPATH=src python examples/recipes/observation_subblock.py \
  --config examples/recipes/observation_subblock_template/prescription.yaml
```

3. Update this template’s `inputs.cube`/`inputs.trace` to point at those
   generated artifacts, then run inference:

```bash
PYTHONPATH=src python examples/recipes/observation_subblock_inference.py \
  --config examples/recipes/observation_subblock_inference_template/prescription.yaml
```

## Output artifacts

The inference recipe writes:

- `manifest.json`
- `<file_prefix>_<timestamp>_recovered_trace.csv`
- optional `<file_prefix>_<timestamp>_truth_comparison.csv` (when truth trace is available)
- diagnostic plots (loss history, trace plots, image-fit panel)
