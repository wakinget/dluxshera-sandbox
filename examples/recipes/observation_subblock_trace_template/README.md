# Observation Sub-Block Trace Template

## Purpose

This template builds canonical explicit trace CSV files for the observation
sub-block renderer. It is the first step in the Phase 3 two-step workflow:

1. generate an explicit trace CSV
2. render an image cube from that CSV

## Files in this template

- `prescription.yaml` — trace-generation config (`experiment` block only)

## Config contract

Set `experiment.kind: observation_subblock_trace` and define
`experiment.observation_subblock_trace`:

- `n_frames` (required): number of frames
- `dt_s` (required): frame cadence in seconds
- `seed` (optional): reproducible seed (at `experiment.seed` or in the trace block)
- `keys` (required): per-key generation specs for:
  - `source.x_position_as`
  - `source.y_position_as`
  - `source.position_angle_deg`

Each key chooses one mode:

- `explicit`: `values` list with length = `n_frames`
- `linear_drift`: `start`, `rate_per_s`
- `random_walk`: `start`, `sigma_step`
- `iid_jitter`: `center`, `sigma`

`outputs` controls where files are written:

- `outdir` (optional)
- `file_prefix` (optional)
- `write_manifest` (optional, default `true`)

## Usage

Generate a trace from the template:

```bash
PYTHONPATH=src python examples/recipes/observation_subblock_trace.py \
  --config examples/recipes/observation_subblock_trace_template/prescription.yaml
```

Validate only:

```bash
PYTHONPATH=src python examples/recipes/observation_subblock_trace.py \
  --config examples/recipes/observation_subblock_trace_template/prescription.yaml \
  --dry-run
```

## Output contract

The recipe writes:

- `<file_prefix>_<timestamp>_frame_truth.csv`
- `manifest.json` (unless `write_manifest: false`)

The CSV schema matches renderer expectations:

- `frame_index`
- `time_s`
- `source.x_position_as`
- `source.y_position_as`
- `source.position_angle_deg`

## Next step: render a cube

Point the renderer config to the generated CSV via
`experiment.observation_subblock.trace.path`, then run:

```bash
PYTHONPATH=src python examples/recipes/observation_subblock.py \
  --config <renderer_prescription.yaml>
```
