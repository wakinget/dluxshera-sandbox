# Observation Sub-Block Template

## Purpose

This template demonstrates the Phase 2 observation sub-block renderer:

- one resolved base `system`
- one explicit per-frame trace CSV
- frame-varying keys fixed to:
  - `source.x_position_as`
  - `source.y_position_as`
  - `source.position_angle_deg`
- one central-field image per frame
- outputs:
  - timestamped FITS cube
  - `manifest.json`
  - timestamped frame-truth CSV

Use this folder as a starting point for small explicit-trace rendering runs.

## Recommended workflow

Observation sub-block runs are now a two-step flow:

1. build a canonical trace CSV with
   `examples/recipes/observation_subblock_trace.py`
2. render the sub-block cube with
   `examples/recipes/observation_subblock.py`

The renderer still accepts hand-authored CSV traces directly, but the trace
builder recipe is now the recommended way to create reproducible traces.

## Files in this template

- `prescription.yaml` — canonical nested config (`system` + `experiment`)
- `frame_truth.csv` — minimal 3-frame explicit trace

## Config contract (current behavior)

The recipe expects a top-level `system` and `experiment` block.

### `system`

- Use `preset` (for example `SHERA_TESTBED_3P`) and optional overrides.
- The template keeps a minimal detector layer list for lightweight runs.

### `experiment`

- `kind` must be `observation_subblock`.
- `seed` controls optional noise sampling.
- `truth` defines shared sub-block truth overrides.
- `observation_subblock` controls trace input and validation:
  - `varying_keys` is optional metadata in v1.
  - if omitted, renderer defaults to applied keys (`x/y/PA`).
  - if provided and different, renderer still applies fixed `x/y/PA` keys and
    records requested vs applied keys in manifest metadata.
  - `trace.format` is currently `csv` in v1.
  - `trace.path` points to the explicit trace file.
  - `validate.require_contiguous_frame_index` and
    `validate.require_monotonic_time` are honored and default to `true`.
- `noise` uses the existing recipe noise path; set `enabled: false` for
  deterministic/noiseless output.
- `outputs` controls:
  - `outdir`
  - `file_prefix`
  - `frame_truth_format` (currently `csv` only)

## Trace CSV requirements

Required columns:

- `frame_index`
- `time_s`
- `source.x_position_as`
- `source.y_position_as`
- `source.position_angle_deg`

Validation rules:

- rows are sorted by `frame_index` internally
- duplicate `frame_index` values are always invalid
- required numeric fields must be finite
- `frame_index` contiguous check is controlled by
  `validate.require_contiguous_frame_index` (default `true`)
- `time_s` monotonic non-decreasing check is controlled by
  `validate.require_monotonic_time` (default `true`)

Extra columns are allowed and preserved in the output truth CSV, but ignored for
v1 rendering behavior.

## CLI options

Run with:

```bash
python examples/recipes/observation_subblock.py [options]
```

Supported options:

- `--config <path>`: prescription YAML/JSON path  
  default: `examples/recipes/observation_subblock_template/prescription.yaml`
- `--system-preset <name>`: optional system preset override merged before config
- `--results-dir <path>`: output root override
- `--run-name <name>`: run directory label under output root
- `--dry-run`: validate config + trace and print expected outputs without rendering
- `--no-progress`: disable frame-level tqdm progress output

## Usage examples

### 1) Validate only (no rendering)

```bash
PYTHONPATH=src python examples/recipes/observation_subblock.py \
  --config examples/recipes/observation_subblock_template/prescription.yaml \
  --dry-run
```

### 2) Render using template defaults

```bash
PYTHONPATH=src python examples/recipes/observation_subblock.py \
  --config examples/recipes/observation_subblock_template/prescription.yaml
```

### 3) Render to a custom output root + run name

```bash
PYTHONPATH=src python examples/recipes/observation_subblock.py \
  --config examples/recipes/observation_subblock_template/prescription.yaml \
  --results-dir Results/observation_subblock \
  --run-name demo_run
```

### 4) Generate a trace first, then render

```bash
PYTHONPATH=src python examples/recipes/observation_subblock_trace.py \
  --config examples/recipes/observation_subblock_trace_template/prescription.yaml \
  --run-name trace_run
```

Then set `experiment.observation_subblock.trace.path` in your renderer
prescription to the generated `*_frame_truth.csv`, and run the renderer recipe.

## Output layout

The recipe writes into:

`<output_root>/<run_name_or_timestamp>/`

Artifacts:

- `manifest.json`
- `<file_prefix>_<timestamp>_cube.fits`
- `<file_prefix>_<timestamp>_frame_truth.csv`

`manifest.json` includes at least:

- schema version
- created timestamp
- generator id
- frame count
- varying keys
- applied/requested varying key metadata
- trace metadata
- time start/stop
- relative artifact paths

## Current v1 limits

- Only explicit CSV traces are supported.
- Only `x/y/PA` are allowed as frame-varying rendering keys.
- No built-in motion helper generation in this renderer.
- Single central-field cube only (no multi-ROI payload yet).
