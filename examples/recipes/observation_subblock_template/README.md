# Observation Sub-Block Template

## Purpose

This template demonstrates the observation-subblock renderer:

- one resolved base `system`
- one explicit per-frame trace CSV
- configurable per-frame varying keys
- one central-field image per frame
- outputs:
  - timestamped FITS cube
  - `manifest.json`
  - timestamped frame-truth CSV

Use this folder as a starting point for explicit-trace rendering runs.

## Recommended workflow

1. build a canonical trace CSV with
   `examples/recipes/observation_subblock_trace.py`
2. render the sub-block cube with
   `examples/recipes/observation_subblock.py`

Optional follow-ons:

- infer per-frame registration with
  `examples/recipes/observation_subblock_inference.py`
- inspect quick-look diagnostics with
  `examples/scripts/visualize_obs_subblock.py`

## Files in this template

- `prescription.yaml`: canonical nested config (`system` + `experiment`)
- `frame_truth.csv`: minimal explicit trace example

## Config contract (current behavior)

The recipe expects top-level `system` and `experiment` blocks.

### `system`

- use `preset` (for example `SHERA_TESTBED_3P`) and optional overrides
- this template keeps a minimal detector layer list for lightweight runs

### `experiment`

- `kind` must be `observation_subblock`
- `seed` controls optional noise sampling
- `truth` defines shared sub-block truth overrides
- `observation_subblock` controls trace input and validation:
  - `varying_keys` is the applied per-frame varying-key list
  - if omitted, renderer defaults to:
    - `source.x_position_as`
    - `source.y_position_as`
    - `source.position_angle_deg`
  - supported key syntax:
    - scalar: `a.b.c`
    - indexed vector component: `a.b.c[index]`
  - supported key family includes non-structural runtime/inference-facing keys
    (source/runtime scalars, plate scale, and indexed primary/secondary
    Zernike coefficients)
  - `trace.format` currently supports `csv`
  - `trace.path` points to explicit trace CSV
  - `validate.require_contiguous_frame_index` and
    `validate.require_monotonic_time` are honored and default to `true`
- `noise` uses the existing recipe noise path; set `enabled: false` for
  deterministic/noiseless output
- `outputs` controls:
  - `outdir`
  - `file_prefix`
  - `frame_truth_format` (currently `csv` only)

## Trace CSV requirements

Required columns:

- `frame_index`
- `time_s`
- one column for each configured `varying_keys` entry

Validation rules:

- rows are sorted by `frame_index` internally
- duplicate `frame_index` values are always invalid
- required numeric fields must be finite
- contiguous check is controlled by
  `validate.require_contiguous_frame_index` (default `true`)
- monotonic time check is controlled by
  `validate.require_monotonic_time` (default `true`)

Extra columns are allowed and preserved in output truth CSV, but they do not
drive rendering updates unless included in `varying_keys`.

## CLI options

Run with:

```bash
python examples/recipes/observation_subblock.py [options]
```

Supported options:

- `--config <path>`: prescription YAML/JSON path  
  default: `examples/recipes/observation_subblock_template/prescription.yaml`
- `--system-preset <name>`: optional preset override merged before config
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

Then set `experiment.observation_subblock.trace.path` in your renderer config to
the generated `*_frame_truth.csv`, and run the renderer recipe.

### 5) Generate quick-look diagnostics

```bash
PYTHONPATH=src python examples/scripts/visualize_obs_subblock.py \
  --cube Results/observation_subblock/demo_run/obs_subblock_*_cube.fits \
  --manifest Results/observation_subblock/demo_run/manifest.json
```

Default quick-look outputs are written to a `quicklook/` folder alongside the
cube and include `preview.gif`, `summary.png`, and `trace_summary.png`.

### 6) Run registration-only inference on the rendered cube

```bash
PYTHONPATH=src python examples/recipes/observation_subblock_inference.py \
  --config examples/recipes/observation_subblock_inference_template/prescription.yaml
```

## Output layout

Artifacts under `<output_root>/<run_name_or_timestamp>/`:

- `manifest.json`
- `<file_prefix>_<timestamp>_cube.fits`
- `<file_prefix>_<timestamp>_frame_truth.csv`

`manifest.json` includes schema version, generator info, frame count,
varying-key metadata, trace metadata, time bounds, and artifact relative paths.

## Current limits

- only explicit CSV traces are supported
- structural per-frame keys are rejected
- no motion-helper generation inside renderer (use trace-builder recipe)
- single central-field cube only (no multi-ROI payload yet)
