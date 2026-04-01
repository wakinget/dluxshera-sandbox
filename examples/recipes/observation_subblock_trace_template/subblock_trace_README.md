# Observation Sub-Block Trace Template

## Purpose

This template generates canonical explicit trace CSV files for
`examples/recipes/observation_subblock.py`.

Recommended workflow:

1. generate trace CSV with `subblock_trace_generation.py`
2. render cube with `observation_subblock.py`

## Files

- `subblock_trace_prescription.yaml`: generalized trace-generation example

## Path resolution

Relative `experiment.outputs.outdir` values are resolved relative to the
prescription file. That makes it easy to keep a trace prescription inside a
subblock work folder and emit outputs into a sibling `trace/` directory.

## Config shape

Set:

- `experiment.kind: subblock_trace_generation`
- `experiment.trace`:
  - `n_frames` (required)
  - `dt_s` (required)
  - `varying_keys` (required list of key strings)
  - `plan` (required mapping keyed by each varying key)

Optional:

- `experiment.seed` or `experiment.trace.seed`
- `system` + `experiment.truth` (required when any varying key omits `base`)
- `experiment.outputs.{outdir,file_prefix,write_manifest}`

## Supported varying keys

Scalar keys:

- `source.x_position_as`
- `source.y_position_as`
- `source.position_angle_deg`
- `source.separation_as`
- `source.contrast`
- `source.log_flux_total`
- `optics.plate_scale_as_per_pix`

Indexed keys:

- `optics.primary.zernike_coeffs_nm[i]`
- `optics.secondary.zernike_coeffs_nm[i]`

Use one syntax everywhere: `base.path` or `base.path[index]`.

## Anchor and effects semantics

For each varying key:

- anchor = `base` if provided
- otherwise anchor = resolved/refreshed nominal store value
- trace value per frame = `anchor + sum(effects)`

Effects supported in `plan.<key>.effects`:

- `constant_offset`: `offset`
- `linear_drift`: `start`, `rate_per_s`
- `random_walk`: `start`, `sigma_step`
- `iid_jitter`: `center`, `sigma`
- `explicit`: `values` (length must equal `n_frames`)

Effect outputs are additive and order-independent in meaning.

## Usage

Generate trace:

```bash
PYTHONPATH=src python examples/recipes/subblock_trace_generation.py \
  --config examples/recipes/observation_subblock_trace_template/subblock_trace_prescription.yaml
```

Validate only:

```bash
PYTHONPATH=src python examples/recipes/subblock_trace_generation.py \
  --config examples/recipes/observation_subblock_trace_template/subblock_trace_prescription.yaml \
  --dry-run
```

## Output contract

Artifacts:

- `<file_prefix>_<timestamp>_frame_truth.csv`
- `manifest.json` (unless `write_manifest: false`)

The trace manifest includes:

- source `config_path`
- normalized `trace_spec`
- resolved anchor values
- optional resolved `system` config snapshot
- `shared_truth` overrides used when anchors come from the system

CSV columns:

- always: `frame_index`, `time_s`
- plus: one column per configured `varying_keys`

## Next step: render

Set the renderer trace path:
`experiment.observation_subblock.trace.path: <generated_csv>`, then run:

```bash
PYTHONPATH=src python examples/recipes/observation_subblock.py \
  --config <renderer_prescription.yaml>
```

Recommended stage layout for repeated subblocks:

```text
Results/<campaign>/<subblock_id>/
  trace/
  render/
  inference/
```

With the current CLI, one simple convention is:

- trace run: `--results-dir Results/<campaign>/<subblock_id> --run-name trace`
- render run: `--results-dir Results/<campaign>/<subblock_id> --run-name render`
- inference run: `--results-dir Results/<campaign>/<subblock_id> --run-name inference`
