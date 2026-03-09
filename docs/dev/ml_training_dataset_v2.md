# ML training dataset generator V2

`work/experiments/generate_training_dataset_v2.py` adds a richer one-parameter Fisher-scaled sweep workflow while keeping V1 unchanged.

## Key behavior
- V2 is a separate script; V1 (`generate_training_dataset.py`) remains the baseline.
- V2 emits exactly one nominal sample at the beginning of the run.
- Per-parameter sweeps are nonzero only.
- Log spacing is applied in sigma space, then mapped to parameter deltas using Fisher-derived parameter sigma.
- `n_magnitudes` means the positive-side count only; negatives are mirrored automatically.
- Total nonzero samples per swept component are `2 * n_magnitudes`.
- Fisher-derived per-parameter/component sigma values are printed before emission and persisted to run-level metadata.

## Quick usage examples

### 1) Run with defaults
This uses the script defaults for sweep configuration:

```bash
PYTHONPATH=src python work/experiments/generate_training_dataset_v2.py
```

### 2) Set output location and run name
```bash
PYTHONPATH=src python work/experiments/generate_training_dataset_v2.py \
  --outdir Results/custom_runs \
  --run-name sweep_v2_demo
```

### 3) Change global sweep defaults
These values apply to all inference keys unless overridden:

```bash
PYTHONPATH=src python work/experiments/generate_training_dataset_v2.py \
  --min-sigma 0.05 \
  --max-sigma 20.0 \
  --n-magnitudes 10 \
  --spacing log
```

### 4) Exclude secondary Zernikes
```bash
PYTHONPATH=src python work/experiments/generate_training_dataset_v2.py \
  --exclude-secondary-zernikes
```

### 5) Use per-parameter overrides from JSON file
```bash
PYTHONPATH=src python work/experiments/generate_training_dataset_v2.py \
  --sweep-config-json work/experiments/sweep_config_v2_template.json
```

### 6) Inline JSON overrides (no file)
```bash
PYTHONPATH=src python work/experiments/generate_training_dataset_v2.py \
  --sweep-config-json '{
    "parameters": {
      "binary.contrast": {
        "min_sigma": 0.01,
        "max_sigma": 5.0,
        "n_magnitudes": 6,
        "spacing": "log"
      }
    }
  }'
```

## Per-parameter config template
A reusable template is provided at:

- `work/experiments/sweep_config_v2_template.json`

Supported per-key fields:
- `min_sigma`
- `max_sigma`
- `n_magnitudes`
- `spacing` (currently `log`)

Config shape (what `--sweep-config-json` expects):

```json
{
  "parameters": {
    "binary.separation_as": {
      "min_sigma": 0.1,
      "max_sigma": 10.0,
      "n_magnitudes": 8,
      "spacing": "log"
    }
  }
}
```

## Metadata notes
- `manifest.json` records script/version identity, selected parameters, sweep configuration, Fisher sigma summary, and sample counts.
- `samples.jsonl` and sidecar JSON files include sweep fields like `sigma_offset`, `parameter_sigma`, `delta_value`, and `parameter_value`.
- Nominal sample records `is_nominal=true`; perturbed samples record `is_nominal=false` and exactly one swept parameter.
