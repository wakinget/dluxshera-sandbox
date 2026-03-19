# Prescribed Monte Carlo Templates

## Purpose
Prescribed Monte Carlo runs use a native experiment config plus an optional per-run plan. The config lives in `prescription.yaml` (real `system` + `experiment` blocks). The plan usually lives in `run_plan.csv` and is referenced from `experiment.monte_carlo.run_plan` inside the prescription.

## Template layout
- `prescription.yaml` — main experiment config
- `run_plan.csv` — per-run override plan (keys in the first column, runs across columns)

Outputs live under the experiment directory:
- `manifest.json` (metadata, notes, run list)
- `results.csv` (aggregate metrics; default orientation = column)
- `runs/<run_id>/...` (per-run artifacts)

## What belongs where?
- **YAML (`prescription.yaml`)**
  - `system`: preset or full system definition.
  - `experiment.monte_carlo`: `n_runs`, `run_id_prefix`, `results_filename`, `results_orientation`, `run_plan`, `reuse_fim`.
  - `experiment.infer_keys`, `experiment.priors`, `experiment.notes`.
  - Top-level experiment controls such as `optimizer`, `eigenmodes`, `noise`, `outputs`, and `init`.
- **CSV (`run_plan.csv`)**
  - Per-run toggles/notes/seeds.
  - Per-run overrides for `truth.*`, `init.*`, `prior.*`, `optimizer.*`, `eigen.*`, `noise.*`, `fim.*`.
  - Keep structural/system edits in YAML, not CSV.

## Config Resolution
- If `system.preset` is set, the script loads that preset first and then deep-merges your YAML `system` block over it.
- Partial overrides are allowed for mapping-style sections such as `system.source`, `system.optics`, and `system.detector`.
- Omitted mapping fields fall back to the preset. For example, if you override only `source.x_position_as`, the other `source` fields still come from the preset.
- If you omit `detector.model` but keep `system.preset`, the detector model falls back to the preset value.
- Lists are not merged item-by-item. If you override `detector.layers`, that entire layer list replaces the preset layer list.
- As a result, `detector.layers` should usually be written in full when you override it. Supplying only one layer does not patch the preset list; it discards the other preset layers.
- The same general rule applies to other list-valued fields: mappings merge, lists replace.
- If `experiment.monte_carlo.run_plan` is a relative path, it resolves relative to `prescription.yaml`, not the repo root.

## Data Vs Inference Systems
- By default, the script uses the resolved top-level `system` both to generate synthetic data and to run inference. This is the no-mismatch case.
- You can provide `experiment.inference_system` when you want the inference model to differ from the data-generating model.
  - If `experiment.inference_system` is omitted entirely, the script copies the fully resolved top-level `system` and uses that for inference.
    - If `experiment.inference_system` is present, it resolves as its own system block. It does not inherit missing fields from the top-level `system`. This means that any non-default `system` fields that you don't intend to change must be present in `experiment.inference_system`.
      - For example, if `system.source.exposure_time_s` is set, and you don't intend to change it, it must be present in `experiment.inference_system`:
        - ```
          system:
            preset: SHERA_FLIGHT_3P
            source:
              exposure_time_s: 180000.0
            detector:
              layers:
                - name: downsample
                  kernel_size: 3
                - name: pixel_offsets
                  dx_path: src/dluxshera/data/pixel_offsets/dx_fpa_realization_01.fits
                  dy_path: src/dluxshera/data/pixel_offsets/dy_fpa_realization_01.fits
          
          experiment:
            inference_system:
              preset: SHERA_FLIGHT_3P
              source:
                exposure_time_s: 180000.0 # <-- this is required
              detector:
                layers:
                  - name: downsample
                    kernel_size: 3        # <-- this too
                  - name: pixel_offsets
                    dx_path: src/dluxshera/data/pixel_offsets/dx_fpa_realization_01.fits
                    dy_path: src/dluxshera/data/pixel_offsets/dy_fpa_realization_01.fits
                    knowledge_error:
                        model: gaussian
                        scale: 1e-3
          ```
- If `experiment.inference_system.preset` is set, omitted mapping fields inside `inference_system` fall back to that preset's values.
- If `experiment.inference_system` is present without its own `preset`, it must be complete enough to resolve as a standalone system.
- The same merge rules apply inside `inference_system` as for the top-level `system`: mappings merge, lists replace.
- This is useful for model-mismatch studies such as using different Noll index sets in the optics model, or using different detector calibration maps or detector-layer knowledge errors during inference.

## Outdir Resolution
- The experiment root is resolved in this order: `--outdir`, then `experiment.outputs.outdir`, then `--run-name`, then the default timestamped `Results/prescribed_mc_<timestamp>`.
- If `experiment.outputs.outdir` is a relative path, it resolves relative to `prescription.yaml`, not the repo root.
- Set `experiment.outputs.outdir: .` to write results into the same directory that contains the prescription file.
- Keep `outputs.outdir` experiment-level. Do not try to steer the experiment root from `run_plan.csv`.

## Quick start
Dry run with templates:
```bash
python examples/recipes/prescribed_monte_carlo.py --dry-run
```

Explicit files:
```bash
python examples/recipes/prescribed_monte_carlo.py \
  --prescription examples/recipes/prescribed_mc_template/prescription.yaml \
  --dry-run
```

Run for real:
```bash
python examples/recipes/prescribed_monte_carlo.py \
  --prescription examples/recipes/prescribed_mc_template/prescription.yaml \
  --outdir Results/my_experiment
```

Use the config-defined outdir:
```bash
python examples/recipes/prescribed_monte_carlo.py \
  --prescription examples/recipes/prescribed_mc_template/prescription.yaml
```

Name the output directory without typing the full path:
```bash
python examples/recipes/prescribed_monte_carlo.py \
  --prescription examples/recipes/prescribed_mc_template/prescription.yaml \
  --run-name my_experiment
```

Aggregate from existing run artifacts only:
```bash
python examples/recipes/prescribed_monte_carlo.py \
  --outdir Results/my_experiment \
  --aggregate-only
```

## Create your own experiment
1. Copy the templates:
   ```bash
   cp -R examples/recipes/prescribed_mc_template Results/my_experiment
   ```
2. Edit `prescription.yaml`:
   - Set `system.preset` or inline your system block.
   - Add `experiment.inference_system` only if you want the inference model to differ from the data-generating system.
   - Fill `experiment.infer_keys` and `experiment.priors`.
   - Tweak `experiment.monte_carlo`, `optimizer`, `eigenmodes`, `noise`, `outputs`, and `init`.
   - Keep `experiment.monte_carlo.run_plan: run_plan.csv` if you want to use the bundled plan template.
   - Set `experiment.outputs.outdir` if you want the experiment root to live in a fixed location.
   - Use `outdir: .` if the prescription file already lives inside the directory where you want outputs written.
   - Set `run_plan: null` or omit the key if you want default-only runs with no CSV plan.
3. Edit `run_plan.csv` for per-run changes.
   - Blank cell = keep defaults. Use `null` to clear.
   - Arrays in CSV must be quoted JSON (e.g., `"[1, 2, 3]"`).
4. Preview with `--dry-run --num-preview 5`, then run without `--dry-run`.

## Notes, orientation, discovery
- Experiment notes: `experiment.notes` in YAML; persisted to `manifest.json`.
- Per-run notes: `note/notes/comment/comments` columns in CSV; persisted to runs/results/manifest.
- `results.csv` orientation: default `col`; use `--results-orientation row` for row-oriented compatibility output.
- If `--outdir` is provided without `--prescription`, the script looks for `prescription.*` (yaml/yml/json) under that directory. This discovery step only helps locate the prescription; `--outdir` still wins as the experiment root once execution starts.
- If `--prescription` is provided and `experiment.outputs.outdir` is set, you can omit `--outdir` and let the prescription control the experiment root.

## CSV override hints
- `run_id`, `enabled`, `seed`, `note`/`comment` supported.
- `prior.<infer_key>.sigma|dist` applies when `init.mode` resolves to `prior`.
- If `run_plan` is disabled, the script still runs `experiment.monte_carlo.n_runs` default-only runs.
- Keep `system.*` or other structural changes out of CSV; place them in YAML.
