# Prescribed Monte Carlo Templates

## Purpose
Prescribed Monte Carlo runs use a native experiment config plus a per-run plan. The config lives in `prescription.yaml` (real `system` + `experiment` blocks). The plan lives in `overrides.csv` (per-run overrides).

## Template layout
- `prescription.yaml` — main experiment config
- `overrides.csv` — per-run override plan (keys in the first column, runs across columns)

Outputs live under the experiment directory:
- `manifest.json` (metadata, notes, run list)
- `results.csv` (aggregate metrics; default orientation = column)
- `runs/<run_id>/...` (per-run artifacts)

## What belongs where?
- **YAML (`prescription.yaml`)**
  - `system`: preset or full system definition.
  - `experiment.infer_keys`, `experiment.priors`, `experiment.notes`.
  - `experiment.prescribed_mc`: `n_runs`, `run_id_prefix`, `results_filename`, `results_orientation`, and `defaults` (seed, truth/init/optimizer/eigen/noise/fim/outputs).
- **CSV (`overrides.csv`)**
  - Per-run toggles/notes/seeds.
  - Per-run overrides for `truth.*`, `init.*`, `prior.*`, `optimizer.*`, `eigen.*`, `noise.*`, `fim.*`.
  - Keep structural/system edits in YAML, not CSV.

## Quick start
Dry run with templates:
```bash
python examples/recipes/prescribed_monte_carlo.py --dry-run
```

Explicit files:
```bash
python examples/recipes/prescribed_monte_carlo.py \
  --prescription examples/recipes/prescribed_mc_template/prescription.yaml \
  --overrides examples/recipes/prescribed_mc_template/run_plan.csv \
  --dry-run
```

Run for real:
```bash
python examples/recipes/prescribed_monte_carlo.py \
  --prescription examples/recipes/prescribed_mc_template/prescription.yaml \
  --overrides examples/recipes/prescribed_mc_template/run_plan.csv \
  --outdir Results/my_experiment
```

## Create your own experiment
1. Copy the templates:
   ```bash
   cp -R examples/recipes/prescribed_mc_template Results/my_experiment
   ```
2. Edit `prescription.yaml`:
   - Set `system.preset` or inline your system block.
   - Fill `experiment.infer_keys` and `experiment.priors`.
   - Tweak `experiment.prescribed_mc.defaults` for seed, truth, init mode, optimizer, eigen, noise, fim, outputs.
3. Edit `overrides.csv` for per-run changes.
   - Blank cell = keep defaults. Use `null` to clear.
   - Arrays in CSV must be quoted JSON (e.g., `"[1, 2, 3]"`).
4. Preview with `--dry-run`, then run without it.

## Notes, orientation, discovery
- Experiment notes: `experiment.notes` in YAML; persisted to `manifest.json`.
- Per-run notes: `note/notes/comment/comments` columns in CSV; persisted to runs/results/manifest.
- `results.csv` orientation: default `col`; override via `experiment.prescribed_mc.results_orientation` or `--results-orientation row`.
- If `--outdir` is provided without explicit files, the script looks for `prescription.*` (yaml/yml/json) and `overrides*.csv` under that directory. Multiple matches require explicit paths.

## CSV override hints
- `run_id`, `enabled`, `seed`, `note`/`comment` supported.
- `prior.<infer_key>.sigma|dist` applies when `init.mode` resolves to `prior`.
- Keep `system.*` or other structural changes out of CSV; place them in YAML.
