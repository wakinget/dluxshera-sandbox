# Prescribed Monte Carlo Templates

## Purpose
Prescribed Monte Carlo in this repo means an experiment-wide **prescription** plus a per-run **plan** that drives repeated optimizations. You define a single `prescription.json` for the experiment and a `plan.csv` that enumerates the run-specific inputs. The runner then executes each run with consistent model/config defaults and per-run overrides.

## Template layout
A template directory contains:
- `prescription.json`
- `plan.csv` (transposed matrix format; **keys-as-rows, runs-as-columns**)

Outputs go to an experiment output directory:
- `manifest.json` (run metadata and configuration used)
- `results.csv` (summary of run outputs)
- `runs/<run_id>/...` (per-run meta/summary/trace artifacts)

## Quick start
1) **Copy a template directory**

```bash
cp -R work/experiments/prescription_templates work/experiments/my_experiment
```

2) **Edit `prescription.json`** (experiment-wide settings)

Key sections you will typically touch:
- `model` / `config` (base model/config for all runs)
- `overrides.config` (experiment-wide config overrides)
- `overrides.store` (experiment-wide data store overrides)
- `infer_keys`
- `priors`
- `defaults`

3) **Edit `plan.csv`** (per-run inputs)

- Transposed format: **first column = keys**, each remaining column = a run.
- Blank cells mean “no override for this run.”
- Use `null` (literal text) to set an explicit JSON null.
- Vector-valued cells must be **JSON arrays** (e.g., `[0.1, 0.2, 0.3]`).

4) **Dry-run preview**

```bash
python work/experiments/prescribed_monte_carlo.py \
  --prescription work/experiments/my_experiment/prescription.json \
  --plan work/experiments/my_experiment/plan.csv \
  --dry-run
```

5) **Run for real**

```bash
python work/experiments/prescribed_monte_carlo.py \
  --prescription work/experiments/my_experiment/prescription.json \
  --plan work/experiments/my_experiment/plan.csv \
  --outdir Results/my_experiment
```

6) **Inspect outputs**
- `Results/my_experiment/results.csv` for run-level summaries
- `Results/my_experiment/runs/<run_id>/` for per-run artifacts
- `Results/my_experiment/manifest.json` for the resolved configuration and plan

## Plan formats (brief)
The default template uses the **transposed** format (keys-as-rows, runs-as-columns). The legacy wide format is no longer part of the templates.

## Auto-discovery
When you pass `--outdir` without explicit `--prescription` or `--plan`, the script scans the output directory for:
- **Prescription candidates**: JSON files whose names contain `prescription` (case-insensitive).
- **Plan candidates**: CSV files whose names contain `plan` (case-insensitive), with `plan.csv` treated as the preferred naming convention.

If multiple candidates are found, you must disambiguate by passing explicit paths. If none are found, the script falls back to these templates and prints a warning.

## Key semantics / policies
- Structural config overrides are **experiment-wide**. Do **not** put `model.*` or `overrides.config.*` in the plan.
- Each run gets **one seed**; JAX splits it internally for stochastic components.
- `init.mode`:
  - `prior`: samples around truth using priors.
  - `explicit`: only uses explicitly provided init values; missing values follow normal resolution/derived refresh.
- Paths like `diffractive_pupil_path` are **repo-root-relative**.
- Vector cells in `plan.csv` must be **JSON arrays**.

## Troubleshooting
- **Strict config validation**: a typo in any override key will error out.
- **Malformed JSON arrays**: ensure vector cells are valid JSON (e.g., `[1, 2, 3]`).
- **Forbidden plan keys**: `model.*` or `overrides.config.*` in the plan are rejected.
- **Zernike length mismatches**: ensure vector lengths match configured Noll indices.

## Future note
This workflow may later move from `work/experiments` to `examples/recipes`, but the copy/edit/run pattern will remain.
