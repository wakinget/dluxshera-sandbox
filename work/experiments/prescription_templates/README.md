# Prescribed Monte Carlo Templates

## Purpose
Prescribed Monte Carlo in this repo means an experiment-wide **prescription** plus a per-run **plan** that drives repeated optimizations. You define a single `prescription.json` for the experiment and an `overrides.csv` that enumerates the run-specific inputs. The runner then executes each run with consistent model/config defaults and per-run overrides.

## Template layout
A template directory contains:
- `prescription.json`
- `overrides.csv` (transposed matrix format; **keys-as-rows, runs-as-columns**)

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
  - `defaults.fim.reuse_fim` controls whether FIM reuse is forced even when the strict cache key misses.

3) **Edit `overrides.csv`** (per-run inputs)

- Transposed format: **first column = keys**, each remaining column = a run.
- Blank cells mean “no override for this run.”
- Use `null` (literal text) to set an explicit JSON null.
- Vector-valued cells must be **JSON arrays inside quoted CSV cells** (e.g., `"[0.1, 0.2, 0.3]"`).
  - Why: CSV uses commas as delimiters, so unquoted arrays can be split across multiple columns.

4) **Dry-run preview**

```bash
python work/experiments/prescribed_monte_carlo.py \
  --prescription work/experiments/my_experiment/prescription.json \
  --overrides work/experiments/my_experiment/overrides.csv \
  --dry-run
```

5) **Run for real**

```bash
python work/experiments/prescribed_monte_carlo.py \
  --prescription work/experiments/my_experiment/prescription.json \
  --overrides work/experiments/my_experiment/overrides.csv \
  --outdir Results/my_experiment
```

6) **Inspect outputs**
- `Results/my_experiment/results.csv` for run-level summaries
- `Results/my_experiment/runs/<run_id>/` for per-run artifacts
- `Results/my_experiment/manifest.json` for the resolved configuration and plan

## Plan formats (brief)
The default template uses the **transposed** format (keys-as-rows, runs-as-columns). The legacy wide format is no longer part of the templates.

## Auto-discovery
When you pass `--outdir` without explicit `--prescription` or `--overrides`, the script scans the output directory for:
- **Prescription candidates**: JSON files whose names contain `prescription` (case-insensitive).
- **Overrides candidates**: CSV files whose names contain `overrides` (case-insensitive), with `overrides.csv` treated as the preferred naming convention.

If multiple candidates are found, you must disambiguate by passing explicit paths.

Resolution behavior:
- If a prescription is found but no overrides file is found, the run proceeds with **no per-run overrides** (no template overrides fallback).
- If an overrides file is found but no prescription is found, the script raises an error.
- If neither is found, the script warns and falls back to **both** templates.
- Supplying `--prescription` without `--overrides` explicitly also runs with no per-run overrides.

## Key semantics / policies
- Structural config overrides are **experiment-wide**. Do **not** put `model.*` or `overrides.config.*` in the plan.
- Each run gets **one seed**; JAX splits it internally for stochastic components.
  - If a plan row does **not** specify `seed`, the runner derives a per-run seed
    by folding the run index into the prescription default seed (deterministic
    and reproducible across runs).
- `experiment.n_runs` is **authoritative** when set:
  - If the plan defines fewer runs, the remaining runs execute with prescription defaults (no per-run overrides).
  - If the plan defines more runs, extra plan-defined runs are ignored.
  - `--dry-run` prints the resolved run set (after padding/truncation).
- `init.mode`:
  - `prior`: samples around truth using priors.
  - `explicit`: only uses explicitly provided init values; missing values follow normal resolution/derived refresh.
- Per-run prior overrides (overrides.csv):
  - Use keys `prior.<infer_key>.sigma` or `prior.<infer_key>.dist` to override the
    prescription priors for a single run without changing `init.mode`.
  - These overrides apply **only** when `init.mode` resolves to `prior`.
- Vector-valued sigmas should be quoted JSON arrays (e.g., `"[1, 1, 1]"`). Scalar sigmas will broadcast
    across vector-valued parameters.
  - Examples:
    - Scalar sigma: `prior.binary.x_position_as.sigma = 0.05`
    - Vector sigma: `prior.primary.zernike_coeffs_nm.sigma = [1, 1, 1, 1, 1, 1, 1, 1]`
    - Dist override: `prior.binary.contrast.dist = LogNormal`
- `fim.reuse_fim`:
  - `false` (default): reuse only when the strict FIM cache key matches.
  - `true`: reuse the most recent cached FIM even when the strict cache key misses (with a warning).
- Paths like `diffractive_pupil_path` are **repo-root-relative**.
- Vector cells in `overrides.csv` must be **quoted JSON arrays**.

### CSV + JSON array nuance (important)
- The plan parser accepts JSON arrays for vector overrides, but CSV parsing happens first.
- If you write an unquoted array like `[1, 2, 3]` in a CSV cell, commas may be interpreted as column separators.
- Always quote vector arrays so the full JSON value stays in a single cell:
  - ✅ Correct: `"[1, 2, 3]"`
  - ❌ Risky: `[1, 2, 3]`

## Troubleshooting
- **Strict config validation**: a typo in any override key will error out.
- **Malformed JSON arrays**: ensure vector cells are valid JSON (e.g., `[1, 2, 3]`).
- **Forbidden overrides keys**: `model.*` or `overrides.config.*` in the overrides file are rejected.
- **Zernike length mismatches**: ensure vector lengths match configured Noll indices.

## Future note
This workflow may later move from `work/experiments` to `examples/recipes`, but the copy/edit/run pattern will remain.
