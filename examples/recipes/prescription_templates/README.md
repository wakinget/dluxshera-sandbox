# Prescribed Monte Carlo Templates

## Purpose
Prescribed Monte Carlo in this repo means an experiment-wide **prescription** plus a per-run **plan** that drives repeated optimizations. You define a single `prescription.json` for the experiment and an `overrides.csv` that enumerates the run-specific inputs. The runner then executes each run with consistent model/config defaults and per-run overrides.

## Template layout
A template directory contains:
- `prescription.json`
- `overrides.csv` (keys defined down rows, each run as a column)

Outputs go to an experiment output directory:
- `manifest.json` (run metadata and configuration used)
- `results.csv` (summary of run outputs)
- `runs/<run_id>/...` (per-run meta/summary/trace artifacts)

### Aggregate schema notes
- `results.csv` includes run metadata columns such as `run_id`, `status`, `created_at`,
  `run_note`, `plan_label`, and `seed` before optimizer/noise fields and parameter columns.
- `manifest.json` includes top-level `notes` from `experiment.notes` (aliases accepted: `experiment.note`, `experiment.comment`, `experiment.comments`).
- `manifest.json` also includes one record per run under `runs[]`; when a run summary contains
  `run_note`, it is surfaced in that run record for quick annotation lookups.

## Quick Start

Call the script using prescription templates (dry run). Calling the script without any additional arguments defaults to using the template prescription and overrides files. The `--dry-run` argument is used to parse the inputs, and display a preview, but halts the script before running the experiment.

```bash
python examples/recipes/prescribed_monte_carlo.py --dry-run
```

You can specify that you want to use a specific prescription file with the `--prescription` argument. This will run the experiment with **no overrides**, relying only on the default values within `prescription.json`.

```bash
python examples/recipes/prescribed_monte_carlo.py \
  --prescription examples/recipes/prescription_templates/prescription.json \
  --dry-run
```

You can additionally specify any per-run overrides that you want to use with the `--overrides` argument:

```bash
python examples/recipes/prescribed_monte_carlo.py \
  --prescription examples/recipes/prescription_templates/prescription.json \
  --overrides examples/recipes/prescription_templates/overrides.csv \
  --dry-run
```

Finally, you can also specify an explicit `--outdir` for the saved results:

```bash
python examples/recipes/prescribed_monte_carlo.py \
  --prescription examples/recipes/prescription_templates/prescription.json \
  --overrides examples/recipes/prescription_templates/overrides.csv \
  --outdir Results/my_experiment \
  --dry-run
```

## Creating your own experiment

### 1) Copy the templates to a new experiment directory

```bash
cp -R examples/recipes/prescription_templates Results/my_experiment
```

### 2) Edit `prescription.json` (experiment-wide settings)

Key sections you will typically touch:
- `model` / `config` (base model/config for all runs)
- `overrides.config` (experiment-wide config overrides)
- `overrides.store` (experiment-wide data store overrides)
- `infer_keys`
- `priors`
- `defaults`
  - `defaults.fim.reuse_fim` controls forced FIM reuse even when the cache doesn't match.

### 3) Edit `overrides.csv` (per-run inputs)

- Transposed format: **first column = keys**, each remaining column = a run.
- Blank cells mean “no override for this run.”
- Use `null` (literal text) to set an explicit JSON null (resolves to `None` in python)
- Vector-valued cells in CSV files must be **quoted JSON arrays** (e.g., `"[0.1, 0.2, 0.3]"`).
  - CSV uses commas as delimiters, so unquoted arrays can be split across multiple columns.

### 4) Dry-run preview

Use the `--dry-run` argument to ensure that the input files load correctly. Prescription and override files can be auto-discovered if present within the provided `--outdir`.

Using auto-discovery:
```bash
python examples/recipes/prescribed_monte_carlo.py \
  --outdir Results/my_experiment \
  --dry-run
```

Using explicit prescription/overrides files:
```bash
python examples/recipes/prescribed_monte_carlo.py \
  --prescription Results/my_experiment/prescription.json \
  --overrides Results/my_experiment/overrides.csv \
  --dry-run
```
An output directory is automatically generated if `--outdir` isn't provided.

### 5) Run for real

Just remove the `--dry-run` argument (either syntax from above should work):

```bash
python examples/recipes/prescribed_monte_carlo.py \
  --outdir Results/my_experiment
```

### 6) **Inspect outputs**
- `Results/my_experiment/results.csv` for run-level summaries
- `Results/my_experiment/runs/<run_id>/` for per-run artifacts
- `Results/my_experiment/manifest.json` for the resolved configuration and plan


## Auto-discovery
When you pass `--outdir` without explicit `--prescription` or `--overrides`, the script scans the output directory for:
- **Prescription candidates**: JSON files whose names contain `prescription` (case-insensitive).
- **Overrides candidates**: CSV files whose names contain `overrides` (case-insensitive), with `overrides.csv` treated as the preferred naming convention.

If multiple candidates are found, you must disambiguate by passing explicit paths.

Resolution behavior:
- If a prescription is found but no overrides file is found, the run proceeds with **no per-run overrides** (no template overrides fallback).
- If an overrides file is found but no prescription is found, the script raises an error.
- If neither is found, the script warns and falls back to **both** templates.
- Supplying `--prescription` without `--overrides` runs with no per-run overrides.

## Key semantics / policies
- Prescription JSON supports **private keys** that start with `_` anywhere in the object tree.
  - Private keys are recursively stripped immediately after `json.load`, so runtime parsing/validation never sees them.
  - Use `_comment` freely for inline template notes.
  - Use underscore-prefixed keys (for example `_bandwidth_m` or `_binary.x_position_as`) to keep optional overrides in place but disabled by default.
  - Only a **leading** underscore is special; regular keys like `run_id_prefix` remain active.
  - JSON does not allow meaningful duplicate keys in the same object (parsers keep only the last one), so do not rely on repeated `_comment` keys side-by-side.
- Structural config overrides are **experiment-wide**. Do **not** put `model.*` or `overrides.config.*` in the overrides.
- Each run gets **one seed**; JAX splits it internally for stochastic components.
  - If a plan row does **not** specify `seed`, the runner derives a per-run seed
    by folding the run index into the prescription default seed (deterministic
    and reproducible across runs).
- Notes fields:
  - **Experiment-level note**: put this in `experiment.notes` (recommended) in `prescription.json`; aliases `experiment.note`, `experiment.comment`, and `experiment.comments` are accepted for compatibility. This is persisted once in `manifest.json` as top-level `notes`.
  - **Per-run note**: put `note` / `notes` / `comment` / `comments` in `overrides.csv`; these are persisted per run as `run_note` in summaries, `results.csv`, and `manifest.json.runs[]`.
- Paths are usually specified **relative to the repo-root**.
  - `examples/recipes/prescribed_monte_carlo.py`
- `experiment.n_runs` is **authoritative** when set:
  - If the plan defines fewer runs, the remaining runs execute with prescription defaults (no per-run overrides).
  - If the plan defines more runs, extra plan-defined runs are ignored.
  - `--dry-run` prints the resolved run set (after padding/truncation).
- `init.mode`:
  - `prior`: samples around truth using priors.
  - `explicit`: only uses explicitly provided init values; missing values follow normal resolution/derived refresh.
  - In `explicit` mode, precedence for the common `imaging.exposure_time_s` / `binary.log_flux_total` pair is:

    | Inputs provided in `init.*` | Resolved `binary.log_flux_total` |
    | --- | --- |
    | `imaging.exposure_time_s` only | Recomputed by transform from primitives (updates with exposure time). |
    | `binary.log_flux_total` only | Uses the explicit `binary.log_flux_total` value. |
    | Both | Explicit `binary.log_flux_total` wins (applied after `refresh_derived`). |
- **Per-run prior overrides (overrides.csv):**
  - Vector cells in `overrides.csv` must be **quoted** JSON arrays.
  - First row: `key, run_001, run_002, run_003, etc.`
    - The first row 
  - `run_id`: Overrides for the name of each run directory. If not specified, a `run_id` is auto-generated using the prescription's `run_id_prefix`
  - `enabled`: True/False flag for disabling particular runs. Defaults to True (enabled) if blank or missing.
  - `note`: Individual note field for each run, saved in the aggregated results file.
  - `seed`: Specific seed to be used in each run. Seeds are incremented automatically by default unless explicitly set.
  - `init.mode`: Used to override the init mode for any particular run. Used to force explicit init values that would otherwise be drawn from priors, or conversely to draw init values from priors when they would otherwise be set explicitly.
    - `explicit`: All init values are set explicitly. Any parameters left blank default to the truth store value.
    - `prior`: Init values are drawn from priors, except where explicitly set.
  - `optimizer.n_iter`: Per-run override for optimizer iteration count.
    - Use this to shorten exploratory runs or extend convergence for selected runs without changing experiment-wide defaults.
  - `optimizer.base_lr`: Per-run override for optimizer learning rate.
    - Useful for quick sweeps over step size to assess stability or convergence speed.
  - `eigen.use_eigen`: Per-run True/False toggle for eigen-truncation behavior in the optimizer/preconditioner path.
    - Set `false` to run without eigen truncation for comparison runs.
  - `eigen.truncate_k`: Per-run override for fixed-rank eigen truncation.
    - Typically an integer rank; leave blank to use the prescription default.
  - `eigen.truncate_by_eigval`: Per-run threshold for eigenvalue-based truncation.
    - Use this instead of `eigen.truncate_k` when you want value-threshold truncation rather than a fixed rank.
  - `noise.add_noise`: Per-run True/False toggle for simulated noise injection.
    - Useful when pairing noisy and noise-free runs under otherwise identical settings.
  - `fim.reuse_fim`: Configures FIM reuse behavior. The FIM is calculated and cached for potential reuse.
    - `false` (default): Recomputes FIM if any key differs from previously cached FIM (cache miss). If all inputs are constant, the cache will 'hit' and the cached FIM is reused.
    - `true`: Always reuses the most recent cached FIM even when the strict cache key misses (issues a warning for clarity).
  - `truth.*`: Used to override the true data value for any particular run.
    - Certain keys may depend on each other and can conflict if both are overridden. `imaging.exposure_time_s` and `binary.log_flux_total` are good examples. The overwrite behavior depends on the spec used by `refresh_derived`: if `binary.log_flux_total` is derived in that active spec it is recomputed from primitives, but if you explicitly set `truth.binary.log_flux_total` after refresh that explicit value wins.
  - `init.*`: Used to override the initial value in the model for any particular run.
  - `prior.*`: Used to override default priors for any particular run.
    - Use `prior.<infer_key>.sigma` or `prior.<infer_key>.dist` to override the default priors for a single run without changing `init.mode`.
    - These overrides apply **only** when `init.mode` resolves to `prior` (otherwise init values are set explicitly).
    - Vector-valued sigmas should be **quoted** JSON arrays. Scalar sigmas will broadcast across vector-valued parameters.
      - Examples (single comma separated pair):
        - Scalar sigma: `prior.binary.x_position_as.sigma, 1e-2`
        - Scalar sigma (broadcast to length of vector): `prior.primary.zernike_coeffs_nm.sigma, 1.0`
        - Vector sigma (note the quotes): `prior.primary.zernike_coeffs_nm.sigma, "[1, 2, 3, 4, 1, 2, 3, 4]"`
        - Dist override: `prior.binary.contrast.dist, LogNormal`

### CSV + JSON Array Nuance
- The plan parser accepts JSON arrays for vector overrides, but CSV parsing happens first.
- In the `overrides.csv`, you must wrap arrays with quotes like `"[1, 2, 3]"`, otherwise commas may be interpreted as column separators.
- Annoyingly, the `prescription.json` file *only* accepts bare arrays like `[1, 2, 3]`, and quoted arrays will error.
- CSV Files:
  - ✅ `"[1, 2, 3]"`
- JSON Files:
  - ✅ `[1, 2, 3]`

## Troubleshooting
- **Strict config validation**: a typo in any override key will error out.
- **Malformed JSON arrays**: ensure vector cells are valid JSON (e.g., `[1, 2, 3]`).
- **Forbidden overrides keys**: `model.*` or `overrides.config.*` in the overrides file are rejected.
- **Zernike length mismatches**: ensure vector lengths match configured Noll indices.
