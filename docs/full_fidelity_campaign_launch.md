# Full-Fidelity Campaign Launch

This is the recommended launch path for full-fidelity binary iterative campaign
configs.

## Checklist

1. Edit the source YAML.

   Start from a source config with:

   ```yaml
   experiment:
     kind: full_fidelity_binary_iterative
   ```

   For cadence sweeps, edit `experiment.run_name`,
   `experiment.iterative.windows_per_draw`, and
   `experiment.iterative.subblocks_per_window`. Do not hand-edit stale realized
   duplicates such as `subblocks.n_subblocks` or
   `iterative_forecast.actual_windows` unless they exactly match the canonical
   cadence.

2. Run the cheap schema audit on a login node.

   ```bash
   PYTHONPATH=src python examples/scripts/audit_campaign_config_schema.py \
     --config examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_info_damped_detector_ke_projected_30min_v1.yaml
   ```

   This only parses YAML and reports resolved cadence, forecast fields, and
   obvious contradictions. It does not build plans, render models, write
   templates, or run subblock preflights.

3. Prepare shards.

   Use the parent results directory, not a directory already ending in
   `observation_bias_campaign`:

   ```bash
   PYTHONPATH=src python examples/scripts/prepare_full_fidelity_campaign_shards.py \
     --config path/to/full_fidelity.yaml \
     --outdir path/to/shards \
     --run-name-prefix my_campaign \
     --mode draw \
     --results-root /projects/.../dLuxShera-Results \
     --time 12:00:00 \
     --cpus-per-task 10 \
     --mem 128G \
     --max-workers 5
   ```

   Use `--mode condition` for fewer larger jobs and `--mode draw` for one job
   per condition/draw.

4. Inspect the manifest.

   ```bash
   PYTHONPATH=src python examples/scripts/audit_campaign_config_schema.py \
     --config path/to/full_fidelity.yaml \
     --check-shard-manifest path/to/shards/shard_manifest.csv
   ```

   Check row count, expected subblocks, expected windows, resources, run roots,
   and doubled `observation_bias_campaign/observation_bias_campaign` paths.

5. Review Slurm commands.

   Generated submit scripts use plain `sbatch`. If your environment requires
   Edge submission, add `-M edge` to the generated `sbatch` commands before
   launch.

6. Submit from a compute-appropriate environment.

   ```bash
   cd /path/to/dluxshera-sandbox
   ./path/to/shards/submit_draw_shards.sh
   ```

   The submit script sets `RESULTS_ROOT`, `CONFIG`, `RUN_NAME`, and
   `MAX_WORKERS` for each shard. The sbatch wrapper derives:

   ```text
   RUN_RESULTS_ROOT="$RESULTS_ROOT/observation_bias_campaign"
   ```

   and passes that directory to the full-fidelity wrapper.

7. Check progress.

   ```bash
   ./path/to/shards/summarize_shard_status.sh
   ```

   This reads expected shard roots and reports plan, summary, status, and
   completed subblock counts.

8. Interpret review warnings separately from science completion.

   The sbatch wrapper may run the full-fidelity review analyzer after the
   campaign finishes. Required science artifacts still control strict analyzer
   success, but optional review plots are best-effort. Degenerate plot inputs
   such as empty, all-NaN, single-value, or constant forecast columns are
   recorded in `analysis/full_fidelity_review/review_warnings.json` and should
   not invalidate a shard whose campaign summary and subblock status show
   complete science output.

9. Use the standard family-analysis contract for post-run summaries.

   See
   `docs/dev/notes/full_fidelity_campaign_family_analysis_standard.md`. Family
   summaries should use actual realized final separation error as the headline
   result; projected 30-minute endpoints are diagnostics unless the campaign is
   explicitly testing projections. Report signed bias metrics
   (`mean_final_sep_err_uas`, `std_final_sep_err_uas`,
   `sem_final_sep_err_uas`, `mad_final_sep_err_uas`) separately from achieved
   absolute-error magnitude (`mean_abs_final_sep_err_uas`,
   `median_abs_final_sep_err_uas`). Preserve axes such as target, low-order WFE
   condition, KE amplitude, and detector calibration term unless a row is
   explicitly labeled as pooled.

10. Choose rendered-FITS retention deliberately.

   Existing configs that omit the setting keep rendered cube/variance FITS.
   Future long iterative runs can opt in to pruning completed windows:

   ```yaml
   experiment:
     subblocks:
       render_retention: delete_after_window
   ```

   The runner preserves all rendered FITS while a window is incomplete. After a
   window has valid subblock summaries, a valid `posterior_by_label.csv`,
   `science_summary.csv`, `iterative_window_diagnostics.csv`, and a successful
   `iterative_reference_update.json`, it may delete only `*_cube.fits` and
   `*_variance.fits` from that window's subblock `render/` directories. Schur
   summaries/matrices, manifests, frame truth, configs, diagnostics, logs,
   posterior products, status/progress files, and analysis outputs remain.

## Edge Launch Checklist

Use this checklist before launching full-fidelity shards on Gattaca2 Edge:

- Submit Edge-side jobs with `sbatch -M edge`.
- Do not rely on `/scratch` for paths that must refer to one specific side's
  filesystem. On Gattaca2, `/scratch` resolves to `/scratch-jpl` on JPL/default
  nodes and to `/scratch-edge` on Edge nodes. `/scratch-jpl` and
  `/scratch-edge` are independent filesystems and are not mirrored.
- If a Conda environment was created on the JPL/default side under `/scratch`,
  Edge jobs must reference it through `/scratch-jpl`, not `/scratch`.
- For the current dLuxShera environment, Edge jobs should use
  `/scratch-jpl/shera_hpc/dmckeith/conda/envs/dluxshera-py311`, not
  `/scratch/shera_hpc/dmckeith/conda/envs/dluxshera-py311`, and not only
  `conda activate dluxshera-py311`.
- Use the shared Miniforge initialization and activate by explicit prefix:

  ```bash
  source /cm/shared/apps/miniforge/etc/profile.d/conda.sh
  conda activate /scratch-jpl/shera_hpc/dmckeith/conda/envs/dluxshera-py311
  export PYTHONPATH="${PYTHONPATH:-src}"
  ```

- Print environment diagnostics at the start of Edge sbatch jobs.

Use this block exactly so the here-doc delimiter stays at column zero:

```bash
echo "Conda env: ${CONDA_DEFAULT_ENV:-unset}"
echo "CONDA_PREFIX: ${CONDA_PREFIX:-unset}"
echo "Python executable: $(which python)"
python - <<'PYENV'
import sys
print("sys.executable:", sys.executable)
import jax
print("jax:", jax.__version__)
import dluxshera
print("dluxshera import ok")
PYENV
```

The here-doc delimiter must be exact: the closing `PYENV` has no quotes, no
indentation, and no duplicated Python code after it.

- Run one Edge smoke job or one test draw before submitting a full wave of
  10-30 jobs.
- Jobs that fail in 1-4 seconds with about 5 MB MaxRSS are usually
  launch/environment/shell failures, not science/model failures.
- Healthy Edge launch states include `RUNNING`, `PENDING` with
  `QOSMaxMemoryPerUser`, or jobs that at least print the Python/import
  diagnostics before later model execution.

## Expensive Steps

Do not run generated `preflight_*_shards.sh` scripts on login or head nodes.
Those scripts call the full-fidelity wrapper with `--dry-run`, which can build
campaign plans, resolve systems, and write model/template artifacts. That is
much cheaper than a full campaign, but it is still not a login-node schema
check.

Use `audit_campaign_config_schema.py` for login-node validation. Reserve
preflight and real campaign execution for a compute node or scheduled Slurm job.

## Results Root Rule

Use this:

```text
/projects/.../dLuxShera-Results
```

Do not use this with the shard helper:

```text
/projects/.../dLuxShera-Results/observation_bias_campaign
```

The helper and sbatch wrapper add `observation_bias_campaign` themselves.
Passing the nested path produces doubled run roots.

## Resume And Render-Retention Validation

Production campaign roots under `/projects/shera_hpc/...` must be audited on
Gattaca2 before broad relaunch. Do not infer compatibility only from expected
paths; check the actual representative root.

Repository-side checks before copying/launching:

```bash
git rev-parse HEAD
git status --short
PYTHONPATH=src python examples/scripts/audit_campaign_config_schema.py \
  --config path/to/full_fidelity.yaml
```

For one representative partial M1 root, run non-destructive inventory commands
that count current-contract artifacts by window:

```bash
ROOT=/projects/.../dLuxShera-Results/observation_bias_campaign/<run_name>

find "$ROOT/cases" -path '*/windows/window_*/posterior_by_label.csv' -type f | sort
find "$ROOT/cases" -path '*/windows/window_*/iterative_reference_update.json' -type f | sort
find "$ROOT/subblock_runs" -path '*/window_*/subblock_*/study/schur_summary/subblock_summary.json' -type f | sort | wc -l
find "$ROOT/subblock_runs" -path '*/window_*/subblock_*/render/*_cube.fits' -type f | sort | wc -l
find "$ROOT/subblock_runs" -path '*/window_*/subblock_*/render/*_variance.fits' -type f | sort | wc -l
```

To identify the partial window, compare per-window counts of canonical
subblock summaries with the configured `iterative.subblocks_per_window`, and
compare completed-window posterior/reference-update files with the current
runner contract. A completed already-pruned window should have the posterior and
reference-update artifacts even when render FITS count is zero.

If using `--dry-run`, treat it as a compute-node preflight, not a login-node
schema audit: it can build plans and write model/template artifacts. There is
no separate fake dry-run resume planner for render cleanup.

For the first real-root smoke resume with `render_retention: delete_after_window`,
watch for these behaviors before broad relaunch:

- already completed windows are reused and any residual completed-window render
  FITS are pruned;
- the current partial window keeps existing render FITS until missing subblocks
  finish and the window aggregates;
- completed subblocks inside the partial window are reused;
- missing subblocks execute normally;
- once the partial window writes posterior/reference-update artifacts, only its
  `*_cube.fits` and `*_variance.fits` are removed;
- the next untouched window starts normally.

Post-smoke, check `campaign_summary.json`,
`analysis/aggregate_status.json`, `subblock_status_iterative.csv`, per-window
`render_retention/cleanup_latest.json`, and FITS counts by window before
launching a larger wave.

## HO-WFE Field-Dither Family

The 2026-07 per-mirror high-order WFE knowledge-error and field-dither family is
prepared by:

```bash
PYTHONPATH=src:. python examples/scripts/prepare_howfe_field_dither_campaign_family.py --overwrite
```

This writes source configs, draw-level shard configs, manifests, status helpers,
and Gattaca Edge submit scripts under:

```text
examples/recipes/full_fidelity_howfe_field_dither_20260720/
```

The helper delegates sharding to `prepare_full_fidelity_campaign_shards.py` and
does not execute validation, preflight, or production jobs. The generated
scientific YAMLs keep scheduler details outside the config; Edge-specific launch
details live in `submit_draw_shards_edge.sh`.

Cheap login-node audits:

```bash
PYTHONPATH=src python examples/scripts/audit_campaign_config_schema.py \
  --config examples/recipes/full_fidelity_howfe_field_dither_20260720/configs/production/ff_howfe_field_m1_hoke_0p1nm_xp0p0_yp0p0_w10x30_v1.yaml

PYTHONPATH=src python examples/scripts/audit_campaign_config_schema.py \
  --config examples/recipes/full_fidelity_howfe_field_dither_20260720/configs/production/ff_howfe_field_m2_hoke_0p1nm_xp0p0_yp0p0_w10x30_v1.yaml

PYTHONPATH=src python examples/scripts/audit_campaign_config_schema.py \
  --config examples/recipes/full_fidelity_howfe_field_dither_20260720/configs/controls/ff_howfe_field_noke_xp0p0_yp0p0_w10x30_v1.yaml

PYTHONPATH=src python examples/scripts/audit_campaign_config_schema.py \
  --config examples/recipes/full_fidelity_howfe_field_dither_20260720/configs/production/ff_howfe_field_m2_hoke_0p1nm_xp1p0_yp0p0_w10x30_v1.yaml

PYTHONPATH=src python examples/scripts/audit_campaign_config_schema.py \
  --config examples/recipes/full_fidelity_howfe_field_dither_20260720/configs/production/ff_howfe_field_m1_hoke_0p1nm_xp0p0_yp0p0_w10x30_v1.yaml \
  --check-shard-manifest examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/production_center/shard_manifest.csv
```

Validation launch, not run during preparation:

```bash
./examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/validation/submit_draw_shards_edge.sh
```

Production priority order, not run during preparation:

```bash
./examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/production_center/submit_draw_shards_edge.sh
./examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/production_m2_offaxis/submit_draw_shards_edge.sh
./examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/production_m1_offaxis/submit_draw_shards_edge.sh
./examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/submit_draw_shards_edge.sh
```

The generated preflight scripts are still expensive because they call the
full-fidelity wrapper `--dry-run`; reserve them for compute nodes.

For TACC, reuse the same scientific YAMLs and provide a site wrapper with module
or Conda activation, repository/data paths, results root, Slurm
account/partition, CPU and memory requests, JAX cache path, and submission
syntax appropriate to that system.
