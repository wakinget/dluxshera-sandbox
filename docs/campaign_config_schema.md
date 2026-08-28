# Full-Fidelity Campaign Config Schema

This guide documents the executable full-fidelity binary iterative campaign
schema used by
`examples/scripts/run_full_fidelity_binary_iterative_campaign.py` and
`examples/scripts/prepare_full_fidelity_campaign_shards.py`.

It focuses on source configs with:

```yaml
experiment:
  kind: full_fidelity_binary_iterative
```

The wrapper translates those configs into the older executable
`observation_bias_campaign` schema before delegating to
`examples/scripts/run_observation_bias_campaign.py`.

## Schema Layers

`full_fidelity_binary_iterative` is the canonical source schema for current
full-fidelity binary iterative launches. Use it for campaign YAMLs that will be
sharded or submitted through the full-fidelity wrapper.

`observation_bias_campaign` is the translated executable schema. The
observation-bias runner reads this kind directly, but the shard helper currently
expects the source full-fidelity schema, not an already translated config.

`full_fidelity_algorithm_campaign` is a design skeleton, not an executable
campaign kind for this path.

## Canonical Cadence

The realized cadence is owned by `experiment.iterative`:

```yaml
experiment:
  iterative:
    windows_per_draw: 5
    subblocks_per_window: 60
```

These two fields define the actual generated work per prior draw:

```text
total_realized_subblocks = windows_per_draw * subblocks_per_window
```

For example, `5 x 60` produces 300 realized subblocks per draw.

When the full-fidelity wrapper translates to `observation_bias_campaign`, it
derives:

- `experiment.subblocks.n_subblocks`
- `experiment.subblocks.trace_source.window.n_subblocks`, when a trace source
  window is present
- `experiment.iterative_forecast.actual_windows`
- `experiment.iterative_forecast.subblocks_per_window`

If any of these redundant realized fields are present in the source config,
they must match the canonical iterative cadence.

## Forecast Fields

`experiment.iterative_forecast` is the active projection path for iterative
full-fidelity campaigns. Its realized fields usually mirror
`experiment.iterative`, while `projected_windows` is intentionally separate:

```yaml
experiment:
  iterative:
    windows_per_draw: 5
    subblocks_per_window: 60
  iterative_forecast:
    projected_windows: 60
    observation_duration_s: 1800.0
```

`iterative_forecast.actual_windows` should match
`iterative.windows_per_draw` when present. `iterative_forecast.subblocks_per_window`
should match `iterative.subblocks_per_window` when present. If either field is
intentionally different, do not rely on the current wrapper path without a code
change: the wrapper and runner treat mismatches as contradictions.

`iterative_forecast.projected_windows` is the projection target. A 30-minute
projection may use `projected_windows: 60` even when the realized run only uses
3, 5, or 10 update windows.

`experiment.forecast` is the legacy non-iterative forecast path in
`run_observation_bias_campaign.py`. Keep it disabled for current iterative
full-fidelity templates unless you are deliberately exercising that older path.
If both forecast blocks are enabled, the iterative projection and legacy
forecast analysis are separate consumers with different output conventions.

## Subblocks And Trace Source

`experiment.subblocks.n_subblocks` is derived for iterative full-fidelity source
configs. When present, it must equal:

```text
iterative.windows_per_draw * iterative.subblocks_per_window
```

`experiment.subblocks.trace_source.window.start_s` selects the start of the
continuous trajectory interval. `trace_source.window.n_subblocks` is also
derived when present; it must match the total realized subblock count.

For iterative-disabled observation-bias configs, `subblocks.n_subblocks` remains
the canonical total count. That is a different execution mode from the
full-fidelity binary iterative source schema described here.

Trajectory trace sources may add constant registration offsets after trajectory
filtering:

```yaml
experiment:
  subblocks:
    trace_source:
      mode: trajectory
      processing:
        filter:
          enabled: true
          kind: high_pass
          apply_stage: before_window
        offsets:
          source.x_position_as: 1.0
          source.y_position_as: 0.0
          source.position_angle_deg: 0.0
```

Supported offset keys are `source.x_position_as`, `source.y_position_as`, and
`source.position_angle_deg`. The processing order is raw trajectory load,
canonical X/Y/PA mapping, configured filtering, constant offsets, frame
interpolation, subblock split, then frame-truth and starting-guess artifact
writing. For the diagnostic `filter.apply_stage: after_window` mode, offsets
are applied after that frame-sampled filter and before subblock splitting, so a
high-pass filter cannot remove the requested DC field displacement. Zero
offsets preserve previous trajectory values to numerical precision.

When offsets are active, trajectory preparation writes
`trajectory_offset_provenance.json` and `trajectory_offset_summary.csv` beside
the existing trajectory filter artifacts. These files record requested offsets,
application stage, and pre/post mean, standard deviation, min, max, and
peak-to-peak statistics.

## Render Retention

Rendered subblock cube and variance FITS are transient inputs to the
image-backed Schur-summary export. Iterative full-fidelity campaigns may opt in
to post-window cleanup with:

```yaml
experiment:
  subblocks:
    render_retention: delete_after_window
```

Supported values are:

- `keep`: default when omitted; preserve rendered FITS exactly as previous
  campaign runs did.
- `delete_after_window`: after a full iterative window has aggregated and the
  runner has written valid completed-window artifacts, delete only
  `*_cube.fits` and `*_variance.fits` under that window's subblock `render/`
  directories.

The safety boundary is the iterative window, not an individual subblock.
Incomplete windows retain all rendered FITS, including FITS for subblocks that
already have `study/schur_summary/subblock_summary.json`. Persistent Schur
summaries, Schur matrix products, render manifests, frame-truth metadata,
configs, diagnostics, logs, status/progress files, posterior tables, reference
updates, campaign/window summaries, and analysis products are not pruned.

Per-window cleanup provenance is written under:

```text
cases/<case>/windows/window_###/render_retention/
```

when pruning is active. The latest JSON and JSONL history record the policy,
window index, completion guard artifacts, deleted file count, deleted logical
bytes, allowed suffixes, skipped candidates, and unlink errors.

## Subprocess Timeout

Subblock subprocess execution is unbounded by default for backward
compatibility. Recovery campaigns may opt in to a finite parent-side timeout:

```yaml
experiment:
  subblocks:
    subprocess_timeout_s: 21600
```

The same value can be overridden from the observation-bias or full-fidelity
wrapper CLIs with `--subprocess-timeout-s <seconds>`. Values must be positive
finite seconds; unset/null preserves the historical no-timeout behavior.

On timeout, the parent terminates the child process group where supported,
writes `subprocess_diagnostics.json` with `failure_class: timeout`, preserves
any already-written `study/schur_summary/subblock_summary.json`, and reports the
subblock as failed for the current invocation. A later `--resume` continues to
use the existing science summary as the completion marker only when the
canonical science summary is present and valid/loadable.

## High-Order WFE

`experiment.high_order_wfe` remains backward-compatible with the original scalar
knowledge-error block:

```yaml
high_order_wfe:
  enabled: true
  inference:
    knowledge_error:
      enabled: true
      amplitude_nm_rms: 0.1
```

When no mirror-specific block is present, the scalar knowledge-error amplitude
is applied to every active truth mirror. Per-mirror overrides can enable or
disable the additive high-order knowledge-error residual independently:

```yaml
high_order_wfe:
  enabled: true
  truth:
    enabled: true
    mirrors: [primary, secondary]
    mode: synthetic
    npix: 256
    amplitude_nm_rms: 20.0
    power_law_alpha: 2.5
    seed: 20260610
    pairing: independent
    remove_low_order_zernikes: true
    remove_zernike_modes: [4, 5, 6, 7, 8, 9, 10, 11]
  inference:
    enabled: true
    mode: knowledge_error
    use_truth_common_map: true
    knowledge_error:
      enabled: true
      seed: 20260720
      pairing: independent
      power_law_alpha: same_as_truth
      remove_low_order_zernikes: true
      mirrors:
        primary:
          enabled: true
          amplitude_nm_rms: 0.1
        secondary:
          enabled: false
          amplitude_nm_rms: 0.0
```

Mirror-specific `enabled: false` means the reference/inference map for that
mirror is truth-matched; it does not remove that mirror's truth high-order WFE.
Truth maps remain present for all mirrors listed under `truth.mirrors`.

`truth.seed` controls truth-map realization. `inference.knowledge_error.seed`
controls knowledge-error morphology independently. With an explicit KE seed, the
same seed and mirror produce the same normalized residual morphology regardless
of run name, prior draw, field offset, shard name, or requested RMS. Changing
`amplitude_nm_rms` rescales that same morphology. Provenance records per-mirror
truth seeds, KE seeds, enabled state, requested and measured KE RMS, truth vs
reference difference RMS, truth-match state, and raw/normalized map hashes.

## Results Root Semantics

The safest convention for full-fidelity shard launches is:

```text
RESULTS_ROOT=/projects/.../dLuxShera-Results
```

Do not pass a root that already ends in `observation_bias_campaign` to the shard
helper or generated submit scripts.

The scripts interpret roots as follows:

| Script | Expected argument | Final run root |
| --- | --- | --- |
| `run_observation_bias_campaign.py --results-root` | campaign family directory | `<results-root>/<run_name>` |
| `run_full_fidelity_binary_iterative_campaign.py --results-root` | translated observation-bias family directory | `<results-root>/<run_name>` |
| `prepare_full_fidelity_campaign_shards.py --results-root` | parent results directory | `<results-root>/observation_bias_campaign/<shard_name>` in the manifest |
| generated `submit_draw_shards.sh` / `submit_condition_shards.sh` | parent `RESULTS_ROOT` | sbatch wrapper computes `RUN_RESULTS_ROOT="$RESULTS_ROOT/observation_bias_campaign"` |
| `full_fidelity_iterative_campaign_hpc.sbatch` | parent `RESULTS_ROOT` env var | passes `$RESULTS_ROOT/observation_bias_campaign` to the wrapper |

Passing `/projects/.../dLuxShera-Results/observation_bias_campaign` to the shard
helper creates doubled paths such as:

```text
observation_bias_campaign/observation_bias_campaign
```

Use the audit utility to catch this in manifests:

```bash
python examples/scripts/audit_campaign_config_schema.py \
  --config examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_info_damped_detector_ke_projected_30min_v1.yaml \
  --check-shard-manifest path/to/shard_manifest.csv
```

## Sharding Semantics

`prepare_full_fidelity_campaign_shards.py` accepts source
`full_fidelity_binary_iterative` configs. It does not accept translated
`observation_bias_campaign` configs.

`--mode condition` creates one shard per prior-draw condition. Each shard keeps
all draws for that condition.

`--mode draw` creates one shard per condition and draw. This is useful for
tighter scheduling or rerunning a single draw, but it creates more jobs.

Draw mode writes:

- `shard_manifest.csv`
- `submit_draw_shards.sh`
- `preflight_draw_shards.sh`
- `summarize_shard_status.sh`
- `configs/*.yaml`
- `README.md`

The manifest columns are:

```text
shard_name
shard_mode
source_config_path
config_path
expected_run_root
condition_label
m1_sigma_nm
m2_sigma_nm
draw_start
draw_stop
draw_index
expected_subblocks
expected_windows
expected_subblocks_per_window
expected_n_theta
recommended_time
recommended_cpus_per_task
recommended_mem
recommended_max_workers
sbatch_command
ho_ke_active_mirror
ho_ke_primary_enabled
ho_ke_secondary_enabled
ho_ke_primary_amplitude_nm_rms
ho_ke_secondary_amplitude_nm_rms
field_offset_x_as
field_offset_y_as
field_offset_pa_deg
truth_seed
knowledge_error_seed
primary_ke_map_hash
secondary_ke_map_hash
map_group
```

The most important count fields should resolve from the source cadence:

- `expected_subblocks = selected_draws * windows_per_draw * subblocks_per_window`
- `expected_windows = selected_draws * windows_per_draw`
- `expected_subblocks_per_window = subblocks_per_window`

## Slurm And Edge

Generated submit scripts currently emit plain `sbatch` commands. On clusters
where Edge submission is required, inspect or edit the command to include:

```bash
sbatch -M edge ...
```

On Gattaca2 Edge, also make environment and shared data paths side-explicit.
Do not use `/scratch` when the job needs a JPL-side environment; Edge resolves
`/scratch` to `/scratch-edge`, while the default/JPL side resolves it to
`/scratch-jpl`. See `docs/full_fidelity_campaign_launch.md` for the current
Edge launch checklist.

A future helper option such as `--slurm-cluster edge` would make this less
manual. For now, treat cluster selection as an explicit launch review step.

## Cheap Audit

Use:

```bash
python examples/scripts/audit_campaign_config_schema.py --config path/to/config.yaml
```

The audit only parses config and optional manifest CSVs. It is safe for login
nodes because it does not generate campaign plans, render models, write smear
templates, or run subblock preflights.
