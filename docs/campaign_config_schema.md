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
