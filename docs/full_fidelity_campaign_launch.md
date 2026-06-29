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
