# Full-fidelity runtime benchmark configs, 2026-07-03

These benchmark jobs are for runtime decomposition, not science. They are separate from the science shard campaigns and should not be submitted automatically.

## Cases

- `full_fidelity_runtime_benchmark_truth_2x20f_v1`: `phi_ref=truth_when_available`, 2 subblocks x 20 frames, `MAX_WORKERS=1`
- `full_fidelity_runtime_benchmark_recovered_2x20f_v1`: `phi_ref=recovered`, 2 subblocks x 20 frames, `MAX_WORKERS=1`

Source YAMLs:

- `examples/recipes/full_fidelity_next_campaigns_20260703/full_fidelity_runtime_benchmark_truth_2x20f_v1.yaml`
- `examples/recipes/full_fidelity_next_campaigns_20260703/full_fidelity_runtime_benchmark_recovered_2x20f_v1.yaml`

Sbatch scripts:

- `work/full_fidelity_runtime_benchmarks_20260703/full_fidelity_runtime_benchmark_truth_2x20f_v1.sbatch`
- `work/full_fidelity_runtime_benchmarks_20260703/full_fidelity_runtime_benchmark_recovered_2x20f_v1.sbatch`

Both scripts request `--cpus-per-task=4`, `--mem=96G`, and `MAX_WORKERS=1`. They preserve one-thread-per-child JAX/BLAS settings and enable:

- `--resource-time auto`
- `--profile-runtime`
- `--profile-runtime-detail basic`
- `--memory-diagnostics`

## First run

Run the truth-reference benchmark first:

```bash
sbatch work/full_fidelity_runtime_benchmarks_20260703/full_fidelity_runtime_benchmark_truth_2x20f_v1.sbatch
```

If that completes and memory is stable, run the recovered-reference benchmark:

```bash
sbatch work/full_fidelity_runtime_benchmarks_20260703/full_fidelity_runtime_benchmark_recovered_2x20f_v1.sbatch
```

## Outputs to inspect

Under each run root in `/projects/shera_hpc/dmckeith/dLuxShera-Results/observation_bias_campaign/`, inspect:

- `execution_context.json`
- `campaign_plan.json`
- `subblock_status_iterative.csv`
- each subblock `subprocess_diagnostics.json`
- runtime profile JSON/JSONL artifacts emitted by `run_obs_subblock_study.py`
- memory diagnostics artifacts emitted by `run_obs_subblock_study.py`

The expected comparison is the incremental cost of recovered-reference solving over truth-when-available for the same 2 x 20-frame layout.
