# Full-fidelity iterative campaign shards

Source config: `examples/recipes/full_fidelity_next_campaigns_20260703/full_fidelity_info_damped_no_ke_single_w30x30_actual15min_v1.yaml`

This directory contains 1 `draw` shards. The original 300-subblock
campaign exceeded a 24-hour Slurm allocation: 50 successful rows had a median
subblock runtime of about 4278 seconds. Iterative windows within a draw are
sequential because each posterior/reference update feeds the next window.
Conditions and prior draws are independent, while subblocks within one window
are independent.

Condition sharding is the recommended production mode because it reduces the
current 2x2 campaign to four jobs without fragmenting it into 20 small jobs.
Draw mode is available for tighter scheduling or reruns. `MAX_WORKERS=5` matches
the current five subblocks per window; larger values do not add useful
parallelism unless the runner gains another safe parallel axis.

## Workflow

From the repository root:

```bash
./preflight_draw_shards.sh
./submit_draw_shards.sh
./summarize_shard_status.sh
```

Recommended resources are `4-00:00:00`, 20 CPUs,
`400G`, and `MAX_WORKERS=15`. Preflight uses
`--dry-run --max-workers 1 --resource-time auto`, verifies required plan
artifacts, checks expected counts and theta layout, and rejects a first shard
that accidentally contains the full parent campaign.

Each shard keeps its own run root and runs the existing analyzer independently.
`shard_manifest.csv` is the source of truth connecting shards to the parent
campaign and should be referenced from the Campaign Tracker with submitted job
IDs. A future multi-run aggregator may concatenate compatible per-shard
analysis tables; no cross-shard science aggregation is performed here.

## GPU benchmark

Do not switch production submissions to GPU by default. Benchmark one
condition, one draw, one window, and five subblocks separately with
`MAX_WORKERS=1` or `2`, then compare CPU and GPU subblock runtimes.
