# Subblock inference analysis

## Runtime profiling and cacheability diagnostics

Use `--profile-runtime` with `examples/scripts/run_obs_subblock_study.py --mode schur_summary` to emit:
- `runtime_profile_summary.json`
- `runtime_profile_timeline.jsonl`

These artifacts separate case preparation, inference, Schur export, and diagnostics overhead for wall-time auditing and cacheability review.
