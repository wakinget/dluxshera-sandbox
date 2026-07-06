# ALPHA_CEN draw shards

This directory contains 10 draw-mode shards for `ALPHA_CEN` in the target-registry No-KE hard-low-order campaign. Each shard is one prior draw and expects 10 windows x 30 subblocks/window = 300 subblocks.

Recommended resources are `5-00:00:00`, 10 CPUs, `250G`, and `MAX_WORKERS=10`. The generated submit script intentionally uses plain `sbatch`; add `-M edge` only if required by the local launch convention.

From the repository root:

```bash
PYTHONPATH=src python examples/scripts/audit_campaign_config_schema.py \
  --config examples/recipes/full_fidelity_algorithm_campaign_template/target_registry_no_ke_hardloworder/ff_targetreg_no_ke_hardlo_ALPHA_CEN_n10_w10x30_v1.yaml \
  --check-shard-manifest work/full_fidelity_target_registry_no_ke_hardloworder_n10_w10x30_projected_30min_v1_shards/ALPHA_CEN/shard_manifest.csv

./work/full_fidelity_target_registry_no_ke_hardloworder_n10_w10x30_projected_30min_v1_shards/ALPHA_CEN/submit_draw_shards.sh
./work/full_fidelity_target_registry_no_ke_hardloworder_n10_w10x30_projected_30min_v1_shards/ALPHA_CEN/summarize_shard_status.sh
```

Use the top-level `VALIDATION_AND_LAUNCH_README.md` before submitting science jobs.
