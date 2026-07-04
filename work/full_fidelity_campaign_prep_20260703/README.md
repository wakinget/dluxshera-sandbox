# Next full-fidelity campaign prep, 2026-07-03

This prep only generates configs, draw shards, submit scripts, and audits. It does not submit Slurm jobs.

## Source configs

Configs live in `examples/recipes/full_fidelity_next_campaigns_20260703/`.

Science families:

- `full_fidelity_info_damped_hoke_0p1nm_loz0p01nm_n10_w10x30_projected_30min_v1`
- `full_fidelity_info_damped_hoke_1p0nm_loz0p01nm_n10_w10x30_projected_30min_v1`
- `full_fidelity_info_damped_pixelposke_1em4pix_n10_w10x30_projected_30min_v1`
- `full_fidelity_info_damped_pixelposke_5em4pix_n10_w10x30_projected_30min_v1`
- `full_fidelity_info_damped_pixelposke_1em3pix_n10_w10x30_projected_30min_v1`
- `full_fidelity_info_damped_no_ke_single_w30x30_actual15min_v1`
- `full_fidelity_info_damped_no_ke_single_w60x30_actual30min_v1`

Runtime benchmark configs:

- `full_fidelity_runtime_benchmark_truth_2x20f_v1`
- `full_fidelity_runtime_benchmark_recovered_2x20f_v1`

## Shard roots

Draw-mode shards live under `work/full_fidelity_condition_draw_shards/`.

Expected row counts:

- `hoke_0p1nm_loz0p01nm_n10_w10x30_projected_30min_v1`: 10 shards, 300 subblocks each
- `hoke_1p0nm_loz0p01nm_n10_w10x30_projected_30min_v1`: 10 shards, 300 subblocks each
- `pixelposke_1em4pix_n10_w10x30_projected_30min_v1`: 10 shards, 300 subblocks each
- `pixelposke_5em4pix_n10_w10x30_projected_30min_v1`: 10 shards, 300 subblocks each
- `pixelposke_1em3pix_n10_w10x30_projected_30min_v1`: 10 shards, 300 subblocks each
- `no_ke_single_w30x30_actual15min_v1`: 1 shard, 900 subblocks
- `no_ke_single_w60x30_actual30min_v1`: 1 shard, 1800 subblocks

Resource settings in generated submit scripts:

- HO-KE and pixel-position KE science: `MAX_WORKERS=10`, `--mem=250G`, `--cpus-per-task=16`, `--time=4-00:00:00`
- Long No-KE pilots: `MAX_WORKERS=15`, `--mem=400G`, `--cpus-per-task=20`
- w30x30 long pilot: `--time=4-00:00:00`
- w60x30 long pilot: `--time=7-00:00:00`, with a submit-script gate requiring `ALLOW_W60X30_LONG_PILOT_SUBMIT=1`

## Required audits

Schema audit all generated source YAMLs:

```bash
for f in examples/recipes/full_fidelity_next_campaigns_20260703/*.yaml; do
  PYTHONPATH=src python3 examples/scripts/audit_campaign_config_schema.py --config "$f"
done
```

Shard manifest audit:

```bash
for manifest in work/full_fidelity_condition_draw_shards/*/shard_manifest.csv; do
  PYTHONPATH=src python3 examples/scripts/audit_campaign_config_schema.py \
    --config "$(awk -F, 'NR==2 {print $3}' "$manifest")" \
    --check-shard-manifest "$manifest"
done
```

Dry-run/preflight audit, to run on an appropriate compute/preflight environment, not as an expensive login-node launch:

```bash
./work/full_fidelity_condition_draw_shards/hoke_0p1nm_loz0p01nm_n10_w10x30_projected_30min_v1/preflight_draw_shards.sh
```

Repeat for each shard root.

Check no doubled result path:

```bash
grep -R "observation_bias_campaign/observation_bias_campaign" \
  work/full_fidelity_condition_draw_shards/*/shard_manifest.csv
```

Expected result: no matches.

## HO-WFE map hash audit

After dry-run/preflight artifacts exist, hash the map artifacts for all draw run roots in one HO-KE family:

```bash
python3 work/full_fidelity_campaign_prep_20260703/audit_high_order_wfe_map_hashes.py \
  /path/to/preflight_results/full_fidelity_info_damped_hoke_0p1nm_loz0p01nm_n10_w10x30_projected_30min_v1_cond_m1_0p01nm_m2_0p01nm_draw_* \
  --out work/full_fidelity_campaign_prep_20260703/hoke_0p1nm_map_hashes.csv
```

The truth common map and knowledge-error map hashes should be identical across prior draws within a family. Repeat for the 1.0 nm HO-KE family.

## Detector KE audit

Verify pixel offsets are enabled and pixel response KE is disabled/zero:

```bash
python3 work/full_fidelity_campaign_prep_20260703/audit_detector_pixelpos_ke_configs.py \
  examples/recipes/full_fidelity_next_campaigns_20260703/full_fidelity_info_damped_pixelposke_*pix_n10_w10x30_projected_30min_v1.yaml
```

## Science KE block audit

Verify the five HO-KE and pixel-position KE sweep families have the required
knowledge-error blocks under `experiment`, with the expected cadence and
low-order prior condition:

```bash
python3 work/full_fidelity_campaign_prep_20260703/audit_next_science_ke_configs.py \
  examples/recipes/full_fidelity_next_campaigns_20260703/full_fidelity_info_damped_hoke_0p1nm_loz0p01nm_n10_w10x30_projected_30min_v1.yaml \
  examples/recipes/full_fidelity_next_campaigns_20260703/full_fidelity_info_damped_hoke_1p0nm_loz0p01nm_n10_w10x30_projected_30min_v1.yaml \
  examples/recipes/full_fidelity_next_campaigns_20260703/full_fidelity_info_damped_pixelposke_1em4pix_n10_w10x30_projected_30min_v1.yaml \
  examples/recipes/full_fidelity_next_campaigns_20260703/full_fidelity_info_damped_pixelposke_5em4pix_n10_w10x30_projected_30min_v1.yaml \
  examples/recipes/full_fidelity_next_campaigns_20260703/full_fidelity_info_damped_pixelposke_1em3pix_n10_w10x30_projected_30min_v1.yaml
```

The audit prints `run_name`, cadence, `high_order_wfe`,
`detector_calibration_knowledge_error`, and primary/secondary low-order sigma.
It exits nonzero if an HO-KE config lacks an enabled `high_order_wfe` knowledge
error block, or if a pixel-position KE config lacks an enabled detector
calibration knowledge-error block.

## Long-pilot walltime estimate

Using the recent 66-85 min/subblock range and assuming 30 subblocks/window with `MAX_WORKERS=15`, each window takes roughly two waves:

- 66 min/subblock: about 132 min/window
- 85 min/subblock: about 170 min/window

The w30x30 pilot therefore brackets roughly 66-85 hours, plus aggregation overhead. The w60x30 pilot brackets roughly 132-170 hours, plus aggregation overhead. Both are likely too long for a single uninterrupted allocation on many Slurm partitions. Treat generated scripts as review artifacts; prefer a split-by-window/resume strategy if the local walltime limit is below these estimates.
