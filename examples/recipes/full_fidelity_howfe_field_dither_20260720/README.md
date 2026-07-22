# HO-WFE Field-Dither Campaign Family

Prepared artifacts only. No validation or production jobs were submitted.

Base config: `examples/recipes/full_fidelity_next_campaigns_20260703/full_fidelity_info_damped_hoke_0p1nm_loz0p01nm_n10_w10x30_projected_30min_v1.yaml`

Fixed seeds:
- truth maps: `20260610`
- knowledge-error maps: `20260720`
- M1/M2 low-order prior sigma: `1.0` nm

Production HO-WFE KE matrix:
- 2 mirrors x 5 amplitudes x 5 fields x 10 draws = 500 draw shards
- no-KE field controls: 5 fields x 10 draws = 50 draw shards

Priority launch order:
1. `shards/production_center/submit_draw_shards_edge.sh`
2. `shards/production_m2_offaxis/submit_draw_shards_edge.sh`
3. `shards/production_m1_offaxis/submit_draw_shards_edge.sh`
4. `shards/controls/submit_draw_shards_edge.sh`

Validation wave:
- `shards/validation/submit_draw_shards_edge.sh`

Cheap audits:

```bash
PYTHONPATH=src python examples/scripts/audit_campaign_config_schema.py --config examples/recipes/full_fidelity_howfe_field_dither_20260720/configs/production/ff_howfe_field_m1_hoke_0p1nm_xp0p0_yp0p0_w10x30_v1.yaml
PYTHONPATH=src python examples/scripts/audit_campaign_config_schema.py --config examples/recipes/full_fidelity_howfe_field_dither_20260720/configs/production/ff_howfe_field_m2_hoke_0p1nm_xp0p0_yp0p0_w10x30_v1.yaml
PYTHONPATH=src python examples/scripts/audit_campaign_config_schema.py --config examples/recipes/full_fidelity_howfe_field_dither_20260720/configs/controls/ff_howfe_field_noke_xp0p0_yp0p0_w10x30_v1.yaml
PYTHONPATH=src python examples/scripts/audit_campaign_config_schema.py --config examples/recipes/full_fidelity_howfe_field_dither_20260720/configs/production/ff_howfe_field_m2_hoke_0p1nm_xp1p0_yp0p0_w10x30_v1.yaml
PYTHONPATH=src python examples/scripts/audit_campaign_config_schema.py --config examples/recipes/full_fidelity_howfe_field_dither_20260720/configs/production/ff_howfe_field_m1_hoke_0p1nm_xp0p0_yp0p0_w10x30_v1.yaml --check-shard-manifest examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/production_center/shard_manifest.csv
```

Compute-node preflight commands are generated as `preflight_draw_shards.sh` in
each shard group. They are intentionally marked expensive and were not run.

TACC portability note: the scientific YAMLs avoid scheduler-specific fields.
A TACC wrapper still needs to provide module/environment activation, repository
and data paths, results root, Slurm account/partition, CPU/memory requests, JAX
cache path, and any submission wrapper differences.
