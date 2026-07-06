# Target-Registry No-KE Hard-Low-Order Campaign

Campaign family:

```bash
full_fidelity_target_registry_no_ke_hardloworder_n10_w10x30_projected_30min_v1
```

Source configs are canonical `experiment.kind: full_fidelity_binary_iterative` YAMLs under:

```bash
examples/recipes/full_fidelity_algorithm_campaign_template/target_registry_no_ke_hardloworder/
```

Shards are draw-mode, one target/draw per job, under this directory. Each target has 10 draw shards. Each draw shard expects 10 realized windows, 30 subblocks/window, and 300 total subblocks. The projection horizon remains 60 windows / 1800 seconds.

The parent results root is:

```bash
/projects/shera_hpc/dmckeith/dLuxShera-Results
```

The helper-generated run roots append `observation_bias_campaign/<run_name>`.

## Target Audit

Compact audit table:

```bash
work/full_fidelity_target_registry_no_ke_hardloworder_n10_w10x30_projected_30min_v1_shards/target_audit.csv
```

## Login-Node-Safe Audits

Run all source schema and shard manifest audits:

```bash
./work/full_fidelity_target_registry_no_ke_hardloworder_n10_w10x30_projected_30min_v1_shards/scripts/audit_all_schema_and_manifests.sh
```

Single-target schema audit pattern:

```bash
PYTHONPATH=src python examples/scripts/audit_campaign_config_schema.py \
  --config examples/recipes/full_fidelity_algorithm_campaign_template/target_registry_no_ke_hardloworder/ff_targetreg_no_ke_hardlo_61_CYG_n10_w10x30_v1.yaml
```

Single-target manifest audit pattern:

```bash
PYTHONPATH=src python examples/scripts/audit_campaign_config_schema.py \
  --config examples/recipes/full_fidelity_algorithm_campaign_template/target_registry_no_ke_hardloworder/ff_targetreg_no_ke_hardlo_61_CYG_n10_w10x30_v1.yaml \
  --check-shard-manifest work/full_fidelity_target_registry_no_ke_hardloworder_n10_w10x30_projected_30min_v1_shards/61_CYG/shard_manifest.csv
```

The manifest audit warns that commands use plain `sbatch`. Preserve that unless the local launch convention requires editing commands to include `-M edge`.

## Validation Gate

Run this only from a compute-appropriate environment because preflight builds campaign plans and model/template artifacts:

```bash
./work/full_fidelity_target_registry_no_ke_hardloworder_n10_w10x30_projected_30min_v1_shards/scripts/preflight_validation_two_draws.sh
```

This validates draw 000 for `61_CYG` and draw 000 for `HR_2667_2668`. It writes a temporary two-row manifest at:

```bash
work/full_fidelity_target_registry_no_ke_hardloworder_n10_w10x30_projected_30min_v1_shards/validation_two_draw_manifest.csv
```

and preflight artifacts under:

```bash
work/full_fidelity_target_registry_no_ke_hardloworder_n10_w10x30_projected_30min_v1_shards/preflight_validation_results/
```

## Science Submission Waves

Do not submit science jobs until schema audits, manifest audits, and the two-draw validation gate pass.

Wave 1: `ALPHA_CEN`, `61_CYG`

```bash
./work/full_fidelity_target_registry_no_ke_hardloworder_n10_w10x30_projected_30min_v1_shards/scripts/submit_wave1.sh
```

Wave 2: `70_OPH`, `36_OPH`, `XI_BOO`

```bash
./work/full_fidelity_target_registry_no_ke_hardloworder_n10_w10x30_projected_30min_v1_shards/scripts/submit_wave2.sh
```

Wave 3: `P_ERI`, `HR_2667_2668`

```bash
./work/full_fidelity_target_registry_no_ke_hardloworder_n10_w10x30_projected_30min_v1_shards/scripts/submit_wave3.sh
```

Per-target submission remains available:

```bash
./work/full_fidelity_target_registry_no_ke_hardloworder_n10_w10x30_projected_30min_v1_shards/61_CYG/submit_draw_shards.sh
```

An aggregate helper exists but should only be used when queue pressure is acceptable:

```bash
./work/full_fidelity_target_registry_no_ke_hardloworder_n10_w10x30_projected_30min_v1_shards/scripts/submit_all_do_not_run_until_queue_pressure_is_acceptable.sh
```

## Status

Single target:

```bash
./work/full_fidelity_target_registry_no_ke_hardloworder_n10_w10x30_projected_30min_v1_shards/61_CYG/summarize_shard_status.sh
```

All targets:

```bash
for target in ALPHA_CEN 61_CYG 70_OPH 36_OPH XI_BOO P_ERI HR_2667_2668; do
  echo "== ${target} =="
  ./work/full_fidelity_target_registry_no_ke_hardloworder_n10_w10x30_projected_30min_v1_shards/${target}/summarize_shard_status.sh
done
```

## Shard Preparation Command Pattern

The shard directories were generated with:

```bash
PYTHONPATH=src python examples/scripts/prepare_full_fidelity_campaign_shards.py \
  --config <target_config.yaml> \
  --outdir work/full_fidelity_target_registry_no_ke_hardloworder_n10_w10x30_projected_30min_v1_shards/<TARGET_KEY> \
  --run-name-prefix ff_targetreg_no_ke_hardlo_<TARGET_KEY>_n10_w10x30_v1 \
  --mode draw \
  --results-root /projects/shera_hpc/dmckeith/dLuxShera-Results \
  --time 5-00:00:00 \
  --cpus-per-task 10 \
  --mem 250G \
  --max-workers 10
```
