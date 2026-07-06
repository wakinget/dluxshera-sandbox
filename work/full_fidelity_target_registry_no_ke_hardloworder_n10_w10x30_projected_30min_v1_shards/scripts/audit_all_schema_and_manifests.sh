#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. This audit is login-node safe.
TARGETS=(ALPHA_CEN 61_CYG 70_OPH 36_OPH XI_BOO P_ERI HR_2667_2668)
ROOT="work/full_fidelity_target_registry_no_ke_hardloworder_n10_w10x30_projected_30min_v1_shards"
CFG_ROOT="examples/recipes/full_fidelity_algorithm_campaign_template/target_registry_no_ke_hardloworder"
PYTHON_BIN="${PYTHON_BIN:-python}"

for target in "${TARGETS[@]}"; do
  config="${CFG_ROOT}/ff_targetreg_no_ke_hardlo_${target}_n10_w10x30_v1.yaml"
  manifest="${ROOT}/${target}/shard_manifest.csv"
  echo "== ${target}: schema =="
  PYTHONPATH=src "${PYTHON_BIN}" examples/scripts/audit_campaign_config_schema.py \
    --config "${config}"
  echo "== ${target}: manifest =="
  PYTHONPATH=src "${PYTHON_BIN}" examples/scripts/audit_campaign_config_schema.py \
    --config "${config}" \
    --check-shard-manifest "${manifest}"
done

if find "${ROOT}" \
  \( -name 'shard_manifest.csv' -o -name 'submit_draw_shards.sh' -o -name '*.yaml' \) \
  -type f -print0 | xargs -0 grep -H "observation_bias_campaign/observation_bias_campaign"; then
  echo "Found doubled observation_bias_campaign path." >&2
  exit 1
fi

echo "All schema and manifest audits passed."
