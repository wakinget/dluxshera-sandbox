#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on the cluster.
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate dluxshera-py311

PREFLIGHT_ROOT="${PREFLIGHT_ROOT:-work/full_fidelity_target_registry_no_ke_hardloworder_n10_w10x30_projected_30min_v1_shards/61_CYG/preflight_results}"
mkdir -p "$PREFLIGHT_ROOT"
PYTHONPATH=src python examples/scripts/check_full_fidelity_campaign_shards.py preflight --manifest work/full_fidelity_target_registry_no_ke_hardloworder_n10_w10x30_projected_30min_v1_shards/61_CYG/shard_manifest.csv --results-root "$PREFLIGHT_ROOT"
