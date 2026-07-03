#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on the cluster.
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate dluxshera-py311

PREFLIGHT_ROOT="${PREFLIGHT_ROOT:-work/full_fidelity_condition_draw_shards/no_ke_single_w30x30_actual15min_v1/preflight_results}"
mkdir -p "$PREFLIGHT_ROOT"
PYTHONPATH=src python examples/scripts/check_full_fidelity_campaign_shards.py preflight --manifest work/full_fidelity_condition_draw_shards/no_ke_single_w30x30_actual15min_v1/shard_manifest.csv --results-root "$PREFLIGHT_ROOT"
