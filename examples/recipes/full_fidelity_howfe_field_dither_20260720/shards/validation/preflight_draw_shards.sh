#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on the cluster.
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate dluxshera-py311

PREFLIGHT_ROOT="${PREFLIGHT_ROOT:-examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/validation/preflight_results}"
mkdir -p "$PREFLIGHT_ROOT"
PYTHONPATH=src python examples/scripts/check_full_fidelity_campaign_shards.py preflight --manifest examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/validation/shard_manifest.csv --results-root "$PREFLIGHT_ROOT"
