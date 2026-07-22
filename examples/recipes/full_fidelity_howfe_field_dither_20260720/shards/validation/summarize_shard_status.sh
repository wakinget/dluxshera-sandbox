#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on the cluster.
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate dluxshera-py311

RESULTS_ROOT="${RESULTS_ROOT:-/projects/shera_hpc/dmckeith/dLuxShera-Results}"
PYTHONPATH=src python examples/scripts/check_full_fidelity_campaign_shards.py status --manifest examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/validation/shard_manifest.csv --results-root "$RESULTS_ROOT"
