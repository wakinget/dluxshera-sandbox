#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on the cluster.
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate dluxshera-py311

RESULTS_ROOT="${RESULTS_ROOT:-/projects/shera_hpc/dmckeith/dLuxShera-Results}"
PYTHONPATH=src python examples/scripts/check_full_fidelity_campaign_shards.py status --manifest work/full_fidelity_condition_draw_shards/pixelposke_1em3pix_n10_w10x30_projected_30min_v1/shard_manifest.csv --results-root "$RESULTS_ROOT"
