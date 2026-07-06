#!/usr/bin/env bash
set -euo pipefail

# Wave 2 science: additional moderate targets, all 10 draws each.
ROOT="work/full_fidelity_target_registry_no_ke_hardloworder_n10_w10x30_projected_30min_v1_shards"
"${ROOT}/70_OPH/submit_draw_shards.sh"
"${ROOT}/36_OPH/submit_draw_shards.sh"
"${ROOT}/XI_BOO/submit_draw_shards.sh"
