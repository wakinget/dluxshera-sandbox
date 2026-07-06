#!/usr/bin/env bash
set -euo pipefail

# Wave 1 science: Alpha Cen plus one non-Alpha target, all 10 draws each.
ROOT="work/full_fidelity_target_registry_no_ke_hardloworder_n10_w10x30_projected_30min_v1_shards"
"${ROOT}/ALPHA_CEN/submit_draw_shards.sh"
"${ROOT}/61_CYG/submit_draw_shards.sh"
