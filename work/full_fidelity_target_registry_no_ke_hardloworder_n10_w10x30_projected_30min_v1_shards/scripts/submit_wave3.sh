#!/usr/bin/env bash
set -euo pipefail

# Wave 3 science: remaining targets, all 10 draws each.
ROOT="work/full_fidelity_target_registry_no_ke_hardloworder_n10_w10x30_projected_30min_v1_shards"
"${ROOT}/P_ERI/submit_draw_shards.sh"
"${ROOT}/HR_2667_2668/submit_draw_shards.sh"
