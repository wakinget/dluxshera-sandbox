#!/usr/bin/env bash
set -euo pipefail

# Aggregate submission helper. Prefer wave scripts unless queue pressure is acceptable.
ROOT="work/full_fidelity_target_registry_no_ke_hardloworder_n10_w10x30_projected_30min_v1_shards"
"${ROOT}/scripts/submit_wave1.sh"
"${ROOT}/scripts/submit_wave2.sh"
"${ROOT}/scripts/submit_wave3.sh"
