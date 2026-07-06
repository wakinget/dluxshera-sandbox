#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on the cluster.
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate dluxshera-py311

RESULTS_ROOT="${RESULTS_ROOT:-/projects/shera_hpc/dmckeith/dLuxShera-Results}"
mkdir -p "$RESULTS_ROOT/slurm_logs"

# full_fidelity_info_damped_no_ke_single_w30x30_actual15min_v1_cond_m1_0p3nm_m2_0p3nm_draw_000
CONFIG=work/full_fidelity_condition_draw_shards/no_ke_single_w30x30_actual15min_v1/configs/full_fidelity_info_damped_no_ke_single_w30x30_actual15min_v1_cond_m1_0p3nm_m2_0p3nm_draw_000.yaml
RUN_NAME=full_fidelity_info_damped_no_ke_single_w30x30_actual15min_v1_cond_m1_0p3nm_m2_0p3nm_draw_000
MAX_WORKERS=15
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=10-00:00:00 --cpus-per-task=20 --mem=400G --job-name=ff-m1_0p3nm_m2_0p3nm-d000 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/projects/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=work/full_fidelity_condition_draw_shards/no_ke_single_w30x30_actual15min_v1/configs/full_fidelity_info_damped_no_ke_single_w30x30_actual15min_v1_cond_m1_0p3nm_m2_0p3nm_draw_000.yaml,RUN_NAME=full_fidelity_info_damped_no_ke_single_w30x30_actual15min_v1_cond_m1_0p3nm_m2_0p3nm_draw_000,MAX_WORKERS=15,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch
