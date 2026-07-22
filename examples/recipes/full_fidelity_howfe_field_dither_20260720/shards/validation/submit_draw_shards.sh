#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on the cluster.
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate dluxshera-py311

RESULTS_ROOT="${RESULTS_ROOT:-/projects/shera_hpc/dmckeith/dLuxShera-Results}"
mkdir -p "$RESULTS_ROOT/slurm_logs"

# ff_howfe_validation_cond_noke_xp0p0_yp0p0_w1x30_draw_000
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/validation/configs/ff_howfe_validation_cond_noke_xp0p0_yp0p0_w1x30_draw_000.yaml
RUN_NAME=ff_howfe_validation_cond_noke_xp0p0_yp0p0_w1x30_draw_000
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp0p0_yp0p0_w1x30-d000 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/projects/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/validation/configs/ff_howfe_validation_cond_noke_xp0p0_yp0p0_w1x30_draw_000.yaml,RUN_NAME=ff_howfe_validation_cond_noke_xp0p0_yp0p0_w1x30_draw_000,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_validation_cond_m1_hoke_0p1nm_xp0p0_yp0p0_w1x30_draw_000
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/validation/configs/ff_howfe_validation_cond_m1_hoke_0p1nm_xp0p0_yp0p0_w1x30_draw_000.yaml
RUN_NAME=ff_howfe_validation_cond_m1_hoke_0p1nm_xp0p0_yp0p0_w1x30_draw_000
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-m1_hoke_0p1nm_xp0p0_yp0p0_w1x30-d000 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/projects/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/validation/configs/ff_howfe_validation_cond_m1_hoke_0p1nm_xp0p0_yp0p0_w1x30_draw_000.yaml,RUN_NAME=ff_howfe_validation_cond_m1_hoke_0p1nm_xp0p0_yp0p0_w1x30_draw_000,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_validation_cond_m2_hoke_0p1nm_xp0p0_yp0p0_w1x30_draw_000
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/validation/configs/ff_howfe_validation_cond_m2_hoke_0p1nm_xp0p0_yp0p0_w1x30_draw_000.yaml
RUN_NAME=ff_howfe_validation_cond_m2_hoke_0p1nm_xp0p0_yp0p0_w1x30_draw_000
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-m2_hoke_0p1nm_xp0p0_yp0p0_w1x30-d000 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/projects/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/validation/configs/ff_howfe_validation_cond_m2_hoke_0p1nm_xp0p0_yp0p0_w1x30_draw_000.yaml,RUN_NAME=ff_howfe_validation_cond_m2_hoke_0p1nm_xp0p0_yp0p0_w1x30_draw_000,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_validation_cond_m2_hoke_0p1nm_xp1p0_yp0p0_w1x30_draw_000
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/validation/configs/ff_howfe_validation_cond_m2_hoke_0p1nm_xp1p0_yp0p0_w1x30_draw_000.yaml
RUN_NAME=ff_howfe_validation_cond_m2_hoke_0p1nm_xp1p0_yp0p0_w1x30_draw_000
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-m2_hoke_0p1nm_xp1p0_yp0p0_w1x30-d000 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/projects/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/validation/configs/ff_howfe_validation_cond_m2_hoke_0p1nm_xp1p0_yp0p0_w1x30_draw_000.yaml,RUN_NAME=ff_howfe_validation_cond_m2_hoke_0p1nm_xp1p0_yp0p0_w1x30_draw_000,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_validation_cond_m2_hoke_0p1nm_xm1p0_yp0p0_w1x30_draw_000
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/validation/configs/ff_howfe_validation_cond_m2_hoke_0p1nm_xm1p0_yp0p0_w1x30_draw_000.yaml
RUN_NAME=ff_howfe_validation_cond_m2_hoke_0p1nm_xm1p0_yp0p0_w1x30_draw_000
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-m2_hoke_0p1nm_xm1p0_yp0p0_w1x30-d000 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/projects/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/validation/configs/ff_howfe_validation_cond_m2_hoke_0p1nm_xm1p0_yp0p0_w1x30_draw_000.yaml,RUN_NAME=ff_howfe_validation_cond_m2_hoke_0p1nm_xm1p0_yp0p0_w1x30_draw_000,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_validation_cond_m2_hoke_0p1nm_xp0p0_yp1p0_w1x30_draw_000
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/validation/configs/ff_howfe_validation_cond_m2_hoke_0p1nm_xp0p0_yp1p0_w1x30_draw_000.yaml
RUN_NAME=ff_howfe_validation_cond_m2_hoke_0p1nm_xp0p0_yp1p0_w1x30_draw_000
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-m2_hoke_0p1nm_xp0p0_yp1p0_w1x30-d000 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/projects/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/validation/configs/ff_howfe_validation_cond_m2_hoke_0p1nm_xp0p0_yp1p0_w1x30_draw_000.yaml,RUN_NAME=ff_howfe_validation_cond_m2_hoke_0p1nm_xp0p0_yp1p0_w1x30_draw_000,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_validation_cond_m2_hoke_0p1nm_xp0p0_ym1p0_w1x30_draw_000
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/validation/configs/ff_howfe_validation_cond_m2_hoke_0p1nm_xp0p0_ym1p0_w1x30_draw_000.yaml
RUN_NAME=ff_howfe_validation_cond_m2_hoke_0p1nm_xp0p0_ym1p0_w1x30_draw_000
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-m2_hoke_0p1nm_xp0p0_ym1p0_w1x30-d000 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/projects/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/validation/configs/ff_howfe_validation_cond_m2_hoke_0p1nm_xp0p0_ym1p0_w1x30_draw_000.yaml,RUN_NAME=ff_howfe_validation_cond_m2_hoke_0p1nm_xp0p0_ym1p0_w1x30_draw_000,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_validation_cond_m1_hoke_0p1nm_xp1p0_yp0p0_w1x30_draw_000
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/validation/configs/ff_howfe_validation_cond_m1_hoke_0p1nm_xp1p0_yp0p0_w1x30_draw_000.yaml
RUN_NAME=ff_howfe_validation_cond_m1_hoke_0p1nm_xp1p0_yp0p0_w1x30_draw_000
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-m1_hoke_0p1nm_xp1p0_yp0p0_w1x30-d000 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/projects/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/validation/configs/ff_howfe_validation_cond_m1_hoke_0p1nm_xp1p0_yp0p0_w1x30_draw_000.yaml,RUN_NAME=ff_howfe_validation_cond_m1_hoke_0p1nm_xp1p0_yp0p0_w1x30_draw_000,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch
