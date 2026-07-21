#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on the cluster.
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate dluxshera-py311

RESULTS_ROOT="${RESULTS_ROOT:-/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results}"
mkdir -p "$RESULTS_ROOT/slurm_logs"

# ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_000
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_000.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_000
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp0p0_yp0p0_w10x30-d000 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_000.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_000,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_001
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_001.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_001
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp0p0_yp0p0_w10x30-d001 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_001.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_001,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_002
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_002.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_002
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp0p0_yp0p0_w10x30-d002 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_002.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_002,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_003
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_003.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_003
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp0p0_yp0p0_w10x30-d003 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_003.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_003,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_004
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_004.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_004
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp0p0_yp0p0_w10x30-d004 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_004.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_004,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_005
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_005.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_005
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp0p0_yp0p0_w10x30-d005 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_005.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_005,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_006
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_006.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_006
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp0p0_yp0p0_w10x30-d006 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_006.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_006,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_007
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_007.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_007
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp0p0_yp0p0_w10x30-d007 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_007.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_007,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_008
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_008.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_008
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp0p0_yp0p0_w10x30-d008 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_008.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_008,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_009
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_009.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_009
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp0p0_yp0p0_w10x30-d009 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_009.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp0p0_w10x30_draw_009,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_000
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_000.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_000
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp1p0_yp0p0_w10x30-d000 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_000.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_000,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_001
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_001.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_001
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp1p0_yp0p0_w10x30-d001 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_001.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_001,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_002
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_002.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_002
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp1p0_yp0p0_w10x30-d002 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_002.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_002,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_003
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_003.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_003
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp1p0_yp0p0_w10x30-d003 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_003.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_003,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_004
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_004.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_004
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp1p0_yp0p0_w10x30-d004 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_004.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_004,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_005
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_005.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_005
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp1p0_yp0p0_w10x30-d005 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_005.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_005,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_006
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_006.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_006
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp1p0_yp0p0_w10x30-d006 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_006.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_006,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_007
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_007.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_007
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp1p0_yp0p0_w10x30-d007 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_007.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_007,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_008
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_008.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_008
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp1p0_yp0p0_w10x30-d008 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_008.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_008,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_009
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_009.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_009
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp1p0_yp0p0_w10x30-d009 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_009.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp1p0_yp0p0_w10x30_draw_009,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_000
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_000.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_000
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xm1p0_yp0p0_w10x30-d000 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_000.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_000,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_001
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_001.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_001
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xm1p0_yp0p0_w10x30-d001 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_001.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_001,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_002
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_002.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_002
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xm1p0_yp0p0_w10x30-d002 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_002.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_002,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_003
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_003.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_003
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xm1p0_yp0p0_w10x30-d003 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_003.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_003,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_004
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_004.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_004
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xm1p0_yp0p0_w10x30-d004 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_004.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_004,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_005
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_005.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_005
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xm1p0_yp0p0_w10x30-d005 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_005.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_005,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_006
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_006.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_006
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xm1p0_yp0p0_w10x30-d006 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_006.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_006,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_007
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_007.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_007
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xm1p0_yp0p0_w10x30-d007 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_007.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_007,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_008
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_008.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_008
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xm1p0_yp0p0_w10x30-d008 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_008.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_008,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_009
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_009.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_009
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xm1p0_yp0p0_w10x30-d009 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_009.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xm1p0_yp0p0_w10x30_draw_009,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_000
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_000.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_000
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp0p0_yp1p0_w10x30-d000 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_000.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_000,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_001
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_001.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_001
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp0p0_yp1p0_w10x30-d001 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_001.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_001,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_002
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_002.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_002
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp0p0_yp1p0_w10x30-d002 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_002.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_002,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_003
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_003.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_003
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp0p0_yp1p0_w10x30-d003 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_003.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_003,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_004
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_004.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_004
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp0p0_yp1p0_w10x30-d004 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_004.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_004,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_005
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_005.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_005
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp0p0_yp1p0_w10x30-d005 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_005.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_005,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_006
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_006.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_006
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp0p0_yp1p0_w10x30-d006 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_006.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_006,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_007
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_007.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_007
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp0p0_yp1p0_w10x30-d007 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_007.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_007,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_008
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_008.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_008
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp0p0_yp1p0_w10x30-d008 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_008.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_008,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_009
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_009.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_009
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp0p0_yp1p0_w10x30-d009 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_009.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_yp1p0_w10x30_draw_009,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_000
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_000.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_000
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp0p0_ym1p0_w10x30-d000 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_000.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_000,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_001
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_001.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_001
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp0p0_ym1p0_w10x30-d001 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_001.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_001,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_002
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_002.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_002
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp0p0_ym1p0_w10x30-d002 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_002.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_002,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_003
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_003.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_003
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp0p0_ym1p0_w10x30-d003 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_003.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_003,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_004
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_004.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_004
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp0p0_ym1p0_w10x30-d004 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_004.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_004,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_005
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_005.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_005
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp0p0_ym1p0_w10x30-d005 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_005.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_005,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_006
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_006.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_006
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp0p0_ym1p0_w10x30-d006 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_006.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_006,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_007
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_007.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_007
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp0p0_ym1p0_w10x30-d007 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_007.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_007,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_008
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_008.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_008
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp0p0_ym1p0_w10x30-d008 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_008.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_008,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch

# ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_009
CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_009.yaml
RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_009
MAX_WORKERS=5
export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT
sbatch --time=12:00:00 --cpus-per-task=10 --mem=128G --job-name=ff-noke_xp0p0_ym1p0_w10x30-d009 --output="$RESULTS_ROOT/slurm_logs/%x-%j.out" --error="$RESULTS_ROOT/slurm_logs/%x-%j.err" --export=ALL,RESULTS_ROOT=/scratch-jpl/shera_hpc/dmckeith/dLuxShera-Results,CONFIG=examples/recipes/full_fidelity_howfe_field_dither_20260720/shards/controls/configs/ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_009.yaml,RUN_NAME=ff_howfe_controls_cond_noke_xp0p0_ym1p0_w10x30_draw_009,MAX_WORKERS=5,FAIL_FAST=1,ANALYZE_AFTER_RUN=1,USE_RESOURCE_TIME=1 examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_iterative_campaign_hpc.sbatch
