#!/bin/bash
#SBATCH --job-name=binary_iter_v1
#SBATCH --partition=compute
#SBATCH --account=shera_hpc
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

source /cm/shared/apps/miniforge/etc/profile.d/conda.sh
conda activate dluxshera-py311
cd ~/dluxshera-sandbox

export DLUX_RESULTS=/scratch/shera_hpc/$USER/dluxshera
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export JAX_COMPILATION_CACHE_DIR=/scratch/shera_hpc/$USER/jax_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export PYTHONPATH=src

CONFIG=examples/recipes/observation_bias_campaign_template/binary_iterative_cluster_validation.yaml
RUN_NAME=binary_iterative_cluster_validation_v1

# Dry-run first outside sbatch, or uncomment for a scheduler-side preflight.
# python3 examples/scripts/run_observation_bias_campaign.py \
#   --config "$CONFIG" \
#   --results-root "$DLUX_RESULTS" \
#   --run-name "$RUN_NAME" \
#   --dry-run

python3 examples/scripts/run_observation_bias_campaign.py \
  --config "$CONFIG" \
  --results-root "$DLUX_RESULTS" \
  --run-name "$RUN_NAME" \
  --max-workers 1 \
  --resource-time auto

python3 examples/scripts/run_observation_bias_campaign.py \
  --config "$CONFIG" \
  --results-root "$DLUX_RESULTS" \
  --run-name "$RUN_NAME" \
  --aggregate-only
