#!/bin/bash
#SBATCH --job-name=full_fidelity_runtime_benchmark_recovered_2x20f_v1
#SBATCH --output=/projects/shera_hpc/dmckeith/dLuxShera-Results/slurm_logs/%x-%j.out
#SBATCH --error=/projects/shera_hpc/dmckeith/dLuxShera-Results/slurm_logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G

set -euo pipefail

REPO="${REPO:-$HOME/dluxshera-sandbox}"
RESULTS_ROOT="${RESULTS_ROOT:-/projects/shera_hpc/dmckeith/dLuxShera-Results}"
RUN_RESULTS_ROOT="${RESULTS_ROOT%/}/observation_bias_campaign"
CONFIG="${CONFIG:-examples/recipes/full_fidelity_next_campaigns_20260703/full_fidelity_runtime_benchmark_recovered_2x20f_v1.yaml}"
RUN_NAME="${RUN_NAME:-full_fidelity_runtime_benchmark_recovered_2x20f_v1}"
MAX_WORKERS="${MAX_WORKERS:-1}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export JAX_COMPILATION_CACHE_DIR="/scratch/shera_hpc/$USER/jax_cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1

mkdir -p "$RUN_RESULTS_ROOT" "$JAX_COMPILATION_CACHE_DIR" "$RESULTS_ROOT/slurm_logs"
cd "$REPO"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dluxshera-py311

PYTHONPATH=src python examples/scripts/run_full_fidelity_binary_iterative_campaign.py \
  --config "$CONFIG" \
  --run-name "$RUN_NAME" \
  --results-root "$RUN_RESULTS_ROOT" \
  --max-workers "$MAX_WORKERS" \
  --fail-fast \
  --resource-time auto \
  --profile-runtime \
  --profile-runtime-detail basic \
  --memory-diagnostics
