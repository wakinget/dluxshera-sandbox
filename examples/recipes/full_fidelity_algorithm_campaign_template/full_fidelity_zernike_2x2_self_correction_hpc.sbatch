#!/bin/bash
#SBATCH --job-name=ff-zernike-2x2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

REPO="${REPO:-$HOME/dluxshera-sandbox}"
CONFIG="${CONFIG:-examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_zernike_2x2_self_correction_hpc_v1.yaml}"
RUN_NAME="${RUN_NAME:-full_fidelity_zernike_2x2_self_correction_hpc_v1}"
RESULTS_ROOT="${RESULTS_ROOT:-/scratch/shera_hpc/$USER/dluxshera}"
RUN_RESULTS_ROOT="${RESULTS_ROOT%/}/observation_bias_campaign"
MAX_WORKERS="${MAX_WORKERS:-4}"
FAIL_FAST="${FAIL_FAST:-1}"
ANALYZE_AFTER_RUN="${ANALYZE_AFTER_RUN:-1}"
USE_RESOURCE_TIME="${USE_RESOURCE_TIME:-0}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export JAX_COMPILATION_CACHE_DIR="/scratch/shera_hpc/$USER/jax_cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1

mkdir -p "$RUN_RESULTS_ROOT" "$JAX_COMPILATION_CACHE_DIR" logs
cd "$REPO"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dluxshera-py311

RUN_ARGS=(
  --config "$CONFIG"
  --run-name "$RUN_NAME"
  --results-root "$RUN_RESULTS_ROOT"
  --max-workers "$MAX_WORKERS"
)
if [[ "$FAIL_FAST" == "1" ]]; then
  RUN_ARGS+=(--fail-fast)
else
  RUN_ARGS+=(--no-fail-fast)
fi
if [[ "$USE_RESOURCE_TIME" != "1" ]]; then
  RUN_ARGS+=(--no-resource-time)
fi

PYTHONPATH=src python examples/scripts/run_full_fidelity_binary_iterative_campaign.py "${RUN_ARGS[@]}"

if [[ "$ANALYZE_AFTER_RUN" == "1" ]]; then
  RUN_ROOT="$RUN_RESULTS_ROOT/$RUN_NAME"
  PYTHONPATH=src python examples/scripts/analyze_full_fidelity_binary_iterative_campaign.py \
    --run-root "$RUN_ROOT" \
    --outdir "$RUN_ROOT/analysis/full_fidelity_review" \
    --strict \
    --max-image-examples 4
fi
