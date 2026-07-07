#!/bin/bash
#SBATCH --job-name=ff-iterative
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G

set -euo pipefail

REPO="${REPO:-$HOME/dluxshera-sandbox}"
CONFIG="${CONFIG:?Set CONFIG to a full-fidelity iterative campaign YAML}"
RUN_NAME="${RUN_NAME:?Set RUN_NAME for this campaign shard}"
RESULTS_ROOT="${RESULTS_ROOT:-/scratch/shera_hpc/$USER/dluxshera}"
RUN_RESULTS_ROOT="${RESULTS_ROOT%/}/observation_bias_campaign"
MAX_WORKERS="${MAX_WORKERS:-5}"
FAIL_FAST="${FAIL_FAST:-1}"
ANALYZE_AFTER_RUN="${ANALYZE_AFTER_RUN:-1}"
USE_RESOURCE_TIME="${USE_RESOURCE_TIME:-1}"
PROFILE_RUNTIME="${PROFILE_RUNTIME:-0}"
PROFILE_RUNTIME_DETAIL="${PROFILE_RUNTIME_DETAIL:-basic}"
MEMORY_DIAGNOSTICS="${MEMORY_DIAGNOSTICS:-0}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export JAX_COMPILATION_CACHE_DIR="/scratch/shera_hpc/$USER/jax_cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1

mkdir -p "$RUN_RESULTS_ROOT" "$JAX_COMPILATION_CACHE_DIR" "$RESULTS_ROOT/slurm_logs" logs
cd "$REPO"

# Use the shared cluster Miniforge install explicitly, but activate by prefix.
# Important: /scratch is side-dependent on Gattaca2. On Edge, /scratch points
# to /scratch-edge, so use /scratch-jpl explicitly for the JPL-side env.
source /cm/shared/apps/miniforge/etc/profile.d/conda.sh

DLUXSHERA_CONDA_ENV_PREFIX="${DLUXSHERA_CONDA_ENV_PREFIX:-/scratch-jpl/shera_hpc/dmckeith/conda/envs/dluxshera-py311}"
conda activate "$DLUXSHERA_CONDA_ENV_PREFIX"

export PYTHONPATH="${PYTHONPATH:-src}"

echo "Conda env: ${CONDA_DEFAULT_ENV:-unset}"
echo "CONDA_PREFIX: ${CONDA_PREFIX:-unset}"
echo "Python executable: $(which python)"
python - <<'PYENV'
import sys
print("sys.executable:", sys.executable)
import jax
print("jax:", jax.__version__)
import dluxshera
print("dluxshera import ok")
PYENV

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
if [[ "$USE_RESOURCE_TIME" == "1" ]]; then
  RUN_ARGS+=(--resource-time auto)
else
  RUN_ARGS+=(--no-resource-time)
fi
if [[ "$PROFILE_RUNTIME" == "1" ]]; then
  RUN_ARGS+=(--profile-runtime --profile-runtime-detail "$PROFILE_RUNTIME_DETAIL")
fi
if [[ "$MEMORY_DIAGNOSTICS" == "1" ]]; then
  RUN_ARGS+=(--memory-diagnostics)
fi

python examples/scripts/run_full_fidelity_binary_iterative_campaign.py "${RUN_ARGS[@]}"

if [[ "$ANALYZE_AFTER_RUN" == "1" ]]; then
  RUN_ROOT="$RUN_RESULTS_ROOT/$RUN_NAME"
  python examples/scripts/analyze_full_fidelity_binary_iterative_campaign.py \
    --run-root "$RUN_ROOT" \
    --outdir "$RUN_ROOT/analysis/full_fidelity_review" \
    --strict \
    --max-image-examples 4
fi
