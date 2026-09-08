#!/bin/bash
#SBATCH --account=shera_hpc
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch-jpl/shera_hpc/%u/dLuxShera-ML/render_v4/logs/%x-%A_%a.out
#SBATCH --error=/scratch-jpl/shera_hpc/%u/dLuxShera-ML/render_v4/logs/%x-%A_%a.err

set -euo pipefail

REPO_ROOT="${V4_REPO_ROOT:-/home/dmckeith/dluxshera-sandbox}"
PLAN_ROOT="${V4_PLAN_ROOT:?V4_PLAN_ROOT is required}"
OUTPUT_ROOT="${V4_OUTPUT_ROOT:?V4_OUTPUT_ROOT is required}"
SCRATCH_ROOT="${V4_SCRATCH_ROOT:-/scratch-jpl/shera_hpc/${USER}/dLuxShera-ML/render_v4}"
RENDERS_PER_TASK="${V4_RENDERS_PER_TASK:?V4_RENDERS_PER_TASK is required}"
RENDER_COUNT="${V4_RENDER_COUNT:?V4_RENDER_COUNT is required}"
CONDA_SH="${V4_CONDA_SH:-/cm/shared/apps/miniforge/etc/profile.d/conda.sh}"
CONDA_ENV="${V4_CONDA_ENV:-dluxshera-py311}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

mkdir -p "${SCRATCH_ROOT}/summaries" "${SCRATCH_ROOT}/jax_cache"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${SCRATCH_ROOT}/jax_cache}"
export XLA_FLAGS="${XLA_FLAGS:---xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK:-1}}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

cd "${REPO_ROOT}"

if command -v module >/dev/null 2>&1; then
  module load git >/dev/null 2>&1 || true
fi

if command -v git >/dev/null 2>&1; then
  ACTUAL_SHA="$(git rev-parse HEAD)"
  if [[ -n "${EXPECTED_REPO_SHA:-}" && "${ACTUAL_SHA}" != "${EXPECTED_REPO_SHA}" ]]; then
    echo "Repository SHA mismatch: expected ${EXPECTED_REPO_SHA}, got ${ACTUAL_SHA}" >&2
    exit 2
  fi
  export V4_RENDERER_REPOSITORY_SHA="${V4_RENDERER_REPOSITORY_SHA:-${ACTUAL_SHA}}"
elif [[ -n "${EXPECTED_REPO_SHA:-}" ]]; then
  export V4_RENDERER_REPOSITORY_SHA="${V4_RENDERER_REPOSITORY_SHA:-${EXPECTED_REPO_SHA}}"
fi

START_INDEX=$(( TASK_ID * RENDERS_PER_TASK ))
STOP_INDEX=$(( START_INDEX + RENDERS_PER_TASK ))
if (( STOP_INDEX > RENDER_COUNT )); then
  STOP_INDEX="${RENDER_COUNT}"
fi
if (( START_INDEX >= RENDER_COUNT )); then
  echo "Task ${TASK_ID} starts beyond render count ${RENDER_COUNT}; nothing to do." >&2
  exit 3
fi

SUMMARY_PATH="${SCRATCH_ROOT}/summaries/render_master_v4_${SLURM_ARRAY_JOB_ID:-manual}_${TASK_ID}.json"

python -m work.experiments.ml.datasets.render_master_v4 \
  --plan-root "${PLAN_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --start-index "${START_INDEX}" \
  --stop-index "${STOP_INDEX}" \
  --summary-path "${SUMMARY_PATH}" \
  --renderer-repository-sha "${V4_RENDERER_REPOSITORY_SHA:-}"
