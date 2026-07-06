#!/bin/bash
#SBATCH --job-name=ss_fastagg
#SBATCH --partition=compute
#SBATCH --account=shera_hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/shera_hpc/%u/dluxshera/slurm_logs/ss_fastagg-%j.out
#SBATCH --error=/scratch/shera_hpc/%u/dluxshera/slurm_logs/ss_fastagg-%j.err

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

fast_aggregate_root () {
  local ROOT="$1"
  local GLOB_PREFIX="$2"

  echo
  echo "################################################################################"
  echo "Fast aggregate root: $ROOT"
  echo "################################################################################"

  for RUN_ROOT in "$ROOT"/"$GLOB_PREFIX"*; do
    [ -d "$RUN_ROOT" ] || continue
    [ -f "$RUN_ROOT/resolved_config.json" ] || continue

    run="$(basename "$RUN_ROOT")"

    if [ -f "$RUN_ROOT/cases/zero_bias_case/posterior_by_parameter.csv" ] && \
       [ -f "$RUN_ROOT/cases/zero_bias_case/posterior_history.csv" ]; then
      echo "===== skipping existing posterior products: $run ====="
      continue
    fi

    TMP_CFG="$(mktemp --suffix=.json)"
    python - "$RUN_ROOT/resolved_config.json" "$TMP_CFG" <<'PY'
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
with open(src, "r", encoding="utf-8") as f:
    cfg = json.load(f)

exp = cfg.setdefault("experiment", {})
exp.setdefault("forecast", {})["enabled"] = False
exp.setdefault("eigenbasis", {})["enabled"] = False

with open(dst, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
PY

    echo
    echo "===== fast aggregate-only: $run ====="
    PYTHONPATH=src python examples/scripts/run_single_star_calibration_demo.py \
      --config "$TMP_CFG" \
      --results-root "$ROOT" \
      --aggregate-only \
      --quiet \
      --max-workers 1 \
      --no-resource-time

    rm -f "$TMP_CFG"
  done
}

fast_aggregate_root "$DLUX_RESULTS/single_star_knowledge_capture_v1" "single_star_knowledge_capture_v1_"
fast_aggregate_root "$DLUX_RESULTS/single_star_truth_wfe_capture_v1" "single_star_truth_wfe_capture_v1_"
