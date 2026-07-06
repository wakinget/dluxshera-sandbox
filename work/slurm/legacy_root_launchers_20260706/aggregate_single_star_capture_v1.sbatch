#!/bin/bash
#SBATCH --job-name=sscap_agg
#SBATCH --partition=compute
#SBATCH --account=shera_hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/shera_hpc/%u/dluxshera/slurm_logs/sscap_agg-%j.out
#SBATCH --error=/scratch/shera_hpc/%u/dluxshera/slurm_logs/sscap_agg-%j.err

set -euo pipefail

source /cm/shared/apps/miniforge/etc/profile.d/conda.sh
conda activate dluxshera-py311
cd ~/dluxshera-sandbox

export DLUX_RESULTS=/scratch/shera_hpc/$USER/dluxshera
export SS_ROOT="$DLUX_RESULTS/single_star_capture_v1"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

export JAX_COMPILATION_CACHE_DIR=/scratch/shera_hpc/$USER/jax_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1

for run in \
  single_star_capture_jitter_001mas_32block_recovered20 \
  single_star_capture_jitter_010mas_32block_recovered20 \
  single_star_capture_jitter_020mas_32block_recovered20 \
  single_star_capture_jitter_050mas_32block_recovered20 \
  single_star_capture_jitter_100mas_32block_recovered20
do
  echo
  echo "===== aggregate-only: $run ====="
  RUN_ROOT="$SS_ROOT/$run"

  PYTHONPATH=src python examples/scripts/run_single_star_calibration_demo.py \
    --config "$RUN_ROOT/resolved_config.json" \
    --results-root "$SS_ROOT" \
    --aggregate-only \
    --quiet \
    --max-workers 1 \
    --no-resource-time
done
