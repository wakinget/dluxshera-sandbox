#!/bin/bash
#SBATCH -J rt-recov20
#SBATCH -p compute
#SBATCH -A shera_hpc
#SBATCH -n 4
#SBATCH --mem=96G
#SBATCH -t 10:00:00
#SBATCH -o /projects/shera_hpc/dmckeith/dLuxShera-Results/slurm_logs/runtime_benchmark_recovered20-%j.out
#SBATCH -e /projects/shera_hpc/dmckeith/dLuxShera-Results/slurm_logs/runtime_benchmark_recovered20-%j.err

set -eo pipefail

# Do not enable nounset until after cluster/conda shell init.
# /etc/bashrc may reference variables before defining them.
set +u
source ~/.bashrc

# Initialize conda for non-interactive Slurm shells.
eval "$(conda shell.bash hook)"
conda activate dluxshera-py311

set -u

cd ~/dluxshera-sandbox

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export JAX_COMPILATION_CACHE_DIR="/scratch/shera_hpc/$USER/jax_cache"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

export RESULTS_ROOT=/projects/shera_hpc/dmckeith/dLuxShera-Results
export BENCH_RUN=runtime_benchmark_recovered20_2subblocks_profile_exec_v2
export BENCH_CONFIG=work/full_fidelity_runtime_benchmarks/runtime_benchmark_recovered20_2subblocks_profile_v1.json

PYTHONPATH=src python examples/scripts/run_full_fidelity_binary_iterative_campaign.py \
  --config "$BENCH_CONFIG" \
  --results-root "$RESULTS_ROOT/observation_bias_campaign" \
  --run-name "$BENCH_RUN" \
  --max-workers 1 \
  --resource-time \
  --profile-runtime \
  --profile-runtime-detail basic \
  --memory-diagnostics
