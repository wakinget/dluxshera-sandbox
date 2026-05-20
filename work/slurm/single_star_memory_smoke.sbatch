#!/usr/bin/env bash
#SBATCH --job-name=single-star-smoke
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --account=shera_hpc
#SBATCH --output=/scratch/shera_hpc/dmckeith/dluxshera/slurm_logs/%x-%j.out
#SBATCH --error=/scratch/shera_hpc/dmckeith/dluxshera/slurm_logs/%x-%j.err

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID}"
echo "Job name: ${SLURM_JOB_NAME}"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Submitted from: ${SLURM_SUBMIT_DIR}"

source /cm/shared/apps/miniforge/etc/profile.d/conda.sh
conda activate dluxshera-py311

cd "$HOME/dluxshera-sandbox"

export DLUX_RESULTS=/scratch/shera_hpc/$USER/dluxshera
mkdir -p "$DLUX_RESULTS/slurm_logs"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

echo "Python: $(which python)"
python --version

python -c "import jax; print('JAX:', jax.__version__); print('devices:', jax.devices())"
python -c "import dLux; print('dLux import OK')"
python -c "import dluxshera; print('dluxshera import OK')"

python examples/scripts/run_single_star_calibration_demo.py \
  --results-root "$DLUX_RESULTS/single_star_calibration_demo" \
  --run-name hpc_single_star_sbatch_memory_smoke \
  --n-subblocks 1 \
  --n-frames 3 \
  --noise disabled \
  --phi-ref truth_when_available \
  --reference-diagnostics-profile none \
  --max-workers 1 \
  --memory-diagnostics

echo "Finished: $(date)"
