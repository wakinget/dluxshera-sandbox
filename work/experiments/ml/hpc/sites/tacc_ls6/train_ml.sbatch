#!/bin/bash
#SBATCH --account=JPL-PUB
#SBATCH --partition=gpu-a100-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=work/experiments/ml/hpc/logs/%x-%j.out
#SBATCH --error=work/experiments/ml/hpc/logs/%x-%j.err

set -euo pipefail

export ML_SITE="${ML_SITE:-tacc_ls6}"
export ML_TACC_ISOLATE_PYTHON=1
REPO_ROOT="${ML_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
exec bash "$REPO_ROOT/work/experiments/ml/hpc/run_study_training.sh"
