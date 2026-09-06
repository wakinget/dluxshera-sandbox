#!/bin/bash
#SBATCH --account=shera_hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=work/experiments/ml/hpc/logs/%x-%j.out
#SBATCH --error=work/experiments/ml/hpc/logs/%x-%j.err

set -euo pipefail

export ML_SITE="${ML_SITE:-gattaca2}"
REPO_ROOT="${ML_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
exec bash "$REPO_ROOT/work/experiments/ml/hpc/run_study_training.sh"
