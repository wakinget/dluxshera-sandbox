#!/bin/bash
#SBATCH --job-name=S01-E01-R001
#SBATCH --account=shera_hpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=work/experiments/ml/s01/logs/%x-%j.out
#SBATCH --error=work/experiments/ml/s01/logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${S01_REPO_ROOT:-$HOME/dluxshera-sandbox}"
cd "$REPO_ROOT"

if [[ "${S01_SCRATCH_SIDE:-}" != "jpl" && "${S01_SCRATCH_SIDE:-}" != "edge" ]]; then
  echo "Set S01_SCRATCH_SIDE to 'jpl' or 'edge'." >&2
  exit 2
fi

SCRATCH_ROOT="${S01_SCRATCH_ROOT:-/scratch-${S01_SCRATCH_SIDE}/shera_hpc/$USER/dLuxShera-ML}"
PROJECT_RESULTS_ROOT="${S01_PROJECT_RESULTS_ROOT:-/projects/shera_hpc/$USER/dLuxShera-Results/ml}"
S01_RUN_ID_RESOLVED="${S01_RUN_ID:-S01-E01-R001}"
S01_CONDA_SH_RESOLVED="${S01_CONDA_SH:-}"
if [[ -z "$S01_CONDA_SH_RESOLVED" ]] && [[ -f /cm/shared/apps/miniforge/etc/profile.d/conda.sh ]]; then
  S01_CONDA_SH_RESOLVED=/cm/shared/apps/miniforge/etc/profile.d/conda.sh
fi

export ML_REPO_ROOT="$REPO_ROOT"
export ML_CONDA_SH="$S01_CONDA_SH_RESOLVED"
export ML_CONDA_ENV="${S01_CONDA_ENV:-}"
export ML_CONDA_PREFIX="${S01_CONDA_PREFIX:-}"
export ML_STUDY_PATH="work/experiments/ml/s01/study.yaml"
export ML_EXPERIMENT_ID="S01-E01"
export ML_RUN_ID="$S01_RUN_ID_RESOLVED"
export ML_PREPARED_ROOT="${S01_PREPARED_ROOT:-$SCRATCH_ROOT/data/PREP-V3-nuisance-v1}"
export ML_SPLIT_REGISTRY="${S01_SPLIT_REGISTRY:-$SCRATCH_ROOT/artifacts/S01/split/SPLIT-ML-v1.json}"
export ML_VALIDATION_MANIFEST="${S01_VALIDATION_MANIFEST:-$SCRATCH_ROOT/artifacts/S01/validation_pairs/S01-VALIDATION-PAIRS-v1}"
export ML_TEST_MANIFEST="${S01_TEST_MANIFEST:-$SCRATCH_ROOT/artifacts/S01/test_pairs/S01-TEST-PAIRS-v1}"
export ML_RUN_DIR="${S01_RUN_DIR:-$SCRATCH_ROOT/runs/S01/S01-E01/$S01_RUN_ID_RESOLVED}"
export ML_PERSIST_DIR="${S01_PERSIST_DIR:-$PROJECT_RESULTS_ROOT/S01/S01-E01/$S01_RUN_ID_RESOLVED}"
export ML_PERSIST_ARTIFACT_ROOT="${S01_PROJECT_ARTIFACT_ROOT:-$PROJECT_RESULTS_ROOT/S01/artifacts}"
export ML_DEVICE="${S01_DEVICE:-cuda:0}"
export ML_RESUME_CHECKPOINT="${S01_RESUME_CHECKPOINT:-}"
export ML_OVERWRITE="${S01_OVERWRITE:-}"
export ML_SOURCE_COMMIT="${S01_SOURCE_COMMIT:-}"
export DLUXSHERA_SOURCE_COMMIT="${S01_SOURCE_COMMIT:-}"
export ML_SOURCE_ARCHIVE_ID="${S01_SOURCE_ARCHIVE_ID:-}"
export DLUXSHERA_SOURCE_ARCHIVE_ID="${S01_SOURCE_ARCHIVE_ID:-}"

mkdir -p work/experiments/ml/s01/logs
exec bash "$REPO_ROOT/work/experiments/ml/hpc/run_study_training.sh"
