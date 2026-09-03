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

if [[ -n "${S01_CONDA_SH:-}" ]]; then
  source "$S01_CONDA_SH"
elif [[ -f /cm/shared/apps/miniforge/etc/profile.d/conda.sh ]]; then
  source /cm/shared/apps/miniforge/etc/profile.d/conda.sh
fi

if [[ -n "${S01_CONDA_ENV:-}" ]]; then
  conda activate "$S01_CONDA_ENV"
elif [[ -n "${S01_CONDA_PREFIX:-}" ]]; then
  conda activate "$S01_CONDA_PREFIX"
else
  echo "Set S01_CONDA_ENV or S01_CONDA_PREFIX to a CUDA-enabled PyTorch environment." >&2
  exit 2
fi

if [[ "${S01_SCRATCH_SIDE:-}" != "jpl" && "${S01_SCRATCH_SIDE:-}" != "edge" ]]; then
  echo "Set S01_SCRATCH_SIDE to 'jpl' or 'edge'." >&2
  exit 2
fi

export PYTHONPATH="src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

SCRATCH_ROOT="${S01_SCRATCH_ROOT:-/scratch-${S01_SCRATCH_SIDE}/shera_hpc/$USER/dLuxShera-ML}"
PROJECT_RESULTS_ROOT="${S01_PROJECT_RESULTS_ROOT:-/projects/shera_hpc/$USER/dLuxShera-Results/ml}"
PROJECT_ARTIFACT_ROOT="${S01_PROJECT_ARTIFACT_ROOT:-$PROJECT_RESULTS_ROOT/S01/artifacts}"
STUDY="work/experiments/ml/s01/study.yaml"
PREPARED_ROOT="${S01_PREPARED_ROOT:-$SCRATCH_ROOT/data/PREP-V3-nuisance-v1}"
SPLIT_REGISTRY="${S01_SPLIT_REGISTRY:-$SCRATCH_ROOT/artifacts/S01/split/SPLIT-ML-v1.json}"
VALIDATION_MANIFEST="${S01_VALIDATION_MANIFEST:-$SCRATCH_ROOT/artifacts/S01/validation_pairs/S01-VALIDATION-PAIRS-v1}"
TEST_MANIFEST="${S01_TEST_MANIFEST:-$SCRATCH_ROOT/artifacts/S01/test_pairs/S01-TEST-PAIRS-v1}"
RUN_DIR="${S01_RUN_DIR:-$SCRATCH_ROOT/runs/S01/S01-E01/S01-E01-R001}"
PERSIST_DIR="${S01_PERSIST_DIR:-$PROJECT_RESULTS_ROOT/S01/S01-E01/S01-E01-R001}"

mkdir -p work/experiments/ml/s01/logs

if [[ -n "$(git status --short)" ]]; then
  echo "WARNING: Git worktree is dirty; run_manifest.json will record dirty=true." >&2
fi

python work/experiments/ml/s01/hpc/preflight_s01_gpu.py \
  --study "$STUDY" \
  --experiment-id S01-E01 \
  --prepared-root "$PREPARED_ROOT" \
  --split-registry "$SPLIT_REGISTRY" \
  --validation-manifest "$VALIDATION_MANIFEST" \
  --test-manifest "$TEST_MANIFEST" \
  --device cuda:0

mkdir -p \
  "$PROJECT_ARTIFACT_ROOT/split" \
  "$PROJECT_ARTIFACT_ROOT/validation_pairs" \
  "$PROJECT_ARTIFACT_ROOT/test_pairs"
rsync -a "$SPLIT_REGISTRY" "$PROJECT_ARTIFACT_ROOT/split/SPLIT-ML-v1.json"
rsync -a --delete "$VALIDATION_MANIFEST/" "$PROJECT_ARTIFACT_ROOT/validation_pairs/S01-VALIDATION-PAIRS-v1/"
rsync -a --delete "$TEST_MANIFEST/" "$PROJECT_ARTIFACT_ROOT/test_pairs/S01-TEST-PAIRS-v1/"

python work/experiments/ml/train_from_study.py \
  --study "$STUDY" \
  --experiment-id S01-E01 \
  --prepared-root "$PREPARED_ROOT" \
  --split-registry "$SPLIT_REGISTRY" \
  --validation-manifest "$VALIDATION_MANIFEST" \
  --test-manifest "$TEST_MANIFEST" \
  --output-dir "$RUN_DIR" \
  --device cuda:0 \
  --copy-final-to "$PERSIST_DIR"
