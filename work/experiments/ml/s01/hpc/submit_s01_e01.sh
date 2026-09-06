#!/bin/bash
set -euo pipefail

REPO_ROOT="${S01_REPO_ROOT:-$HOME/dluxshera-sandbox}"
cd "$REPO_ROOT"

if [[ "${S01_SCRATCH_SIDE:-}" != "jpl" && "${S01_SCRATCH_SIDE:-}" != "edge" ]]; then
  echo "Set S01_SCRATCH_SIDE to 'jpl' or 'edge' before submitting." >&2
  exit 2
fi

if [[ -z "${S01_GPU_SBATCH_ARGS:-}" ]]; then
  cat >&2 <<'EOF'
Set S01_GPU_SBATCH_ARGS to the site-verified GPU scheduling options.
Examples, only after confirming with sinfo/scontrol on Gattaca2:
  export S01_GPU_SBATCH_ARGS="--partition=<gpu_partition> --gres=<gpu_resource>"
  export S01_GPU_SBATCH_ARGS="--partition=<gpu_partition> --gpus=1"
This wrapper intentionally does not invent Gattaca2 GPU partition syntax.
EOF
  exit 2
fi

mkdir -p work/experiments/ml/s01/logs

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

cluster_args=()
if [[ -n "${S01_SLURM_CLUSTER:-}" ]]; then
  cluster_args=(-M "$S01_SLURM_CLUSTER")
fi

read -r -a gpu_args <<< "$S01_GPU_SBATCH_ARGS"
sbatch_args=(--parsable)
if [[ ${#cluster_args[@]} -gt 0 ]]; then
  sbatch_args+=("${cluster_args[@]}")
fi
sbatch_args+=("${gpu_args[@]}")
sbatch_args+=(work/experiments/ml/s01/hpc/train_s01_e01.sbatch)
submit_output="$(sbatch "${sbatch_args[@]}")"
printf '%s\n' "$submit_output"
job_id="$(printf '%s\n' "$submit_output" | awk '/^[0-9]+(;[A-Za-z0-9_.-]+)?$/ { split($0, parts, ";"); id=parts[1] } END {print id}')"
if [[ -z "$job_id" ]]; then
  echo "Could not parse Slurm job ID from sbatch output." >&2
  exit 1
fi
echo "Parsed Slurm job ID: $job_id"
