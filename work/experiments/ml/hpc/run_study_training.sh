#!/bin/bash
set -euo pipefail

REPO_ROOT="${ML_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"

if [[ "${ML_TACC_ISOLATE_PYTHON:-0}" == "1" ]]; then
  module unload python3/3.9.7 2>/dev/null || true
  unset PYTHONPATH
  unset PYTHONHOME
  export PYTHONNOUSERSITE=1
fi

if [[ -n "${ML_CONDA_SH:-}" ]]; then
  source "$ML_CONDA_SH"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is unavailable after initialization. Set ML_CONDA_SH to the site conda.sh setup script or load a module that defines conda." >&2
  exit 2
fi

if [[ -n "${ML_CONDA_ENV:-}" ]]; then
  conda activate "$ML_CONDA_ENV"
elif [[ -n "${ML_CONDA_PREFIX:-}" ]]; then
  conda activate "$ML_CONDA_PREFIX"
else
  echo "Set ML_CONDA_ENV or ML_CONDA_PREFIX to a CUDA-enabled PyTorch environment before launching the batch job." >&2
  exit 2
fi

export PYTHONPATH="src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

: "${ML_STUDY_PATH:?Set ML_STUDY_PATH.}"
: "${ML_EXPERIMENT_ID:?Set ML_EXPERIMENT_ID.}"
: "${ML_PREPARED_ROOT:?Set ML_PREPARED_ROOT.}"
: "${ML_SPLIT_REGISTRY:?Set ML_SPLIT_REGISTRY.}"
: "${ML_VALIDATION_MANIFEST:?Set ML_VALIDATION_MANIFEST.}"
: "${ML_TEST_MANIFEST:?Set ML_TEST_MANIFEST.}"
: "${ML_RUN_DIR:?Set ML_RUN_DIR.}"

ML_DEVICE="${ML_DEVICE:-cuda:0}"
run_args=()
if [[ -n "${ML_RUN_ID:-}" ]]; then
  run_args+=(--run-id "$ML_RUN_ID")
fi
preflight_args=()
if [[ -n "${ML_PERSIST_ARTIFACT_ROOT:-}" ]]; then
  preflight_args+=(--persist-artifact-root "$ML_PERSIST_ARTIFACT_ROOT")
fi

copy_compact_outputs() {
  if [[ -z "${ML_PERSIST_DIR:-}" || ! -d "$ML_RUN_DIR" ]]; then
    return 0
  fi
  mkdir -p "$ML_PERSIST_DIR"
  local name
  for name in \
    run_manifest.json \
    run_config_resolved.json \
    history.csv \
    metrics.json \
    evaluation_predictions.npz \
    checkpoint_best.pt \
    checkpoint_last.pt
  do
    if [[ -e "$ML_RUN_DIR/$name" ]]; then
      cp -p "$ML_RUN_DIR/$name" "$ML_PERSIST_DIR/$name"
    fi
  done
}

trap copy_compact_outputs EXIT

python work/experiments/ml/hpc/preflight_ml_gpu.py \
  --study "$ML_STUDY_PATH" \
  --experiment-id "$ML_EXPERIMENT_ID" \
  "${run_args[@]}" \
  --prepared-root "$ML_PREPARED_ROOT" \
  --split-registry "$ML_SPLIT_REGISTRY" \
  --validation-manifest "$ML_VALIDATION_MANIFEST" \
  --test-manifest "$ML_TEST_MANIFEST" \
  --device "$ML_DEVICE" \
  "${preflight_args[@]}"

train_args=()
if [[ -n "${ML_PERSIST_DIR:-}" ]]; then
  train_args+=(--copy-final-to "$ML_PERSIST_DIR")
fi
if [[ -n "${ML_RESUME_CHECKPOINT:-}" ]]; then
  train_args+=(--resume-checkpoint "$ML_RESUME_CHECKPOINT")
fi
if [[ "${ML_OVERWRITE:-0}" == "1" ]]; then
  train_args+=(--overwrite)
fi

python work/experiments/ml/train_from_study.py \
  --study "$ML_STUDY_PATH" \
  --experiment-id "$ML_EXPERIMENT_ID" \
  "${run_args[@]}" \
  --prepared-root "$ML_PREPARED_ROOT" \
  --split-registry "$ML_SPLIT_REGISTRY" \
  --validation-manifest "$ML_VALIDATION_MANIFEST" \
  --test-manifest "$ML_TEST_MANIFEST" \
  --output-dir "$ML_RUN_DIR" \
  --device "$ML_DEVICE" \
  "${train_args[@]}"
