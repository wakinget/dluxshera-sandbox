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

cluster_args=()
if [[ -n "${S01_SLURM_CLUSTER:-}" ]]; then
  cluster_args=(-M "$S01_SLURM_CLUSTER")
fi

read -r -a gpu_args <<< "$S01_GPU_SBATCH_ARGS"
exec sbatch "${cluster_args[@]}" "${gpu_args[@]}" work/experiments/ml/s01/hpc/train_s01_e01.sbatch
