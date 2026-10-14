#!/bin/bash
# Parametric sbatch job for ablation training runs.
#
# Usage (submit from repo root):
#   sbatch --export=CONFIG=configs/grpo/a0_baseline/games.yaml \
#          jobs/common/submit_pipeline.sbatch
#
# Submit all ablations for a dataset at once:
#   for cfg in configs/grpo/*/games.yaml; do
#     sbatch --export=CONFIG=$cfg jobs/common/submit_pipeline.sbatch
#   done
#
# Override GPU count or time for RL-heavy runs:
#   sbatch --export=CONFIG=...,NPROC=8 --gpus=8 --time=48:00:00 jobs/common/submit_pipeline.sbatch

#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --job-name=sid-ablation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=07:00:00
#SBATCH --output=jobs/outputs/%x-%j.out
#SBATCH --error=jobs/outputs/%x-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "${SLURM_SUBMIT_DIR}"
fi

if [[ -z "${CONFIG:-}" ]]; then
    echo "[error] CONFIG env variable not set. Pass via --export=CONFIG=<path>"
    exit 1
fi

# Point uv at the venv built by setup_env.sh (inside the submodule, not the repo root)
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$(pwd)/baselines/repos/SIDReasoner/.venv}"

# Apply our local submodule modifications (kept as a patch in this repo, not
# committed upstream). Must run before sourcing cluster_env.sh — the patch edits it.
bash scripts/common/apply_submodule_patch.sh

# Load the cluster modules and set TORCHRUN_CMD / PYTHON_CMD
source baselines/repos/SIDReasoner/scripts/cluster_env.sh

echo "[submit_pipeline] Config: ${CONFIG}"
echo "[submit_pipeline] Job ID: ${SLURM_JOB_ID:-local}"

${PYTHON_CMD} scripts/common/train/train_pipeline.py --config "${CONFIG}"
