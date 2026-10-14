#!/bin/bash
# Parametric sbatch job for ablation EVALUATION.
#
# Usage (submit from repo root):
#   sbatch --export=CONFIG=configs/grpo/a0_baseline/games.yaml \
#          jobs/common/submit_eval.sbatch
#
# Picks the evaluation mode automatically from the config:
#   - alignment_only            -> beam-search eval (evaluate_Qwen3.py)
#   - all other ablations       -> thinking-mode vLLM eval (evaluate_Qwen3_think.py)
# For S3 (RL) ablations it first merges the latest verl FSDP checkpoint
# (stage3_rl/global_step_N/actor) into a single HF model, then evaluates it.

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=sid-eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
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

# Checkpoint merging imports verl from the baseline source tree.
export PYTHONPATH="$(pwd)/baselines/repos/SIDReasoner:$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"

echo "[submit_eval] Config: ${CONFIG}"
echo "[submit_eval] Job ID: ${SLURM_JOB_ID:-local}"

${PYTHON_CMD} scripts/common/eval/evaluate.py --config "${CONFIG}"
