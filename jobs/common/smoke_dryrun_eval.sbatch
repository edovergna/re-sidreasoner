#!/bin/bash
# Smoke test 3: verify evaluation mode selection and checkpoint inference.
# Tests uses_thinking() and infer_checkpoint() directly, then runs evaluate.py
# --dry-run for all 8 configs and checks think mode + eval script selection.
# CPU-only, no GPU needed.
#
# Usage (from repo root):
#   sbatch jobs/common/smoke_dryrun_eval.sbatch

#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --job-name=sid-smoke-eval
#SBATCH --output=jobs/outputs/%x-%j.out
#SBATCH --error=jobs/outputs/%x-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "${SLURM_SUBMIT_DIR}"
fi

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$(pwd)/baselines/repos/SIDReasoner/.venv}"
source baselines/repos/SIDReasoner/scripts/cluster_env.sh

echo "[smoke_dryrun_eval] Repo root: $(pwd)"
echo "[smoke_dryrun_eval] Venv: ${UV_PROJECT_ENVIRONMENT}"

# Install pytest into the venv if not already present
uv pip install --quiet pytest --python "${UV_PROJECT_ENVIRONMENT}/bin/python"

"${UV_PROJECT_ENVIRONMENT}/bin/python" -m pytest tests/common/smoke/test_dryrun_eval.py -v --tb=short
