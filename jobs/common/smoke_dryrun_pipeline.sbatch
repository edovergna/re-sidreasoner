#!/bin/bash
# Smoke test 2: verify training pipeline command generation via --dry-run.
# Runs train_pipeline.py --dry-run for all 8 ablation configs and checks that
# enabled stages produce correct subprocess commands and disabled stages are absent.
# CPU-only, no GPU needed.
#
# Usage (from repo root):
#   sbatch jobs/common/smoke_dryrun_pipeline.sbatch

#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --job-name=sid-smoke-pipeline
#SBATCH --output=jobs/outputs/%x-%j.out
#SBATCH --error=jobs/outputs/%x-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "${SLURM_SUBMIT_DIR}"
fi

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$(pwd)/baselines/repos/SIDReasoner/.venv}"
source baselines/repos/SIDReasoner/scripts/cluster_env.sh

echo "[smoke_dryrun_pipeline] Repo root: $(pwd)"
echo "[smoke_dryrun_pipeline] Venv: ${UV_PROJECT_ENVIRONMENT}"

# Install pytest into the venv if not already present
uv pip install --quiet pytest --python "${UV_PROJECT_ENVIRONMENT}/bin/python"

"${UV_PROJECT_ENVIRONMENT}/bin/python" -m pytest tests/common/smoke/test_dryrun_pipeline.py -v --tb=short
