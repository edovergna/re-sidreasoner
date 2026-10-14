#!/bin/bash
# Smoke test 1: verify all YAML-referenced paths resolve to existing files/directories.
# Covers all 8 ablation configs (4 types × games + office). CPU-only, no GPU needed.
#
# Usage (from repo root):
#   sbatch jobs/common/smoke_config_paths.sbatch

#SBATCH --partition=rome
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --job-name=sid-smoke-paths
#SBATCH --output=jobs/outputs/%x-%j.out
#SBATCH --error=jobs/outputs/%x-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "${SLURM_SUBMIT_DIR}"
fi

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$(pwd)/baselines/repos/SIDReasoner/.venv}"
source baselines/repos/SIDReasoner/scripts/cluster_env.sh

echo "[smoke_config_paths] Repo root: $(pwd)"
echo "[smoke_config_paths] Venv: ${UV_PROJECT_ENVIRONMENT}"

# Install pytest into the venv if not already present
uv pip install --quiet pytest --python "${UV_PROJECT_ENVIRONMENT}/bin/python"

"${UV_PROJECT_ENVIRONMENT}/bin/python" -m pytest tests/common/smoke/test_config_paths.py -v --tb=short
