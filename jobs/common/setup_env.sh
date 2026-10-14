#!/bin/bash
# Create and install the SIDReasoner Python environment.
#
# Submit from the repo root:
#   sbatch jobs/common/setup_env.sh
#
# Optional extras (can be combined):
#   sbatch jobs/common/setup_env.sh --sglang
#   sbatch jobs/common/setup_env.sh --megatron
#   sbatch jobs/common/setup_env.sh --all

#SBATCH --partition=rome
#SBATCH --job-name=sid-setup-env
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=jobs/outputs/%x-%j.out
#SBATCH --error=jobs/outputs/%x-%j.err

set -euo pipefail

# ── Parse optional flags ──────────────────────────────────────────────────────
WITH_MEGATRON=0
WITH_SGLANG=0
for arg in "$@"; do
    case "$arg" in
        --megatron) WITH_MEGATRON=1 ;;
        --sglang)   WITH_SGLANG=1   ;;
        --all)      WITH_MEGATRON=1; WITH_SGLANG=1 ;;
        *)
            echo "[error] Unknown argument: $arg"
            echo "Usage: sbatch jobs/common/setup_env.sh [--sglang] [--megatron] [--all]"
            exit 2
            ;;
    esac
done

# ── Locate repo root ──────────────────────────────────────────────────────────
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${REPO_ROOT}"

SIDREASONER_DIR="${REPO_ROOT}/baselines/repos/SIDReasoner"
if [[ ! -f "${SIDREASONER_DIR}/pyproject.toml" ]]; then
    echo "[error] SIDReasoner submodule not found at ${SIDREASONER_DIR}"
    echo "        Run: python3 scripts/common/download_baselines.py"
    exit 1
fi

# Apply our local submodule modifications (kept as a patch in this repo, not
# committed upstream). Idempotent.
bash "${REPO_ROOT}/scripts/common/apply_submodule_patch.sh"

# ── Load cluster modules ──────────────────────────────────────────────────────
unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_PROMPT_MODIFIER CONDA_SHLVL || true
module purge
module load 2023
module load CUDA/12.4.0

echo "[setup] REPO_ROOT=${REPO_ROOT}"
echo "[setup] SIDREASONER_DIR=${SIDREASONER_DIR}"
echo "[setup] CUDA_HOME=${CUDA_HOME:-<not set>}"

# ── Ensure uv is available ────────────────────────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] uv not found — installing to ~/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi
echo "[setup] uv $(uv --version)"

# ── Build the environment inside the submodule ────────────────────────────────
cd "${SIDREASONER_DIR}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${SIDREASONER_DIR}/.cache/uv}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${SIDREASONER_DIR}/.venv}"
export MAX_JOBS="${MAX_JOBS:-8}"

mkdir -p "${UV_CACHE_DIR}"

echo "[setup] UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"
echo "[setup] UV_CACHE_DIR=${UV_CACHE_DIR}"
echo "[setup] Syncing Python 3.10 environment ..."

uv sync --python 3.10

echo "[setup] Installing cuDNN runtime override (nvidia-cudnn-cu12==9.8.0.87)"
uv pip install --no-deps "nvidia-cudnn-cu12==9.8.0.87"

# Deps for our analysis/plotting helpers, not declared in the SIDReasoner submodule's
# pyproject: matplotlib (model_size figures) + tabulate (pandas .to_markdown() for the
# RQ5 item-side diversity tables). Install them explicitly.
echo "[setup] Installing matplotlib + tabulate (analysis/plotting helpers)"
uv pip install matplotlib tabulate

# ── Optional extras ───────────────────────────────────────────────────────────
if [[ "${WITH_SGLANG}" == "1" ]]; then
    echo "[setup] Installing optional SGLang (--no-deps to preserve pinned FlashInfer)"
    uv pip install --no-deps "sglang[all]==0.4.6.post1"
fi

if [[ "${WITH_MEGATRON}" == "1" ]]; then
    echo "[setup] Installing optional Megatron-LM and TransformerEngine"
    uv pip install pip setuptools wheel
    NVTE_FRAMEWORK=pytorch uv pip install --no-build-isolation --no-deps \
        "git+https://github.com/NVIDIA/TransformerEngine.git@v2.2.1"
    uv pip install --no-build-isolation --no-deps \
        "git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.12.2"
fi

echo "[setup] Environment ready at ${UV_PROJECT_ENVIRONMENT}"
echo "[setup] Test with: uv run --no-sync python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
