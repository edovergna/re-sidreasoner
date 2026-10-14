#!/bin/bash
# Download all SIDReasoner author checkpoints from heyingzhi on HuggingFace
# into data/checkpoints/<model-name>/.
#
# Run standalone:
#   bash scripts/common/data_prep/download_checkpoints.sh
# Or via SLURM:
#   sbatch jobs/common/download_checkpoints.job

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CHECKPOINTS_DIR="${REPO_ROOT}/data/checkpoints"

MODELS=(
    "heyingzhi/sidreasoner_sft_games"
    "heyingzhi/sidreaonser_sft_office"
    "heyingzhi/sidreasoner_ra_games"
    "heyingzhi/sidreasoner_ra_office"
)

export PATH="${HOME}/.local/bin:${PATH}"

echo "[download_checkpoints] Target dir: ${CHECKPOINTS_DIR}"

for MODEL in "${MODELS[@]}"; do
    MODEL_NAME="${MODEL##*/}"
    LOCAL_DIR="${CHECKPOINTS_DIR}/${MODEL_NAME}"
    mkdir -p "${LOCAL_DIR}"
    echo "[download_checkpoints] Downloading ${MODEL} → ${LOCAL_DIR}"
    uvx --from huggingface_hub hf download \
        "${MODEL}" \
        --local-dir "${LOCAL_DIR}"
    echo "[download_checkpoints] Done: ${MODEL_NAME}"
done

echo "[download_checkpoints] All checkpoints downloaded."
