#!/bin/bash
# Apply our local modifications to the SIDReasoner submodule WITHOUT committing
# them into the submodule. The submodule stays pinned at the upstream commit
# recorded by the parent repository; all changes live here as
# patches/common/sidreasoner-local.patch and are materialised into the
# submodule's working tree here.
#
# Idempotent: safe to run repeatedly. Must run before sourcing the submodule's
# scripts/cluster_env.sh (the patch modifies that file).
#
# Usage (from anywhere):
#   bash scripts/common/apply_submodule_patch.sh

set -euo pipefail

# Walk up from this script to the repo root (depth-independent: sentinel is
# .gitmodules next to the baselines/ dir). Robust to the script's location.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ "${REPO_ROOT}" != "/" && ! ( -f "${REPO_ROOT}/.gitmodules" && -d "${REPO_ROOT}/baselines" ) ]]; do
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
SUBMODULE_DIR="${REPO_ROOT}/baselines/repos/SIDReasoner"
PATCH_FILE="${REPO_ROOT}/patches/common/sidreasoner-local.patch"

if [[ ! -f "${SUBMODULE_DIR}/pyproject.toml" ]]; then
    echo "[apply-patch] ERROR: submodule not found at ${SUBMODULE_DIR}"
    echo "              Run: python3 scripts/common/download_baselines.py"
    exit 1
fi

if [[ ! -f "${PATCH_FILE}" ]]; then
    echo "[apply-patch] ERROR: patch file not found at ${PATCH_FILE}"
    exit 1
fi

python3 "${REPO_ROOT}/scripts/common/normalize_baseline_env.py" "${SUBMODULE_DIR}"

if git -C "${SUBMODULE_DIR}" apply --reverse --check "${PATCH_FILE}" 2>/dev/null; then
    echo "[apply-patch] Patch already applied — nothing to do."
elif git -C "${SUBMODULE_DIR}" apply --check "${PATCH_FILE}" 2>/dev/null; then
    git -C "${SUBMODULE_DIR}" apply "${PATCH_FILE}"
    echo "[apply-patch] Applied ${PATCH_FILE} to ${SUBMODULE_DIR}"
else
    echo "[apply-patch] ERROR: patch does not apply cleanly. Submodule may have"
    echo "              drifted from the expected upstream commit. Check with:"
    echo "              git -C ${SUBMODULE_DIR} log --oneline -1"
    exit 1
fi
