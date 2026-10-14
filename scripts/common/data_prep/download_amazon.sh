#!/usr/bin/env bash
# Download Amazon dataset files used by SIDReasoner.
#
# Usage:
#   bash scripts/common/data_prep/download_amazon.sh Video_Games
#   bash scripts/common/data_prep/download_amazon.sh Office_Products
#   bash scripts/common/data_prep/download_amazon.sh Industrial_and_Scientific
#
# Files are placed under data/raw/<dataset_slug>/
# Run from the repo root.

set -euo pipefail

DATASET="${1:-}"
if [[ -z "$DATASET" ]]; then
    echo "Usage: $0 <dataset_name>"
    echo "  Available: Video_Games  Office_Products  Industrial_and_Scientific"
    exit 1
fi

# Normalise to lowercase slug for our directory structure
SLUG=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]' | tr '_' '-' | sed \
    -e 's/video-games/games/' \
    -e 's/office-products/office/' \
    -e 's/industrial-and-scientific/industrial/')

OUTDIR="data/raw/${SLUG}"
mkdir -p "$OUTDIR"

echo "[download] Dataset:    $DATASET"
echo "[download] Output dir: $OUTDIR"

# Mirror the make data target from baselines/repos/SIDReasoner/Makefile
SIDROOT="baselines/repos/SIDReasoner"
if [[ ! -f "$SIDROOT/Makefile" ]]; then
    echo "[error] SIDReasoner submodule not initialised. Run: python3 scripts/common/download_baselines.py"
    exit 1
fi

# Delegate to SIDReasoner's own download mechanism (reads Makefile targets).
# We pass CATEGORY and DATA_DIR so its scripts write into our data/raw/ layout.
make -C "$SIDROOT" data \
    CATEGORY="$DATASET" \
    DATA_DIR="$(pwd)/$OUTDIR" \
    2>&1 | tee "$OUTDIR/download.log"

echo "[download] Done. Files in $OUTDIR:"
ls -lh "$OUTDIR"
