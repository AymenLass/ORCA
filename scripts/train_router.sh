#!/usr/bin/env bash
# ===========================================================================
# AgenticVLM — Router Training Script
# ===========================================================================
# Usage:
#   bash scripts/train_router.sh --train-data data/train.csv --image-dir data/images
#
# Environment variables:
#   AGENTICVLM_CONFIG — path to YAML config (default: config/default.yaml)
#   OUTPUT_DIR        — override output directory

set -euo pipefail

CONFIG="${AGENTICVLM_CONFIG:-config/default.yaml}"
OUTPUT="${OUTPUT_DIR:-outputs/router}"

python -m agenticvlm.cli.train \
    --config "$CONFIG" \
    --output-dir "$OUTPUT" \
    "$@"
