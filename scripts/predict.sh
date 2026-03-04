#!/usr/bin/env bash
# ===========================================================================
# AgenticVLM — Prediction Script
# ===========================================================================
# Usage:
#   bash scripts/predict.sh --data data/test.csv --output results/predictions.csv

set -euo pipefail

CONFIG="${AGENTICVLM_CONFIG:-config/default.yaml}"

python -m agenticvlm.cli.predict \
    --config "$CONFIG" \
    "$@"
