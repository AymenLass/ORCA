#!/usr/bin/env bash
# ===========================================================================
# AgenticVLM — Evaluation Script
# ===========================================================================
# Usage:
#   bash scripts/evaluate.sh --predictions results/predictions.csv

set -euo pipefail

python -m agenticvlm.cli.evaluate \
    --threshold 0.5 \
    "$@"
