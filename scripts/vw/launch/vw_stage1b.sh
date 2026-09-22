#!/bin/bash
# Stage 1b: the streaming score pass over the whole training set. Split from 1a because the
# cluster is GPU-saturated and slurm backfills SHORT jobs into gaps -- 1a is ~7 minutes and
# gets scheduled quickly, this one asks for hours and would otherwise hold both back.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
PY="uv run python"
echo "=== scoring benchmark (2M positions) ==="
$PY scripts/vw/vw_scores.py --run results/vw/base --positions 2000000 --tag tl_bench
echo "=== full scoring pass ==="
$PY scripts/vw/vw_scores.py --run results/vw/base --time-budget 2400
