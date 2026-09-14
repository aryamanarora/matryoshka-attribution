#!/usr/bin/env bash
# Build the MIB leaderboard submission (all cells in make_mib_submission.CELLS, node + edge) from
# the headline TEST dirs, then verify it with MIB's own run_evaluation.py on ioi/gpt2 (CPU).
# Does NOT upload: `hf upload` publishes to a public repo, so that step stays manual (printed at
# the end). Run as a cluster job -- on sc the login node's NFS makes torch imports crawl:
#
#   nlprun -q jag -g 1 -c 4 -r 48G -n mib-build --dependency afterok:<edge job> \
#       "bash scripts/mib/launch/build_and_verify_submission.sh"
#
# OUT / NODE / EDGE / HEAD override the defaults below.
set -euo pipefail
ABS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$ABS"
OUT=${OUT:-/nlp/scr/aryaman/mib_submit}
NODE=${NODE:-results/test_node_softlog_sgd_lr_1.0}
EDGE=${EDGE:-results/test_edge_softlog_sgd_lr_3.0}
HEAD=${HEAD:-100}

echo "== build: node=$NODE edge=$EDGE -> $OUT"
uv run python scripts/mib/make_mib_submission.py --node-dir "$NODE" --edge-dir "$EDGE" --out "$OUT"

echo; echo "== verify (ioi/gpt2, MIB run_evaluation.py, --head $HEAD)"
bash scripts/mib/launch/verify_mib_submission.sh "$OUT" "$NODE" "$HEAD"

echo; echo "== contents"
du -sh "$OUT/node" "$OUT/edge"
find "$OUT" -type f | sort | sed "s|$OUT/||"

echo; echo "== hf auth (upload is manual)"
(uv run hf auth whoami 2>&1 || uv run huggingface-cli whoami 2>&1) | head -3 || true
cat <<EOF

NEXT, manual (publishes):
  uv run hf upload aryaman/mattr-mib-circuits $OUT . --repo-type model     # repo must be PUBLIC
  https://huggingface.co/spaces/mib-bench/leaderboard -> Submit, track "Circuit Localization":
    https://huggingface.co/aryaman/mattr-mib-circuits/tree/main/node   level "Node (submodule)"
    https://huggingface.co/aryaman/mattr-mib-circuits/tree/main/edge   level "Edge"
  Expect only the "interpbench missing" warning -> Proceed Anyway. Save the submission IDs.
EOF
