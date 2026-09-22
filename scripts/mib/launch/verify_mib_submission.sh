#!/usr/bin/env bash
# Score a built submission (make_mib_submission.py) with MIB's OWN run_evaluation.py on one
# cheap cell (ioi/gpt2, public test split), so the files we upload are known to reproduce
# before spending one of the leaderboard's 2 submissions/week. CPU-only: gpt2 with --head N
# examples runs in minutes. Three runs:
#   node_full  the untouched eval_mib.py importances.json (edges dict included)
#   node_slim  the submission JSON (edges: {})        -> must equal node_full EXACTLY
#   edge       the submission importances.pt          -> sanity (runs, CPR-AUC in range)
#
# Usage: scripts/mib/launch/verify_mib_submission.sh <submission dir> <node results dir> [head=100]
#
# The MIB checkout is resolved the same way the eval scripts do (deps.find_mib_path: --mib-path /
# $L2A_MIB_PATH / deps/MIB-circuit-track). The interpreter is this project's default env via
# `uv run` -- fine here because the cell is gpt2, which is version-stable; a gemma2 cell would
# need the `tl2` group instead (CLAUDE.md). On sc, run this as a john CPU job, not on the login
# node: importing torch from the NFS venv there crawls (nfs_wait_bit_killable for 20+ min on 2026-09-14).
set -euo pipefail
SUB=${1:?submission dir}; NODE_DIR=${2:?node results dir (for the full json)}; HEAD=${3:-100}
ABS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
[[ "$NODE_DIR" = /* ]] || NODE_DIR="$ABS/$NODE_DIR"
[[ "$SUB" = /* ]] || SUB="$ABS/$SUB"
MIB=$(cd "$ABS" && uv run python -c "from learning_to_attribute.deps import find_mib_path; print(find_mib_path())")
PY="uv run --project $ABS python"
OUT=$ABS/results/mib_submission_check
echo "MIB checkout: $MIB ($(git -C "$MIB" rev-parse --short HEAD))"; echo "python: $PY"
cd "$MIB"
run() {  # name level circuit-file
  PYTHONPATH=EAP-IG/src CUDA_VISIBLE_DEVICES= $PY run_evaluation.py --models gpt2 --tasks ioi \
    --level $2 --ablation patching --split test --batch-size 20 --head $HEAD \
    --method $1 --circuit-files $3 --output-dir $OUT 2>&1 | grep -E "CPR-AUC|Error|error" || true
}
run node_full node $NODE_DIR/ioi_gpt2_importances.json
run node_slim node $SUB/node/ioi_gpt2/importances.json
run edge      edge $SUB/edge/ioi_gpt2/importances.pt
$PY - <<EOF
import pickle, glob
d = {}
for name in ["node_full", "node_slim", "edge"]:
    p = glob.glob("$OUT/%s_patching_*/ioi_gpt2_test_abs-False.pkl" % name)[0]
    d[name] = pickle.load(open(p, "rb"))
    print(f"{name:10s} CPR-AUC={d[name]['area_under']:.6f}  curve={[round(x,4) for x in d[name]['faithfulnesses']]}")
assert d["node_full"]["faithfulnesses"] == d["node_slim"]["faithfulnesses"], "slim JSON changed the node eval"
print("OK: node_slim == node_full")
EOF
