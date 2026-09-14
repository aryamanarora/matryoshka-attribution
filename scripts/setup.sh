#!/usr/bin/env bash
# One-shot setup for a fresh checkout: clone the MIB benchmark repo (OUR FORK, with its EAP-IG
# submodule) into deps/ at the pinned commit, then `uv sync`, then a no-model smoke check.
#
#     bash scripts/setup.sh              # clone the pinned commit into deps/, uv sync, smoke check
#     bash scripts/setup.sh --latest     # clone the fork's default branch instead of the pin
#     bash scripts/setup.sh --no-sync    # clone only
#
# WHY A FORK, AND WHY PINNED. Every scripts/mib/eval_*.py calls MIB's own
# `evaluate_area_under_curve` so our CPR numbers are produced by the benchmark's code, not a
# reimplementation. Our fork extends its return value from 5 to 7 (`accuracies`, `acc_auc` -- the
# acc-AUC columns) and the scripts unpack 7, so UPSTREAM MIB, or a stale fork clone, trains for
# hours and then dies with `ValueError: not enough values to unpack (expected 7, got 5)` after
# eval and before scores.pt is written (2026-09-14: cost a 1.5 h llama3 node eval on sc). The
# EAP-IG submodule carries our fixes too (logits-node to_json, EAP-IG-inputs-mc at edge level).
# The pin below is what every number in results/ was produced against; move it in the same
# commit as the numbers it changes.
#
# WHERE THE CODE LOOKS (src/learning_to_attribute/deps.py, first hit wins): --mib-path,
# $L2A_MIB_PATH, deps/MIB-circuit-track, then the older gitignored symlink ./MIB-circuit-track,
# so a machine provisioned before deps/ existed keeps working and this script will not clone a
# second copy next to it.
#
# NOT HANDLED HERE: the gemma2 venv. Gemma-2 cells must run under TL 2.15.4 (CLAUDE.md), which
# lives in a separate environment inside the MIB checkout (`deps/MIB-circuit-track/.venv`); build
# it from that repo's own environment.yml / pyproject when you need a gemma2 cell.
set -euo pipefail

MIB_SHA=f329461         # aryamanarora/MIB-circuit-track main, 2026-09-14: 7-value evaluate + refs/percentages kwargs
                        # (its .gitmodules pins EAP-IG at 41e9b9c "Add EAP-IG-inputs-mc at edge level")
MIB_URL="${MIB_URL:-https://github.com/aryamanarora/MIB-circuit-track.git}"

PIN=1; SYNC=1
for arg in "$@"; do
  case "$arg" in
    --latest) PIN=0 ;;
    --no-sync) SYNC=0 ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p deps

NAME=MIB-circuit-track; DEST="deps/$NAME"; MARKER=run_evaluation.py
echo "benchmark repo (our fork, called unmodified from scripts/mib/):"
if [ -f "$DEST/$MARKER" ]; then
  echo "  $NAME: already present at $DEST ($(git -C "$DEST" rev-parse --short HEAD 2>/dev/null || echo 'no git metadata'))"
elif [ -f "$NAME/$MARKER" ]; then
  echo "  $NAME: found the older checkout/symlink at ./$NAME ($(git -C "$NAME" rev-parse --short HEAD 2>/dev/null || echo '?'))"
  echo "            deps.py falls back to it, so not cloning a second copy. Remove it to move to deps/."
else
  echo "  $NAME: cloning $MIB_URL (with submodules)"
  git clone --quiet --recurse-submodules "$MIB_URL" "$DEST"
  if [ "$PIN" = 1 ]; then
    git -C "$DEST" checkout --quiet "$MIB_SHA"
    git -C "$DEST" submodule update --init --quiet
    echo "            pinned to $MIB_SHA (EAP-IG @ $(git -C "$DEST/EAP-IG" rev-parse --short HEAD))"
  else
    echo "            at $(git -C "$DEST" rev-parse --short HEAD) (default branch, --latest)"
  fi
fi

if [ "$SYNC" = 1 ]; then
  echo; echo "uv sync:"; uv sync
  echo; echo "smoke check (no model download): the checkout resolves and its evaluate returns 7 values"
  uv run python - <<'PY'
import inspect, sys
from learning_to_attribute.deps import add_mib_to_sys_path
p = add_mib_to_sys_path()
from MIB_circuit_track.evaluation import evaluate_area_under_curve as f
from eap.graph import Graph  # the EAP-IG submodule is checked out
src = inspect.getsource(f)
assert "return weighted_edge_counts, area_under, area_from_1, average, faithfulnesses, accuracies, acc_auc" in src, \
    f"{p}: evaluate_area_under_curve is upstream's 5-value version -- wrong clone or stale pin"
print(f"  OK  {p}  (7-value evaluate_area_under_curve, EAP-IG importable)")
PY
fi

cat <<'EOF2'

Done. Not covered here, and each optional:
  HF_TOKEN            meta-llama/* and google/gemma-2* are gated.
  gemma2 cells        need TL 2.15.4 in deps/MIB-circuit-track/.venv (CLAUDE.md), built from that repo.
  results/            the paper's results dirs live on the clusters (CLAUDE.md's table); not in git.
EOF2
