#!/usr/bin/env bash
# One-shot setup for a fresh checkout: use the MIB benchmark repo (OUR FORK, with its EAP-IG
# submodule) vendored in deps/ at the pinned commit (cloned from $MIB_URL only if missing; the
# URL is withheld for anonymous review), then `uv sync`, then a no-model smoke check.
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
# eval and before scores.pt is written (2026-09-14: cost a 1.5 h llama3 node eval on cluster B). The
# EAP-IG submodule carries our fixes too (logits-node to_json, EAP-IG-inputs-mc at edge level).
# The pin below is what every number in results/ was produced against; move it in the same
# commit as the numbers it changes.
#
# WHERE THE CODE LOOKS (src/matryoshka_attribution/deps.py, first hit wins): --mib-path,
# $MATTR_MIB_PATH, then deps/MIB-circuit-track -- which find_mib_path clones at the pin if it is
# missing, so a fresh checkout works from the first `uv run` even without this script.
#
# NO SEPARATE GEMMA2 VENV. Gemma-2 cells must run under TL 2.15.4 (README.md); that stack is the
# `tl2` dependency group of pyproject.toml, locked in uv.lock, and a job syncs it into its own env
# (UV_PROJECT_ENVIRONMENT=.venv-tl2 uv run --no-default-groups --group tl2 ...), exactly like the
# ViT group. Nothing here to build.
set -euo pipefail

# The pin and URL live in src/matryoshka_attribution/deps.py (deps.find_mib_path clones at the
# pin on first use too; this script just does it up front, with --latest as the escape hatch).
DEPS_PY="$(dirname "${BASH_SOURCE[0]}")/../src/matryoshka_attribution/deps.py"
MIB_SHA=$(sed -n 's/^MIB_SHA = "\(.*\)"/\1/p' "$DEPS_PY")
MIB_URL="${MIB_URL:-$(sed -n 's/^MIB_URL = "\(.*\)"/\1/p' "$DEPS_PY")}"

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
from matryoshka_attribution.deps import add_mib_to_sys_path
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
  gemma2 cells        run in the TL 2.15.4 env, the `tl2` dependency group (pyproject.toml), which the job
                      syncs itself:  UV_PROJECT_ENVIRONMENT=.venv-tl2 uv run --no-default-groups --group tl2 python ...
  results/            the paper's results dirs live on the clusters (README.md's table); not in git.
EOF2
