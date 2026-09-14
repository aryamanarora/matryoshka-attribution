#!/bin/bash
# Finalize the compute-matched gradient-baseline wave (see submit_grad_compute_match.sh).
# For every file that landed in results/_gradcm_staging/<tree>/:
#   1. archive the live counterpart: real files -> results/_deprecated_grad_smallN/<tree>/,
#      symlinks (the 5k trees' pointers into sva_sweep{,_ferr}) are removed -- their targets
#      stay where they are and ARE the archive;
#   2. move the staged file into the live tree.
# Idempotent and partial-failure-tolerant: cells whose job died simply never staged a file and
# keep serving the deprecated run; rerun this script after backfilling. Logs every action.
set -uo pipefail
cd "$(dirname "$0")/../../.."
STAGE=results/_gradcm_staging
DEP=results/_deprecated_grad_smallN
moved=0; archived=0; unlinked=0; missing=0

for tree_dir in "$STAGE"/*/; do
  tree=$(basename "$tree_dir")
  live="results/$tree"
  mkdir -p "$DEP/$tree"
  for f in "$tree_dir"*; do
    [ -e "$f" ] || continue
    b=$(basename "$f")
    if [ -L "$live/$b" ]; then
      rm "$live/$b"; unlinked=$((unlinked+1))
      echo "unlink  $live/$b (target archived in place)"
    elif [ -e "$live/$b" ]; then
      mv "$live/$b" "$DEP/$tree/$b"; archived=$((archived+1))
      echo "archive $live/$b -> $DEP/$tree/"
    fi
    mv "$f" "$live/$b"; moved=$((moved+1))
  done
done

# report cells that never staged (job died / still pending) so the gap is visible
echo "== finalize: $moved files went live, $archived archived, $unlinked symlinks removed =="
find "$STAGE" -type f | head -20 | sed 's/^/UNMOVED /'
find "$STAGE" -type d -empty -delete 2>/dev/null
exit 0
