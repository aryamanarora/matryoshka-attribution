#!/bin/bash
# Fill EVERY method the test-set bar chart draws (plots/plot_mib_test_avg.py, via
# make_mib_test_table.collect()) for the one cell the paper's 11-cell grid never had:
# arithmetic_addition / llama3, TEST split. cluster B / gpujob, run INSIDE tmux (gpujob is not on PATH
# over plain ssh). Companion to submit_arith_add_headline_sc.sh, which did the two headline dirs.
#
# What gets launched (all llama3 / arithmetic_addition / test), and where the protocol comes from:
#
#   MAttr node x3   test_node_softuni_sgd_lr_3.0   submit_test_sgd.sh      500 steps, uniform k, sgd lr 3.0
#                   test_node_topk_log_lr05        submit_test_lr05.sh     500 steps, log k, adam lr 0.05
#                   test_node_topk_uniform_lr05    submit_softuni_lr05.sh  500 steps, uniform k, adam lr 0.05
#                   (all: --include-input --batch-size 2, full test split; a6000)
#   MAttr edge x3   test_edge_softuni_sgd_lr_3.0   submit_test_edge_sgd.sh   5000 steps, uniform k, sgd lr 3.0
#                   test_edge_topk_log_lr05        submit_edge_arc_llama3.sh 5000 steps, log k, adam lr 0.05
#                   test_edge_topk_uniform_lr05    submit_softuni_lr05.sh    5000 steps, uniform k, adam lr 0.05
#                   (all: --batch-size 2 --eval-examples 200 = the $\dagger$ cap; large-memory h100)
#   grad x9         run_evaluation.py --split test --batch-size 1 on circuits that ALREADY exist in
#                   deps/MIB-circuit-track/results/<cdir>/<Method>_patching_node/arithmetic-addition_llama3
#                   (the attribution wave included addition; only the test evals were never run):
#                   AttnLRP, GIM, RelP, RelP-qkgrad, EAP-IG-inputs x4 (ig1/napig_ref/napig10/napig30),
#                   EAP-IG-inputs-mc (napig_mc)  -- mirrors run_attnlrp_test.sh / run_gim_relpqk_test.sh /
#                   run_stepless_test.sh in the fork. Output pkl: results/<odir>/<Method>_patching_node/
#                   arithmetic-addition_llama3_test_abs-False.pkl, which make_mib_test_table.load_run_eval_cpr reads.
#   DBM ladder x8   eprun_node_ld_sig_lr0.3_l1{0.2,0.6,2.0,6.0,20.0,40.0,60.0,200.0}: node level, 3000
#                   steps, gate=sigmoid, loss=logit_diff, lr 0.3 (run_edge_pruning.sbatch's recipe). No
#                   addition graph exists at any rung. The l1=6.0 job ALSO runs the EdgePruning test eval
#                   into eprun_eval_ld_sig_lr0.3_l16.0 (the "DBM" row); the other rungs only train.
#   DBM sweep x1    eval_dbm_multisparsity.py --split test -> results/dbm_multisparsity/arithmetic_addition_llama3_test.json
#                   (the "DBM (multi-sparsity)" row), submitted with --dependency afterok:<all 8 rungs>.
#
# NOT COVERED, and cannot be from here: the rows transcribed from MIB's Table 1
# (make_mib_test_table.NODE_BASELINES / EDGE_BASELINES) carry no arithmetic_addition cell, so those
# methods' averages stay 11-cell unless the numbers are transcribed from the MIB paper.
#
# Every job uses `uv run python` (env syncs in the job) and the deps/ fork clone via
# deps.find_mib_path(). A cell whose final artifact already exists is skipped, so re-running after
# a partial wave is safe.
#
#   bash scripts/mib/launch/submit_arith_add_fill_sc.sh                 # everything
#   ONLY="mattr grad" DRYRUN=1 bash scripts/mib/launch/submit_arith_add_fill_sc.sh   # subset / preview
#   ONLY groups: mattr_node mattr_edge grad dbm dbm_sweep  (mattr = mattr_node mattr_edge)
set -u
ABS=${ABS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; cd "$ABS"
MIB=$ABS/deps/MIB-circuit-track
DRYRUN=${DRYRUN:-0}
ONLY=${ONLY:-"mattr_node mattr_edge grad dbm dbm_sweep"}
ONLY=${ONLY/mattr /mattr_node mattr_edge }; [ "$ONLY" = mattr ] && ONLY="mattr_node mattr_edge"
MODEL=llama3; TASK=arithmetic_addition; HT=arithmetic-addition   # HT = MIB's hyphenated spelling
EXP="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
NODE_RES="-q gpu -d a6000 -c 4 -r 96G"          # 8B at batch 2 fits 48 GB (headline node run did)
EDGE_RES="-q gpu-big -d h100 -r 128G"            # edge / DBM training need 80 GB (submit_bigmem_retry.sh)
want() { case " $ONLY " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
n=0; skip=0
submit() {  # name resources command  -> prints job id on stdout (empty on dry run)
  local name=$1 res=$2 cmd=$3
  n=$((n+1))
  if [ "$DRYRUN" = "1" ]; then echo "DRY gpujob -g 1 $res -n $name" >&2; echo "    $cmd" >&2; echo ""; return; fi
  local out; out=$(gpujob -g 1 $res -n "$name" -o "$ABS/logs/${name}.out" "$cmd" 2>&1)
  local jid; jid=$(echo "$out" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+$')
  [ -n "$jid" ] && echo "submitted $name -> job $jid" >&2 || { echo "FAILED $name: $out" >&2; }
  sleep 1; echo "$jid"
}
skipping() { echo "SKIP $1: already on disk" >&2; skip=$((skip+1)); }

# ---------------- MAttr node (3) ----------------
if want mattr_node; then
  for spec in "softuni-sgd|uniform|sgd|3.0|test_node_softuni_sgd_lr_3.0" \
              "log-adam|log|adam|0.05|test_node_topk_log_lr05" \
              "uni-adam|uniform|adam|0.05|test_node_topk_uniform_lr05"; do
    IFS='|' read -r tag sched opt lr out <<< "$spec"
    if [ -f "results/$out/${TASK}_${MODEL}_scores.pt" ]; then skipping "$out"; continue; fi
    submit "aa-node-$tag" "$NODE_RES" "$EXP uv run python scripts/mib/eval_mib.py --model $MODEL --task $TASK \
--steps 500 --k-schedule $sched --masking topk --optimizer $opt --mode iso --lr $lr \
--split test --train-split train --include-input --batch-size 2 --output results/$out" >/dev/null
  done
fi

# ---------------- MAttr edge (3) ----------------
if want mattr_edge; then
  for spec in "softuni-sgd|uniform|sgd|3.0|test_edge_softuni_sgd_lr_3.0" \
              "log-adam|log|adam|0.05|test_edge_topk_log_lr05" \
              "uni-adam|uniform|adam|0.05|test_edge_topk_uniform_lr05"; do
    IFS='|' read -r tag sched opt lr out <<< "$spec"
    if [ -f "results/$out/${TASK}_${MODEL}_scores.pt" ]; then skipping "$out"; continue; fi
    submit "aa-edge-$tag" "$EDGE_RES" "$EXP uv run python scripts/mib/eval_mib_edge.py --model $MODEL --task $TASK \
--steps 5000 --k-schedule $sched --masking topk --optimizer $opt --mode iso --lr $lr \
--split test --train-split train --batch-size 2 --eval-examples 200 --output results/$out" >/dev/null
  done
fi

# ---------------- gradient baselines: test evals on existing circuits (9) ----------------
if want grad; then
  for spec in "attnlrp|AttnLRP|attnlrp|attnlrp_eval" \
              "gim|GIM|gim|gim_eval" \
              "relp|RelP|relp|relp_eval" \
              "relpqk|RelP-qkgrad|relp_qkgrad|relp_qkgrad_eval" \
              "ig1|EAP-IG-inputs|ig1|ig1_test" \
              "ig5|EAP-IG-inputs|napig_ref|napig_ref_test" \
              "ig10|EAP-IG-inputs|napig10|napig10_test" \
              "ig30|EAP-IG-inputs|napig30|napig30_test" \
              "eg|EAP-IG-inputs-mc|napig_mc|napig_mc_test"; do
    IFS='|' read -r tag method cdir odir <<< "$spec"
    cpath="$MIB/results/$cdir/${method}_patching_node/${HT}_${MODEL}/importances.json"
    [ -f "$cpath" ] || { echo "MISSING circuit for $tag: $cpath" >&2; continue; }
    if [ -f "results/$odir/${method}_patching_node/${HT}_${MODEL}_test_abs-False.pkl" ]; then skipping "$odir/$tag"; continue; fi
    submit "aa-grad-$tag" "$NODE_RES" "cd $MIB && PYTHONPATH=.:EAP-IG/src $EXP uv run --project $ABS python run_evaluation.py \
--models $MODEL --tasks $TASK --method $method --level node --ablation patching --split test --batch-size 1 \
--circuit-dir results/$cdir --output-dir $ABS/results/$odir" >/dev/null
  done
fi

# ---------------- DBM ladder (8 trainings; l1=6.0 also evaluated) ----------------
LADDER_JOBS=""
if want dbm; then
  for l1 in 0.2 0.6 2.0 6.0 20.0 40.0 60.0 200.0; do
    out=results/eprun_node_ld_sig_lr0.3_l1$l1; graph=$out/graph_${TASK}_${MODEL}.json
    evaldir=results/eprun_eval_ld_sig_lr0.3_l1$l1
    train="$EXP uv run python scripts/mib/eval_mib_edge_pruning.py --model $MODEL --task $TASK --level node \
--steps 3000 --split test --output $out --loss logit_diff --gate sigmoid --lr 0.3 --l1-coeff $l1 --l1-target gate --skip-eval"
    evalc="cd $MIB && PYTHONPATH=.:EAP-IG/src $EXP uv run --project $ABS python run_evaluation.py --models $MODEL --tasks $TASK \
--level node --ablation patching --split test --method EdgePruning --circuit-files $ABS/$graph --batch-size 2 --output-dir $ABS/$evaldir"
    if [ "$l1" = "6.0" ]; then
      if [ -f "$evaldir/EdgePruning_patching_node/${HT}_${MODEL}_test_abs-False.pkl" ]; then skipping "$evaldir"; continue; fi
      [ -f "$graph" ] && cmd="$evalc" || cmd="$train && $evalc"
    else
      if [ -f "$graph" ]; then skipping "$out"; continue; fi
      cmd="$train"
    fi
    jid=$(submit "aa-dbm-l1$l1" "$EDGE_RES" "$cmd"); [ -n "$jid" ] && LADDER_JOBS="$LADDER_JOBS:$jid"
  done
fi

# ---------------- DBM multi-sparsity test eval (after the ladder) ----------------
if want dbm_sweep; then
  if [ -f "results/dbm_multisparsity/${TASK}_${MODEL}_test.json" ]; then skipping "dbm_multisparsity"; else
    dep=""; [ -n "$LADDER_JOBS" ] && dep="--dependency afterok${LADDER_JOBS}"
    submit "aa-dbm-sweep" "$EDGE_RES $dep" "PYTHONPATH=$MIB:$MIB/EAP-IG/src $EXP uv run python scripts/mib/eval_dbm_multisparsity.py \
--model $MODEL --task $TASK --split test --batch-size 2" >/dev/null
  fi
fi

[ "$DRYRUN" = "1" ] && pfx="DRY " || pfx=""
echo "== ${pfx}total $n arithmetic_addition fill jobs, $skip skipped ==" >&2
