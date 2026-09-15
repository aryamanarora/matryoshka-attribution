#!/bin/bash
# arithmetic_addition / llama3 for the SPARSITY SWEEP (make_lr_table.SPARSITY_METHODS ->
# tabs/sparsity_sweep.tex, tabs/lr_sweep_accauc.tex and plots/plot_sparsity_sweep_summary.py),
# VALIDATION split, so those two blocks average over 12 cells like the results tables do.
# sc / nlprun, run INSIDE tmux.
#
#   Node Pruning, logit-diff, s in {0.1 .. 2.0}   9 rungs: TRAIN (no addition graph exists) + eval
#   DBM + L1, lr 0.3, lambda = 0                  1 rung:  TRAIN + eval (the no-penalty control)
#   DBM + L1, lr 0.3, lambda in {0.2 .. 200}      8 rungs: EVAL ONLY -- graphs were trained by
#                                                 submit_arith_add_fill_sc.sh (test-split protocol;
#                                                 training never reads the eval split, so the graph
#                                                 is split-agnostic)
#
# Recipes are read off each dir's arithmetic_subtraction_llama3_scores.pt `args` (2026-09-15):
#   Node Pruning: 3000 steps, gate hard_concrete, lr 0.8, reg_lr 0.8, --target-sparsity s,
#                 warmup 0.83 / 0.07 linear (defaults), loss logit_diff, seed 42
#   DBM:          3000 steps, gate sigmoid, lr 0.3, --l1-coeff lambda --l1-target gate, loss logit_diff
# and the eval is MIB's run_evaluation.py --method EdgePruning --split validation --batch-size 2
# --head 200 (llama3 validation cap, run_edge_pruning.sbatch), into results/eprun_eval<suffix>.
# Output dir names are passed explicitly and match run_edge_pruning.sbatch's suffix grammar
# (_s<S>_ld for Node Pruning logit-diff; _ld_sig_lr0.3[_l1<L>] for DBM), so the table reads them.
#
# Training jobs go to sphinx h100 (node-level llama3 mask training, like the DBM ladder); the
# eval-only jobs fit jag a6000. Cells with a validation pkl are skipped.
#   bash scripts/mib/launch/submit_arith_add_sweep_sc.sh          # 18 jobs
#   ONLY=dbm DRYRUN=1 bash ...                                     # ONLY: np | dbm
set -u
ABS=${ABS:-/juice3/scr3/nlp/interp/learning-to-attribute}; cd "$ABS"; mkdir -p logs
MIB=$ABS/deps/MIB-circuit-track
DRYRUN=${DRYRUN:-0}; ONLY=${ONLY:-"np dbm"}
MODEL=llama3; TASK=arithmetic_addition; HT=arithmetic-addition
EXP="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
TRAIN_RES="-q sphinx -d h100 -r 128G"; EVAL_RES="-q jag -d a6000 -c 4 -r 96G"
n=0; skip=0
evalcmd() {  # graph evaldir
  echo "cd $MIB && PYTHONPATH=.:EAP-IG/src $EXP uv run --project $ABS python run_evaluation.py --models $MODEL --tasks $TASK \
--level node --ablation patching --split validation --method EdgePruning --circuit-files $ABS/$1 --batch-size 2 --head 200 --output-dir $ABS/$2"
}
submit() {  # name res cmd
  n=$((n+1))
  if [ "$DRYRUN" = 1 ]; then echo "DRY nlprun -g 1 $2 -n $1"; echo "    $3"; else
    nlprun -g 1 $2 -n "$1" -o "$ABS/logs/$1.out" "$3" 2>&1 | grep -E 'Submitted batch job' | sed "s/^/$1: /"; sleep 1
  fi
}
launch() {  # tag suffix train-extra-args
  local tag=$1 suf=$2 extra=$3
  local out=results/eprun_node$suf
  local evaldir=results/eprun_eval$suf
  local graph=$out/graph_${TASK}_${MODEL}.json   # separate statements: set -u + one `local` line expands $out before assigning it
  if [ -f "$evaldir/EdgePruning_patching_node/${HT}_${MODEL}_validation_abs-False.pkl" ]; then echo "SKIP $evaldir"; skip=$((skip+1)); return; fi
  local ev; ev=$(evalcmd "$graph" "$evaldir")
  if [ -f "$graph" ]; then submit "aas-$tag" "$EVAL_RES" "$ev"; else
    submit "aas-$tag" "$TRAIN_RES" "$EXP uv run python scripts/mib/eval_mib_edge_pruning.py --model $MODEL --task $TASK --level node \
--steps 3000 --split validation --output $out --loss logit_diff $extra --skip-eval && $ev"
  fi
}
case " $ONLY " in *" np "*)
  for s in 0.1 0.25 0.5 0.8 0.9 0.95 0.99 1.25 2.0; do launch "np-s$s" "_s${s}_ld" "--gate hard_concrete --target-sparsity $s"; done ;;
esac
case " $ONLY " in *" dbm "*)
  launch "dbm-l10" "_ld_sig_lr0.3" "--gate sigmoid --lr 0.3"
  for l1 in 0.2 0.6 2.0 6.0 20.0 40.0 60.0 200.0; do launch "dbm-l1$l1" "_ld_sig_lr0.3_l1$l1" "--gate sigmoid --lr 0.3 --l1-coeff $l1 --l1-target gate"; done ;;
esac
echo "== $([ "$DRYRUN" = 1 ] && echo 'DRY ')total $n sparsity-sweep arithmetic_addition jobs, $skip skipped =="
