"""Evaluate a DBM sparsity ladder at each run's OWN empirical L0, not on MIB's fixed grid.

WHY THIS EXISTS. MIB scores a method by sweeping ONE score vector over ten fixed proportions
(.001 ... 1). That is the right question for a ranking, and the wrong one for a mask trained at a
particular sparsity: DBM's sigmoid gate saturates, so nodes whose gate is off stop receiving
gradient and their logits drift into a narrow band -- at l1=60 on ioi/gpt2, 152 of 156 scores sit
inside a 0.39-wide window. MIB then asks that run to rank at p=0.2 and p=0.5, i.e. deep inside the
band, and most of its CPR integral is computed over noise.

This script instead evaluates each run at the ONE point it was trained for: the empirical L0, the
expected number of open gates at the last training step (`k_log[-1]` in the run's scores .pt).
Collecting those points across the l1 ladder gives a size-vs-performance frontier for the METHOD
FAMILY, from which scripts/mib/dbm_multisparsity.py computes CPR and IIA log-AUC.

THE BINARISED MASK IS THE TOP-L0 SET, exactly. The gate is a monotone function of the stored
score, so thresholding the gate at 0.5 and taking the top-L0 scores select the same nodes. That is
why this can reuse Graph.apply_topn and needs no new masking code -- and why the numbers are
directly comparable to MIB's, which applies the same operator at different k.

*** ONE MODEL LOAD PER CELL, NOT ONE PER RUN. *** The clean and fully-ablated references are
computed once and shared across the whole ladder, so a cell costs (n_rungs + 2) graph evaluations
instead of n_rungs * 12. That also guarantees every rung is normalised against the same two
reference scores, which a per-run invocation would not.

Run under the MIB venv (TL 2.15.4) -- mandatory for gemma2, whose forward is wrong under the L2A
venv's TL 3.x. See scripts/mib/launch/submit_dbm_multisparsity.sh.

    PYTHONPATH=$MIB:$MIB/EAP-IG/src $MIB/.venv/bin/python scripts/mib/eval_dbm_multisparsity.py \
        --model gpt2 --task ioi --split validation

Out: results/dbm_multisparsity/{task}_{model}_{split}.json
"""
import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path

import math
import torch
from learning_to_attribute.deps import find_mib_path

# Default ladder: submit_dbm_l1.sh's original five plus the 2026-09-08 extension. A rung with no
# graph on disk is skipped and named in the output, so a partial ladder is visible rather than
# silently shortening the frontier.
# 200 added 2026-09-09 to fill the SPARSE END. Under the hold protocol a new rung can only fill
# budgets that were previously pinned to a sparser mask, so adding rungs is weakly monotone -- a
# rung that trains badly costs nothing, it just supplies a low value at a low budget.
#
# *** IT BOUGHT ALMOST NOTHING, AND THE REASON IS WORTH KEEPING. *** Measured on the 8-rung
# ladder against the 7-rung one: IIA moved +0.000 (ioi/gpt2), +0.002 (ioi/qwen2.5, ioi/gemma2),
# +0.005 (mcqa/qwen2.5). The sparse end was NOT where the deficit was. Decomposing the IIA gap
# to MAttr by grid point puts essentially all of it at p=0.05 and p=0.1 -- e.g. 100% of
# mcqa/qwen2.5's 0.103 sits at p=0.05 alone -- where the ladder has a large multiplicative GAP:
# lambda=60 lands at k=26 and lambda=200 at k=8, while p=0.05 asks for k <= 18. DBM's k=26 mask
# scores accuracy 1.000, BETTER than MAttr's 0.920 at k=18; it simply cannot be spent at that
# budget. So the deficit is ladder RESOLUTION in the band k ~ [0.02N, 0.1N], not mask quality
# and not coverage at the extreme. The control is ioi/gpt2, whose ladder happens to have rungs
# at k=1,5,15,27 straight through that band: its deficit is 0.003.
#
# The ladder ENDS AT 200 (user decision, 2026-09-09) rather than being refined further. Closing
# the gap would need lambda in (60, 200) on every cell, i.e. more runs on a row whose cost column
# already reads 21k against the other rows' 3k -- and that cost IS the finding. Note also that
# raising the penalty is the wrong lever regardless: an L1 term controls sparsity only indirectly
# (at lambda=20 the converged density spans 0.009 to 0.256 across cells, a 27x range) and stops
# being a knob once it dominates the task loss -- at lambda=200, 3 of 11 cells collapse to zero
# open gates. To place a rung at a chosen size, train against an explicit target the way Node
# Pruning's hard-concrete gates do.
# 600 was launched with 200 and CANCELLED unstarted: lambda=200 already collapsed on 3 of 8 cells
# and produced 2-node masks on three more, so 600 would have added no coverage below what 200
# reaches -- and per the above, coverage below 200 was never the binding constraint anyway.
L1S = ["0.2", "0.6", "2.0", "6.0", "20.0", "40.0", "60.0", "200.0"]
NODE_DIR = "results/eprun_node_ld_sig_lr0.3_l1{l1}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--l1s", nargs="+", default=L1S)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--head", type=int, default=None,
                    help="cap the eval set; MIB caps llama3 VALIDATION at 200, test uncapped")
    ap.add_argument("--mib-path", default=None)
    ap.add_argument("--output", default="results/dbm_multisparsity")
    ap.add_argument("--absolute", action="store_true",
                    help="rank by |score| instead of score. MIB's run_evaluation.py defaults "
                         "this OFF and encodes it in the pkl name (`abs-False`), so the default "
                         "here must stay off too -- see the note above `score`.")
    ap.add_argument("--verify", action="store_true",
                    help="ALSO evaluate MIB's ten fixed proportions with this script's own "
                         "machinery and print them beside the stored pkl. Proves the pipeline "
                         "reproduces run_evaluation.py before any own-L0 number is trusted.")
    a = ap.parse_args()

    a.mib_path = str(find_mib_path(a.mib_path))
    sys.path.insert(0, a.mib_path)
    sys.path.insert(0, os.path.join(a.mib_path, "EAP-IG", "src"))
    from transformer_lens import HookedTransformer
    from eap.graph import Graph
    from MIB_circuit_track.evaluation import evaluate_area_under_curve
    from MIB_circuit_track.metrics import get_metric
    from MIB_circuit_track.utils import TASKS_TO_HF_NAMES, MODEL_NAME_TO_FULLNAME
    from MIB_circuit_track.dataset import HFEAPDataset

    # Collect the ladder BEFORE loading the model: a missing graph should fail in seconds, not
    # after an 8B checkpoint is resident.
    rungs = []
    for l1 in a.l1s:
        d = NODE_DIR.format(l1=l1)
        g = f"{d}/graph_{a.task}_{a.model}.json"
        s = f"{d}/{a.task}_{a.model}_scores.pt"
        if not (os.path.exists(g) and os.path.exists(s)):
            print(f"SKIP l1={l1}: no graph/scores at {d}")
            continue
        sd = torch.load(s, map_location="cpu")
        # k_log[-1] is the EXPECTED open-gate count at the last step -- the run's own L0. Not
        # args.target_sparsity, which the sigmoid gate ignores entirely.
        rungs.append((l1, float(sd["k_log"][-1]), g, len(sd["scores"])))
    if not rungs:
        raise SystemExit(f"no DBM rungs on disk for {a.task}/{a.model}")

    if a.model in ("qwen2.5", "gemma2", "llama3"):
        model = HookedTransformer.from_pretrained(
            MODEL_NAME_TO_FULLNAME[a.model], attn_implementation="eager",
            torch_dtype=torch.bfloat16)
    else:
        model = HookedTransformer.from_pretrained(MODEL_NAME_TO_FULLNAME[a.model])
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True

    ds = HFEAPDataset(f"mib-bench/{TASKS_TO_HF_NAMES[a.task]}", model.tokenizer,
                      split=a.split, task=a.task, model_name=a.model)
    if a.head is not None:
        ds.head(min(a.head, len(ds)))
    dl = ds.to_dataloader(batch_size=a.batch_size)
    metric = get_metric("logit_diff", a.task, model.tokenizer, model)
    metrics = partial(metric, mean=False, loss=False)

    # n_scored_items exactly as MIB computes it (evaluation.py counts non-NaN nodes_scores):
    # every proportion below is int(p * this), so a different node count silently shifts every k.
    n_nodes = int((~torch.isnan(Graph.from_json(rungs[0][2]).nodes_scores)).sum().item())
    print(f"{a.task}/{a.model} [{a.split}]  n_nodes={n_nodes}, {len(rungs)} rungs")

    # *** THE NUMBERS ARE PRODUCED BY MIB'S OWN evaluate_area_under_curve, NOT BY A COPY OF IT.
    # *** An earlier version of this script reimplemented the per-point loop and disagreed with
    # the stored pkl at exactly one grid point even after the absolute-ranking bug was fixed --
    # a reimplementation of someone else's metric is a liability, and this row has to be
    # comparable to the column it sits in. MIB's function now takes an optional `percentages`
    # (a backward-compatible kwarg added to the MIB clone), so this passes its OWN grid and
    # everything else -- normalisation, pruning, accuracy definition, both trapezoid rules --
    # is their code.
    #
    # THE GRID: 0.001, then each rung's own converged L0 as a proportion, then 1. The two
    # anchors are MIB's own integration limits, and because they are evaluated rather than
    # assumed, p=0.001 (k=0 here, the empty circuit) contributes a MEASURED faithfulness and
    # accuracy instead of the 0-and-corrupted-accuracy this script used to assert.
    grid = {}
    n_gates = rungs[0][3]
    unmasked = n_nodes - n_gates
    print(f"  graph scores {n_nodes}, trainer masks {n_gates} -> {unmasked} always-on "
          f"(expect 1: `input`); k = round(L0) + {unmasked}")
    for l1, L0, gpath, _ in rungs:
        k = int(round(L0)) + unmasked
        if k < 1 or k >= n_nodes:
            print(f"  l1={l1}: k={k} outside [1,{n_nodes}) -- degenerate run, skipped")
            continue
        # int(pct * n) must round-trip to k, so nudge up off the floor boundary.
        pct = (k + 0.5) / n_nodes
        if int(pct * n_nodes) != k:
            pct = k / n_nodes
        grid.setdefault(round(pct, 12), []).append((l1, L0, k, gpath))

    ps = sorted(p for p in grid if 0.001 < p < 1.0)
    if not ps:
        raise SystemExit("no rung proportions strictly inside (0.001, 1)")
    percentages = (0.001,) + tuple(ps) + (1.0,)

    # The clean and fully-ablated references are graph-independent (the corrupted score is the
    # EMPTY circuit), so MIB's function fills this dict on the first call and reuses it after.
    # Without it every rung re-measured both, which was three quarters of the cost: the two
    # llama3 ARC cells timed out at 8h having evaluated the full circuit six times over.
    refs = {}

    def run_grid(gpath, pcts):
        g = Graph.from_json(gpath)
        return evaluate_area_under_curve(
            model, g, dl, metrics, quiet=True, level="node", absolute=a.absolute,
            intervention="patching", percentages=pcts, refs=refs)

    # ANCHORS FIRST, and they double as the reference-cache warm-up. p=0.001 is k=0 on every cell
    # here (the empty circuit) and p=1 is the full one, so both are graph-independent and one
    # two-point call supplies them -- and it leaves `refs` populated for everything below.
    anchor = run_grid(rungs[0][2], (0.001, 1.0))

    # Each rung is read at ITS OWN proportion, so the frontier MIXES GRAPHS: rung i supplies the
    # point at p_i and nothing else. ONE-POINT call per rung, which is now literally one dataset
    # pass: the references come from the cache and the p=1 point is already in hand from the
    # anchor call. (It used to be a (p_i, 1.0) call with uncached references -- 4 passes to get
    # 1, i.e. the same full circuit re-evaluated once per rung.) Every point still lands on one
    # normalisation, now by construction rather than by coincidence.
    points = []
    for p in ps:
        l1, L0, k, gpath = grid[p][0]
        _, _, _, _, faiths, accs, _ = run_grid(gpath, (p,))
        points.append(dict(l1=float(l1), L0=L0, k=k, p=p,
                           faithfulness=faiths[0], accuracy=accs[0]))
        print(f"  l1={l1:>5}  L0={L0:8.2f} -> k={k:4d} (p={p:.4f})  "
              f"faith={faiths[0]:+.3f}  acc={accs[0]:.3f}")

    # The frontier's own scalars: MIB's two trapezoid rules over the mixed-graph curve. Computed
    # here rather than taken from any single run_grid call, because each of those integrates ONE
    # rung's ranking over the whole grid, which is the very thing this row exists to avoid.
    faith_curve = [anchor[4][0]] + [q["faithfulness"] for q in points] + [anchor[4][-1]]
    acc_curve = [anchor[5][0]] + [q["accuracy"] for q in points] + [anchor[5][-1]]
    xs = list(percentages)
    cpr = sum((xs[i + 1] - xs[i]) * (faith_curve[i] + faith_curve[i + 1]) / 2
              for i in range(len(xs) - 1))
    lx = [math.log(x) for x in xs]
    iia = (sum((lx[i + 1] - lx[i]) * (acc_curve[i] + acc_curve[i + 1]) / 2
               for i in range(len(xs) - 1)) / (lx[-1] - lx[0]))
    print(f"  frontier: CPR={cpr:.4f}  IIA={iia:.4f}  over {len(points)} rungs")

    verify = None
    if a.verify:
        # MIB's DEFAULT grid through the same call, on one rung, against its stored pkl. If this
        # does not match, nothing above is comparable to the rest of the table.
        v = run_grid(rungs[0][2], None)
        verify = dict(l1=rungs[0][0], faithfulnesses=v[4], accuracies=v[5],
                      area_under=v[1], acc_auc=v[6])
        print(f"  VERIFY l1={rungs[0][0]}: CPR={v[1]:.4f} acc_auc={v[6]:.4f}")
        print(f"    faith {[round(x, 3) for x in v[4]]}")

    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    fn = out / f"{a.task}_{a.model}_{a.split}.json"
    json.dump(dict(task=a.task, model=a.model, split=a.split, n_nodes=n_nodes, head=a.head,
                   absolute=a.absolute, l1s_requested=list(a.l1s), n_gates=n_gates,
                   unmasked=unmasked, percentages=list(percentages),
                   faith_curve=faith_curve, acc_curve=acc_curve, cpr=cpr, iia=iia,
                   verify=verify, points=points),
              open(fn, "w"), indent=1)
    print(f"wrote {fn} ({len(points)} rungs)")


if __name__ == "__main__":
    main()
