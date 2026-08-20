"""Read out the neuron-substrate LR sweep from submit_sva_mlp_lr.sh.

Prints acc-AUC / faith-AUC / k*_50 for the sweep next to every baseline on the same cell
(addition / llama3 / --nodes mlp), plus the score-spread diagnostic: std(scores) against the
sigmoid_topk gate temperature T=1.0. Spread far below T means the gates never left the linear
region (undertrained, ranking ~ one-shot gradient); far above means saturated.

RESULT, 2026-08-20 -- lr=0.05 (the headline) is already the argmax: a 0.30-0.36 plateau over
0.005-0.3 decaying either side, against IG's 0.500. LR does move saturation across four orders
of magnitude (std 0.17 -> 261 vs T=1), so the knob works and is simply already tuned. This
falsifies the "gate slope ~ k/n so useful LR scales like n/k, and 0.05 must be far too small
at 2.29M units" argument, for the Adam/soft-top-k arm. (SGD has never been swept at neuron
scale, so it survives there.)

DO NOT READ THE FLAT CURVE AS A CEILING. Every run here is still rising at step 1999 (+0.03 to
+0.09 over its last 800 probe steps), and per eval_sva.py:773 a rising probe means UNDER-
CONVERGED. The pre-existing results/probe_* sweep -- 29 runs, same cell, --loss acc, lr crossed
with step budget -- shows the budget is the binding constraint and that the lr optimum moves
with it: topk goes 0.367 (2k) -> 0.465 (8k) -> 0.472 (16k) at lr=0.02, i.e. ~5x the entire
spread of this grid, and 0.02 overtakes 0.05 once the budget grows. id-STE at lr=0.05/16000
reaches 0.502, matching IG. So the gradient-methods win at this substrate is NOT established
at convergence, and the open experiment is an 8000-step step-matched re-run at logit_diff --
not more lr points. See plots/plot_sva_mlp_lr_probe.py for both halves in one figure.

Both halves read only from disk; safe to re-run. Run from the repo root.
"""

import json, glob, os, re

def rd(p):
    try: d = json.load(open(p))
    except Exception: return None
    return d

def f3(v): return "   n/a" if v is None else f"{v:6.3f}"
def fk(v): return "       n/a" if v is None else f"{v:10.0f}"

print("== reference (results/sva_sweep, addition/llama3/mlp, 2,293,760 units) ==")
print(f"{'method':40s} {'acc_auc':>7s} {'faith':>7s} {'k*50':>10s}")
refs = [
    ("IG",                     "addition_llama3_mlp_ig.json"),
    ("AttnLRP",                "addition_llama3_mlp_attnlrp.json"),
    ("IxG",                    "addition_llama3_mlp_ixg.json"),
    ("DBM sig_lr0.3_l16.0",    "addition_llama3_mlp_sig_lr0.3_l16.0.json"),
    ("eprun s090",             "addition_llama3_mlp_eprun_s090.json"),
    ("id-STE / SGD",           "addition_llama3_mlp_sufficient_hard_topk_identity_sgd_bs1.json"),
    ("+hard (hard_topk/Adam)", "addition_llama3_mlp_sufficient_hard_topk_adam_bs1.json"),
    ("MAttr headline lr=0.05", "addition_llama3_mlp_sufficient_topk_adam_bs1.json"),
]
for name, fn in refs:
    r = rd(os.path.join("results/sva_sweep", fn))
    if r is None:
        print(f"{name:40s}   MISSING"); continue
    print(f"{name:40s} {f3(r.get('acc_auc'))} {f3(r.get('faith_auc'))} {fk(r.get('kstar_50'))}")

print()
print("== sweep (results/sva_mlp_lr/topk_adam) ==")
print(f"{'lr':>8s} {'acc_auc':>7s} {'faith':>7s} {'k*50':>10s}  file")
def lrkey(p):
    m = re.search(r"lr_([0-9.]+)", p)
    return float(m.group(1)) if m else 0.0
for d in sorted(glob.glob("results/sva_mlp_lr/topk_adam/lr_*"), key=lrkey):
    js = sorted(p for p in glob.glob(os.path.join(d, "*.json")))
    if not js:
        print(f"{lrkey(d):8g}   (no json yet)"); continue
    for p in js:
        r = rd(p)
        print(f"{lrkey(d):8g} {f3(r.get('acc_auc'))} {f3(r.get('faith_auc'))} {fk(r.get('kstar_50'))}  {os.path.basename(p)}")

print()
print("== score spread vs gate temperature T=1.0 ==")
import torch
sp = [("0.05", "results/sva_sweep/addition_llama3_mlp_sufficient_topk_adam_bs1.scores.pt")]
for d in sorted(glob.glob("results/sva_mlp_lr/topk_adam/lr_*"), key=lrkey):
    g = glob.glob(os.path.join(d, "*.scores.pt"))
    if g:
        sp.append((f"{lrkey(d):g}", g[0]))
print(f"{'lr':>8s} {'n':>10s} {'std':>9s} {'max-min':>9s} {'frac>T from med':>16s}")
for lab, p in sorted(sp, key=lambda x: float(x[0])):
    s = torch.load(p, map_location="cpu")
    if isinstance(s, dict):
        s = list(s.values())[0]
    s = s.float().flatten()
    med = s.median()
    print(f"{lab:>8s} {s.numel():10d} {s.std():9.4f} {(s.max() - s.min()):9.3f} "
          f"{((s - med).abs() > 1).float().mean():16.4f}")
