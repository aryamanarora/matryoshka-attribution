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

RESULT OF THE OTHER TWO ARMS (2026-08-20) -- THE OPTIMIZER IS THE WHOLE STORY AT THIS
SUBSTRATE, NOT THE GATE. Completing the optimizer x gate cross (submit_sva_sweep.sh bundles
them and never crosses) moves acc-AUC far more than lr ever did:

    variant / optimizer        best acc-AUC over its lr grid
    topk  / sgd                0.496  (lr=1)      <- ties IG's 0.500
    id-STE / sgd               0.437  (on disk, sva_sweep)
    topk  / adam               0.361  (lr=0.05)
    id-STE / adam              0.349  (lr=0.05/0.1)

Same gate, swap Adam->SGD: 0.361 -> 0.496. Same optimizer, swap gate: at most 0.02. And
id-STE's 0.437 was bought entirely by its optimizer -- run it with Adam and the best MAttr
variant on this cell becomes the WORST, below the soft/Adam headline.

DO NOT REPORT "MAttr TIES IG" WITHOUT THE TWO CAVEATS BELOW.
 1. The winning SGD runs are in the UNDERTRAINED regime and are converging on IG's own answer.
    std(scores) is 0.001-0.03 against gate T=1.0, i.e. 30-1000x BELOW it, so the gates never
    leave sigmoid's linear region and the score is ~ -lr * sum_t g_t, an accumulated-gradient
    ranking. Measured: top-2082 overlap with the IG score vector is 0.73-0.77, HIGHER than
    IxG's own 0.55 overlap with IG. Where the gate actually engages (lr 30-100, spread ~1-2xT)
    overlap falls to 0.43-0.55 and acc-AUC falls to 0.44-0.46. So the tie is a gradient method
    in disguise, not evidence that mask learning solves this substrate.
 2. It costs 2000 forward+backward passes against IG's 10. Compute-matched, IG still wins by a
    mile; this is a "200x the compute buys a tie" result.
Faithfulness corroborates: the SGD arm's faith-AUC is 0.50-0.53, right on IG's 0.488, while
every Adam arm sits at 1.0-1.2 -- the logit_diff gap-padding signature (see the sweep memo).

ON THE lr-INVARIANCE PREDICTION FOR hard_topk_identity. It holds only MODULO FLOAT
TIE-BREAKING, which the node-level check could not see. Hard fwd reads only the ranking and
identity bwd reads no magnitude, so scores = lr x (fixed vector) -- but `scores` init to
exactly ZERO, so the step-0 top-k is fully degenerate and rounding breaks those ties
differently at different lr. Ratios that are exact in binary preserve the tie order and give
bit-identical results (2*s(0.005) - s(0.01) is EXACTLY 0 over all 2,293,760 entries, argsort
identical, acc-AUC 0.3317 both; likewise 0.05 and 0.1 both 0.3487). Ratios that are not (0.001
vs 0.005) diverge: Pearson 0.54, top-k overlap 0.35. Net effect is still a flat arm
(0.328-0.349), just not a constant one.
WHY ADAM IS SO BAD UNDER IDENTITY STE, precisely: with dL/ds = dL/dm exactly, Adam's update is
m_hat/(sqrt(v_hat)+eps) ~ sign(g), so the score becomes a signed COUNT of steps and all effect
magnitude is discarded. s/lr lands on a near-integer lattice (23% of units within 0.01 of an
integer) with std ~64 against sqrt(2000)=45 for a pure coin-flip walk -- the ranking is only
~1.4x above a random walk. That is the mechanism behind the node-level "SGD 5/5, +0.404".

Both halves read only from disk; safe to re-run. Run from the repo root (uv run python, for torch).
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

def lrkey(p):
    m = re.search(r"lr_([0-9.]+)", p)
    return float(m.group(1)) if m else 0.0

# The pre-existing headline run is the topk_adam arm's lr=0.05 point; it lives in the shared
# sweep dir because run_tag() does not encode lr, so it is grafted in rather than re-run.
CENTRE = {"topk_adam": ("0.05",
          "results/sva_sweep/addition_llama3_mlp_sufficient_topk_adam_bs1.json")}
ARMS = sorted(d for d in glob.glob("results/sva_mlp_lr/*") if os.path.isdir(d))

for arm in ARMS:
    tag = os.path.basename(arm)
    print()
    print(f"== arm {tag} ({arm}) ==")
    print(f"{'lr':>8s} {'acc_auc':>7s} {'faith':>7s} {'k*50':>10s}  file")
    rows = []
    if tag in CENTRE:
        lab, p = CENTRE[tag]
        if os.path.exists(p):
            rows.append((float(lab), p))
    for d in glob.glob(os.path.join(arm, "lr_*")):
        js = sorted(glob.glob(os.path.join(d, "*.json")))
        rows.extend((lrkey(d), p) for p in js)
        if not js:
            rows.append((lrkey(d), None))
    for lr, p in sorted(rows):
        if p is None:
            print(f"{lr:8g}   (no json yet)"); continue
        r = rd(p)
        print(f"{lr:8g} {f3(r.get('acc_auc'))} {f3(r.get('faith_auc'))} {fk(r.get('kstar_50'))}"
              f"  {os.path.basename(p)}")

print()
print("== score spread vs gate temperature T=1.0 ==")
import torch
for arm in ARMS:
    tag = os.path.basename(arm)
    sp = []
    if tag in CENTRE:
        lab, p = CENTRE[tag]
        p = p.replace(".json", ".scores.pt")
        if os.path.exists(p):
            sp.append((float(lab), p))
    for d in glob.glob(os.path.join(arm, "lr_*")):
        g = glob.glob(os.path.join(d, "*.scores.pt"))
        if g:
            sp.append((lrkey(d), g[0]))
    if not sp:
        continue
    print(f"-- {tag}")
    print(f"{'lr':>8s} {'n':>10s} {'std':>9s} {'max-min':>9s} {'frac>T from med':>16s}")
    for lr, p in sorted(sp):
        s = torch.load(p, map_location="cpu")
        if isinstance(s, dict):
            s = list(s.values())[0]
        s = s.float().flatten()
        med = s.median()
        print(f"{lr:8g} {s.numel():10d} {s.std():9.4f} {(s.max() - s.min()):9.3f} "
              f"{((s - med).abs() > 1).float().mean():16.4f}")
