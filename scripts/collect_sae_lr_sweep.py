"""Read the residual-SAE LR sweep: does a smaller lr stop MAttr selecting dead latents?

WHY THREE COLUMNS AND NOT ONE. The sweep tests a specific causal story, and acc_auc alone cannot
confirm or refute it -- an lr that raises the score while still selecting 100% inert latents has
found a different effect, not fixed this one. So every row reports:

  acc_auc    the outcome (IIA AUC, 100 held-out examples)
  dead@10k   the mechanism: fraction of the top-10,000 that never fires, so f_b = f_c = 0 and
             masking it is a no-op. At the shipped lr this is 1.000 for BOTH MAttr arms on all 8
             tasks, against 0.000 for IG and 0.978 for Random -- MAttr is worse than chance.
  range      max-min of the learned scores. This is the saturation tell: sigmoid_topk's backward
             carries a factor sp = m(1-m), which dies once |s-tau| >> T, so a large range means
             most of the mask froze early and the plateau in the training probe is gradient
             death rather than convergence. At lr=1.0 the range is 2.6e7 on addition.
  trainable  fraction of units still inside the active band (|s-tau_k| <= 5T) at k=10^4 -- the
             same statement as `range`, but read at the k the selection actually happens at.

*** THE FIX CONDITION IS dead@10k COLLAPSING, NOT acc_auc RISING. *** If the best-scoring lr still
shows dead@10k near 1.0, the diagnosis in scripts/submit_sae_lr_sweep.sh is wrong and the score
moved for some other reason.

lr AND T ARE ONE KNOB (lr/T^2), so there is no T axis here -- see the submit script's header for
the first-step derivation. Do not add one.

DEAD IS DEFINED BY IG'S NONZERO SUPPORT on its own grad-examples, which is a proxy for "inert on
the eval examples" rather than the thing itself: a latent that fires only outside IG's sample
would be miscounted. The proxy is load-bearing enough to state, and it is self-consistent --
Random's dead@k lands exactly at the dead fraction of the basis (0.978 vs ~98% dead), which is
what it must do if the support estimate is right.

Run:  uv run python scripts/collect_sae_lr_sweep.py
"""
import glob
import json
import os
import re
import sys

import numpy as np
import torch

ROOT = "results/sae_lr_sweep"
D_SAE = 32768
K = 10_000
T = 0.5
TAG = {"sgd": "sufficient_topk_sgd_bs1", "adam": "sufficient_topk_adam_eps1e-2_bs1"}


def stats(scores_path, supp):
    s = torch.load(scores_path, map_location="cpu").float().numpy()
    tk = np.argpartition(s, -K)[-K:]
    tau = np.partition(s, -K)[-K]
    return (1.0 - supp[tk].mean(), float(s.max() - s.min()),
            float(np.mean(np.abs(s - tau) <= 5 * T)))


def main():
    dirs = sorted(glob.glob(f"{ROOT}/*_lr*"))
    if not dirs:
        raise SystemExit(f"no sweep dirs under {ROOT} -- run scripts/submit_sae_lr_sweep.sh")

    # The reference support comes from the MAIN sweep's IG run on the same (task, substrate),
    # not from this dir -- the sweep only trains MAttr, so there is no IG here to compare to.
    # Two directory spellings coexist on purpose: the first wave was submitted before the script
    # grew a VARIANT knob and landed in `<opt>_lr<x>`, the later arms in `<variant>_<opt>_lr<x>`.
    # Renaming the old dirs would orphan the running jobs' baked-in --output, so parse both and
    # take the VARIANT from the filename tag (`sufficient_<variant>_<opt>[_eps..]_bs1`), which is
    # authoritative either way because run_tag() encodes --variant but not --lr.
    rows, ref = [], {}
    for d in dirs:
        # The optional `_eps<x>` segment sits BETWEEN the optimizer and the lr
        # (topk_log_adam_eps1e-8_lr1.0), so it has to be matched explicitly -- without it the
        # low-eps dirs simply do not match and are silently dropped from the table.
        m = re.match(r".*?/(?:(.+?)_)?(sgd|adam)(?:_eps([^_]+))?_lr(.+)$", os.path.normpath(d))
        if not m:
            continue
        opt, eps, lr = m.group(2), m.group(3), m.group(4)
        for f in glob.glob(f"{d}/*_sufficient_*_{opt}*_bs1.json"):
            base = os.path.basename(f)
            task = base.split("_llama3_")[0]
            nodes = "resid_sae_span"
            variant = base.split("_sufficient_")[1].split(f"_{opt}")[0]
            # eps is an IDENTITY knob at this scale (it decides whether the update
            # keeps gradient MAGNITUDE or collapses to sign(g)), so it belongs in the
            # row label, not hidden in the directory name.
            if eps:
                variant = f"{variant} eps={eps}"
            if task not in ref:
                p = f"results/sva_sweep/{task}_llama3_{nodes}_ig.scores.pt"
                if not os.path.exists(p):
                    print(f"  ! no IG reference for {task}, skipping dead@k")
                    ref[task] = None
                else:
                    ig = torch.load(p, map_location="cpu").float().numpy()
                    ref[task] = ig != 0
            d_ = json.load(open(f))
            sp = f.replace(".json", ".scores.pt")
            dead = rng = tr = float("nan")
            if os.path.exists(sp) and ref[task] is not None:
                dead, rng, tr = stats(sp, ref[task])
            rows.append((task, variant, opt, float(lr), d_.get("acc_auc"), d_.get("faith_auc"),
                         dead, rng, tr))

    if not rows:
        raise SystemExit(f"{len(dirs)} dirs but no finished runs yet")
    rows.sort(key=lambda r: (r[0], r[1], r[2], -r[3]))

    print(f"{'task':<10}{'variant':<20}{'opt':<6}{'lr':>9}{'acc_auc':>9}{'faith':>8}"
          f"{'dead@10k':>10}{'range':>11}{'trainable':>11}")
    prev = None
    for task, variant, opt, lr, acc, fa, dead, rng, tr in rows:
        if prev and prev != (task, variant, opt):
            print()
        prev = (task, variant, opt)
        a = f"{acc:>9.3f}" if acc is not None and np.isfinite(acc) else f"{'nan':>9}"
        b = f"{fa:>8.3f}" if fa is not None and np.isfinite(fa) else f"{'nan':>8}"
        print(f"{task:<10}{variant:<20}{opt:<6}{lr:>9.0e}{a}{b}"
              f"{dead:>10.3f}{rng:>11.3g}{tr:>11.2%}")

    print("\nbest acc_auc per (task, variant, optimizer), with the mechanism column beside it:")
    for key in sorted({(r[0], r[1], r[2]) for r in rows}):
        c = [r for r in rows if (r[0], r[1], r[2]) == key
             and r[4] is not None and np.isfinite(r[4])]
        if not c:
            continue
        b = max(c, key=lambda r: r[4])
        verdict = ("MECHANISM FIXED" if b[6] < 0.5 else
                   "score moved but still ~all-dead -> NOT this mechanism")
        print(f"  {key[0]:<10} {key[1]:<20} {key[2]:<5} lr={b[3]:<9.0e} "
              f"acc_auc={b[4]:.3f} dead@10k={b[6]:.3f}   {verdict}")
    print("\nreference on this task/basis (results/sva_sweep): IG acc_auc=0.302 dead@10k=0.000, "
          "Random dead@10k~0.97")
    return 0


if __name__ == "__main__":
    sys.exit(main())
