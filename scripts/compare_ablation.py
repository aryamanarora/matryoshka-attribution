"""Compare the two SVA+ ablation SETTINGS: patch (results/sva_sweep) vs zero (results/sva_zeroabl).

The zero setting sets every non-top-k unit to 0 instead of to the counterfactual source
activation. It is a setting, not a rescoring -- MAttr trains through the same intervention it is
scored under, and the gradient baselines change estimator (IxG -> Gradient x Input, IG ->
textbook zero-baseline IG). So only RANKINGS WITHIN a setting are meaningful; the absolute
numbers are on different scales, for a reason this script quantifies.

Two things it exists to handle:

1. **acc-AUC has a chance floor under zeroing and must be corrected.** Zeroing everything
   destroys the model to logit_diff ~ 0, so `acc_base` -- a binary base-vs-source preference --
   sits at 0.5 rather than patching's 0.0. Raw `acc_auc` therefore reads ~0.5 for a circuit that
   carries no signal at all, which compresses the method spread and can invert it. The fix is to
   renormalise against the measured floor:

       acc' = (acc - acc[0]) / (1 - acc[0])

   `acc[0]` is the smallest-k point, i.e. the (near-)fully-ablated model. Under patching
   acc[0] = 0 and this reduces to the identity, so the same column is valid in both settings.
   Caveat: acc[0] is a finite-sample estimate (~+-0.05 at 100 examples), so the corrected
   column is noisier than the patched raw one. Prefer faith-AUC when the two disagree.

2. **faith-AUC is correctly normalised but changes SCALE between settings.** It divides by
   (F_clean - F_patch), and F_patch moves from ~-3.0 (patched: the model flips to the source
   answer) to ~-0.05 (zeroed: the model is simply destroyed). The denominator roughly halves,
   so zero-setting faith values are inflated ~1.9x relative to patched ones. Within-setting
   rankings are unaffected; cross-setting absolute comparisons are meaningless.

Run:  uv run python scripts/compare_ablation.py
"""
import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "plots"))
# This file deliberately keeps its OWN parse_method (see below), but the task->model pin is not a
# parsing choice -- it is a fact about what is on disk, so it is imported rather than restated.
from plot_accauc_vs_faithauc import on_model   # noqa: E402


def parse_method(fname, nodes):
    """Method label from the filename tag.

    NOTE: this deliberately does NOT reuse plots/plot_sva_sweep.py's parse_method, which tests
    `"hard_topk" in tag` and so lets the plain-`topk` variant fall through to a catch-all that
    returns "IG". Plain `topk` (soft top-k forward) is the HEADLINE MAttr variant as of
    2026-07-21, so that path silently averages the headline into the IG series. Order the tests
    most-specific-first to keep that from recurring here.
    """
    tag = fname.split(f"_{nodes.replace('+', '-')}_", 1)[1].rsplit(".json", 1)[0]
    tag = tag.replace("_zeroabl", "")
    if "hard_topk_identity" in tag:
        fam = "MAttr-idSTE"
    elif "hard_topk" in tag:
        fam = "MAttr+hard"
    elif "sufficient_topk" in tag:
        # Soft top-k forward = headline. The optimizer has to be in the family name: the
        # 2026-08-21 `topk:sgd` arm shares this tag prefix and would otherwise be averaged into
        # the headline MAttr rows -- the same silent-folding failure this docstring warns about.
        fam = "MAttr-SGD" if "_topk_sgd" in tag else "MAttr"
    elif tag.startswith("ixg"):
        return "IxG"
    elif tag.startswith("attnlrp"):
        return "AttnLRP"
    elif tag.startswith("ig"):
        return "IG"
    else:
        return None                        # random / conductance / eprun: not in this grid
    return f"{fam}-{'unif' if 'uniformk' in tag else 'log'}"


def auc_of(xs, ya):
    """Trapezoid on a log-x grid, normalised by the log span -- matches eval_sva.py:738."""
    lx = np.log10(np.asarray(xs, float))
    ya = np.asarray(ya, float)
    return float(np.sum((lx[1:] - lx[:-1]) * (ya[1:] + ya[:-1]) / 2) / (lx[-1] - lx[0]))


def load(resdir):
    """(task, nodes, loss, method) -> record, for one setting."""
    out = {}
    for f in sorted(glob.glob(str(ROOT / resdir / "*.json"))):
        d = json.load(open(f))
        m = parse_method(Path(f).name, d["nodes"])
        if m is None:
            continue
        # results/sva_sweep holds a llama3 IOI wave; the key below has no model in it, so the
        # patch-vs-zero comparison could otherwise read the two settings off different models.
        if not on_model(d):
            continue
        acc = d["iso_metrics"]["acc_base"]
        a0 = acc[0]
        # n_nodes is the actual evaluated grid (verified identical to reconstructing it from
        # `total`); use it rather than rebuilding, so a change to the sweep grid can't desync.
        corr = auc_of(d["n_nodes"], [(a - a0) / (1 - a0) for a in acc]) if a0 != 1 else np.nan
        out[(d["task"], d["nodes"], d["loss"], m)] = dict(
            acc_auc=d["acc_auc"], acc_auc_corr=corr, faith_auc=d["faith_auc"],
            kstar_50=d["kstar_50"], a0=a0, F_clean=d["F_clean"], F_patch=d["F_patch"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patch-dir", default="results/sva_sweep")
    ap.add_argument("--zero-dir", default="results/sva_zeroabl")
    ap.add_argument("--loss", default=None, help="restrict to one training loss")
    a = ap.parse_args()

    P, Z = load(a.patch_dir), load(a.zero_dir)
    keys = sorted(set(P) & set(Z))
    if a.loss:
        keys = [k for k in keys if k[2] == a.loss]
    if not keys:
        raise SystemExit("no matched patch/zero cells")
    print(f"matched cells: {len(keys)}  (patch {len(P)}, zero {len(Z)})")
    print(f"tasks: {sorted({k[0] for k in keys})}")
    print(f"losses: {sorted({k[2] for k in keys})}\n")

    # --- why the two settings are not on one scale -------------------------------------------
    print("normalisation endpoints (median over matched cells):")
    for lab, S in (("patch", P), ("zero", Z)):
        fc = np.median([S[k]["F_clean"] for k in keys])
        fp = np.median([S[k]["F_patch"] for k in keys])
        print(f"  {lab:5s}  F_clean {fc:7.3f}   F_patch {fp:8.3f}   denom {fc - fp:7.3f}")
    ratio = np.median([(P[k]["F_clean"] - P[k]["F_patch"]) / (Z[k]["F_clean"] - Z[k]["F_patch"])
                       for k in keys])
    print(f"  -> zero denominator is {ratio:.2f}x smaller; zero faith-AUC is inflated by that.")
    print(f"  chance floor acc[0]: patch median {np.median([P[k]['a0'] for k in keys]):.3f}, "
          f"zero median {np.median([Z[k]['a0'] for k in keys]):.3f}")
    kdeg = sum(1 for k in keys if Z[k]["kstar_50"] == 1.0)
    print(f"  kstar_50 degenerate (==1.0): patch {sum(1 for k in keys if P[k]['kstar_50'] == 1.0)}"
          f"/{len(keys)}, zero {kdeg}/{len(keys)}\n")

    # --- per-method table --------------------------------------------------------------------
    by = defaultdict(lambda: defaultdict(list))
    for k in keys:
        for lab, S in (("patch", P), ("zero", Z)):
            for met in ("acc_auc", "acc_auc_corr", "faith_auc"):
                by[k[3]][f"{lab}.{met}"].append(S[k][met])
    hdr = (f"{'method':14s} {'n':>3s} | {'acc-AUC raw':>22s} | {'acc-AUC corrected':>22s} "
           f"| {'faith-AUC':>22s}")
    print(hdr)
    print(f"{'':14s} {'':>3s} | {'patch':>10s} {'zero':>11s} | {'patch':>10s} {'zero':>11s} "
          f"| {'patch':>10s} {'zero':>11s}")
    print("-" * len(hdr))
    order = sorted(by, key=lambda m: -np.median(by[m]["zero.faith_auc"]))
    for m in order:
        n = len(by[m]["patch.acc_auc"])
        cells = "".join(
            f" {np.median(by[m][f'patch.{met}']):10.3f} {np.median(by[m][f'zero.{met}']):11.3f} |"
            for met in ("acc_auc", "acc_auc_corr", "faith_auc"))
        print(f"{m:14s} {n:3d} |{cells}")

    # --- do the settings rank methods the same way? ------------------------------------------
    from scipy.stats import spearmanr
    print("\ncross-setting agreement over matched cells (Spearman):")
    for met in ("faith_auc", "acc_auc", "acc_auc_corr"):
        x = [P[k][met] for k in keys]
        y = [Z[k][met] for k in keys]
        ok = [i for i in range(len(x)) if np.isfinite(x[i]) and np.isfinite(y[i])]
        rho = spearmanr([x[i] for i in ok], [y[i] for i in ok]).statistic
        print(f"  {met:14s} rho = {rho:+.3f}")
    print("\nA low rho is the POINT: it means zeroing induces a genuinely different ranking,")
    print("which is what makes it worth reporting as a second setting rather than a rescaling.")


if __name__ == "__main__":
    main()
