"""The 2026-08-20 neuron-substrate lr x {gate, optimizer} grid: what actually moves the score
at addition / llama3 / --nodes mlp (2,293,760 units).

Data: results/sva_mlp_lr/<variant>_<optimizer>/lr_*, 2000 steps, bs=1, log-k, --loss
logit_diff, --mode sufficient. Three arms, completing the cross that submit_sva_sweep.sh's
MATTR_CONFIGS never runs (it bundles gate WITH optimizer and never crosses them):

    topk  / sgd    lr 0.05..300   best 0.496 @ lr=1
    topk  / adam   lr 0.001..10   best 0.361 @ lr=0.05   (the paper headline)
    id-STE/ adam   lr 0.001..10   best 0.349 @ lr=0.05
    id-STE/ sgd    single run on disk in results/sva_sweep, 0.437 (drawn as a reference line)

THE HEADLINE IS THE OPTIMIZER, NOT THE GATE OR THE lr. Holding the gate fixed and swapping
Adam -> SGD moves acc-AUC 0.361 -> 0.496; holding the optimizer fixed and swapping the gate
moves it at most ~0.02. The lr grid inside any one arm spans less than the gap between arms.

TWO THINGS THE FIGURE DELIBERATELY SHOWS SO THE TIE WITH IG IS NOT OVERSOLD.
 1. Left panel: the SGD arm is CONVERGED and the Adam arms are NOT. The three best SGD runs
    move -0.009/+0.009/-0.004 over their last 800 probe steps; every Adam run is still rising
    (+0.02 to +0.04). Per eval_sva.py:773 a rising probe means under-converged, so the earlier
    "the flat lr curve is an under-convergence artifact" reading applies to the Adam arms only.
 2. Right panel, second axis: top-2082 overlap between each run's score vector and IG's. The
    SGD runs that tie IG (0.73-0.77) overlap with IG MORE than IxG does (0.55, dotted line).
    Their score spread is 0.001-0.03 against gate T=1.0, i.e. 30-1000x below it, so the gates
    never leave sigmoid's linear region and the score is ~ -lr * sum_t g_t -- an
    accumulated-gradient ranking. Where the gate actually engages (lr 30-100) overlap falls to
    0.43-0.55 and so does acc-AUC. The tie is a gradient method in disguise, and it costs 2000
    passes against IG's 10.

Run:  uv run python plots/plot_sva_mlp_lr_probe.py   -> plots/sva_mlp_lr_probe.pdf
"""
import glob
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SWEEP = ROOT / "results/sva_mlp_lr"
SVA = ROOT / "results/sva_sweep"
# lr=0.05 of the topk/adam arm predates this grid and lives in the shared sweep dir (run_tag()
# does not encode lr), so it is grafted in as that arm's centre rather than re-run.
CENTRE = {"topk_adam": (0.05, SVA / "addition_llama3_mlp_sufficient_topk_adam_bs1.json")}
ARMS = [("topk_sgd", "soft top-k / SGD", "#1f77b4"),
        ("topk_adam", "soft top-k / Adam  (headline)", "#d62728"),
        ("hard_topk_identity_adam", "id-STE / Adam", "#7f7f7f")]
REFS = [("IG", "addition_llama3_mlp_ig.json", "crimson", ":"),
        ("id-STE / SGD", "addition_llama3_mlp_sufficient_hard_topk_identity_sgd_bs1.json",
         "#2ca02c", "--")]
K = 2082  # IG's k*_50 on this cell; the log-k grid point the overlap is measured at


def lrkey(p):
    return float(re.search(r"lr_([0-9.]+)", str(p)).group(1))


def load_scores(p):
    s = torch.load(p, map_location="cpu")
    if isinstance(s, dict):
        s = list(s.values())[0]
    return s.float().flatten()


def load_arm(tag):
    """(lr, probe curve, final test acc-AUC, scores path) per run, ascending in lr."""
    runs = []
    if tag in CENTRE:
        lr, f = CENTRE[tag]
        if f.exists():
            j = json.load(open(f))
            runs.append({"lr": lr, "probe": j.get("train_eval_log", []),
                         "test": j.get("acc_auc"),
                         "scores": f.with_suffix(".scores.pt")})
    for d in glob.glob(str(SWEEP / tag / "lr_*")):
        for f in glob.glob(f"{d}/*.json"):
            j = json.load(open(f))
            runs.append({"lr": lrkey(d), "probe": j.get("train_eval_log", []),
                         "test": j.get("acc_auc"),
                         "scores": Path(f).with_suffix(".scores.pt")})
    return sorted(runs, key=lambda r: r["lr"])


def main():
    arms = {tag: load_arm(tag) for tag, _, _ in ARMS}
    refs = {name: json.load(open(SVA / fn))["acc_auc"]
            for name, fn, _, _ in REFS if (SVA / fn).exists()}

    ig_scores = load_scores(SVA / "addition_llama3_mlp_ig.scores.pt")
    ig_top = set(torch.topk(ig_scores, K).indices.tolist())
    ixg_ov = len(ig_top & set(torch.topk(
        load_scores(SVA / "addition_llama3_mlp_ixg.scores.pt"), K).indices.tolist())) / K

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # ---- left: convergence. Whether a curve has flattened is the finding, not its height.
    for tag, label, c in ARMS:
        best = max(arms[tag], key=lambda r: r["test"])
        for r in arms[tag]:
            pts = [(p["step"], p["acc_auc"]) for p in r["probe"]
                   if p.get("acc_auc") is not None]
            if pts:
                ax0.plot(*zip(*pts), color=c, lw=2.0 if r is best else 0.9,
                         alpha=1.0 if r is best else 0.35, zorder=4 if r is best else 2)
        ax0.plot(2000, best["test"], "*", color=c, ms=14, mec="k", mew=.5, zorder=6)
    for name, _, c, ls in REFS:
        if name in refs:
            ax0.axhline(refs[name], color=c, ls=ls, lw=1.3, zorder=1)
            ax0.text(1990, refs[name] + .008, f"{name} = {refs[name]:.3f}",
                     color=c, size=8.5, ha="right")
    ax0.set_xlabel("training step")
    ax0.set_ylabel("acc-AUC $\\uparrow$")
    ax0.set_title("SGD converges by 2000 steps; Adam is still climbing\n"
                  "(bold = each arm's best lr, star = its 100-ex TEST value)", size=10)
    ax0.grid(alpha=.25, lw=.5)
    ax0.legend(handles=[plt.Line2D([], [], color=c, lw=2, label=lab) for _, lab, c in ARMS],
               fontsize=8.5, loc="lower right", frameon=False)

    # ---- right: the lr surface per arm, with "is this just IG?" on the twin axis.
    for tag, label, c in ARMS:
        ax1.plot([r["lr"] for r in arms[tag]], [r["test"] for r in arms[tag]],
                 "o-", color=c, lw=1.7, ms=5, label=label)
    for name, _, c, ls in REFS:
        if name in refs:
            ax1.axhline(refs[name], color=c, ls=ls, lw=1.3, zorder=1)
    ax1.set_xscale("log")
    ax1.set_xlabel("learning rate")
    ax1.set_ylabel("final test acc-AUC $\\uparrow$")
    ax1.grid(alpha=.25, lw=.5)
    ax1.legend(fontsize=8.5, loc="lower left", frameon=False)

    ax2 = ax1.twinx()
    for tag, _, c in ARMS:
        ov = [len(ig_top & set(torch.topk(load_scores(r["scores"]), K).indices.tolist())) / K
              if Path(r["scores"]).exists() else np.nan for r in arms[tag]]
        ax2.plot([r["lr"] for r in arms[tag]], ov, "^:", color=c, lw=1.0, ms=4, alpha=.55)
    ax2.axhline(ixg_ov, color="k", ls=(0, (1, 3)), lw=1.0)
    ax2.text(ax1.get_xlim()[1], ixg_ov + .012, f"I$\\times$G vs IG = {ixg_ov:.2f}",
             size=8, ha="right", color="k")
    ax2.set_ylabel(f"top-{K} overlap with IG (dotted $\\triangle$)", size=9)
    ax2.set_ylim(0, 1)
    ax1.set_title("the arms that tie IG are the arms that AGREE with IG\n"
                  "(solid = score, dotted = overlap with IG's ranking)", size=10)

    fig.suptitle("addition / llama3 / mlp (2,293,760 units): the optimizer moves acc-AUC "
                 "~7x more than the lr or the gate", size=11)
    fig.tight_layout(rect=[0, 0, 1, .93])
    out = ROOT / "plots/sva_mlp_lr_probe.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    print(f"wrote {out}")
    for tag, label, _ in ARMS:
        for r in arms[tag]:
            pts = [p["acc_auc"] for p in r["probe"] if p.get("acc_auc") is not None]
            rise = (pts[-1] - pts[-5]) if len(pts) >= 5 else float("nan")
            print(f"  {tag:24s} lr={r['lr']:<7g} test={r['test']:.3f}  "
                  f"probe last-800-step rise={rise:+.3f}")


if __name__ == "__main__":
    main()
