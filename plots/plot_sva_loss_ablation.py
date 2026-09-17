"""Attribution-loss ablation of the headline \\ourmethod{} on SVA+: logit-diff vs CE vs soft-acc,
per task, CPR and Compactness.

    uv run python plots/plot_sva_loss_ablation.py [--method stopk-unif-eps1e-2] [--sub node]
    -> plots/sva_loss_ablation.pdf (+ .png); copied to figs/ for sections/detailed-sva.tex.

THE RUNS. The node-substrate headline arm (soft top-k, uniform k, Adam eps 1e-2, 2000 steps;
scripts/sva/launch/submit_unifk_eps_node_sc.sh) trained under the three task losses eval_sva.py
offers -- `logit_diff` (the paper's default, base minus source logit), `ce` (cross-entropy on
the base answer) and `acc` (a soft accuracy, sigmoid of the margin at --acc-temp) -- on all ten
SVA+ tasks (IOI on Qwen-2.5, the rest on Llama-3, as everywhere in this family). Scores come
from plot_accauc_vs_faithauc.load, the loader behind the SVA tables, so a cell here is the
table's cell: CPR is the linear-p AUC (tabs/sva_results.tex), Compactness is `acc_auc`.

LAYOUT. One group per task in the table's order (SVA, Arith, ARC-E, IOI), three bars per group
in the loss order above, a 13th "Avg" group after a rule with each loss's mean over the ten
tasks (a plain mean over cells, like the table's Avg; NOT the macro-average the scatter
figures use). Two stacked full-width panels, separate y scales. Logit-diff keeps the method's
own colour, since it is the headline; CE and soft-acc take the two Wong hues the objective
ablation uses for its non-headline arms, so the two ablation figures read the same way.

A missing cell leaves a gap and is named on stdout. scripts/sva/loss_paired_tests.py has the
paired significance tests for the same comparison.
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import palette as P                              # noqa: E402
import plot_accauc_vs_faithauc as V              # load(), SVA, ARITH, TASK_MODEL  # noqa: E402

LOSSES = [("logit_diff", "logit-diff", P.METHOD["MAttr"]), ("ce", "CE", "#e69f00"), ("acc", "soft-acc", "#009e73")]
TASKS = list(V.SVA) + list(V.ARITH) + ["arc_easy", "ioi"]
TASK_LABEL = {"nounpp": "NounPP", "rc": "RC", "simple": "Simple", "within_rc": "Within RC",
              "addition": "Addition", "months": "Months", "weekdays": "Weekdays", "hours": "Hours",
              "arc_easy": "ARC-E", "ioi": "IOI"}
# CPR is the LINEAR AUC (index 2, V._cpr_of), never the log-weighted faith_auc (index 1) --
# user decision 2026-09-17, applied to tabs/sva_results.tex the same day.
METRICS = [(2, "CPR (↑)"), (0, "Compactness (↑)")]     # index into load()'s (acc_auc, faith_auc, cpr)
FIG_W, PANEL_H, FOOT, HEAD = 5.5, 1.05, 0.40, 0.22
FS_AXIS, FS_TICK, FS_ANNOT, FS_LEG = 6.5, 5.5, 4.0, 6.0
BAR_W = 0.26


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results/sva_sweep")
    ap.add_argument("--method", default="stopk-unif-eps1e-2", help="parse_method key of the arm")
    ap.add_argument("--sub", default="node")
    ap.add_argument("--out", default="plots/sva_loss_ablation.pdf")
    a = ap.parse_args()
    raw = V.load(a.res)
    data = {loss: {t: raw.get((a.method, loss, a.sub, t)) for t in TASKS} for loss, _, _ in LOSSES}
    for loss, lab, _ in LOSSES:
        miss = [t for t in TASKS if data[loss][t] is None]
        print(f"  {lab}: {len(TASKS) - len(miss)}/{len(TASKS)} cells" + (f"; missing {miss}" if miss else ""))
    if all(v is None for d in data.values() for v in d.values()):
        raise SystemExit(f"no cells for method {a.method!r} / substrate {a.sub!r} in {a.res}")

    plt.rcParams.update(P.RC)
    fh = HEAD + len(METRICS) * PANEL_H + FOOT
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(FIG_W, fh), sharex=True)
    groups = TASKS + ["avg"]
    xs = list(range(len(groups)))
    for ax, (key, ylab) in zip(axes, METRICS):
        for j, (loss, lab, colour) in enumerate(LOSSES):
            vals = []
            for g in groups:
                if g == "avg":
                    v = [data[loss][t][key] for t in TASKS if data[loss][t] is not None]
                    vals.append(sum(v) / len(v) if v else None)
                else:
                    r = data[loss][g]
                    vals.append(None if r is None else r[key])
            off = (j - (len(LOSSES) - 1) / 2) * BAR_W
            ax.bar([x + off for x in xs], [0 if v is None else v for v in vals], width=BAR_W,
                   color=colour, lw=0, zorder=2, label=lab)
            for x, v in zip(xs, vals):
                if v is not None:
                    ax.annotate(f"{v:.2f}", (x + off, v), textcoords="offset points", xytext=(0, 1.2),
                                ha="center", va="bottom", fontsize=FS_ANNOT, rotation=90, zorder=6)
        # task-group rules: after SVA, after Arith, after ARC-E, and before Avg
        for cut in (len(V.SVA) - 0.5, len(V.SVA) + len(V.ARITH) - 0.5, len(TASKS) - 1.5, len(TASKS) - 0.5):
            ax.axvline(cut, color="#999999", lw=0.5, ls=(0, (2, 2)), zorder=1)
        ax.set_ylabel(ylab, fontsize=FS_AXIS)
        top = max(v[key] for d in data.values() for v in d.values() if v is not None)
        ax.set_ylim(0, top * 1.28)
        ax.grid(True, lw=0.25, color="#dddddd"); ax.set_axisbelow(True); ax.grid(False, axis="x")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
        ax.tick_params(axis="y", labelsize=FS_TICK)
        ax.tick_params(axis="x", length=0)
    axes[-1].set_xticks(xs)
    axes[-1].set_xticklabels([TASK_LABEL[t] + ("\n(Qwen)" if V.TASK_MODEL[t] == "qwen2.5" else "") for t in TASKS] + ["Avg"],
                             fontsize=FS_TICK)
    axes[-1].set_xlim(-0.5 - BAR_W, len(groups) - 0.5 + BAR_W)
    fig.tight_layout(pad=0.3, h_pad=0.5)
    fig.subplots_adjust(top=1.0 - HEAD / fh)
    fig.legend(handles=[Patch(facecolor=c, label=lab) for _, lab, c in LOSSES],
               fontsize=FS_LEG, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.998),
               frameon=False, handlelength=1.2, handleheight=1.0, handletextpad=0.4, columnspacing=1.4)
    fig.savefig(a.out)
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=200)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
