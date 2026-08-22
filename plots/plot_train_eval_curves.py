"""Eval score vs TRAINING STEP for every method, on one task/substrate, patched vs zeroed.

Every mask-learning run stores an 11-point `train_eval_log` (steps 0, 200, ..., 1800, 1999),
each entry a full re-scoring of the current mask on the 24-point sparsity grid. Plotting it
answers a question the end-of-run scatter cannot: whether a method's final circuit is its best
circuit, or whether the training objective walked away from the eval metric.

That gap is the point of the figure. Measured on this data (see the module note in
plots/plot_accauc_vs_faithauc.py for the settings themselves), MAttr's best-minus-final acc-AUC
is 0.007 under patching and 0.036 under zeroing, while its training loss is still falling at
step 1999 in 90% of zero runs. It is not overfitting -- the loss goes down and the metric goes
down with it -- it is the objective being sampled in the wrong place: the log-k schedule spends
62% (node) / 82% (mlp) of its steps below 7% density, which is exactly where the zero-ablated
model is already destroyed and both loss and gradient are noise. Under patching the same
schedule is well matched, because patched discrimination lives at 1-7%.

The three GRADIENT baselines are single-pass -- there is no training loop, so no trajectory.
They are drawn as horizontal dashed references at their one score, which is what they are:
a constant the trained methods have to beat.

Substrate note: the patched sweep only carries `train_eval_log` on the ARITHMETIC tasks at the
mlp substrates (the four SVA cells there predate the logging), so the default task is `weekdays`
-- the one with all 8 trained method-configs x 3 losses present in BOTH settings. `--task` will
happily accept an SVA task and then draw an empty patched row; the script prints the per-panel
series count so that is visible rather than silent.

Run:  uv run python plots/plot_train_eval_curves.py            -> plots/train_eval_curves.pdf
      uv run python plots/plot_train_eval_curves.py --metric faith_auc --task hours
"""
import argparse
import glob
import json
import os
import sys

import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_point, geom_hline, facet_grid, labs, theme, theme_set,
    theme_bw, element_text, element_line, element_blank, scale_color_manual, scale_linetype_manual,
    guides, guide_legend,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_accauc_vs_faithauc import METHODS, parse_method, on_model  # noqa: E402

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(6.5, 3.4),
        axis_title=element_text(size=8),
        axis_text=element_text(size=6),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.03,
        panel_spacing_y=0.03,
        strip_background=element_blank(),
        strip_text=element_text(size=7),
        legend_title=element_blank(),
        legend_text=element_text(size=6),
        legend_key_size=8,
        legend_position="top",
        legend_direction="horizontal",
        legend_margin=0,
    )
)

# (results dir, ablation-row label). Same two SETTINGS as the scatter figure: `Patched` ablates
# every non-top-k unit to its counterfactual source activation, `Zero-abl.` sets it to 0. The
# trained methods train THROUGH whichever one they are scored under, so a row is a different
# experiment, not a rescoring -- compare shapes across rows, not heights.
SOURCES = [("results/sva_sweep", "Patched"), ("results/sva_zeroabl", "Zero-abl.")]
LOSS_LABEL = {"ce": "CE", "acc": "acc", "logit_diff": "logit-diff"}
LOSS_ORDER = ["CE", "acc", "logit-diff"]
# Methods with a training loop, in legend order. Keys are METHODS keys so the colours come from
# palette.py like every other figure. `soft-log` (the "+hard" STE ablation) is kept here even
# though the scatter figure drops it -- there it overlaps MAttr, but its trajectory is the whole
# reason to look: it diverges 0.069 best-to-final under zeroing, twice MAttr's 0.036.
TRAINED = ["stopk-log", "soft-log", "eprun-s090", "sig_lr0.3_l16.0"]
UNTRAINED = ["IG", "IxG", "AttnLRP"]
METRIC_LABEL = {
    "acc_auc": "accuracy-AUC",
    "faith_auc": "faithfulness-AUC",
    "kstar_50": "k* (density for 50% faithfulness)",
    "cause_accsrc_auc": "cause accuracy-AUC",
}


def collect(task, substrate, metric):
    """-> (long frame of trajectories, frame of single-pass baseline levels)."""
    traj, flat = [], []
    for res, abl in SOURCES:
        for f in glob.glob(f"{res}/*.json"):
            d = json.load(open(f))
            if d["task"] != task or d["nodes"] != substrate:
                continue
            # `--task ioi` would otherwise draw BOTH the canonical qwen2.5 run and the stray
            # llama3 one as separate rows of the same method, since nothing here keys on model.
            if not on_model(d):
                continue
            m = parse_method(os.path.basename(f), d)
            if m not in TRAINED and m not in UNTRAINED:
                continue
            row = dict(method=METHODS[m][0], abl=abl, loss=LOSS_LABEL[d["loss"]])
            if m in UNTRAINED:
                # No training loop: one score, drawn as a level. `train_eval_log` is absent
                # rather than length-1, so read the top-level field.
                flat.append({**row, "y": d[metric]})
                continue
            for e in d.get("train_eval_log") or []:
                traj.append({**row, "step": e["step"], "y": e[metric]})
    return pd.DataFrame(traj), pd.DataFrame(flat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="weekdays")
    ap.add_argument("--substrate", default="mlp", choices=["mlp", "mlp+attn_head", "node"])
    ap.add_argument("--metric", default="acc_auc", choices=sorted(METRIC_LABEL))
    ap.add_argument("--out", default="plots/train_eval_curves.pdf")
    a = ap.parse_args()

    traj, flat = collect(a.task, a.substrate, a.metric)
    if traj.empty:
        sys.exit(f"no train_eval_log for task={a.task} substrate={a.substrate}")

    order = [METHODS[m][0] for m in TRAINED + UNTRAINED]
    cmap = {METHODS[m][0]: METHODS[m][1] for m in TRAINED + UNTRAINED}
    for df in (traj, flat):
        if df.empty:
            continue
        df["method"] = pd.Categorical(df["method"], [c for c in order if c in set(df["method"])])
        df["loss"] = pd.Categorical(df["loss"], LOSS_ORDER)
        df["abl"] = pd.Categorical(df["abl"], [s[1] for s in SOURCES])

    for (abl, loss), g in traj.groupby(["abl", "loss"], observed=True):
        n = g["method"].nunique()
        print(f"{abl:>10s} / {loss:<10s} {n} trained series"
              + ("" if n == len(TRAINED) else f"  (MISSING {len(TRAINED) - n})"))

    p = (
        ggplot(traj, aes("step", "y", color="method"))
        + geom_hline(flat, aes(yintercept="y", color="method"), linetype="dashed", size=0.4)
        + geom_line(size=0.5)
        + geom_point(size=0.7)
        + facet_grid("abl ~ loss")
        + scale_color_manual(values=cmap, name="")
        + labs(x="training step", y=METRIC_LABEL[a.metric])
        + guides(color=guide_legend(nrow=1))
    )
    p.save(a.out, verbose=False)
    print(f"\n{a.out}  ({a.task}, {a.substrate}, {a.metric}; "
          f"dashed = single-pass baselines, no training loop)")

    # The number the prose actually quotes: how far the FINAL circuit sits below the best one
    # the run passed through. Positive = training walked away from the metric.
    d = (traj.sort_values("step").groupby(["abl", "loss", "method"], observed=True)["y"]
         .agg(best="max", final="last").reset_index())
    d["drop"] = d["best"] - d["final"]
    print("\nbest - final (" + a.metric + "):")
    print(d.pivot_table(index="method", columns="abl", values="drop", observed=True)
          .round(3).to_string())


if __name__ == "__main__":
    main()
