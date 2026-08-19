"""Paper-ready SVA+ sweep figure: facet grid of metrics (rows) x task (cols), one PDF
per substrate. Bars = attribution method, filled by training/target loss
(logit_diff / ce / acc). Data: results/sva_sweep/*.json, plus results/sva_zeroabl/*.json
for the zero-ablation row.

The last-but-one row is the ZERO-ABLATION setting: non-top-k units are set to 0 rather than to
the counterfactual source activation. It is a distinct setting, not a rescoring -- MAttr trains
through it, and the gradient baselines change estimator (IxG -> Gradient x Input, IG ->
textbook zero-baseline IG). Reading it:

  * **The row is chance-corrected accuracy-AUC by default** (``--zero-metric``). Raw acc-AUC is
    meaningless here -- zeroing destroys the model to logit_diff ~ 0, so accuracy sits at a 0.5
    chance floor and a circuit with no signal reads ~0.5 -- and k* degenerates for the same
    reason. Correcting against the measured floor is the identity under patching, so this row
    IS on the same scale as the Accuracy-AUC row at the top and can be read against it.
    ``--zero-metric faith`` swaps in faith-AUC, which has no estimation noise but is unbounded
    and gap-paddable; in practice logit_diff-trained circuits pad to ~8 there and squash every
    other loss flat, which is why it is not the default.
  * **What to conclude from it is the ORDERING, not the heights.** The two settings agree only
    at Spearman ~0.44 over matched cells, which is the whole reason the row is worth drawing:
    zeroing induces a genuinely different ranking rather than a rescaling of the patched one.

Cells with no zero-ablation counterpart yet are NaN and are simply absent, not zero-height.

Run:  uv run python plots/plot_sva_sweep.py [--zero-metric acc|faith]
"""
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_col, geom_hline, facet_grid, labs, theme, theme_set, theme_bw,
    element_text, element_line, element_blank, position_dodge, scale_fill_brewer,
)

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(2.4, 1.7),  # ~70% — LaTeX upscales so text/lines feel larger
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=6, rotation=45, hjust=0.5, vjust=1.0),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.02,
        panel_spacing_y=0.02,
        strip_background=element_blank(),
        strip_text=element_text(size=7),
        legend_title=element_text(size=7),
        legend_text=element_text(size=6),
        legend_key_size=6,
        legend_position="top",
        legend_direction="horizontal",
        legend_box_margin=0,
    )
)

RES = Path("results/sva_sweep")
ZERO = Path("results/sva_zeroabl")   # the zero-ablation setting (see load_zero)
# SVA tasks are llama3; MIB tasks carry a /model suffix (node substrate only).
TASKS = ["nounpp", "rc", "simple", "within_rc",
         "addition", "months", "weekdays", "hours",   # arithmetic-wild, all three substrates
         "arc_easy", "ioi/qwen2.5"]
# metrics that have a measured random-ranking baseline (drawn as a dashed red line)
RANDOM_METRICS = {"acc_auc", "faith_auc"}
# MAttr split into gate/optimizer family: soft = hard_topk + Adam (sigmoid-STE);
# idSTE = hard_topk_identity + SGD (identity-STE). Each x {log, uniform} k; -IG = mask-space
# integrated-gradient score update (--mattr-ig-steps>1), swept on log-k only.
METHOD_ORDER = ["IG", "IxG", "AttnLRP", "Cond",
                "stopk-log", "stopk-unif", "stopk-fixed",       # soft top-k fwd = headline
                "soft-log", "soft-unif", "soft-fixed", "idSTE-log", "idSTE-unif", "idSTE-fixed",
                "stopk-log-IG", "soft-log-IG", "idSTE-log-IG"]  # -fixed = fixed k=10%, no sched
LOSS_ORDER = ["logit_diff", "ce", "acc"]
# (json key, facet-strip label, log10-transform?)
# Row label for the zero-ablation facet, keyed by --zero-metric. Both are "higher is better",
# but only the corrected-accuracy one shares a scale with the patched Accuracy-AUC row.
# Two lines on purpose: facet_grid's right-hand strips render rotated, so label LENGTH is
# constrained by ROW HEIGHT (~1.45in), and a single-line version overruns into the row above.
ZERO_LABELS = {"acc": "Acc AUC zero-abl\n(chance-corr., ↑)", "faith": "Faith AUC\nzero-abl (↑)"}
METRICS = [
    ("acc_auc", "Accuracy AUC (↑)", False),
    ("kstar_50", "log₁₀ k* iso (↓)", True),      # iso: first k s.t. acc_base>=0.5 (in JSON)
    ("faith_auc", "Faithfulness AUC (↑)", False),
    # Zero-ablation SETTING (results/sva_zeroabl): non-top-k units set to 0 rather than to the
    # counterfactual source activation. Which metric fills this row is --zero-metric; the label
    # is rewritten to match in main(), so read it off the strip rather than assuming.
    ("zero_auc", ZERO_LABELS["acc"], False),
    ("cause_kstar", "log₁₀ k* cause (↓)", True), # cause: first k s.t. acc_source>=0.5 (computed)
]
CAUSE_THR = 0.5


def parse_method(fname: str, d: dict) -> str | None:
    """Method label from the filename tag (JSON doesn't store ig/ixg/mattr).

    Ordered most-specific-first, and STRICT: an unrecognised tag returns None rather than
    falling through to a baseline. The previous version ended in
    ``return "IxG" if tag.startswith("ixg") else "IG"``, which silently folded 113 headline
    soft-top-k MAttr runs and 52 AttnLRP runs into the "IG" bar -- so the plotted "IG" series
    was mostly not IG. This is the same failure already documented for ``eprun_`` below and in
    plot_accauc_vs_faithauc.py; keep the catch-all strict.
    """
    nodes_safe = d["nodes"].replace("+", "-")   # filenames use '-' not '+'
    tag = fname.split(f"_{nodes_safe}_", 1)[1].rsplit(".json", 1)[0]
    tag = tag.replace("_zeroabl", "")   # the ablation setting is a facet row, not a method
    if tag.startswith("random"):
        return "RANDOM"
    if tag.startswith("conductance"):
        return "Cond"
    if "hard_topk" in tag:
        fam = "idSTE" if "identity" in tag else "soft"
        ks = "fixed" if "fixedk" in tag else ("unif" if "uniformk" in tag else "log")
        ig = "-IG" if re.search(r"_ig\d+", tag) else ""
        return f"{fam}-{ks}{ig}"
    if "sufficient_topk_" in tag:   # soft top-k forward, no STE -- the headline since 2026-07-21
        ks = "fixed" if "fixedk" in tag else ("unif" if "uniformk" in tag else "log")
        ig = "-IG" if re.search(r"_ig\d+", tag) else ""
        return f"stopk-{ks}{ig}"
    # Node Pruning (eval_sva.py --method edge_pruning) matches none of the tests above.
    # This figure has no Node Pruning entry in its METHOD_ORDER, so exclude rather than
    # mislabel; see plot_accauc_vs_faithauc.py, which does plot the series.
    if tag.startswith("eprun_"):
        return None
    if tag.startswith("ixg"):
        return "IxG"
    if tag.startswith("attnlrp"):
        return "AttnLRP"
    if tag.startswith("ig"):
        return "IG"
    return None    # cause-trained (`necessary_*`), sig_*, and anything new: drop, don't guess


def _task_label(d):
    """SVA (llama3) tasks keep their name; MIB tasks on other models get a /model suffix."""
    return d["task"] if d["model"] == "llama3" else f"{d['task']}/{d['model']}"


def _key(d, method):
    return (_task_label(d), d["nodes"], d["loss"], method)


def _auc_of(xs, ya):
    """Trapezoid on a log-x grid, normalised by the log span. Mirrors eval_sva.py:738, which is
    the source of truth (scripts/compare_ablation.py carries the same three lines)."""
    lx = np.log10(np.asarray(xs, float))
    ya = np.asarray(ya, float)
    return float(np.sum((lx[1:] - lx[:-1]) * (ya[1:] + ya[:-1]) / 2) / (lx[-1] - lx[0]))


def load_zero(metric: str) -> dict:
    """(task, nodes, loss, method) -> one metric under the ZERO-ablation setting.

    Neither available metric is unambiguously right, so this is a flag (--zero-metric):

    * ``acc`` (default) -- **chance-corrected** accuracy-AUC. Raw acc-AUC is unusable here:
      zeroing destroys the model to logit_diff ~ 0, so `acc_base` (a binary base-vs-source
      preference) sits at a 0.5 CHANCE FLOOR rather than patching's 0.0, and a circuit carrying
      no signal reads ~0.5. Renormalising against the measured floor,
      ``acc' = (acc - acc[0]) / (1 - acc[0])``, restores the spread; it is the identity when
      acc[0] = 0, so this row is directly comparable to the patched Accuracy-AUC row above.
      Cost: acc[0] is a finite-sample estimate (~+-0.05 at 100 examples), so the row is noisier
      than the patched one.
    * ``faith`` -- faith-AUC. No estimation noise, but unbounded and gap-paddable, and its
      denominator (F_clean - F_patch) roughly halves under zeroing (F_patch -3.57 -> -0.05),
      inflating values ~1.9x. In practice logit_diff-trained circuits pad to ~8 and squash every
      other loss to invisibility, which is why it is not the default.

    Either way, k* is NOT offered: it is an absolute 0.5 threshold (eval_sva.py:754) and
    degenerates outright under zeroing. See scripts/compare_ablation.py.
    """
    out = {}
    for f in sorted(glob.glob(str(ZERO / "*.json"))):
        d = json.load(open(f))
        method = parse_method(Path(f).name, d)
        if method is None or method == "RANDOM":
            continue
        if metric == "faith":
            v = float(d["faith_auc"])
        else:
            acc = d["iso_metrics"]["acc_base"]
            a0 = acc[0]
            # n_nodes is the actual evaluated grid, so a change to the sweep's sparsity
            # schedule can't silently desync this from the stored curve.
            v = np.nan if a0 == 1 else _auc_of(d["n_nodes"], [(a - a0) / (1 - a0) for a in acc])
        out[_key(d, method)] = v
    return out


def load(zero_metric: str = "acc") -> pd.DataFrame:
    """Method runs only (random-baseline files are excluded here; see random_lines())."""
    zero = load_zero(zero_metric)
    rows = []
    for f in sorted(glob.glob(str(RES / "*.json"))):
        d = json.load(open(f))
        method = parse_method(Path(f).name, d)
        if method is None or method == "RANDOM":
            continue
        rec = {"task": _task_label(d), "nodes": d["nodes"], "loss": d["loss"], "method": method}
        # cause k*: first k (n_nodes) at which corrupting the top-k gives acc_source >= thr
        cs = d["cause_metrics"]["acc_source"]
        cause_kstar = next((float(k) for k, a in zip(d["n_nodes"], cs) if a >= CAUSE_THR), None)
        for key, _, is_log in METRICS:
            if key == "cause_kstar":
                v = cause_kstar
            elif key == "zero_auc":
                # NaN where the zero sweep has no counterpart yet -- plotnine drops those bars,
                # so a missing cell reads as absent rather than as a zero-height result.
                v = zero.get(_key(d, method))
            else:
                v = d.get(key)
            rec[key] = (np.nan if v is None
                        else float(np.log10(v)) if is_log else float(v))
        rows.append(rec)
    return pd.DataFrame(rows)


def random_lines() -> pd.DataFrame:
    """Measured random-ranking baselines -> dashed lines, one per (nodes, task, metric)."""
    lbl = {k: l for k, l, _ in METRICS}
    acc = {}   # (nodes, task) -> list of (acc_auc, faith_auc)
    for f in glob.glob(str(RES / "*.json")):
        d = json.load(open(f))
        if parse_method(Path(f).name, d) != "RANDOM":
            continue
        acc.setdefault((d["nodes"], _task_label(d)), []).append((d["acc_auc"], d["faith_auc"]))
    rows = []
    for (nodes, task), vals in acc.items():
        arr = np.array(vals)
        for key, col in (("acc_auc", 0), ("faith_auc", 1)):
            rows.append({"nodes": nodes, "task": task, "metric": lbl[key],
                         "yintercept": float(arr[:, col].mean())})
    return pd.DataFrame(rows)


def long_form(df: pd.DataFrame) -> pd.DataFrame:
    m = df.melt(id_vars=["task", "nodes", "loss", "method"],
                value_vars=[k for k, _, _ in METRICS],
                var_name="metric", value_name="value")
    m["metric"] = pd.Categorical(m["metric"].map({k: lbl for k, lbl, _ in METRICS}),
                                 categories=[lbl for _, lbl, _ in METRICS], ordered=True)
    m["task"] = pd.Categorical(m["task"], categories=TASKS, ordered=True)
    m["method"] = pd.Categorical(m["method"], categories=METHOD_ORDER, ordered=True)
    m["loss"] = pd.Categorical(m["loss"], categories=LOSS_ORDER, ordered=True)
    return m


def plot_substrate(m: pd.DataFrame, nodes: str, out: Path, rand: pd.DataFrame):
    sub = m[m["nodes"] == nodes].copy()
    sub["task"] = sub["task"].cat.remove_unused_categories()   # drop tasks absent for substrate
    keep = list(sub["task"].cat.categories)
    n_tasks = len(keep)
    p = (
        ggplot(sub, aes("method", "value", fill="loss"))
        + geom_col(position=position_dodge(width=0.8), width=0.72)
        + facet_grid("metric ~ task", scales="free_y")
        + scale_fill_brewer(type="qual", palette="Set1")
        + labs(x="", y="", fill="Loss")
        # Width scales with #tasks, height with #metric rows (1.3in each). The per-task width
        # went 1.6 -> 2.1in when parse_method stopped folding AttnLRP and the stopk-* family
        # into "IG": 15 correctly-labelled methods need more room than the 8 that used to show.
        # Vertical tick labels for the same reason -- at 45 deg these names overlap.
        + theme(figure_size=(2.1 * n_tasks + 1.1, 1.45 * len(METRICS)),
                axis_text_x=element_text(size=5.5, rotation=90, hjust=0.5, vjust=1.0))
    )
    # dashed red random-ranking baseline where measured (acc/faith rows only)
    rl = rand[(rand["nodes"] == nodes) & (rand["task"].isin(keep))].copy()
    if len(rl):
        rl["metric"] = pd.Categorical(rl["metric"], categories=[l for _, l, _ in METRICS], ordered=True)
        rl["task"] = pd.Categorical(rl["task"], categories=keep, ordered=True)
        p = p + geom_hline(rl, aes(yintercept="yintercept"), linetype="dashed",
                           color="red", size=0.4)
    p.save(out, verbose=False)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zero-metric", default="acc", choices=["acc", "faith"],
                    help="metric for the zero-ablation row; see load_zero(). acc (default) is "
                         "the chance-corrected accuracy-AUC, which shares a scale with the "
                         "patched Accuracy-AUC row; faith is faith-AUC, which is gap-paddable "
                         "and in practice lets logit_diff dominate the row.")
    zm = ap.parse_args().zero_metric
    # METRICS carries the default label; rewrite it so the strip never disagrees with the data.
    METRICS[[k for k, _, _ in METRICS].index("zero_auc")] = ("zero_auc", ZERO_LABELS[zm], False)
    df = load(zm)
    rand = random_lines()
    print(f"loaded {len(df)} runs; nodes={sorted(df.nodes.unique())}; random lines={len(rand)}")
    m = long_form(df)
    for nodes in sorted(df["nodes"].unique()):
        plot_substrate(m, nodes, RES / f"sva_sweep_facet_{nodes.replace('+', '-')}.pdf", rand)


if __name__ == "__main__":
    main()
