"""Appendix tables for experiments 1-4 (SVA sweep / granularity / harder tasks / STE x k-sched).

One LaTeX table per granularity (substrate). Rows = (method, train-loss); columns = tasks.
Cell = a single eval metric (default acc-AUC). No-input data (results/sva_sweep), the
MIB-comparable headline. Best method per (task) column is bolded.

Run:  uv run python scripts/make_sva_appendix_tables.py [--metric acc_auc|faith_auc|kstar_50]
Writes paper/tabs/sva_grid_<metric>_<substrate>.tex
"""
import argparse
import glob
import json
import os
from collections import defaultdict

RES = "results/sva_sweep"
TABDIR = "paper/tabs"

SUBSTRATES = [("mlp", "MLP neurons (per-token)"),
              ("mlp+attn_head", "MLP neurons + attn heads (per-token)"),
              ("node", "Node / MIB granularity (MLP block + attn head)")]
# task -> column label; node adds the two MIB tasks
SVA_TASKS = [("nounpp", "nounpp"), ("rc", "rc"), ("simple", "simple"), ("within_rc", "within-rc")]
NODE_EXTRA = [("arc_easy", "ARC-E"), ("ioi", "IOI (qwen)")]
# method key (from parse_method) -> latex row label; order defines row order
METHODS = [("IG", "IG"), ("IxG", "IxG"),
           ("soft-log", r"soft, log-$k$"), ("soft-unif", r"soft, unif-$k$"),
           ("soft-fixed", r"soft, fixed-$k$"),
           ("idSTE-log", r"idSTE, log-$k$"), ("idSTE-unif", r"idSTE, unif-$k$"),
           ("idSTE-fixed", r"idSTE, fixed-$k$")]
LOSSES = [("ce", "CE"), ("acc", "acc"), ("logit_diff", "logit-diff")]
METRIC_LABEL = {"acc_auc": "Accuracy AUC", "faith_auc": "Faithfulness AUC",
                "kstar_50": r"$k^\\star$ (acc$>$0.5)"}


def parse_method(fname, d):
    import re
    tag = fname.split("_" + d["nodes"].replace("+", "-") + "_", 1)[1].rsplit(".json", 1)[0]
    if tag.startswith("random"): return "RANDOM"
    if tag.startswith("conductance"): return "Cond"
    if "hard_topk" in tag:
        fam = "idSTE" if "identity" in tag else "soft"
        ks = "fixed" if "fixedk" in tag else ("unif" if "uniformk" in tag else "log")
        if re.search(r"_ig\d+", tag): return None      # skip MAttr-IG (exp 5)
        return f"{fam}-{ks}"
    return "IxG" if tag.startswith("ixg") else "IG"


def load(metric):
    # data[(nodes, method, loss, task)] = value
    data = {}
    for f in glob.glob(RES + "/*.json"):
        d = json.load(open(f))
        m = parse_method(os.path.basename(f), d)
        if m is None or m == "RANDOM":
            continue
        task = d["task"] if d["model"] == "llama3" else "ioi"   # ioi is qwen
        data[(d["nodes"], m, d["loss"], task)] = d.get(metric)
    return data


def fmt(v, metric, best):
    if v is None:
        return "---"
    s = (f"{v:.0f}" if metric.startswith("kstar") else f"{v:.2f}")
    return f"\\textbf{{{s}}}" if best else s


def make_table(data, nodes, subdesc, metric):
    tasks = SVA_TASKS + (NODE_EXTRA if nodes == "node" else [])
    # which methods actually have data for this substrate
    methods = [(mk, ml) for mk, ml in METHODS
               if any((nodes, mk, lk, tk) in data and data[(nodes, mk, lk, tk)] is not None
                      for lk, _ in LOSSES for tk, _ in tasks)]
    higher_better = not metric.startswith("kstar")
    # per-(task) best value across all (method,loss) for bolding
    best = {}
    for tk, _ in tasks:
        vals = [data.get((nodes, mk, lk, tk)) for mk, _ in methods for lk, _ in LOSSES]
        vals = [v for v in vals if v is not None]
        best[tk] = (max(vals) if higher_better else min(vals)) if vals else None

    col = "ll" + "c" * len(tasks)
    lines = [r"\begin{tabular}{" + col + "}", r"\toprule",
             "Method & Loss & " + " & ".join(tl for _, tl in tasks) + r" \\", r"\midrule"]
    for mi, (mk, ml) in enumerate(methods):
        if mi: lines.append(r"\midrule")
        for li, (lk, ll) in enumerate(LOSSES):
            cells = []
            for tk, _ in tasks:
                v = data.get((nodes, mk, lk, tk))
                cells.append(fmt(v, metric, v is not None and best[tk] is not None
                                 and abs(v - best[tk]) < 1e-9))
            name = ml if li == 0 else ""
            lines.append(f"{name} & {ll} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="acc_auc", choices=list(METRIC_LABEL))
    args = ap.parse_args()
    data = load(args.metric)
    os.makedirs(TABDIR, exist_ok=True)
    for nodes, desc in SUBSTRATES:
        tex = make_table(data, nodes, desc, args.metric)
        out = f"{TABDIR}/sva_grid_{args.metric}_{nodes.replace('+', '-')}.tex"
        open(out, "w").write(tex + "\n")
        print(f"wrote {out}  ({desc}; metric={METRIC_LABEL[args.metric]})")


if __name__ == "__main__":
    main()
