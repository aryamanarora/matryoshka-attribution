"""Colored 'fingerprint' LaTeX tables for experiments 1-3 (method x loss, per granularity).

One table per substrate. Rows = method x train-loss; columns = task(-group) x metric
(acc-AUC, faith-AUC, k* as % of total nodes). Per-column colour scale (per-task): greenest =
best method for that task/metric; column-best bolded. RdYlGn \\cellcolor, MIB header style.
No fixed-k (that's its own STE-interaction figure).

  node:            task-groups SVA (mean of 4) / ARC-E / IOI    (MIB granularity)
  mlp, mlp+attn_head: the 4 SVA tasks separately (per-token; no arc/ioi there)

Run:  uv run python scripts/make_fingerprint_tables.py  ->  paper/tabs/fingerprint_<sub>.tex
"""
import glob
import json
import os
import re

import numpy as np

RES = "results/sva_sweep"
TABDIR = "paper/tabs"
# (metric key, header label, higher_is_better); k* rendered as % of total
METRICS = [("acc_auc", "acc", True), ("faith_auc", "faith", True), ("kstar_pct", r"$k^\star$\%", False)]
# MAttr headline = soft top-k fwd; the STE variants are "+ hard" ablations
SECTIONS = [
    ("Gradient attribution", [("IG", "IG"), ("IxG", "IxG")]),
    # Node Pruning through eval_sva.py's own loss_fn, so it shares MAttr's objective and
    # substrate exactly and differs only in mask parameterization (annealed L0 vs top-k).
    ("Mask learning", [("Node Pruning", "eprun-s090")]),
    (r"MAttr (soft top-$k$ fwd, Adam)", [(r"log-$k$", "stopk-log"), (r"unif-$k$", "stopk-unif")]),
    (r"$+$ hard (sigmoid-STE, Adam)", [(r"log-$k$", "soft-log"), (r"unif-$k$", "soft-unif")]),
    (r"$+$ hard (identity-STE, SGD)", [(r"log-$k$", "idSTE-log"), (r"unif-$k$", "idSTE-unif")]),
]
LOSSES = [("ce", "CE"), ("acc", "acc"), ("logit_diff", "logit-diff")]
SVA = ["nounpp", "rc", "simple", "within_rc"]

# ---------------------------------------------------------------------------
# "Bwd." = backward passes needed to produce one circuit, counted in SEQUENCES
# (sum over steps of batch size for the mask learners; attribution examples x IG
# steps for the gradient methods). Steps alone would not be comparable: a mask
# step here is one sequence, but a gradient "step" is a batch of 32--100.
#
#   mask learners  submit_sva_sweep.sh STEPS=2000 --train-batch-size 1, and
#                  submit_sva_node_pruning.sh STEPS=2000 ("keep it matched")
#                  -> 2000 for every MAttr and Node Pruning row, every table.
#   gradient       eval_sva.py runs all n_examples as ONE batch and loops
#                  alphas = --ig-steps times (default 10; every method except
#                  ig/conductance is forced to 1). n_examples is --eval-examples
#                  100, except the MIB-sourced tasks (arc_easy, ioi), which pass
#                  --grad-examples 32 -- hence the ranges in the node table.
#   +input         --include-input adds input_node_effect(), a second pass of S
#                  backwards over the embedding path, i.e. exactly double.
COST_MASK = "2k"                                        # 2000 steps x batch 1
COSTS_SVA = {"IG": "1k", "IxG": "100"}                  # 100 x 10 / 100 x 1
COSTS_MIXED = {"IG": "0.3--1k", "IxG": "32--100"}       # 32 examples on arc_easy/ioi
COSTS_INPUT = {"IG": "0.6--2k", "IxG": "64--200"}       # the above, doubled

SUBSTRATES = [("node", [("SVA", set(SVA)), ("ARC-E", {"arc_easy"}), ("IOI", {"ioi"})], COSTS_MIXED),
              ("mlp", [(t, {t}) for t in SVA], COSTS_SVA),
              ("mlp+attn_head", [(t, {t}) for t in SVA], COSTS_SVA)]


def parse_method(fname, d):
    tag = fname.split("_" + d["nodes"].replace("+", "-") + "_", 1)[1].rsplit(".json", 1)[0]
    if tag.startswith(("random", "conductance")) or "fixedk" in tag:
        return None
    if "hard_topk" in tag:
        if re.search(r"_ig\d+", tag):
            return None
        fam = "idSTE" if "identity" in tag else "soft"
        ks = "unif" if "uniformk" in tag else "log"
        return f"{fam}-{ks}"
    if "sufficient_topk_" in tag:   # soft top-k forward (differentiable, no STE)
        if re.search(r"_ig\d+", tag):
            return None
        ks = "unif" if "uniformk" in tag else "log"
        return f"stopk-{ks}"
    # eprun_s090[_ce|_acc] -> one key per budget. This branch must stay ABOVE the catch-all:
    # the tag matches none of the tests above, so without it every Node Pruning run is
    # silently averaged into the IG rows.
    if tag.startswith("eprun_s"):
        return "eprun-s" + tag.split("_")[1][1:]
    # sig_lr0.3_l16.0[_ce|_acc] -> the pyvene sigmoid-mask baseline (DBM in the paper). One key
    # per recipe, since lr and the L1 weight are what decide the circuit. Same placement rule as
    # eprun: ABOVE the catch-all, or every DBM run is silently averaged into the IG rows.
    if tag.startswith("sig_"):
        return re.sub(r"_(ce|acc)$", "", tag)
    return "IxG" if tag.startswith("ixg") else "IG"


def load(nodes, res=RES):
    raw = {}   # (method, loss, task) -> {metric: value}
    for f in glob.glob(res + "/*.json"):
        d = json.load(open(f))
        if d["nodes"] != nodes:
            continue
        m = parse_method(os.path.basename(f), d)
        if m is None:
            continue
        ks = d.get("kstar_50")
        raw[(m, d["loss"], d["task"])] = {
            "acc_auc": d["acc_auc"], "faith_auc": d["faith_auc"],
            "kstar_pct": 100.0 * (ks if ks is not None else d["total"]) / d["total"]}
    return raw


def hexcol(g):
    red, yel, grn = (0xF6, 0xA5, 0x82), (0xFF, 0xFF, 0xCC), (0xA6, 0xD9, 0x6A)
    a, b, t = (red, yel, g / 0.5) if g < 0.5 else (yel, grn, (g - 0.5) / 0.5)
    return "%02X%02X%02X" % tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def make(nodes, groups, costs=COSTS_SVA, res=RES, suffix=""):
    raw = load(nodes, res)
    rows = [(mk, lk) for _, ms in SECTIONS for _, mk in ms for lk, _ in LOSSES]
    ncol = 3 * len(groups)

    def gval(mk, lk, tasks, metric):
        vs = [raw[(mk, lk, t)][metric] for t in tasks if (mk, lk, t) in raw]
        return float(np.mean(vs)) if vs else None

    good, best, txt = {}, {}, {}
    for gi, (gl, tasks) in enumerate(groups):
        for metric, _, hib in METRICS:
            vals = [gval(mk, lk, tasks, metric) for mk, lk in rows]
            xs = [v for v in vals if v is not None]
            lo, hi = (min(xs), max(xs)) if xs else (0, 1)
            bg = -1
            for (mk, lk), v in zip(rows, vals):
                k = (mk, lk, gi, metric)
                if v is None:
                    good[k], txt[k] = None, "---"; continue
                g = (v - lo) / (hi - lo + 1e-9)
                good[k] = (1 - g) if not hib else g
                txt[k] = ("%.0f" % v if metric == "kstar_pct" and v >= 99.5 else "%.2f" % v)
                bg = max(bg, good[k])
            for mk, lk in rows:
                k = (mk, lk, gi, metric)
                best[k] = good.get(k) is not None and abs(good[k] - bg) < 1e-9

    def cell(mk, lk, gi, metric):
        k = (mk, lk, gi, metric); g = good.get(k); s = txt[k]
        if best.get(k):
            s = r"\textbf{%s}" % s
        return (r"\cellcolor[HTML]{%s}%s" % (hexcol(g), s)) if g is not None else s

    hdr_groups, cmids, c = [], [], 4   # 4: Method, Loss, Bwd. come first
    for gl, _ in groups:
        hdr_groups.append(r"\multicolumn{3}{c}{%s}" % gl.replace("_", r"\_"))
        cmids.append(r"\cmidrule(lr){%d-%d}" % (c, c + 2)); c += 3
    L = [r"\begin{adjustbox}{max width=\textwidth}",
         r"\begin{tabular}{llr *{%d}{c}}" % ncol, r"\toprule",
         "& & & " + " & ".join(hdr_groups) + r" \\", " ".join(cmids),
         r"\textbf{Method} & \textbf{Loss} & \textbf{Bwd.} & "
         + " & ".join(ml for _ in groups for _, ml, _ in METRICS) + r" \\", r"\midrule"]
    for si, (sec, methods) in enumerate(SECTIONS):
        if si:
            L.append(r"\midrule")
        L.append(r"\multicolumn{%d}{l}{\textit{%s}} \\" % (ncol + 3, sec))
        for mlabel, mk in methods:
            for li, (lk, ll) in enumerate(LOSSES):
                cells = [cell(mk, lk, gi, mt) for gi in range(len(groups)) for mt, _, _ in METRICS]
                # cost is a property of the method, not the loss: print it once per block,
                # on the same row as the method name.
                name, cost = (mlabel, costs.get(mk, COST_MASK)) if li == 0 else ("", "")
                L.append(f"{name} & {ll} & {cost} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}"]
    out = f"{TABDIR}/fingerprint_{nodes.replace('+', '-')}{suffix}.tex"
    os.makedirs(TABDIR, exist_ok=True)
    open(out, "w").write("\n".join(L) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    for nodes, groups, costs in SUBSTRATES:
        make(nodes, groups, costs)
    # +input node runs (input-embedding node scored+ablated; learnable-input, harder mode)
    node_groups = SUBSTRATES[0][1]
    make("node", node_groups, COSTS_INPUT, res="results/sva_sweep_input", suffix="_input")
