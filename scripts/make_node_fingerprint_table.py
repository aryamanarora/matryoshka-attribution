"""Colored 'fingerprint' LaTeX table for node granularity (experiments 1-4).

Rows = method x train-loss; columns = task-group x metric (SVA / ARC-E / IOI) x (acc-AUC,
faith-AUC, k*). Each column has its OWN colour scale (per-task normalization): greenest =
best method for that task/metric; column-best is also bolded. RdYlGn \\cellcolor, MIB header
style (adjustbox + \\multicolumn groups + \\cmidrule + \\textit section rows).

Run:  uv run python scripts/make_node_fingerprint_table.py  ->  paper/tabs/node_fingerprint.tex
"""
import glob
import json
import os
import re

import numpy as np

RES = "results/sva_sweep"
OUT = "paper/tabs/node_fingerprint.tex"
GRPS = ["SVA", "ARC-E", "IOI"]
METRICS = [("acc_auc", "acc", True), ("faith_auc", "faith", True), ("kstar_50", r"$k^\star$", False)]
SECTIONS = [
    ("Gradient attribution", [("IG", "IG"), ("IxG", "IxG")]),
    (r"MAttr (sigmoid-STE, Adam)", [(r"log-$k$", "soft-log"), (r"unif-$k$", "soft-unif"), (r"fixed-$k$", "soft-fixed")]),
    (r"MAttr (identity-STE, SGD)", [(r"log-$k$", "idSTE-log"), (r"unif-$k$", "idSTE-unif"), (r"fixed-$k$", "idSTE-fixed")]),
]
LOSSES = [("ce", "CE"), ("acc", "acc"), ("logit_diff", "logit-diff")]
SVA = {"nounpp", "rc", "simple", "within_rc"}


def parse_method(fname, d):
    tag = fname.split("_" + d["nodes"].replace("+", "-") + "_", 1)[1].rsplit(".json", 1)[0]
    if tag.startswith(("random", "conductance")):
        return None
    if "hard_topk" in tag:
        if re.search(r"_ig\d+", tag):
            return None
        fam = "idSTE" if "identity" in tag else "soft"
        ks = "fixed" if "fixedk" in tag else ("unif" if "uniformk" in tag else "log")
        return f"{fam}-{ks}"
    return "IxG" if tag.startswith("ixg") else "IG"


def load():
    raw = {}   # (method, loss, grp) -> {metric: [values over tasks]}
    for f in glob.glob(RES + "/*.json"):
        d = json.load(open(f))
        if d["nodes"] != "node":
            continue
        m = parse_method(os.path.basename(f), d)
        if m is None:
            continue
        grp = "SVA" if d["task"] in SVA else ("ARC-E" if d["task"] == "arc_easy" else "IOI")
        key = (m, d["loss"], grp)
        for mk, _, _ in METRICS:
            raw.setdefault(key, {}).setdefault(mk, []).append(d.get(mk))
    val = {}   # (method, loss, grp, metric) -> mean value (None-safe)
    for key, md in raw.items():
        for mk, vs in md.items():
            fs = [v for v in vs if v is not None]
            val[(*key, mk)] = (float(np.mean(fs)) if fs else None)
    return val


def hexcol(g):
    red, yel, grn = (0xF6, 0xA5, 0x82), (0xFF, 0xFF, 0xCC), (0xA6, 0xD9, 0x6A)
    a, b, t = (red, yel, g / 0.5) if g < 0.5 else (yel, grn, (g - 0.5) / 0.5)
    return "%02X%02X%02X" % tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def main():
    val = load()
    rows = [(mk, lk) for _, ms in SECTIONS for _, mk in ms for lk, _ in LOSSES]

    # per-column (grp, metric) goodness + best, over all method/loss rows
    good, best, txt = {}, {}, {}
    for grp in GRPS:
        for metric, _, hib in METRICS:
            vs = [val.get((mk, lk, grp, metric)) for mk, lk in rows]
            xs = [(np.log10(max(v, 1)) if metric.startswith("kstar") else v)
                  for v in vs if v is not None]
            lo, hi = (min(xs), max(xs)) if xs else (0, 1)
            bestg = -1
            for (mk, lk), v in zip(rows, vs):
                if v is None:
                    good[(mk, lk, grp, metric)] = None
                    txt[(mk, lk, grp, metric)] = "---"
                    continue
                x = np.log10(max(v, 1)) if metric.startswith("kstar") else v
                g = (x - lo) / (hi - lo + 1e-9)
                if not hib:
                    g = 1 - g
                good[(mk, lk, grp, metric)] = g
                txt[(mk, lk, grp, metric)] = ("%.0f" % v if metric.startswith("kstar") else "%.2f" % v)
                bestg = max(bestg, g)
            for mk, lk in rows:
                g = good.get((mk, lk, grp, metric))
                best[(mk, lk, grp, metric)] = g is not None and abs(g - bestg) < 1e-9

    def cell(mk, lk, grp, metric):
        g = good.get((mk, lk, grp, metric)); s = txt[(mk, lk, grp, metric)]
        if best.get((mk, lk, grp, metric)):
            s = r"\textbf{%s}" % s
        return (r"\cellcolor[HTML]{%s}%s" % (hexcol(g), s)) if g is not None else s

    L = [r"\begin{adjustbox}{max width=\textwidth}",
         r"\begin{tabular}{ll *{9}{c}}", r"\toprule",
         "& & " + " & ".join(r"\multicolumn{3}{c}{%s}" % g for g in GRPS) + r" \\",
         r"\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11}",
         r"\textbf{Method} & \textbf{Loss} & "
         + " & ".join(ml for _ in GRPS for _, ml, _ in METRICS) + r" \\", r"\midrule"]
    for si, (sec, methods) in enumerate(SECTIONS):
        if si:
            L.append(r"\midrule")
        L.append(r"\multicolumn{11}{l}{\textit{%s}} \\" % sec)
        for mlabel, mk in methods:
            for li, (lk, ll) in enumerate(LOSSES):
                cells = [cell(mk, lk, g, mt) for g in GRPS for mt, _, _ in METRICS]
                name = mlabel if li == 0 else ""
                L.append(f"{name} & {ll} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(L) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
