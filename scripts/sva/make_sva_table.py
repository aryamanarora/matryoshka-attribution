"""SVA+ results tables in the style of paper/tabs/mib_results.tex, one section per circuit
granularity (variable set H): node, MLP neurons, MLP neurons + attention heads, SAE latents.

    uv run python scripts/sva/make_sva_table.py            # -> paper/tabs/sva_results.tex
                                                           #    paper/tabs/sva_accauc_results.tex

Two files, mirroring the MIB pair:
  sva_results.tex         CPR  -- the LINEAR-p faithfulness AUC (V._cpr_of; MIB's measure), NOT faith_auc, the log-weighted one that
                          figs/accauc_vs_faithauc.pdf plots on y ("CPR" on the paper's axes)
  sva_accauc_results.tex  Compactness -- `acc_auc`, the IIA AUC the same figure plots on x

EVERYTHING IS READ THROUGH plots/plot_accauc_vs_faithauc.py -- its `load` (which runs count,
how Random's seeds average), `parse_method` (filename tag -> method key), `on_model` (the
task -> model pin: ioi is qwen2.5, everything else llama3), `SUBSTRATE_RES` (the neuron
substrates read the 5k-budget trees, SAE the frozen-error tree) and `REQUIRED`/`GROUPS`
(which task groups a substrate is scored on). So a cell here is the number behind the
corresponding point in the figure, and the Avg column is EXACTLY the figure's plotted
coordinate: the macro-average over task GROUPS (SVA, Arith, ARC-E, IOI), not over tasks, so
the four-task groups do not outvote the single-task MIB cells. The cut is the figure's default
one: logit-diff loss, patched (counterfactual) ablation, no input-embedding variables.

Rows follow mib_results.tex's blocks -- gradient attribution, mask learning, ours + ablations
-- plus Random as a reference row. Random is excluded from the bold/underline and from the
colour range (it is a floor, not a competitor). A row that has no runs on a substrate is
omitted from that section rather than drawn as dashes (AttnLRP and Expected Gradients were not
run on SAE latents). Bold = best in column within the section, underline = second; the blue
ramp is per column within the section, same ramp as the MIB tables.
"""

import sys
from pathlib import Path

import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "plots"))
sys.path.insert(0, str(ROOT / "scripts" / "mib"))
import plot_accauc_vs_faithauc as V  # noqa: E402
import mattr_variants as MV  # the MIB tables' headline definition + label grammar  # noqa: E402

TABS = ROOT / "paper" / "tabs"

# (task, column header). Groups follow V.GROUPS so the header cmidrules match the figure's
# task groups; the model per column is V.TASK_MODEL (Llama-3 everywhere except IOI = Qwen).
COLUMNS = [
    ("nounpp", "NounPP"), ("rc", "RC"), ("simple", "Simple"), ("within_rc", "Within-RC"),
    ("addition", "Add."), ("months", "Months"), ("weekdays", "Weekd."), ("hours", "Hours"),
    ("arc_easy", "Llama"), ("ioi", "Qwen"),
]
GROUP_HEADER = [("SVA", 4), ("Arithmetic", 4), ("ARC (E)", 1), ("IOI", 1)]

SUBSTRATES = [
    ("node", "results/sva_sweep", "Node-level (attention heads + MLPs)"),
    ("mlp", V.SUBSTRATE_RES["mlp"], "MLP neurons"),
    ("mlp+attn_head", V.SUBSTRATE_RES["mlp+attn_head"], "MLP neurons + attention heads"),
    ("mlp_sae_span", V.SUBSTRATE_RES["mlp_sae_span"], "SAE latents (MLP out)"),
]
# --zero: the same four substrates under ZERO ablation, all from results/sva_zeroabl_5k (5000 steps
# for every trained method and the patched trees' attribution budgets for the gradient rows, all
# launched by submit_zero_5k_sc.sh, 2026-09-18; the older results/sva_zeroabl is 2k and unused here).
# The 10x-steps rows do not exist under zero, so build() passes no 10x tree in this mode.
ZERO_RES = "results/sva_zeroabl_5k"
SUBSTRATES_ZERO = [(sub, ZERO_RES, title) for sub, _, title in SUBSTRATES]

# (block title, [(method key in V.METHODS, display name)]). MAttr labels come from
# scripts/mib/mattr_variants.py, the same definition the MIB tables use: the headline is
# uniform k + Adam, and on these substrates (where Adam's epsilon is swept) eps = 1e-2; every
# other arm is marked relative to it. Headline first, then one-attribute ablations.
BLOCKS = [
    ("Gradient attribution", [
        ("IG", "IG"),
        ("IxG", "I$\\times$G"),
        ("mc_ig", "Expected Gradients"),
        ("AttnLRP", "AttnLRP"),
    ]),
    ("Mask learning", [
        ("eprun-s090", "Node Pruning"),
        ("sig_lr0.3_l16.0", "DBM"),
    ]),
    ("\\ourmethod{}", [
        ("stopk-unif-eps1e-2", MV.label(eps="1e-2")),                 # headline: unif k, Adam, eps 1e-2
        ("stopk-log-eps1e-2",  MV.label(k="log", eps="1e-2")),        # $+$ log $k$
        ("stopk-log",          MV.label(k="log", eps="1e-8")),        # $+$ log $k$, eps=1e-8 (Adam default)
        ("softsgd-log",        MV.label(k="log", opt="sgd")),         # $+$ log $k$, $+$ SGD
        ("soft-log",           MV.label(k="log", fwd="hard", eps="1e-8")),   # hard STE fwd, Adam default eps
        # The headline trained 10x longer (50k steps; submit_unifk_eps_50k_sc.sh, 2026-09-17) at
        # both Adam epsilons. Loaded from V.TENX_RES's own trees and re-keyed by V.TENX_KEYS --
        # see build(). The eps=1e-8 twin has no 5k counterpart on these substrates (the default-eps
        # uniform arm was only ever run at 2k on MLP / MLP+Attn and never on SAE), so its row is
        # read against the eps=1e-2 headline, one row above the 10x pair.
        ("stopk-unif-eps1e-2-10x", MV.label(eps="1e-2", extra=("$+$ $10\\times$ steps",))),
        ("stopk-unif-10x",         MV.label(eps="1e-8", extra=("$+$ $10\\times$ steps",))),
    ]),
]
REFERENCE = ("Random", "Random")
LOSS = "logit_diff"
# CPR IS THE LINEAR AUC, ALWAYS (user decision 2026-09-17): index 2 of V.load's tuple, the
# MIB-style linear trapezoid over the kept proportion (V._cpr_of), not the log-weighted
# faith_auc (index 1) this column carried until then. Same integrand as the curves figure.
METRICS = {"cpr": (2, "sva_results.tex", "CPR"),
           "accauc": (0, "sva_accauc_results.tex", "Compactness")}
METRICS_ZERO = {"cpr": (2, "sva_zero_results.tex", "CPR"),
                "accauc": (0, "sva_zero_accauc_results.tex", "Compactness (chance-corrected)")}
# Compactness under ZERO ablation is CHANCE-CORRECTED: the per-example indicator is `lb > ls`,
# and a zeroed model (logit diff ~ 0) satisfies it with probability 1/2, so the IIA log-AUC has
# a floor of 0.5 (Random reads 0.47-0.52 across the four substrates) where the interchange
# tables' floor is 0. The table prints (AUC - 1/2) / (1 - 1/2) = 2 AUC - 1: the SAME affine map
# for every method in a cell, so orderings are untouched and Random lands at ~0, on the scale of
# the interchange tables. The floor is the ANALYTIC one, not a per-run or per-cell measurement:
# a destroyed model's preference is a coherent coin flip per run (plot_accauc_vs_faithauc.load's
# note on the bimodal acc[0] under zeroing), so an empirical floor would divide each row by its
# own noise draw. CPR is unaffected (its floor is ~0 either way).
ZERO_CHANCE = {0: 0.5}   # metric index -> chance level under zero ablation

# === Bwd. column ============================================================================
# Backward passes through the model, in sequences, to fit ONE cell -- the unit and the reading
# of make_mib_table's column (compute-proportional, order of magnitude, not wall clock). Read
# off each run's saved `config`: trained methods spend steps x train batch, gradient methods
# spend attribution examples x path points (I x G, Expected Gradients and AttnLRP are one
# backward per example; IG is grad_examples x ig_steps). Random draws a ranking and costs 0.
# Older files have no `config` block (the 2k node-level NP / DBM runs, whose defaults are
# 2000 steps x batch 1); NO_CONFIG_COST covers those by method key so the cell is not a dash.
# A (method, substrate) whose runs differ across tasks prints the range, like MIB's "0.1--1k".
NO_CONFIG_COST = {"eprun-s090": 2000, "sig_lr0.3_l16.0": 2000}
ONE_PASS = {"ixg", "mc_ig", "attnlrp"}


def run_cost(d, m):
    """Backward passes of one run, from its config; None if it cannot be determined."""
    c = d.get("config") or {}
    if m == "Random":
        return 0
    meth = c.get("method")
    if meth is None:
        return NO_CONFIG_COST.get(m)
    if meth in ("mattr", "edge_pruning", "sigmoid_mask"):
        return c["steps"] * (c.get("train_batch_size") or 1)
    if meth in ONE_PASS:
        return c.get("grad_examples")
    if meth == "ig":
        ge = c.get("grad_examples")
        return None if ge is None else ge * c["ig_steps"]
    return None


def fmt_cost(lo, hi):
    k = lambda v: "0" if v == 0 else f"{v / 1000:g}k"
    return k(lo) if lo == hi else f"{k(lo)}--{k(hi)}".replace("k--", "--")


def costs(res, sub, tenx=None):
    """{method key: Bwd. string} over every file of the tree(s) that renders in this section."""
    import glob, os
    seen = {}
    for tree, keymap in ((res, None), (tenx, V.TENX_KEYS)):
        if tree is None:
            continue
        for f in glob.glob(tree + "/*.json"):
            d = json.load(open(f))
            if d.get("nodes") != sub or d.get("loss") != LOSS or not V.on_model(d):
                continue
            m = V.parse_method(os.path.basename(f), d)
            if keymap is not None:
                m = keymap.get(m)
            if m is None:
                continue
            v = run_cost(d, m)
            if v is not None:
                seen.setdefault(m, []).append(v)
    return {m: fmt_cost(min(v), max(v)) for m, v in seen.items()}

# Cell styling mirrored from scripts/mib/make_mib_table.py (cell_color / fmt) so the two table
# families read identically; copied rather than imported because that module's import-time
# work is the MIB results tree.
CELL_HI = (0x92, 0xC5, 0xDE)


def cell_color(v, rng):
    if v is None or rng is None:
        return None
    lo, hi = rng
    t = 0.0 if hi <= lo else (v - lo) / (hi - lo)
    return "%02X%02X%02X" % tuple(round(255 + t * (c - 255)) for c in CELL_HI)


def fmt(v, bold=False, underline=False, color=None):
    if v is None:
        return "---"
    s = f"{v:.2f}"
    if bold:
        s = f"\\mathbf{{{s}}}"
    elif underline:
        s = f"\\underline{{{s}}}"
    s = f"${s}$"
    return f"\\cellcolor[HTML]{{{color}}}{s}" if color else s


def row_values(raw, m, sub, idx, chance=None):
    """{task: value} for one (method, substrate) under the figure's cut; `chance` (a floor in
    [0, 1)) rescales the value to (v - chance) / (1 - chance), see ZERO_CHANCE."""
    vals = {t: raw[(m, LOSS, sub, t)][idx] for t, _ in COLUMNS if (m, LOSS, sub, t) in raw}
    if chance:
        vals = {t: (v - chance) / (1 - chance) for t, v in vals.items()}
    return vals


def row_avg(vals, sub):
    """The figure's coordinate: macro-average over the substrate's REQUIRED task groups, None
    unless every group is complete (same all-or-nothing rule as V.group_avg)."""
    gs = []
    for gname, tasks in V.GROUPS:
        if gname not in V.REQUIRED[sub]:
            continue
        if not set(tasks) <= set(vals):
            return None
        gs.append(np.mean([vals[t] for t in tasks]))
    return float(np.mean(gs)) if gs else None


def section(raw, sub, title, idx, cost, chance=None):
    rows = []   # (display, is_ref, {task: v}, avg)
    for _, members in BLOCKS:
        for m, disp in members:
            vals = row_values(raw, m, sub, idx, chance)
            if vals:
                rows.append((m, disp, False, vals, row_avg(vals, sub)))
    rv = row_values(raw, REFERENCE[0], sub, idx, chance)
    ref = (REFERENCE[0], REFERENCE[1], True, rv, row_avg(rv, sub)) if rv else None

    # best / second / colour range per column, competitors only
    comp = [r for r in rows]
    rank, rng = {}, {}
    for t, _ in COLUMNS + [("avg", "")]:
        xs = sorted({(r[4] if t == "avg" else r[3].get(t)) for r in comp} - {None}, reverse=True)
        rank[t] = (xs[0] if xs else None, xs[1] if len(xs) > 1 else None)
        rng[t] = (min(xs), max(xs)) if xs else None

    def render(m, disp, vals, avg, is_ref):
        cells = [cost.get(m, "---")]
        for t, _ in COLUMNS:
            v = vals.get(t)
            b = (not is_ref) and v is not None and v == rank[t][0]
            u = (not is_ref) and v is not None and not b and v == rank[t][1]
            cells.append(fmt(v, b, u, None if is_ref else cell_color(v, rng[t])))
        b = (not is_ref) and avg is not None and avg == rank["avg"][0]
        u = (not is_ref) and avg is not None and not b and avg == rank["avg"][1]
        cells.append(fmt(avg, b, u, None if is_ref else cell_color(avg, rng["avg"])))
        return f"\\quad {disp} & " + " & ".join(cells) + " \\\\"

    ncols = len(COLUMNS) + 3
    out = [f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{{title}}}}} \\\\"]
    by_key = {r[0]: r for r in rows}
    for btitle, members in BLOCKS:
        present = [by_key[m] for m, _ in members if m in by_key]
        if not present:
            continue
        out.append(f"\\textbf{{{btitle}}} \\\\")
        out += [render(m, d, v, a, False) for m, d, _, v, a in present]
    if ref:
        out.append("\\textbf{Reference} \\\\")
        out.append(render(*[ref[0], ref[1], ref[3], ref[4], True]))
    return out


def build(idx, metric_label, zero=False):
    ncols = len(COLUMNS)
    lines = ["\\begin{adjustbox}{max width=\\textwidth}",
             "\\begin{tabular}{lr" + "c" * ncols + "@{\\quad}c}",
             "\\toprule"]
    heads, rules, c = [], [], 3   # column 2 is Bwd.
    for g, n in GROUP_HEADER:
        heads.append(f"\\multicolumn{{{n}}}{{c}}{{{g}}}" if n > 1 else g)
        rules.append(f"\\cmidrule(lr){{{c}-{c + n - 1}}}")
        c += n
    lines.append("& & " + " & ".join(heads) + " & \\\\")
    lines.append(" ".join(rules))
    lines.append("\\textbf{Method} & \\textbf{Bwd.} & " + " & ".join(h for _, h in COLUMNS)
                 + f" & \\textbf{{Avg}} \\\\")
    cache = {}
    for sub, res, title in (SUBSTRATES_ZERO if zero else SUBSTRATES):
        raw = dict(cache.setdefault(res, V.load(res)))
        # The 10x-steps rows live in their own trees (one budget per tree, V.SUBSTRATE_RES's
        # rule); merge them in under their synthetic keys, the same re-keying V's tenx_for does.
        tenx = None if zero else V.TENX_RES.get(sub)
        if tenx:
            raw.update({(V.TENX_KEYS[m], l, ss, t): v
                        for (m, l, ss, t), v in cache.setdefault(tenx, V.load(tenx)).items()
                        if m in V.TENX_KEYS})
        lines.append("\\midrule")
        lines += section(raw, sub, f"{title}, {metric_label}", idx, costs(res, sub, tenx),
                         chance=ZERO_CHANCE.get(idx) if zero else None)
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}"]
    return "\n".join(lines) + "\n"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--zero", action="store_true",
                    help="the zero-ablation pair (sva_zero_results / sva_zero_accauc_results)")
    a = ap.parse_args()
    for key, (idx, fname, label) in (METRICS_ZERO if a.zero else METRICS).items():
        tex = build(idx, label, zero=a.zero)
        (TABS / fname).write_text(tex)
        print(f"-> {TABS / fname}  ({tex.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
