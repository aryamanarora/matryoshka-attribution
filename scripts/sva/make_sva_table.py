"""SVA+ results tables in the style of paper/tabs/mib_results.tex, one section per circuit
granularity (variable set H): node, MLP neurons, MLP neurons + attention heads, SAE latents.

    uv run python scripts/sva/make_sva_table.py            # -> paper/tabs/sva_results.tex
                                                           #    paper/tabs/sva_accauc_results.tex

Two files, mirroring the MIB pair:
  sva_results.tex         CPR  -- `faith_auc`, the log-sparsity-weighted faithfulness AUC that
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
    ]),
]
REFERENCE = ("Random", "Random")
LOSS = "logit_diff"
METRICS = {"cpr": (1, "sva_results.tex", "CPR"),
           "accauc": (0, "sva_accauc_results.tex", "Compactness")}

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


def row_values(raw, m, sub, idx):
    """{task: value} for one (method, substrate) under the figure's cut."""
    return {t: raw[(m, LOSS, sub, t)][idx] for t, _ in COLUMNS if (m, LOSS, sub, t) in raw}


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


def section(raw, sub, title, idx):
    rows = []   # (display, is_ref, {task: v}, avg)
    for _, members in BLOCKS:
        for m, disp in members:
            vals = row_values(raw, m, sub, idx)
            if vals:
                rows.append((m, disp, False, vals, row_avg(vals, sub)))
    rv = row_values(raw, REFERENCE[0], sub, idx)
    ref = (REFERENCE[0], REFERENCE[1], True, rv, row_avg(rv, sub)) if rv else None

    # best / second / colour range per column, competitors only
    comp = [r for r in rows]
    rank, rng = {}, {}
    for t, _ in COLUMNS + [("avg", "")]:
        xs = sorted({(r[4] if t == "avg" else r[3].get(t)) for r in comp} - {None}, reverse=True)
        rank[t] = (xs[0] if xs else None, xs[1] if len(xs) > 1 else None)
        rng[t] = (min(xs), max(xs)) if xs else None

    def render(m, disp, vals, avg, is_ref):
        cells = []
        for t, _ in COLUMNS:
            v = vals.get(t)
            b = (not is_ref) and v is not None and v == rank[t][0]
            u = (not is_ref) and v is not None and not b and v == rank[t][1]
            cells.append(fmt(v, b, u, None if is_ref else cell_color(v, rng[t])))
        b = (not is_ref) and avg is not None and avg == rank["avg"][0]
        u = (not is_ref) and avg is not None and not b and avg == rank["avg"][1]
        cells.append(fmt(avg, b, u, None if is_ref else cell_color(avg, rng["avg"])))
        return f"\\quad {disp} & " + " & ".join(cells) + " \\\\"

    ncols = len(COLUMNS) + 2
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


def build(idx, metric_label):
    ncols = len(COLUMNS)
    lines = ["\\begin{adjustbox}{max width=\\textwidth}",
             "\\begin{tabular}{l" + "c" * ncols + "@{\\quad}c}",
             "\\toprule"]
    heads, rules, c = [], [], 2
    for g, n in GROUP_HEADER:
        heads.append(f"\\multicolumn{{{n}}}{{c}}{{{g}}}" if n > 1 else g)
        rules.append(f"\\cmidrule(lr){{{c}-{c + n - 1}}}")
        c += n
    lines.append("& " + " & ".join(heads) + " & \\\\")
    lines.append(" ".join(rules))
    lines.append("\\textbf{Method} & " + " & ".join(h for _, h in COLUMNS)
                 + f" & \\textbf{{Avg}} \\\\")
    cache = {}
    for sub, res, title in SUBSTRATES:
        raw = cache.setdefault(res, V.load(res))
        lines.append("\\midrule")
        lines += section(raw, sub, f"{title}, {metric_label}", idx)
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}"]
    return "\n".join(lines) + "\n"


def main():
    for key, (idx, fname, label) in METRICS.items():
        tex = build(idx, label)
        (TABS / fname).write_text(tex)
        print(f"-> {TABS / fname}  ({tex.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
