"""Paired Wilcoxon signed-rank tests on MIB TEST CPR AUC: the headline \\ourmethod{} row against
every other row of tabs/mib_test_results.tex, at node and at edge level.

    uv run python scripts/mib/test_paired_tests.py [--tex paper/tabs/paired_tests.tex]

THE CLAIM UNDER TEST: "On both node- and edge-level circuit localisation, \\ourmethod{}
statistically significantly outperforms all baselines (p < 0.05)." The rows are exactly the
test table's (make_mib_test_table.collect() + collect_extra() plus its NODE_BASELINES /
EDGE_BASELINES literals transcribed from the MIB paper), so a row that is in the table is in the
test and vice versa.
Pairs are the (task, model) cells both rows have (12 for our runs; 11 for the transcribed
rows, which have no arithmetic_addition cell). Two-sided Wilcoxon on the per-cell differences,
with the one-sided (MAttr greater) p beside it, and Holm-Bonferroni over the family "headline
vs each baseline" within a level -- the question of interest, not all C(k,2) pairs.

n is small (11-12), so the smallest two-sided p a Wilcoxon can produce is 2 / 2^n ~ 0.001:
a row that is beaten on every cell reaches that floor, one that wins a few cells does not.
Ties (identical rounded values) are dropped, as scipy's default does.

Our own ablation rows (other \\ourmethod{} variants) are listed separately: they are not
baselines, but the numbers are useful and cost nothing.

--tex writes the same numbers as a tabular fragment (paper/tabs/paired_tests.tex): one block per
level, baselines in table order, then the variants. Rows with fewer than MIN_N cells (UGS, 3
cells: the smallest two-sided p is 0.25) show n / diff / wins but no p, and are not in the Holm
family; the caption in detailed-mib.tex says so.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_mib_test_table as T          # noqa: E402
import mattr_variants as MV              # noqa: E402


def paired(ref, other):
    cells = sorted(c for c in ref if c in other and ref[c] is not None and other[c] is not None)
    x = np.array([ref[c] for c in cells], float)
    y = np.array([other[c] for c in cells], float)
    d = x - y
    nz = d[d != 0]
    if len(nz) < 1:
        return len(cells), float(d.mean()), int((d > 0).sum()), float("nan"), float("nan")
    p2 = wilcoxon(x, y, alternative="two-sided").pvalue
    p1 = wilcoxon(x, y, alternative="greater").pvalue
    return len(cells), float(d.mean()), int((d > 0).sum()), p2, p1


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (monotone), NaNs passed through."""
    idx = [i for i, p in enumerate(pvals) if not np.isnan(p)]
    m = len(idx)
    order = sorted(idx, key=lambda i: pvals[i])
    adj = [float("nan")] * len(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        v = min(1.0, (m - rank) * pvals[i])
        running = max(running, v)
        adj[i] = running
    return adj


MIN_N = 6   # below this a two-sided Wilcoxon cannot reach 0.05 (floor 2 / 2^n)


def stats_for(ref, rows):
    """[(name, n, mean diff, wins, p2, p1, holm)] with the Holm family = rows with n >= MIN_N."""
    out = [(name,) + paired(ref, data) for name, data in rows]
    fam = [s[4] if s[1] >= MIN_N else float("nan") for s in out]
    adj = holm(fam)
    return [s + (a,) for s, a in zip(out, adj)]


def report(level, ref_name, ref, rows, ablations):
    print(f"\n=== {level} level: {ref_name} vs each row, MIB test CPR AUC ===")
    print(f"{'row':<30}{'n':>4}{'mean diff':>11}{'wins':>7}{'p two-sided':>13}{'p greater':>11}{'Holm (2s)':>11}")
    st = stats_for(ref, rows)
    names, adj = [s[0] for s in st], [s[6] for s in st]
    for name, n, md, wins, p2, p1, pa in st:
        flag = "  (n < MIN_N; not tested)" if n < MIN_N else ("  *" if pa < 0.05 else "  (n.s.)")
        print(f"{name:<30}{n:>4}{md:>+11.3f}{wins:>4}/{n:<2}{p2:>13.4f}{p1:>11.4f}{pa:>11.4f}{flag}")
    if ablations:
        print(f"--- our other variants (not baselines; not in the Holm family) ---")
        for name, data in ablations:
            n, md, wins, p2, p1 = paired(ref, data)
            print(f"{name:<30}{n:>4}{md:>+11.3f}{wins:>4}/{n:<2}{p2:>13.4f}{p1:>11.4f}")
    sig = [nm for nm, pa in zip(names, adj) if not np.isnan(pa) and pa < 0.05]
    tested = [(pa, nm) for nm, pa in zip(names, adj) if not np.isnan(pa)]
    worst = max(tested)
    print(f"significant after Holm at 0.05: {len(sig)}/{len(tested)} tested baselines "
          f"({len(names) - len(tested)} untested, n < {MIN_N}); "
          f"largest adjusted p = {worst[0]:.4f} ({worst[1]})")
    return st


def _p(p):
    if np.isnan(p):
        return "---"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def write_tex(path, blocks):
    """blocks = [(level title, [(name, n, md, wins, p2, p1, holm)], [variant rows])].

    TWO LEVELS SIDE BY SIDE (2026-09-21, requested): the node block on the left, the edge block
    on the right, zipped row by row so the table is as tall as the longer block rather than the
    sum of both. Each side repeats the column header; a short side is padded with empty cells.
    """
    hdr = "\\textbf{Baseline} & $n$ & $\\Delta$ CPR & wins & $p$ & $p_{\\mathrm{Holm}}$"
    sides = []
    for title, rows, variants in blocks:
        side = [f"\\multicolumn{{6}}{{l}}{{\\textit{{{title}}}}}"]
        for name, n, md, wins, p2, p1, pa in rows:
            small = n < MIN_N
            side.append(f"\\quad {name} & {n} & {md:+.2f} & {wins}/{n} & "
                        f"{'---' if small else _p(p2)} & {'---' if small else _p(pa)}")
        if variants:
            side.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{title}, other \\ourmethod{{}} variants "
                        f"(not in the Holm family)}}}}")
            for name, n, md, wins, p2, p1 in variants:
                side.append(f"\\quad {name} & {n} & {md:+.2f} & {wins}/{n} & {_p(p2)} & ---")
        sides.append(side)
    empty = " & & & & & "
    width = max(len(sd) for sd in sides)
    lines = ["\\begin{tabular}{lrrrrr@{\\qquad}lrrrrr}", "\\toprule",
             hdr + " & " + hdr + " \\\\", "\\midrule"]
    for i in range(width):
        cells = [sd[i] if i < len(sd) else empty for sd in sides]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=None, help="also write a tabular fragment here")
    a = ap.parse_args()
    ours_nodes, mask_nodes, grad_nodes, ours_edges = T.collect()
    causal_nodes, grad_edges, mask_edges = T.collect_extra()
    head = MV.OURMETHOD
    # node level
    ref = ours_nodes[head]
    node_rows = ([(k, v) for k, v in T.NODE_BASELINES.items()]
                 + [(k, v) for k, v in grad_nodes.items()]
                 + [(k, v) for k, v in mask_nodes.items()]
                 + [(k, v) for k, v in causal_nodes.items()])
    node_abl = [(k, v) for k, v in ours_nodes.items() if k != head]
    node_st = report("node", head, ref, node_rows, node_abl)
    # edge level
    ref_e = ours_edges[head]
    edge_rows = ([(k, v) for k, v in T.EDGE_BASELINES.items()]
                 + [(k, v) for k, v in grad_edges.items()]
                 + [(k, v) for k, v in mask_edges.items()])
    edge_abl = [(k, v) for k, v in ours_edges.items() if k != head]
    edge_st = report("edge", head, ref_e, edge_rows, edge_abl)
    if a.tex:
        write_tex(a.tex, [
            ("Node-level", node_st, [(k,) + paired(ref, v) for k, v in node_abl]),
            ("Edge-level", edge_st, [(k,) + paired(ref_e, v) for k, v in edge_abl]),
        ])


if __name__ == "__main__":
    main()
