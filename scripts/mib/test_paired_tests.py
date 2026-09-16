"""Paired Wilcoxon signed-rank tests on MIB TEST CPR AUC: the headline \\ourmethod{} row against
every other row of tabs/mib_test_results.tex, at node and at edge level.

    uv run python scripts/mib/test_paired_tests.py [--metric cpr]

THE CLAIM UNDER TEST: "On both node- and edge-level circuit localisation, \\ourmethod{}
statistically significantly outperforms all baselines (p < 0.05)." The rows are exactly the
test table's (make_mib_test_table.collect() plus its NODE_BASELINES / EDGE_BASELINES literals
transcribed from the MIB paper), so a row that is in the table is in the test and vice versa.
Pairs are the (task, model) cells both rows have (12 for our runs; 11 for the transcribed
rows, which have no arithmetic_addition cell). Two-sided Wilcoxon on the per-cell differences,
with the one-sided (MAttr greater) p beside it, and Holm-Bonferroni over the family "headline
vs each baseline" within a level -- the question of interest, not all C(k,2) pairs.

n is small (11-12), so the smallest two-sided p a Wilcoxon can produce is 2 / 2^n ~ 0.001:
a row that is beaten on every cell reaches that floor, one that wins a few cells does not.
Ties (identical rounded values) are dropped, as scipy's default does.

Our own ablation rows (other \\ourmethod{} variants) are listed separately: they are not
baselines, but the numbers are useful and cost nothing.
"""
import argparse
import os
import sys

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


def report(level, ref_name, ref, rows, ablations):
    print(f"\n=== {level} level: {ref_name} vs each row, MIB test CPR AUC ===")
    print(f"{'row':<30}{'n':>4}{'mean diff':>11}{'wins':>7}{'p two-sided':>13}{'p greater':>11}{'Holm (2s)':>11}")
    names, stats = [], []
    for name, data in rows:
        names.append(name); stats.append(paired(ref, data))
    adj = holm([s[3] for s in stats])
    for name, (n, md, wins, p2, p1), pa in zip(names, stats, adj):
        flag = "" if np.isnan(pa) else ("  *" if pa < 0.05 else "  (n.s.)")
        print(f"{name:<30}{n:>4}{md:>+11.3f}{wins:>4}/{n:<2}{p2:>13.4f}{p1:>11.4f}{pa:>11.4f}{flag}")
    if ablations:
        print(f"--- our other variants (not baselines; not in the Holm family) ---")
        for name, data in ablations:
            n, md, wins, p2, p1 = paired(ref, data)
            print(f"{name:<30}{n:>4}{md:>+11.3f}{wins:>4}/{n:<2}{p2:>13.4f}{p1:>11.4f}")
    sig = [nm for nm, pa in zip(names, adj) if not np.isnan(pa) and pa < 0.05]
    worst = max((pa, nm) for nm, pa in zip(names, adj) if not np.isnan(pa))
    print(f"significant after Holm at 0.05: {len(sig)}/{len(names)} baselines; "
          f"largest adjusted p = {worst[0]:.4f} ({worst[1]})")


def main():
    ap = argparse.ArgumentParser()
    a = ap.parse_args()
    ours_nodes, mask_nodes, grad_nodes, ours_edges = T.collect()
    head = MV.OURMETHOD
    # node level
    ref = ours_nodes[head]
    node_rows = ([(k, v) for k, v in T.NODE_BASELINES.items()]
                 + [(k, v) for k, v in grad_nodes.items()]
                 + [(k, v) for k, v in mask_nodes.items()])
    node_abl = [(k, v) for k, v in ours_nodes.items() if k != head]
    report("node", head, ref, node_rows, node_abl)
    # edge level
    ref_e = ours_edges[head]
    edge_rows = [(k, v) for k, v in T.EDGE_BASELINES.items()]
    edge_abl = [(k, v) for k, v in ours_edges.items() if k != head]
    report("edge", head, ref_e, edge_rows, edge_abl)


if __name__ == "__main__":
    main()
