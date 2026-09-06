"""Per method: is training on CE or soft-acc different from training on logit-diff?

PAIRED, and the pairing is what makes this worth doing. Method-vs-method comparisons on SVA+
are confounded by which cells a method covers; a loss-vs-loss comparison inside ONE method is
not, because the same method under two losses runs the same 36 cells. So each test asks a
within-method question -- "on this exact cell, did swapping the objective move the score?" --
and the cell effect, which is by far the largest source of variance here (IIA AUC ranges 0.02
to 0.88 across cells), differences out.

UNIT OF PAIRING is a CELL: (results dir, substrate, task) over the two PATCHED dirs
(results/sva_sweep, results/sva_sweep_input), i.e. 36 of them -- node/mlp/mlp+attn x SVA+Arith,
plus ARC-E and IOI at node, plus the ten +input node cells. Zero-ablation dirs are excluded for
the reason plot_accauc_vs_faithauc.FIGURE_SOURCES gives: their IIA AUC sits on a ~0.51 random
floor against the patched ~0.03, so a difference there is mostly floor and would dilute the
paired sample with cells that cannot move.

*** NOT AVERAGED OVER TASK-GROUPS. *** The figures macro-average SVA and Arith to one point
each so the four-task groups do not outvote the single-task MIB cells. That is right for a
scatter and wrong here: it would throw away 30 of the 36 pairs and leave n=4, which no paired
test can say anything with. The cost is that the 4 SVA and 4 Arith cells are not independent of
each other (same model, related prompts), so the effective sample size is below 36 and the
p-values are somewhat optimistic. Read them as screening, not as confirmatory.

WILCOXON SIGNED-RANK is the headline test: n<=36, the per-cell differences are not normal (a
method that collapses under CE produces a long one-sided tail), and the AUCs are bounded. The
paired t is printed beside it as a sanity check, not as the answer -- where the two disagree,
trust Wilcoxon.

HOLM-BONFERRONI across all tests within a metric (8 methods x 2 losses = 16), because that is
the family actually being screened. Both raw and adjusted p are printed; a claim in the paper
should cite the adjusted one.

TWO METRICS, reported separately. They do not measure the same thing and a loss can move one
without the other -- IIA AUC is a bounded fraction-of-decided integral, faith AUC normalises by
(F_clean - F_patch) and is the noisy axis (~10% run-to-run at fixed config, against ~0.002 for
IIA), so faith needs a bigger effect before it means anything.

Run:  uv run python scripts/sva/loss_paired_tests.py [--metric acc|faith|both]
"""
import argparse
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "plots"))
import plot_accauc_vs_faithauc as V   # noqa: E402  loaders, task lists, the task->model pin

REF = "logit_diff"                      # the loss everything is compared against
OTHERS = ["ce", "acc"]
# Methods to test: every registry entry with a training loss. Random is excluded because it has
# none (a random ranking is not an optimisation), and the eps arm because it was only ever run
# at logit-diff, so it has nothing to pair.
SKIP = {"Random", "stopk-log-eps1e-2"}
METRIC = {"acc": (0, "IIA AUC"), "faith": (1, "Faith AUC")}


def cells():
    """{(method, loss): {cell: (acc_auc, faith_auc)}} over the patched dirs, unaveraged."""
    out = {}
    for res, _inp, _abl in V.FIGURE_SOURCES:
        raw = V.load(res)                       # already applies the task->model pin
        for (m, loss, sub, task), v in raw.items():
            out.setdefault((m, loss), {})[(res, sub, task)] = v
    return out


def holm(ps):
    """Holm-Bonferroni adjusted p-values, order preserved. Monotone (cummax) as the method
    requires -- without it an adjusted p can come out below an earlier, smaller one."""
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (len(ps) - i) * ps[idx])
        adj[idx] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="both", choices=["acc", "faith", "both"])
    a = ap.parse_args()

    data = cells()
    methods = [m for m in V.METHODS if m not in SKIP]

    for mkey in (["acc", "faith"] if a.metric == "both" else [a.metric]):
        idx, mlabel = METRIC[mkey]
        rows = []
        for m in methods:
            for loss in OTHERS:
                A, B = data.get((m, REF), {}), data.get((m, loss), {})
                shared = sorted(set(A) & set(B))
                if len(shared) < 6:             # a signed-rank test on <6 pairs cannot reach .05
                    rows.append((m, loss, len(shared), None, None, None, None, None))
                    continue
                x = np.array([A[c][idx] for c in shared])
                y = np.array([B[c][idx] for c in shared])
                d = y - x                        # positive = the alternative loss scored HIGHER
                if np.allclose(d, 0):
                    rows.append((m, loss, len(shared), 0.0, 0.0, 1.0, 1.0, 0.0))
                    continue
                w = stats.wilcoxon(y, x, zero_method="wilcox", alternative="two-sided")
                t = stats.ttest_rel(y, x)
                dz = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else np.nan
                rows.append((m, loss, len(shared), float(np.median(d)), float(d.mean()),
                             float(w.pvalue), float(t.pvalue), dz))
        live = [i for i, r in enumerate(rows) if r[5] is not None]
        adj = holm([rows[i][5] for i in live])
        adjmap = {i: p for i, p in zip(live, adj)}

        print(f"\n=== {mlabel}: each loss vs logit-diff, paired by cell "
              f"(patched dirs, {len(data.get(('IG', REF), {}))} cells max)")
        print(f"{'method':<16}{'loss':>5}{'n':>4}{'med Δ':>9}{'mean Δ':>9}"
              f"{'dz':>7}{'p(wilcox)':>11}{'p(t)':>9}{'p_holm':>9}  ")
        for i, (m, loss, n, med, mean, pw, pt, dz) in enumerate(rows):
            name = V.METHODS[m][0]
            if pw is None:
                print(f"{name:<16}{loss:>5}{n:>4}{'':>9}{'':>9}{'':>7}"
                      f"{'too few pairs':>11}")
                continue
            pa = adjmap[i]
            star = "***" if pa < .001 else "**" if pa < .01 else "*" if pa < .05 else ""
            print(f"{name:<16}{loss:>5}{n:>4}{med:>+9.3f}{mean:>+9.3f}{dz:>+7.2f}"
                  f"{pw:>11.2e}{pt:>9.2e}{pa:>9.2e}  {star}")
        print("  Δ = (alternative loss) − (logit-diff); positive means the alternative scored "
              "higher.\n  p_holm is Holm-corrected over the tests in this block; "
              "* <.05  ** <.01  *** <.001")


if __name__ == "__main__":
    sys.exit(main())
