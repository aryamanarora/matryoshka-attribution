"""Score `impossibility_bench.py` output: ROC AUC per method per task, and the
query-budget curve for MAttr vs the brute-force (Theorem 5.1) reference.

The paper's own summary is a 40-threshold sensitivity/specificity scatter against the
diagonal; the threshold sweep of a scalar test IS a ROC, so we report its exact area, which
is the same information as one number. Random guessing = 0.5 by construction (the `random`
arm measures the finite-sample spread around it).

Restricted to `ordered_feature_idxs`, the non-categorical features -- the only ones the
paper's oracle loop is evaluated on.

    uv run python scripts/impossibility_report.py                      # table
    uv run python scripts/impossibility_report.py --budget-curve       # + MAttr vs brute force
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

TASKS = ("recourse", "spurious")
# base methods are task-agnostic (one phi, both tests applied); MAttr/brute force are
# task-specific by construction, and carry a budget.
BASE = ["random", "grad", "smoothgrad", "ig_zero", "ig_min", "lime", "shap"]
ARM_RE = re.compile(r"(mattr_(?:adam|sgd)(?:_(?:lr|eps)[0-9.eE+-]+)?|bruteforce)@(\d+)$")


def auc(scores, labels):
    """Mann-Whitney ROC AUC with mid-ranks for ties. nan if a class is absent."""
    labels = np.asarray(labels, dtype=float)
    n1, n0 = labels.sum(), (1 - labels).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(np.asarray(scores, dtype=float), kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    s = np.asarray(scores, dtype=float)[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def collect(path, readout="paper"):
    """-> {task: {method: {model: (scores, labels)}}}, plus per-method mean budget.

    `readout="paper"` scores their `_normalize`d phi (per-example min-max onto [-1,1]);
    `readout="raw"` scores the method's unnormalized output.

    AUC is invariant to a positive rescaling of the whole pooled score vector, so `raw` is
    exactly what ANY per-model (global) normalization would give -- the two readouts are
    "normalize within each example" vs "normalize once across them", not "normalized vs not".
    That distinction is the whole ballgame here: the spurious ORACLE is a single global
    variance threshold (the model's own 80th quantile over 100 reference examples), so the
    information it tests lives in the BETWEEN-example scale, which a per-example min-max
    throws away before the test ever sees it. The `bruteforce` arm is the proof, because it
    estimates the oracle's own statistic and nothing else: it scores ~0.99 raw and ~0.37 under
    their normalization. Read that as a property of the protocol, not of any method.
    """
    d = json.load(open(path))
    field = "phi" if readout == "paper" else "phi_raw"
    acc = {t: defaultdict(lambda: defaultdict(lambda: ([], []))) for t in TASKS}
    budgets = defaultdict(list)
    for rec in d["records"]:
        keep = rec["ordered"]
        mi = rec["model"]
        for task in TASKS:
            y = [rec["oracle"][task][j] for j in keep]
            for name, phi in rec[field].items():
                if "|" in name:
                    base, mode = name.split("|")
                    if mode != task:
                        continue
                    key = base
                else:
                    if name not in BASE:
                        continue
                    key = name
                v = [phi[j] for j in keep]
                # recourse tests the SIGN of phi, spurious tests its MAGNITUDE
                v = v if task == "recourse" else list(np.abs(v))
                s, l = acc[task][key][mi]
                s.extend(v)
                l.extend(y)
                budgets[key].append(rec["budget"][name])
    return d, acc, {k: float(np.mean(v)) for k, v in budgets.items()}


def rung(method, budget, p):
    """Budget curves are averaged over datasets, so the x index has to mean the same thing in
    every dataset. `bruteforce` spends n queries PER FEATURE, so its row count is n*p and a
    given row count exists in only one dataset (p differs) -- index it by n and report the
    mean row count. MAttr's step count is dataset-independent, so its rung IS its budget."""
    return budget / p if method.startswith("bruteforce") else budget


def summarize(acc):
    """-> {task: {method: (mean AUC over models, std)}}"""
    out = {}
    for task in TASKS:
        out[task] = {}
        for m, per_model in acc[task].items():
            a = [auc(s, l) for s, l in per_model.values()]
            a = [x for x in a if not np.isnan(x)]
            out[task][m] = (float(np.mean(a)), float(np.std(a)), len(a))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/impossibility")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--at-budget", type=int, default=1024,
                    help="budget (rows) at which MAttr is quoted in the main table")
    ap.add_argument("--budget-curve", action="store_true")
    ap.add_argument("--readout", choices=("paper", "raw"), default="paper",
                    help="'paper' = their min-max _normalize; 'raw' = unnormalized scores")
    args = ap.parse_args()

    paths = sorted(Path(args.results).glob("*.json"))
    if args.datasets:
        paths = [p for p in paths if p.stem in args.datasets]
    if not paths:
        raise SystemExit(f"no results in {args.results}")

    per_ds, budgets_all = {}, {}
    for p in paths:
        d, acc, bud = collect(p, args.readout)
        per_ds[p.stem] = (d, summarize(acc), bud)
        budgets_all.update(bud)

    def pick(summ, m):
        return summ.get(m)

    for task in TASKS:
        rows = BASE + sorted(k for k in budgets_all
                             if ARM_RE.match(k) and not k.startswith("bruteforce")
                             and int(k.split("@")[1]) == args.at_budget)
        rows += [k for k in sorted(budgets_all) if k.startswith("bruteforce@")]
        print(f"\n=== {task}  (ROC AUC, mean +- sd over models; 0.5 = random) ===")
        hdr = f"{'method':<22}{'budget':>8}  " + "".join(f"{n:>16}" for n in per_ds) + f"{'MEAN':>10}"
        print(hdr)
        print("-" * len(hdr))
        for m in rows:
            cells, vals = [], []
            for name, (_, summ, _) in per_ds.items():
                r = pick(summ[task], m)
                cells.append(f"{r[0]:.3f}+-{r[1]:.3f}" if r else "--")
                if r:
                    vals.append(r[0])
            b = budgets_all.get(m, float("nan"))
            print(f"{m:<22}{b:>8.0f}  " + "".join(f"{c:>16}" for c in cells)
                  + (f"{np.mean(vals):>10.3f}" if vals else f"{'--':>10}"))

    if args.budget_curve:
        print("\n=== query-budget curve (mean AUC over datasets) ===")
        for task in TASKS:
            print(f"\n{task}:")
            fam = defaultdict(lambda: defaultdict(list))
            for d, summ, bud in per_ds.values():
                for m in summ[task]:
                    mm = ARM_RE.match(m)
                    if mm:
                        k = rung(mm.group(1), int(mm.group(2)), d["p"])
                        fam[mm.group(1)][k].append((bud[m], summ[task][m][0]))
            for f, curve in fam.items():
                pts = "  ".join(
                    f"{np.mean([x for x, _ in v]):.0f}:{np.mean([y for _, y in v]):.3f}"
                    for _, v in sorted(curve.items()))
                print(f"  {f:<12} {pts}")


if __name__ == "__main__":
    main()
