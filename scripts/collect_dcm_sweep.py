"""Collect the DCM lr sweep and report CPR/accuracy AT THE PINNED DENSITY.

DCM produces a set, not a ranking, so its CPR-AUC over MIB's full sparsity sweep is not a
fair summary — the interior of that curve is decided by tie order among scores that
``clamp_(0,1)`` piled onto exactly 0.0 and 1.0. What IS fair is the single sweep point whose
k equals the density the PID was pinned to, where ``round(mask)`` and top-k are the same set.

So this reports ``faithfulnesses[i]`` and ``accuracies[i]`` at that point, not ``area_under``.
The AUC columns are printed too, greyed out in meaning: use them only to see how much the
tie mass is worth, never as the headline.

    python scripts/collect_dcm_sweep.py
    python scripts/collect_dcm_sweep.py --densities 0.05
"""

import argparse
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

# MIB's sweep points, as fractions of the substrate (run_evaluation.py). The pinned
# densities must be drawn from this grid or there is no matching point to read.
SWEEP_POINTS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]


def sweep_index(density: float) -> int:
    for i, p in enumerate(SWEEP_POINTS):
        if abs(p - density) < 1e-9:
            return i
    raise SystemExit(
        f"density {density} is not one of MIB's sweep points {SWEEP_POINTS}; "
        f"there is no curve point where round(mask) and top-k coincide.")


def achieved_counts(train_dir: Path) -> dict:
    """cell -> (units left in round(mask), total maskable units).

    Counts, not densities, because the substrates differ by 2.3x (gpt2 has 156 nodes,
    qwen2.5 360) and the question "did it land" is about whether the kept set matches the
    k units MIB will take, which is a count. At the 1% pin that is 2 units on gpt2 and 4
    on qwen2.5, so a density-space tolerance means completely different things per cell.

    The pinned density is a *target*, and DCM does not always reach it: the additive
    penalty's per-unit gradient is the constant ``mult/total``, so a single scalar
    threshold decides the whole circuit, and the PID has to land ``mult`` in the gap
    between the k-th and (k+1)-th task-gradient magnitudes. Where that gap is narrow the
    controller steps over it and the mask collapses toward empty.

    That matters for reading the table, not just for curiosity -- see ``verdict``.
    """
    out = {}
    for graph in train_dir.glob("graph_*.json"):
        try:
            with open(graph) as f:
                nodes = json.load(f)["nodes"]
        except (OSError, ValueError, KeyError):
            continue
        scores = [v["score"] for v in nodes.values() if isinstance(v, dict) and "score" in v]
        if not scores:
            continue
        # `input` is forced into every circuit and is not one of the maskable units.
        stem = graph.stem[len("graph_"):]              # <task>_<model>, task underscored
        out[stem.replace("-", "_")] = (sum(1 for s in scores if s >= 0.5) - 1, len(scores) - 1)
    return out


def verdict(kept: int, k: int) -> str:
    """How much of the scored circuit is actually the set DCM converged to.

    MIB always takes exactly k units by score. DCM's scores are 0/1 plus a pruning-order
    tie-break, so what k selects depends on how many units DCM left standing:

      ``=``    kept == k. Top-k IS DCM's set. The only fully apples-to-apples case.
      ``sub``  kept > k. Top-k is a subset of DCM's set, with the tie-break choosing
               which subset. Still DCM's units, weaker claim about which.
      ``pad``  kept < k. DCM's set is too small, so top-k fills the remainder from units
               DCM discarded, ordered by when they died. Partly a trajectory ranking.
      ``TIE``  kept == 0. Nothing survived; the score is entirely the trajectory ranking
               and says nothing about the set DCM converged to.
    """
    if kept == 0:
        return "TIE"
    return "=" if kept == k else ("sub" if kept > k else "pad")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--densities", nargs="+", type=float, default=[0.01, 0.05, 0.2])
    ap.add_argument("--split", default="validation")
    args = ap.parse_args()

    root = Path(args.results)
    pat = re.compile(r"^eprun_eval_ld_dcm_d([0-9.]+)_lr([0-9.]+)$")

    # (density, lr) -> {cell: (cpr_at_pin, acc_at_pin, auc, kept, n_units)}
    table = defaultdict(dict)
    for d in sorted(root.glob("eprun_eval_ld_dcm_d*_lr*")):
        m = pat.match(d.name)
        if not m:
            continue
        density, lr = float(m.group(1)), float(m.group(2))
        if density not in args.densities:
            continue
        idx = sweep_index(density)
        counts = achieved_counts(root / d.name.replace("eprun_eval_", "eprun_node_"))
        for pkl in d.rglob(f"*_{args.split}_abs-False.pkl"):
            cell = pkl.name.replace(f"_{args.split}_abs-False.pkl", "")
            with open(pkl, "rb") as f:
                r = pickle.load(f)
            faith, acc = r.get("faithfulnesses"), r.get("accuracies")
            if not faith or len(faith) <= idx:
                continue
            kept, n_units = counts.get(cell.replace("-", "_"), (None, None))
            table[(density, lr)][cell] = (faith[idx], acc[idx], r.get("area_under"),
                                          kept, n_units)

    if not table:
        raise SystemExit("no DCM eval dirs found — has the sweep finished?")

    for density in sorted(args.densities):
        rows = {lr: cells for (dd, lr), cells in table.items() if dd == density}
        if not rows:
            continue
        cells = sorted({c for v in rows.values() for c in v})
        pin_pct = density * 100
        print(f"\n=== pinned density {density:g} ({pin_pct:g}% of substrate) — "
              f"CPR at that sweep point ===")
        # Each cell gets CPR at the pin plus "kept/k v" -- how many units DCM actually
        # left standing, how many MIB took, and which of the four cases that is.
        head = "  ".join(f"{c:>15} {'kept/k':>9} {'v':>3}" for c in cells)
        print(f"{'lr':>6}  {head}  {'mean':>8}  {'mean AUC':>9}  exact")
        best = None
        for lr in sorted(rows):
            vals = [rows[lr].get(c) for c in cells]
            got = [v[0] for v in vals if v]
            aucs = [v[2] for v in vals if v and v[2] is not None]
            scored = [v for v in vals if v and v[3] is not None]
            mean = sum(got) / len(got) if got else float("nan")
            mean_auc = sum(aucs) / len(aucs) if aucs else float("nan")
            n_exact = sum(1 for v in scored if verdict(v[3], round(density * v[4])) == "=")
            parts = []
            for v in vals:
                if not v:
                    parts.append(f"{'-':>15} {'-':>9} {'-':>3}")
                elif v[3] is None:
                    parts.append(f"{v[0]:>15.3f} {'?':>9} {'?':>3}")
                else:
                    k = round(density * v[4])
                    parts.append(f"{v[0]:>15.3f} {f'{v[3]}/{k}':>9} {verdict(v[3], k):>3}")
            n_missing = sum(1 for v in vals if v is None)
            flag = f"  ({n_missing} missing)" if n_missing else ""
            print(f"{lr:>6g}  {'  '.join(parts)}  {mean:>8.3f}  {mean_auc:>9.3f}  "
                  f"{n_exact}/{len(scored)}{flag}")
            # Only a run that hit its own setpoint everywhere is eligible to be "best":
            # comparing a landed circuit against a collapsed one's tie-break ranking
            # would pick the winner on which artifact got scored, not on which lr is good.
            if got and scored and n_exact == len(scored) and (best is None or mean > best[1]):
                best = (lr, mean)
        if best:
            print(f"  -> best lr at density {density:g}: {best[0]:g}  (mean CPR {best[1]:.3f})")
        else:
            print(f"  -> NO lr hit density {density:g} exactly in every cell; read the "
                  f"kept/k columns before quoting any CPR here (TIE = pure tie-break).")


if __name__ == "__main__":
    main()
