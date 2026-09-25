"""Seed variance of the headline node-level MAttr on the MIB TEST split (2026-09-21).

Seeds 42 (results/test_node_topk_uniform_lr05, the paper's headline), 43 and 44
(results/test_node_topk_uniform_lr05_s43 / _s44, submit_headline_seeds_sc.sh): CPR (area_under)
and Compactness (acc_auc) per cell and seed, the per-cell mean and sample sd over seeds, and
the 12-cell Avg per seed. The sd column is what the "single seed" reviewer point asks for.

--spearman adds the agreement of the attributions themselves: per cell, the pairwise Spearman rho
of the node scores (`nodes` of `<task>_<model>_importances.json`, heads + MLPs + input, `logits`
dropped) between seeds, and the overlap of their top-10% node sets.

    uv run python scripts/mib/seed_stats.py [--tex paper/tabs/headline_seeds.tex] [--spearman]
"""
import argparse
import itertools
import json
import pickle
import statistics
import sys
from pathlib import Path

from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_mib_test_table as T   # COLUMNS, the test table's 12 cells   # noqa: E402
import ladder_nestedness as LN    # TASK_TEX / MODEL_TEX, the nestedness table's cell names   # noqa: E402

SEEDS = {42: "results/test_node_topk_uniform_lr05",
         43: "results/test_node_topk_uniform_lr05_s43",
         44: "results/test_node_topk_uniform_lr05_s44"}
METRICS = [("area_under", "CPR"), ("acc_auc", "Compactness")]


def cell(d, task, model, key):
    p = Path(d) / f"{task}_{model}_test.pkl"
    if not p.exists():
        return None
    return float(pickle.load(open(p, "rb"))[key])


def node_scores(d, task, model):
    p = Path(d) / f"{task}_{model}_importances.json"
    if not p.exists():
        return None
    nodes = json.load(open(p))["nodes"]
    return {k: v["score"] for k, v in nodes.items() if k != "logits" and v.get("score") is not None}


def rank_agreement():
    pairs = list(itertools.combinations(SEEDS, 2))
    print("\n=== Seed agreement of node scores (Spearman rho; top-10% overlap) ===")
    print(f"{'cell':28s}{'n':>6s}" + "".join(f"{f'{a}-{b}':>8s}" for a, b in pairs) + f"{'mean':>8s}{'top10%':>8s}")
    rhos, ovs = [], []
    for t, m, _ in T.COLUMNS:
        sc = {s: node_scores(d, t, m) for s, d in SEEDS.items()}
        if any(v is None for v in sc.values()):
            print(f"{t + '/' + m:28s}  missing importances for seed(s) {[s for s, v in sc.items() if v is None]}")
            continue
        keys = sorted(set.intersection(*(set(v) for v in sc.values())))
        k = max(1, len(keys) // 10)
        top = {s: set(sorted(keys, key=lambda n: -sc[s][n])[:k]) for s in SEEDS}
        r = [spearmanr([sc[a][n] for n in keys], [sc[b][n] for n in keys]).statistic for a, b in pairs]
        ov = statistics.mean(len(top[a] & top[b]) / k for a, b in pairs)
        rhos.append(statistics.mean(r)); ovs.append(ov)
        print(f"{t + '/' + m:28s}{len(keys):6d}" + "".join(f"{x:8.3f}" for x in r) + f"{rhos[-1]:8.3f}{ov:8.2f}")
    print(f"{'Avg (' + str(len(rhos)) + ' cells)':28s}{'':6s}" + " " * 8 * len(pairs)
          + f"{statistics.mean(rhos):8.3f}{statistics.mean(ovs):8.2f}")
    print(f"rho: median {statistics.median(rhos):.3f}, min {min(rhos):.3f}, max {max(rhos):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=None)
    ap.add_argument("--spearman", action="store_true")
    a = ap.parse_args()
    vals = {key: {s: {(t, m): cell(d, t, m, key) for t, m, _ in T.COLUMNS} for s, d in SEEDS.items()}
            for key, _ in METRICS}
    lines_tex = []
    for key, name in METRICS:
        print(f"\n=== {name} (MIB test, node level, headline MAttr) ===")
        print(f"{'cell':28s}" + "".join(f"{'s' + str(s):>8s}" for s in SEEDS) + f"{'mean':>8s}{'sd':>7s}")
        sds, means = [], []
        rows = []
        for t, m, h in T.COLUMNS:
            v = [vals[key][s][(t, m)] for s in SEEDS]
            have = [x for x in v if x is not None]
            mu = statistics.mean(have) if have else None
            sd = statistics.stdev(have) if len(have) > 1 else None
            if mu is not None:
                means.append(mu)
            if sd is not None:
                sds.append(sd)
            print(f"{t + '/' + m:28s}" + "".join(f"{x:8.2f}" if x is not None else f"{'---':>8s}" for x in v)
                  + (f"{mu:8.2f}" if mu is not None else f"{'---':>8s}") + (f"{sd:7.3f}" if sd is not None else f"{'---':>7s}"))
            rows.append((t, m, h, v, mu, sd))
        avg = {s: statistics.mean([x for x in vals[key][s].values() if x is not None]) for s in SEEDS
               if any(x is not None for x in vals[key][s].values())}
        print(f"{'Avg (12 cells)':28s}" + "".join(f"{avg[s]:8.2f}" if s in avg else f"{'---':>8s}" for s in SEEDS)
              + f"{statistics.mean(avg.values()):8.2f}{statistics.stdev(avg.values()) if len(avg) > 1 else float('nan'):7.3f}")
        print(f"median per-cell sd {statistics.median(sds):.3f}, max {max(sds):.3f}; seed range of the Avg "
              f"{min(avg.values()):.2f}..{max(avg.values()):.2f}")
        lines_tex.append((name, rows, avg))

    if a.tex:
        L = ["\\begin{tabular}{l" + "rrr" * len(METRICS) + "}", "\\toprule",
             "& " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{name}}}" for name, _, _ in lines_tex) + " \\\\",
             " ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(METRICS))),
             "\\textbf{Task / model} & " + " & ".join("seed 42 & mean & sd" for _ in METRICS) + " \\\\",
             "\\midrule"]
        for i in range(len(T.COLUMNS)):
            t, m, h = T.COLUMNS[i]
            cells = []
            for name, rows, _ in lines_tex:
                _, _, _, v, mu, sd = rows[i]
                cells.append(f"{v[0]:.2f} & {mu:.2f} & {sd:.3f}" if sd is not None else
                             (f"{v[0]:.2f} & --- & ---" if v[0] is not None else "--- & --- & ---"))
            L.append(f"{LN.TASK_TEX[t]} / {LN.MODEL_TEX[m]} & " + " & ".join(cells) + " \\\\")
        L.append("\\midrule")
        cells = []
        for name, rows, avg in lines_tex:
            cells.append(f"{avg[42]:.2f} & {statistics.mean(avg.values()):.2f} & "
                         f"{statistics.stdev(avg.values()):.3f}" if len(avg) > 1 else f"{avg.get(42, float('nan')):.2f} & --- & ---")
        L.append("\\textbf{Avg} & " + " & ".join(cells) + " \\\\")
        L += ["\\bottomrule", "\\end{tabular}"]
        Path(a.tex).parent.mkdir(parents=True, exist_ok=True)
        Path(a.tex).write_text("\n".join(L) + "\n")
        print(f"\nwrote {a.tex}")

    if a.spearman:
        rank_agreement()


if __name__ == "__main__":
    main()
