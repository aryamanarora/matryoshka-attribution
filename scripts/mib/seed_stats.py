"""Seed variance of the headline node-level MAttr on the MIB TEST split (2026-09-21).

Seeds 42 (results/test_node_topk_uniform_lr05, the paper's headline), 43 and 44
(results/test_node_topk_uniform_lr05_s43 / _s44, submit_headline_seeds_sc.sh): CPR (area_under)
and Compactness (acc_auc) per cell and seed, the per-cell mean and sample sd over seeds, and
the 12-cell Avg per seed. The sd column is what the "single seed" reviewer point asks for.

    uv run python scripts/mib/seed_stats.py [--tex paper/tabs/headline_seeds.tex]
"""
import argparse
import pickle
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_mib_test_table as T   # COLUMNS, the test table's 12 cells   # noqa: E402

SEEDS = {42: "results/test_node_topk_uniform_lr05",
         43: "results/test_node_topk_uniform_lr05_s43",
         44: "results/test_node_topk_uniform_lr05_s44"}
METRICS = [("area_under", "CPR"), ("acc_auc", "Compactness")]


def cell(d, task, model, key):
    p = Path(d) / f"{task}_{model}_test.pkl"
    if not p.exists():
        return None
    return float(pickle.load(open(p, "rb"))[key])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=None)
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
            L.append(f"{T.TASK_TEX[t] if hasattr(T, 'TASK_TEX') else t.replace('_', ' ')} / {h} & " + " & ".join(cells) + " \\\\")
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


if __name__ == "__main__":
    main()
