"""Print the 13x13 cross-task transfer matrix from whatever has landed on disk.

Rows = TARGET task (whose eval set is scored), columns = SOURCE task (whose MAttr ranking is
evaluated). Each cell is the target-task metric of the source's ranking DIVIDED by the
diagonal (the target's own ranking, recomputed by the same job on the same 200 examples), so
1.00 on the diagonal by construction and "how much of the task's own performance does a
foreign circuit retain" off it. Raw values print with --raw.

The two harness halves use their own headline metric -- MIB targets: CPR (area_under);
eval_sva targets: faith_auc (iso/sufficiency) -- which is exactly why cells are only
comparable within a row. --metric acc switches to acc-AUC on both halves.

Partial-tolerant: missing cells print as "." and a summary of what's still pending follows.
"""
import argparse
import json
from pathlib import Path

R = Path("results")
MIB_TASKS = ["ioi", "arithmetic_subtraction", "mcqa", "arc_easy", "arc_challenge"]
SVA_TASKS = ["simple", "nounpp", "rc", "within_rc", "addition", "months", "weekdays", "hours"]
TASKS = MIB_TASKS + SVA_TASKS   # heatmap order (plots/plot_task_corr_heatmap.py)
SHORT = {"ioi": "ioi", "arithmetic_subtraction": "arith", "mcqa": "mcqa", "arc_easy": "arc-e",
         "arc_challenge": "arc-c", "simple": "simple", "nounpp": "nounpp", "rc": "rc",
         "within_rc": "w-rc", "addition": "add", "months": "months", "weekdays": "wkday",
         "hours": "hours"}


def load_row(target, metric, method=""):
    """{source: value} for one target task, from whichever harness owns it.

    ``method``: "" for the original MAttr(SGD) round (results/transfer_{mib,sva}); anything
    else reads the per-method round submit_transfer.sh METHOD=<m> wrote to
    results/transfer_{mib,sva}_<m>.
    """
    suf = f"_{method}" if method else ""
    if target in MIB_TASKS:
        fn = R / f"transfer_mib{suf}" / f"{target}_llama3_transfer.json"
        if not fn.exists():
            return {}
        key = "acc_auc" if metric == "acc" else "area_under"
        return {src: d[key] for src, d in json.load(open(fn)).items()}
    row = {}
    key = "acc_auc" if metric == "acc" else "faith_auc"
    for src in TASKS:
        fn = R / f"transfer_sva{suf}" / f"{target}_llama3_node_xfer_{src}.json"
        if fn.exists():
            row[src] = json.load(open(fn))[key]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="cpr", choices=["cpr", "acc"],
                    help="cpr = CPR / faith_auc per harness; acc = acc-AUC on both")
    ap.add_argument("--raw", action="store_true", help="print raw values, not diagonal-relative")
    ap.add_argument("--method", default="", help="per-method round suffix (adam/mc_ig/attnlrp); empty = MAttr(SGD)")
    args = ap.parse_args()

    rows = {t: load_row(t, args.metric, args.method) for t in TASKS}
    done = sum(len(r) for r in rows.values())
    print(f"{done}/169 cells on disk  (metric={args.metric}, "
          f"{'raw' if args.raw else 'relative to diagonal'})\n")

    hdr = "target\\src" + "".join(f"{SHORT[s]:>8}" for s in TASKS)
    print(hdr)
    for t in TASKS:
        r, diag = rows[t], rows[t].get(t)
        cells = []
        for s in TASKS:
            v = r.get(s)
            if v is None or (not args.raw and diag is None):
                cells.append(f"{'.':>8}")
            else:
                cells.append(f"{v if args.raw else v / diag:8.3f}")
        print(f"{SHORT[t]:<10}" + "".join(cells))

    missing = {t: [s for s in TASKS if s not in rows[t]] for t in TASKS}
    missing = {t: m for t, m in missing.items() if m}
    if missing:
        print("\npending:")
        for t, m in missing.items():
            print(f"  {t}: {len(m)} cells" + ("" if len(m) == 13 else f" ({', '.join(SHORT[s] for s in m)})"))


if __name__ == "__main__":
    main()
