"""LaTeX table: how learning rate influences node CPR AUC, per task, for each method
we swept LR on (hard_topk / MAttr and bernoulli_reinforce / +hard bwd).

Reads results/<dir>/<task>_<model>_validation.pkl. lr-sweep dirs (htk_lr_*, bern_lr_*)
currently only hold ioi/gpt2; the lr=0.01 baselines hold all tasks.
Run from repo root on sc:  uv run python scripts/make_lr_table.py
"""
import pickle
from pathlib import Path

RESULTS_BASE = Path("results")
OUTPUT = Path("paper/tabs/lr_sweep.tex")

COLUMNS = [
    ("ioi", "gpt2", "GPT"), ("ioi", "qwen2.5", "Qwen"), ("ioi", "gemma2", "Gemma"),
    ("ioi", "llama3", "Llama"), ("arithmetic_subtraction", "llama3", "Llama"),
    ("mcqa", "qwen2.5", "Qwen"), ("mcqa", "gemma2", "Gemma"), ("mcqa", "llama3", "Llama"),
    ("arc_easy", "gemma2", "Gemma"), ("arc_easy", "llama3", "Llama"),
    ("arc_challenge", "llama3", "Llama"),
]

# method -> list of (lr-label, results-dir). lr=0.01 dirs are the main runs (all tasks).
# MAttr headline = soft top-k fwd, log k; "+ hard" = sigmoid-STE hard forward.
METHODS = [
    ("\\ourmethod{}", [
        ("0.005", "topklog_lr_0.005"), ("0.01", "mib_node_topk_log"),
        ("0.05", "topklog_lr_0.05"), ("0.1", "topklog_lr_0.1"), ("0.3", "topklog_lr_0.3"),
    ]),
    ("$+$ hard", [
        ("0.005", "htklog_lr_0.005"), ("0.01", "mib_node_hard_topk_log"),
        ("0.05", "htklog_lr_0.05"), ("0.1", "htklog_lr_0.1"), ("0.3", "htklog_lr_0.3"),
    ]),
    ("$+$ unif $k$, $+$ hard", [
        ("0.005", "htk_lr_0.005"), ("0.01", "mib_node_hard_topk"),
        ("0.05", "htk_lr_0.05"), ("0.1", "htk_lr_0.1"), ("0.3", "htk_lr_0.3"),
    ]),
    ("$+$ hard bwd (REINFORCE)", [
        ("0.01", "bern_lr_0.01"),                  # baseline (orig dir was overwritten by lr0.1 rerun)
        ("0.05", "bern_lr_0.05"),
        ("0.1", "mib_node_bernoulli_reinforce"),   # lr0.1 rerun = the main +hard row (all tasks)
        ("0.3", "bern_lr_0.3"),
        ("0.1, 2k", "bern_lr_0.1_2k"),
    ]),
]


# llama3/ioi (10k val, 8B) is evaluated on a reduced 200-example subset (daggered). The
# lr=0.01 anchor for that one cell therefore reads the capped rerun, not the full-eval dir.
# llama3/ioi is eval'd on a reduced 200-example subset (daggered) in every MAttr block;
# the lr=0.01 anchor for that cell reads the capped rerun, not the full-eval main dir.
DIR_OVERRIDE = {("mib_node_hard_topk_log", "ioi", "llama3"): "htklog_lr_0.01",
                ("mib_node_topk_log", "ioi", "llama3"): "topklog_lr_0.01",
                ("mib_node_hard_topk", "ioi", "llama3"): "htk_lr_0.01"}
DAGGER_CELLS = {("ioi", "llama3")}  # capped at 200 in all 3 MAttr blocks (not REINFORCE)


def cpr(d, task, model):
    d = DIR_OVERRIDE.get((d, task, model), d)
    p = RESULTS_BASE / d / f"{task}_{model}_validation.pkl"
    if not p.exists():
        return None
    try:
        return round(pickle.load(open(p, "rb"))["area_under"], 2)
    except Exception:
        return None


def fmt(v, bold=False, dagger=False):
    if v is None:
        return "---"
    s = f"\\textbf{{{v:.2f}}}" if bold else f"{v:.2f}"
    return ("$^{\\dagger}$" + s) if dagger else s


def main():
    ncols = len(COLUMNS)
    lines = ["\\begin{adjustbox}{max width=\\textwidth}",
             "\\begin{tabular}{lr@{\\quad}" + "r" * ncols + "}", "\\toprule",
             "& & \\multicolumn{4}{c}{IOI} & Arith & \\multicolumn{3}{c}{MCQA} & "
             "\\multicolumn{2}{c}{ARC (E)} & ARC (C) \\\\",
             "\\cmidrule(lr){3-6} \\cmidrule(lr){7-7} \\cmidrule(lr){8-10} "
             "\\cmidrule(lr){11-12} \\cmidrule(lr){13-13}",
             "\\textbf{Method / LR} & \\textbf{Avg} & " + " & ".join(h for _, _, h in COLUMNS) + " \\\\",
             "\\midrule"]

    for mi, (method, lrs) in enumerate(METHODS):
        if mi:
            lines.append("\\midrule")
        data = {lr: {(t, m): cpr(d, t, m) for t, m, _ in COLUMNS} for lr, d in lrs}
        best = {}
        for t, m, _ in COLUMNS:
            vals = [data[lr][(t, m)] for lr, _ in lrs if data[lr][(t, m)] is not None]
            best[(t, m)] = max(vals) if len(vals) > 1 else None  # only bold when there's a sweep
        is_mattr = "REINFORCE" not in method   # 3 MAttr blocks cap llama/ioi; REINFORCE does not
        lines.append(f"\\multicolumn{{{ncols + 2}}}{{l}}{{\\textit{{{method}}}}} \\\\")
        for lr, _ in lrs:
            present = [data[lr][(t, m)] for t, m, _ in COLUMNS if data[lr][(t, m)] is not None]
            avg = f"{sum(present) / len(present):.2f}" if present else "---"
            cells = [fmt(data[lr][(t, m)],
                         bold=(data[lr][(t, m)] is not None and data[lr][(t, m)] == best[(t, m)]),
                         dagger=(is_mattr and (t, m) in DAGGER_CELLS and data[lr][(t, m)] is not None))
                     for t, m, _ in COLUMNS]
            lines.append(f"\\quad LR$=${lr} & {avg} & " + " & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines += ["\\end{tabular}", "\\end{adjustbox}"]
    table = "\n".join(lines) + "\n"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(table)
    print(f"Wrote {OUTPUT}\n")
    print(table)


if __name__ == "__main__":
    main()
