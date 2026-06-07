"""Generate LaTeX table of hybrid MLP/attn swap results.

Usage:
    uv run python scripts/make_hybrid_table.py
"""

import pickle
from pathlib import Path

RESULTS_BASE = Path("results/hybrid_eval")
OUTPUT = Path("paper/tabs/hybrid_results.tex")

COLUMNS = [
    ("ioi", "gpt2", "IOI/GPT"),
    ("ioi", "qwen2.5", "IOI/Qwen"),
    ("mcqa", "qwen2.5", "MCQA/Qwen"),
    ("mcqa", "gemma2", "MCQA/Gemma"),
    ("arc_easy", "gemma2", "ARC-E/Gemma"),
]

METHODS = [
    ("ours", "Ours"),
    ("napig", "NAP-IG"),
    ("napig_ours_mlp", "NAP-IG attn + our MLPs"),
    ("ours_napig_mlp", "Our attn + NAP-IG MLPs"),
]

OURS_DIR = Path("results/mib_node_hard_topk_log")
NAPIG_DIR = Path("results/napig_repro_eval/EAP-IG-inputs_patching_node")


def load(task, model, method):
    stask = task.replace("_", "-")
    if method == "ours":
        p = OURS_DIR / f"{task}_{model}_validation.pkl"
    elif method == "napig":
        p = NAPIG_DIR / f"{stask}_{model}_validation_abs-False.pkl"
    else:
        p = RESULTS_BASE / f"{task}_{model}_{method}_validation.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return round(pickle.load(f)["area_under"], 2)


def fmt(v, bold=False, underline=False):
    if v is None:
        return "---"
    s = f"{v:.2f}"
    if bold:
        s = f"\\textbf{{{s}}}"
    elif underline:
        s = f"\\underline{{{s}}}"
    return s


def main():
    # Collect results
    data = {}
    for hybrid, _ in METHODS:
        data[hybrid] = {}
        for task, model, _ in COLUMNS:
            data[hybrid][(task, model)] = load(task, model, hybrid)

    # Find best/second per column
    best = {}
    second = {}
    for task, model, _ in COLUMNS:
        vals = [data[h][(task, model)] for h, _ in METHODS if data[h][(task, model)] is not None]
        if vals:
            sorted_vals = sorted(set(vals), reverse=True)
            best[(task, model)] = sorted_vals[0]
            second[(task, model)] = sorted_vals[1] if len(sorted_vals) > 1 else None
        else:
            best[(task, model)] = None
            second[(task, model)] = None

    # Generate LaTeX
    ncols = len(COLUMNS)
    lines = []
    lines.append("\\begin{adjustbox}{max width=\\textwidth}")
    lines.append("\\begin{tabular}{l" + "r" * ncols + "}")
    lines.append("\\toprule")
    lines.append("\\textbf{Ranking} & " + " & ".join(label for _, _, label in COLUMNS) + " \\\\")
    lines.append("\\midrule")

    for hybrid, name in METHODS:
        vals = []
        for task, model, _ in COLUMNS:
            v = data[hybrid][(task, model)]
            is_best = v is not None and best.get((task, model)) == v
            is_second = v is not None and not is_best and second.get((task, model)) == v
            vals.append(fmt(v, bold=is_best, underline=is_second))
        lines.append(f"{name} & " + " & ".join(vals) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{adjustbox}")

    table = "\n".join(lines) + "\n"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(table)
    print(f"Wrote {OUTPUT}")
    print()
    print(table)


if __name__ == "__main__":
    main()
