"""Generate LaTeX table of MIB CPR AUC results from saved .pkl files.

Reads results from results/ directories and outputs to paper/tabs/.
Run from the repo root on the cluster:
    uv run python scripts/make_mib_table.py
"""

import pickle
import math
from pathlib import Path

RESULTS_BASE = Path("results")
OUTPUT = Path("paper/tabs/mib_results.tex")

# Column definitions: (task, model, col_header)
COLUMNS = [
    ("ioi", "gpt2", "GPT-2"),
    ("ioi", "qwen2.5", "Qwen-2.5"),
    ("ioi", "gemma2", "Gemma-2"),
    ("ioi", "llama3", "Llama-3.1"),
    ("arithmetic_subtraction", "llama3", "Llama-3.1"),
    ("mcqa", "qwen2.5", "Qwen-2.5"),
    ("mcqa", "gemma2", "Gemma-2"),
    ("mcqa", "llama3", "Llama-3.1"),
    ("arc_easy", "gemma2", "Gemma-2"),
    ("arc_easy", "llama3", "Llama-3.1"),
    ("arc_challenge", "llama3", "Llama-3.1"),
]

# Our result directories: (display_name, results_subdir)
OUR_METHODS = [
    # Node level
    ("Ours: node (sigmoid top-k)", "final_node"),
    ("Ours: node (hard top-k + ST)", "mib_node_hard_topk"),
    ("Ours: node (hard concrete + L0)", "mib_node_hard_concrete"),
    # Edge level
    ("Ours: edge (sigmoid top-k)", "final_edge"),
    ("Ours: edge (hard top-k + ST)", "mib_edge_hard_topk"),
    ("Ours: edge (hard concrete + L0)", "mib_edge_hard_concrete"),
]

# Seed run directories (for mean ± std)
SEED_DIRS = {
    "topk": "mib_node_seeds/topk",
    "hard_topk": "mib_node_seeds/hard_topk",
}

# MIB baselines (from paper Table 1)
BASELINES = {
    "Random": {
        ("ioi", "gpt2"): 0.25, ("ioi", "qwen2.5"): 0.28, ("ioi", "gemma2"): 0.30,
        ("ioi", "llama3"): 0.25, ("arithmetic_subtraction", "llama3"): 0.25,
        ("mcqa", "qwen2.5"): 0.27, ("mcqa", "gemma2"): 0.32, ("mcqa", "llama3"): 0.26,
        ("arc_easy", "gemma2"): 0.32, ("arc_easy", "llama3"): 0.26,
        ("arc_challenge", "llama3"): 0.25,
    },
    "EAP-IG-inputs (CF)": {
        ("ioi", "gpt2"): 1.85, ("ioi", "qwen2.5"): 1.63, ("ioi", "gemma2"): 3.20,
        ("ioi", "llama3"): 2.08, ("arithmetic_subtraction", "llama3"): 0.99,
        ("mcqa", "qwen2.5"): 1.16, ("mcqa", "gemma2"): 1.64, ("mcqa", "llama3"): 1.05,
        ("arc_easy", "gemma2"): 1.53, ("arc_easy", "llama3"): 1.04,
        ("arc_challenge", "llama3"): 0.98,
    },
    "NAP-IG (CF)": {
        ("ioi", "gpt2"): 0.76, ("ioi", "qwen2.5"): 0.29, ("ioi", "gemma2"): 1.52,
        ("ioi", "llama3"): 0.42, ("arithmetic_subtraction", "llama3"): 0.39,
        ("mcqa", "qwen2.5"): 0.77, ("mcqa", "gemma2"): 1.71, ("mcqa", "llama3"): 1.87,
        ("arc_easy", "gemma2"): 1.53, ("arc_easy", "llama3"): 0.26,
        ("arc_challenge", "llama3"): 0.26,
    },
    "UGS": {
        ("ioi", "gpt2"): 0.97, ("ioi", "qwen2.5"): 0.98,
        ("mcqa", "qwen2.5"): 1.17,
    },
}


def load_cpr_auc(results_dir, task, model):
    """Load CPR AUC from a MIB results pickle."""
    pkl_path = RESULTS_BASE / results_dir / f"{task}_{model}_validation.pkl"
    if not pkl_path.exists():
        return None
    try:
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
        return d["area_under"]
    except Exception:
        return None


def fmt(v, bold=False):
    if v is None:
        return "-"
    s = f"{v:.2f}"
    if bold:
        s = f"\\textbf{{{s}}}"
    return s


def main():
    # Collect all our results
    all_results = {}  # method_name -> {(task, model): cpr_auc}

    for method_name, results_dir in OUR_METHODS:
        data = {}
        for task, model, _ in COLUMNS:
            v = load_cpr_auc(results_dir, task, model)
            if v is not None:
                data[(task, model)] = round(v, 2)
        all_results[method_name] = data

    # Find best per column (across all methods including baselines)
    best_per_col = {}
    all_methods_data = {**BASELINES, **all_results}
    for task, model, _ in COLUMNS:
        best = -math.inf
        for method_data in all_methods_data.values():
            v = method_data.get((task, model))
            if v is not None and v > best:
                best = v
        best_per_col[(task, model)] = best if best > -math.inf else None

    # Generate LaTeX
    lines = []
    lines.append("\\begin{tabular}{l" + "r" * len(COLUMNS) + "}")
    lines.append("\\toprule")
    lines.append("& \\multicolumn{4}{c}{IOI} & Arithmetic & \\multicolumn{3}{c}{MCQA} & \\multicolumn{2}{c}{ARC (E)} & ARC (C) \\\\")
    lines.append("\\cmidrule(lr){2-5} \\cmidrule(lr){6-6} \\cmidrule(lr){7-9} \\cmidrule(lr){10-11} \\cmidrule(lr){12-12}")
    header = "\\textbf{Method} & " + " & ".join(h for _, _, h in COLUMNS) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    # Baselines
    for name, data in BASELINES.items():
        vals = []
        for task, model, _ in COLUMNS:
            v = data.get((task, model))
            is_best = v is not None and best_per_col.get((task, model)) == v
            vals.append(fmt(v, bold=is_best))
        lines.append(f"{name} & " + " & ".join(vals) + " \\\\")
    lines.append("\\midrule")

    # Our node methods
    for method_name in [n for n, d in OUR_METHODS[:3]]:
        data = all_results[method_name]
        vals = []
        for task, model, _ in COLUMNS:
            v = data.get((task, model))
            is_best = v is not None and best_per_col.get((task, model)) == v
            vals.append(fmt(v, bold=is_best))
        lines.append(f"{method_name} & " + " & ".join(vals) + " \\\\")
    lines.append("\\midrule")

    # Our edge methods
    for method_name in [n for n, d in OUR_METHODS[3:]]:
        data = all_results[method_name]
        vals = []
        for task, model, _ in COLUMNS:
            v = data.get((task, model))
            is_best = v is not None and best_per_col.get((task, model)) == v
            vals.append(fmt(v, bold=is_best))
        lines.append(f"{method_name} & " + " & ".join(vals) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    table = "\n".join(lines) + "\n"

    # Write
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(table)
    print(f"Wrote {OUTPUT}")
    print()
    print(table)


if __name__ == "__main__":
    main()
