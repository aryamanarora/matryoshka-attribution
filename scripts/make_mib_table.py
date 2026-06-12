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
    ("ioi", "gpt2", "GPT"),
    ("ioi", "qwen2.5", "Qwen"),
    ("ioi", "gemma2", "Gemma"),
    ("ioi", "llama3", "Llama"),
    ("arithmetic_subtraction", "llama3", "Llama"),
    ("mcqa", "qwen2.5", "Qwen"),
    ("mcqa", "gemma2", "Gemma"),
    ("mcqa", "llama3", "Llama"),
    ("arc_easy", "gemma2", "Gemma"),
    ("arc_easy", "llama3", "Llama"),
    ("arc_challenge", "llama3", "Llama"),
]

# Our method + ablations: (display_name, results_subdir, level, is_ours)
# (display_name, results_subdir, level, group)
# group: "ours" = default, "uniform" = uniform k ablation
OUR_METHODS = [
    # Node level (log k-schedule = default)
    ("\\ourmethod{}", "mib_node_hard_topk_log", "node", "ours"),
    ("$+$ soft fwd", "mib_node_topk_log", "node", "ours"),
    ("$+$ soft fwd, $-$ $c_k$ grad", "mib_node_detached_tau_log", "node", "ours"),
    ("$+$ hard bwd", "mib_node_bernoulli_reinforce_log", "node", "ours"),
    # Node level (uniform k-schedule = ablation)
    ("\\ourmethod{}", "mib_node_hard_topk", "node", "uniform"),
    ("$+$ Gumbel sel.", "mib_node_hard_topk_gumbel", "node", "uniform"),
    ("$+$ soft fwd", "final_node", "node", "uniform"),
    ("$+$ soft fwd, $-$ $c_k$ grad", "mib_node_detached_tau", "node", "uniform"),
    ("$+$ hard bwd", "mib_node_bernoulli_reinforce", "node", "uniform"),
    # Edge level (log k-schedule = default)
    ("\\ourmethod{}", "mib_edge_hard_topk", "edge", "ours"),
    ("$+$ soft fwd", "final_edge", "edge", "ours"),
    ("$+$ soft fwd, $-$ $c_k$ grad", "mib_edge_detached_tau", "edge", "ours"),
    ("$+$ hard bwd", "mib_edge_bernoulli_reinforce", "edge", "ours"),
    # Edge level (uniform k-schedule)
    ("\\ourmethod{}", "mib_edge_hard_topk_uniform", "edge", "uniform"),
]

# Seed run directories (for mean ± std)
SEED_DIRS = {
    "topk": "mib_node_seeds/topk",
    "hard_topk": "mib_node_seeds/hard_topk",
}

# Node baselines (reproduced on validation set)
NODE_BASELINES = {}

# NAP-IG reproduced: read from results/napig_repro_eval/
NAPIG_REPRO_DIR = "napig_repro_eval"

EAPIG_REPRO_DIR = "eapig_repro_eval"

# Edge baselines (reproduced on validation set)
EDGE_BASELINES = {}


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
    # Collect all our results
    all_results = {}  # method_key -> {(task, model): cpr_auc}

    for method_name, results_dir, level, group in OUR_METHODS:
        key = f"{method_name}_{level}_{group}"
        data = {}
        for task, model, _ in COLUMNS:
            v = load_cpr_auc(results_dir, task, model)
            if v is not None:
                data[(task, model)] = round(v, 2)
        all_results[key] = data

    # Find best per column per level
    node_ours = [(n, d, l, g) for n, d, l, g in OUR_METHODS if l == "node" and g == "ours"]
    node_uniform = [(n, d, l, g) for n, d, l, g in OUR_METHODS if l == "node" and g == "uniform"]
    edge_ours = [(n, d, l, g) for n, d, l, g in OUR_METHODS if l == "edge" and g == "ours"]
    edge_uniform = [(n, d, l, g) for n, d, l, g in OUR_METHODS if l == "edge" and g == "uniform"]

    def best_in_col(level):
        baselines = NODE_BASELINES if level == "node" else EDGE_BASELINES
        our = {f"{n}_{l}_{g}": all_results.get(f"{n}_{l}_{g}", {}) for n, _, l, g in OUR_METHODS if l == level}
        best = {}
        second = {}
        for task, model, _ in COLUMNS:
            vals = []
            for data in list(baselines.values()) + list(our.values()):
                v = data.get((task, model))
                if v is not None:
                    vals.append(v)
            if vals:
                sorted_vals = sorted(set(vals), reverse=True)
                best[(task, model)] = sorted_vals[0]
                second[(task, model)] = sorted_vals[1] if len(sorted_vals) > 1 else None
            else:
                best[(task, model)] = None
                second[(task, model)] = None
        return best, second

    best_node, second_node = best_in_col("node")
    best_edge, second_edge = best_in_col("edge")

    # Cells evaluated on a reduced subset (OOM fallback) -> mark with a dagger.
    DAGGER = {
        "NAP-IG (CF, repro)": {("mcqa", "llama3")},
        "EAP-IG-inp (CF, repro)": {("arc_challenge", "llama3")},
    }

    def logk(name):
        # "+ log k" precedes other attributes (no parens)
        if name.startswith("\\ourmethod"):
            return name + " $+$ log $k$"
        return "$+$ log $k$ " + name

    def make_row(name, data, best_col, second_col, indent=False, dagger=None):
        dcells = dagger if dagger is not None else DAGGER.get(name, set())
        vals = []
        for task, model, _ in COLUMNS:
            v = data.get((task, model))
            is_best = v is not None and best_col.get((task, model)) == v
            is_second = v is not None and not is_best and second_col.get((task, model)) == v
            cell = fmt(v, bold=is_best, underline=is_second)
            if v is not None and (task, model) in dcells:
                cell = cell + "$^\\dagger$"
            vals.append(cell)
        if indent:
            prefix = f"\\quad {name}"
        else:
            prefix = name
        return f"{prefix} & " + " & ".join(vals) + " \\\\"

    # Generate LaTeX
    ncols = len(COLUMNS)
    lines = []
    lines.append("\\begin{adjustbox}{max width=\\textwidth}")
    lines.append("\\begin{tabular}{l" + "r" * ncols + "}")
    lines.append("\\toprule")
    lines.append("& \\multicolumn{4}{c}{IOI} & Arithmetic & \\multicolumn{3}{c}{MCQA} & \\multicolumn{2}{c}{ARC (E)} & ARC (C) \\\\")
    lines.append("\\cmidrule(lr){2-5} \\cmidrule(lr){6-6} \\cmidrule(lr){7-9} \\cmidrule(lr){10-11} \\cmidrule(lr){12-12}")
    header = "\\textbf{Method} & " + " & ".join(h for _, _, h in COLUMNS) + " \\\\"
    lines.append(header)

    # === Node-level section ===
    lines.append("\\midrule")
    lines.append(f"\\multicolumn{{{ncols + 1}}}{{l}}{{\\textit{{Node-level}}}} \\\\")
    # Load NAP-IG repro results
    napig_repro = {}
    for task, model, _ in COLUMNS:
        stask = task.replace("_", "-")
        pkl = RESULTS_BASE / NAPIG_REPRO_DIR / f"EAP-IG-inputs_patching_node" / f"{stask}_{model}_validation_abs-False.pkl"
        if pkl.exists():
            try:
                with open(pkl, "rb") as f:
                    d = pickle.load(f)
                napig_repro[(task, model)] = round(d["area_under"], 2)
            except Exception:
                pass
    NODE_BASELINES["NAP-IG (CF, repro)"] = napig_repro

    # Recompute best after adding repro
    best_node, second_node = best_in_col("node")

    for name, data in NODE_BASELINES.items():
        lines.append(make_row(name, data, best_node, second_node))
    lines.append("\\textbf{Ours} \\\\")
    # uniform-k = main method
    for method_name, _, _, group in node_uniform:
        key = f"{method_name}_node_{group}"
        lines.append(make_row(method_name, all_results.get(key, {}), best_node, second_node, indent=True))
    # log-k variants, annotated (no separate section)
    for method_name, _, _, group in node_ours:
        key = f"{method_name}_node_{group}"
        lines.append(make_row(logk(method_name), all_results.get(key, {}), best_node, second_node, indent=True))

    # === Edge-level section ===
    lines.append("\\midrule")
    lines.append(f"\\multicolumn{{{ncols + 1}}}{{l}}{{\\textit{{Edge-level}}}} \\\\")

    # Load EAP-IG repro results
    eapig_repro = {}
    for task, model, _ in COLUMNS:
        stask = task.replace("_", "-")
        pkl = RESULTS_BASE / EAPIG_REPRO_DIR / f"EAP-IG-inputs_patching_edge" / f"{stask}_{model}_validation_abs-False.pkl"
        if pkl.exists():
            try:
                with open(pkl, "rb") as f:
                    d = pickle.load(f)
                eapig_repro[(task, model)] = round(d["area_under"], 2)
            except Exception:
                pass
    EDGE_BASELINES["EAP-IG-inp (CF, repro)"] = eapig_repro
    best_edge, second_edge = best_in_col("edge")

    # MAttr edge llama3 cells use a reduced eval subset (sphinx 80GB rerun) -> dagger.
    EDGE_LLAMA_DAGGER = {("ioi", "llama3"), ("arithmetic_subtraction", "llama3"), ("mcqa", "llama3")}
    for name, data in EDGE_BASELINES.items():
        lines.append(make_row(name, data, best_edge, second_edge))
    lines.append("\\textbf{Ours} \\\\")
    # uniform-k = main method
    for method_name, _, _, group in edge_uniform:
        key = f"{method_name}_edge_{group}"
        lines.append(make_row(method_name, all_results.get(key, {}), best_edge, second_edge, indent=True, dagger=EDGE_LLAMA_DAGGER))
    # log-k variants, annotated (no separate section)
    for method_name, _, _, group in edge_ours:
        key = f"{method_name}_edge_{group}"
        lines.append(make_row(logk(method_name), all_results.get(key, {}), best_edge, second_edge, indent=True, dagger=EDGE_LLAMA_DAGGER))

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{adjustbox}")

    table = "\n".join(lines) + "\n"

    # Write
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(table)
    print(f"Wrote {OUTPUT}")
    print()
    print(table)


if __name__ == "__main__":
    main()
