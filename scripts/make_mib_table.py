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
OUR_METHODS = [
    # Node level (log k-schedule = default)
    ("\\ourmethod{}", "mib_node_hard_topk_log", "node", True),
    ("$+$ soft fwd", "mib_node_topk_log", "node", True),
    ("$+$ soft fwd, $-$ $c_k$ grad", "mib_node_detached_tau_log", "node", True),
    ("$+$ hard bwd", "mib_node_bernoulli_reinforce_log", "node", True),
    # Node level (uniform k-schedule = ablation)
    ("\\ourmethod{} ($-$ log $k$)", "mib_node_hard_topk", "node", True),
    ("$+$ soft fwd ($-$ log $k$)", "final_node", "node", True),
    ("$+$ soft fwd, $-$ $c_k$ grad ($-$ log $k$)", "mib_node_detached_tau", "node", True),
    ("$+$ hard bwd ($-$ log $k$)", "mib_node_bernoulli_reinforce", "node", True),
    # Edge level (log k-schedule = default)
    ("\\ourmethod{}", "mib_edge_hard_topk", "edge", True),
    ("$+$ soft fwd", "final_edge", "edge", True),
    ("$+$ soft fwd, $-$ $c_k$ grad", "mib_edge_detached_tau", "edge", True),
    ("$+$ hard bwd", "mib_edge_bernoulli_reinforce", "edge", True),
]

# Seed run directories (for mean ± std)
SEED_DIRS = {
    "topk": "mib_node_seeds/topk",
    "hard_topk": "mib_node_seeds/hard_topk",
}

# MIB baselines (from paper Table 1), grouped by level
NODE_BASELINES = {
    "Random$^\\dagger$": {
        ("ioi", "gpt2"): 0.25, ("ioi", "qwen2.5"): 0.28, ("ioi", "gemma2"): 0.30,
        ("ioi", "llama3"): 0.25, ("arithmetic_subtraction", "llama3"): 0.25,
        ("mcqa", "qwen2.5"): 0.27, ("mcqa", "gemma2"): 0.32, ("mcqa", "llama3"): 0.26,
        ("arc_easy", "gemma2"): 0.32, ("arc_easy", "llama3"): 0.26,
        ("arc_challenge", "llama3"): 0.25,
    },
    "NAP (CF)$^\\dagger$": {
        ("ioi", "gpt2"): 0.28, ("ioi", "qwen2.5"): 0.30, ("ioi", "gemma2"): 0.30,
        ("ioi", "llama3"): 0.26, ("arithmetic_subtraction", "llama3"): 0.27,
        ("mcqa", "qwen2.5"): 0.38, ("mcqa", "gemma2"): 1.47, ("mcqa", "llama3"): 1.69,
        ("arc_easy", "gemma2"): 1.01, ("arc_easy", "llama3"): 0.26,
        ("arc_challenge", "llama3"): 0.26,
    },
    "NAP-IG (CF)$^\\dagger$": {
        ("ioi", "gpt2"): 0.76, ("ioi", "qwen2.5"): 0.29, ("ioi", "gemma2"): 1.52,
        ("ioi", "llama3"): 0.42, ("arithmetic_subtraction", "llama3"): 0.39,
        ("mcqa", "qwen2.5"): 0.77, ("mcqa", "gemma2"): 1.71, ("mcqa", "llama3"): 1.87,
        ("arc_easy", "gemma2"): 1.53, ("arc_easy", "llama3"): 0.26,
        ("arc_challenge", "llama3"): 0.26,
    },
}

# NAP-IG reproduced: read from results/napig_repro_eval/
NAPIG_REPRO_DIR = "napig_repro_eval"

EAPIG_REPRO_DIR = "eapig_repro_eval"

EDGE_BASELINES = {
    "Random$^\\dagger$": {
        ("ioi", "gpt2"): 0.25, ("ioi", "qwen2.5"): 0.28, ("ioi", "gemma2"): 0.30,
        ("ioi", "llama3"): 0.25, ("arithmetic_subtraction", "llama3"): 0.25,
        ("mcqa", "qwen2.5"): 0.27, ("mcqa", "gemma2"): 0.32, ("mcqa", "llama3"): 0.26,
        ("arc_easy", "gemma2"): 0.32, ("arc_easy", "llama3"): 0.26,
        ("arc_challenge", "llama3"): 0.25,
    },
    "EAP-IG-inp (CF)$^\\dagger$": {
        ("ioi", "gpt2"): 1.85, ("ioi", "qwen2.5"): 1.63, ("ioi", "gemma2"): 3.20,
        ("ioi", "llama3"): 2.08, ("arithmetic_subtraction", "llama3"): 0.99,
        ("mcqa", "qwen2.5"): 1.16, ("mcqa", "gemma2"): 1.64, ("mcqa", "llama3"): 1.05,
        ("arc_easy", "gemma2"): 1.53, ("arc_easy", "llama3"): 1.04,
        ("arc_challenge", "llama3"): 0.98,
    },
    "UGS$^\\dagger$": {
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
    all_results = {}  # method_name -> {(task, model): cpr_auc}

    for method_name, results_dir, level, is_ours in OUR_METHODS:
        key = f"{method_name}_{level}"
        data = {}
        for task, model, _ in COLUMNS:
            v = load_cpr_auc(results_dir, task, model)
            if v is not None:
                data[(task, model)] = round(v, 2)
        all_results[key] = data

    # Find best per column per level
    node_ours = [(n, d, l, o) for n, d, l, o in OUR_METHODS if l == "node"]
    edge_ours = [(n, d, l, o) for n, d, l, o in OUR_METHODS if l == "edge"]

    def best_in_col(level):
        baselines = NODE_BASELINES if level == "node" else EDGE_BASELINES
        our = {f"{n}_{l}": all_results.get(f"{n}_{l}", {}) for n, _, l, _ in OUR_METHODS if l == level}
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

    def make_row(name, data, best_col, second_col, indent=False):
        vals = []
        for task, model, _ in COLUMNS:
            v = data.get((task, model))
            is_best = v is not None and best_col.get((task, model)) == v
            is_second = v is not None and not is_best and second_col.get((task, model)) == v
            vals.append(fmt(v, bold=is_best, underline=is_second))
        if indent:
            prefix = f"\\quad {name}" if "\\our" in name else f"\\quad \\textbf{{{name}}}"
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
    for method_name, _, _, _ in node_ours:
        key = f"{method_name}_node"
        lines.append(make_row(method_name, all_results.get(key, {}), best_node, second_node, indent=True))

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

    for name, data in EDGE_BASELINES.items():
        lines.append(make_row(name, data, best_edge, second_edge))
    lines.append("\\textbf{Ours} \\\\")
    for method_name, _, _, _ in edge_ours:
        key = f"{method_name}_edge"
        lines.append(make_row(method_name, all_results.get(key, {}), best_edge, second_edge, indent=True))

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
