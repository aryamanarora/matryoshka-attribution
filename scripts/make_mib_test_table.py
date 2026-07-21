"""Generate LaTeX table of MIB CPR AUC results on the TEST set.

Includes MIB paper baselines (test set) + our best method (test set).
Run from the repo root:
    uv run python scripts/make_mib_test_table.py
"""

import pickle
from pathlib import Path

RESULTS_BASE = Path("results")
OUTPUT = Path("paper/tabs/mib_test_results.tex")

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

# MIB paper baselines (from Table 1, test set)
NODE_BASELINES = {
    "Random": {
        ("ioi", "gpt2"): 0.25, ("ioi", "qwen2.5"): 0.28, ("ioi", "gemma2"): 0.30,
        ("ioi", "llama3"): 0.25, ("arithmetic_subtraction", "llama3"): 0.25,
        ("mcqa", "qwen2.5"): 0.27, ("mcqa", "gemma2"): 0.32, ("mcqa", "llama3"): 0.26,
        ("arc_easy", "gemma2"): 0.32, ("arc_easy", "llama3"): 0.26,
        ("arc_challenge", "llama3"): 0.25,
    },
    "NAP (CF)": {
        ("ioi", "gpt2"): 0.28, ("ioi", "qwen2.5"): 0.30, ("ioi", "gemma2"): 0.30,
        ("ioi", "llama3"): 0.26, ("arithmetic_subtraction", "llama3"): 0.27,
        ("mcqa", "qwen2.5"): 0.38, ("mcqa", "gemma2"): 1.47, ("mcqa", "llama3"): 1.69,
        ("arc_easy", "gemma2"): 1.01, ("arc_easy", "llama3"): 0.26,
        ("arc_challenge", "llama3"): 0.26,
    },
    "NAP-IG (CF)": {
        ("ioi", "gpt2"): 0.76, ("ioi", "qwen2.5"): 0.29, ("ioi", "gemma2"): 1.52,
        ("ioi", "llama3"): 0.42, ("arithmetic_subtraction", "llama3"): 0.39,
        ("mcqa", "qwen2.5"): 0.77, ("mcqa", "gemma2"): 1.71, ("mcqa", "llama3"): 1.87,
        ("arc_easy", "gemma2"): 1.53, ("arc_easy", "llama3"): 0.26,
        ("arc_challenge", "llama3"): 0.26,
    },
}

EDGE_BASELINES = {
    "EAP-IG-inp (CF)": {
        ("ioi", "gpt2"): 1.85, ("ioi", "qwen2.5"): 1.63, ("ioi", "gemma2"): 3.20,
        ("ioi", "llama3"): 2.08, ("arithmetic_subtraction", "llama3"): 0.99,
        ("mcqa", "qwen2.5"): 1.16, ("mcqa", "gemma2"): 1.64, ("mcqa", "llama3"): 1.05,
        ("arc_easy", "gemma2"): 1.53, ("arc_easy", "llama3"): 1.04,
        ("arc_challenge", "llama3"): 0.98,
    },
    "UGS": {
        ("ioi", "gpt2"): 0.97, ("ioi", "qwen2.5"): 0.98,
        ("mcqa", "qwen2.5"): 1.17,
    },
}

# Our method on test set (train on train, eval on test). The 3 headline MAttr variants at
# lr=0.05 (best LR from the sweep): (display name, node results dir).
OUR_NODE_METHODS = [   # MAttr = soft top-k fwd, log k; "+ hard" = sigmoid-STE hard forward
    ("\\ourmethod{}",                  "test_node_topk_log_lr05"),
    ("$+$ hard",                       "test_node_hard_topk_log_lr05"),
    ("$+$ unif $k$, $+$ hard",         "test_node_hard_topk_uniform_lr05"),
]
OUR_EDGE_METHODS = [   # same 3 variants at lr=0.05, edge level (test)
    ("\\ourmethod{}",                  "test_edge_topk_log_lr05"),
    ("$+$ hard",                       "test_edge_hard_topk_log_lr05"),
    ("$+$ unif $k$, $+$ hard",         "test_edge_hard_topk_uniform_lr05"),
]


def load_cpr_auc(results_dir, task, model):
    pkl_path = RESULTS_BASE / results_dir / f"{task}_{model}_test.pkl"
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
    # Load our test results: 3 node variants (name -> {cell: cpr}) + 1 edge.
    ours_nodes = {}
    for name, d in OUR_NODE_METHODS:
        data = {}
        for task, model, _ in COLUMNS:
            v = load_cpr_auc(d, task, model)
            if v is not None:
                data[(task, model)] = round(v, 2)
        ours_nodes[name] = data
    ours_edges = {}
    for name, d in OUR_EDGE_METHODS:
        data = {}
        for task, model, _ in COLUMNS:
            v = load_cpr_auc(d, task, model)
            if v is not None:
                data[(task, model)] = round(v, 2)
        ours_edges[name] = data

    # Best per column
    def find_best(baselines, ours_list):
        best = {}
        second = {}
        for task, model, _ in COLUMNS:
            vals = []
            for data in list(baselines.values()) + list(ours_list):
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

    best_node, second_node = find_best(NODE_BASELINES, list(ours_nodes.values()))
    best_edge, second_edge = find_best(EDGE_BASELINES, list(ours_edges.values()))

    def row_avg(data):
        vs = [v for v in (data.get((t, m)) for t, m, _ in COLUMNS) if v is not None]
        return round(sum(vs) / len(vs), 2) if vs else None

    def section_avg_best(data_dicts):
        avs = sorted({a for a in (row_avg(d) for d in data_dicts) if a is not None}, reverse=True)
        return (avs[0] if avs else None, avs[1] if len(avs) > 1 else None)

    def make_row(name, data, best_col, second_col, dagger=None, avg_best=None, avg_second=None, indent=False):
        dcells = dagger or set()
        vals = []
        for task, model, _ in COLUMNS:
            v = data.get((task, model))
            is_best = v is not None and best_col.get((task, model)) == v
            is_second = v is not None and not is_best and second_col.get((task, model)) == v
            cell = fmt(v, bold=is_best, underline=is_second)
            if v is not None and (task, model) in dcells:
                cell = "$^{\\dagger}$" + cell
            vals.append(cell)
        a = row_avg(data)
        vals.append(fmt(a, bold=(a is not None and a == avg_best),
                        underline=(a is not None and a != avg_best and a == avg_second)))
        prefix = f"\\quad {name}" if indent else name
        return f"{prefix} & " + " & ".join(vals) + " \\\\"

    # Generate LaTeX
    ncols = len(COLUMNS)
    lines = []
    lines.append("\\begin{adjustbox}{max width=\\textwidth}")
    lines.append("\\begin{tabular}{l" + "r" * ncols + "@{\\quad}r}")
    lines.append("\\toprule")
    lines.append("& \\multicolumn{4}{c}{IOI} & Arithmetic & \\multicolumn{3}{c}{MCQA} & \\multicolumn{2}{c}{ARC (E)} & ARC (C) & \\\\")
    lines.append("\\cmidrule(lr){2-5} \\cmidrule(lr){6-6} \\cmidrule(lr){7-9} \\cmidrule(lr){10-11} \\cmidrule(lr){12-12}")
    header = "\\textbf{Method} & " + " & ".join(h for _, _, h in COLUMNS) + " & \\textbf{Avg} \\\\"
    lines.append(header)

    # Node level
    lines.append("\\midrule")
    lines.append(f"\\multicolumn{{{ncols + 2}}}{{l}}{{\\textit{{Node-level}}}} \\\\")
    navb, navs = section_avg_best(list(NODE_BASELINES.values()) + list(ours_nodes.values()))
    for name, data in NODE_BASELINES.items():
        lines.append(make_row(name, data, best_node, second_node, avg_best=navb, avg_second=navs))
    for name, data in ours_nodes.items():
        lines.append(make_row(name, data, best_node, second_node, avg_best=navb, avg_second=navs))

    # Edge level
    lines.append("\\midrule")
    lines.append(f"\\multicolumn{{{ncols + 2}}}{{l}}{{\\textit{{Edge-level}}}} \\\\")
    eavb, eavs = section_avg_best(list(EDGE_BASELINES.values()) + list(ours_edges.values()))
    for name, data in EDGE_BASELINES.items():
        lines.append(make_row(name, data, best_edge, second_edge, avg_best=eavb, avg_second=eavs))
    # MAttr edge llama3 cells use a reduced (200-example) subset -> dagger.
    EDGE_DAGGER = {(t, m) for t, m, _ in COLUMNS if m == "llama3"}
    for name, data in ours_edges.items():
        lines.append(make_row(name, data, best_edge, second_edge, dagger=EDGE_DAGGER, avg_best=eavb, avg_second=eavs))

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
