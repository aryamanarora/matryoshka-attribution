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
    # Node level (log k-schedule = default). Swept methods use lr=0.05 (best); llama/ioi capped 200.
    # MAttr headline = SOFT top-k forward, log k. "+ hard" = sigmoid-STE hard forward.
    ("\\ourmethod{}", "topklog_lr_0.05", "node", "ours"),
    ("$+$ hard", "htklog_lr_0.05", "node", "ours"),
    ("$-$ $c_k$", "mib_node_detached_tau_log", "node", "ours"),
    ("$+$ hard bwd", "mib_node_bernoulli_reinforce_log", "node", "ours"),
    ("$+$ id-STE", "mib_node_identity_sgd_log", "node", "ours"),
    ("$+$ id-STE, Gumbel sel.", "mib_node_identity_gumbel_sgd_log", "node", "ours"),
    # Node level (uniform k-schedule = ablation). Swept -> lr=0.05.
    ("\\ourmethod{}", "final_node", "node", "uniform"),
    ("$+$ hard", "htk_lr_0.05", "node", "uniform"),
    ("$+$ hard, $+$ Gumbel sel.", "mib_node_hard_topk_gumbel", "node", "uniform"),
    ("$-$ $c_k$", "mib_node_detached_tau", "node", "uniform"),
    ("$+$ hard bwd", "mib_node_bernoulli_reinforce", "node", "uniform"),
    ("$+$ id-STE", "mib_node_identity_sgd", "node", "uniform"),
    ("$+$ id-STE, Gumbel sel.", "mib_node_identity_gumbel_sgd_uniform", "node", "uniform"),
    # Edge level (log k-schedule = default). Swept methods -> lr=0.05.
    ("\\ourmethod{}", "mib_edge_topk_log_lr05", "edge", "ours"),
    ("$+$ hard", "mib_edge_hard_topk_log_lr05", "edge", "ours"),
    ("$-$ $c_k$", "mib_edge_detached_tau", "edge", "ours"),
    ("$+$ hard bwd", "mib_edge_bernoulli_reinforce", "edge", "ours"),
    ("$+$ id-STE", "mib_edge_identity_sgd_log", "edge", "ours"),
    # Edge level (uniform k-schedule). Swept -> lr=0.05. (No soft-fwd uniform edge run.)
    ("$+$ hard", "mib_edge_hard_topk_uniform_lr05", "edge", "uniform"),
    ("$+$ id-STE", "mib_edge_identity_sgd_uniform", "edge", "uniform"),
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

# Mask-learning baselines, emitted under their own header in both sections.
# UGS (MIB's own mask baseline) is edge-level and only runs on gpt2-small/qwen, so it can
# never fill more than 3 of the 11 columns (docs/ugs_baseline.md). Edge Pruning is not tied
# to an architecture or a level and covers everything (docs/edge_pruning_baseline.md).
UGS_DIR = "ugs_eval"
PARTIAL_COVERAGE = {"UGS"}
MASK_NODE_BASELINES = {}
MASK_EDGE_BASELINES = {}

# A mask learner optimizes ONE operating point, and its target sparsity is the knob that
# decides where the circuit switches on -- so each budget is a separate row rather than a
# hidden default. (label, results dir); the unsuffixed dir is the runner's own default
# (0.9 node / 0.99 edge), the _s* dirs come from `run_edge_pruning.sbatch ... <sparsity>`.
EPRUN_SPARSITIES = [
    ("0.9", "eprun_eval"),
    ("0.95", "eprun_eval_s0.95"),
    ("0.99", "eprun_eval_s0.99"),
]


def eprun_rows(level):
    """[(display, {(task, model): AUC})], one row per target sparsity that has results."""
    rows = []
    for label, dirn in EPRUN_SPARSITIES:
        data = load_run_eval(dirn, f"EdgePruning_patching_{level}")
        if data:
            rows.append((f"Edge Pruning ($s{{=}}{label}$)", data))
    return rows


def load_run_eval(results_dir, sub):
    """{(task, model): CPR AUC} from a run_evaluation.py output folder."""
    data = {}
    for task, model, _ in COLUMNS:
        pkl = (RESULTS_BASE / results_dir / sub
               / f"{task.replace('_', '-')}_{model}_validation_abs-False.pkl")
        if pkl.exists():
            try:
                with open(pkl, "rb") as f:
                    data[(task, model)] = round(pickle.load(f)["area_under"], 2)
            except Exception:
                pass
    return data


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
        baselines = {**NODE_BASELINES, **MASK_NODE_BASELINES} if level == "node" \
            else {**EDGE_BASELINES, **MASK_EDGE_BASELINES}
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
        "NAP-IG": {("mcqa", "llama3")},
        "EAP-IG-inp (CF, repro)": {("arc_challenge", "llama3")},
    }

    def unifk(name):
        # log k is the default (unmarked); uniform k is the marked ablation. "+ unif k"
        # precedes other attributes, comma-separated; main row drops \ourmethod.
        if name.startswith("\\ourmethod"):
            return "$+$ unif $k$"
        return "$+$ unif $k$, " + name

    def row_avg(data):
        vs = [v for v in (data.get((t, m)) for t, m, _ in COLUMNS) if v is not None]
        return round(sum(vs) / len(vs), 2) if vs else None

    def section_avg_best(data_dicts):
        avs = sorted({a for a in (row_avg(d) for d in data_dicts) if a is not None}, reverse=True)
        return (avs[0] if avs else None, avs[1] if len(avs) > 1 else None)

    def make_row(name, data, best_col, second_col, indent=False, dagger=None,
                 avg_best=None, avg_second=None, suppress_avg=False):
        dcells = dagger if dagger is not None else DAGGER.get(name, set())
        vals = []
        for task, model, _ in COLUMNS:
            v = data.get((task, model))
            is_best = v is not None and best_col.get((task, model)) == v
            is_second = v is not None and not is_best and second_col.get((task, model)) == v
            cell = fmt(v, bold=is_best, underline=is_second)
            if v is not None and (task, model) in dcells:
                cell = "$^{\\dagger}$" + cell
            vals.append(cell)
        # A row that covers only some cells (UGS: 3 of 11) gets no average -- it would not
        # be comparable to the full-coverage rows.
        a = None if (suppress_avg or name in PARTIAL_COVERAGE) else row_avg(data)
        vals.append(fmt(a, bold=(a is not None and a == avg_best),
                        underline=(a is not None and a != avg_best and a == avg_second)))
        prefix = f"\\quad {name}" if indent else name
        return f"{prefix} & " + " & ".join(vals) + " \\\\"

    def opt_of(results_dir):
        # id-STE variants are trained with SGD; everything else with Adam.
        return "sgd" if "identity" in results_dir else "adam"

    # lr=0.05 swept dirs cap llama/ioi at 200 -> dagger just that cell for those rows.
    LR05_CAPPED = {"htklog_lr_0.05", "topklog_lr_0.05", "htk_lr_0.05"}
    LR05_DAGGER = {("ioi", "llama3")}

    def emit_ours(uniform_list, ours_list, level, best, second, avb, avs, dagger=None):
        # Split the "Ours" rows into two optimizer sets, each with a header.
        # Within a set: log-k = main rows (default, unmarked), then annotated uniform-k variants.
        for opt, label in [("adam", "\\ourmethod{}-Adam"), ("sgd", "\\ourmethod{}-SGD")]:
            rows_u = [(n, r, g) for n, r, _, g in uniform_list if opt_of(r) == opt]
            rows_o = [(n, r, g) for n, r, _, g in ours_list if opt_of(r) == opt]
            if not rows_u and not rows_o:
                continue
            lines.append(f"\\textbf{{{label}}} \\\\")
            for n, r, g in rows_o:
                dg = LR05_DAGGER if r in LR05_CAPPED else dagger
                lines.append(make_row(n, all_results.get(f"{n}_{level}_{g}", {}), best, second,
                                      indent=True, dagger=dg, avg_best=avb, avg_second=avs))
            for n, r, g in rows_u:
                dg = LR05_DAGGER if r in LR05_CAPPED else dagger
                lines.append(make_row(unifk(n), all_results.get(f"{n}_{level}_{g}", {}), best, second,
                                      indent=True, dagger=dg, avg_best=avb, avg_second=avs))

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

    # === Node-level section ===
    lines.append("\\midrule")
    lines.append(f"\\multicolumn{{{ncols + 2}}}{{l}}{{\\textit{{Node-level}}}} \\\\")
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
    NODE_BASELINES["NAP-IG"] = napig_repro

    # Additional baselines fetched from Tilde (node-level); each dir has one method subfolder
    EXTRA_NODE_BASELINES = [
        ("Conductance", "napig_local_eval", "EAP-IG-inputs-local_patching_node"),
        ("I$\\times$G", "ig1_eval",         "EAP-IG-inputs_patching_node"),
        ("RelP",        "relp_eval",        "RelP_patching_node"),
        ("RelP+QK",     "relp_qkgrad_eval", "RelP-qkgrad_patching_node"),
        ("AttnRLP",     "attnrlp_eval",     "AttnRLP_patching_node"),
        ("GIM",         "gim_eval",         "GIM_patching_node"),
    ]
    # Tilde baselines used a reduced subset for the llama3 cells only -> dagger those.
    TILDE_LLAMA3_DAGGER = {(t, m) for t, m, _ in COLUMNS if m == "llama3"}
    for disp, dirn, sub in EXTRA_NODE_BASELINES:
        data = {}
        for task, model, _ in COLUMNS:
            stask = task.replace("_", "-")
            pkl = RESULTS_BASE / dirn / sub / f"{stask}_{model}_validation_abs-False.pkl"
            if pkl.exists():
                try:
                    with open(pkl, "rb") as f:
                        d = pickle.load(f)
                    data[(task, model)] = round(d["area_under"], 2)
                except Exception:
                    pass
        NODE_BASELINES[disp] = data
        DAGGER[disp] = TILDE_LLAMA3_DAGGER

    # Mask learning at node level: Edge Pruning (all four models), one row per sparsity budget.
    # Its llama3 cells use the same --head 200 subset as the gradient baselines -> same dagger.
    for name, data in eprun_rows("node"):
        MASK_NODE_BASELINES[name] = data
        DAGGER[name] = TILDE_LLAMA3_DAGGER

    # Recompute best after adding repro
    best_node, second_node = best_in_col("node")
    node_dicts = list(NODE_BASELINES.values()) + list(MASK_NODE_BASELINES.values()) \
        + [all_results.get(f"{n}_node_{g}", {}) for n, _, _, g in node_uniform] \
        + [all_results.get(f"{n}_node_{g}", {}) for n, _, _, g in node_ours]
    avb, avs = section_avg_best(node_dicts)

    lines.append("\\textbf{Gradient attribution} \\\\")
    for name, data in NODE_BASELINES.items():
        lines.append(make_row(name, data, best_node, second_node, indent=True, avg_best=avb, avg_second=avs))
    if MASK_NODE_BASELINES:
        lines.append("\\textbf{Mask learning} \\\\")
        for name, data in MASK_NODE_BASELINES.items():
            lines.append(make_row(name, data, best_node, second_node, indent=True,
                                  avg_best=avb, avg_second=avs,
                                  suppress_avg=len(data) < len(COLUMNS)))
    emit_ours(node_uniform, node_ours, "node", best_node, second_node, avb, avs)

    # === Edge-level section ===
    lines.append("\\midrule")
    lines.append(f"\\multicolumn{{{ncols + 2}}}{{l}}{{\\textit{{Edge-level}}}} \\\\")

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

    # Mask learning at edge level: UGS (reg_lamb=0.001, gpt2/qwen only) + Edge Pruning
    ugs = load_run_eval(UGS_DIR, "UGS_patching_edge")
    if ugs:
        MASK_EDGE_BASELINES["UGS"] = ugs
    for name, data in eprun_rows("edge"):
        MASK_EDGE_BASELINES[name] = data
        DAGGER[name] = TILDE_LLAMA3_DAGGER

    best_edge, second_edge = best_in_col("edge")
    edge_dicts = list(EDGE_BASELINES.values()) + list(MASK_EDGE_BASELINES.values()) \
        + [all_results.get(f"{n}_edge_{g}", {}) for n, _, _, g in edge_uniform] \
        + [all_results.get(f"{n}_edge_{g}", {}) for n, _, _, g in edge_ours]
    eavb, eavs = section_avg_best(edge_dicts)

    # MAttr edge llama3 cells use a reduced eval subset (sphinx rerun) -> dagger.
    EDGE_LLAMA_DAGGER = {(t, m) for t, m, _ in COLUMNS if m == "llama3"}
    lines.append("\\textbf{Gradient attribution} \\\\")
    for name, data in EDGE_BASELINES.items():
        lines.append(make_row(name, data, best_edge, second_edge, indent=True, avg_best=eavb, avg_second=eavs))
    # Mask learners rank by a learned gate rather than a gradient, so they get their own header.
    if MASK_EDGE_BASELINES:
        lines.append("\\textbf{Mask learning} \\\\")
        for name, data in MASK_EDGE_BASELINES.items():
            lines.append(make_row(name, data, best_edge, second_edge, indent=True,
                                  avg_best=eavb, avg_second=eavs,
                                  suppress_avg=len(data) < len(COLUMNS)))
    emit_ours(edge_uniform, edge_ours, "edge", best_edge, second_edge, eavb, eavs, dagger=EDGE_LLAMA_DAGGER)

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
