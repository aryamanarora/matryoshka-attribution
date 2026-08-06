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

# Our method on test set (train on train, eval on test), at lr=0.05 (best LR from the sweep).
#
# SOFT FORWARD ONLY. The hard sigmoid-STE variants (test_*_hard_topk_*_lr05) are deliberately
# not here: the headline method is the soft forward, and the hard forward is an ablation whose
# place is the validation tables, which carry the full ablation grid. They stay on disk and in
# make_mib_table.py -- dropping them here removes two rows from one table, not any result.
#
# So the test table shows the k-schedule contrast at a fixed (soft) forward: log k vs uniform k.
# The uniform-k dirs come from submit_softuni_lr05.sh -- soft + uniform k had never been run at
# lr=0.05 on either split, nor at edge level at all, so the row could not simply be pointed at
# an existing dir.
OUR_NODE_METHODS = [
    ("\\ourmethod{}",          "test_node_topk_log_lr05"),
    ("$+$ unif $k$",           "test_node_topk_uniform_lr05"),
]
OUR_EDGE_METHODS = [
    ("\\ourmethod{}",          "test_edge_topk_log_lr05"),
    ("$+$ unif $k$",           "test_edge_topk_uniform_lr05"),
]


# Mask-learning baseline: Node Pruning (Bhaskar et al., 2024's recipe at node granularity) at
# the headline budget s=0.9 -- the one that wins CPR AUC on validation (1.00 vs 0.96 / 0.91).
# Only this budget is carried to test; the other two stay a validation-only sparsity sweep.
#
# TWO caveats that make this row not quite like its neighbours, both deliberate:
#  1. It is scored by MIB's run_evaluation.py, while the \ourmethod{} rows come from our
#     eval_mib.py. Those harnesses do NOT agree cell-for-cell (worst on Gemma), so a small
#     gap between this row and a MAttr row is inside harness noise.
#  2. Its cells are full-split, including llama3. The validation table daggers llama3 because
#     run_variants.sh caps it at 200 examples there; test splits are <=1188 so nothing is
#     capped and no dagger is owed.
#  3. Only the best budget appears (make_mib_table.EPRUN_BEST_SPARSITY); the validation tables
#     carry the full budget sweep and the logit-diff objective ablation. Note this row trains
#     on Edge Pruning's KL, not MAttr's logit-diff -- see EPRUN_SPARSITIES on that confound.
import make_mib_table as _M   # noqa: E402  (label/dir are defined there, one source of truth)

NODE_PRUNING = (_M.eprun_label("node", _M.EPRUN_BEST_SPARSITY[0]),
                _M.EPRUN_BEST_SPARSITY[1], "EdgePruning_patching_node")

# Gradient node baselines WE ran (unlike the NODE_BASELINES literals above, which are
# transcribed from MIB's Table 1). Same circuits as the validation table -- both methods
# attribute on the train split, so the test pass is eval-only and the circuit is unchanged
# across the two splits (MIB-circuit-track/run_gim_relpqk_test.sh).
#
# These carry the same harness caveat as the Node Pruning row: they are scored by MIB's
# run_evaluation.py while the \ourmethod{} rows come from our eval_mib.py, and the two do not
# agree cell-for-cell (worst on Gemma). A small gap either way is inside harness noise.
GRAD_NODE_BASELINES = [
    ("GIM",      "gim_eval",         "GIM_patching_node"),
    ("RelP$+$QK", "relp_qkgrad_eval", "RelP-qkgrad_patching_node"),
]


def load_run_eval_cpr(results_dir, sub, task, model):
    """CPR AUC from a run_evaluation.py output pkl (baseline layout, dashed task names)."""
    pkl_path = RESULTS_BASE / results_dir / sub / f"{task.replace('_', '-')}_{model}_test_abs-False.pkl"
    if not pkl_path.exists():
        return None
    try:
        with open(pkl_path, "rb") as f:
            return pickle.load(f)["area_under"]
    except Exception:
        return None


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



# Dirs produced by a wave that ran in the L2A venv and whose gemma2 cells are therefore wrong
# until scripts/reeval_gemma_mib.py has been run over them. Gated on the stamp that script
# writes, so an entry clears itself once the re-eval lands -- nothing here has to be pruned by
# hand, and a dir that never gets re-evaluated never silently publishes bad Gemma numbers.
#
# ADD EVERY NEW DIR HERE at the same time you add it to reeval_gemma_mib.py's DIRS. There is no
# way to detect the condition from the pkls: a re-evaluated pkl and an L2A-venv pkl are both
# just a pkl, and the numbers differ by less than the amount that would look obviously wrong.
# (mtime nearly works and was tried; it misreports any dir whose non-gemma cells were topped up
# after the re-eval, which is most of the edge dirs.)
GEMMA_REEVAL_PENDING = {
    "mib_node_topk_uniform_lr05", "test_node_topk_uniform_lr05",
    "mib_edge_topk_uniform_lr05", "test_edge_topk_uniform_lr05",
}
GEMMA_TASKS = ("ioi", "mcqa", "arc_easy")


def gemma_unstamped(d, level, split):
    """Gemma tasks in dir `d` still awaiting re-eval under the MIB venv."""
    if d not in GEMMA_REEVAL_PENDING:
        return []
    return [t for t in GEMMA_TASKS
            if not (RESULTS_BASE / d / f".gemma_reeval_{level}_{split}_{t}").exists()]


def complete_or_skip(name, level, d, data, split="test"):
    """Hold one of OUR rows until every cell has landed. Returns False to skip it.

    Stricter than the baseline rows, which print with a suppressed Avg when partial, and
    deliberately so. A baseline's missing cell can be a real limitation (UGS genuinely has no
    number for most columns), so the dashes are informative. Ours are always run to completion,
    so a gap is only ever a pending job -- and a partial row of ours is actively misleading
    twice over: its Avg is not comparable to the row above it, and its gemma2 cells come out of
    the L2A venv's broken Gemma-2 forward until reeval_gemma_mib.py has been run over the dir,
    which by construction cannot have happened while jobs are still landing in it.
    """
    if len(data) < len(COLUMNS):
        why = "no test cells" if not data else f"only {len(data)}/{len(COLUMNS)} test cells"
        print(f"SKIP {name} ({level}): {why} in results/{d} (jobs still pending)")
        return False
    pend = gemma_unstamped(d, level, split)
    if pend:
        print(f"SKIP {name} ({level}): results/{d} is complete but its gemma2 cells "
              f"({', '.join(pend)}) have not been re-evaluated under the MIB venv yet -- "
              f"run scripts/reeval_gemma_mib.py")
        return False
    return True


def main():
    # Load our test results: 3 node variants (name -> {cell: cpr}) + 1 edge.
    ours_nodes = {}
    for name, d in OUR_NODE_METHODS:
        data = {}
        for task, model, _ in COLUMNS:
            v = load_cpr_auc(d, task, model)
            if v is not None:
                data[(task, model)] = round(v, 2)
        if not complete_or_skip(name, "node", d, data):
            continue
        ours_nodes[name] = data
    # Node Pruning row (empty dict -> row is skipped entirely, not printed as all-dashes)
    np_name, np_dir, np_sub = NODE_PRUNING
    node_pruning = {}
    for task, model, _ in COLUMNS:
        v = load_run_eval_cpr(np_dir, np_sub, task, model)
        if v is not None:
            node_pruning[(task, model)] = round(v, 2)
    mask_nodes = {np_name: node_pruning} if node_pruning else {}
    if node_pruning and len(node_pruning) < len(COLUMNS):
        print(f"WARNING: {np_name} has {len(node_pruning)}/{len(COLUMNS)} test cells; "
              f"missing {[f'{t}/{m}' for t, m, _ in COLUMNS if (t, m) not in node_pruning]}")

    # GIM / RelP+QK, same loader and same "no cells -> no row" rule as Node Pruning: a row of
    # eleven dashes reads as "the method scored nothing", not "the jobs have not landed yet".
    grad_nodes = {}
    for name, d, sub in GRAD_NODE_BASELINES:
        data = {}
        for task, model, _ in COLUMNS:
            v = load_run_eval_cpr(d, sub, task, model)
            if v is not None:
                data[(task, model)] = round(v, 2)
        if not data:
            print(f"SKIP {name}: no test cells in results/{d}/{sub} (jobs still pending)")
            continue
        if len(data) < len(COLUMNS):
            print(f"WARNING: {name} has {len(data)}/{len(COLUMNS)} test cells; "
                  f"missing {[f'{t}/{m}' for t, m, _ in COLUMNS if (t, m) not in data]}")
        grad_nodes[name] = data

    ours_edges = {}
    for name, d in OUR_EDGE_METHODS:
        data = {}
        for task, model, _ in COLUMNS:
            v = load_cpr_auc(d, task, model)
            if v is not None:
                data[(task, model)] = round(v, 2)
        if not complete_or_skip(name, "edge", d, data):
            continue
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

    best_node, second_node = find_best(NODE_BASELINES,
                                       list(grad_nodes.values()) + list(mask_nodes.values())
                                       + list(ours_nodes.values()))
    best_edge, second_edge = find_best(EDGE_BASELINES, list(ours_edges.values()))

    def row_avg(data):
        vs = [v for v in (data.get((t, m)) for t, m, _ in COLUMNS) if v is not None]
        return round(sum(vs) / len(vs), 2) if vs else None

    def section_avg_best(data_dicts):
        avs = sorted({a for a in (row_avg(d) for d in data_dicts) if a is not None}, reverse=True)
        return (avs[0] if avs else None, avs[1] if len(avs) > 1 else None)

    def make_row(name, data, best_col, second_col, dagger=None, avg_best=None, avg_second=None,
                 indent=False, suppress_avg=False):
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
        # suppress_avg is for a row that is partial RELATIVE TO ITS SECTION -- e.g. a Node
        # Pruning sweep still running. It is NOT keyed off "missing any cell": every
        # edge-level row is missing the same two ARC/llama3 cells (no edge circuits there),
        # and those averages are comparable to each other, so blanket-dashing them is wrong.
        a = None if suppress_avg else row_avg(data)
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
    navb, navs = section_avg_best(list(NODE_BASELINES.values()) + list(grad_nodes.values())
                                  + list(mask_nodes.values()) + list(ours_nodes.values()))
    for name, data in NODE_BASELINES.items():
        lines.append(make_row(name, data, best_node, second_node, avg_best=navb, avg_second=navs))
    for name, data in grad_nodes.items():
        # Same Avg rule as the mask rows below: every node cell exists, so a gap is an
        # unfinished job rather than something the method cannot do.
        lines.append(make_row(name, data, best_node, second_node, avg_best=navb, avg_second=navs,
                              suppress_avg=len(data) < len(COLUMNS)))
    for name, data in mask_nodes.items():
        # Node level has all 11 cells, so anything missing here is an unfinished job rather
        # than a cell the method cannot do -- dash the Avg until the sweep completes.
        lines.append(make_row(name, data, best_node, second_node, avg_best=navb, avg_second=navs,
                              suppress_avg=len(data) < len(COLUMNS)))
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
