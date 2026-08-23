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
#
# The last two rows are the SGD optimiser arm, added 2026-08-23. Together with the two Adam rows
# above them they form a 2x2 over {Adam, SGD} x {log k, uniform k}.
#
# THE TWO SGD ROWS ARE AT DIFFERENT LEARNING RATES ON PURPOSE (1.0 and 3.0), and that is not a
# sloppiness to be tidied up into one number. MAttr+SGD is LR-invariant by construction (zero
# init, no momentum), so its useful LR scales like n/k, and the uniform-k schedule additionally
# has E[alpha(1-alpha)] = 1/6 against log-k's 1/(2 ln n) -- about 3x larger, which moves the
# optimum by roughly that factor on its own. Each row therefore sits at its own block-argmax
# from the LR sweep. Forcing both to a single LR would compare one tuned row against one
# detuned one; see the edge LR sweep, where reading SGD at the node table's LR made a
# competitive optimiser look broken.
OUR_NODE_METHODS = [
    ("\\ourmethod{}",          "test_node_topk_log_lr05"),
    ("$+$ unif $k$",           "test_node_topk_uniform_lr05"),
    ("$+$ SGD",                "test_node_softlog_sgd_lr_1.0"),
    ("$+$ SGD, unif $k$",      "test_node_softuni_sgd_lr_3.0"),
]
# Edge level mirrors the node block, including the SGD arm, so the two levels of this table and
# the edge section of the VALIDATION table (paper/tabs/mib_results.tex) all agree about which
# rows exist. The validation table has carried a full \ourmethod{}-SGD block at edge level for a
# while; the test table carrying only the two Adam rows was the same table-disagreement defect
# that the node SGD rows above were added to fix, one granularity down.
#
# The two dirs do not exist yet -- the runs are submitted by scripts/submit_test_edge_sgd.sh
# (2 configs x 11 cells). Declaring them here before they land is safe and self-filling:
# complete_or_skip() prints a SKIP and omits the row until all 11 cells are present, so this
# table stays correct in the meantime and gains the rows the moment the wave finishes.
#
# SAME OWN-BEST-LR POLICY as the node rows above (1.0 for log k, 3.0 for uniform k). One caveat
# specific to edge level: these two LRs are the NODE optima carried over, not an edge-level
# argmax -- submit_mib_edge_soft_sgd.sh:10-11 did the same for the two validation dirs these
# mirror. Re-tuning here and not there would mean the two tables report different
# hyperparameters under one row label, which is worse than untuned-but-consistent.
OUR_EDGE_METHODS = [
    ("\\ourmethod{}",          "test_edge_topk_log_lr05"),
    ("$+$ unif $k$",           "test_edge_topk_uniform_lr05"),
    ("$+$ SGD",                "test_edge_softlog_sgd_lr_1.0"),
    ("$+$ SGD, unif $k$",      "test_edge_softuni_sgd_lr_3.0"),
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

# Plain "Node Pruning", NOT eprun_label()'s "Node Pruning (s=0.5, logit-diff)". That parenthetical
# earns its place in the validation table, where several budgets and both objectives appear as
# separate rows and the label is what tells them apart. Here exactly ONE configuration is carried
# to test, so the suffix distinguishes the row from nothing -- it just states a hyperparameter
# next to a set of baselines whose own hyperparameters are not in their labels.
#
# The dir still comes from EPRUN_BEST_SPARSITY, so which configuration this is stays defined in
# make_mib_table.py; only the display name is overridden. Anyone needing the setting can read
# it there or in this comment: s=0.5, logit-diff objective.
NODE_PRUNING = (_M.EPRUN_NAME["node"],
                _M.EPRUN_BEST_SPARSITY[1], "EdgePruning_patching_node")

# Gradient node baselines WE ran (unlike the NODE_BASELINES literals above, which are
# transcribed from MIB's Table 1). Same circuits as the validation table -- both methods
# attribute on the train split, so the test pass is eval-only and the circuit is unchanged
# across the two splits (MIB-circuit-track/run_gim_relpqk_test.sh).
#
# These carry the same harness caveat as the Node Pruning row: they are scored by MIB's
# run_evaluation.py while the \ourmethod{} rows come from our eval_mib.py, and the two do not
# agree cell-for-cell (worst on Gemma). A small gap either way is inside harness noise.
#
# *** GIM'S ROW IS THE CORRECTED RUN AS OF 2026-08-23. Do NOT repoint it at gim_nomlp_eval. ***
# The 11-cell test wave of the CORRECTED (post scale_mlp_gate) circuits landed that day and this
# entry filled itself with no edit, exactly as the paragraph below predicted; the row now reads
# 1.36 1.33 1.44 1.83 1.14 1.23 1.55 1.02 1.48 1.03 1.02, avg 1.31 -- consistent with the
# corrected validation row's 1.32 and with AttnLRP, which is the check that it is the right run.
#
# The history matters because the failure mode is silent. Until 2026-08-10 this table shipped a
# GIM row that WAS the pre-scale_mlp_gate BUGGY run, matching gim_nomlp_eval cell-for-cell
# (1.36 0.71 0.78 0.25 1.12 0.25 1.24 0.99 1.34 1.11 1.06, avg 0.93). Those buggy pkls used to
# sit in gim_eval, so the row generated cleanly and then FROZE in the .tex while the dirs were
# reorganised underneath it -- the paper showed buggy GIM on test beside corrected GIM on
# validation (1.32) under one label, the confusion make_mib_accauc_table.py:60 warns about. The
# buggy wave is now quarantined in results/_stale_gim_nomlp/ (same treatment as the TL 3.2.1
# wave in _stale_tl321). Note ioi/gpt2 is 1.36 in BOTH runs, so that one cell cannot tell them
# apart; qwen2.5 (0.71 buggy vs 1.33 corrected) is the discriminating column.
#
# AttnLRP was validated on all 12 cells but had never been evaluated on test at all, so the
# strongest gradient baseline in the validation table was simply absent from this one. Its 11
# test evals come from MIB-circuit-track/run_attnlrp_test.sh: EVAL-ONLY, reusing the train-split
# circuits in results/attnlrp that the validation row already scored, so the row below and the
# validation row describe the SAME circuits on two splits rather than two separate attributions.
# Full-split including llama3, per the no-dagger rule documented above.
#
# The last four entries are the QUADRATURE LADDER for NAP-IG (added 2026-08-23), all four scoring
# circuits attributed on the train split by MIB-circuit-track/run_stepless_test.sh -- eval-only,
# same circuits the validation wave scored, so these rows and their validation twins describe the
# same circuits on two splits.
#
# What the ladder varies is ONLY how the input-path integral is quadratured:
#   m=1 grid   -- the right-endpoint rule at m=1 degenerates to alpha=1, the clean input, so this
#                 row IS input x gradient. It is the COMPUTE-MATCHED CONTROL for the MC row, not
#                 a weaker setting of it: one forward+backward per batch either way.
#
#                 *** AND IT IS THE SAME ESTIMATOR AS THE "NAP (CF)" ROW ABOVE IT. ***
#                 get_scores_eap_ig at steps=1 sets new_input = corrupted + (1/1)(clean -
#                 corrupted) = the clean input (attribute_node.py:239), so its forward is the
#                 plain clean forward and its gradient is taken at the clean input -- identical
#                 to get_scores_eap (attribute_node.py:182-185), which is attribution patching.
#                 There is no separate NAP method in attribute_node's dispatch list; NAP *is*
#                 EAP, and EAP-IG-inputs at m=1 collapses onto it. Hence the label: this row is
#                 our run of NAP, not a second method.
#
#                 THE TWO ROWS DO NOT AGREE NUMERICALLY AND THAT IS UNEXPLAINED. NAP (CF) is
#                 transcribed from MIB's Table 1; this row is ours. They match on the IOI cells
#                 (both at chance) and diverge on mcqa/llama3 (1.69 vs 0.38), mcqa/gemma2
#                 (1.47 vs 0.92), arc_easy/gemma2 (1.01 vs 1.25) and arc_challenge/llama3
#                 (0.26 vs 0.59) -- in BOTH directions, so it is not a scaling factor or a sign
#                 convention. Either MIB's NAP differs from EAP in some detail we have not
#                 found, or the two harnesses disagree by more than the "worst on Gemma" caveat
#                 elsewhere in this file allows. Do not present the two rows as independent
#                 methods, and do not quietly drop one: the gap is a reproduction finding.
#   m=5 grid   -- Hanna et al.'s defended default (COLM'24 App. C), 5x cost.
#   m=30 grid  -- converged reference, 30x cost.
#   MC alpha   -- alpha ~ U(0,1) drawn PER EXAMPLE, unbiased for the same integral at every m,
#                 here at m=1. Same cost as the I x G row.
#
# So the honest reading of these four rows is a cost-vs-quality ladder in which the first and
# last are free and the middle two are not. Do NOT reorder them by score; the ordering is the
# cost ordering and that is the point.
#
# The MC row's subfolder is EAP-IG-inputs-mc_patching_node, not EAP-IG-inputs_patching_node --
# run_stepless_test.sh passes --method EAP-IG-inputs-mc precisely so its output does not land in
# the grid arms' folder and overwrite them.
#
# These are distinct from the "NAP-IG (CF)" literal above, which is transcribed from MIB's
# Table 1 and uses a counterfactual ablation; everything here is patching, our own runs.
GRAD_NODE_BASELINES = [
    ("AttnLRP",  "attnlrp_eval",     "AttnLRP_patching_node"),
    ("GIM",      "gim_eval",         "GIM_patching_node"),
    ("RelP$+$QK", "relp_qkgrad_eval", "RelP-qkgrad_patching_node"),
    ("NAP $=$ I$\\times$G (ours)", "ig1_test",        "EAP-IG-inputs_patching_node"),
    ("NAP-IG ($m{=}5$)",          "napig_ref_test",  "EAP-IG-inputs_patching_node"),
    ("NAP-IG ($m{=}30$)",         "napig30_test",    "EAP-IG-inputs_patching_node"),
    ("NAP-IG (MC $\\alpha$)",     "napig_mc_test",   "EAP-IG-inputs-mc_patching_node"),
]

# DBM, the other mask-learning baseline (pyvene's SigmoidMaskIntervention). Same loader and
# same layout as Node Pruning above -- run_evaluation.py output, EdgePruning_patching_node --
# and the same "no cells -> no row" rule, so an entry can be declared here before its jobs land
# and it appears on the next regeneration with no edit.
#
# ONE row, the tuned one: lr=0.3 (best Avg CPR of {0.001..1.0} on validation) and lambda=6.0
# (best of {0, 0.2, 0.6, 2.0, 6.0, 20.0}; validation Avg CPR 1.50 vs 1.31 unpenalised), both
# peaks interior to their grids.
#
# This file used to carry the unpenalised run as "DBM" and the penalised one as "DBM $+$ L1",
# on the argument that folding the penalty into "DBM" attributes to pyvene a term it does not
# have. That split was dropped on 2026-08-07: pyvene's own masking tutorial and Boundless DAS
# both put an L1 on the mask, so the penalty is the library's practice even though it is not
# the class default, and naming the unpenalised run "DBM" gives the baseline its weakest
# operating point. See the longer note above SIGMOID_MASK_ROWS in make_mib_table.py, which
# this must agree with -- the two tables naming the same baseline after different recipes is
# the exact failure that motivated the change. lambda=0 is still reported, as the anchor of the
# lambda sweep in tabs/lr_sweep.tex and as a point in the scatter's DBM series.
#
# Caveat for the prose: lambda also drops achieved density 0.56 -> 0.30, so the CPR gain is
# confounded with the sparsity change and this sweep alone does not establish that the penalty
# *helps*; it establishes the best-tuned operating point.
MASK_NODE_BASELINES = [
    ("DBM", "eprun_eval_ld_sig_lr0.3_l16.0", "EdgePruning_patching_node"),
]


# --- family assignment for the LITERAL baseline dicts -------------------------------------
# NODE_BASELINES / EDGE_BASELINES are plain name->cells dicts with no family field, so the
# grouping has to be declared out here. The rule is deliberately EXHAUSTIVE-BY-DEFAULT: only
# controls and mask methods are named, and everything else falls through to "Gradient-based".
#
# Why that direction. The obvious implementation -- listing the members of each group -- means a
# baseline added to NODE_BASELINES later matches no group and is silently dropped from the table,
# which is invisible in the output because the row simply is not there. Defaulting to a family
# makes the failure mode "a new baseline appears under a possibly-wrong heading", which a reader
# notices. If you add a mask-learning or control baseline, name it here; if you add a gradient
# one, do nothing.
CONTROL_NAMES = ("Random",)
MASK_BASELINE_NAMES = ("UGS",)   # UGS learns edge masks; it is not a gradient attribution.


def classify(baselines, family):
    """Members of `baselines` belonging to `family`, in the dict's own (declaration) order.

    `family` is "gradient" or "mask"; controls are excluded from both and emitted ungrouped.
    """
    out = []
    for name, data in baselines.items():
        if name in CONTROL_NAMES:
            continue
        fam = "mask" if name in MASK_BASELINE_NAMES else "gradient"
        if fam == family:
            out.append((name, data))
    return out


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




gemma_unstamped = _M.gemma_unstamped   # single source of truth, see make_mib_table.py


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
        print(f"SKIP {name} ({level}): {why} in results/{d} (no results on disk -- job pending, or never launched)")
        return False
    pend = gemma_unstamped(d, level, split)
    if pend:
        print(f"SKIP {name} ({level}): results/{d} is complete but its gemma2 cells "
              f"({', '.join(pend)}) have not been re-evaluated under the MIB venv yet -- "
              f"run scripts/reeval_gemma_mib.py")
        return False
    return True


# A baseline row needs at least this fraction of the columns before it is printed at all.
#
# The old rule was `if not data: skip`, i.e. only a completely empty dir was held back, with the
# stated reason that "a row of eleven dashes reads as 'the method scored nothing', not 'the jobs
# have not landed yet'". That reason does not stop applying at one cell. GIM is the case that
# exposed it: its 11-job test wave went out, ioi/gpt2 landed first, and the very next regeneration
# produced `GIM & 1.36 & --- & ... & ---`, a row that is WORSE than the all-dashes one the guard
# was written to prevent -- a real number next to ten dashes invites the reader to conclude the
# dashes are failures rather than pending jobs, and 1.36 sits right in the range where GIM looks
# like it beat several complete rows on the one task it has.
#
# Half is the threshold because that is where the row stops being a progress report and starts
# being a comparison: with most columns present the dashes read as gaps in an otherwise real row
# (which is what they are for UGS, whose missing cells are a genuine limitation, not a queue),
# and the Avg is suppressed anyway whenever a row is partial, so a mostly-complete row cannot be
# misread as a complete average. Below half there is not enough row left to carry that reading.
#
# This is deliberately LOOSER than complete_or_skip, which holds OUR rows until 11/11: a
# baseline's gap can be a real limitation, ours is only ever a pending job.
MIN_BASELINE_FRAC = 0.5


def baseline_or_skip(name, where, data):
    """Print/skip decision for a BASELINE row; warns about stragglers. False means skip."""
    if len(data) < MIN_BASELINE_FRAC * len(COLUMNS):
        why = "no test cells" if not data else f"only {len(data)}/{len(COLUMNS)} test cells"
        print(f"SKIP {name}: {why} in results/{where} "
              f"(no results on disk -- job pending, or never launched)")
        return False
    if len(data) < len(COLUMNS):
        print(f"WARNING: {name} has {len(data)}/{len(COLUMNS)} test cells; "
              f"missing {[f'{t}/{m}' for t, m, _ in COLUMNS if (t, m) not in data]}")
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
    mask_nodes = ({np_name: node_pruning}
                  if baseline_or_skip(np_name, f"{np_dir}/{np_sub}", node_pruning) else {})

    # DBM rows, same rule as Node Pruning, via the shared guard below.
    for name, d, sub in MASK_NODE_BASELINES:
        data = {}
        for task, model, _ in COLUMNS:
            v = load_run_eval_cpr(d, sub, task, model)
            if v is not None:
                data[(task, model)] = round(v, 2)
        if not baseline_or_skip(name, f"{d}/{sub}", data):
            continue
        mask_nodes[name] = data

    # GIM / RelP+QK, same loader and same rule.
    grad_nodes = {}
    for name, d, sub in GRAD_NODE_BASELINES:
        data = {}
        for task, model, _ in COLUMNS:
            v = load_run_eval_cpr(d, sub, task, model)
            if v is not None:
                data[(task, model)] = round(v, 2)
        if not baseline_or_skip(name, f"{d}/{sub}", data):
            continue
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

    def level_header(text):
        return f"\\multicolumn{{{ncols + 2}}}{{l}}{{\\textit{{{text}}}}} \\\\"

    def group_header(text):
        """Second-level heading INSIDE a level section (Gradient-based / Mask-based / MAttr).

        MATCHES make_mib_table.py (the validation table), which is the layout to copy rather
        than invent against: a family heading there is a bare bold cell in the FIRST COLUMN
        (`\\textbf{Gradient attribution} \\\\`, make_mib_table.py:802/814) and the rows under it
        carry the \\quad via make_row(indent=True). The hierarchy is therefore produced by
        indenting the ROWS, not the heading.

        The first version of this function wrapped the heading in \\multicolumn AND prefixed it
        with \\quad, which put heading and rows at the SAME indent and destroyed the nesting --
        the level headers (\\textit{Node-level}) are the ones that use \\multicolumn, and copying
        their form one level down is what broke it.

        Emitted only when the group has at least one row: a heading over zero rows reads as
        "this family scored nothing", which is exactly the confusion the "no cells -> no row"
        rule elsewhere in this file exists to prevent. GIM is the live case -- it has no test
        pkls (see the note above GRAD_NODE_BASELINES), so it contributes no row, and if the
        whole gradient family were ever in that state the heading must vanish with it.
        """
        return f"{text} \\\\"

    # Bold, matching the validation table's family headings. The MAttr heading says "(ours)" so
    # it is not verbatim identical to the \ourmethod{} ROW directly beneath it -- the heading
    # names the family, the row names the headline configuration, and the three "$+$" rows below
    # are ablations OF that row rather than siblings of it.
    #
    # NAMES DIFFER FROM THE VALIDATION TABLE ON PURPOSE-ISH: that one says "Gradient attribution"
    # and "Mask learning". These are the names asked for. If the two tables should agree, change
    # them here (one line) rather than renaming the validation table's, which several captions
    # may refer to.
    GRAD_H, MASK_H = "\\textbf{Gradient-based}", "\\textbf{Mask-based}"
    OURS_H = "\\textbf{\\ourmethod{} (ours)}"

    def emit(group, rows, best, second, avb, avs, dagger=None, suppress_partial=True):
        """One group heading + its rows, indented one level under the heading."""
        if not rows:
            return
        if group:
            lines.append(group_header(group))
        for name, data in rows:
            lines.append(make_row(name, data, best, second, dagger=dagger,
                                  avg_best=avb, avg_second=avs, indent=bool(group),
                                  suppress_avg=suppress_partial and len(data) < len(COLUMNS)))

    # Node level
    lines.append("\\midrule")
    lines.append(level_header("Node-level"))
    navb, navs = section_avg_best(list(NODE_BASELINES.values()) + list(grad_nodes.values())
                                  + list(mask_nodes.values()) + list(ours_nodes.values()))
    # Random is deliberately OUTSIDE the three families and unindented. It is a control, not a
    # method: filing it under "Gradient-based" would be false, and giving it its own heading
    # would imply a family with one member. It stays the first row of the section, which is also
    # where a reader looks for the floor.
    emit(None, [(n, NODE_BASELINES[n]) for n in CONTROL_NAMES if n in NODE_BASELINES],
         best_node, second_node, navb, navs)
    # NAP / NAP-IG are MIB's own gradient baselines, so they head the gradient family rather
    # than sitting in a separate "transcribed from Table 1" block -- provenance is a comment
    # concern, not a reader-facing grouping.
    emit(GRAD_H,
         classify(NODE_BASELINES, "gradient") + list(grad_nodes.items()),
         best_node, second_node, navb, navs)
    emit(MASK_H, list(mask_nodes.items()), best_node, second_node, navb, navs)
    # Ours are held to completeness by complete_or_skip, so a partial one never reaches here and
    # suppress_partial has nothing to act on -- passed explicitly so the asymmetry is visible.
    emit(OURS_H, list(ours_nodes.items()), best_node, second_node, navb, navs,
         suppress_partial=False)

    # Edge level. Same three families, so that a reader who has learned the node section's
    # structure does not have to relearn it here. The groups are thinner (one row each for the
    # two baselines), which is itself informative: the edge baselines are one gradient method
    # and one mask method, and the flat version of this section did not say so.
    lines.append("\\midrule")
    lines.append(level_header("Edge-level"))
    eavb, eavs = section_avg_best(list(EDGE_BASELINES.values()) + list(ours_edges.values()))
    emit(GRAD_H, classify(EDGE_BASELINES, "gradient"),
         best_edge, second_edge, eavb, eavs, suppress_partial=False)
    emit(MASK_H, classify(EDGE_BASELINES, "mask"),
         best_edge, second_edge, eavb, eavs, suppress_partial=False)
    # MAttr edge llama3 cells use a reduced (200-example) subset -> dagger.
    EDGE_DAGGER = {(t, m) for t, m, _ in COLUMNS if m == "llama3"}
    emit(OURS_H, list(ours_edges.items()), best_edge, second_edge, eavb, eavs,
         dagger=EDGE_DAGGER, suppress_partial=False)

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
