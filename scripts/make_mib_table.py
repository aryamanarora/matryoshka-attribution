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

# Dirs produced by a wave that ran in the L2A venv, whose gemma2 cells are therefore computed
# with TL 3.2.1's broken Gemma-2 forward until scripts/reeval_gemma_mib.py has been run over
# them. Gated on the stamp that script writes, so an entry clears itself when the re-eval lands.
#
# ADD EVERY NEW DIR HERE at the same time you add it to reeval_gemma_mib.py's DIRS. The
# condition is not detectable from the pkls -- a re-evaluated pkl and an L2A-venv pkl are both
# just a pkl, and the numbers differ by less than the amount that would look obviously wrong.
# (mtime was tried and rejected: it misreports every dir whose non-gemma cells were topped up
# after the re-eval, which is most of the edge dirs.)
#
# Defined here rather than in make_mib_test_table.py because that module already imports this
# one, so this is the side of the dependency that can hold shared state.
GEMMA_REEVAL_PENDING = {
    "mib_node_topk_uniform_lr05", "test_node_topk_uniform_lr05",
    "mib_edge_topk_uniform_lr05", "test_edge_topk_uniform_lr05",
}
GEMMA_TASKS = ("ioi", "mcqa", "arc_easy")


def gemma_unstamped(d, level, split):
    """Gemma tasks in results/<d> still awaiting re-evaluation under the MIB venv."""
    if d not in GEMMA_REEVAL_PENDING:
        return []
    return [t for t in GEMMA_TASKS
            if not (RESULTS_BASE / d / f".gemma_reeval_{level}_{split}_{t}").exists()]

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
    # Was final_node, which is the SAME variant at the default lr=0.01 -- the one dir in this
    # block not at lr=0.05, so "\ourmethod{}, uniform k" silently meant a different LR here than
    # everywhere else, and than the test table's row of the same name. Repointed once
    # submit_softuni_lr05.sh produced the lr=0.05 run. final_node stays on disk.
    ("\\ourmethod{}", "mib_node_topk_uniform_lr05", "node", "uniform"),
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
    # Edge level (uniform k-schedule). Swept -> lr=0.05.
    ("\\ourmethod{}", "mib_edge_topk_uniform_lr05", "edge", "uniform"),
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
# never fill more than 3 of the 11 columns (docs/ugs_baseline.md). Node/Edge Pruning is not
# tied to an architecture or a level and covers everything (docs/edge_pruning_baseline.md).
UGS_DIR = "ugs_eval"
PARTIAL_COVERAGE = {"UGS"}
MASK_NODE_BASELINES = {}
MASK_EDGE_BASELINES = {}

# A mask learner optimizes ONE operating point, and its target sparsity is the knob that
# decides where the circuit switches on -- so each budget is a separate row rather than a
# hidden default. (latex label suffix, results dir); the unsuffixed dir is the runner's own
# default (0.9 node / 0.99 edge), the _s* dirs come from `run_edge_pruning.sbatch ... <S>`
# and _ld from `LOSS=logit_diff`. The full-size VALIDATION tables show every variant that has
# results; the space-constrained figures and the test table show EPRUN_BEST_SPARSITY only.
#
# Ordered sparse-ward, then the objective ablation last. A dir with no results is skipped by
# eprun_rows, so entries can be listed here before their jobs land.
#
# The _ld row matters more than it looks: every Node Pruning run before 2026-08-02 trained on
# Edge Pruning's KL while MAttr trains on logit-diff, so the KL rows differ from \ourmethod{}
# in BOTH objective and mask parameterization. Only the _ld row isolates the parameterization.
EPRUN_SPARSITIES = [
    ("$s{=}0.5$", "eprun_eval_s0.5"),
    ("$s{=}0.8$", "eprun_eval_s0.8"),
    ("$s{=}0.9$", "eprun_eval"),
    ("$s{=}0.95$", "eprun_eval_s0.95"),
    ("$s{=}0.99$", "eprun_eval_s0.99"),
    # logit-diff objective (_ld): MAttr's own training signal instead of Edge Pruning's KL.
    # Not a side ablation -- the KL rows compare MAttr against a baseline optimizing something
    # other than what CPR measures, and matching the objective is worth a lot. All five budgets
    # complete 2026-08-02 (row avg over 11 cells, KL row at the same budget in parens):
    #
    #   s=0.5  1.67 (0.86, 6/11)   s=0.8  1.46 (--, 9/11)   s=0.9  1.28 (1.00)
    #   s=0.95 1.36 (0.96)         s=0.99 1.24 (0.91)       MAttr node row: 1.88
    #
    # So the honest node-level gap is 1.88 vs 1.67, not 1.88 vs 1.00, and at s=0.5 the baseline
    # beats MAttr on 3 of 11 cells (all llama3: mcqa 2.41/1.90, arc_easy 2.11/2.04,
    # arc_challenge 2.07/1.79). Lower budgets (0.25, 0.1) are registered below to find the peak.
    #
    # Report "mean delta vs KL", NOT "% of the MAttr gap closed" -- the delta is uncorrelated
    # with how far behind a cell starts (corr = +0.09, n=8), so the percentage is a constant
    # numerator over a varying denominator and invents a per-cell story that is not there.
    #
    # And do NOT read per-cell budget-to-budget differences as budget effects. The L0 anneal
    # does not bind at high targets on the 1056-unit llama3 cells, so nominally different runs
    # land on the same circuit and differ only by seed: mcqa/llama3 keeps 380 units at target
    # 0.8 and 379 at target 0.9, yet scores 1.34 vs 0.67. Per-cell spread at fixed size is
    # ~0.7 AUC; only ROW MEANS (SE ~0.10 over 11 cells) are interpretable. s=0.5 is different --
    # there the constraint does bind (mcqa/llama3 keeps 561/1056, achieved 0.469), which is why
    # its lead over s=0.8/0.9 is a real budget effect rather than the same artifact.
    ("$s{=}0.1$, logit-diff", "eprun_eval_s0.1_ld"),
    ("$s{=}0.25$, logit-diff", "eprun_eval_s0.25_ld"),
    ("$s{=}0.5$, logit-diff", "eprun_eval_s0.5_ld"),
    ("$s{=}0.8$, logit-diff", "eprun_eval_s0.8_ld"),
    ("$s{=}0.9$, logit-diff", "eprun_eval_s0.9_ld"),
    ("$s{=}0.95$, logit-diff", "eprun_eval_s0.95_ld"),
    ("$s{=}0.99$, logit-diff", "eprun_eval_s0.99_ld"),
]

# The single config the test table and the figures show. Best by CPR AUC, which is the metric
# the paper leads with -- validation row avg over 11 cells is 1.67 for logit-diff s=0.5 against
# 1.00 for the KL s=0.9 run this used to name, and 1.67 is an interior optimum (s=0.25 -> 0.84
# below it, s=0.8 -> 1.46 above), not the edge of the swept range.
#
# Picking the KL run made \ourmethod{}'s margin look like 1.88 vs 1.00 when the honest node-level
# comparison is 1.88 vs 1.67: most of that apparent gap was the OBJECTIVE mismatch (Edge
# Pruning's KL vs MAttr's logit-diff), not the mask parameterization. Holding the loss fixed is
# the comparison the paper actually wants to make, and it costs us most of the headline gap.
# plot_method_corr_heatmap.EPRUN_BEST was moved for the same reason (rho vs MAttr 0.26 -> 0.57).
#
# Flipped 2026-08-03, once all 11 test cells existed. Order matters: make_mib_test_table and
# plot_mib_accauc_cpr_scatter both read this, so pointing it at a dir with no test pkls would
# silently drop the only mask-learning baseline out of the headline test table rather than
# error. Test row avg for this config is 1.655 over 11/11 cells.
#
# Caveat kept from the old comment: acc-AUC ranks the budgets differently from CPR, and the
# budgets disagree with each other on the node ranking itself (cross-budget rho 0.39-0.56), so
# "best" here means best-by-CPR and nothing stronger.
EPRUN_BEST_SPARSITY = ("$s{=}0.5$, logit-diff", "eprun_eval_s0.5_ld")


# DBM = differentiable binary masking, i.e. pyvene's SigmoidMaskIntervention: a third mask
# parameterization (deterministic sigmoid(mask/tau), tau annealed 50->0.1, NO sparsity term),
# same 3000 steps and same logit-diff loss as the _ld Node Pruning rows. Not an
# EPRUN_SPARSITIES entry -- those are all one method at different budgets and get labelled
# "Node Pruning (...)", which this is not.
#
# "DBM" is the DISPLAY name only. Everything on disk keeps the implementation name (gate
# "sigmoid", results/eprun_*_sig* dirs, the EdgePruning_patching_node subfolder MIB's
# run_evaluation.py writes) -- same rule as EPRUN_NAME above. Renaming those would orphan
# every pkl.
#
# The lr shown is swept, not pyvene's published 1e-3, and the label says so because the
# difference is large enough to change the ranking: 0.74 avg at 1e-3 vs 1.32 at 0.3 (11 vs 10
# cells), i.e. untuned it loses to KL Node Pruning (1.00) and tuned it clearly beats it.
# pyvene chose 1e-3 for a few rotation parameters at one intervention site; here the same
# optimizer drives 156--1056 gate logits, so that value has no reason to transfer and
# reporting it would be measuring our tuning rather than the method. Both points are in
# paper/tabs/lr_sweep.tex; only the tuned one belongs in the headline table.
#
# Structural caveat for the prose: because there is no sparsity term, achieved density is
# 38--54% at EVERY lr. lr changes how well-ordered the logits are within that half, not how
# many units survive -- which is why this cannot reach the L0-annealed rows no matter how it
# is tuned. That argument does not depend on any hyperparameter choice.
SIGMOID_MASK_ROWS = [
    ("DBM (tuned LR)", "eprun_eval_ld_sig_lr0.3"),
]


# === Training-cost column ===
#
# Unit: BACKWARD PASSES THROUGH THE MODEL, counted in sequences, for fitting ONE cell -- i.e.
# summed over optimizer steps of (batch size x mask samples per step) for the mask learners,
# and (attribution examples x IG steps) for the gradient methods. Counting optimizer *steps*
# instead would flatter whichever method batches hardest (UGS by a factor of 60), since a
# backward over a batch of 20 costs ~20x one over a batch of 1.
#
# Where each number comes from:
#   gradient methods  MIB-circuit-track/run_variants.sh (and run_relp/gim/attnrlp.sh, which
#                     share its CELLS): --num-examples 1000 on the IOI cells and 100 on all
#                     others (mcqa's "full" train split is 100 examples), times --ig-steps
#                     (5 for NAP-IG/Conductance, 1 for the rest). The range is a property of
#                     the dataset sizes, not of the method.
#   \ourmethod{}      scripts/eval_mib.py --steps with --train-batch-size 1, --k-avg 1. NODE
#                     runs are 500 steps but EDGE runs are 5000 -- read off the saved `args`
#                     in results/<dir>/*_scores.pt; do not assume one number for both levels.
#   Node/Edge Pruning scripts/run_edge_pruning.sbatch STEPS=3000, one example per step.
#   UGS               ~/optimalablation/edge_pruning_unif_mib.py makes one pass over the train
#                     split at batch_size 5 (gpt2) / 2 (qwen), and EdgeInferenceConfig sets
#                     n_samples=12 mask draws per batch, so a step is 60 (24) sequences:
#                     ioi = 9500 x 12 = 114k, mcqa = 100 examples x 6 repeats x 12 = 7.2k.
#
# Compute-proportional, NOT wall clock: the gradient methods run their passes in large batches
# on an unhooked model while the mask learners go one example at a time through patching
# hooks, and Edge Pruning's KL variant adds an unmasked forward per step that is not counted
# here. An order-of-magnitude column.
COST_GRAD_IG5 = "0.5--5k"    # 5 IG steps x 100--1000 examples
COST_GRAD_IG1 = "0.1--1k"    # 1 backward x 100--1000 examples
COST_EPRUN = "3k"            # 3000 steps x batch 1
COST_UGS = "7--114k"         # the 12 mask samples per step are what make this so large
COST_OURS = {"node": "0.5k", "edge": "5k"}
# ig_steps=5 rows; every other gradient row is a single backward per example.
COST_IG5_ROWS = {"NAP-IG", "Conductance", "EAP-IG-inp (CF, repro)"}


def eprun_label(level, suffix):
    """Row label for one Node/Edge Pruning variant -- the single formatting site."""
    return f"{EPRUN_NAME[level]} ({suffix})"


# Bhaskar et al. (2024) named the method for the granularity it prunes at, so the display name
# follows the level we actually ran: "Node Pruning" for node-level rows, "Edge Pruning" for
# edge-level ones. Same recipe, same code (src/learning_to_attribute/edge_pruning.py) -- only
# the label tracks the substrate. Do NOT hardcode one name for both; a node-level row called
# "Edge Pruning" (or vice versa) misstates what was pruned.
EPRUN_NAME = {"node": "Node Pruning", "edge": "Edge Pruning"}


def eprun_rows(level):
    """[(display, {(task, model): AUC})], one row per variant that has any results.

    Partial variants ARE shown -- a half-finished sweep is visible progress. But note the
    dashes mean something different here than for UGS: UGS is in PARTIAL_COVERAGE because it
    genuinely cannot run those cells, whereas a dashed Node Pruning cell just has not finished
    yet. The count is printed so an in-progress row is never mistaken for a final one, and the
    Avg column of a partial row averages only the cells present.
    """
    rows = []
    for suffix, dirn in EPRUN_SPARSITIES:
        data = load_run_eval(dirn, f"EdgePruning_patching_{level}")
        if not data:
            continue
        label = eprun_label(level, suffix)
        if len(data) < len(COLUMNS):
            print(f"  NOTE {label}: {len(data)}/{len(COLUMNS)} cells ({dirn}) -- still running")
        rows.append((label, data))
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
        # Drop gemma2 cells that have not been re-evaluated under the MIB venv yet: they were
        # computed with TL 3.2.1's broken Gemma-2 forward. Dashes for a pending job are honest;
        # a plausible wrong number is not, and nothing downstream can tell the two apart.
        pend = gemma_unstamped(results_dir, level, "validation")
        for t in pend:
            data.pop((t, "gemma2"), None)
        if pend:
            print(f"HOLD {method_name} ({level}, {group}): gemma2 cells {pend} in "
                  f"results/{results_dir} await scripts/reeval_gemma_mib.py")
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
                 avg_best=None, avg_second=None, suppress_avg=False, cost=None):
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
        return f"{prefix} & {cost or '---'} & " + " & ".join(vals) + " \\\\"

    def grad_cost(name):
        return COST_GRAD_IG5 if name in COST_IG5_ROWS else COST_GRAD_IG1

    def mask_cost(name):
        return COST_UGS if name == "UGS" else COST_EPRUN

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
            # suppress_avg on partial rows, same rule the mask-baseline rows already use. An Avg
            # over whatever cells happen to be present is not comparable to the full-coverage row
            # above it, and the bias is not zero-mean: the cells that go missing are the gemma
            # ones held for re-eval and the slow llama3 ones, which sit at opposite ends of the
            # range, so a partial row can read as either better or worse than it is. The edge
            # "+ unif k" row showed 7.90 over 8 cells against 6.99 over 11 purely because its
            # three held gemma cells are the lowest-scoring columns in that section.
            for n, r, g in rows_o:
                dg = LR05_DAGGER if r in LR05_CAPPED else dagger
                d = all_results.get(f"{n}_{level}_{g}", {})
                lines.append(make_row(n, d, best, second,
                                      indent=True, dagger=dg, avg_best=avb, avg_second=avs,
                                      suppress_avg=len(d) < len(COLUMNS),
                                      cost=COST_OURS[level]))
            for n, r, g in rows_u:
                dg = LR05_DAGGER if r in LR05_CAPPED else dagger
                d = all_results.get(f"{n}_{level}_{g}", {})
                lines.append(make_row(unifk(n), d, best, second,
                                      indent=True, dagger=dg, avg_best=avb, avg_second=avs,
                                      suppress_avg=len(d) < len(COLUMNS),
                                      cost=COST_OURS[level]))

    # Generate LaTeX
    ncols = len(COLUMNS)
    lines = []
    lines.append("\\begin{adjustbox}{max width=\\textwidth}")
    # Column 2 is the training-cost column, so every cmidrule below is shifted by one.
    lines.append("\\begin{tabular}{lr@{\\quad}" + "r" * ncols + "@{\\quad}r}")
    lines.append("\\toprule")
    lines.append("& & \\multicolumn{4}{c}{IOI} & Arithmetic & \\multicolumn{3}{c}{MCQA} & \\multicolumn{2}{c}{ARC (E)} & ARC (C) & \\\\")
    lines.append("\\cmidrule(lr){3-6} \\cmidrule(lr){7-7} \\cmidrule(lr){8-10} \\cmidrule(lr){11-12} \\cmidrule(lr){13-13}")
    header = ("\\textbf{Method} & \\textbf{Bwd.} & "
              + " & ".join(h for _, _, h in COLUMNS) + " & \\textbf{Avg} \\\\")
    lines.append(header)

    # === Node-level section ===
    lines.append("\\midrule")
    lines.append(f"\\multicolumn{{{ncols + 3}}}{{l}}{{\\textit{{Node-level}}}} \\\\")
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
    # pyvene sigmoid mask -- same runner, so same --head 200 llama3 subset and same dagger.
    for name, dirn in SIGMOID_MASK_ROWS:
        data = load_run_eval(dirn, "EdgePruning_patching_node")
        if not data:
            continue
        if len(data) < len(COLUMNS):
            print(f"  NOTE {name}: {len(data)}/{len(COLUMNS)} cells ({dirn}) -- still running")
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
        lines.append(make_row(name, data, best_node, second_node, indent=True, avg_best=avb,
                              avg_second=avs, cost=grad_cost(name)))
    if MASK_NODE_BASELINES:
        lines.append("\\textbf{Mask learning} \\\\")
        for name, data in MASK_NODE_BASELINES.items():
            lines.append(make_row(name, data, best_node, second_node, indent=True,
                                  avg_best=avb, avg_second=avs,
                                  suppress_avg=len(data) < len(COLUMNS),
                                  cost=mask_cost(name)))
    emit_ours(node_uniform, node_ours, "node", best_node, second_node, avb, avs)

    # === Edge-level section ===
    lines.append("\\midrule")
    lines.append(f"\\multicolumn{{{ncols + 3}}}{{l}}{{\\textit{{Edge-level}}}} \\\\")

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
        lines.append(make_row(name, data, best_edge, second_edge, indent=True, avg_best=eavb,
                              avg_second=eavs, cost=grad_cost(name)))
    # Mask learners rank by a learned gate rather than a gradient, so they get their own header.
    if MASK_EDGE_BASELINES:
        lines.append("\\textbf{Mask learning} \\\\")
        for name, data in MASK_EDGE_BASELINES.items():
            lines.append(make_row(name, data, best_edge, second_edge, indent=True,
                                  avg_best=eavb, avg_second=eavs,
                                  suppress_avg=len(data) < len(COLUMNS),
                                  cost=mask_cost(name)))
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
