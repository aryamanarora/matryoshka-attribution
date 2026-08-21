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
    # Optimizer ablation: identical forward and backward to the row above, Adam -> SGD, EACH AT
    # ITS OWN BEST LR. This was pinned to lr=0.05 (matched to the headline, a single-knob
    # contrast) until submit_softlog_sgd_lr.sh's grid came in, at which point the condition the
    # old comment set out was met: SGD peaks well away from Adam's optimum. Validation CPR over
    # the grid is 0.05 -> 1.413, 0.1 -> 1.584, 0.3 -> 1.592, *1.0 -> 1.886*, 3.0 -> 1.750,
    # 10 -> 1.681, against Adam's 1.879 at lr=0.05.
    #
    # The repoint MATERIALLY CHANGES THE CLAIM and that is the point: at the matched LR the
    # ablation reads "SGD costs 0.47 CPR", which is really a statement about SGD being 20x off
    # its optimum, not about the optimizer. At its own optimum SGD matches Adam (1.886 vs
    # 1.879), and the honest ablation is "the optimizer does not matter once tuned; the LR at
    # which it is tuned does". Do NOT re-pin these to a shared LR to recover a bigger gap.
    # The matched-LR numbers are not lost -- the whole grid is in tabs/lr_sweep.tex.
    #
    # Labelled "\ourmethod{}" rather than "+ SGD" because emit_ours() files it under the
    # \ourmethod{}-SGD header (opt_of matches _sgd), where it IS the plain method -- the
    # optimizer is already named by the header, so "+ SGD" there would read as a second one.
    ("\\ourmethod{}", "softlog_sgd_lr_1.0", "node", "ours"),
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
    # The uniform-k twin of the SGD row above, same own-best-LR policy. submit_softuni_sgd_lr.sh
    # completes the forward x k-schedule x optimizer square, and SGD peaks at lr=3.0 here rather
    # than 1.0: 0.05 -> 1.563, 0.1 -> 1.676, 0.3 -> 1.881, 1.0 -> 1.977, *3.0 -> 2.031*,
    # 10 -> 1.985, against Adam's 2.092. So the optimum MOVES with the k-schedule, which is why
    # each of the two SGD rows carries its own LR instead of sharing one.
    ("\\ourmethod{}", "softuni_sgd_lr_3.0", "node", "uniform"),
    ("$+$ hard", "htk_lr_0.05", "node", "uniform"),
    ("$+$ hard, $+$ Gumbel sel.", "mib_node_hard_topk_gumbel", "node", "uniform"),
    ("$-$ $c_k$", "mib_node_detached_tau", "node", "uniform"),
    ("$+$ hard bwd", "mib_node_bernoulli_reinforce", "node", "uniform"),
    ("$+$ id-STE", "mib_node_identity_sgd", "node", "uniform"),
    ("$+$ id-STE, Gumbel sel.", "mib_node_identity_gumbel_sgd_uniform", "node", "uniform"),
    # Edge level (log k-schedule = default). Swept methods -> lr=0.05.
    ("\\ourmethod{}", "mib_edge_topk_log_lr05", "edge", "ours"),
    # Edge twin of the node SGD row, submit_mib_edge_soft_sgd.sh. SAME LABEL POLICY as node
    # (filed under the \ourmethod{}-SGD header, so "+ SGD" would name the optimizer twice), but
    # NOT the same LR policy, and the difference matters:
    #
    #   node rows  = SGD at its OWN swept optimum (1.0 log / 3.0 uniform, both bracketed)
    #   edge rows  = those same two LRs IMPORTED, never swept at edge scale
    #
    # Edge n is 207-1507x node n (gpt2 157 -> 32,491; llama3 1057 -> 1,592,881) and soft-fwd SGD
    # is LR-sensitive, so the import is not justified by the node bracket. It also demonstrably
    # did not land: over the 9 cells shared with the headline edge row, log-k reads -1.221 mean
    # area_under, llama3-concentrated (arith_sub -3.45, ioi -3.74, mcqa -2.42) while gemma2 gains;
    # uniform-k is at parity (+0.154) and loses only on llama3. So these rows say "the node LR
    # does not transfer", NOT "SGD is worse than Adam at edge level" -- the second claim needs an
    # edge LR sweep (LR= override in submit_mib_edge_soft_sgd.sh). Do not caption them as an
    # optimiser result until that exists.
    ("\\ourmethod{}", "mib_edge_softlog_sgd_lr_1.0", "edge", "ours"),
    ("$+$ hard", "mib_edge_hard_topk_log_lr05", "edge", "ours"),
    ("$-$ $c_k$", "mib_edge_detached_tau", "edge", "ours"),
    ("$+$ hard bwd", "mib_edge_bernoulli_reinforce", "edge", "ours"),
    ("$+$ id-STE", "mib_edge_identity_sgd_log", "edge", "ours"),
    # Edge level (uniform k-schedule). Swept -> lr=0.05.
    ("\\ourmethod{}", "mib_edge_topk_uniform_lr05", "edge", "uniform"),
    # Uniform-k twin of the imported-LR edge SGD row above (lr=3.0). See that comment.
    ("\\ourmethod{}", "mib_edge_softuni_sgd_lr_3.0", "edge", "uniform"),
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

# NAP-IG / EAP-IG-inputs reproduced by us (NOT leaderboard numbers): run_attribution.py
# --method EAP-IG-inputs --ig-steps 5, then run_evaluation.py, train -> validation.
#
# Both dirs used to point at the June wave (napig_repro_eval / eapig_repro_eval), which ran
# under the L2A venv (TL 3.2.1) -- see scripts/submit_missing_baselines.sh, the one surviving
# submitter from that wave. That TL computes a wrong Gemma-2 forward (commit 525673a), so all
# six gemma2 cells across the two rows were suspect; on the node row the non-gemma cells agreed
# with a clean rerun to <=0.008 while ioi/gemma2 moved +0.270 and mcqa/gemma2 +0.106, which is
# what pinned it on the venv rather than run noise. Both dirs are now parked in
# results/_stale_tl321/ and nothing reads them.
#
# Replacements are both TL 2.15.4 (MIB-circuit-track/.venv) and both use the CELLS block that
# run_variants.sh / run_relp.sh / run_gim.sh / run_attnlrp.sh share, so NAP-IG is now
# flag-identical to the other gradient baselines in its column rather than merely close.
NAPIG_REPRO_DIR = "napig_ref_eval"        # MIB-circuit-track run_variants.sh, `ref` arm

# === NAP-IG step-count rows ==================================================================
#
# MIB's harness ships --ig-steps 5 (run_attribution.py:46) and that is NOT a converged
# integral. Measured over all 12 cells on the attribution output itself (importances.json, via
# MIB-circuit-track/napig_step_convergence.py), 5 -> 10 steps moves the ranking by rho 0.87
# with only 57% top-5 overlap, and 27 nodes CHANGE SIGN while sitting in the top 10 by |score|
# (26 of the 27 are MLPs). CPR is probed at 0.1--1% sparsity, so that unstable head is most of
# what the metric reads -- which is why the row mean jumps 0.77 -> 1.27 from 5 to 10 steps.
#
# Reporting only the shipped default would flatter \ourmethod{} by ~0.5 CPR AUC against a
# baseline that is merely under-integrated. Reporting only the converged setting would
# misdescribe the MIB leaderboard, whose published NAP-IG numbers are the 5-step ones. Hence
# both, as separate rows, which is also what makes the compute column above worth reading.
#
# Circuits come from MIB-circuit-track/run_napig{10,30}.sh. Those are copies of run_variants.sh
# with --ig-steps as the ONLY difference -- same CELLS, same TL 2.15.4 venv, same --head 200
# llama3 eval cap, same train -> validation direction -- so the gap between these rows and the
# NAP-IG row is the integration grid and nothing else.
#
# Is 30 itself converged? Yes, on all 12 cells, and by a wide margin. 10 -> 30 gives rho 0.994
# / 86.7% top-5 / ZERO sign flips, against 5 -> 10's rho 0.866 / 56.7% top-5 / 27 flips. The
# decisive cell is mcqa/llama3, the WORST at 5 -> 10 (20% top-5 overlap) and rho 0.992 with
# 100% top-5 at 10 -> 30: the instability is not merely smaller on average, it is gone from the
# cell that had the most of it. Same story for ioi/llama3, loosest in the 5 -> 10 column (rho
# 0.750, 40% top-5) and 100% top-5 with zero flips at 10 -> 30. llama3 as a family carried 13
# of the 27 flips and now carries none. CPR agrees independently: per-cell |30 minus 10| is at
# most 0.04, and the row averages are 1.31 vs 1.30.
#
# So the honest reading is that TEN steps is already converged and 30 is the confirmation, not
# that 30 is a distinct better setting. Prose may claim convergence at 10 steps.
# The IG grid is a COMPUTE knob, so these rows must move the Bwd. column with them --
# attribution cost is exactly linear in --ig-steps (same unit as the COST_* block below:
# backward passes in sequences, = examples x ig-steps, 100--1000 examples depending on cell).
# Leaving them at COST_GRAD_IG5 would show a converged NAP-IG costing what the 5-step run
# costs, and that trade is the point of the rows: 30 steps buys most of the CPR gap back, at
# 3--30k backwards against \ourmethod{}'s 0.5k node budget. Accuracy gap narrows, cost gap widens.
NAPIG_STEP_ROWS = [
    ("$+$ 10 IG steps", "napig10_eval", "1--10k"),
    ("$+$ 30 IG steps", "napig30_eval", "3--30k"),
]

# These dirs are written by the MIB repo and have not been copied into L2A's results/ (unlike
# napig_ref_eval, which was). Read them where they actually are rather than snapshotting: jobs
# are still landing, and a stale copy would silently under-report a row as partial forever.
# Same dual-root idea as make_mib_accauc_table.ROOTS, L2A first so a local copy wins if made.
MIB_RESULTS = Path("/home/guests/aryaman/MIB-circuit-track/results")

# NOT eapig_repro_accauc: that dir is clean but was attributed with --num-examples 1000 on every
# cell, off-convention for arc/arithmetic (100) and mcqa (full). It stays the acc-AUC source;
# this row comes from MIB-circuit-track/run_eapig_edge.sh, which follows CELLS.
EAPIG_REPRO_DIR = "eapig_clean_eval"

# Edge twin of NAPIG_STEP_ROWS, from MIB-circuit-track/run_eapig_edge10.sh (same CELLS, same
# venv, same --head 200 cap; --ig-steps is the only difference from EAPIG_REPRO_DIR).
#
# The node ladder is the reason this exists, and the edge answer is the OPPOSITE one, which is
# exactly why the row belongs in the table rather than in a footnote. At node level 5 -> 10
# steps moves CPR 0.85 -> 1.31; here it moves 1.63 -> 1.67 (+0.04, 9/11 cells) and acc-AUC not
# at all (0.933 -> 0.933, 6/12 cells -- a coin flip). So the 5-step edge baseline this table has
# always reported is NOT under-integrated, and our edge-level margin (6.37) does not depend on
# the baseline's IG grid. Only 10 is run: 30 would cost 3--30k backwards to confirm a delta
# that is already inside the noise at 10.
#
# Do not "simplify" by reusing NAPIG_STEP_ROWS' entry -- that one points at a node dir. The
# display name is deliberately identical so the row reads the same way in both sections, which
# also means it inherits the right STEP_COST and DAGGER entries for free.
EAPIG_EDGE_STEP_ROWS = [
    ("$+$ 10 IG steps", "eapig_clean10_eval", "1--10k"),
]

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

# The sweep above is the full grid we RAN; this is what the CPR table SHOWS -- the best budget
# per objective, one KL row and one logit-diff row. Twelve near-identical Node Pruning rows
# buried every other mask-learning baseline in the table, and the budget sweep is not the point
# being made there (it is a hyperparameter search we ran to give the baseline a fair shot).
#
# Selected by validation row mean on 2026-08-08, over the SAME 11 cells:
#   KL:         s=0.9 1.00  >  s=0.95 0.96  >  s=0.99 0.91   (s=0.5, s=0.8 partial at 6/11 and
#               9/11; on their own subsets s=0.9 still wins, 1.05 vs 0.86 and 0.99 vs 0.90, so
#               they cannot overtake it by finishing)
#   logit-diff: s=0.5 1.67  >  s=0.8 1.46  >  s=0.95 1.36  >  s=0.9 1.28  >  s=0.99 1.24
#               >  s=0.25 0.84  >  s=0.1 0.34   (interior optimum, agrees with HEADLINE_EPRUN)
#
# Other consumers (make_mib_accauc_table, plot_mib_accauc_cpr_scatter) still iterate the FULL
# EPRUN_SPARSITIES on purpose -- the scatter wants every budget as a point. Only this table
# filters. Set to None to restore all rows.
EPRUN_SHOW = {"eprun_eval", "eprun_eval_s0.5_ld"}

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
# parameterization (deterministic sigmoid(mask/tau), tau annealed 50->0.1, plus an L1 on the
# gate), same 3000 steps and same logit-diff loss as the _ld Node Pruning rows. Not an
# EPRUN_SPARSITIES entry -- those are all one method at different budgets and get labelled
# "Node Pruning (...)", which this is not.
#
# "DBM" is the DISPLAY name only. Everything on disk keeps the implementation name (gate
# "sigmoid", results/eprun_*_sig* dirs, the EdgePruning_patching_node subfolder MIB's
# run_evaluation.py writes) -- same rule as EPRUN_NAME above. Renaming those would orphan
# every pkl.
#
# BOTH knobs are swept, not pyvene's published defaults, because both change the ranking:
#   lr      1e-3 -> 0.3 moves avg CPR 0.74 -> 1.32. pyvene chose 1e-3 for a few rotation
#           parameters at one intervention site; here the same optimizer drives 156--1056 gate
#           logits, so that value has no reason to transfer.
#   lambda  0 -> 6.0 moves validation avg CPR 1.31 -> 1.50, best of {0, 0.2, 0.6, 2, 6, 20},
#           interior to the grid.
# The headline row is that tuned point, and the test table names the same recipe "DBM".
#
# Why the PENALISED recipe carries the name (changed 2026-08-07): the pyvene class ships with
# no sparsity term, but pyvene's own masking tutorial and Boundless DAS both put an L1 on the
# mask, so the penalty is the library's practice even if it is not the class default. Naming
# the unpenalised run "DBM" would hand the baseline its weakest operating point on a
# technicality. lambda=0 stays visible as the anchor of the lambda sweep in tabs/lr_sweep.tex
# and as a point in plot_mib_accauc_cpr_scatter's DBM series -- it is still reported, it is
# just no longer what the name refers to.
#
# Caveat for the prose: lambda also drops achieved density 0.56 -> 0.30, so the CPR gain is
# confounded with the sparsity change. The sweep fixes the best-tuned operating point; it does
# not establish that the penalty per se is what helps. (The old comment here claimed density
# was 38--54% "at EVERY lr" and concluded DBM structurally could not reach the L0-annealed
# rows. That held for the unpenalised mask only, and the lambda sweep is exactly what refutes
# it -- do not carry that argument into the prose.)
SIGMOID_MASK_ROWS = [
    ("DBM", "eprun_eval_ld_sig_lr0.3_l16.0"),
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
#   gradient methods  MIB-circuit-track/run_variants.sh (and run_relp/gim/relpshapley.sh, which
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
# The ig-steps 10 / 30 rows carry their own cost, declared with the rows in NAPIG_STEP_ROWS
# (defined above, since it needs them) and looked up via grad_cost's STEP_COST.
# ig_steps=5 rows; every other gradient row is a single backward per example.
COST_IG5_ROWS = {"NAP-IG", "Conductance", "EAP-IG-inp (CF, repro)"}


# MODULE level, not nested inside the renderer: make_mib_accauc_table.py renders the same rows
# under the same optimizer headers with the same daggers, and while it kept private copies of
# these they went stale the moment a second SGD arm was added -- that table filed softlog_sgd_*
# under "\ourmethod{}-Adam" while this one had it under SGD. Two tables contradicting each
# other about which optimizer a run used is a wrong claim about the run, not a layout nit, so
# the definitions live here once and that module imports them.
def opt_of(results_dir):
    # id-STE variants are trained with SGD, and so is the soft-topk optimizer ablation
    # (softlog_sgd_*); everything else with Adam. Matching on "identity" alone was enough
    # while id-STE was the ONLY SGD arm, but it silently files any other SGD dir under the
    # \ourmethod{}-Adam header.
    return "sgd" if ("identity" in results_dir or "_sgd" in results_dir) else "adam"


# Swept dirs that cap ioi/llama3 at --eval-examples 200 -> dagger just that ONE cell for those
# rows, rather than the whole-row dagger the uncapped baselines get.
#
# Named for the CONDITION, not the LR. It was IOI_LLAMA_CAPPED while every member happened to be an
# lr=0.05 dir; that stopped being true when the SGD rows were repointed to their own optima
# (softlog_sgd_lr_1.0, softuni_sgd_lr_3.0), and a set literally named "LR05" holding an lr=3.0
# dir is the kind of drift this file warns about everywhere else.
#
# MEMBERSHIP IS A PROPERTY OF THE SUBMIT SCRIPT, not of the LR: submit_softlog_sgd_lr.sh:91 and
# submit_softuni_sgd_lr.sh:67 both apply `--eval-examples 200` to ioi/llama3 at EVERY lr in the
# grid, so any dir from those sweeps belongs here whichever LR the table ends up pointing at.
IOI_LLAMA_CAPPED = {"htklog_lr_0.05", "topklog_lr_0.05", "htk_lr_0.05",
                    "softlog_sgd_lr_1.0", "softuni_sgd_lr_3.0"}
IOI_LLAMA_DAGGER = {("ioi", "llama3")}


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
        if EPRUN_SHOW is not None and dirn not in EPRUN_SHOW:
            continue
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


def load_eval_dual(results_dir, sub):
    """load_run_eval, but searching L2A results/ then the MIB repo's results/.

    Separate from load_run_eval rather than folded into it: every existing caller resolves
    L2A-side, and silently widening their search could pull a cell out of a MIB-side dir that
    happens to share a name with an L2A one (napig_ref_eval exists on BOTH sides).
    """
    data = {}
    for task, model, _ in COLUMNS:
        fn = f"{task.replace('_', '-')}_{model}_validation_abs-False.pkl"
        for root in (RESULTS_BASE, MIB_RESULTS):
            pkl = root / results_dir / sub / fn
            if not pkl.exists():
                continue
            try:
                with open(pkl, "rb") as f:
                    data[(task, model)] = round(pickle.load(f)["area_under"], 2)
                break
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

    # Keyed on the RESULTS DIR, not the display name. The display name is not unique -- the
    # same label legitimately appears once per (level, group) block, and now also once per
    # optimizer -- so a name-based key silently made the last row with a given label overwrite
    # every earlier one's data. Adding the SGD row as "\ourmethod{}" at node/ours blanked the
    # Adam headline row that way: both hashed to "\ourmethod{}_node_ours". The dir is the one
    # thing that is unique per run, which is what this key needs to be.
    def mkey(results_dir, level, group):
        return f"{results_dir}_{level}_{group}"

    for method_name, results_dir, level, group in OUR_METHODS:
        key = mkey(results_dir, level, group)
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
        our = {mkey(d, l, g): all_results.get(mkey(d, l, g), {})
               for _, d, l, g in OUR_METHODS if l == level}
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
        # Only COMPLETE rows compete for the best-Avg bold. A partial row's Avg is suppressed at
        # render time, so if it won here the bold would simply vanish from the section: the
        # winner would be an average that is never printed. Ranking partial against complete
        # averages is meaningless anyway -- they are over different cell sets.
        full = [d for d in data_dicts if len(d) == len(COLUMNS)]
        avs = sorted({a for a in (row_avg(d) for d in full) if a is not None}, reverse=True)
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

    # Rows whose IG grid is neither 5 nor 1 declare their own cost in NAPIG_STEP_ROWS.
    STEP_COST = {disp: cost for disp, _, cost in NAPIG_STEP_ROWS}

    def grad_cost(name):
        if name in STEP_COST:
            return STEP_COST[name]
        return COST_GRAD_IG5 if name in COST_IG5_ROWS else COST_GRAD_IG1

    def mask_cost(name):
        return COST_UGS if name == "UGS" else COST_EPRUN

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
                dg = IOI_LLAMA_DAGGER if r in IOI_LLAMA_CAPPED else dagger
                d = all_results.get(mkey(r, level, g), {})
                # A row with no populated cells renders as 12 "---" and claims a run exists
                # that scored nothing, which is worse than not listing it. Skip until the
                # first cell lands; it then appears on the next regeneration with no edit
                # here. Announced, never silent -- same rule as make_lr_table's block skip.
                if not d:
                    print(f"SKIP row {n!r} ({r}, {level}/{g}): no results yet")
                    continue
                lines.append(make_row(n, d, best, second,
                                      indent=True, dagger=dg, avg_best=avb, avg_second=avs,
                                      suppress_avg=len(d) < len(COLUMNS),
                                      cost=COST_OURS[level]))
            for n, r, g in rows_u:
                dg = IOI_LLAMA_DAGGER if r in IOI_LLAMA_CAPPED else dagger
                d = all_results.get(mkey(r, level, g), {})
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
    # Every node-level baseline in this section (NAP-IG and its step variants, the Tilde
    # methods, the mask learners) is scored by a runner that caps llama3 at --head 200, so they
    # all share one dagger set. Hoisted above the NAP-IG block because that block now needs it
    # too; it used to be defined further down, next to its first use.
    TILDE_LLAMA3_DAGGER = {(t, m) for t, m, _ in COLUMNS if m == "llama3"}
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
    # All six llama3 cells of run_variants.sh are scored with --head 200 (run_variants.sh:21-26),
    # not just mcqa -- the pre-existing DAGGER["NAP-IG"] entry above marked only mcqa/llama3, so
    # five capped cells were rendering as if they were full-validation numbers. It matters most
    # in exactly the columns being argued over: \ourmethod{}'s lr05 dirs cap ONLY ioi/llama3
    # (IOI_LLAMA_DAGGER), so e.g. the mcqa/llama3 column puts a full-val MAttr number next to a
    # 200-example NAP-IG one, and the dagger is the table's only disclosure of that.
    DAGGER["NAP-IG"] = TILDE_LLAMA3_DAGGER
    # ig-steps 10 / 30 rows, same runner and same cap -> same dagger set.
    for disp, dirn, _cost in NAPIG_STEP_ROWS:
        data = load_eval_dual(dirn, "EAP-IG-inputs_patching_node")
        if not data:
            print(f"  NOTE {disp}: 0/{len(COLUMNS)} cells ({dirn}) -- not started; row omitted")
            continue
        if len(data) < len(COLUMNS):
            print(f"  NOTE {disp}: {len(data)}/{len(COLUMNS)} cells ({dirn}) -- still running; "
                  f"Avg suppressed until complete")
        NODE_BASELINES[disp] = data
        DAGGER[disp] = TILDE_LLAMA3_DAGGER

    # Additional baselines fetched from Tilde (node-level); each dir has one method subfolder
    EXTRA_NODE_BASELINES = [
        ("Conductance", "napig_local_eval", "EAP-IG-inputs-local_patching_node"),
        ("I$\\times$G", "ig1_eval",         "EAP-IG-inputs_patching_node"),
        ("RelP",        "relp_eval",        "RelP_patching_node"),
        ("RelP+QK",     "relp_qkgrad_eval", "RelP-qkgrad_patching_node"),
        ("RelP+Shapley",     "relpshapley_eval",     "RelPShapley_patching_node"),
        ("AttnLRP",     "attnlrp_eval",     "AttnLRP_patching_node"),
        ("GIM",         "gim_eval",         "GIM_patching_node"),
    ]
    # Tilde baselines used a reduced subset for the llama3 cells only -> dagger those
    # (TILDE_LLAMA3_DAGGER is defined at the top of this section).
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
        # Same partial-row warning eprun_rows() prints, and for a sharper reason here: the Avg
        # column of a partial row averages ONLY the cells present, so a row with 3 of 11 cells
        # gets an Avg that looks directly comparable to an 11-cell row and is not. That bit us
        # mid-rerun -- a 3-cell GIM and a 9-cell AttnLRP both landed on Avg 1.39, which reads as
        # a tie between two things that were never measured on the same cells.
        if data and len(data) < len(COLUMNS):
            print(f"  NOTE {disp}: {len(data)}/{len(COLUMNS)} cells ({dirn}) -- still running; "
                  f"its Avg covers only those {len(data)}")
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
        + [all_results.get(mkey(d, "node", g), {}) for _, d, _, g in node_uniform] \
        + [all_results.get(mkey(d, "node", g), {}) for _, d, _, g in node_ours]
    avb, avs = section_avg_best(node_dicts)

    lines.append("\\textbf{Gradient attribution} \\\\")
    for name, data in NODE_BASELINES.items():
        # suppress_avg on partial rows -- the same rule the mask-baseline and \ourmethod{} rows
        # already use, and it was the one block missing it. An Avg over whichever cells happen
        # to have finished sits in the same column as an 11-cell Avg and reads as comparable.
        # Live risk right now: the 30-step row fills cheap-model cells first, and those are the
        # LOW-scoring columns for NAP-IG, so a partial Avg would understate it and overstate
        # our margin -- the exact direction of error we should be most reluctant to publish.
        lines.append(make_row(name, data, best_node, second_node, indent=True, avg_best=avb,
                              avg_second=avs, suppress_avg=len(data) < len(COLUMNS),
                              cost=grad_cost(name)))
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
    # run_eapig_edge.sh scores ALL SIX llama3 cells with --head 200 (its CELLS block is verbatim
    # from run_variants.sh), not just arc_challenge. The static DAGGER entry above marked only
    # that one cell, so five 200-example numbers were rendering as if they were full validation
    # -- the identical defect already fixed on the node NAP-IG row, which is where this same
    # TILDE_LLAMA3_DAGGER assignment comes from. Overriding here rather than editing the static
    # dict keeps the two fixes side by side with their sections.
    DAGGER["EAP-IG-inp (CF, repro)"] = TILDE_LLAMA3_DAGGER

    for disp, dirn, _cost in EAPIG_EDGE_STEP_ROWS:
        data = load_eval_dual(dirn, "EAP-IG-inputs_patching_edge")
        if not data:
            print(f"  NOTE {disp} (edge): 0/{len(COLUMNS)} cells ({dirn}) -- not started; row omitted")
            continue
        if len(data) < len(COLUMNS):
            print(f"  NOTE {disp} (edge): {len(data)}/{len(COLUMNS)} cells ({dirn}) -- still "
                  f"running; Avg suppressed until complete")
        EDGE_BASELINES[disp] = data
        DAGGER[disp] = TILDE_LLAMA3_DAGGER

    # Mask learning at edge level: UGS (reg_lamb=0.001, gpt2/qwen only) + Edge Pruning
    ugs = load_run_eval(UGS_DIR, "UGS_patching_edge")
    if ugs:
        MASK_EDGE_BASELINES["UGS"] = ugs
    for name, data in eprun_rows("edge"):
        MASK_EDGE_BASELINES[name] = data
        DAGGER[name] = TILDE_LLAMA3_DAGGER

    best_edge, second_edge = best_in_col("edge")
    edge_dicts = list(EDGE_BASELINES.values()) + list(MASK_EDGE_BASELINES.values()) \
        + [all_results.get(mkey(d, "edge", g), {}) for _, d, _, g in edge_uniform] \
        + [all_results.get(mkey(d, "edge", g), {}) for _, d, _, g in edge_ours]
    eavb, eavs = section_avg_best(edge_dicts)

    # MAttr edge llama3 cells use a reduced eval subset (sphinx rerun) -> dagger.
    EDGE_LLAMA_DAGGER = {(t, m) for t, m, _ in COLUMNS if m == "llama3"}
    lines.append("\\textbf{Gradient attribution} \\\\")
    for name, data in EDGE_BASELINES.items():
        # suppress_avg on partial rows, same rule as every other section. This loop predates
        # any incomplete edge row (the one baseline here was always 11/11), but the step-ladder
        # row lands cell by cell over ~a day of jobs, and an Avg over whichever cells finished
        # first is actively misleading: the first four to land were three mcqa cells, the only
        # task where more IG steps HURT, which read as "10 steps is worse" until the rest came in.
        lines.append(make_row(name, data, best_edge, second_edge, indent=True, avg_best=eavb,
                              avg_second=eavs, suppress_avg=len(data) < len(COLUMNS),
                              cost=grad_cost(name)))
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
