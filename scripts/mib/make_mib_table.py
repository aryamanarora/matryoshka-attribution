"""Generate LaTeX table of MIB CPR AUC results from saved .pkl files.

Reads results from results/ directories and outputs to paper/tabs/.
Run from the repo root on the cluster:
    uv run python scripts/mib/make_mib_table.py
"""

import pickle
import math
from pathlib import Path

import mattr_variants as MV   # which MAttr variant is the unmarked headline, and the label grammar
from learning_to_attribute import deps as MV_DEPS   # find_mib_path / mib_results_dir

import torch

RESULTS_BASE = Path("results")
OUTPUT = Path("paper/tabs/mib_results.tex")

# Dirs produced by a wave that ran in the L2A venv, whose gemma2 cells are therefore computed
# with TL 3.2.1's broken Gemma-2 forward until scripts/mib/reeval_gemma_mib.py has been run over
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
    # Both arithmetic tasks are llama3-only in MIB. Addition joined on 2026-09-15 (the paper's
    # grid was 11 cells; the leaderboard needs 12). A row whose addition cell has not landed is
    # treated exactly like any other partial row: Avg suppressed, excluded from section best/avg,
    # and a NOTE printed -- so the tables never silently mix 11- and 12-cell averages.
    ("arithmetic_addition", "llama3", "Llama ($+$)"),
    ("arithmetic_subtraction", "llama3", "Llama ($-$)"),
    ("mcqa", "qwen2.5", "Qwen"),
    ("mcqa", "gemma2", "Gemma"),
    ("mcqa", "llama3", "Llama"),
    ("arc_easy", "gemma2", "Gemma"),
    ("arc_easy", "llama3", "Llama"),
    ("arc_challenge", "llama3", "Llama"),
]

# Our method + ablations: (display_name, results_subdir, level, group)
# group: "ours" = LOG k-schedule, "uniform" = UNIFORM k-schedule (mattr_variants.GROUP_K).
# Names are RELATIVE to the row's (k-schedule, optimizer) block: the k mark and the optimizer
# header are added at emission from scripts/mib/mattr_variants.py, which is where the headline
# (uniform k, Adam, soft forward since 2026-09-15) is defined. The comments below that say
# "log k = default" or "SGD is the unmarked default" describe earlier conventions and the
# reasoning behind each dir's LR; they are kept as the record, not as the current layout.
OUR_METHODS = [
    # Node level, log k-schedule. Swept methods use lr=0.05 (best); llama/ioi capped 200.
    # SOFT top-k forward; "+ hard" = sigmoid-STE hard forward.
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
    # Labelled "\ourmethod{}" because emit_ours() files it under the (now-default) SGD header
    # (opt_of matches _sgd), where it IS the plain method -- the optimizer is already named by
    # the header, so "+ SGD" there would read as a second one.
    ("\\ourmethod{}", "softlog_sgd_lr_1.0", "node", "ours"),
    ("$+$ hard", "htklog_lr_0.05", "node", "ours"),
    ("$-$ $c_k$", "mib_node_detached_tau_log", "node", "ours"),
    ("$+$ hard bwd", "mib_node_bernoulli_reinforce_log", "node", "ours"),
    ("$+$ id-STE", "mib_node_identity_sgd_log", "node", "ours"),
    ("$+$ id-STE, Gum.", "mib_node_identity_gumbel_sgd_log", "node", "ours"),
    # Node level (uniform k-schedule = ablation). Swept -> lr=0.05.
    # Was final_node, which is the SAME variant at the default lr=0.01 -- the one dir in this
    # block not at lr=0.05, so "\ourmethod{}, uniform k" silently meant a different LR here than
    # everywhere else, and than the test table's row of the same name. Repointed once
    # submit_softuni_lr05.sh produced the lr=0.05 run. final_node stays on disk.
    ("\\ourmethod{}", "mib_node_topk_uniform_lr05", "node", "uniform"),
    # The headline at Adam eps=1e-2 (submit_mib_node_eps_sc.sh, 2026-09-17): the eps the SVA+
    # substrates ship with, run on MIB to ask whether one eps serves every section (reviewer W6).
    # Labelled with the eps mark directly rather than through MV.label(eps=...), whose eps
    # grammar is the SVA one (1e-2 unmarked); on MIB the headline is 1e-8 and unmarked.
    (MV.eps_mark("1e-2"), "mib_node_topk_uniform_lr05_eps1e-2", "node", "uniform"),
    # 10x the steps (5000; submit_node_unif_10x_sc.sh), node twin of the edge 50k row below.
    ("$+$ $10\\times$ steps", "mib_node_topk_uniform_lr05_5k", "node", "uniform"),
    # No learning (submit_mib_node_frozen_sc.sh, 2026-09-18): --optimizer none, scores stay 0
    # and the attribution is the mean negated gradient over the 500 k-draws (the gradient
    # appendix's first-step update by Monte Carlo). No lr; the column prints the saved 0.05.
    # topk_identity (soft fwd, identity bwd): the k-averaged gradient at zero scores is IG along
    # the mask path. Uniform k only (2026-09-18, requested).
    ("$-$ learning", "mib_node_topkid_uniform_frozen", "node", "uniform"),
    # The OBJECTIVE ablation of this headline (--mode cause / joint, 2026-09-16,
    # submit_mib_node_mode_ablation_sc.sh -> mib_node_{cause,joint}_topk_uniform_lr05 and the
    # test_node_ twins) is NOT a row here (user decision, same day): it has its own per-task
    # figure, plots/plot_objective_ablation.py -> figs/objective_ablation.pdf, in its own
    # appendix section, and scripts/mib/summarize_mode_ablation.py prints the numbers.
    # The uniform-k twin of the SGD row above, same own-best-LR policy. submit_softuni_sgd_lr.sh
    # completes the forward x k-schedule x optimizer square, and SGD peaks at lr=3.0 here rather
    # than 1.0: 0.05 -> 1.563, 0.1 -> 1.676, 0.3 -> 1.881, 1.0 -> 1.977, *3.0 -> 2.031*,
    # 10 -> 1.985, against Adam's 2.092. So the optimum MOVES with the k-schedule, which is why
    # each of the two SGD rows carries its own LR instead of sharing one.
    ("\\ourmethod{}", "softuni_sgd_lr_3.0", "node", "uniform"),
    ("$+$ hard", "htk_lr_0.05", "node", "uniform"),
    ("$+$ hard, $+$ Gum.", "mib_node_hard_topk_gumbel", "node", "uniform"),
    ("$-$ $c_k$", "mib_node_detached_tau", "node", "uniform"),
    ("$+$ hard bwd", "mib_node_bernoulli_reinforce", "node", "uniform"),
    ("$+$ id-STE", "mib_node_identity_sgd", "node", "uniform"),
    ("$+$ id-STE, Gum.", "mib_node_identity_gumbel_sgd_uniform", "node", "uniform"),
    # Edge level (log k-schedule = default). Swept methods -> lr=0.05.
    ("\\ourmethod{}", "mib_edge_topk_log_lr05", "edge", "ours"),
    # Edge twin of the node SGD row, submit_mib_edge_soft_sgd.sh. SAME LABEL POLICY as node
    # (filed under the now-default SGD header, so "+ SGD" would name the optimizer twice), but
    # the SAME LR policy: SGD at its own block-argmax. lr=3.0 is SGD's EDGE optimum, bracketed by
    # submit_mib_edge_lr_sweep.sh over its 4 cells (0.3 -> 3.55, 1.0 -> 4.72, *3.0 -> 6.89*,
    # 10 -> 6.53), so this row is tuned-vs-tuned against the Adam edge row like every other row.
    #
    # HISTORY: until 2026-09-04 this pointed at mib_edge_softlog_sgd_lr_1.0, the NODE optimum
    # imported without an edge sweep. Edge n is 207-1507x node n and soft-fwd SGD is
    # LR-sensitive, and the sweep confirmed the import cost ~2.2 CPR on the shared cells. That
    # dir is still on disk and is a "wrong LR" artefact, not an optimizer result -- if a number
    # here looks worse than Adam by ~3, check you're not reading it.
    #
    # WHAT THIS ROW DOES NOT SAY: even tuned, Adam still leads SGD at edge scale on the sweep
    # cells (7.65 vs 6.89) -- the node-level tie does not transfer. Never caption this as "SGD
    # is the better edge optimizer".
    ("\\ourmethod{}", "mib_edge_softlog_sgd_lr_3.0", "edge", "ours"),
    ("$+$ hard", "mib_edge_hard_topk_log_lr05", "edge", "ours"),
    ("$-$ $c_k$", "mib_edge_detached_tau", "edge", "ours"),
    ("$+$ hard bwd", "mib_edge_bernoulli_reinforce", "edge", "ours"),
    ("$+$ id-STE", "mib_edge_identity_sgd_log", "edge", "ours"),
    # Edge level (uniform k-schedule). Swept -> lr=0.05.
    ("\\ourmethod{}", "mib_edge_topk_uniform_lr05", "edge", "uniform"),
    # 10x the steps (50000; submit_edge_unif_10x_sc.sh, 2026-09-17): does the dead sparse end of
    # the uniform-k edge ranking (acc-AUC 0.74 vs log-k 0.96 on gpt2/ioi) come from under-training?
    ("$+$ $10\\times$ steps", "mib_edge_topk_uniform_lr05_50k", "edge", "uniform"),
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
# under the L2A venv (TL 3.2.1) -- see scripts/mib/launch/submit_missing_baselines.sh, the one surviving
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
# Resolved through deps.find_mib_path (deps/MIB-circuit-track/results on juice3; the Tilde path
# above as the last resort). Hardcoded to Tilde until 2026-09-16, which on sc dropped every row
# read from here ("0/12 cells -- not started; row omitted") from the generated table.
MIB_RESULTS = MV_DEPS.mib_results_dir()

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
# Single-node interchange intervention, "IntInv" in the paper (eval_mib_actpatch.py; the dirs keep
# the actpatch name). Exists only where it is affordable --
# gpt2 and qwen2.5, the three small-model cells -- so, like UGS, its rows are partial by design
# and print no Avg. Own header ("Interchange intervention"): it is neither a gradient nor a learned
# mask but the causal quantity both approximate. denoise = restore one node in the corrupted
# run (the quantity CPR integrates), noise = corrupt one node in the clean run.
ACTPATCH_NODE_ROWS = [("IntInv (denoise)", "mib_node_actpatch_denoise"),
                      ("IntInv (noise)", "mib_node_actpatch_noise")]
PARTIAL_COVERAGE = {"UGS"} | {n for n, _ in ACTPATCH_NODE_ROWS}
MASK_NODE_BASELINES = {}
MASK_EDGE_BASELINES = {}
CAUSAL_NODE_BASELINES = {}

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
# *** ONE BUDGET, s=0.95, SHARED WITH THE ACC-AUC TABLE -- and it is deliberately NOT either
# metric's argmax. *** Until 2026-09-02 each table named its own best budget: this one took the
# CPR argmax per objective (KL s=0.9, LD s=0.5) and make_mib_accauc_table took the acc-AUC one
# (s=0.99 for both). Both picks were defensible alone and together they were indefensible --
# two tables, one page apart, with identically-named "Node Pruning" rows that were different
# runs. CPR and acc-AUC rank the budgets in near-opposite orders (validation row means, 11
# cells, higher s = sparser):
#
#   logit-diff   s=0.5   s=0.8   s=0.9   s=0.95  s=0.99
#     CPR AUC     1.67    1.46    1.28    1.36    1.24     <- densest wins
#     acc-AUC     0.23    0.31    0.34    0.36    0.38     <- sparsest wins, the other way
#   KL           s=0.9  1.00/0.40    s=0.95  0.96/0.46    s=0.99  0.91/0.46
#
# so the single budget had to be chosen against BOTH metrics or it would flatter us under one
# of them. s=0.95 is the rank-sum optimum for each objective: KL is the acc-AUC argmax (0.458,
# tied with s=0.99 at 2dp) and second on CPR by 0.04; LD is second on acc-AUC (0.36 vs 0.38)
# and third on CPR. Carrying the CPR pick into the acc-AUC table instead would have shown the
# baseline at its WORST acc budget (0.23 vs 0.38), and carrying the acc-AUC pick in here would
# have widened our node margin from 1.88-vs-1.36 to 1.88-vs-1.24. Both were rejected for the
# same reason: consistency should not be bought out of the baseline's score.
#
# *** EPRUN_BEST_SPARSITY IS STILL s=0.5 LD, AND THAT IS NOT AN OVERSIGHT. *** The test split
# has 11/11 pkls for eprun_eval_s0.5_ld and 0/11 for eprun_eval_s0.95_ld, so repointing it
# would silently drop the only mask-learning baseline out of the headline test table. Until
# scripts/mib/launch/submit_test_node_pruning.sh has been run for s=0.95, the test table shows s=0.5 LD
# while these two validation tables show s=0.95 -- a real remaining inconsistency, recorded
# here rather than papered over. Repoint it, and delete this paragraph, once those cells land.
#
# Other consumers (make_mib_accauc_table, plot_mib_accauc_cpr_scatter) still iterate the FULL
# EPRUN_SPARSITIES on purpose -- the scatter wants every budget as a point. Only this table
# filters. Set to None to restore all rows (they then keep their EPRUN_SPARSITIES suffixes).
#
# A DICT, not a set: the value REPLACES eprun_label's "$s{=}...$" suffix for this table only.
# Once the twelve rows are filtered to one per objective the budget is no longer what
# distinguishes them -- the objective is -- so the parenthetical names that instead.
#
# *** THE BUDGET IS INVISIBLE IN THE LABEL, SO IT HAD BETTER BE THE SAME EVERYWHERE. *** Both
# rows are s=0.95, and make_mib_accauc_table now reads THIS dict rather than keeping its own
# pick, so the two tables cannot drift into showing identically-named rows from different runs.
# That is the whole reason the budget could be dropped from the label at all. If you ever point
# the two tables at different budgets again, put s= back in these strings first.
# The caption should still name the budget, since no row does any more.
# 2026-09-16 (user decision): the KL-objective row is dropped from the paper altogether -- the
# sweep figure, this table and the acc-AUC table show the logit-diff run only, so the label
# names the budget again (the objective no longer distinguishes anything).
# PER LEVEL since 2026-09-17: edge-level Edge Pruning (submit_eprun_edge_sc.sh) runs at the
# script's edge default s=0.99, so the node pick cannot be reused for it. Same dict shape per
# level (dir -> label suffix); make_mib_accauc_table reads the level it renders.
EPRUN_SHOW = {"node": {"eprun_eval_s0.95_ld": "$s{=}0.95$"},
              "edge": {"eprun_eval_s0.99_ld": "$s{=}0.99$"}}

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
# The same method read as a FRONTIER instead of a ranking: seven lambda rungs, each evaluated
# only at the L0 it converged to, integrated over MIB's p range. See
# scripts/mib/dbm_multisparsity.py for the anchors and, more importantly, for why this row is
# not the same object as every other row in the table -- it spends seven training runs where a
# ranking spends one, and the caption has to say so.
#
# THE ROW IS IN THE TEST TABLE ONLY (make_mib_test_table.py), NOT THIS ONE. These two constants
# live here only because that module imports them.
#
# Both validation tables (this one and make_mib_accauc_table.py) were wired up on 2026-09-09 and
# UNWIRED the same day: the row needs its own validation-split evaluation of the ladder -- the
# test numbers cannot stand in, or it would be the one row in the table scored on different data
# from its neighbours -- and that eval was declined as not worth the GPU time. If it is ever
# wanted, the work is one `bash scripts/mib/launch/submit_dbm_multisparsity.sh` (SPLIT defaults
# to validation) plus a MASK_NODE_BASELINES entry reading _DBMMS.cell(task, model,
# "validation"); do NOT point the row at the test JSONs to save the wave.
DBM_MULTI_ROW = "DBM (multi-sparsity)"
# Cost is the LADDER's, not one run's: 8 rungs x COST_EPRUN (3k backward passes each) = 24k.
# Stated as a literal rather than derived because COST_EPRUN is already a display string; if the
# ladder length in eval_dbm_multisparsity.L1S changes, change this with it.
#
# 21k -> 24k on 2026-09-09 when lambda=200 joined the ladder. THE COUNT IS THE ATTEMPTED LADDER,
# not the rungs that land: a lambda that collapses to zero open gates (3 of 11 cells at 200) was
# still trained and still cost 3k, and a cost column that only charged for successful runs would
# understate exactly the method whose knob is hardest to aim.
#
# *** NOTHING EMITS THIS. *** The test table, the row's only home, has no cost column, and this
# table has no such row. It is kept because the number is real and the CAPTION has to carry it
# -- 8 training runs against every other row's 1 -- and deriving it again from L1S at writing
# time is how it would end up stated as 21k in the paper.
DBM_MULTI_COST = "24k"
# The same frontier reading of NODE PRUNING (2026-09-16): its nine target-s runs from
# tabs/sparsity_sweep.tex, each read at the size of the mask it emits, through the same reader
# (dbm_multisparsity.cell at NP_RESULTS). 9 rungs x 3k = 27k, counted like DBM's: the attempted
# ladder, whether or not a rung lands on the frontier (0.9/0.95/0.99 collide on most cells).
NP_MULTI_ROW = "Node Pruning (multi-sparsity)"
NP_MULTI_COST = "27k"


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
#   \ourmethod{}      scripts/mib/eval_mib.py --steps with --train-batch-size 1, --k-avg 1. NODE
#                     runs are 500 steps but EDGE runs are 5000 -- read off the saved `args`
#                     in results/<dir>/*_scores.pt; do not assume one number for both levels.
#   Node/Edge Pruning scripts/mib/launch/run_edge_pruning.sbatch STEPS=3000, one example per step.
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
COST_EPRUN_EDGE = "5k"       # edge-level Edge Pruning at MAttr's edge step count (submit_eprun_edge_sc.sh)
# Single-node interchange intervention (scripts/mib/eval_mib_actpatch.py): 2 + 2N FORWARDS per batch
# over 200 train pairs, N = 157 nodes on gpt2 / 361 on qwen2.5-0.5B, i.e. 63k / 144k sequences.
# Forwards, not backwards, so cheaper per unit than the column's gradient rows -- but it is the
# brute-force baseline and the count is the point: it only ever runs on the two small models.
COST_ACTPATCH = "63--144k"
COST_UGS = "7--114k"         # the 12 mask samples per step are what make this so large
COST_OURS = {"node": "0.5k", "edge": "5k"}
# Per-dir overrides for rows that do not train the level's default step count: the 10x-steps
# twins (submit_node_unif_10x_sc.sh: 5000 steps; submit_edge_unif_10x_sc.sh: 50000). Without
# this the edge 10x row printed the headline's 5k, i.e. a 10x run costing what 1x costs.
COST_OURS_DIR = {"mib_node_topk_uniform_lr05_5k": "5k", "mib_edge_topk_uniform_lr05_50k": "50k"}


def ours_cost(results_dir, level):
    return COST_OURS_DIR.get(results_dir, COST_OURS[level])


# === Learning-rate column ====================================================================
#
# Every trained row in this table carries an LR, and until 2026-08-24 none of them showed it --
# which stopped being survivable once the \ourmethod{} rows were repointed to their own swept
# optima (OUR_METHODS, above): the table then had rows at 0.05, 1.0 and 3.0 stacked on top of
# each other, all reading as if they shared a setting, and the whole argument of that repoint
# ("the optimizer does not matter once tuned, the LR it is tuned at does") was invisible.
# tabs/lr_sweep.tex and figs/lr_sweep_summary.pdf are where the sweeps live; this column is the
# pointer from a headline number back into them.
#
# NOTHING HERE IS HARDCODED PER ROW. Our rows read the `lr` out of the saved `args` in
# results/<dir>/<task>_<model>_scores.pt -- the run's own record of what it did -- and the mask
# baselines read the `_lr<x>` suffix that run_edge_pruning.sbatch:118 puts in the dir name
# ("Different lr = different circuit, so it must not share a dir"), falling back to the gate's
# documented default when there is no suffix. A hand-maintained {dir: lr} dict would go stale on
# the next repoint in exactly the silent way this column exists to prevent.
LR_GATE_DEFAULT = {           # eval_mib_edge_pruning.py:160, when --lr is not passed
    "hard_concrete": 0.8, "sigmoid": 1e-3, "dcm": 1e-1,
}
# UGS is not ours and not run_edge_pruning's: ~/optimalablation/edge_pruning_unif_mib.py:149-159
# derives its LR from reg_lamb and the ablation type rather than taking a flag, and
# scripts/mib/launch/run_ugs.sbatch fixes both (LAMB=0.001, `-e cf`) -> 5e-2 for lamb > 5e-4, then /5 for
# cf. Recompute from those two lines if either is ever changed; it cannot be read off disk.
LR_UGS = 5e-2 / 5


def fmt_lr(v):
    return "---" if v is None else f"{v:g}"


_ours_lr_cache = {}


def ours_lr(results_dir):
    """The LR the cells of results/<dir> were actually trained at, from their saved args.

    Returns the shared value when all cells agree. When they DO NOT, returns every value
    present, sorted -- e.g. "0.01/0.1". That is not a formatting flourish: the node REINFORCE
    dirs really do hold one cell (ioi/llama3, the capped-eval one) trained 10x below the other
    ten, and a column that printed the mode would launder a mixed row into a clean one.
    """
    if results_dir in _ours_lr_cache:
        return _ours_lr_cache[results_dir]
    seen = {}
    for task, model, _ in COLUMNS:
        p = RESULTS_BASE / results_dir / f"{task}_{model}_scores.pt"
        if not p.exists():
            continue
        try:
            args = torch.load(p, map_location="cpu", weights_only=False).get("args")
        except Exception:
            continue
        args = args if isinstance(args, dict) else vars(args)
        lr = args.get("lr")
        if lr is not None:
            seen.setdefault(lr, []).append(f"{task}/{model}")
    if not seen:
        out = None
    elif len(seen) == 1:
        out = fmt_lr(next(iter(seen)))
    else:
        odd = {k: v for k, v in seen.items() if len(v) < max(len(x) for x in seen.values())}
        print(f"  MIXED LR in results/{results_dir}: "
              + ", ".join(f"{fmt_lr(k)} on {len(v)} cell(s)" for k, v in sorted(seen.items()))
              + f" -- off-mode cells: {sorted(c for v in odd.values() for c in v)}")
        out = "/".join(fmt_lr(k) for k in sorted(seen))
    _ours_lr_cache[results_dir] = out
    return out


def eprun_lr(results_dir):
    """The mask LR of a run_edge_pruning.sbatch dir, from its `_lr<x>` suffix or the gate default."""
    for part in results_dir.split("_"):
        if part.startswith("lr") and part[2:]:
            try:
                return fmt_lr(float(part[2:]))
            except ValueError:
                pass
    gate = ("sigmoid" if "_sig" in results_dir else
            "dcm" if "_dcm" in results_dir else "hard_concrete")
    return fmt_lr(LR_GATE_DEFAULT[gate])
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
                    "softlog_sgd_lr_1.0", "softuni_sgd_lr_3.0",
                    "mib_node_topk_uniform_lr05_eps1e-2",   # submit_mib_node_eps_sc.sh caps it too
                    "mib_node_topk_uniform_lr05_5k",        # submit_node_unif_10x_sc.sh likewise
                    "mib_node_topkid_uniform_frozen"}       # submit_mib_node_frozen_sc.sh likewise
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
    """[(display, {(task, model): AUC}, results_dir)], one row per variant that has any results.

    The dir is returned alongside the data because the caller needs it for the LR column and
    the display name it keys rows by ("Node Pruning ($s{=}0.5$, logit-diff)") does not encode
    the LR -- eprun_label() drops everything but the sparsity and objective.

    Partial variants ARE shown -- a half-finished sweep is visible progress. But note the
    dashes mean something different here than for UGS: UGS is in PARTIAL_COVERAGE because it
    genuinely cannot run those cells, whereas a dashed Node Pruning cell just has not finished
    yet. The count is printed so an in-progress row is never mistaken for a final one, and the
    Avg column of a partial row averages only the cells present.
    """
    rows = []
    show = EPRUN_SHOW.get(level) if EPRUN_SHOW is not None else None
    for suffix, dirn in EPRUN_SPARSITIES:
        if show is not None and dirn not in show:
            continue
        data = load_run_eval(dirn, f"EdgePruning_patching_{level}")
        if not data:
            continue
        label = eprun_label(level, show[dirn] if show else suffix)
        if len(data) < len(COLUMNS):
            print(f"  NOTE {label}: {len(data)}/{len(COLUMNS)} cells ({dirn}) -- still running")
        rows.append((label, data, dirn))
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


# Per-cell heat, white at a column's minimum and CELL_HI at its maximum.
#
# *** NORMALISED PER (COLUMN, LEVEL), NOT GLOBALLY. *** CPR AUC is not comparable across columns
# or across levels -- node ioi/gpt2 spans 0.25--1.85 while edge ioi/gpt2 spans 0.30--10.59 -- so
# one shared scale would paint the entire node section white and say nothing. The colour answers
# "where does this method sit among the methods, in this cell", which is the only comparison the
# metric supports. It therefore cannot be read across a row: a dark cell in the Arithmetic column
# and a dark cell in the ARC (C) column are both column-winners at different absolute scores.
#
# LIGHT RAMP ON PURPOSE. Every cell carries black text, some of it bold or underlined, so the
# darkest end has to stay well above the legibility floor; ColorBrewer's light blue (relative
# luminance 0.55) is about as dark as this can go before \mathbf on a 6pt digit starts to fill in.
CELL_HI = (0x92, 0xC5, 0xDE)


def cell_color(v, rng):
    """Hex for one cell, or None where there is nothing to shade."""
    if v is None or rng is None:
        return None
    lo, hi = rng
    t = 0.0 if hi <= lo else (v - lo) / (hi - lo)
    return "%02X%02X%02X" % tuple(round(255 + t * (c - 255)) for c in CELL_HI)


def fmt(v, bold=False, underline=False, dagger=False, color=None):
    """One table cell. Numbers are MATH mode -- so the digits, the \\mathbf of a column winner
    and the dagger all set in the same face as the rest of the paper's numerals, instead of the
    text figures \\textbf gave. \\cellcolor must lead the cell, before any content."""
    if v is None:
        return "---"
    s = f"{v:.2f}"
    if bold:
        s = f"\\mathbf{{{s}}}"
    elif underline:
        s = f"\\underline{{{s}}}"
    if dagger:
        s = "^{\\dagger}" + s
    s = f"${s}$"
    return f"\\cellcolor[HTML]{{{color}}}{s}" if color else s


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
                  f"results/{results_dir} await scripts/mib/reeval_gemma_mib.py")
        all_results[key] = data

    # Find best per column per level
    node_ours = [(n, d, l, g) for n, d, l, g in OUR_METHODS if l == "node" and g == "ours"]
    node_uniform = [(n, d, l, g) for n, d, l, g in OUR_METHODS if l == "node" and g == "uniform"]
    edge_ours = [(n, d, l, g) for n, d, l, g in OUR_METHODS if l == "edge" and g == "ours"]
    edge_uniform = [(n, d, l, g) for n, d, l, g in OUR_METHODS if l == "edge" and g == "uniform"]

    def best_in_col(level):
        baselines = {**NODE_BASELINES, **MASK_NODE_BASELINES, **CAUSAL_NODE_BASELINES} \
            if level == "node" else {**EDGE_BASELINES, **MASK_EDGE_BASELINES}
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

    # The k-schedule mark ("$+$ log $k$" / "$+$ unif $k$") is added by mattr_variants.mark_k,
    # which knows which schedule is the headline; the OUR_METHODS names are RELATIVE labels.

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

    def range_in_col(level):
        """{(task, model): (min, max)} + "avg", over every row that level will RENDER.

        Deliberately the same membership rule as best_in_col -- baselines, mask learners and our
        rows -- and, like it, CALLED FROM THE SECTION BODY rather than up here: the baseline
        dicts are still empty at this point in main(), so a range computed now would be taken
        over our rows alone and every baseline would clip to white or to full saturation.

        The Avg entry follows section_avg_best and spans COMPLETE rows only. A partial row's Avg
        is suppressed at render time, so including it would stretch the scale to fit a number
        the table never prints.
        """
        baselines = {**NODE_BASELINES, **MASK_NODE_BASELINES, **CAUSAL_NODE_BASELINES} \
            if level == "node" else {**EDGE_BASELINES, **MASK_EDGE_BASELINES}
        dicts = list(baselines.values()) + [all_results.get(mkey(d, l, g), {})
                                            for _, d, l, g in OUR_METHODS if l == level]
        rng = {}
        for task, model, _ in COLUMNS:
            vals = [v for v in (d.get((task, model)) for d in dicts) if v is not None]
            rng[(task, model)] = (min(vals), max(vals)) if vals else None
        avs = [a for a in (row_avg(d) for d in dicts if len(d) == len(COLUMNS))
               if a is not None]
        rng["avg"] = (min(avs), max(avs)) if avs else None
        return rng

    # display name -> LR string, populated where each row's RESULTS DIR is in scope. Same shape
    # as DAGGER, and for the same reason: the baseline dicts are keyed by display name, so by
    # the time the render loop sees a row the dir it came from is gone. emit_ours passes lr=
    # directly instead (it has the dir, and its labels are rewritten by unifk()).
    ROW_LR = {}

    def make_row(name, data, best_col, second_col, indent=False, dagger=None,
                 avg_best=None, avg_second=None, suppress_avg=False, cost=None, lr=None,
                 crange=None):
        dcells = dagger if dagger is not None else DAGGER.get(name, set())
        cr = crange or {}
        vals = []
        for task, model, _ in COLUMNS:
            v = data.get((task, model))
            is_best = v is not None and best_col.get((task, model)) == v
            is_second = v is not None and not is_best and second_col.get((task, model)) == v
            vals.append(fmt(v, bold=is_best, underline=is_second,
                            dagger=(task, model) in dcells,
                            color=cell_color(v, cr.get((task, model)))))
        # A row that covers only some cells (UGS: 3 of 11) gets no average -- it would not
        # be comparable to the full-coverage rows.
        a = None if (suppress_avg or name in PARTIAL_COVERAGE) else row_avg(data)
        vals.append(fmt(a, bold=(a is not None and a == avg_best),
                        underline=(a is not None and a != avg_best and a == avg_second),
                        color=cell_color(a, cr.get("avg"))))
        prefix = f"\\quad {name}" if indent else name
        # The gradient rows are UNTRAINED, so their LR cell is "---" in the same sense as a
        # missing result: there is no such number, not one we failed to look up.
        lr = lr if lr is not None else ROW_LR.get(name)
        return f"{prefix} & {lr or '---'} & {cost or '---'} & " + " & ".join(vals) + " \\\\"

    # Rows whose IG grid is neither 5 nor 1 declare their own cost in NAPIG_STEP_ROWS.
    STEP_COST = {disp: cost for disp, _, cost in NAPIG_STEP_ROWS}

    def grad_cost(name):
        if name in STEP_COST:
            return STEP_COST[name]
        return COST_GRAD_IG5 if name in COST_IG5_ROWS else COST_GRAD_IG1

    def mask_cost(name):
        if name == "UGS":
            return COST_UGS
        return COST_EPRUN_EDGE if name.startswith(EPRUN_NAME["edge"]) else COST_EPRUN

    def emit_ours(uniform_list, ours_list, level, best, second, avb, avs, dagger=None,
                  crange=None):
        # Split the "Ours" rows into two optimizer sets, each with a header. SGD IS THE UNMARKED
        # DEFAULT AT BOTH LEVELS as of 2026-08-26; Adam is the annotated ablation.
        #
        # Why SGD: at node level the two arms, each read at its own bracketed optimum, TIE (SGD
        # lr=1.0 -> 1.886, Adam lr=0.05 -> 1.879). Once it is a tie, SGD is the better default --
        # it is LR-invariant by construction (zero init, no momentum), so the headline row is the
        # more reproducible one, and one optimizer across both levels is one fewer caveat.
        #
        # WHAT THIS COSTS AT EDGE LEVEL, stated plainly because the number is not small.
        # submit_mib_edge_lr_sweep.sh brackets both arms at edge scale (4/11 cells, paired) and
        # Adam wins tuned-vs-tuned: Adam lr=0.1 -> 7.65 (lr=0.05 -> 7.59) against SGD lr=3.0 ->
        # 6.89. Worse, the edge SGD dirs on disk are at lr=1.0 -- the node optimum carried over,
        # not an edge argmax -- which that sweep measures at 4.72. On the full test table the bare
        # edge row therefore reads 4.96 where Adam's reads 6.23: this flip understates our own
        # edge result by ~1.27 CPR. It is a presentation choice (uniform default across levels),
        # NOT a claim that SGD is the stronger edge optimizer.
        #
        # THE FIX IS NOT TO FLIP BACK. It is to finish the held esgd3-* wave (lr=3.0) so the edge
        # SGD rows sit at their own optimum like every other row in this table; only 4/11
        # validation cells and 0 test cells exist there today. Repoint the dirs when it lands.
        # HEADLINE AND ORDER COME FROM scripts/mib/mattr_variants.py (2026-09-15): the block of
        # the headline optimizer is emitted first and unmarked, and within a block the headline
        # k-schedule's rows come first and unmarked while the other schedule's rows carry its
        # mark. Everything above this line about SGD-as-default is history the module records.
        for opt, label in MV.OPT_ORDER:
            rows_u = [(n, r, g) for n, r, _, g in uniform_list if opt_of(r) == opt]
            rows_o = [(n, r, g) for n, r, _, g in ours_list if opt_of(r) == opt]
            if not rows_u and not rows_o:
                continue
            lines.append(f"\\textbf{{{label}}} \\\\")
            blocks = {"ours": rows_o, "uniform": rows_u}
            rows_o = []   # emitted below via `blocks`, in MV.GROUP_ORDER, each with its k mark
            rows_u = [(MV.mark_k(n, MV.GROUP_K[grp]), r, g)
                      for grp in MV.GROUP_ORDER for n, r, g in blocks[grp]]
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
                                      cost=ours_cost(r, level), lr=ours_lr(r), crange=crange))
            for n, r, g in rows_u:
                dg = IOI_LLAMA_DAGGER if r in IOI_LLAMA_CAPPED else dagger
                d = all_results.get(mkey(r, level, g), {})
                if not d:
                    print(f"SKIP row {n!r} ({r}, {level}/{g}): no results yet")
                    continue
                lines.append(make_row(n, d, best, second,
                                      indent=True, dagger=dg, avg_best=avb, avg_second=avs,
                                      suppress_avg=len(d) < len(COLUMNS),
                                      cost=ours_cost(r, level), lr=ours_lr(r), crange=crange))

    # Generate LaTeX
    ncols = len(COLUMNS)
    lines = []
    lines.append("\\begin{adjustbox}{max width=\\textwidth}")
    # Columns 2-3 are the two config columns (LR, then training cost), so every cmidrule below
    # is shifted by two: the first task column is 4, not 2.
    # CENTRED, not right-aligned, since the cells took a background colour. \cellcolor paints the
    # whole cell including \tabcolsep, so a right-aligned number sits hard against the right edge
    # of its own colour block with all the slack on the left -- which reads as a misalignment
    # rather than as alignment. Every value here is two decimals of the same width, so the
    # decimal points still line up; centring costs nothing and the swatches become a grid.
    lines.append("\\begin{tabular}{lcc@{\\quad}" + "c" * ncols + "@{\\quad}c}")
    lines.append("\\toprule")
    lines.append("& & & \\multicolumn{4}{c}{IOI} & \\multicolumn{2}{c}{Arithmetic} & \\multicolumn{3}{c}{MCQA} & \\multicolumn{2}{c}{ARC (E)} & ARC (C) & \\\\")
    lines.append("\\cmidrule(lr){4-7} \\cmidrule(lr){8-9} \\cmidrule(lr){10-12} \\cmidrule(lr){13-14} \\cmidrule(lr){15-15}")
    header = ("\\textbf{Method} & \\textbf{LR} & \\textbf{Bwd.} & "
              + " & ".join(h for _, _, h in COLUMNS) + " & \\textbf{Avg} \\\\")
    lines.append(header)

    # === Node-level section ===
    # Every node-level baseline in this section (NAP-IG and its step variants, the Tilde
    # methods, the mask learners) is scored by a runner that caps llama3 at --head 200, so they
    # all share one dagger set. Hoisted above the NAP-IG block because that block now needs it
    # too; it used to be defined further down, next to its first use.
    TILDE_LLAMA3_DAGGER = {(t, m) for t, m, _ in COLUMNS if m == "llama3"}
    lines.append("\\midrule")
    lines.append(f"\\multicolumn{{{ncols + 4}}}{{l}}{{\\textit{{Node-level}}}} \\\\")
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
    # Expected Gradients (EAP-IG-inputs-mc, alpha ~ U(0,1) per example, one backward): MIB-side
    # dir, same as the acc-AUC and test tables already carry. Cost is the I x G budget.
    eg = load_eval_dual("napig_mc_eval", "EAP-IG-inputs-mc_patching_node")
    if eg:
        NODE_BASELINES["Expected Gradients"] = eg
        DAGGER["Expected Gradients"] = TILDE_LLAMA3_DAGGER

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
    for name, data, dirn in eprun_rows("node"):
        MASK_NODE_BASELINES[name] = data
        DAGGER[name] = TILDE_LLAMA3_DAGGER
        ROW_LR[name] = eprun_lr(dirn)
    # pyvene sigmoid mask -- same runner, so same --head 200 llama3 subset and same dagger.
    for name, dirn in SIGMOID_MASK_ROWS:
        data = load_run_eval(dirn, "EdgePruning_patching_node")
        if not data:
            continue
        if len(data) < len(COLUMNS):
            print(f"  NOTE {name}: {len(data)}/{len(COLUMNS)} cells ({dirn}) -- still running")
        MASK_NODE_BASELINES[name] = data
        DAGGER[name] = TILDE_LLAMA3_DAGGER
        ROW_LR[name] = eprun_lr(dirn)

    # Interchange intervention (IntInv): eval_mib-format pkls (<task>_<model>_validation.pkl).
    for name, dirn in ACTPATCH_NODE_ROWS:
        data = {}
        for task, model, _ in COLUMNS:
            v = load_cpr_auc(dirn, task, model)
            if v is not None:
                data[(task, model)] = round(v, 2)
        if data:
            CAUSAL_NODE_BASELINES[name] = data

    # Recompute best after adding repro
    best_node, second_node = best_in_col("node")
    range_node = range_in_col("node")
    node_dicts = list(NODE_BASELINES.values()) + list(MASK_NODE_BASELINES.values()) \
        + list(CAUSAL_NODE_BASELINES.values()) \
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
                              cost=grad_cost(name), crange=range_node))
    if MASK_NODE_BASELINES:
        lines.append("\\textbf{Mask learning} \\\\")
        for name, data in MASK_NODE_BASELINES.items():
            lines.append(make_row(name, data, best_node, second_node, indent=True,
                                  avg_best=avb, avg_second=avs,
                                  suppress_avg=len(data) < len(COLUMNS),
                                  cost=mask_cost(name), crange=range_node))
    if CAUSAL_NODE_BASELINES:
        lines.append("\\textbf{Interchange intervention} \\\\")
        for name, data in CAUSAL_NODE_BASELINES.items():
            lines.append(make_row(name, data, best_node, second_node, indent=True,
                                  avg_best=avb, avg_second=avs, dagger=set(),
                                  cost=COST_ACTPATCH, crange=range_node))
    emit_ours(node_uniform, node_ours, "node", best_node, second_node, avb, avs,
              crange=range_node)

    # === Edge-level section ===
    lines.append("\\midrule")
    lines.append(f"\\multicolumn{{{ncols + 4}}}{{l}}{{\\textit{{Edge-level}}}} \\\\")

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
    eg = load_eval_dual("eapig_mc_eval", "EAP-IG-inputs-mc_patching_edge")   # as at node level
    if eg:
        EDGE_BASELINES["Expected Gradients"] = eg
        DAGGER["Expected Gradients"] = TILDE_LLAMA3_DAGGER

    ugs = load_run_eval(UGS_DIR, "UGS_patching_edge")
    if ugs:
        MASK_EDGE_BASELINES["UGS"] = ugs
        ROW_LR["UGS"] = fmt_lr(LR_UGS)
    for name, data, dirn in eprun_rows("edge"):
        MASK_EDGE_BASELINES[name] = data
        DAGGER[name] = TILDE_LLAMA3_DAGGER
        ROW_LR[name] = eprun_lr(dirn)

    best_edge, second_edge = best_in_col("edge")
    range_edge = range_in_col("edge")
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
                              cost=grad_cost(name), crange=range_edge))
    # Mask learners rank by a learned gate rather than a gradient, so they get their own header.
    if MASK_EDGE_BASELINES:
        lines.append("\\textbf{Mask learning} \\\\")
        for name, data in MASK_EDGE_BASELINES.items():
            lines.append(make_row(name, data, best_edge, second_edge, indent=True,
                                  avg_best=eavb, avg_second=eavs,
                                  suppress_avg=len(data) < len(COLUMNS),
                                  cost=mask_cost(name), crange=range_edge))
    emit_ours(edge_uniform, edge_ours, "edge", best_edge, second_edge, eavb, eavs,
              dagger=EDGE_LLAMA_DAGGER, crange=range_edge)

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
