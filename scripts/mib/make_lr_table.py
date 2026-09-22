"""LaTeX tables: how learning rate influences node CPR AUC -- and, in a third table with the
same blocks, node acc-AUC -- per task, for each method we swept LR on (hard_topk / MAttr,
bernoulli_reinforce / +hard bwd, and the pyvene sigmoid-mask baseline).

Reads results/<dir>/<task>_<model>_validation.pkl, or MIB's own
<dir>/**/<task-with-dashes>_<model>_validation_abs-*.pkl for the eprun_eval_* dirs.
lr-sweep dirs (htk_lr_*, bern_lr_*) currently only hold ioi/gpt2; the lr=0.01 baselines
hold all tasks.
Run from repo root on sc:  uv run python scripts/mib/make_lr_table.py
"""
import json
import pickle
from pathlib import Path

RESULTS_BASE = Path("results")
# Two tables, split by which knob the ROWS of a block vary. Together they were one table of
# 12 blocks / ~60 rows, which only fits by page-breaking -- and a longtable cannot be scaled
# to the 5.5in \textwidth (adjustbox typesets into a single box, and boxes cannot break
# across pages). Split, each half fits one page, so each can go back to a plain
# adjustbox+tabular in a float and the width takes care of itself.
LR_OUTPUT = Path("paper/tabs/lr_sweep.tex")
SPARSITY_OUTPUT = Path("paper/tabs/sparsity_sweep.tex")
# Same blocks as LR_OUTPUT, same pkls, read at "acc_auc" instead of "area_under". A third
# table rather than extra columns in the first: 13 columns already need adjustbox to fit
# \textwidth, and CPR and acc-AUC do not share a scale (CPR is unbounded and ~1 at chance,
# acc-AUC is in [0,1]), so interleaving them would put two units under one set of headers.
# Worth having separately because the two metrics disagree about which LR wins -- acc-AUC is
# log-x weighted and so is dominated by the sparse end, where these methods actually differ
# (see the cpr-auc-is-dense-end-dominated note: ~90% of a CPR AUC is k>=20%, where they tie).
LR_ACCAUC_OUTPUT = Path("paper/tabs/lr_sweep_accauc.tex")

# The LR table's 11 MIB validation cells. The LR sweeps were run before arithmetic_addition
# was wired in (2026-09) and none of their dirs has that cell, so adding the column here would
# suppress the Avg of every LR row (render() prints no Avg below full coverage). It stays at
# 11 until the LR sweeps are rerun on addition; the sparsity table is at 12, below.
COLUMNS = [
    ("ioi", "gpt2", "GPT"), ("ioi", "qwen2.5", "Qwen"), ("ioi", "gemma2", "Gemma"),
    ("ioi", "llama3", "Llama"), ("arithmetic_subtraction", "llama3", "Llama"),
    ("mcqa", "qwen2.5", "Qwen"), ("mcqa", "gemma2", "Gemma"), ("mcqa", "llama3", "Llama"),
    ("arc_easy", "gemma2", "Gemma"), ("arc_easy", "llama3", "Llama"),
    ("arc_challenge", "llama3", "Llama"),
]
# The sparsity table's 12 cells: the same 11 plus arithmetic_addition/llama3, which both
# sparsity ladders (Node Pruning s, DBM L1) were run on in 2026-09
# (scripts/mib/launch/submit_arith_add_sweep_sc.sh). This is the column set of
# mib_results.tex, so the sweep figure (plots/plot_sparsity_sweep_summary.py) averages over
# the same cells as the headline table.
SPARSITY_COLUMNS = (COLUMNS[:4]
                    + [("arithmetic_addition", "llama3", "Llama ($+$)"),
                       ("arithmetic_subtraction", "llama3", "Llama ($-$)")]
                    + COLUMNS[5:])
assert [c[:2] for c in SPARSITY_COLUMNS if c[0] != "arithmetic_addition"] == [c[:2] for c in COLUMNS]

# Task -> header group label. Consecutive columns with the same group share one
# \multicolumn and one \cmidrule, so both column sets render from the same code.
TASK_GROUP = {"ioi": "IOI", "arithmetic_addition": "Arithmetic",
              "arithmetic_subtraction": "Arithmetic", "mcqa": "MCQA",
              "arc_easy": "ARC (E)", "arc_challenge": "ARC (C)"}


def header_lines(columns, stub):
    r"""\toprule through \midrule for `columns`. Task columns start at 2 (col 1 is the row
    label) and Avg is the rightmost column, matching mib_results.tex / mib_test_results.tex /
    mib_accauc_results.tex, so a reader moving between tables finds the summary in the same
    place. Arithmetic's group spans one column in the LR table and two in the sparsity table;
    a lone column gets a plain header cell, not a 1-wide \multicolumn."""
    groups = []  # [group, first_col, last_col]
    for i, (t, _, _) in enumerate(columns, start=2):
        g = TASK_GROUP[t]
        if groups and groups[-1][0] == g:
            groups[-1][2] = i
        else:
            groups.append([g, i, i])
    cells = [f"\\multicolumn{{{b - a + 1}}}{{c}}{{{g}}}" if b > a else g for g, a, b in groups]
    rules = " ".join(f"\\cmidrule(lr){{{a}-{b}}}" for _, a, b in groups)
    return ["\\toprule",
            "& " + " & ".join(cells) + " & \\\\",
            rules,
            f"\\textbf{{{stub}}} & " + " & ".join(h for _, _, h in columns) + " & \\textbf{Avg} \\\\",
            "\\midrule"]

# method -> list of (lr-label, results-dir), optionally followed by the row-label prefix for
# blocks that sweep something other than the learning rate (default "LR$=$").
# lr=0.01 dirs are the main runs (all tasks).
# MAttr headline = soft top-k fwd, log k; "+ hard" = sigmoid-STE hard forward.
#
# Table 1: blocks whose rows vary the LEARNING RATE at a fixed sparsity setting.
LR_METHODS = [
    # HEADLINE since 2026-09-15 (mattr_variants.HEADLINE): soft top-k, UNIFORM k, Adam. The
    # sweep dirs are submit_topkuni_lr.sh's (11 cells, no arithmetic_addition) plus the headline
    # lr=0.05 dir (12 cells); there is no uniform-k Adam run at lr=0.01. Relabelled 2026-09-20:
    # this block was "$+$ unif $k$" while the log-k block below was the bare \ourmethod{}.
    ("\\ourmethod{}", [
        ("0.005", "topkuni_lr_0.005"), ("0.05", "mib_node_topk_uniform_lr05"),
        ("0.1", "topkuni_lr_0.1"), ("0.3", "topkuni_lr_0.3"),
    ]),
    ("$+$ log $k$", [
        ("0.005", "topklog_lr_0.005"), ("0.01", "mib_node_topk_log"),
        ("0.05", "topklog_lr_0.05"), ("0.1", "topklog_lr_0.1"), ("0.3", "topklog_lr_0.3"),
    ]),
    # Same soft top-k forward AND same sigmoid-slope backward as the block above; only the
    # optimizer differs. Included because the three MAttr arms on disk never isolated the
    # optimizer -- the id-STE arm flips the backward and switches to SGD at once -- so this is
    # the block that says whether Adam's per-parameter normalisation is doing the work.
    # submit_softlog_sgd_lr.sh. COVERAGE IS UNEVEN AND THE AVG COLUMN IS THE ONLY SAFE READ:
    # the first submission was killed after 10/55 jobs, so 0.005 and 0.01 are 5/11 cells
    # (ioi gpt2/llama3/qwen2.5 + mcqa llama3/qwen2.5) and permanently so -- they were NOT
    # resubmitted, because at 5/11 they already lose to Adam on every one and the SVA neuron
    # substrate puts this arm's optimum at lr>=1. The resubmission runs 0.05/0.1/0.3/1.0 on all
    # 11. render() suppresses Avg below full coverage, so the two short rows show cells but no
    # Avg; do not hand-average them against the 11-cell rows.
    #
    # lr=1.0 is OUTSIDE the 0.005-0.3 grid every other block in this table uses. It is here
    # because at the SVA neuron substrate soft+SGD peaks at lr=1 (acc-AUC 0.496 vs soft+Adam's
    # 0.361, results/sva_mlp_lr) -- i.e. the shared grid was chosen for Adam and there is no
    # reason it brackets SGD's optimum. If 1.0 wins, the row is not comparable to the Adam rows
    # at equal lr and needs saying so in prose.
    #
    # 3.0/10.0 exist to BRACKET that 1.0 peak from above, which they do: on acc-AUC both lose
    # to 1.0 nearly cell-for-cell (2/11 and 1/9 wins). Read them on acc-AUC, not on CPR. The
    # CPR deficit at 3.0 is one cell -- ioi/gpt2 drops 1.84 -> 0.25 because the two top-ranked
    # nodes (`input`, `m0`) get the largest gradients, so at 3x the step they overshoot furthest
    # past zero, saturate the gate (sp = m(1-m) -> 0), and never come back; `input` ends ranked
    # last, so every circuit below k=100% reads corrupted input and sits at the corrupt floor.
    # Excluding that one cell, 3.0's CPR mean vs 1.0 is +0.009 at 6/10 -- i.e. a wash, and the
    # bracketing claim rests on acc-AUC alone. Do not read the CPR Avg here as an LR effect.
    ("$+$ log $k$, $+$ SGD", [
        ("0.005", "softlog_sgd_lr_0.005"), ("0.01", "softlog_sgd_lr_0.01"),
        ("0.05", "softlog_sgd_lr_0.05"), ("0.1", "softlog_sgd_lr_0.1"),
        ("0.3", "softlog_sgd_lr_0.3"), ("1.0", "softlog_sgd_lr_1.0"),
        ("3.0", "softlog_sgd_lr_3.0"), ("10.0", "softlog_sgd_lr_10.0"),
    ]),
    # submit_softuni_sgd_lr.sh. Crosses the k-schedule with the optimizer: "$+$ SGD" above is
    # soft/log/SGD and the "$+$ unif $k$" row of mib_results.tex is soft/uniform/Adam, so
    # neither says whether the uniform-k and SGD effects are the same effect. They look alike
    # on disk -- both score better on CPR-AUC and worse on acc-AUC than the headline -- which
    # is exactly the dense-end/sparse-end split, so the corner is needed to separate them.
    #
    # THE ANSWER, from this block against the one above: they are NOT the same effect and are
    # separable. Moving the optimizer (Adam -> SGD) moves acc-AUC and CPR-AUC the SAME way;
    # moving the k-schedule (log -> uniform) TRADES them -- uniform-k is best-in-sweep on
    # CPR-AUC (9/11 vs the Adam headline) and among the worst on acc-AUC. That is the
    # dense-end/sparse-end split showing up as a k-schedule effect, not an optimizer one.
    #
    # This arm is also far FLATTER in LR than log-k: 3.0 is -0.010 CPR / -0.003 acc vs 1.0,
    # against log-k's -0.15 / -0.035, and it has not blown up on any cell. Consistent with it
    # spending most steps at large k, where the gate stays far from saturation and so cannot
    # take the overshoot-past-zero path that kills log-k's ioi/gpt2 at 3.0.
    ("$+$ SGD", [
        ("0.05", "softuni_sgd_lr_0.05"), ("0.1", "softuni_sgd_lr_0.1"),
        ("0.3", "softuni_sgd_lr_0.3"), ("1.0", "softuni_sgd_lr_1.0"),
        ("3.0", "softuni_sgd_lr_3.0"), ("10.0", "softuni_sgd_lr_10.0"),
    ]),
    ("$+$ log $k$, $+$ hard", [
        ("0.005", "htklog_lr_0.005"), ("0.01", "mib_node_hard_topk_log"),
        ("0.05", "htklog_lr_0.05"), ("0.1", "htklog_lr_0.1"), ("0.3", "htklog_lr_0.3"),
    ]),
    ("$+$ hard", [
        ("0.005", "htk_lr_0.005"), ("0.01", "mib_node_hard_topk"),
        ("0.05", "htk_lr_0.05"), ("0.1", "htk_lr_0.1"), ("0.3", "htk_lr_0.3"),
    ]),
    ("$+$ hard bwd (REINFORCE)", [
        ("0.01", "bern_lr_0.01"),                  # baseline (orig dir was overwritten by lr0.1 rerun)
        ("0.05", "bern_lr_0.05"),
        ("0.1", "mib_node_bernoulli_reinforce"),   # lr0.1 rerun = the main +hard row (all tasks)
        ("0.3", "bern_lr_0.3"),
        ("0.1, 2k", "bern_lr_0.1_2k"),
    ]),
    # pyvene's SigmoidMaskIntervention baseline. Its published lr=1e-3 was chosen for a few
    # rotation parameters at one intervention site; here it drives 156--1056 gate logits
    # against a task loss, so it is swept like everything else. Only the 0.001 row runs all
    # 11 cells -- the sweep is on the three cheap gpt2/qwen2.5 cells (submit_sigmoid_mask_lr.sh).
    ("DBM", [
        ("0.001 (pyvene)", "eprun_eval_ld_sig"),
        ("0.01", "eprun_eval_ld_sig_lr0.01"),
        ("0.1", "eprun_eval_ld_sig_lr0.1"),
        ("0.3", "eprun_eval_ld_sig_lr0.3"),   # added to bracket the 0.1 peak against 1.0
        ("1.0", "eprun_eval_ld_sig_lr1.0"),
    ]),
    # Node Pruning's rows in mib_results.tex are a SPARSITY sweep at ONE learning rate, so
    # "Node Pruning underperforms \ourmethod{}" rested on its default LR being a good one.
    # These two blocks close that gap (submit_node_pruning_lr.sh). Budgets s=0.5 and s=0.8 are
    # the best and second-best logit-diff rows by Avg CPR, and they bracket the peak.
    #
    # The 0.8 row is the EXISTING default-LR run, not a new one: 0.8 is the hard-concrete
    # default. That differs from the sigmoid default of 1e-3, so this grid is deliberately NOT
    # the DBM grid -- each gate is swept around its own default rather than on a shared one.
    # How those two budgets were picked is visible in the sparsity table, whose Node Pruning
    # block sweeps s at this same LR; its s=0.5 and s=0.8 rows ARE the "0.8 (default)" rows
    # below, so the two tables agree exactly there by construction.
    ("Node Pruning ($s{=}0.5$, logit-diff)", [
        ("0.1", "eprun_eval_s0.5_ld_lr0.1"),
        ("0.3", "eprun_eval_s0.5_ld_lr0.3"),
        ("0.8 (default)", "eprun_eval_s0.5_ld"),
        ("1.5", "eprun_eval_s0.5_ld_lr1.5"),
        ("3.0", "eprun_eval_s0.5_ld_lr3.0"),
    ]),
    ("Node Pruning ($s{=}0.8$, logit-diff)", [
        ("0.1", "eprun_eval_s0.8_ld_lr0.1"),
        ("0.3", "eprun_eval_s0.8_ld_lr0.3"),
        ("0.8 (default)", "eprun_eval_s0.8_ld"),
        ("1.5", "eprun_eval_s0.8_ld_lr1.5"),
        ("3.0", "eprun_eval_s0.8_ld_lr3.0"),
    ]),
]

# Table 2: blocks whose rows (or, for DCM, whose blocks) vary a SPARSITY knob -- the budget
# s, the L1 coefficient, or the pinned density. Separated from the LR table because together
# they were one ~60-row table that could only fit by page-breaking, which in turn ruled out
# scaling it to \textwidth.
SPARSITY_METHODS = [
    # The Node Pruning sparsity sweep at the default LR. It is the companion to the two
    # "Node Pruning ($s{=}...$)" blocks in the LR table, which are each CONDITIONED on a
    # budget: without this block a reader cannot see how those two budgets were picked, or
    # how much of the block-to-block spread there is the budget rather than the LR. Same 11
    # cells and same objective, so the three blocks stay directly comparable across tables.
    #
    # NOT independent of them: the s=0.5 and s=0.8 rows here ARE the "0.8 (default)" rows of
    # those two blocks -- same dirs, listed twice on purpose so each block reads as a complete
    # sweep around its own centre. Averages therefore agree exactly across the two tables;
    # that is a consistency check, not a duplicated run.
    #
    # The KL sparsity sweep (eprun_eval_s0.5 / _s0.8 / _s0.95 / _s0.99 and the s=0.9 default
    # `eprun_eval`) is deliberately NOT here: two of its dirs are 6/11 and 9/11, so it would
    # render with suppressed Avgs, and the LR blocks are all logit-diff anyway.
    #
    # s > 1 IS A LEGAL SETTING, NOT A TYPO (added 2026-08-25). `s` is the L0 anneal's Lagrangian
    # TARGET, not a sparsity: edge_pruning.py:149-151 ascends both multipliers, so s>1 makes the
    # density constraint permanently violated and the sparsity pressure unbounded and still
    # correctly signed. It is how the grid probes past the ~0.86 achieved-sparsity saturation
    # that node-pruning-l0-anneal-steps documents. These two rows are what BRACKET the sweep --
    # without them every acc-AUC curve was still climbing at the right-hand edge and the figure
    # could only say "the grid does not contain the optimum".
    ("Node Pruning (logit-diff, LR $=$ 0.8)", [
        ("0.1", "eprun_eval_s0.1_ld"),
        ("0.25", "eprun_eval_s0.25_ld"),
        ("0.5", "eprun_eval_s0.5_ld"),
        ("0.8", "eprun_eval_s0.8_ld"),
        ("0.9", "eprun_eval_s0.9_ld"),
        ("0.95", "eprun_eval_s0.95_ld"),
        ("0.99", "eprun_eval_s0.99_ld"),
        ("1.25", "eprun_eval_s1.25_ld"),
        # 8/11 as of 2026-08-25 (ioi/gemma2, arc_easy/llama3, arc_challenge/llama3 still
        # training), so its Avg renders suppressed here and the point is DROPPED entirely from
        # plot_sparsity_sweep_summary.py, which requires 12/12 per metric. Listed anyway so the
        # row appears the moment the cells land; do not read its per-cell numbers as a mean.
        ("2.0", "eprun_eval_s2.0_ld"),
    ], "$s{=}$"),
    # DBM with the sparsity penalty it is normally trained with (submit_dbm_l1.sh). The DBM
    # rows in the LR table have none, which is faithful to the pyvene *library* but not to how
    # pyvene trains this class: its own tutorial uses loss + 1.0*||mask||_1, and Boundless DAS
    # uses 2.0*intervention_boundaries.sum(). The penalty here is Boundless DAS's --
    # coeff*z.mean(), i.e. L1 on the density -- because their `intervention_boundaries` scalar
    # IS the density, so the constant transfers and 2.0 is a published default, not a guess.
    # (The mask tutorial's ||mask||_1 penalises pre-sigmoid logits that init at 0, so it drives
    # gates to z=0.5 -- toward the ~50% density the unpenalised rows already show. It cannot
    # sparsify; L1_TARGET=logit runs it if we ever want that row.)
    #
    # Swept at lr=0.3, the best of the five DBM LRs by Avg CPR over all 11 cells, so the
    # penalty is not confounded with a bad LR. The 0 row is the existing lr=0.3 run reused as
    # the control, not a new job.
    ("DBM $+$ L1 (lr $=$ 0.3)", [
        ("0 (no penalty)", "eprun_eval_ld_sig_lr0.3"),
        ("0.2", "eprun_eval_ld_sig_lr0.3_l10.2"),
        ("0.6", "eprun_eval_ld_sig_lr0.3_l10.6"),
        ("2.0 (Boundless DAS)", "eprun_eval_ld_sig_lr0.3_l12.0"),
        ("6.0", "eprun_eval_ld_sig_lr0.3_l16.0"),
        ("20.0", "eprun_eval_ld_sig_lr0.3_l120.0"),
        # Extended 2026-09-08/09 (the L1 ladder eval_dbm_multisparsity.L1S walks): three more
        # rungs past the CPR peak at 6, each x1.5-3 apart, so the sparse end of the knob is
        # bracketed instead of stopping one point after the turnover. All 11 validation cells
        # of each are on disk; wired into this table and plots/plot_sparsity_sweep_summary.py
        # (which imports this block) on 2026-09-15.
        ("40.0", "eprun_eval_ld_sig_lr0.3_l140.0"),
        ("60.0", "eprun_eval_ld_sig_lr0.3_l160.0"),
        ("200.0", "eprun_eval_ld_sig_lr0.3_l1200.0"),
    ], "$\\lambda_{\\mathrm{L1}}{=}$"),
    # --- DCM: PULLED FROM THE PAPER 2026-08-13 and its port removed 2026-09-22. Do not re-add
    # without reading this. ---
    #
    # Three blocks used to live here (pinned density 1/5/20%, five LRs each, dirs
    # results/eprun_eval_ld_dcm_d{0.01,0.05,0.2}_lr*). The runs are still on disk; the port
    # (edge_pruning.py:learn_scores_dcm, --gate dcm, collect_dcm_sweep.py, dcm_rank_agreement.py)
    # is in git history before 2026-09-22. Three reasons, in order of weight:
    #
    # 1. THE NUMBER WE PRINTED CANNOT BE PRODUCED BY DCM. Every block in this table reports
    #    `area_under`, but DCM's mask saturates -- 91-100% of units land at exactly 0.0 or
    #    1.0, fully degenerate on 2 of 3 cells at the 1% pin -- so a sweep needs an order for
    #    the two blocks, and that order is OUR pruning-order tie-break (edge_pruning.py:507),
    #    not the method's. Upstream never sorts and gets its sparsity curve by retraining per
    #    lambda (DCM.py:144). So a DCM CPR-AUC is an artifact of our port; a reviewer who
    #    knows the method would be right to object.
    # 2. It made the artifact visible in the worst way: `area_under` is a LINEAR trapezoid
    #    over 0.001..1.0 (evaluation.py:63), ~90% of it from k >= 20% where every method is
    #    near-identical, so collapsed-mask runs (ZERO units kept) outscored converged ones.
    #    The $\varnothing$ marks flagged this, but the caption never defined the symbol.
    # 3. Coverage: 3 of 12 cells (ioi/gpt2, ioi/qwen2.5, mcqa/qwen2.5), hence an all-`---`
    #    Avg column -- the only block in either table that cannot report one.
    #
    # If it comes back, it must report CPR AT THE PIN (the old collect_dcm_sweep.py computed
    # it; that IS a faithful evaluation of the set DCM emits, and the three pins are the
    # honest analogue of upstream's lambda sweep), and the tie-break must be described as our
    # adaptation. Do NOT re-add it reporting area_under.
]


# llama3/ioi (10k val, 8B) is evaluated on a reduced 200-example subset (daggered). The
# lr=0.01 anchor for that one cell therefore reads the capped rerun, not the full-eval dir.
# llama3/ioi is eval'd on a reduced 200-example subset (daggered) in every MAttr block;
# the lr=0.01 anchor for that cell reads the capped rerun, not the full-eval main dir.
DIR_OVERRIDE = {("mib_node_hard_topk_log", "ioi", "llama3"): "htklog_lr_0.01",
                ("mib_node_topk_log", "ioi", "llama3"): "topklog_lr_0.01",
                ("mib_node_hard_topk", "ioi", "llama3"): "htk_lr_0.01"}
DAGGER_CELLS = {("ioi", "llama3")}  # capped at 200 in all 3 MAttr blocks (not REINFORCE)

# Training steps per block. This is NOT cosmetic: the mask baselines get 3000 steps and every
# MAttr variant gets 500, a 6x budget gap that runs in the BASELINES' favour, so a reader
# comparing block Avgs without it is reading a handicapped-in-our-disfavour comparison as if it
# were matched. Verified from the runs themselves rather than the submit scripts -- last row of
# results/<dir>/*_trainlog.csv is step 499 (1999 for bern_lr_0.1_2k) for the MAttr blocks, and
# logs/eprun_*.out counts to /3000 for every eprun_* dir including the default-LR rows.
STEPS = {
    "\\ourmethod{}": "500 steps",
    "$+$ log $k$": "500 steps",
    "$+$ log $k$, $+$ SGD": "500 steps",
    # submit_softuni_sgd_lr.sh passes --steps 500, same as every other MAttr block. Without an
    # entry here the block renders with NO step note while the blocks around it carry one,
    # which reads as "unknown/unmatched budget" for the one block whose whole job is to be
    # matched to the block above.
    "$+$ SGD": "500 steps",
    "$+$ log $k$, $+$ hard": "500 steps",
    "$+$ hard": "500 steps",
    "$+$ hard bwd (REINFORCE)": "500 steps; 2000 in the last row",
    "DBM": "3000 steps",
    "DBM $+$ L1 (lr $=$ 0.3)": "3000 steps",
    "Node Pruning (logit-diff, LR $=$ 0.8)": "3000 steps",
    "Node Pruning ($s{=}0.5$, logit-diff)": "3000 steps",
    "Node Pruning ($s{=}0.8$, logit-diff)": "3000 steps",
    "DCM (pinned density $=$ 1\\%)": "3000 steps",
    "DCM (pinned density $=$ 5\\%)": "3000 steps",
    "DCM (pinned density $=$ 20\\%)": "3000 steps",
}


def steps_note(method):
    """Upright, small parenthetical after the italic block header. A block with no STEPS entry
    renders exactly as before, so adding a block does not silently claim a step count."""
    v = STEPS.get(method)
    return "" if v is None else f"\\quad{{\\footnotesize ({v})}}"


# Legacy acc-AUC re-evaluations, in the MIB tree. Runs scored before MIB's evaluation.py
# started returning acc_auc have acc_auc=None in their eval_mib pkl -- that is every lr=0.01
# main-run dir (mib_node_*), htk_lr_*, and bern_lr_*, i.e. most of two blocks and all of a
# third. Those cells were re-evaluated later into these two folders, so the values EXIST and a
# table that showed "---" there would be reporting a gap in its own routing as a gap in the
# results. Same fallback chain, same folders and same layout as
# make_mib_accauc_table.acc_mattr, so the two acc-AUC tables agree cell-for-cell by
# construction rather than by coincidence.
#
# Mixing an ordinary eval with a re-eval is safe and was checked there rather than assumed:
# over every cell present in both sources the values agree to 2dp, and the largest CPR AUC
# difference between any method's _eval and its _accauc rerun is 0.02 -- run-to-run
# nondeterminism, not a different evaluation setting.
#
# CPR never needs this (area_under is in every pkl ever written), so the chain is only
# consulted for acc_auc and the default table is byte-identical to before.
from learning_to_attribute.deps import mib_results_dir   # deps/MIB-circuit-track/results, else the Tilde path
ACCAUC_FALLBACK_ROOTS = [
    mib_results_dir() / "mattr_accauc",
    mib_results_dir() / "mattr_accauc_val",
]


def _read(p, key):
    """`key` from one pkl, or None if the file, the key, or the value is absent.

    An older pkl can carry the key with a None VALUE rather than not carrying it at all;
    both mean "this metric was not computed for this run" and both must fall through to the
    next source in the chain, so they are collapsed here instead of relying on round() to
    raise and be swallowed.
    """
    if not p.exists():
        return None
    try:
        v = pickle.load(open(p, "rb"))[key]
    except Exception:
        return None
    return None if v is None else round(v, 2)


def cpr(d, task, model, key="area_under"):
    """The pkl's `key` for one cell, rounded to the 2dp the table prints.

    `key` selects the metric: "area_under" is CPR AUC (the default, what every block was
    written for), "acc_auc" the log-x-weighted decision accuracy in [0,1]. Both live in the
    SAME pkl for anything evaluated recently -- acc_auc comes free with the CPR run -- so the
    acc-AUC table needs no new evaluation pass; older runs come from the fallback roots above.
    """
    d = DIR_OVERRIDE.get((d, task, model), d)
    stask = task.replace("_", "-")
    v = _read(RESULTS_BASE / d / f"{task}_{model}_validation.pkl", key)
    if v is None:
        # MIB's own run_evaluation.py (what the eprun_eval_* dirs come from) writes a
        # different layout: <dir>/EdgePruning_patching_node/<task-with-dashes>_<model>_
        # validation_abs-False.pkl. Same keys inside.
        for p in (RESULTS_BASE / d).glob(f"**/{stask}_{model}_validation_abs-*.pkl"):
            v = _read(p, key)
            if v is not None:
                break
    if v is None and key == "acc_auc":
        for root in ACCAUC_FALLBACK_ROOTS:
            v = _read(root / f"{d}_patching_node"
                      / f"{stask}_{model}_validation_abs-False.pkl", key)
            if v is not None:
                break
    return v


def empty_circuit(d, task, model):
    """True if a DCM run's PID missed its pin badly enough to leave round(mask) empty.

    Only DCM can hit this. Its scores are 0/1 plus a pruning-order tie-break, so when no
    unit survives, MIB's top-k still returns k units -- ordered by when they died. The
    resulting AUC is a property of the training trajectory, not of any circuit the method
    converged to, and empirically it is HIGHER than a successful run's. Left in the table
    (the run happened, and hiding it would misrepresent the sweep) but marked and excluded
    from the per-column best, so it can never be read as the winning learning rate.
    """
    if "_dcm_" not in d:
        return False
    graph = RESULTS_BASE / d.replace("eprun_eval_", "eprun_node_") / f"graph_{task}_{model}.json"
    try:
        with open(graph) as f:
            nodes = json.load(f)["nodes"]
    except (OSError, ValueError, KeyError):
        return False
    scores = [v["score"] for v in nodes.values() if isinstance(v, dict) and "score" in v]
    # `input` is forced into every circuit and is not one of the maskable units.
    return bool(scores) and sum(1 for s in scores if s >= 0.5) - 1 == 0


def fmt(v, bold=False, dagger=False, empty=False):
    if v is None:
        return "---"
    s = f"\\textbf{{{v:.2f}}}" if bold else f"{v:.2f}"
    if empty:
        return "$^{\\varnothing}$" + s
    return ("$^{\\dagger}$" + s) if dagger else s


def render(methods, output, stub, key="area_under", columns=COLUMNS):
    """Emit one adjustbox+tabular fragment, to be \\input inside a table float.

    No float, no caption, no \\label here: this file is overwritten on every run, so anything
    a human would want to edit has to live next to the \\input in sections/detailed-mib.tex
    instead, or it gets silently reverted the next time a sweep finishes.

    `stub` is the top-left header cell -- the two tables sweep different knobs, so calling
    both columns "LR" would mislabel half the rows of the sparsity one.

    `key` is the pkl metric (see cpr): the CPR and acc-AUC tables are the SAME blocks,
    the same bolding and the same daggers, read from the same pkls at a different key.

    `columns` is the cell set (COLUMNS for the LR tables, SPARSITY_COLUMNS for the sparsity
    one); full coverage of it is what gates the Avg column.
    """
    ncols = len(columns)
    # Both tables render the same blocks, so every SKIP/WARNING below fires twice per run
    # with identical text. Tag them or the acc-AUC table's coverage holes read as duplicates
    # of the CPR table's, which they are not -- acc_auc is missing from strictly more cells.
    tag = "acc-AUC" if key == "acc_auc" else "CPR"
    # adjustbox rather than a hand-tuned \footnotesize: 13 columns want ~500pt against a
    # 5.5in (~397pt) \textwidth, and `max width` shrinks to fit whatever the content turns
    # out to be instead of relying on a font-size guess. It works here only because each of
    # these two tables fits on one page -- adjustbox typesets into a single box, and a box
    # cannot break across pages, which is why the combined 12-block version had to be a
    # longtable and could not be scaled at all.
    lines = ["\\begin{adjustbox}{max width=\\textwidth}",
             "\\begin{tabular}{l" + "r" * ncols + "@{\\quad}r}"]
    lines += header_lines(columns, stub)

    emitted = 0
    for entry in methods:
        method, lrs = entry[0], entry[1]
        prefix = entry[2] if len(entry) > 2 else "LR$=$"
        data = {lr: {(t, m): cpr(d, t, m, key) for t, m, _ in columns} for lr, d in lrs}
        empty = {lr: {(t, m): empty_circuit(d, t, m) for t, m, _ in columns} for lr, d in lrs}
        # A block whose only populated row is the control (an existing run reused as the
        # sweep's zero point) is not yet a sweep -- it would render as one row of numbers
        # over four rows of "---". Skip it until a second point lands; it then appears on
        # the next regeneration with no edit here.
        if sum(any(v is not None for v in data[lr].values()) for lr, _ in lrs) < 2:
            print(f"[{tag}] SKIP block {method!r}: <2 populated rows (jobs still pending)")
            continue
        if emitted:
            lines.append("\\midrule")
        emitted += 1
        best = {}
        for t, m, _ in columns:
            # Collapsed DCM runs are excluded here, not just marked: they routinely score
            # above the runs that hit their pin, so leaving them in would bold an empty
            # circuit as the block's best learning rate.
            vals = [data[lr][(t, m)] for lr, _ in lrs
                    if data[lr][(t, m)] is not None and not empty[lr][(t, m)]]
            best[(t, m)] = max(vals) if len(vals) > 1 else None  # only bold when there's a sweep
        # 3 MAttr blocks cap llama/ioi at 200 val examples, and so does the sigmoid-mask
        # block (run_edge_pruning.sbatch passes --head 200); the REINFORCE runs do not.
        is_capped = "REINFORCE" not in method
        lines.append(f"\\multicolumn{{{ncols + 2}}}{{l}}{{\\textit{{{method}}}"
                     f"{steps_note(method)}}} \\\\")
        for lr, _ in lrs:
            present = [data[lr][(t, m)] for t, m, _ in columns if data[lr][(t, m)] is not None]
            # Same rule as the block-level skip above, one level down: a row with no populated
            # cells renders as 12 "---" and reads as an LR that was run and scored nothing,
            # which is a wrong claim rather than a gap. Drop it until its first cell lands; it
            # reappears on the next regeneration with no edit here. Announced, never silent.
            if not present:
                print(f"[{tag}] SKIP row {method} {prefix}{lr}: no cells yet")
                continue
            # An Avg over populated cells only is NOT comparable to the row above it when the two
            # rows have different cell counts, and the bias is not even zero-mean: the columns that
            # go missing are the slow llama3 ones, which are also the high-CPR ones, so a partial
            # row reads as a worse LR than it is. Suppress it until the row is complete rather than
            # print a number that invites exactly the comparison it cannot support.
            if len(present) == len(columns):
                avg = f"{sum(present) / len(present):.2f}"
            else:
                avg = "---"
                if present:
                    print(f"[{tag}] WARNING: {method} {prefix}{lr} has {len(present)}/{len(columns)} cells; "
                          f"Avg suppressed (missing "
                          f"{[f'{t}/{m}' for t, m, _ in columns if data[lr][(t, m)] is None]})")
            cells = [fmt(data[lr][(t, m)],
                         bold=(data[lr][(t, m)] is not None and not empty[lr][(t, m)]
                               and data[lr][(t, m)] == best[(t, m)]),
                         dagger=(is_capped and (t, m) in DAGGER_CELLS and data[lr][(t, m)] is not None),
                         empty=empty[lr][(t, m)])
                     for t, m, _ in columns]
            # An Avg over cells that are all empty circuits is an average of trajectory
            # rankings; mark it so the block-level number carries the same warning as the
            # cells it came from, rather than laundering it into a clean-looking mean.
            n_empty = sum(1 for t, m, _ in columns
                          if empty[lr][(t, m)] and data[lr][(t, m)] is not None)
            if n_empty:
                print(f"[{tag}] WARNING: {method} {prefix}{lr} has {n_empty}/{len(present)} cells whose "
                      f"circuit is EMPTY; those scores are the pruning-order tie-break")
                if avg != "---" and n_empty == len(present):
                    avg = "$^{\\varnothing}$" + avg
            lines.append(f"\\quad {prefix}{lr} & " + " & ".join(cells) + f" & {avg} \\\\")

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}"]
    table = "\n".join(lines) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(table)
    print(f"Wrote {output} ({emitted} blocks, {len(lines)} lines)\n")
    print(table)


def main():
    render(LR_METHODS, LR_OUTPUT, "Method / LR")
    render(SPARSITY_METHODS, SPARSITY_OUTPUT, "Method / sparsity", columns=SPARSITY_COLUMNS)
    render(LR_METHODS, LR_ACCAUC_OUTPUT, "Method / LR", key="acc_auc")


if __name__ == "__main__":
    main()
