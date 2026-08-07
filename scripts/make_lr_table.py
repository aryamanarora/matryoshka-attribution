"""LaTeX table: how learning rate influences node CPR AUC, per task, for each method
we swept LR on (hard_topk / MAttr, bernoulli_reinforce / +hard bwd, and the pyvene
sigmoid-mask baseline).

Reads results/<dir>/<task>_<model>_validation.pkl, or MIB's own
<dir>/**/<task-with-dashes>_<model>_validation_abs-*.pkl for the eprun_eval_* dirs.
lr-sweep dirs (htk_lr_*, bern_lr_*) currently only hold ioi/gpt2; the lr=0.01 baselines
hold all tasks.
Run from repo root on sc:  uv run python scripts/make_lr_table.py
"""
import json
import pickle
from pathlib import Path

RESULTS_BASE = Path("results")
OUTPUT = Path("paper/tabs/lr_sweep.tex")

COLUMNS = [
    ("ioi", "gpt2", "GPT"), ("ioi", "qwen2.5", "Qwen"), ("ioi", "gemma2", "Gemma"),
    ("ioi", "llama3", "Llama"), ("arithmetic_subtraction", "llama3", "Llama"),
    ("mcqa", "qwen2.5", "Qwen"), ("mcqa", "gemma2", "Gemma"), ("mcqa", "llama3", "Llama"),
    ("arc_easy", "gemma2", "Gemma"), ("arc_easy", "llama3", "Llama"),
    ("arc_challenge", "llama3", "Llama"),
]

# method -> list of (lr-label, results-dir), optionally followed by the row-label prefix for
# blocks that sweep something other than the learning rate (default "LR$=$").
# lr=0.01 dirs are the main runs (all tasks).
# MAttr headline = soft top-k fwd, log k; "+ hard" = sigmoid-STE hard forward.
METHODS = [
    ("\\ourmethod{}", [
        ("0.005", "topklog_lr_0.005"), ("0.01", "mib_node_topk_log"),
        ("0.05", "topklog_lr_0.05"), ("0.1", "topklog_lr_0.1"), ("0.3", "topklog_lr_0.3"),
    ]),
    ("$+$ hard", [
        ("0.005", "htklog_lr_0.005"), ("0.01", "mib_node_hard_topk_log"),
        ("0.05", "htklog_lr_0.05"), ("0.1", "htklog_lr_0.1"), ("0.3", "htklog_lr_0.3"),
    ]),
    ("$+$ unif $k$, $+$ hard", [
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
    # DBM with the sparsity penalty it is normally trained with (submit_dbm_l1.sh). The rows
    # above have none, which is faithful to the pyvene *library* but not to how pyvene trains
    # this class: its own tutorial uses loss + 1.0*||mask||_1, and Boundless DAS uses
    # 2.0*intervention_boundaries.sum(). The penalty here is Boundless DAS's -- coeff*z.mean(),
    # i.e. L1 on the density -- because their `intervention_boundaries` scalar IS the density,
    # so the constant transfers and 2.0 is a published default, not a guess. (The mask
    # tutorial's ||mask||_1 penalises pre-sigmoid logits that init at 0, so it drives gates to
    # z=0.5 -- toward the ~50% density the unpenalised rows already show. It cannot sparsify;
    # L1_TARGET=logit runs it if we ever want that row.)
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
    ], "$\\lambda_{\\mathrm{L1}}{=}$"),
    # Node Pruning's rows in mib_results.tex are a SPARSITY sweep at ONE learning rate, so
    # "Node Pruning underperforms \ourmethod{}" rested on its default LR being a good one.
    # These two blocks close that gap (submit_node_pruning_lr.sh). Budgets s=0.5 and s=0.8 are
    # the best and second-best logit-diff rows by Avg CPR, and they bracket the peak.
    #
    # The 0.8 row is the EXISTING default-LR run, not a new one: 0.8 is the hard-concrete
    # default. That differs from the sigmoid default of 1e-3, so this grid is deliberately NOT
    # the DBM grid -- each gate is swept around its own default rather than on a shared one.
    # The sparsity sweep itself, at the default LR. It belongs in this table because the two LR
    # blocks below are each CONDITIONED on a budget (s=0.5 and s=0.8), so without it a reader
    # cannot see how those two budgets were picked or how much of the block-to-block spread is
    # the budget rather than the LR. Same 11 cells and same objective as the LR blocks, so the
    # three blocks are directly comparable.
    #
    # NOT independent of the blocks below: the s=0.5 and s=0.8 rows here ARE the "0.8 (default)"
    # rows of the two LR blocks -- same dirs, listed twice on purpose so each block reads as a
    # complete sweep around its own centre. Averages will therefore agree exactly across blocks;
    # that is a consistency check, not a duplicated run.
    #
    # The KL sparsity sweep (eprun_eval_s0.5 / _s0.8 / _s0.95 / _s0.99 and the s=0.9 default
    # `eprun_eval`) is deliberately NOT here: two of its dirs are 6/11 and 9/11, so it would
    # render with suppressed Avgs, and the LR blocks are all logit-diff anyway.
    ("Node Pruning (sparsity sweep, logit-diff, LR $=$ 0.8)", [
        ("0.1", "eprun_eval_s0.1_ld"),
        ("0.25", "eprun_eval_s0.25_ld"),
        ("0.5", "eprun_eval_s0.5_ld"),
        ("0.8", "eprun_eval_s0.8_ld"),
        ("0.9", "eprun_eval_s0.9_ld"),
        ("0.95", "eprun_eval_s0.95_ld"),
        ("0.99", "eprun_eval_s0.99_ld"),
    ], "$s{=}$"),
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
    # DCM (Prakash et al. 2024 / roonbug/belief_dynamics): a raw coefficient clamped to
    # [0,1], circuit = round(mask), with a PID controller on the sparsity weight. Unlike
    # every other block here it has no ranking of its own -- clamp_ piles the scores onto
    # exactly 0.0 and 1.0 -- so it can only be trained toward a density, and the density is
    # a THIRD axis on top of LR. Hence three blocks, pinned to three of MIB's own sweep
    # points, where round(mask) and top-k are the same set.
    #
    # 0.1 is the published LR and is identical in Prakash et al. and belief_dynamics, so
    # unlike pyvene's 1e-3 there is a real prior here; the grid brackets it either way.
    #
    # READ THE $\varnothing$ MARKS BEFORE READING THE NUMBERS. Where the PID overshoots, the
    # mask collapses to zero units and MIB still scores the run -- on the pruning-order
    # tie-break, which is a trajectory ranking rather than any circuit DCM converged to.
    # Those scores are not just meaningless but ANTI-correlated with success: the collapsed
    # runs post a HIGHER AUC than the runs that hit their pin. Marked, and never bolded.
    ("DCM (pinned density $=$ 1\\%)", [
        ("0.01", "eprun_eval_ld_dcm_d0.01_lr0.01"),
        ("0.03", "eprun_eval_ld_dcm_d0.01_lr0.03"),
        ("0.1 (published)", "eprun_eval_ld_dcm_d0.01_lr0.1"),
        ("0.3", "eprun_eval_ld_dcm_d0.01_lr0.3"),
        ("1.0", "eprun_eval_ld_dcm_d0.01_lr1.0"),
    ]),
    ("DCM (pinned density $=$ 5\\%)", [
        ("0.01", "eprun_eval_ld_dcm_d0.05_lr0.01"),
        ("0.03", "eprun_eval_ld_dcm_d0.05_lr0.03"),
        ("0.1 (published)", "eprun_eval_ld_dcm_d0.05_lr0.1"),
        ("0.3", "eprun_eval_ld_dcm_d0.05_lr0.3"),
        ("1.0", "eprun_eval_ld_dcm_d0.05_lr1.0"),
    ]),
    ("DCM (pinned density $=$ 20\\%)", [
        ("0.01", "eprun_eval_ld_dcm_d0.2_lr0.01"),
        ("0.03", "eprun_eval_ld_dcm_d0.2_lr0.03"),
        ("0.1 (published)", "eprun_eval_ld_dcm_d0.2_lr0.1"),
        ("0.3", "eprun_eval_ld_dcm_d0.2_lr0.3"),
        ("1.0", "eprun_eval_ld_dcm_d0.2_lr1.0"),
    ]),
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
    "$+$ hard": "500 steps",
    "$+$ unif $k$, $+$ hard": "500 steps",
    "$+$ hard bwd (REINFORCE)": "500 steps; 2000 in the last row",
    "DBM": "3000 steps",
    "DBM $+$ L1 (lr $=$ 0.3)": "3000 steps",
    "Node Pruning (sparsity sweep, logit-diff, LR $=$ 0.8)": "3000 steps",
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


def cpr(d, task, model):
    d = DIR_OVERRIDE.get((d, task, model), d)
    p = RESULTS_BASE / d / f"{task}_{model}_validation.pkl"
    if not p.exists():
        # MIB's own run_evaluation.py (what the eprun_eval_* dirs come from) writes a
        # different layout: <dir>/EdgePruning_patching_node/<task-with-dashes>_<model>_
        # validation_abs-False.pkl. Same "area_under" key inside.
        hits = list((RESULTS_BASE / d).glob(
            f"**/{task.replace('_', '-')}_{model}_validation_abs-*.pkl"))
        if not hits:
            return None
        p = hits[0]
    try:
        return round(pickle.load(open(p, "rb"))["area_under"], 2)
    except Exception:
        return None


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


def main():
    ncols = len(COLUMNS)
    span = ncols + 2
    # longtable, not tabular-in-a-float: at 12 blocks this is ~80 lines, which overruns a
    # page even at \scriptsize, and a `table` float cannot break across pages. That also
    # rules out adjustbox (it cannot break either), so the fit is done with \footnotesize
    # and a tighter \tabcolsep instead of by scaling the whole box.
    #
    # The caption TEXT deliberately does not live here. This file is regenerated on every
    # run, so a caption written into it would be silently reverted the next time the sweep
    # finishes. It is referenced through \lrsweepcaption, which is defined next to the
    # \input in sections/detailed-mib.tex; the \providecommand below is only a fallback so
    # this table still compiles standalone, and is a no-op whenever that definition exists.
    header = ["\\toprule",
              "& & \\multicolumn{4}{c}{IOI} & Arith & \\multicolumn{3}{c}{MCQA} & "
              "\\multicolumn{2}{c}{ARC (E)} & ARC (C) \\\\",
              "\\cmidrule(lr){3-6} \\cmidrule(lr){7-7} \\cmidrule(lr){8-10} "
              "\\cmidrule(lr){11-12} \\cmidrule(lr){13-13}",
              "\\textbf{Method / LR} & \\textbf{Avg} & "
              + " & ".join(h for _, _, h in COLUMNS) + " \\\\",
              "\\midrule"]
    lines = ["\\providecommand{\\lrsweepcaption}{\\textbf{Learning rate sweep of "
             "\\ourmethod{} variants on MIB.}}",
             "\\begingroup",
             "\\footnotesize",
             "\\setlength{\\tabcolsep}{4pt}",
             "\\begin{longtable}{lr@{\\quad}" + "r" * ncols + "}",
             # \normalsize so the caption matches every other table's rather than
             # inheriting the \footnotesize the table body needs to fit the width.
             "\\caption{{\\normalsize\\lrsweepcaption}}",
             "\\label{tab:lr-sweep} \\\\"]
    lines += header + ["\\endfirsthead"]
    lines += [f"\\multicolumn{{{span}}}{{l}}{{\\textit{{Table \\ref{{tab:lr-sweep}}, "
              f"continued from the previous page.}}}} \\\\"]
    lines += header + ["\\endhead"]
    lines += ["\\midrule",
              f"\\multicolumn{{{span}}}{{r}}{{\\textit{{continued on the next page}}}} \\\\",
              "\\endfoot",
              "\\bottomrule",
              "\\endlastfoot"]

    emitted = 0
    for entry in METHODS:
        method, lrs = entry[0], entry[1]
        prefix = entry[2] if len(entry) > 2 else "LR$=$"
        data = {lr: {(t, m): cpr(d, t, m) for t, m, _ in COLUMNS} for lr, d in lrs}
        empty = {lr: {(t, m): empty_circuit(d, t, m) for t, m, _ in COLUMNS} for lr, d in lrs}
        # A block whose only populated row is the control (an existing run reused as the
        # sweep's zero point) is not yet a sweep -- it would render as one row of numbers
        # over four rows of "---". Skip it until a second point lands; it then appears on
        # the next regeneration with no edit here.
        if sum(any(v is not None for v in data[lr].values()) for lr, _ in lrs) < 2:
            print(f"SKIP block {method!r}: <2 populated rows (jobs still pending)")
            continue
        if emitted:
            lines.append("\\midrule")
        emitted += 1
        best = {}
        for t, m, _ in COLUMNS:
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
            present = [data[lr][(t, m)] for t, m, _ in COLUMNS if data[lr][(t, m)] is not None]
            # An Avg over populated cells only is NOT comparable to the row above it when the two
            # rows have different cell counts, and the bias is not even zero-mean: the columns that
            # go missing are the slow llama3 ones, which are also the high-CPR ones, so a partial
            # row reads as a worse LR than it is. Suppress it until the row is complete rather than
            # print a number that invites exactly the comparison it cannot support.
            if len(present) == len(COLUMNS):
                avg = f"{sum(present) / len(present):.2f}"
            else:
                avg = "---"
                if present:
                    print(f"WARNING: {method} {prefix}{lr} has {len(present)}/{len(COLUMNS)} cells; "
                          f"Avg suppressed (missing "
                          f"{[f'{t}/{m}' for t, m, _ in COLUMNS if data[lr][(t, m)] is None]})")
            cells = [fmt(data[lr][(t, m)],
                         bold=(data[lr][(t, m)] is not None and not empty[lr][(t, m)]
                               and data[lr][(t, m)] == best[(t, m)]),
                         dagger=(is_capped and (t, m) in DAGGER_CELLS and data[lr][(t, m)] is not None),
                         empty=empty[lr][(t, m)])
                     for t, m, _ in COLUMNS]
            # An Avg over cells that are all empty circuits is an average of trajectory
            # rankings; mark it so the block-level number carries the same warning as the
            # cells it came from, rather than laundering it into a clean-looking mean.
            n_empty = sum(1 for t, m, _ in COLUMNS
                          if empty[lr][(t, m)] and data[lr][(t, m)] is not None)
            if n_empty:
                print(f"WARNING: {method} {prefix}{lr} has {n_empty}/{len(present)} cells whose "
                      f"circuit is EMPTY; those scores are the pruning-order tie-break")
                if avg != "---" and n_empty == len(present):
                    avg = "$^{\\varnothing}$" + avg
            lines.append(f"\\quad {prefix}{lr} & {avg} & " + " & ".join(cells) + " \\\\")

    # \bottomrule is in \endlastfoot, so longtable draws it once, after the final page.
    lines += ["\\end{longtable}", "\\endgroup"]
    table = "\n".join(lines) + "\n"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(table)
    print(f"Wrote {OUTPUT}\n")
    print(table)


if __name__ == "__main__":
    main()
