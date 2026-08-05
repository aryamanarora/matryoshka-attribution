"""LaTeX table: how learning rate influences node CPR AUC, per task, for each method
we swept LR on (hard_topk / MAttr, bernoulli_reinforce / +hard bwd, and the pyvene
sigmoid-mask baseline).

Reads results/<dir>/<task>_<model>_validation.pkl, or MIB's own
<dir>/**/<task-with-dashes>_<model>_validation_abs-*.pkl for the eprun_eval_* dirs.
lr-sweep dirs (htk_lr_*, bern_lr_*) currently only hold ioi/gpt2; the lr=0.01 baselines
hold all tasks.
Run from repo root on sc:  uv run python scripts/make_lr_table.py
"""
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


# llama3/ioi (10k val, 8B) is evaluated on a reduced 200-example subset (daggered). The
# lr=0.01 anchor for that one cell therefore reads the capped rerun, not the full-eval dir.
# llama3/ioi is eval'd on a reduced 200-example subset (daggered) in every MAttr block;
# the lr=0.01 anchor for that cell reads the capped rerun, not the full-eval main dir.
DIR_OVERRIDE = {("mib_node_hard_topk_log", "ioi", "llama3"): "htklog_lr_0.01",
                ("mib_node_topk_log", "ioi", "llama3"): "topklog_lr_0.01",
                ("mib_node_hard_topk", "ioi", "llama3"): "htk_lr_0.01"}
DAGGER_CELLS = {("ioi", "llama3")}  # capped at 200 in all 3 MAttr blocks (not REINFORCE)


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


def fmt(v, bold=False, dagger=False):
    if v is None:
        return "---"
    s = f"\\textbf{{{v:.2f}}}" if bold else f"{v:.2f}"
    return ("$^{\\dagger}$" + s) if dagger else s


def main():
    ncols = len(COLUMNS)
    lines = ["\\begin{adjustbox}{max width=\\textwidth}",
             "\\begin{tabular}{lr@{\\quad}" + "r" * ncols + "}", "\\toprule",
             "& & \\multicolumn{4}{c}{IOI} & Arith & \\multicolumn{3}{c}{MCQA} & "
             "\\multicolumn{2}{c}{ARC (E)} & ARC (C) \\\\",
             "\\cmidrule(lr){3-6} \\cmidrule(lr){7-7} \\cmidrule(lr){8-10} "
             "\\cmidrule(lr){11-12} \\cmidrule(lr){13-13}",
             "\\textbf{Method / LR} & \\textbf{Avg} & " + " & ".join(h for _, _, h in COLUMNS) + " \\\\",
             "\\midrule"]

    emitted = 0
    for entry in METHODS:
        method, lrs = entry[0], entry[1]
        prefix = entry[2] if len(entry) > 2 else "LR$=$"
        data = {lr: {(t, m): cpr(d, t, m) for t, m, _ in COLUMNS} for lr, d in lrs}
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
            vals = [data[lr][(t, m)] for lr, _ in lrs if data[lr][(t, m)] is not None]
            best[(t, m)] = max(vals) if len(vals) > 1 else None  # only bold when there's a sweep
        # 3 MAttr blocks cap llama/ioi at 200 val examples, and so does the sigmoid-mask
        # block (run_edge_pruning.sbatch passes --head 200); the REINFORCE runs do not.
        is_capped = "REINFORCE" not in method
        lines.append(f"\\multicolumn{{{ncols + 2}}}{{l}}{{\\textit{{{method}}}}} \\\\")
        for lr, _ in lrs:
            present = [data[lr][(t, m)] for t, m, _ in COLUMNS if data[lr][(t, m)] is not None]
            avg = f"{sum(present) / len(present):.2f}" if present else "---"
            cells = [fmt(data[lr][(t, m)],
                         bold=(data[lr][(t, m)] is not None and data[lr][(t, m)] == best[(t, m)]),
                         dagger=(is_capped and (t, m) in DAGGER_CELLS and data[lr][(t, m)] is not None))
                     for t, m, _ in COLUMNS]
            lines.append(f"\\quad {prefix}{lr} & {avg} & " + " & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines += ["\\end{tabular}", "\\end{adjustbox}"]
    table = "\n".join(lines) + "\n"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(table)
    print(f"Wrote {OUTPUT}\n")
    print(table)


if __name__ == "__main__":
    main()
