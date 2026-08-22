"""LaTeX tables: how learning rate influences EDGE-level CPR AUC and acc-AUC.

The edge counterpart to make_lr_table.py. Same blocks, same bolding, same dagger convention,
same two-metric split -- so the edge and node LR tables can be read against each other without
the reader having to check whether a difference is a table-construction difference.

FOUR COLUMNS, AND THAT IS FINAL -- DO NOT "BACKFILL" IT. submit_mib_edge_lr_sweep.sh defines
nine cells; the qwen2.5 (24 jobs) and gemma2 (36 jobs) arms were CANCELLED on 2026-08-22 with
the sweep's question already answered, to free the 2-node QOS cap for the stepless-IG wave.
Those five columns are a deliberate scope decision, not jobs that are still pending, so a
future reader who sees four columns and resubmits gemma2 is spending ~113 GPU-hours to widen
a table whose conclusion does not depend on it. If they are ever wanted, the recipe is
`ONLY=gemma2 bash scripts/submit_mib_edge_lr_sweep.sh` and this script picks them up with no
edit -- COLUMNS below already lists all nine.

WHAT THE SWEEP ANSWERED, and why four cells were enough:
 - Adam's imported lr=0.05 is essentially at the argmax. CPR-AUC peaks at 0.1 on three of the
   four cells and 0.05 is within 0.15 everywhere, so the paper's edge numbers do not rest on a
   badly chosen LR and no headline re-run is needed.
 - SGD's edge optimum is lr ~ 3-10, an ORDER OF MAGNITUDE above the node table's argmax of 1.0
   -- which is what the n/k scaling predicts, since edges outnumber nodes. Read at the node's
   LR the SGD block looks broken (0.3/1.0 sit at half Adam's CPR-AUC on every llama3 cell);
   read at its own optimum it is competitive, Avg 6.89 at lr=3 vs Adam's best 7.65, and it
   BEATS Adam on arithmetic_subtraction (6.34 at lr=10 vs Adam's best 4.78). An earlier draft
   of this docstring concluded "SGD does not transfer to llama3" from the truncated 0.3/1.0
   grid; that was an artifact of reading an LR-invariant optimizer off a grid tuned for a
   different unit count. Do not reintroduce it, and do not compare the two blocks row-by-row
   at the same numeric LR -- only block-argmax vs block-argmax means anything here.

TWO CEILING EFFECTS THAT DECIDE WHICH TABLE TO READ:
 - acc-AUC SATURATES at 1.00 on llama3 ioi and mcqa for most settings, so in those columns it
   cannot discriminate learning rates at all and every tie gets bolded. The LR argmax there is
   a CPR-AUC statement or it is nothing.
 - Conversely CPR-AUC is the dense-end-dominated metric (a linear trapezoid, ~90% of it from
   k >= 20% where methods tie), so a CPR-AUC win that acc-AUC does not corroborate is a win at
   sparsities nobody reads a circuit at. On ioi/gpt2 the two metrics disagree in SIGN about
   SGD vs Adam. Report both; when they disagree, say so rather than picking one.

HOW FINELY THE LR CAN BE READ: not very. ioi/llama3 under SGD goes 6.83 -> 5.30 -> 6.38 at
lr 3/10/30, which is not a shape any LR story explains -- it bounds run-to-run variance at
roughly +/-0.8 CPR-AUC on that cell. Single-run gaps below ~1 point are not LR effects. There
are no seed replicates in this sweep, so that non-monotonicity is the only variance estimate
available; treat every block-argmax as "somewhere in this range", not a tuned value.

5000 STEPS, not the node table's 500 -- see STEPS below. Do not compare a block Avg here
against a block Avg in lr_sweep.tex as if the budgets matched.

Run from repo root:  uv run python scripts/make_edge_lr_table.py
"""
import pickle
import re
from pathlib import Path

RESULTS_BASE = Path("results")
LR_OUTPUT = Path("paper/tabs/edge_lr_sweep.tex")
LR_ACCAUC_OUTPUT = Path("paper/tabs/edge_lr_sweep_accauc.tex")

# All nine cells submit_mib_edge_lr_sweep.sh defines, in its PAIRS order. The five that were
# cancelled render as all-"---" ROWS and are dropped by the empty-column filter in render(),
# so listing them costs nothing today and makes a future gemma2/qwen2.5 top-up a zero-edit
# regeneration rather than a hunt for where the column list lives.
COLUMNS = [
    ("ioi", "gpt2", "GPT"), ("ioi", "qwen2.5", "Qwen"), ("ioi", "gemma2", "Gemma"),
    ("ioi", "llama3", "Llama"), ("arithmetic_subtraction", "llama3", "Llama"),
    ("mcqa", "qwen2.5", "Qwen"), ("mcqa", "gemma2", "Gemma"), ("mcqa", "llama3", "Llama"),
    ("arc_easy", "gemma2", "Gemma"),
]
# Column group headers, keyed by task, in COLUMNS order. Built at render time from whichever
# columns survive so a cancelled arm cannot leave a \multicolumn spanning nothing.
TASK_LABEL = {"ioi": "IOI", "arithmetic_subtraction": "Arith", "mcqa": "MCQA",
              "arc_easy": "ARC (E)"}

# (block label, [(lr-label, results dir)]). Dirs are submit_mib_edge_lr_sweep.sh's
# "mib_edge_lrsweep_${opt}_${sched}_lr_${lr}" -- optimizer AND lr in the name, because
# eval_mib_edge.py's output filenames encode neither and two sweep points sharing an --output
# would overwrite each other while the sweep looked complete.
#
# The two grids are deliberately DIFFERENT and are not a mistake to be "aligned": MAttr+SGD is
# LR-invariant by construction (zero init, no momentum), so its useful LR scales like n/k and
# the Adam grid does not bracket its optimum. Edges have far more units than nodes, which is
# why the SGD grid runs an order of magnitude higher here than the node table's.
LR_METHODS = [
    ("\\ourmethod{}", [(lr, f"mib_edge_lrsweep_adam_log_lr_{lr}")
                       for lr in ("0.005", "0.01", "0.05", "0.1", "0.3", "1.0")]),
    # 3.0 and 10.0 are complete. Two rows are permanently partial and their Avg is suppressed:
    # 30.0 is missing mcqa/llama3 and 100.0 is gpt2-only, both because those jobs were in the
    # cancelled batch. Extending the grid further up is NOT worth resubmitting -- lr=30 is
    # already past the argmax on every populated cell, and ioi/gpt2's acc-AUC has collapsed to
    # 0.81-0.83 there from 0.98 at lr<=3.
    ("$+$ SGD", [(lr, f"mib_edge_lrsweep_sgd_log_lr_{lr}")
                 for lr in ("0.3", "1.0", "3.0", "10.0", "30.0", "100.0")]),
]

# Every llama3 cell here is scored on 200 validation examples (submit_mib_edge_lr_sweep.sh sets
# ev=200 for llama3, ev=0 elsewhere), per the CLAUDE.md cap. That is the whole llama3 COLUMN,
# unlike the node table where only ioi/llama3 is capped -- so the dagger set is by model.
DAGGER_MODELS = {"llama3"}

# 5000, from the submit script's --steps 5000. The node blocks in lr_sweep.tex get 500. Stated
# for the same reason that table states its own: a reader comparing block Avgs across the two
# tables without it is comparing a 10x budget difference as if it were a granularity effect.
STEPS = {"\\ourmethod{}": "5000 steps", "$+$ SGD": "5000 steps"}


def steps_note(method):
    v = STEPS.get(method)
    return "" if v is None else f"\\quad{{\\footnotesize ({v})}}"


def read(d, task, model, key):
    """`key` for one cell, rounded to the 2dp the table prints; None if absent.

    A pkl can carry the key with a None VALUE rather than omitting it, and both mean "not
    computed"; collapsed here so callers only test for None.
    """
    p = RESULTS_BASE / d / f"{task}_{model}_validation.pkl"
    if not p.exists():
        return None
    try:
        v = pickle.load(open(p, "rb"))[key]
    except Exception:
        return None
    return None if v is None else round(v, 2)


def fmt(v, bold=False, dagger=False):
    if v is None:
        return "---"
    s = f"\\textbf{{{v:.2f}}}" if bold else f"{v:.2f}"
    return ("$^{\\dagger}$" + s) if dagger else s


def header_lines(cols, stub):
    """Group header + cmidrules, spanning only the columns that actually have data.

    Built rather than hardcoded because the surviving column set depends on which arms were
    run: a literal "\\multicolumn{4}{c}{IOI}" written for the nine-cell sweep would span three
    llama3/qwen2.5 columns that are not there and silently mislabel the row.
    """
    groups = []
    for t, m, h in cols:
        if groups and groups[-1][0] == t:
            groups[-1][1].append(h)
        else:
            groups.append((t, [h]))
    span, rules, at = [], [], 3  # cols 1-2 are the stub and Avg
    for t, hs in groups:
        lab = TASK_LABEL.get(t, t)
        span.append(f"\\multicolumn{{{len(hs)}}}{{c}}{{{lab}}}" if len(hs) > 1 else lab)
        rules.append(f"\\cmidrule(lr){{{at}-{at + len(hs) - 1}}}")
        at += len(hs)
    return ["\\toprule",
            "& & " + " & ".join(span) + " \\\\",
            " ".join(rules),
            f"\\textbf{{{stub}}} & \\textbf{{Avg}} & "
            + " & ".join(h for _, _, h in cols) + " \\\\",
            "\\midrule"]


def render(methods, output, stub, key="area_under"):
    """Emit one adjustbox+tabular fragment, to be \\input inside a table float.

    No float, no caption, no \\label: this file is overwritten on every run, so anything a
    human would edit has to live next to the \\input in the section file or it is silently
    reverted the next time the sweep is regenerated.
    """
    tag = "acc-AUC" if key == "acc_auc" else "CPR"
    data = {m: {lr: {(t, mo): read(d, t, mo, key) for t, mo, _ in COLUMNS}
                for lr, d in lrs} for m, lrs in methods}

    # Drop columns no block populated. A cancelled arm would otherwise render as a column of
    # "---" under a real header, which reads as "this cell was run and scored nothing" -- a
    # wrong claim, where an absent column is merely a narrower table.
    cols = [c for c in COLUMNS
            if any(data[m][lr][(c[0], c[1])] is not None for m, lrs in methods for lr, _ in lrs)]
    dropped = [f"{t}/{m}" for t, m, _ in COLUMNS if (t, m) not in {(c[0], c[1]) for c in cols}]
    if dropped:
        print(f"[{tag}] columns with no data, omitted: {', '.join(dropped)}")

    ncols = len(cols)
    lines = ["\\begin{adjustbox}{max width=\\textwidth}",
             "\\begin{tabular}{lr@{\\quad}" + "r" * ncols + "}"]
    lines += header_lines(cols, stub)

    emitted = 0
    for method, lrs in methods:
        d_m = data[method]
        if sum(any(d_m[lr][(t, mo)] is not None for t, mo, _ in cols) for lr, _ in lrs) < 2:
            print(f"[{tag}] SKIP block {method!r}: <2 populated rows")
            continue
        if emitted:
            lines.append("\\midrule")
        emitted += 1
        best = {}
        for t, mo, _ in cols:
            vals = [d_m[lr][(t, mo)] for lr, _ in lrs if d_m[lr][(t, mo)] is not None]
            best[(t, mo)] = max(vals) if len(vals) > 1 else None  # only bold across a sweep
        lines.append(f"\\multicolumn{{{ncols + 2}}}{{l}}{{\\textit{{{method}}}"
                     f"{steps_note(method)}}} \\\\")
        for lr, _ in lrs:
            present = [d_m[lr][(t, mo)] for t, mo, _ in cols if d_m[lr][(t, mo)] is not None]
            if not present:
                print(f"[{tag}] SKIP row {method} LR={lr}: no cells yet")
                continue
            # An Avg over populated cells only is not comparable to a full row above it, and
            # the bias is not zero-mean: the cells that go missing are the slow llama3 ones,
            # which are also the high-CPR ones, so a partial row reads as a worse LR than it
            # is. Suppress rather than print a number that invites that comparison.
            if len(present) == ncols:
                avg = f"{sum(present) / len(present):.2f}"
            else:
                miss = [f"{t}/{mo}" for t, mo, _ in cols if d_m[lr][(t, mo)] is None]
                avg = "---"
                print(f"[{tag}] WARNING: {method} LR={lr} has {len(present)}/{ncols} cells; "
                      f"Avg suppressed (missing {miss})")
            cells = [fmt(d_m[lr][(t, mo)],
                         bold=(d_m[lr][(t, mo)] is not None and d_m[lr][(t, mo)] == best[(t, mo)]),
                         dagger=(mo in DAGGER_MODELS and d_m[lr][(t, mo)] is not None))
                     for t, mo, _ in cols]
            lines.append(f"\\quad LR$=${lr} & {avg} & " + " & ".join(cells) + " \\\\")

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}"]
    table = "\n".join(lines) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(table)
    print(f"Wrote {output} ({emitted} blocks, {ncols} columns)\n")
    print(table)


def main():
    render(LR_METHODS, LR_OUTPUT, "Method / LR")
    render(LR_METHODS, LR_ACCAUC_OUTPUT, "Method / LR", key="acc_auc")


if __name__ == "__main__":
    main()
