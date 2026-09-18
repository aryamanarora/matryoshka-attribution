"""MIB node-level results under ZERO ablation, TEST split -- the SVA+ zero-ablation method set
(scripts/sva/launch/submit_zero_5k_sc.sh) applied to MIB by scripts/mib/launch/submit_mib_zero_sc.sh.

Two tables, one per metric, same 12 columns as make_mib_test_table:
    paper/tabs/mib_zero_results.tex          CPR AUC  (area_under)
    paper/tabs/mib_zero_accauc_results.tex   acc-AUC  (acc_auc)

Every method is trained/attributed under zero and scored by MIB's evaluate_area_under_curve with
intervention='zero'. MAttr rows are eval_mib.py pkls (<task>_<model>_test.pkl); every other row
is run_evaluation.py output (<sub>/<task-dash>_<model>_test_abs-False.pkl) with the `_zero_node`
subfolder name that harness gives a zero-ablation run.

NO chance correction here, unlike the SVA+ zero tables: MIB's accuracy is not a binary
base-vs-source preference, and the Random ranking reads ~0.1 acc-AUC under zero, not 0.5.

Same reading conventions as the test table: bold best / underline second per column, Avg only
for a complete row, a partial row prints its dashes (rows are still landing), a row with no
cells is omitted with a printed line.

    uv run python scripts/mib/make_mib_zero_table.py
"""
import pickle
from pathlib import Path

import make_mib_test_table as T
import mattr_variants as MV

RESULTS = Path("results")
TABS = Path("paper/tabs")

# (block title, [(display, kind, dir, sub)]); kind: "ours" = eval_mib layout, "mib" = run_evaluation
BLOCKS = [
    ("\\textbf{Gradient-based}", [
        ("IG ($m{=}10$)",      "mib", "mib_zero_test/ig",      "EAP-IG-inputs_zero_node"),
        ("I$\\times$G",        "mib", "mib_zero_test/ixg",     "EAP-IG-inputs_zero_node"),
        ("Expected Gradients", "mib", "mib_zero_test/eg",      "EAP-IG-inputs-mc_zero_node"),
        ("AttnLRP",            "mib", "mib_zero_test/attnlrp", "AttnLRP_zero_node"),
    ]),
    ("\\textbf{Mask-based}", [
        ("Node Pruning", "mib", "eprun_eval_s0.5_ld_zero",          "EdgePruning_zero_node"),
        ("DBM",          "mib", "eprun_eval_ld_sig_lr0.3_l16.0_zero", "EdgePruning_zero_node"),
    ]),
    ("\\textbf{\\ourmethod{} (ours)}", [
        (MV.label(),        "ours", "test_node_topk_uniform_lr05_zero", None),
        (MV.label(k="log"), "ours", "test_node_topk_log_lr05_zero",     None),
    ]),
]
REFERENCE = ("Random", "mib", "mib_zero_test/random", "Random_zero_node")
METRICS = {"area_under": "mib_zero_results.tex", "acc_auc": "mib_zero_accauc_results.tex"}


def load(kind, d, sub, task, model, key):
    if kind == "ours":
        p = RESULTS / d / f"{task}_{model}_test.pkl"
    else:
        p = RESULTS / d / sub / f"{task.replace('_', '-')}_{model}_test_abs-False.pkl"
    if not p.exists():
        return None
    try:
        v = pickle.load(open(p, "rb")).get(key)
        return None if v is None else round(float(v), 2)
    except Exception:
        return None


def row(spec, key):
    disp, kind, d, sub = spec
    return disp, {(t, m): v for t, m, _ in T.COLUMNS
                  if (v := load(kind, d, sub, t, m, key)) is not None}


def build(key):
    rows = [(title, [row(s, key) for s in members]) for title, members in BLOCKS]
    ref = row(REFERENCE, key)
    ncols = len(T.COLUMNS)
    all_rows = [r for _, rs in rows for r in rs if r[1]]
    best, second = {}, {}
    for t, m, _ in T.COLUMNS:
        vs = sorted({r[1][(t, m)] for r in all_rows if (t, m) in r[1]}, reverse=True)
        best[(t, m)] = vs[0] if vs else None
        second[(t, m)] = vs[1] if len(vs) > 1 else None

    def avg(data):
        return round(sum(data.values()) / ncols, 2) if len(data) == ncols else None
    avgs = sorted({a for a in (avg(r[1]) for r in all_rows) if a is not None}, reverse=True)
    avb, avs = (avgs[0] if avgs else None), (avgs[1] if len(avgs) > 1 else None)

    def render(disp, data, ref=False):
        cells = []
        for t, m, _ in T.COLUMNS:
            v = data.get((t, m))
            cells.append(T.fmt(v, bold=(not ref and v is not None and v == best[(t, m)]),
                               underline=(not ref and v is not None and v != best[(t, m)]
                                          and v == second[(t, m)])))
        a = avg(data)
        cells.append(T.fmt(a, bold=(not ref and a is not None and a == avb),
                           underline=(not ref and a is not None and a != avb and a == avs)))
        return f"\\quad {disp} & " + " & ".join(cells) + " \\\\"

    L = ["\\begin{adjustbox}{max width=\\textwidth}",
         "\\begin{tabular}{l" + "r" * ncols + "@{\\quad}r}", "\\toprule",
         "& \\multicolumn{4}{c}{IOI} & \\multicolumn{2}{c}{Arithmetic} & \\multicolumn{3}{c}{MCQA} & "
         "\\multicolumn{2}{c}{ARC (E)} & ARC (C) & \\\\",
         "\\cmidrule(lr){2-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-10} \\cmidrule(lr){11-12} \\cmidrule(lr){13-13}",
         "\\textbf{Method} & " + " & ".join(h for _, _, h in T.COLUMNS) + " & \\textbf{Avg} \\\\",
         "\\midrule"]
    label = {"area_under": "CPR", "acc_auc": "Compactness"}[key]
    L.append(f"\\multicolumn{{{ncols + 2}}}{{l}}{{\\textit{{Node-level, zero ablation, test set, {label}}}}} \\\\")
    for title, rs in rows:
        present = [r for r in rs if r[1]]
        for disp, data in rs:
            if not data:
                print(f"SKIP {disp}: no cells yet")
            elif len(data) < ncols:
                print(f"NOTE {disp}: {len(data)}/{ncols} cells")
        if not present:
            continue
        L.append(f"{title} \\\\")
        L += [render(d, v) for d, v in present]
    if ref[1]:
        L.append("\\textbf{Reference} \\\\")
        L.append(render(ref[0], ref[1], ref=True))
    L += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}"]
    return "\n".join(L) + "\n"


def main():
    TABS.mkdir(parents=True, exist_ok=True)
    for key, fname in METRICS.items():
        tex = build(key)
        (TABS / fname).write_text(tex)
        print(f"Wrote {TABS / fname}")


if __name__ == "__main__":
    main()
