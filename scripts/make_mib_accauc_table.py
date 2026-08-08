"""MIB validation table with acc-AUC (log-weighted decision accuracy in [0,1]) instead of CPR.
Mirrors mib_results.tex (node level): gradient baselines + all MAttr variants (Adam/SGD, log-k
default + '+unif k' ablations). Swept methods (hard/soft log, uniform) at lr=0.05.

acc_auc sources: baselines -> the method's ORDINARY eval dir, falling back to a legacy *_accauc
re-eval only for methods scored before acc_auc existed; MAttr -> results/mattr_accauc_val/
(eval-only re-eval of each variant's circuit). 2 decimals; llama cells daggered (n=200 cap).

ADDING A NEW BASELINE: list its ordinary eval dir in BASELINES and stop. run_evaluation.py has
stored acc_auc in every pkl it writes since MIB_circuit_track/evaluation.py started returning
it, so acc-AUC comes free with the CPR run and NO separate acc-AUC pass needs to be launched.

Run from repo root:  uv run python scripts/make_mib_accauc_table.py  ->  paper/tabs/mib_accauc_results.tex
"""
import pickle
from pathlib import Path
import make_mib_table as M   # reuse COLUMNS + node OUR_METHODS + unifk/opt_of conventions

MIB = Path("/home/guests/aryaman/MIB-circuit-track/results")
L2A = Path("results")
MATTR_ACC = MIB / "mattr_accauc"       # lr01 ablations (acc_auc already computed)
MATTR_REEVAL = MIB / "mattr_accauc_val"  # htk_lr_0.05, final_node (re-eval)
OUTPUT = Path("paper/tabs/mib_accauc_results.tex")
COLUMNS = M.COLUMNS
LR05_CAPPED = {"htklog_lr_0.05", "topklog_lr_0.05", "htk_lr_0.05"}
LR05_EVALMIB = {"htklog_lr_0.05", "topklog_lr_0.05"}  # acc_auc in the eval_mib val pkl
REEVAL_DIRS = {"htk_lr_0.05", "final_node"}           # not in mattr_accauc -> re-eval

# (display, [results dirs, PRIMARY FIRST], method_saveable). The primary is the same ordinary
# eval dir make_mib_table.EXTRA_NODE_BASELINES reads for CPR, so the two tables describe the
# same run; the trailing `*_accauc` entries are legacy re-evaluations kept only because those
# methods were scored before evaluation.py returned acc_auc and their original pkls have
# acc_auc=None. Per cell, the first dir that yields a value wins.
#
# Mixing the two is safe and was checked rather than assumed: over every cell present in both,
# the max |CPR AUC difference| between a method's _eval and its _accauc rerun is 0.02
# (RelP 0.009, IxG 0.002, RelP+QK 0.020, RelP+Shapley 0.000, Conductance 0.008), i.e.
# run-to-run nondeterminism and not a different evaluation setting.
#
# AttnLRP and GIM have NO legacy dir on purpose -- they postdate the change, so their ordinary
# eval dir already carries acc_auc and adding a `*_accauc` rerun for them would be wasted GPU.
# (There IS a gim_nomlp_accauc on disk; it belongs to the pre-scale_mlp_gate GIM and must not
# be listed here, or a corrected-GIM row would silently fill from the buggy run.)
BASELINES = [
    ("NAP-IG", ["napig_repro_eval", "napig_ref_accauc"], "EAP-IG-inputs_patching_node"),
    ("Conductance", ["napig_local_eval", "napig_local_accauc"], "EAP-IG-inputs-local_patching_node"),
    ("I$\\times$G", ["ig1_eval", "ig1_accauc"], "EAP-IG-inputs_patching_node"),
    ("RelP", ["relp_eval", "relp_accauc"], "RelP_patching_node"),
    ("RelP+QK", ["relp_qkgrad_eval", "relp_qkgrad_accauc"], "RelP-qkgrad_patching_node"),
    ("RelP+Shapley", ["relpshapley_eval", "relpshapley_accauc"], "RelPShapley_patching_node"),
    ("AttnLRP", ["attnlrp_eval"], "AttnLRP_patching_node"),
    ("GIM", ["gim_eval"], "GIM_patching_node"),
]
NODE_METHODS = [(n, r, g) for n, r, l, g in M.OUR_METHODS if l == "node"]   # (name, dir, group)

# Mask-learning baselines (own header). UGS is edge-only so it cannot appear in this
# node-level table at all; Node Pruning runs at node level on every model. The budget, not the
# ranking, is what a mask learner actually optimizes, so it is a reported setting rather than a
# hidden default -- but the full M.EPRUN_SPARSITIES sweep is twelve near-identical rows, so as
# in the CPR table (M.EPRUN_SHOW) we show the best budget per objective and no more.
#
# *** The pick is NOT M.EPRUN_SHOW's, and that is the point, not an oversight. ***
# CPR and acc-AUC rank the budgets in essentially opposite orders (validation row means, 11
# cells; higher s = sparser = smaller circuit):
#
#   logit-diff   s=0.5   s=0.8   s=0.9   s=0.95  s=0.99
#     CPR AUC     1.67    1.46    1.28    1.36    1.24     <- densest wins
#     acc-AUC     0.23    0.31    0.34    0.36    0.38     <- sparsest wins, monotone the other way
#   KL           s=0.9 1.00 / 0.40   s=0.95 0.96 / 0.46   s=0.99 0.91 / 0.46
#
# That is CPR rewarding a bigger circuit, which is the same gap-padding sensitivity that
# motivated reporting acc-AUC in the first place. So each table names the budget that is best
# under the metric that table reports; carrying the CPR pick over here would show Node Pruning
# at its WORST acc-AUC budget (0.23 vs 0.38) and read as the baseline collapsing on acc-AUC
# when it is the budget selection, not the method.
#
# KL s=0.95 and s=0.99 are a tie at 2dp (0.4582 vs 0.4592); s=0.99 is the argmax but the
# margin is noise, so do not report a preference between them.
EPRUN_SHOW = {"eprun_eval_s0.99", "eprun_eval_s0.99_ld"}
MASK_BASELINES = [(M.eprun_label("node", suf), L2A / d, "EdgePruning_patching_node")
                  for suf, d in M.EPRUN_SPARSITIES if d in EPRUN_SHOW]
# DBM (pyvene sigmoid mask) is in the CPR table's mask block via M.SIGMOID_MASK_ROWS but was
# missing here, even though its eval pkls carry acc_auc like every other run -- so the acc-AUC
# table was silently comparing MAttr against a smaller set of mask learners than the CPR table.
# Its dirs are also written by run_evaluation.py under EdgePruning_patching_node (same code
# path, only the mask parameterization differs), hence the same sub-folder name.
MASK_BASELINES += [(disp, L2A / d, "EdgePruning_patching_node")
                   for disp, d in M.SIGMOID_MASK_ROWS]


def opt_of(d):   # id-STE variants use SGD; everything else Adam (mirrors make_mib_table)
    return "sgd" if "identity" in d else "adam"


def _acc(p):
    if not p.exists():
        return None
    try:
        v = pickle.load(open(p, "rb")).get("acc_auc")
        return round(v, 2) if v is not None else None
    except Exception:
        return None


# L2A/results and MIB-circuit-track/results are mirrors of the same run_evaluation.py outputs,
# but neither is complete: napig_repro_eval exists only in L2A, and a freshly finished job lands
# MIB-side until it is copied over. Searching both is what makes "just add the eval dir" true
# regardless of which side a run happens to be on.
ROOTS = (L2A, MIB)


def acc_base(dirs, sub, t, m):
    """acc_auc for one cell, from the first (root, dir) that has a value.

    Returns None only if NO listed dir has this cell, so a half-copied mirror or a legacy dir
    that never covered a cell degrades to a dash rather than to a wrong number.
    """
    fn = f"{t.replace('_', '-')}_{m}_validation_abs-False.pkl"
    for d in dirs:
        for root in ROOTS:
            v = _acc(root / d / sub / fn)
            if v is not None:
                return v
    return None


def acc_mattr(dir_, t, m):
    if dir_ in LR05_EVALMIB:                            # lr05 swept log methods: eval_mib pkl
        return _acc(L2A / dir_ / f"{t}_{m}_validation.pkl")
    base = MATTR_REEVAL if dir_ in REEVAL_DIRS else MATTR_ACC
    return _acc(base / f"{dir_}_patching_node" / f"{t.replace('_', '-')}_{m}_validation_abs-False.pkl")


def fmt(v, bold=False, dagger=False):
    if v is None:
        return "---"
    s = f"\\textbf{{{v:.2f}}}" if bold else f"{v:.2f}"
    return ("$^{\\dagger}$" + s) if dagger else s


def row_avg(data):
    vs = [v for v in (data.get((t, m)) for t, m, _ in COLUMNS) if v is not None]
    return round(sum(vs) / len(vs), 2) if vs else None


def main():
    rows = []   # (display, data, dagger_cells)
    for disp, d, sub in BASELINES:
        data = {(t, m): acc_base(d, sub, t, m) for t, m, _ in COLUMNS}
        # Same warning make_mib_table prints. A partial row shows no Avg here (see `full`), so
        # it cannot masquerade as a finished one in the table -- but silence at the terminal is
        # how a row stays half-empty for a week without anyone noticing.
        n = sum(v is not None for v in data.values())
        if 0 < n < len(COLUMNS):
            print(f"  NOTE {disp}: {n}/{len(COLUMNS)} cells ({'/'.join(d)}) -- still running")
        rows.append((disp, data, {(t, m) for t, m, _ in COLUMNS if m == "llama3"}))
    mask_rows = []   # (display, data)
    for disp, base, sub in MASK_BASELINES:
        data = {(t, m): _acc(base / sub / f"{t.replace('_', '-')}_{m}_validation_abs-False.pkl")
                for t, m, _ in COLUMNS}
        if any(v is not None for v in data.values()):
            mask_rows.append((disp, data))
    mattr = {}   # dir -> data
    for _, d, _ in NODE_METHODS:
        mattr[d] = {(t, m): acc_mattr(d, t, m) for t, m, _ in COLUMNS}

    all_data = [dd for _, dd, _ in rows] + [dd for _, dd in mask_rows] + list(mattr.values())
    best, second = {}, {}
    for t, m, _ in COLUMNS:
        vals = sorted({dd[(t, m)] for dd in all_data if dd.get((t, m)) is not None}, reverse=True)
        best[(t, m)] = vals[0] if vals else None
        second[(t, m)] = vals[1] if len(vals) > 1 else None
    # An average over a subset of columns is not comparable to one over all 11, so rows with
    # missing cells neither print an average nor compete for the bolded best average.
    def full(dd):
        return all(dd.get((t, m)) is not None for t, m, _ in COLUMNS)

    avs = sorted({a for a in (row_avg(dd) for dd in all_data if full(dd)) if a is not None},
                 reverse=True)
    abest, asec = (avs[0] if avs else None), (avs[1] if len(avs) > 1 else None)

    def emit(disp, data, dcells, indent=True):
        cells = []
        for t, m, _ in COLUMNS:
            v = data.get((t, m))
            cells.append(fmt(v, bold=(v is not None and v == best[(t, m)]),
                             dagger=((t, m) in dcells and v is not None)))
        a = row_avg(data) if full(data) else None
        cells.append(fmt(a, bold=(a is not None and a == abest)))
        pre = f"\\quad {disp}" if indent else disp
        return f"{pre} & " + " & ".join(cells) + " \\\\"

    def unifk(name):
        return "$+$ unif $k$" if name.startswith("\\ourmethod") else "$+$ unif $k$, " + name

    ncols = len(COLUMNS)
    L = ["\\begin{adjustbox}{max width=\\textwidth}",
         "\\begin{tabular}{l" + "r" * ncols + "@{\\quad}r}", "\\toprule",
         "& \\multicolumn{4}{c}{IOI} & Arithmetic & \\multicolumn{3}{c}{MCQA} & "
         "\\multicolumn{2}{c}{ARC (E)} & ARC (C) & \\\\",
         "\\cmidrule(lr){2-5} \\cmidrule(lr){6-6} \\cmidrule(lr){7-9} \\cmidrule(lr){10-11} \\cmidrule(lr){12-12}",
         "\\textbf{Method} & " + " & ".join(h for _, _, h in COLUMNS) + " & \\textbf{Avg} \\\\",
         "\\midrule", f"\\multicolumn{{{ncols + 2}}}{{l}}{{\\textit{{Node-level, acc-AUC}}}} \\\\",
         "\\textbf{Gradient attribution} \\\\"]
    for disp, data, dc in rows:
        L.append(emit(disp, data, dc))
    if mask_rows:
        L.append("\\textbf{Mask learning} \\\\")
        llama_cells = {(t, m) for t, m, _ in COLUMNS if m == "llama3"}
        for disp, data in mask_rows:
            L.append(emit(disp, data, llama_cells))   # llama3 eval is --head 200, as above

    llama_ioi = {("ioi", "llama3")}
    for opt, label in [("adam", "\\ourmethod{}-Adam"), ("sgd", "\\ourmethod{}-SGD")]:
        ours = [(n, d) for n, d, g in NODE_METHODS if g == "ours" and opt_of(d) == opt]
        unif = [(n, d) for n, d, g in NODE_METHODS if g == "uniform" and opt_of(d) == opt]
        if not ours and not unif:
            continue
        L.append(f"\\textbf{{{label}}} \\\\")
        for n, d in ours:
            L.append(emit(n, mattr[d], llama_ioi if d in LR05_CAPPED else set()))
        for n, d in unif:
            L.append(emit(unifk(n), mattr[d], llama_ioi if d in LR05_CAPPED else set()))
    L += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}"]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(L) + "\n")
    print(f"Wrote {OUTPUT} ({len(NODE_METHODS)} MAttr + {len(BASELINES)} baseline rows)")


if __name__ == "__main__":
    main()
