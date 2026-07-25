"""MIB validation table with acc-AUC (log-weighted decision accuracy in [0,1]) instead of CPR.
Mirrors mib_results.tex (node level): gradient baselines + all MAttr variants (Adam/SGD, log-k
default + '+unif k' ablations). Swept methods (hard/soft log, uniform) at lr=0.05.

acc_auc sources: baselines -> MIB-circuit-track/results/*_accauc; MAttr -> results/mattr_accauc_val/
(eval-only re-eval of each variant's circuit). 2 decimals; llama cells daggered (n=200 cap).

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

BASELINES = [  # (display, *_accauc dir, method_saveable)
    ("NAP-IG", "napig_ref_accauc", "EAP-IG-inputs_patching_node"),
    ("Conductance", "napig_local_accauc", "EAP-IG-inputs-local_patching_node"),
    ("I$\\times$G", "ig1_accauc", "EAP-IG-inputs_patching_node"),
    ("RelP", "relp_accauc", "RelP_patching_node"),
    ("RelP+QK", "relp_qkgrad_accauc", "RelP-qkgrad_patching_node"),
    ("AttnRLP", "attnrlp_accauc", "AttnRLP_patching_node"),
    ("GIM", "gim_accauc", "GIM_patching_node"),
]
NODE_METHODS = [(n, r, g) for n, r, l, g in M.OUR_METHODS if l == "node"]   # (name, dir, group)

# Mask-learning baselines (own header). UGS is edge-only so it cannot appear in this
# node-level table at all; Edge Pruning runs at node level on every model. One row per
# target-sparsity budget (M.EPRUN_SPARSITIES) -- the budget, not the ranking, is what a
# mask learner actually optimizes, so it is a reported setting rather than a hidden default.
MASK_BASELINES = [(f"Edge Pruning ($s{{=}}{lab}$)", L2A / d, "EdgePruning_patching_node")
                  for lab, d in M.EPRUN_SPARSITIES]


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


def acc_base(dir_, sub, t, m):
    return _acc(MIB / dir_ / sub / f"{t.replace('_', '-')}_{m}_validation_abs-False.pkl")


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
        rows.append((disp, {(t, m): acc_base(d, sub, t, m) for t, m, _ in COLUMNS},
                     {(t, m) for t, m, _ in COLUMNS if m == "llama3"}))
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
