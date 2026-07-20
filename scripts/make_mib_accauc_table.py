"""LaTeX MIB validation table with acc-AUC (log-weighted decision accuracy in [0,1]) instead of
CPR. Node level. Gradient baselines from the acc-AUC benchmark (MIB-circuit-track/results/*_accauc);
the 3 headline MAttr variants at lr=0.05. Bold = best per column; llama cells daggered (n=200 cap).

Run from repo root:  uv run python scripts/make_mib_accauc_table.py  ->  paper/tabs/mib_accauc_results.tex
"""
import pickle
from pathlib import Path

L2A = Path("results")
MIB = Path("/home/guests/aryaman/MIB-circuit-track/results")
OUTPUT = Path("paper/tabs/mib_accauc_results.tex")

COLUMNS = [
    ("ioi", "gpt2", "GPT"), ("ioi", "qwen2.5", "Qwen"), ("ioi", "gemma2", "Gemma"),
    ("ioi", "llama3", "Llama"), ("arithmetic_subtraction", "llama3", "Llama"),
    ("mcqa", "qwen2.5", "Qwen"), ("mcqa", "gemma2", "Gemma"), ("mcqa", "llama3", "Llama"),
    ("arc_easy", "gemma2", "Gemma"), ("arc_easy", "llama3", "Llama"), ("arc_challenge", "llama3", "Llama"),
]

# Gradient baselines: (display, benchmark dir, method_saveable subdir) under MIB/*_accauc
BASELINES = [
    ("NAP-IG",       "napig_ref_accauc",   "EAP-IG-inputs_patching_node"),
    ("Conductance",  "napig_local_accauc", "EAP-IG-inputs-local_patching_node"),
    ("I$\\times$G",  "ig1_accauc",         "EAP-IG-inputs_patching_node"),
    ("RelP",         "relp_accauc",        "RelP_patching_node"),
    ("RelP+QK",      "relp_qkgrad_accauc", "RelP-qkgrad_patching_node"),
    ("AttnRLP",      "attnrlp_accauc",     "AttnRLP_patching_node"),
    ("GIM",          "gim_accauc",         "GIM_patching_node"),
]
# MAttr lr=0.05: (display, kind, path-info)
MATTR = [
    ("\\ourmethod{} (log $k$)",       "l2a",  "htklog_lr_0.05"),
    ("\\ourmethod{} (soft fwd, log)", "l2a",  "topklog_lr_0.05"),
    ("\\ourmethod{} (unif $k$)",      "mib",  ("htkuni_lr05_accauc", "htkuni-lr05_patching_node")),
]


def acc_l2a(d, task, model):
    p = L2A / d / f"{task}_{model}_validation.pkl"
    return _acc(p)


def acc_mib(dir_, sub, task, model):
    p = MIB / dir_ / sub / f"{task.replace('_', '-')}_{model}_validation_abs-False.pkl"
    return _acc(p)


def _acc(p):
    if not p.exists():
        return None
    try:
        v = pickle.load(open(p, "rb")).get("acc_auc")
        return round(v, 3) if v is not None else None
    except Exception:
        return None


def fmt(v, bold=False, dagger=False):
    if v is None:
        return "---"
    s = f"\\textbf{{{v:.3f}}}" if bold else f"{v:.3f}"
    return ("$^{\\dagger}$" + s) if dagger else s


def row_avg(data):
    vs = [v for v in (data.get((t, m)) for t, m, _ in COLUMNS) if v is not None]
    return round(sum(vs) / len(vs), 3) if vs else None


def main():
    rows = []   # (display, {cell: acc}, dagger_cells)
    for disp, d, sub in BASELINES:
        data = {(t, m): acc_mib(d, sub, t, m) for t, m, _ in COLUMNS}
        rows.append((disp, data, {(t, m) for t, m, _ in COLUMNS if m == "llama3"}))
    for disp, kind, info in MATTR:
        if kind == "l2a":
            data = {(t, m): acc_l2a(info, t, m) for t, m, _ in COLUMNS}
        else:
            data = {(t, m): acc_mib(info[0], info[1], t, m) for t, m, _ in COLUMNS}
        rows.append((disp, data, {("ioi", "llama3")}))   # only llama/ioi capped for MAttr

    # best/second per column
    best, second = {}, {}
    for t, m, _ in COLUMNS:
        vals = sorted({data[(t, m)] for _, data, _ in rows if data.get((t, m)) is not None}, reverse=True)
        best[(t, m)] = vals[0] if vals else None
        second[(t, m)] = vals[1] if len(vals) > 1 else None
    avs = sorted({a for a in (row_avg(d) for _, d, _ in rows) if a is not None}, reverse=True)
    abest, asec = (avs[0] if avs else None), (avs[1] if len(avs) > 1 else None)

    ncols = len(COLUMNS)
    L = ["\\begin{adjustbox}{max width=\\textwidth}",
         "\\begin{tabular}{l" + "r" * ncols + "@{\\quad}r}", "\\toprule",
         "& \\multicolumn{4}{c}{IOI} & Arithmetic & \\multicolumn{3}{c}{MCQA} & "
         "\\multicolumn{2}{c}{ARC (E)} & ARC (C) & \\\\",
         "\\cmidrule(lr){2-5} \\cmidrule(lr){6-6} \\cmidrule(lr){7-9} \\cmidrule(lr){10-11} \\cmidrule(lr){12-12}",
         "\\textbf{Method} & " + " & ".join(h for _, _, h in COLUMNS) + " & \\textbf{Avg} \\\\",
         "\\midrule", f"\\multicolumn{{{ncols + 2}}}{{l}}{{\\textit{{Node-level, acc-AUC}}}} \\\\"]

    def emit(disp, data, dcells, indent=False):
        cells = []
        for t, m, _ in COLUMNS:
            v = data.get((t, m))
            cells.append(fmt(v, bold=(v is not None and v == best[(t, m)]), dagger=((t, m) in dcells and v is not None)))
        a = row_avg(data)
        cells.append(fmt(a, bold=(a is not None and a == abest)))
        pre = f"\\quad {disp}" if indent else disp
        return f"{pre} & " + " & ".join(cells) + " \\\\"

    L.append("\\textbf{Gradient attribution} \\\\")
    for disp, data, dc in rows[:len(BASELINES)]:
        L.append(emit(disp, data, dc, indent=True))
    L.append("\\textbf{\\ourmethod{} (lr $=$ 0.05)} \\\\")
    for disp, data, dc in rows[len(BASELINES):]:
        L.append(emit(disp, data, dc, indent=True))
    L += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}"]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(L) + "\n")
    print(f"Wrote {OUTPUT}\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
