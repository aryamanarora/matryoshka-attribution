"""LaTeX (+plaintext) CPR-AUC table comparing the 3 node-level methods across the 12 cells.

Methods: standard NAP-IG (EAP-IG-inputs, 5 steps), our local-delta fix, and input x grad
(1-step IG). llama3 cells are evaluated on a reduced subset (--head 200, full-val OOMs on 8B),
so every llama3 cell is marked with a dagger, following the L2A table convention.
"""
import pickle, os

# (task, model, column header)
COLUMNS = [
    ("ioi","gpt2","IOI/gpt2"), ("ioi","qwen2.5","IOI/qwen"), ("mcqa","qwen2.5","MCQA/qwen"),
    ("ioi","gemma2","IOI/gemma"), ("mcqa","gemma2","MCQA/gemma"), ("arc_easy","gemma2","ARC-e/gemma"),
    ("ioi","llama3","IOI/llama"), ("mcqa","llama3","MCQA/llama"),
    ("arithmetic_addition","llama3","Add/llama"), ("arithmetic_subtraction","llama3","Sub/llama"),
    ("arc_easy","llama3","ARC-e/llama"), ("arc_challenge","llama3","ARC-c/llama"),
]
# row label -> (output_dir, method_name_saveable)
METHODS = [
    ("NAP-IG (5-step)",        "napig_ref_eval",   "EAP-IG-inputs_patching_node"),
    ("\\;+local-$\\Delta$ (ours)", "napig_local_eval", "EAP-IG-inputs-local_patching_node"),
    ("input$\\times$grad (1-step)", "ig1_eval",         "EAP-IG-inputs_patching_node"),
    ("RelP",                        "relp_eval",        "RelP_patching_node"),
    ("RelP (QK grad)",              "relp_qkgrad_eval", "RelP-qkgrad_patching_node"),
    ("RelP+Shapley",                     "relpshapley_eval",     "RelPShapley_patching_node"),
    ("AttnLRP",                     "attnlrp_eval",     "AttnLRP_patching_node"),
    ("GIM",                         "gim_eval",         "GIM_patching_node"),
]
BASE = "results"
# llama3 cells use subset eval -> dagger
DAGGER = {(t, m) for (t, m, _) in COLUMNS if m == "llama3"}


def auc(odir, mdir, task, model):
    p = f"{BASE}/{odir}/{mdir}/{task.replace('_','-')}_{model}_validation_abs-False.pkl"
    if not os.path.exists(p):
        return None
    try:
        return pickle.load(open(p, "rb"))["area_under"]
    except Exception:
        return None


def main():
    # gather: results[row_label][(task,model)] = auc
    results = {}
    for label, odir, mdir in METHODS:
        results[label] = {(t, m): auc(odir, mdir, t, m) for (t, m, _) in COLUMNS}

    # best (max) per column among available
    best = {}
    for (t, m, _) in COLUMNS:
        vals = [(lbl, results[lbl][(t, m)]) for lbl, _, _ in METHODS if results[lbl][(t, m)] is not None]
        best[(t, m)] = max((v for _, v in vals), default=None)

    # ---- plaintext ----
    print("CPR AUC (area_under, abs-False); higher=better. † = llama3 subset eval (--head 200)\n")
    hdr = f"{'method':26}" + "".join(f"{h:>13}" for _, _, h in COLUMNS) + f"{'mean':>9}"
    print(hdr); print("-" * len(hdr))
    for label, _, _ in METHODS:
        cells = []
        done = []
        for (t, m, _) in COLUMNS:
            v = results[label][(t, m)]
            if v is None:
                cells.append(f"{'--':>13}")
            else:
                done.append(v)
                mark = "*" if (best[(t, m)] is not None and abs(v - best[(t, m)]) < 1e-9) else " "
                dag = "†" if (t, m) in DAGGER else ""
                cells.append(f"{v:>11.3f}{mark}{dag}")
        mean = sum(done) / len(done) if done else float("nan")
        print(f"{label.replace(chr(92),'')[:26]:26}" + "".join(cells) + f"{mean:>9.3f}")

    # ---- LaTeX ----
    def fmt(v, t, m):
        if v is None:
            return "--"
        s = f"{v:.2f}"
        if best[(t, m)] is not None and abs(v - best[(t, m)]) < 1e-9:
            s = f"\\textbf{{{s}}}"
        if (t, m) in DAGGER:
            s = s + "$^\\dagger$"
        return s

    tex = []
    tex.append("% CPR AUC node-level comparison. $^\\dagger$ = llama3 evaluated on a 200-example subset.")
    tex.append("\\begin{tabular}{l" + "c" * len(COLUMNS) + "}")
    tex.append("\\toprule")
    tex.append("Method & " + " & ".join(h for _, _, h in COLUMNS) + " \\\\")
    tex.append("\\midrule")
    for label, _, _ in METHODS:
        row = " & ".join(fmt(results[label][(t, m)], t, m) for (t, m, _) in COLUMNS)
        tex.append(f"{label} & {row} \\\\")
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    print("\n" + "\n".join(tex))


if __name__ == "__main__":
    main()
