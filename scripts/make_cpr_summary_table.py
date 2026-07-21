"""Summarised MIB table: mean CPR (95% CI) per method, averaged over the 11 tasks.
Two side-by-side subtables (node | edge), grouped Gradient attribution vs Ours.
Writes paper/tabs/cpr_summary.tex. Run on sc."""
import pickle
from pathlib import Path
import numpy as np

R = Path("results"); OUT = Path("paper/tabs/cpr_summary.tex")
COLUMNS = [("ioi", "gpt2"), ("ioi", "qwen2.5"), ("ioi", "gemma2"), ("ioi", "llama3"),
           ("arithmetic_subtraction", "llama3"), ("mcqa", "qwen2.5"), ("mcqa", "gemma2"),
           ("mcqa", "llama3"), ("arc_easy", "gemma2"), ("arc_easy", "llama3"),
           ("arc_challenge", "llama3")]

# (label, group, spec). spec: (dir,) = our flat val.pkl; (evaldir, sub) = grad eval abs-False
NODE = [
    ("NAP-IG", "g", ("napig_repro_eval", "EAP-IG-inputs_patching_node")),
    ("Conductance", "g", ("napig_local_eval", "EAP-IG-inputs-local_patching_node")),
    ("I$\\times$G", "g", ("ig1_eval", "EAP-IG-inputs_patching_node")),
    ("RelP", "g", ("relp_eval", "RelP_patching_node")),
    ("RelP+QK", "g", ("relp_qkgrad_eval", "RelP-qkgrad_patching_node")),
    ("AttnRLP", "g", ("attnrlp_eval", "AttnRLP_patching_node")),
    ("GIM", "g", ("gim_eval", "GIM_patching_node")),
    # MAttr headline = soft top-k fwd, log k (lr=0.01 dirs here); "+ hard" = sigmoid-STE fwd
    ("\\ourmethod{}", "o", ("mib_node_topk_log",)),
    ("$+$ unif $k$", "o", ("final_node",)),
    ("$+$ hard", "o", ("mib_node_hard_topk_log",)),
    ("$+$ unif $k$, $+$ hard", "o", ("mib_node_hard_topk",)),
    ("$+$ unif $k$, $+$ hard, $+$ Gumbel", "o", ("mib_node_hard_topk_gumbel",)),
    ("$-c_k$", "o", ("mib_node_detached_tau_log",)),
    ("$+$ unif $k$, $-c_k$", "o", ("mib_node_detached_tau",)),
    ("$+$ hard bwd", "o", ("mib_node_bernoulli_reinforce_log",)),
    ("$+$ unif $k$, $+$ hard bwd", "o", ("mib_node_bernoulli_reinforce",)),
    ("$+$ id-STE, SGD", "o", ("mib_node_identity_sgd_log",)),
    ("$+$ unif $k$, $+$ id-STE, SGD", "o", ("mib_node_identity_sgd",)),
]
EDGE = [
    ("EAP-IG-inp", "g", ("eapig_repro_eval", "EAP-IG-inputs_patching_edge")),
    ("\\ourmethod{}", "o", ("final_edge",)),
    ("$+$ hard", "o", ("mib_edge_hard_topk",)),
    ("$+$ unif $k$, $+$ hard", "o", ("mib_edge_hard_topk_uniform",)),
    ("$-c_k$", "o", ("mib_edge_detached_tau",)),
    ("$+$ hard bwd", "o", ("mib_edge_bernoulli_reinforce",)),
]


def auc(spec, task, model):
    p = (R / spec[0] / f"{task}_{model}_validation.pkl" if len(spec) == 1
         else R / spec[0] / spec[1] / f"{task.replace('_', '-')}_{model}_validation_abs-False.pkl")
    if not p.exists():
        return None
    try:
        v = pickle.load(open(p, "rb"))["area_under"]
        return float(v) if np.isfinite(v) else None
    except Exception:
        return None


def stats(spec):
    vals = [v for v in (auc(spec, t, m) for t, m in COLUMNS) if v is not None]
    if not vals:
        return None
    a = np.array(vals)
    ci = 1.96 * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
    return a.mean(), ci


def best_of(methods):
    return max((s[0] for _, _, spec in methods if (s := stats(spec))), default=None)


def cell_for(spec, best):
    s = stats(spec)
    if s is None:
        return "---"
    m, ci = s
    mtxt = f"\\textbf{{{m:.2f}}}" if m == best else f"{m:.2f}"
    return f"{mtxt} {{\\scriptsize$\\pm${ci:.2f}}}"


def plain_tabular(methods, best):
    """One column of methods (no group sub-headers)."""
    lines = ["\\begin{tabular}[t]{lc}", "\\toprule", "Method & CPR \\\\", "\\midrule"]
    for lab, _, spec in methods:
        lines.append(f"{lab} & {cell_for(spec, best)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def grouped_tabular(methods, best):
    """One column with Gradient/Ours sub-headers (for the edge subtable)."""
    lines = ["\\begin{tabular}[t]{lc}", "\\toprule", "Method & CPR \\\\", "\\midrule"]
    prev = None
    for lab, grp, spec in methods:
        if grp != prev:
            lines.append("\\textit{Gradient} \\\\" if grp == "g" else "\\textbf{Ours} \\\\")
            prev = grp
        lines.append(f"\\quad {lab} & {cell_for(spec, best)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def subgrouped_tabular(groups, best):
    """One column with explicit sub-group headers (for the gradient node subtable).
    groups: list of (subheader, [methods])."""
    lines = ["\\begin{tabular}[t]{lc}", "\\toprule", "Method & CPR \\\\", "\\midrule"]
    for sub, methods in groups:
        lines.append(f"\\textit{{{sub}}} \\\\")
        for lab, _, spec in methods:
            lines.append(f"\\quad {lab} & {cell_for(spec, best)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def main():
    grad = [m for m in NODE if m[1] == "g"]
    ours = [m for m in NODE if m[1] == "o"]
    # sort gradient node methods by CPR descending (None -> bottom)
    grad.sort(key=lambda m: (s[0] if (s := stats(m[2])) else -1), reverse=True)
    # split gradient methods into path-based (NAP-IG, Conductance) vs point-based (rest)
    PATH = {"NAP-IG", "Conductance"}
    grad_groups = [("Path", [m for m in grad if m[0] in PATH]),
                   ("Point", [m for m in grad if m[0] not in PATH])]
    node_best, edge_best = best_of(NODE), best_of(EDGE)
    tex = (
        "{\\small\n"
        "\\begin{minipage}[t]{0.66\\linewidth}\\centering\n"
        "{\\bfseries Node-level}\\par\\vspace{-1.8ex}\n"
        "\\begin{minipage}[t]{0.42\\linewidth}\\centering\n"
        + subgrouped_tabular(grad_groups, node_best)
        + "\n\\end{minipage}\\hfill\n"
        "\\begin{minipage}[t]{0.56\\linewidth}\\centering\n"
        + plain_tabular(ours, node_best)
        + "\n\\end{minipage}\n\\end{minipage}\\hfill\n"
        "\\begin{minipage}[t]{0.32\\linewidth}\\centering\n"
        "{\\bfseries Edge-level}\\par\\vspace{-1.8ex}\n"
        + plain_tabular(EDGE, edge_best)
        + "\n\\end{minipage}\n}\n"
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(tex)
    print(f"Wrote {OUT}\n")
    print(tex)


if __name__ == "__main__":
    main()
