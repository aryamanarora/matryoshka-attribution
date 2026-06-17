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
    ("\\ourmethod{}", "o", ("mib_node_hard_topk",)),
    ("$+$ log $k$", "o", ("mib_node_hard_topk_log",)),
    ("$+$ Gumbel", "o", ("mib_node_hard_topk_gumbel",)),
    ("$+$ soft fwd", "o", ("final_node",)),
    ("$+$ soft, log $k$", "o", ("mib_node_topk_log",)),
    ("$+$ soft, $-c_k$", "o", ("mib_node_detached_tau",)),
    ("$+$ soft, $-c_k$, log $k$", "o", ("mib_node_detached_tau_log",)),
    ("$+$ hard bwd", "o", ("mib_node_bernoulli_reinforce",)),
    ("$+$ hard, log $k$", "o", ("mib_node_bernoulli_reinforce_log",)),
]
EDGE = [
    ("EAP-IG-inp", "g", ("eapig_repro_eval", "EAP-IG-inputs_patching_edge")),
    ("\\ourmethod{}", "o", ("mib_edge_hard_topk_uniform",)),
    ("$+$ log $k$", "o", ("mib_edge_hard_topk",)),
    ("$+$ soft fwd", "o", ("final_edge",)),
    ("$+$ soft, $-c_k$", "o", ("mib_edge_detached_tau",)),
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


def subtable(title, methods):
    rows = [(lab, grp, stats(spec)) for lab, grp, spec in methods]
    best = max((s[0] for _, _, s in rows if s), default=None)
    lines = ["\\begin{tabular}{lc}", "\\toprule",
             f"\\textbf{{{title}}} & Mean CPR \\\\", "\\midrule"]
    prev = None
    for lab, grp, s in rows:
        if grp != prev:
            lines.append("\\textit{Gradient attribution} \\\\" if grp == "g" else "\\textbf{Ours} \\\\")
            prev = grp
        if s is None:
            cell = "---"
        else:
            m, ci = s
            mtxt = f"\\textbf{{{m:.2f}}}" if m == best else f"{m:.2f}"
            cell = f"{mtxt} {{\\scriptsize$\\pm${ci:.2f}}}"
        lines.append(f"\\quad {lab} & {cell} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def main():
    tex = ("\\begin{minipage}[t]{0.5\\linewidth}\\centering\n"
           + subtable("Node-level", NODE)
           + "\n\\end{minipage}\\hfill\n"
           + "\\begin{minipage}[t]{0.46\\linewidth}\\centering\n"
           + subtable("Edge-level", EDGE)
           + "\n\\end{minipage}\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(tex)
    print(f"Wrote {OUT}\n")
    print(tex)


if __name__ == "__main__":
    main()
