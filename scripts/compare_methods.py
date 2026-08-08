"""Wilcoxon signed-rank tests comparing MAttr against every baseline / ablation
on validation-set CPR AUC, over the matched task/model columns.

We test the family "MAttr vs each other method" (not all C(k,2) pairs): that's the
question of interest and keeps Holm-Bonferroni powerful (correcting over ~14 pairs
instead of ~100). Two-sided Wilcoxon; mean_diff > 0 and wins favour MAttr.

Usage:
    uv run python scripts/compare_methods.py
"""

import pickle
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

RESULTS_BASE = Path("results")
REFERENCE = "MAttr"

COLUMNS = [
    ("ioi", "gpt2"), ("ioi", "qwen2.5"), ("ioi", "gemma2"), ("ioi", "llama3"),
    ("arithmetic_subtraction", "llama3"),
    ("mcqa", "qwen2.5"), ("mcqa", "gemma2"), ("mcqa", "llama3"),
    ("arc_easy", "gemma2"), ("arc_easy", "llama3"), ("arc_challenge", "llama3"),
]

# Each method: (name, kind, spec).
#   kind "flat":   spec = dir            -> results/<dir>/{task}_{model}_validation.pkl
#   kind "nested": spec = (evaldir, sub) -> results/<evaldir>/<sub>/{stask}_{model}_validation_abs-False.pkl
# MAttr default = uniform-k (mib_node_hard_topk); log-k is the "+ log k" ablation.
NODE_METHODS = [
    ("MAttr",                   "flat", "mib_node_hard_topk"),
    ("+ log k",                 "flat", "mib_node_hard_topk_log"),
    ("+ Gumbel",                "flat", "mib_node_hard_topk_gumbel"),
    ("+ soft fwd",              "flat", "final_node"),
    ("+ soft fwd, log k",       "flat", "mib_node_topk_log"),
    ("+ soft fwd, -c_k",        "flat", "mib_node_detached_tau"),
    ("+ soft fwd, -c_k, log k", "flat", "mib_node_detached_tau_log"),
    ("+ hard bwd",              "flat", "mib_node_bernoulli_reinforce"),
    ("+ hard bwd, log k",       "flat", "mib_node_bernoulli_reinforce_log"),
    # gradient-attribution baselines
    ("NAP-IG",      "nested", ("napig_ref_eval", "EAP-IG-inputs_patching_node")),
    ("Conductance", "nested", ("napig_local_eval", "EAP-IG-inputs-local_patching_node")),
    ("IxG",         "nested", ("ig1_eval", "EAP-IG-inputs_patching_node")),
    ("RelP",        "nested", ("relp_eval", "RelP_patching_node")),
    ("RelP+QK",     "nested", ("relp_qkgrad_eval", "RelP-qkgrad_patching_node")),
    ("RelP+Shapley",     "nested", ("relpshapley_eval", "RelPShapley_patching_node")),
    ("AttnLRP",     "nested", ("attnlrp_eval", "AttnLRP_patching_node")),
    ("GIM",         "nested", ("gim_eval", "GIM_patching_node")),
]

EDGE_METHODS = [
    ("MAttr",                   "flat", "mib_edge_hard_topk_uniform"),
    ("+ log k",                 "flat", "mib_edge_hard_topk"),
    ("+ soft fwd, log k",       "flat", "final_edge"),
    ("+ soft fwd, -c_k, log k", "flat", "mib_edge_detached_tau"),
    ("+ hard bwd, log k",       "flat", "mib_edge_bernoulli_reinforce"),
    ("EAP-IG-inp", "nested", ("eapig_clean_eval", "EAP-IG-inputs_patching_edge")),
]


def load(kind, spec, task, model):
    if kind == "flat":
        p = RESULTS_BASE / spec / f"{task}_{model}_validation.pkl"
    else:
        evaldir, sub = spec
        p = RESULTS_BASE / evaldir / sub / f"{task.replace('_', '-')}_{model}_validation_abs-False.pkl"
    if not p.exists():
        return None
    try:
        v = pickle.load(open(p, "rb"))["area_under"]
        return float(v) if np.isfinite(v) else None
    except Exception:
        return None


def collect_results(methods):
    """Return {method_name: {(task, model): cpr_auc}}."""
    data = {}
    for name, kind, spec in methods:
        vals = {}
        for task, model in COLUMNS:
            v = load(kind, spec, task, model)
            if v is not None:
                vals[(task, model)] = v
        data[name] = vals
    return data


def vs_reference(data, methods):
    """Wilcoxon: REFERENCE vs each other method, on matched columns."""
    results = []
    ref = data[REFERENCE]
    for name, _, _ in methods:
        if name == REFERENCE:
            continue
        other = data[name]
        matched = [(ref[c], other[c]) for c in COLUMNS if c in ref and c in other]
        n = len(matched)
        if n < 4:
            results.append((name, n, None, None, None))
            continue
        x = np.array([m[0] for m in matched]); y = np.array([m[1] for m in matched])
        diff = x - y
        try:
            _, p = wilcoxon(x, y, alternative="two-sided")
        except ValueError:
            p = 1.0
        results.append((name, n, diff.mean(), p, f"{(diff > 0).sum()}-{(diff < 0).sum()}"))
    return results


def holm_bonferroni(results):
    """Holm-Bonferroni over the family of comparisons with a valid p-value."""
    valid = sorted([(i, r[3]) for i, r in enumerate(results) if r[3] is not None],
                   key=lambda x: x[1])
    m = len(valid)
    corrected, running = {}, 0.0
    for rank, (idx, p) in enumerate(valid):
        running = max(running, min(p * (m - rank), 1.0))  # enforce monotonicity
        corrected[idx] = running
    return [(*r, corrected.get(i)) for i, r in enumerate(results)]


def print_results(level, results):
    print(f"\n{'='*78}")
    print(f"  {level.upper()}-LEVEL: {REFERENCE} vs each method (Wilcoxon signed-rank, Holm-corrected)")
    print(f"{'='*78}")
    print(f"  {REFERENCE} vs {'Method':<24s} n   mean_diff   wins(M-O)   p-val    p-corr")
    print("-" * 78)
    for r in results:
        name, n, mean_diff, p, wins, p_corr = r
        if p is None:
            print(f"  {REFERENCE} vs {name:<24s} {n:2d}   (too few matched columns)")
            continue
        sig = " **" if (p_corr is not None and p_corr < 0.01) else (
              " *" if (p_corr is not None and p_corr < 0.05) else "")
        print(f"  {REFERENCE} vs {name:<24s} {n:2d}   {mean_diff:+.4f}    {wins:>7s}   "
              f"{p:.4f}   {p_corr:.4f}{sig}")
    print()


def main():
    for level, methods in [("node", NODE_METHODS), ("edge", EDGE_METHODS)]:
        data = collect_results(methods)
        print(f"\n{level.upper()}-level columns available:")
        for name, _, _ in methods:
            print(f"  {name:<26s} {len(data[name])} cols")
        results = holm_bonferroni(vs_reference(data, methods))
        print_results(level, results)


if __name__ == "__main__":
    main()
