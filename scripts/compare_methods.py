"""Pairwise Wilcoxon signed-rank tests comparing CPR AUC across methods.

For each level (node, edge), compares all pairs of replicated methods
on their matched task/model columns. Applies Holm-Bonferroni correction.

Usage:
    uv run python scripts/compare_methods.py
"""

import pickle
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy.stats import wilcoxon

RESULTS_BASE = Path("results")

COLUMNS = [
    ("ioi", "gpt2"), ("ioi", "qwen2.5"), ("ioi", "gemma2"), ("ioi", "llama3"),
    ("arithmetic_subtraction", "llama3"),
    ("mcqa", "qwen2.5"), ("mcqa", "gemma2"), ("mcqa", "llama3"),
    ("arc_easy", "gemma2"), ("arc_easy", "llama3"), ("arc_challenge", "llama3"),
]

# Methods we have actual results for (not dagger baselines)
NODE_METHODS = {
    "Ours": "mib_node_hard_topk",
    "+ soft fwd": "final_node",
    "+ soft fwd, - c_k grad": "mib_node_detached_tau",
    "+ hard bwd": "mib_node_bernoulli_reinforce",
}

EDGE_METHODS = {
    "Ours": "mib_edge_hard_topk",
    "+ soft fwd": "final_edge",
    "+ soft fwd, - c_k grad": "mib_edge_detached_tau",
}

# Add repro baselines
NAPIG_REPRO_DIR = "napig_repro_eval"
EAPIG_REPRO_DIR = "eapig_repro_eval"


def load_cpr_auc(results_dir, task, model):
    pkl_path = RESULTS_BASE / results_dir / f"{task}_{model}_validation.pkl"
    if not pkl_path.exists():
        return None
    try:
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
        return d["area_under"]
    except Exception:
        return None


def load_repro(level, task, model):
    stask = task.replace("_", "-")
    if level == "node":
        pkl = RESULTS_BASE / NAPIG_REPRO_DIR / "EAP-IG-inputs_patching_node" / f"{stask}_{model}_validation_abs-False.pkl"
    else:
        pkl = RESULTS_BASE / EAPIG_REPRO_DIR / "EAP-IG-inputs_patching_edge" / f"{stask}_{model}_validation_abs-False.pkl"
    if not pkl.exists():
        return None
    try:
        with open(pkl, "rb") as f:
            d = pickle.load(f)
        return d["area_under"]
    except Exception:
        return None


def collect_results(methods, level):
    """Return {method_name: {(task, model): cpr_auc}}."""
    data = {}
    for name, results_dir in methods.items():
        vals = {}
        for task, model in COLUMNS:
            v = load_cpr_auc(results_dir, task, model)
            if v is not None:
                vals[(task, model)] = v
        data[name] = vals

    # Add repro baseline
    repro_name = "NAP-IG (repro)" if level == "node" else "EAP-IG (repro)"
    vals = {}
    for task, model in COLUMNS:
        v = load_repro(level, task, model)
        if v is not None:
            vals[(task, model)] = v
    data[repro_name] = vals

    return data


def pairwise_wilcoxon(data):
    """Run Wilcoxon signed-rank on all pairs, return results."""
    names = list(data.keys())
    results = []

    for a, b in combinations(names, 2):
        # Find matched columns
        matched = []
        for col in COLUMNS:
            va = data[a].get(col)
            vb = data[b].get(col)
            if va is not None and vb is not None:
                matched.append((va, vb))

        n = len(matched)
        if n < 4:
            results.append((a, b, n, None, None, None))
            continue

        x = np.array([m[0] for m in matched])
        y = np.array([m[1] for m in matched])
        diff = x - y
        mean_diff = diff.mean()
        wins_a = (diff > 0).sum()
        wins_b = (diff < 0).sum()

        # Wilcoxon signed-rank (two-sided)
        try:
            stat, p = wilcoxon(x, y, alternative="two-sided")
        except ValueError:
            # All differences are zero
            stat, p = 0, 1.0

        results.append((a, b, n, mean_diff, p, f"{wins_a}-{wins_b}"))

    return results


def holm_bonferroni(results):
    """Apply Holm-Bonferroni correction to p-values."""
    # Get indices of results with valid p-values
    valid = [(i, r[4]) for i, r in enumerate(results) if r[4] is not None]
    if not valid:
        return results

    # Sort by p-value
    valid.sort(key=lambda x: x[1])
    m = len(valid)

    corrected = {}
    for rank, (idx, p) in enumerate(valid):
        corrected[idx] = min(p * (m - rank), 1.0)

    # Build output with corrected p-values
    out = []
    for i, r in enumerate(results):
        if i in corrected:
            out.append((*r, corrected[i]))
        else:
            out.append((*r, None))
    return out


def print_results(level, results):
    print(f"\n{'='*70}")
    print(f"  {level.upper()}-LEVEL pairwise Wilcoxon signed-rank tests")
    print(f"{'='*70}")
    print(f"{'Method A':>30s}  vs  {'Method B':<30s}  n   mean_diff   wins   p-val   p-corr")
    print("-" * 120)

    for r in results:
        if len(r) == 7:
            a, b, n, mean_diff, p, wins, p_corr = r
        else:
            a, b, n, mean_diff, p, wins = r
            p_corr = None

        if p is None:
            print(f"{a:>30s}  vs  {b:<30s}  {n:2d}   {'(too few pairs)':>40s}")
        else:
            sig = ""
            if p_corr is not None and p_corr < 0.05:
                sig = " *"
            if p_corr is not None and p_corr < 0.01:
                sig = " **"
            p_str = f"{p:.4f}"
            pc_str = f"{p_corr:.4f}" if p_corr is not None else "  n/a"
            print(f"{a:>30s}  vs  {b:<30s}  {n:2d}   {mean_diff:+.4f}   {wins:>5s}   {p_str}   {pc_str}{sig}")

    print()


def main():
    for level, methods in [("node", NODE_METHODS), ("edge", EDGE_METHODS)]:
        data = collect_results(methods, level)

        # Print available data summary
        print(f"\n{level.upper()}-level results available:")
        for name, vals in data.items():
            cols = [f"{t[:3]}/{m[:3]}" for (t, m) in vals.keys()]
            print(f"  {name}: {len(vals)} columns")

        results = pairwise_wilcoxon(data)
        results = holm_bonferroni(results)
        print_results(level, results)


if __name__ == "__main__":
    main()
