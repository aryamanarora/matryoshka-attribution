"""Spearman rank correlation between our method and baselines across all tasks.

Reports overall, attn-head-only, and MLP-only correlations for node level.

Usage:
    uv run python scripts/compare_ranks.py
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

RESULTS_BASE = Path("results")

COLUMNS = [
    ("ioi", "gpt2"), ("ioi", "qwen2.5"), ("ioi", "gemma2"), ("ioi", "llama3"),
    ("arithmetic_subtraction", "llama3"),
    ("mcqa", "qwen2.5"), ("mcqa", "gemma2"), ("mcqa", "llama3"),
    ("arc_easy", "gemma2"), ("arc_easy", "llama3"), ("arc_challenge", "llama3"),
]


def load_node_scores(path):
    d = json.load(open(path))
    nodes = d["nodes"] if "nodes" in d else d
    return {n: info["score"] for n, info in nodes.items()
            if n not in ("input", "logits")}


def rho_or_nan(x, y):
    if len(x) < 4:
        return float("nan"), 1.0
    return spearmanr(x, y)


def compare_node_methods(ours_dir, baseline_dir, baseline_is_mib_repro=False):
    """Compare node-level scores across all available task/model pairs."""
    rows = []
    for task, model in COLUMNS:
        ours_path = RESULTS_BASE / ours_dir / f"{task}_{model}_importances.json"
        if baseline_is_mib_repro:
            stask = task.replace("_", "-")
            bl_path = RESULTS_BASE / baseline_dir / f"{stask}_{model}" / "importances.json"
        else:
            bl_path = RESULTS_BASE / baseline_dir / f"{task}_{model}_importances.json"

        if not ours_path.exists() or not bl_path.exists():
            continue

        ours = load_node_scores(ours_path)
        bl = load_node_scores(bl_path)
        common = sorted(set(ours) & set(bl))
        if len(common) < 4:
            continue

        attn = [n for n in common if not n.startswith("m")]
        mlps = [n for n in common if n.startswith("m")]

        rho_all, p_all = rho_or_nan([ours[n] for n in common], [bl[n] for n in common])
        rho_attn, _ = rho_or_nan([ours[n] for n in attn], [bl[n] for n in attn])
        rho_mlp, _ = rho_or_nan([ours[n] for n in mlps], [bl[n] for n in mlps])

        rows.append({
            "task": task, "model": model,
            "n": len(common), "n_attn": len(attn), "n_mlp": len(mlps),
            "rho_all": rho_all, "rho_attn": rho_attn, "rho_mlp": rho_mlp,
        })
    return rows


def print_table(title, rows):
    print(f"\n{'=' * 95}")
    print(f"  {title}")
    print(f"{'=' * 95}")
    print(f"{'Task/Model':30s}  {'n':>4s}  {'ρ(all)':>8s}  {'ρ(attn)':>8s}  {'n_attn':>6s}  {'ρ(MLP)':>8s}  {'n_mlp':>5s}")
    print("-" * 95)
    for r in rows:
        label = f"{r['task']}/{r['model']}"
        rho_all = f"{r['rho_all']:.3f}" if not np.isnan(r['rho_all']) else "   n/a"
        rho_attn = f"{r['rho_attn']:.3f}" if not np.isnan(r['rho_attn']) else "   n/a"
        rho_mlp = f"{r['rho_mlp']:.3f}" if not np.isnan(r['rho_mlp']) else "   n/a"
        print(f"{label:30s}  {r['n']:4d}  {rho_all:>8s}  {rho_attn:>8s}  {r['n_attn']:6d}  {rho_mlp:>8s}  {r['n_mlp']:5d}")

    # Averages
    if rows:
        avg_all = np.nanmean([r["rho_all"] for r in rows])
        avg_attn = np.nanmean([r["rho_attn"] for r in rows])
        avg_mlp = np.nanmean([r["rho_mlp"] for r in rows])
        print("-" * 95)
        print(f"{'Mean':30s}        {avg_all:8.3f}  {avg_attn:8.3f}          {avg_mlp:8.3f}")


def main():
    # Node level: Ours vs NAP-IG repro
    rows = compare_node_methods(
        "mib_node_hard_topk",
        "napig_repro/EAP-IG-inputs_patching_node",
        baseline_is_mib_repro=True,
    )
    print_table("NODE: Ours (hard top-k) vs NAP-IG (repro)", rows)

    # Node level: Ours vs soft fwd (our ablation)
    rows = compare_node_methods("mib_node_hard_topk", "final_node")
    print_table("NODE: Ours (hard top-k) vs + soft fwd", rows)

    # Node level: Ours vs detached tau
    rows = compare_node_methods("mib_node_hard_topk", "mib_node_detached_tau")
    print_table("NODE: Ours (hard top-k) vs + soft fwd, - c_k grad", rows)


if __name__ == "__main__":
    main()
