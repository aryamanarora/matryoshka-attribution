"""Spearman rank correlation for edge-level scores: Ours vs EAP-IG repro.

Breaks down by edge type: attn->attn, attn->MLP, MLP->attn, MLP->MLP.
Uses the JSON edge ordering directly (no model loading needed).

Usage:
    uv run python scripts/compare_edge_ranks.py
"""

import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

RESULTS_BASE = Path("results")

COLUMNS = [
    ("ioi", "gpt2"), ("ioi", "qwen2.5"), ("ioi", "gemma2"),
    ("mcqa", "qwen2.5"), ("mcqa", "gemma2"),
]


def classify_edge(edge_name):
    """Classify edge by source->dest type."""
    src, rest = edge_name.split("->")
    dest = rest.split("<")[0] if "<" in rest else rest
    src_is_mlp = src.startswith("m") or src == "input"
    dest_is_mlp = dest.startswith("m") or dest == "logits"
    if src_is_mlp and dest_is_mlp:
        return "MLP->MLP"
    elif src_is_mlp and not dest_is_mlp:
        return "MLP->attn"
    elif not src_is_mlp and dest_is_mlp:
        return "attn->MLP"
    else:
        return "attn->attn"


def rho_or_nan(x, y):
    if len(x) < 4:
        return float("nan")
    return spearmanr(x, y)[0]


def main():
    results = []

    for task, model in COLUMNS:
        stask = task.replace("_", "-")
        ours_path = RESULTS_BASE / "mib_edge_hard_topk" / f"{task}_{model}_scores.pt"
        eapig_json = RESULTS_BASE / "eapig_repro" / "EAP-IG-inputs_patching_edge" / f"{stask}_{model}" / "importances.json"

        if not ours_path.exists() or not eapig_json.exists():
            continue

        print(f"Processing {task}/{model}...")

        eapig = json.load(open(eapig_json))
        eapig_edges = eapig["edges"]
        ours_data = torch.load(ours_path, weights_only=False, map_location="cpu")
        ours_scores = ours_data["scores"].numpy()

        assert len(eapig_edges) == len(ours_scores), \
            f"Size mismatch: {len(eapig_edges)} vs {len(ours_scores)}"

        # Pair up: JSON dict order == flat tensor order
        edge_names = list(eapig_edges.keys())
        eapig_vals = np.array([eapig_edges[n]["score"] for n in edge_names])

        # Group by edge type
        by_type = {}
        for i, ename in enumerate(edge_names):
            etype = classify_edge(ename)
            if etype not in by_type:
                by_type[etype] = ([], [])
            by_type[etype][0].append(ours_scores[i])
            by_type[etype][1].append(eapig_vals[i])

        rho_all = rho_or_nan(ours_scores, eapig_vals)
        row = {"task": task, "model": model, "n": len(ours_scores), "rho_all": rho_all}

        for etype in ["attn->attn", "attn->MLP", "MLP->attn", "MLP->MLP"]:
            if etype in by_type:
                x, y = by_type[etype]
                row[f"rho_{etype}"] = rho_or_nan(x, y)
                row[f"n_{etype}"] = len(x)
            else:
                row[f"rho_{etype}"] = float("nan")
                row[f"n_{etype}"] = 0

        results.append(row)
        print(f"  n={len(ours_scores)} rho_all={rho_all:.3f}")
        for etype in ["attn->attn", "attn->MLP", "MLP->attn", "MLP->MLP"]:
            print(f"    {etype}: n={row[f'n_{etype}']} rho={row[f'rho_{etype}']:.3f}")

    # Print summary
    print(f"\n{'='*80}")
    print(f"  EDGE: Ours (hard top-k) vs EAP-IG (repro)")
    print(f"{'='*80}")
    print(f"{'Task/Model':20s} {'n':>6s} {'all':>7s} {'a->a':>7s} {'a->M':>7s} {'M->a':>7s} {'M->M':>7s}")
    print("-" * 65)
    for r in results:
        label = f"{r['task']}/{r['model']}"
        fmtd = []
        for col in ["rho_all"] + [f"rho_{t}" for t in ["attn->attn", "attn->MLP", "MLP->attn", "MLP->MLP"]]:
            v = r[col]
            fmtd.append(f"{v:7.3f}" if not np.isnan(v) else "    n/a")
        print(f"{label:20s} {r['n']:6d} " + " ".join(fmtd))
    if results:
        print("-" * 65)
        print(f"{'Mean':20s}        " + " ".join(
            f"{np.nanmean([r[c] for r in results]):7.3f}"
            for c in ["rho_all"] + [f"rho_{t}" for t in ["attn->attn", "attn->MLP", "MLP->attn", "MLP->MLP"]]
        ))

    # Generate LaTeX
    if results:
        generate_latex(results)


def generate_latex(results):
    TASK_LABELS = {"ioi": "IOI", "mcqa": "MCQA", "arc_easy": "ARC (E)"}
    MODEL_LABELS = {"gpt2": "GPT", "qwen2.5": "Qwen", "gemma2": "Gemma", "llama3": "Llama"}
    from collections import Counter

    task_order = []
    for r in results:
        if r["task"] not in task_order:
            task_order.append(r["task"])
    task_counts = Counter(r["task"] for r in results)

    def fmt(v):
        return "---" if np.isnan(v) else f"{v:.2f}"

    lines = []
    ncols = len(results)
    lines.append("\\begin{adjustbox}{max width=\\textwidth}")
    lines.append("\\begin{tabular}{l" + "r" * ncols + "}")
    lines.append("\\toprule")

    header_parts = []
    for task in task_order:
        n = task_counts[task]
        label = TASK_LABELS.get(task, task)
        header_parts.append(f"\\multicolumn{{{n}}}{{c}}{{{label}}}" if n > 1 else label)
    lines.append("& " + " & ".join(header_parts) + " \\\\")

    col = 2
    cmr = []
    for task in task_order:
        n = task_counts[task]
        cmr.append(f"\\cmidrule(lr){{{col}-{col + n - 1}}}")
        col += n
    lines.append(" ".join(cmr))

    lines.append("& " + " & ".join(MODEL_LABELS.get(r["model"], r["model"]) for r in results) + " \\\\")
    lines.append("\\midrule")

    lines.append("$\\rho$ (all) & " + " & ".join(fmt(r["rho_all"]) for r in results) + " \\\\")
    for etype, label in [("attn->attn", "$\\rho$ (attn$\\to$attn)"),
                          ("attn->MLP", "$\\rho$ (attn$\\to$MLP)"),
                          ("MLP->attn", "$\\rho$ (MLP$\\to$attn)"),
                          ("MLP->MLP", "$\\rho$ (MLP$\\to$MLP)")]:
        lines.append(f"{label} & " + " & ".join(fmt(r[f"rho_{etype}"]) for r in results) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{adjustbox}")

    table = "\n".join(lines) + "\n"
    out = Path("paper/tabs/edge_rank_correlations.tex")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table)
    print(f"\nWrote {out}")
    print(table)


if __name__ == "__main__":
    main()
