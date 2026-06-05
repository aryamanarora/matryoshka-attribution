"""Spearman rank correlation for edge-level scores: Ours vs EAP-IG repro.

Breaks down by edge type: attn->attn, attn->MLP, MLP->attn, MLP->MLP.
Requires transformer_lens for graph construction (run on cluster).

Usage:
    uv run python scripts/compare_edge_ranks.py --mib-path /path/to/MIB-circuit-track
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

RESULTS_BASE = Path("results")

COLUMNS = [
    ("ioi", "gpt2"), ("ioi", "qwen2.5"), ("ioi", "gemma2"),
    ("mcqa", "qwen2.5"), ("mcqa", "gemma2"),
]

MODEL_FULLNAMES = {
    "gpt2": "gpt2-small",
    "qwen2.5": "Qwen/Qwen2.5-0.5B",
    "gemma2": "google/gemma-2-2b",
    "llama3": "meta-llama/Llama-3.1-8B",
}


def classify_edge(edge_name):
    """Classify edge by source->dest type."""
    src, rest = edge_name.split("->")
    # Strip qkv suffix from dest
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--mib-path", type=str, required=True)
    args = parser.parse_args()

    mib_path = Path(args.mib_path).resolve()
    sys.path.insert(0, str(mib_path))

    from transformer_lens import HookedTransformer
    from eap.graph import Graph

    results = []

    for task, model in COLUMNS:
        stask = task.replace("_", "-")
        ours_path = RESULTS_BASE / "mib_edge_hard_topk" / f"{task}_{model}_scores.pt"
        eapig_path = RESULTS_BASE / "eapig_repro" / "EAP-IG-inputs_patching_edge" / f"{stask}_{model}" / "importances.json"

        if not ours_path.exists() or not eapig_path.exists():
            continue

        print(f"Processing {task}/{model}...")

        # Load TL model and build graph
        tl_name = MODEL_FULLNAMES[model]
        if model in ("gemma2", "llama3", "qwen2.5"):
            tl_model = HookedTransformer.from_pretrained(tl_name, attn_implementation="eager",
                                                          torch_dtype=torch.bfloat16)
        else:
            tl_model = HookedTransformer.from_pretrained(tl_name)
        tl_model.cfg.use_split_qkv_input = True
        tl_model.cfg.use_attn_result = True
        tl_model.cfg.use_hook_mlp_in = True
        tl_model.cfg.ungroup_grouped_query_attention = True

        graph = Graph.from_model(tl_model)
        real_mask = graph.real_edge_mask.bool().flatten()

        # Load our scores
        ours_data = torch.load(ours_path, weights_only=False, map_location="cpu")
        ours_flat = ours_data["scores"].numpy()

        # Map our flat scores to edge names
        # graph.scores is [n_forward, n_backward], real_edge_mask selects valid entries
        full_scores = np.full(graph.scores.shape, float("-inf"))
        flat_full = full_scores.flatten()
        flat_full[real_mask.numpy()] = ours_flat
        full_scores = flat_full.reshape(graph.scores.shape)

        # Load EAP-IG scores
        eapig = json.load(open(eapig_path))
        eapig_edges = eapig["edges"]

        # Build paired score vectors by edge name
        paired = {}  # edge_name -> (ours_score, eapig_score)
        for edge_name, edge_info in eapig_edges.items():
            eapig_score = edge_info.get("score", 0)
            # Find this edge in the graph to get indices
            if edge_name in graph.edges:
                edge = graph.edges[edge_name]
                fi = edge.forward_index
                bi = edge.backward_index
                if fi < full_scores.shape[0] and bi < full_scores.shape[1]:
                    our_score = full_scores[fi, bi]
                    if our_score > float("-inf"):
                        paired[edge_name] = (our_score, eapig_score)

        if len(paired) < 10:
            print(f"  Only {len(paired)} matched edges, skipping")
            continue

        # Classify edges
        edge_types = {}
        for ename in paired:
            etype = classify_edge(ename)
            if etype not in edge_types:
                edge_types[etype] = []
            edge_types[etype].append(paired[ename])

        all_ours = [v[0] for v in paired.values()]
        all_eapig = [v[1] for v in paired.values()]
        rho_all = rho_or_nan(all_ours, all_eapig)

        row = {
            "task": task, "model": model, "n": len(paired),
            "rho_all": rho_all,
        }
        for etype in ["attn->attn", "attn->MLP", "MLP->attn", "MLP->MLP"]:
            pairs = edge_types.get(etype, [])
            if pairs:
                x, y = zip(*pairs)
                row[f"rho_{etype}"] = rho_or_nan(list(x), list(y))
                row[f"n_{etype}"] = len(pairs)
            else:
                row[f"rho_{etype}"] = float("nan")
                row[f"n_{etype}"] = 0

        results.append(row)
        print(f"  n={len(paired)} rho_all={rho_all:.3f}")
        for etype in ["attn->attn", "attn->MLP", "MLP->attn", "MLP->MLP"]:
            print(f"    {etype}: n={row[f'n_{etype}']} rho={row[f'rho_{etype}']:.3f}")

        del tl_model, graph
        torch.cuda.empty_cache()

    # Print summary
    print(f"\n{'='*100}")
    print(f"  EDGE: Ours (hard top-k) vs EAP-IG (repro)")
    print(f"{'='*100}")
    print(f"{'Task/Model':20s} {'n':>6s} {'all':>7s} {'a->a':>7s} {'a->M':>7s} {'M->a':>7s} {'M->M':>7s}")
    print("-" * 65)
    for r in results:
        label = f"{r['task']}/{r['model']}"
        vals = [r["rho_all"]] + [r[f"rho_{t}"] for t in ["attn->attn", "attn->MLP", "MLP->attn", "MLP->MLP"]]
        fmtd = [f"{v:7.3f}" if not np.isnan(v) else "    n/a" for v in vals]
        print(f"{label:20s} {r['n']:6d} " + " ".join(fmtd))
    if results:
        print("-" * 65)
        for col in ["rho_all"] + [f"rho_{t}" for t in ["attn->attn", "attn->MLP", "MLP->attn", "MLP->MLP"]]:
            vals = [r[col] for r in results if not np.isnan(r[col])]
        print(f"{'Mean':20s}        " + " ".join(
            f"{np.nanmean([r[c] for r in results]):7.3f}"
            for c in ["rho_all"] + [f"rho_{t}" for t in ["attn->attn", "attn->MLP", "MLP->attn", "MLP->MLP"]]
        ))

    # Generate LaTeX
    if results:
        generate_latex(results)


def generate_latex(results):
    TASK_LABELS = {"ioi": "IOI", "mcqa": "MCQA", "arc_easy": "ARC (E)"}
    MODEL_LABELS = {"gpt2": "GPT-2", "qwen2.5": "Qwen-2.5", "gemma2": "Gemma-2", "llama3": "Llama-3.1"}

    from collections import Counter
    task_order = []
    for r in results:
        if r["task"] not in task_order:
            task_order.append(r["task"])
    task_counts = Counter(r["task"] for r in results)

    def fmt(v):
        if np.isnan(v):
            return "---"
        return f"{v:.2f}"

    lines = []
    ncols = len(results)
    lines.append("\\begin{adjustbox}{max width=\\textwidth}")
    lines.append("\\begin{tabular}{l" + "r" * ncols + "}")
    lines.append("\\toprule")

    header_parts = []
    col = 2
    for task in task_order:
        n = task_counts[task]
        label = TASK_LABELS.get(task, task)
        if n > 1:
            header_parts.append(f"\\multicolumn{{{n}}}{{c}}{{{label}}}")
        else:
            header_parts.append(label)
    lines.append("& " + " & ".join(header_parts) + " \\\\")

    col = 2
    cmr = []
    for task in task_order:
        n = task_counts[task]
        cmr.append(f"\\cmidrule(lr){{{col}-{col + n - 1}}}")
        col += n
    lines.append(" ".join(cmr))

    model_headers = [MODEL_LABELS.get(r["model"], r["model"]) for r in results]
    lines.append("& " + " & ".join(model_headers) + " \\\\")
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
