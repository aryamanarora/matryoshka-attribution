"""Evaluate hybrid node scores: NAP-IG attn heads + our MLP scores.

Tests whether MLP disagreement drives our CPR advantage.

Usage:
    uv run python scripts/eval_hybrid_scores.py --mib-path /path/to/MIB --model gpt2 --task ioi
"""

import argparse
import json
import logging
import pickle
import sys
from functools import partial
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_BASE = Path("results")

MODEL_FULLNAMES = {
    "gpt2": "gpt2-small",
    "qwen2.5": "Qwen/Qwen2.5-0.5B",
    "gemma2": "google/gemma-2-2b",
    "llama3": "meta-llama/Llama-3.1-8B",
}


def load_node_scores(path):
    d = json.load(open(path))
    nodes = d.get("nodes", d)
    return {n: info["score"] for n, info in nodes.items() if n not in ("input", "logits")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mib-path", type=str, default="MIB-circuit-track")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_FULLNAMES.keys()))
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--output", type=str, default="results/hybrid_eval")
    args = parser.parse_args()

    mib_path = Path(args.mib_path).resolve()
    sys.path.insert(0, str(mib_path))
    sys.path.insert(0, str(mib_path / "EAP-IG" / "src"))

    from transformer_lens import HookedTransformer
    from eap.graph import Graph
    from MIB_circuit_track.dataset import HFEAPDataset
    from MIB_circuit_track.metrics import get_metric
    from MIB_circuit_track.evaluation import evaluate_area_under_curve

    task, model = args.task, args.model
    stask = task.replace("_", "-")

    # Load scores from both methods
    ours_path = RESULTS_BASE / "mib_node_hard_topk" / f"{task}_{model}_importances.json"
    if not ours_path.exists():
        ours_path = RESULTS_BASE / "mib_node_hard_topk" / f"{task}_{model}_importances.json"
    napig_path = RESULTS_BASE / "napig_repro" / "EAP-IG-inputs_patching_node" / f"{stask}_{model}" / "importances.json"

    if not ours_path.exists() or not napig_path.exists():
        logger.error("Missing importances: ours=%s napig=%s", ours_path.exists(), napig_path.exists())
        return

    ours = load_node_scores(ours_path)
    napig = load_node_scores(napig_path)
    logger.info("Loaded %d ours scores, %d NAP-IG scores", len(ours), len(napig))

    # Z-score normalize each method's scores so they're on the same scale
    import numpy as np
    common = sorted(set(ours) & set(napig))

    ours_vals = np.array([ours[n] for n in common])
    napig_vals = np.array([napig[n] for n in common])
    ours_z = {n: float((ours[n] - ours_vals.mean()) / ours_vals.std()) for n in common}
    napig_z = {n: float((napig[n] - napig_vals.mean()) / napig_vals.std()) for n in common}

    def make_hybrid(base_z, donor_z, swap_mlp=True):
        """Swap MLPs (swap_mlp=True) or attn heads (swap_mlp=False) from donor into base."""
        result = {}
        for n in common:
            is_mlp = n.startswith("m")
            if (swap_mlp and is_mlp) or (not swap_mlp and not is_mlp):
                result[n] = donor_z[n]
            else:
                result[n] = base_z[n]
        return result

    hybrids = {
        # MLP swaps (only these need re-evaluation)
        "napig_ours_mlp": make_hybrid(napig_z, ours_z, swap_mlp=True),    # NAP-IG base + our MLPs
        "ours_napig_mlp": make_hybrid(ours_z, napig_z, swap_mlp=True),    # Ours base + NAP-IG MLPs
    }

    for name, scores in hybrids.items():
        top5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info("%s top-5: %s", name, [(n, round(s, 2)) for n, s in top5])

    # Load TL model
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
    n_layers = tl_model.cfg.n_layers
    n_heads = tl_model.cfg.n_heads

    # Load eval dataset
    TASKS_TO_HF = {
        "ioi": "ioi", "mcqa": "copycolors_mcqa",
        "arithmetic_subtraction": "arithmetic_subtraction",
        "arc_easy": "arc_easy", "arc_challenge": "arc_challenge",
    }
    hf_task = f"mib-bench/{TASKS_TO_HF[task]}"
    eval_dataset = HFEAPDataset(hf_task, tl_model.tokenizer, split=args.split,
                                 task=task, model_name=model)
    dataloader = eval_dataset.to_dataloader(batch_size=args.batch_size)
    metric = get_metric("logit_diff", task, tl_model.tokenizer, tl_model)
    attribution_metric = partial(metric, mean=False, loss=False)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for hybrid_name, scores_dict in hybrids.items():
        logger.info("Evaluating %s...", hybrid_name)

        # Set node scores on graph
        node_scores = torch.full((graph.n_forward,), float("nan"))

        for name, node in graph.nodes.items():
            if name == "logits":
                continue
            idx = graph.forward_index(node, attn_slice=False)
            if idx >= graph.n_forward:
                continue
            if name == "input":
                node_scores[idx] = float("inf")
            elif name in scores_dict:
                node_scores[idx] = scores_dict[name]

        graph.nodes_scores = node_scores

        weighted_edge_counts, area_under, area_from_1, average, faithfulnesses = \
            evaluate_area_under_curve(tl_model, graph, dataloader, attribution_metric,
                                      level="node", absolute=False)

        logger.info("  %s: CPR AUC=%.4f, Avg=%.4f", hybrid_name, area_under, average)
        percentages = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
        for pct, faith in zip(percentages, faithfulnesses):
            logger.info("    %5.1f%% -> CPR=%.4f", pct * 100, faith)

        result = {
            "weighted_edge_counts": weighted_edge_counts,
            "area_under": area_under,
            "area_from_1": area_from_1,
            "average": average,
            "faithfulnesses": faithfulnesses,
            "hybrid": hybrid_name,
        }
        pkl_path = output_dir / f"{task}_{model}_{hybrid_name}_{args.split}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(result, f)

    logger.info("Done. Results in %s", output_dir)


if __name__ == "__main__":
    main()
