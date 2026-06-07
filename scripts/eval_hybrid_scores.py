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
    parser.add_argument("--mib-path", type=str, required=True)
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
    ours_path = RESULTS_BASE / "mib_node_hard_topk_log" / f"{task}_{model}_importances.json"
    if not ours_path.exists():
        ours_path = RESULTS_BASE / "mib_node_hard_topk" / f"{task}_{model}_importances.json"
    napig_path = RESULTS_BASE / "napig_repro" / "EAP-IG-inputs_patching_node" / f"{stask}_{model}" / "importances.json"

    if not ours_path.exists() or not napig_path.exists():
        logger.error("Missing importances: ours=%s napig=%s", ours_path.exists(), napig_path.exists())
        return

    ours = load_node_scores(ours_path)
    napig = load_node_scores(napig_path)
    logger.info("Loaded %d ours scores, %d NAP-IG scores", len(ours), len(napig))

    # Build hybrids
    hybrids = {
        "napig_ours_mlp": {},      # NAP-IG attn + our MLPs
        "ours_napig_mlp": {},      # Our attn + NAP-IG MLPs
        "napig_only": {},          # Pure NAP-IG (for reference)
        "ours_only": {},           # Pure ours (for reference)
    }

    common = sorted(set(ours) & set(napig))
    for node in common:
        is_mlp = node.startswith("m")
        hybrids["napig_ours_mlp"][node] = ours[node] if is_mlp else napig[node]
        hybrids["ours_napig_mlp"][node] = napig[node] if is_mlp else ours[node]
        hybrids["napig_only"][node] = napig[node]
        hybrids["ours_only"][node] = ours[node]

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
        node_scores[0] = float("inf")  # input
        node_scores[-1] = float("inf")  # logits

        for node_name, score in scores_dict.items():
            if node_name.startswith("a"):
                parts = node_name.split(".")
                L = int(parts[0][1:])
                H = int(parts[1][1:])
                node = graph.nodes.get(node_name)
                if node:
                    idx = graph.forward_index(node)
                    node_scores[idx] = score
            elif node_name.startswith("m"):
                L = int(node_name[1:])
                node = graph.nodes.get(node_name)
                if node:
                    idx = graph.forward_index(node)
                    node_scores[idx] = score

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
