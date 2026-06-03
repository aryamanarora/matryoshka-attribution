"""Train sigmoid top-k node scores and evaluate on MIB circuit track."""

import argparse
import json
import logging
import os
import pickle
import random
import sys
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

import math

from learning_to_attribute import sigmoid_topk
from learning_to_attribute.models import (
    LlamaAttributionHooks, GPTNeoXAttributionHooks, GPT2AttributionHooks,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# MIB model name mapping (same as MIB's utils.py)
MODEL_FULLNAMES = {
    "gpt2": "gpt2",
    "qwen2.5": "Qwen/Qwen2.5-0.5B",
    "gemma2": "google/gemma-2-2b",
    "llama3": "meta-llama/Llama-3.1-8B",
}

# Map to transformer_lens names
MODEL_TL_NAMES = {
    "gpt2": "gpt2-small",
    "qwen2.5": "Qwen/Qwen2.5-0.5B",
    "gemma2": "google/gemma-2-2b",
    "llama3": "meta-llama/Llama-3.1-8B",
}

TASKS_TO_HF = {
    "ioi": "ioi",
    "mcqa": "copycolors_mcqa",
    "arithmetic_addition": "arithmetic_addition",
    "arithmetic_subtraction": "arithmetic_subtraction",
    "arc_easy": "arc_easy",
    "arc_challenge": "arc_challenge",
}

# Auto-detect hooks class from HF model
HOOKS_BY_TYPE = {
    "llama": LlamaAttributionHooks,
    "gpt2": GPT2AttributionHooks,
    "gpt_neox": GPTNeoXAttributionHooks,
    "qwen2": LlamaAttributionHooks,
    "gemma2": LlamaAttributionHooks,
}


def get_hooks_class(model):
    model_type = model.config.model_type
    if model_type in HOOKS_BY_TYPE:
        return HOOKS_BY_TYPE[model_type]
    raise ValueError(f"Unsupported model_type: {model_type}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--mib-path", type=str, required=True,
                        help="Path to cloned MIB-circuit-track repo")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_FULLNAMES.keys()))
    parser.add_argument("--task", type=str, required=True,
                        choices=list(TASKS_TO_HF.keys()))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--T", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--n_iters", type=int, default=30)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-schedule", default="uniform",
                        choices=["uniform", "log"],
                        help="How to sample k: uniform or log-uniform")
    parser.add_argument("--mode", default="necessary",
                        choices=["necessary", "sufficient"],
                        help="necessary: top-k stay clean (like EAP-IG). "
                             "sufficient: top-k get CF (find what flips).")
    parser.add_argument("--masking", default="topk",
                        choices=["topk", "hard_topk", "hard_concrete"],
                        help="topk: sigmoid top-k with random k (ours). "
                             "hard_topk: random k + hard 0/1 mask with straight-through. "
                             "hard_concrete: Bernoulli(sigmoid) + L0 penalty (UGS-style).")
    parser.add_argument("--l0-lambda", type=float, default=1e-3,
                        help="L0 regularization weight for hard_concrete masking")
    parser.add_argument("--include-input", action="store_true",
                        help="Learn a score for the input embedding node")
    parser.add_argument("--eval-examples", type=int, default=500,
                        help="Max examples for MIB eval (default 500, None=all)")
    parser.add_argument("--output", type=str, default="results/mib")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="circuits")
    parser.add_argument("--wandb-name", default=None)

    # Config YAML
    temp_args, _ = parser.parse_known_args()
    if temp_args.config:
        config_path = Path(temp_args.config)
        if not config_path.is_absolute():
            config_path = Path(__file__).parent / config_path
        with open(config_path) as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            dest = key.replace("-", "_")
            for action in parser._actions:
                if action.dest == dest:
                    action.default = value
                    break

    args = parser.parse_args()

    # W&B init
    if args.wandb:
        import wandb
        run_name = args.wandb_name or f"{args.task}_{args.model}_{args.masking}_s{args.seed}"
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
    else:
        wandb = None

    # Add MIB to path
    mib_path = Path(args.mib_path).resolve()
    sys.path.insert(0, str(mib_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # === Phase 1: Train node scores with HF model ===
    hf_model_name = MODEL_FULLNAMES[args.model]
    logger.info("Loading HF model %s for training...", hf_model_name)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        dtype=torch.bfloat16 if args.model in ("gemma2", "llama3", "qwen2.5") else torch.float32,
        device_map="auto" if args.model in ("gemma2", "llama3") else None,
    )
    if args.model not in ("gemma2", "llama3"):
        hf_model = hf_model.to(device)
    hf_model.eval()
    for p in hf_model.parameters():
        p.requires_grad_(False)
    hf_model.gradient_checkpointing_enable()
    logger.info("Model loaded in %.1fs", time.time() - t0)

    # Load MIB dataset for training examples
    from MIB_circuit_track.dataset import HFEAPDataset
    hf_task_name = f"mib-bench/{TASKS_TO_HF[args.task]}"
    dataset = HFEAPDataset(hf_task_name, tokenizer, split=args.split,
                           task=args.task, model_name=args.model)
    logger.info("Loaded %d examples from %s (%s)", len(dataset), args.task, args.split)

    # Set up node-level hooks
    # For node mask, seq_len doesn't matter (position-agnostic), but we need a dummy value
    HooksCls = get_hooks_class(hf_model)
    is_sufficient = args.mode == "sufficient"
    hooker = HooksCls(hf_model, "node", seq_len=1,
                       sufficient=is_sufficient,
                       include_input=args.include_input)
    total = hooker.total
    logger.info("Node scores: %s", hooker.describe())

    scores = nn.Parameter(torch.zeros(total, device=device))
    optimizer = torch.optim.Adam([scores], lr=args.lr)
    hooker.register_hooks()

    # Training loop
    loss_log = []
    logger.info("Training for %d steps on %s/%s...", args.steps, args.task, args.model)
    t0 = time.time()
    n_examples = len(dataset)

    for step in range(args.steps):
        # Sample a random example
        idx = random.randint(0, n_examples - 1)
        clean, corrupted, labels = dataset[idx]
        correct_idx, incorrect_idx = labels[0], labels[1]

        base_ids = tokenizer(clean, return_tensors="pt").input_ids.to(device)
        src_ids = tokenizer(corrupted, return_tensors="pt").input_ids.to(device)

        # Skip if different lengths (node mask needs same seq_len for CF interpolation)
        if base_ids.shape[1] != src_ids.shape[1]:
            continue

        # Cache CF activations
        hooker.cache_cf_activations(src_ids)

        # Forward with mask
        if args.k_schedule == "log":
            log_k = math.log(1) + (math.log(total) - math.log(1)) * torch.rand(1).item()
            k = math.exp(log_k)
        else:
            k = 1.0 + (total - 1.0) * torch.rand(1).item()

        if args.masking == "topk":
            hooker.mask = sigmoid_topk(scores, k=k, T=args.T, n_iters=args.n_iters)
        elif args.masking == "hard_topk":
            # Hard 0/1 top-k with straight-through gradient
            _, top_idx = scores.topk(int(k))
            hard = torch.zeros_like(scores)
            hard[top_idx] = 1.0
            # Straight-through: use sigmoid_topk gradient, hard values in forward
            soft = sigmoid_topk(scores, k=k, T=args.T, n_iters=args.n_iters)
            hooker.mask = hard - soft.detach() + soft
        else:
            # Hard concrete: Bernoulli(sigmoid(scores)) with straight-through
            probs = torch.sigmoid(scores)
            hard = torch.bernoulli(probs)
            hooker.mask = hard - probs.detach() + probs

        logits = hf_model(base_ids).logits[0, -1].float()

        if is_sufficient:
            loss = logits[correct_idx] - logits[incorrect_idx]
        else:
            loss = -(logits[correct_idx] - logits[incorrect_idx])

        # L0 penalty for hard_concrete
        if args.masking == "hard_concrete":
            l0 = torch.sigmoid(scores).sum()
            loss = loss + args.l0_lambda * l0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_log.append(loss_val)
        if wandb:
            wandb.log({"loss": loss_val, "k": k, "k_frac": k / total}, step=step)
        if (step + 1) % 50 == 0 or step == 0:
            rate = (step + 1) / (time.time() - t0)
            logger.info("Step %4d/%d  loss=%.4f  k=%.0f/%d  (%.1f step/s)",
                        step + 1, args.steps, loss_val, k, total, rate)

    train_time = time.time() - t0
    logger.info("Training complete in %.1fs", train_time)
    hooker.remove_hooks()

    # Delete HF model to free memory
    del hf_model
    torch.cuda.empty_cache()

    # === Phase 2: Convert to MIB graph and evaluate ===
    logger.info("Loading transformer_lens model for MIB evaluation...")
    from transformer_lens import HookedTransformer
    from eap.graph import Graph
    from MIB_circuit_track.metrics import get_metric
    from MIB_circuit_track.evaluation import evaluate_area_under_curve

    tl_name = MODEL_TL_NAMES[args.model]
    if args.model in ("gemma2", "llama3", "qwen2.5"):
        tl_model = HookedTransformer.from_pretrained(
            tl_name, attn_implementation="eager", torch_dtype=torch.bfloat16)
    else:
        tl_model = HookedTransformer.from_pretrained(tl_name)
    tl_model.cfg.use_split_qkv_input = True
    tl_model.cfg.use_attn_result = True
    tl_model.cfg.use_hook_mlp_in = True
    tl_model.cfg.ungroup_grouped_query_attention = True

    # Create graph and set node scores
    graph = Graph.from_model(tl_model, node_scores=True)
    logger.info("Graph: %d nodes, %d edges", len(graph.nodes), len(graph.edges))

    # Map our scores to MIB node scores
    n_layers = hooker.num_layers
    n_heads = hooker.num_heads
    off = 1 if args.include_input else 0
    input_score_val = scores.data[0].item() if args.include_input else None
    attn_scores = scores.data[off:off + n_layers * n_heads].view(n_layers, n_heads).cpu()
    mlp_scores = scores.data[off + n_layers * n_heads:].cpu()

    # Map our scores to MIB graph node scores
    # nodes_scores has shape (n_forward,) — use forward_index to map
    # Use NaN for unscored nodes (they stay in graph), but give input
    # a high score so it's always kept even with absolute=False
    max_score = max(attn_scores.abs().max().item(), mlp_scores.abs().max().item()) + 1.0
    node_scores_tensor = torch.full((graph.n_forward,), float("nan"))
    for name, node in graph.nodes.items():
        if name == "logits":
            continue
        idx = graph.forward_index(node, attn_slice=False)
        if idx >= graph.n_forward:
            continue
        if name == "input":
            node_scores_tensor[idx] = input_score_val if input_score_val is not None else max_score
        elif name.startswith("a"):
            parts = name.split(".")
            L = int(parts[0][1:])
            H = int(parts[1][1:])
            node_scores_tensor[idx] = attn_scores[L, H].item()
        elif name.startswith("m"):
            L = int(name[1:])
            node_scores_tensor[idx] = mlp_scores[L].item()

    # absolute=True in eval handles both necessary (positive = important)
    # and sufficient (negative = important for flipping) correctly via |score|
    graph.nodes_scores = node_scores_tensor
    logger.info("Set node scores (%d scored, %d forward nodes)",
                (~torch.isnan(graph.nodes_scores)).sum().item(), graph.n_forward)
    logger.info("Set node scores on graph")

    # Reload dataset for eval (with TL tokenizer)
    eval_dataset = HFEAPDataset(hf_task_name, tl_model.tokenizer, split=args.split,
                                task=args.task, model_name=args.model)
    if args.eval_examples:
        eval_dataset.head(args.eval_examples)
        logger.info("Capped eval set to %d examples", len(eval_dataset))
    dataloader = eval_dataset.to_dataloader(batch_size=args.batch_size)
    metric = get_metric("logit_diff", args.task, tl_model.tokenizer, tl_model)
    attribution_metric = partial(metric, mean=False, loss=False)

    # Run MIB evaluation
    logger.info("Running MIB evaluation (node level)...")
    weighted_edge_counts, area_under, area_from_1, average, faithfulnesses = \
        evaluate_area_under_curve(
            tl_model, graph, dataloader, attribution_metric,
            level="node", absolute=False)

    logger.info("MIB Results:")
    percentages = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
    for pct, faith in zip(percentages, faithfulnesses):
        logger.info("  %5.1f%% -> CPR=%.4f  CMD=%.4f", pct * 100, faith, abs(1 - faith))
    logger.info("  CPR AUC=%.4f  CMD AUC=%.4f  Avg CPR=%.4f",
                area_under, area_from_1, average)

    if wandb:
        log_dict = {"cpr_auc": area_under, "cmd_auc": area_from_1, "avg_cpr": average,
                    "train_time_s": train_time}
        for pct, faith in zip(percentages, faithfulnesses):
            log_dict[f"cpr_{pct}"] = faith
        wandb.log(log_dict)
        wandb.finish()

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save MIB-format results
    mib_results = {
        "weighted_edge_counts": weighted_edge_counts,
        "area_under": area_under,
        "area_from_1": area_from_1,
        "average": average,
        "faithfulnesses": faithfulnesses,
    }
    results_path = output_dir / f"{args.task}_{args.model}_{args.split}.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(mib_results, f)
    logger.info("Saved MIB results to %s", results_path)

    # Save our scores
    scores_path = output_dir / f"{args.task}_{args.model}_scores.pt"
    torch.save({
        "scores": scores.data.cpu(),
        "attn_scores": attn_scores,
        "mlp_scores": mlp_scores,
        "args": vars(args),
        "loss_log": loss_log,
        "train_time_s": train_time,
        "mib_results": mib_results,
    }, scores_path)
    logger.info("Saved scores to %s", scores_path)

    # Save importances.json for MIB submission format
    nodes_dict = {}
    for name in graph.nodes:
        if name == "input" or name == "logits":
            nodes_dict[name] = {"in_graph": True}
        elif name.startswith("a"):
            parts = name.split(".")
            L = int(parts[0][1:])
            H = int(parts[1][1:])
            s = attn_scores[L, H].item()
            nodes_dict[name] = {"in_graph": False, "score": s}
        elif name.startswith("m"):
            L = int(name[1:])
            s = mlp_scores[L].item()
            nodes_dict[name] = {"in_graph": False, "score": s}

    # For edges: propagate node scores (edge score = min of src/dst node scores)
    edges_dict = {}
    for edge_name in graph.edges:
        src, rest = edge_name.split("->")
        dst = rest.split("<")[0]
        src_score = node_score_list[list(graph.nodes.keys()).index(src)] if src in graph.nodes else 0
        dst_score = node_score_list[list(graph.nodes.keys()).index(dst)] if dst in graph.nodes else 0
        s = min(src_score, dst_score) if src_score != float("inf") and dst_score != float("inf") else max(src_score, dst_score)
        if s == float("inf"):
            s = 0.0
        edges_dict[edge_name] = {"score": s, "in_graph": False}

    cfg_dict = {
        "n_layers": tl_model.cfg.n_layers,
        "n_heads": tl_model.cfg.n_heads,
        "parallel_attn_mlp": False,
        "d_model": tl_model.cfg.d_model,
    }
    importances = {"cfg": cfg_dict, "nodes": nodes_dict, "edges": edges_dict}
    imp_path = output_dir / f"{args.task}_{args.model}_importances.json"
    with open(imp_path, "w") as f:
        json.dump(importances, f)
    logger.info("Saved importances to %s", imp_path)


if __name__ == "__main__":
    main()
