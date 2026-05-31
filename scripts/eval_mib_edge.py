"""Train sigmoid top-k edge scores and evaluate on MIB circuit track."""

import argparse
import json
import logging
import math
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_FULLNAMES = {
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


def sample_k(total, schedule="uniform"):
    if schedule == "log":
        return math.exp(random.uniform(0, math.log(total)))
    return 1.0 + (total - 1.0) * random.random()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mib-path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_FULLNAMES.keys()))
    parser.add_argument("--task", type=str, required=True, choices=list(TASKS_TO_HF.keys()))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--T", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--n_iters", type=int, default=30)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-schedule", default="log", choices=["uniform", "log"])
    parser.add_argument("--mode", default="necessary", choices=["necessary", "sufficient"])
    parser.add_argument("--eval-examples", type=int, default=500)
    parser.add_argument("--output", type=str, default="results/mib_edge")
    args = parser.parse_args()

    mib_path = Path(args.mib_path).resolve()
    sys.path.insert(0, str(mib_path))

    # Now import TL and MIB
    from transformer_lens import HookedTransformer
    from eap.graph import Graph
    from eap.utils import tokenize_plus, make_hooks_and_matrices
    from MIB_circuit_track.dataset import HFEAPDataset
    from MIB_circuit_track.metrics import get_metric
    from MIB_circuit_track.evaluation import evaluate_area_under_curve

    # Add sigmoid_topk to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from learning_to_attribute import sigmoid_topk

    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load model via transformer_lens
    tl_name = MODEL_FULLNAMES[args.model]
    logger.info("Loading %s via transformer_lens...", tl_name)
    if args.model in ("gemma2", "llama3", "qwen2.5"):
        model = HookedTransformer.from_pretrained(tl_name, attn_implementation="eager",
                                                   torch_dtype=torch.bfloat16)
    else:
        model = HookedTransformer.from_pretrained(tl_name)
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True

    # Create graph
    graph = Graph.from_model(model)
    n_edges = graph.real_edge_mask.sum().item()
    logger.info("Graph: %d nodes, %d edges (%d real)", len(graph.nodes), len(graph.edges), int(n_edges))

    # Load dataset
    hf_task = f"mib-bench/{TASKS_TO_HF[args.task]}"
    dataset = HFEAPDataset(hf_task, model.tokenizer, split=args.split,
                           task=args.task, model_name=args.model)
    logger.info("Loaded %d examples from %s (%s)", len(dataset), args.task, args.split)

    # Score tensor: one score per real edge
    # We'll work with the full (n_forward, n_backward) matrix but only optimize real edges
    total = int(n_edges)
    scores = nn.Parameter(torch.zeros(total, device=device))
    optimizer = torch.optim.Adam([scores], lr=args.lr)
    logger.info("Edge scores: %d parameters", total)

    is_sufficient = args.mode == "sufficient"

    # Training loop
    loss_log = []
    n_examples = len(dataset)
    logger.info("Training for %d steps (mode=%s, k_schedule=%s)...",
                args.steps, args.mode, args.k_schedule)
    t0 = time.time()

    for step in range(args.steps):
        # Sample example
        idx = random.randint(0, n_examples - 1)
        clean, corrupted, labels = dataset[idx]
        correct_idx, incorrect_idx = labels[0], labels[1]

        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, [clean])
        corrupted_tokens, _, _, _ = tokenize_plus(model, [corrupted])

        if clean_tokens.shape[1] != corrupted_tokens.shape[1]:
            continue

        # Get activation differences (corrupted - clean)
        (fwd_hooks_corrupted, fwd_hooks_clean, _), activation_difference = \
            make_hooks_and_matrices(model, graph, 1, n_pos, None)

        with torch.no_grad():
            with model.hooks(fwd_hooks_corrupted):
                _ = model(corrupted_tokens, attention_mask=attention_mask)
            with model.hooks(fwd_hooks_clean):
                _ = model(clean_tokens, attention_mask=attention_mask)

        # Build soft edge mask: sigmoid_topk over real edges only
        k = sample_k(total, args.k_schedule)
        soft_mask_flat = sigmoid_topk(scores, k=k, T=args.T, n_iters=args.n_iters)

        # Expand to full (n_forward, n_backward) matrix
        # Real edges get the soft mask; non-real edges stay 0
        real_mask = graph.real_edge_mask.to(device=device, dtype=torch.float32)
        full_mask = torch.zeros_like(real_mask)
        full_mask[real_mask.bool()] = soft_mask_flat

        # In necessary mode: mask=1 means "corrupt this edge" (patch to CF)
        # Top-k edges by score get mask≈1 → stay clean (NOT corrupted)
        # So we invert: corruption_mask = 1 - full_mask
        # In sufficient mode: top-k edges get mask≈1 → get corrupted
        if is_sufficient:
            corruption_mask = full_mask
        else:
            corruption_mask = 1 - full_mask

        corruption_mask = corruption_mask.to(model.cfg.dtype)

        # Build input construction hooks using our soft mask
        # This replaces evaluate_graph's binary in_graph with our soft corruption_mask
        from einops import einsum as eeinsum

        def make_soft_input_hook(act_diff, edge_weights):
            def hook(activations, hook):
                update = eeinsum(act_diff[:, :, :len(edge_weights)], edge_weights,
                                 'batch pos previous hidden, previous ... -> batch pos ... hidden')
                return activations + update
            return hook

        input_hooks = []
        for layer in range(model.cfg.n_layers):
            # Attention heads Q/K/V
            if any(graph.nodes[f'a{layer}.h{h}'].in_graph for h in range(model.cfg.n_heads)):
                for i, letter in enumerate('qkv'):
                    node = graph.nodes[f'a{layer}.h0']
                    prev_idx = graph.prev_index(node)
                    bwd_idx = graph.backward_index(node, qkv=letter, attn_slice=True)
                    weights = corruption_mask[:prev_idx, bwd_idx]
                    input_hooks.append((node.qkv_inputs[i],
                                       make_soft_input_hook(activation_difference, weights)))

            # MLP
            node = graph.nodes[f'm{layer}']
            prev_idx = graph.prev_index(node)
            bwd_idx = graph.backward_index(node)
            weights = corruption_mask[:prev_idx, bwd_idx]
            input_hooks.append((node.in_hook,
                               make_soft_input_hook(activation_difference, weights)))

        # Logits
        node = graph.nodes['logits']
        prev_idx = graph.prev_index(node)
        bwd_idx = graph.backward_index(node)
        weights = corruption_mask[:prev_idx, bwd_idx]
        input_hooks.append((node.in_hook,
                           make_soft_input_hook(activation_difference, weights)))

        # Forward with soft patching
        logits = model.run_with_hooks(clean_tokens, fwd_hooks=input_hooks,
                                      attention_mask=attention_mask)
        logit_diff = logits[0, -1, correct_idx] - logits[0, -1, incorrect_idx]

        if is_sufficient:
            loss = logit_diff  # minimize: want to flip
        else:
            loss = -logit_diff  # maximize: want to preserve

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_log.append(loss_val)
        if (step + 1) % 50 == 0 or step == 0:
            rate = (step + 1) / (time.time() - t0)
            logger.info("Step %4d/%d  loss=%.4f  k=%.0f/%d  (%.1f step/s)",
                        step + 1, args.steps, loss_val, k, total, rate)

    train_time = time.time() - t0
    logger.info("Training complete in %.1fs", train_time)

    # === MIB Evaluation ===
    logger.info("Running MIB evaluation (edge level)...")

    # Set edge scores on graph
    real_edges = graph.real_edge_mask.bool()
    graph.scores[:] = float('-inf')  # non-real edges ranked last
    graph.scores[real_edges] = scores.data.cpu()

    # For necessary: high score = keep clean = important. MIB keeps top-k.
    # For sufficient: high score = important for flipping. Need to handle.

    # Reload dataset for eval
    eval_dataset = HFEAPDataset(hf_task, model.tokenizer, split=args.split,
                                task=args.task, model_name=args.model)
    if args.eval_examples:
        eval_dataset.head(args.eval_examples)
    dataloader = eval_dataset.to_dataloader(batch_size=args.batch_size)
    metric = get_metric("logit_diff", args.task, model.tokenizer, model)
    attribution_metric = partial(metric, mean=False, loss=False)

    weighted_edge_counts, area_under, area_from_1, average, faithfulnesses = \
        evaluate_area_under_curve(model, graph, dataloader, attribution_metric,
                                  level="edge", absolute=False)

    logger.info("MIB Results (edge level):")
    percentages = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
    for pct, faith in zip(percentages, faithfulnesses):
        logger.info("  %5.1f%% -> CPR=%.4f", pct * 100, faith)
    logger.info("  CPR AUC=%.4f  Avg CPR=%.4f", area_under, average)

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    mib_results = {
        "weighted_edge_counts": weighted_edge_counts,
        "area_under": area_under,
        "area_from_1": area_from_1,
        "average": average,
        "faithfulnesses": faithfulnesses,
    }
    with open(output_dir / f"{args.task}_{args.model}_{args.split}.pkl", "wb") as f:
        pickle.dump(mib_results, f)

    torch.save({
        "scores": scores.data.cpu(),
        "args": vars(args),
        "loss_log": loss_log,
        "train_time_s": train_time,
        "mib_results": mib_results,
    }, output_dir / f"{args.task}_{args.model}_scores.pt")
    logger.info("Saved results to %s", output_dir)


if __name__ == "__main__":
    main()
