"""Train sigmoid top-k edge scores and evaluate on MIB circuit track.

Uses live (grad-tracked) activations for stronger gradient signal.
"""

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
    parser.add_argument("--mib-path", type=str, default="MIB-circuit-track")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_FULLNAMES.keys()))
    parser.add_argument("--task", type=str, required=True, choices=list(TASKS_TO_HF.keys()))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--T", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"],
                        help="Optimizer for the scores (adam or sgd)")
    parser.add_argument("--n_iters", type=int, default=30)
    parser.add_argument("--split", type=str, default="validation",
                        help="Split to EVALUATE the circuit on (held out)")
    parser.add_argument("--train-split", type=str, default="train",
                        help="Split to TRAIN scores on (must differ from --split to avoid leakage)")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-schedule", default="log", choices=["uniform", "log"])
    parser.add_argument("--mode", default="iso",
                        choices=["iso", "cause", "sufficient", "necessary"],
                        help="iso (=sufficient, denoising): corrupt the complement of the top-k, "
                             "maximize retained clean behavior (MIB CPR; all our runs). "
                             "cause (=necessary, noising): corrupt the top-k, find what breaks "
                             "behavior. (sufficient/necessary still accepted.)")
    parser.add_argument("--masking", default="topk",
                        choices=["topk", "topk_detached", "hard_topk", "hard_topk_identity", "hard_topk_identity_gumbel", "hard_topk_gumbel", "hard_concrete", "bernoulli_reinforce"],
                        help="topk: sigmoid top-k (ours). topk_detached: soft forward, detached tau. "
                             "hard_topk: hard 0/1 + straight-through. "
                             "hard_concrete: Bernoulli(sigmoid) + L0. "
                             "bernoulli_reinforce: Bernoulli fwd + REINFORCE bwd.")
    parser.add_argument("--l0-lambda", type=float, default=1e-3)
    parser.add_argument("--eval-examples", type=int, default=None)
    parser.add_argument("--output", type=str, default="results/mib_edge")
    args = parser.parse_args()

    mib_path = Path(args.mib_path).resolve()
    sys.path.insert(0, str(mib_path))
    sys.path.insert(0, str(mib_path / "EAP-IG" / "src"))

    from transformer_lens import HookedTransformer
    from eap.graph import Graph
    from eap.utils import tokenize_plus
    from MIB_circuit_track.dataset import HFEAPDataset
    from MIB_circuit_track.metrics import get_metric
    from MIB_circuit_track.evaluation import evaluate_area_under_curve
    from einops import einsum

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from learning_to_attribute import sigmoid_topk, learn_scores, normalize_mode
    from learning_to_attribute.sigmoid_topk import sigmoid_topk_detached_tau
    args.mode = normalize_mode(args.mode)   # iso/cause -> sufficient/necessary (both accepted)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load model
    tl_name = MODEL_FULLNAMES[args.model]
    logger.info("Loading %s...", tl_name)
    if args.model in ("gemma2", "llama3", "qwen2.5"):
        model = HookedTransformer.from_pretrained(tl_name, attn_implementation="eager",
                                                   torch_dtype=torch.bfloat16)
    else:
        model = HookedTransformer.from_pretrained(tl_name)
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_model = model.cfg.d_model

    # Create graph for structure
    graph = Graph.from_model(model)
    n_real = int(graph.real_edge_mask.sum().item())
    logger.info("Graph: %d nodes, %d edges (%d real)", len(graph.nodes), len(graph.edges), n_real)

    # Precompute real edge index mapping for differentiable scatter
    real_flat = graph.real_edge_mask.flatten().bool()
    n_full = graph.n_forward * graph.n_backward
    # cumsum trick: maps each full-matrix position to an index in the flat real-edge vector
    cumsum = real_flat.float().cumsum(0).long() - 1
    cumsum = cumsum.clamp(min=0).to(device)
    real_flat_device = real_flat.float().to(device)

    # Load dataset
    hf_task = f"mib-bench/{TASKS_TO_HF[args.task]}"
    dataset = HFEAPDataset(hf_task, model.tokenizer, split=args.train_split,
                           task=args.task, model_name=args.model)
    logger.info("Loaded %d examples", len(dataset))

    # Score tensor
    total = n_real
    logger.info("Edge scores: %d parameters", total)

    # "necessary" (noising) corrupts the top-k edges; "sufficient" (denoising) corrupts
    # the complement (our runs / MIB CPR).
    corrupt_topk = args.mode == "necessary"

    # Precompute source node info for hooks (input, a{l}.h{h}, m{l})
    source_hooks = []  # (hook_name, node_name, forward_index, is_attn)
    input_node = graph.nodes['input']
    source_hooks.append(('hook_embed', 'input', graph.forward_index(input_node), False))
    for l in range(n_layers):
        head0 = graph.nodes[f'a{l}.h0']
        head0_idx = graph.forward_index(head0, attn_slice=False)
        source_hooks.append((f'blocks.{l}.attn.hook_result', f'attn_{l}',
                            head0_idx, True))
        mlp_node = graph.nodes[f'm{l}']
        mlp_idx = graph.forward_index(mlp_node)
        source_hooks.append((f'blocks.{l}.hook_mlp_out', f'm{l}', mlp_idx, False))

    n_examples = len(dataset)
    logger.info("Training for %d steps (mode=%s, k_schedule=%s, live activations)...",
                args.steps, args.mode, args.k_schedule)

    # The optimization (k-sampling, mask variants incl. REINFORCE/L0) is in learn_scores;
    # this closure is the EDGE environment: sample one pair, expand the flat edge mask to the
    # [n_forward, n_backward] adjacency, patch destinations with the live activation diffs,
    # and return the logit-diff loss. Returns None on a length-mismatch (skips the step).
    def loss_fn(mask_flat):
        idx = random.randint(0, n_examples - 1)
        clean, corrupted, labels = dataset[idx]
        correct_idx, incorrect_idx = labels[0], labels[1]
        clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, [clean])
        corrupted_tokens, _, _, _ = tokenize_plus(model, [corrupted])
        if clean_tokens.shape[1] != corrupted_tokens.shape[1]:
            return None

        # Step 1: cache corrupted source outputs (no grad)
        corrupted_acts = {}

        def make_corrupted_hook(name, fwd_idx, is_attn):
            def hook(act, hook):
                if is_attn:
                    for h in range(n_heads):
                        corrupted_acts[fwd_idx + h] = act[:, :, h].detach()
                else:
                    corrupted_acts[fwd_idx] = act.detach()
            return hook

        corrupted_fwd_hooks = [(hname, make_corrupted_hook(nname, fidx, is_a))
                               for hname, nname, fidx, is_a in source_hooks]
        with torch.no_grad():
            model.run_with_hooks(corrupted_tokens, fwd_hooks=corrupted_fwd_hooks,
                                 attention_mask=attention_mask)

        # Step 2: differentiable expansion of mask_flat to full [n_forward, n_backward]
        expanded = mask_flat[cumsum]  # [n_full]
        full_mask = (expanded * real_flat_device).view(graph.n_forward, graph.n_backward)
        corruption_mask = (full_mask if corrupt_topk else 1 - full_mask).to(model.cfg.dtype)

        # Step 3: clean forward with live source capture + destination patching
        clean_acts = {}

        def make_clean_hook(name, fwd_idx, is_attn):
            def hook(act, hook):
                if is_attn:
                    for h in range(n_heads):
                        clean_acts[fwd_idx + h] = act[:, :, h]  # LIVE, grad-tracked
                else:
                    clean_acts[fwd_idx] = act  # LIVE
                return act
            return hook

        def make_dest_hook(dest_node, letter=None):
            prev_idx = graph.prev_index(dest_node)
            bwd_idx = graph.backward_index(dest_node, qkv=letter, attn_slice=True)
            weights = corruption_mask[:prev_idx, bwd_idx]  # [prev, ...] or [prev, n_heads]

            def hook(activations, hook):
                diffs = []
                for src_i in range(prev_idx):
                    if src_i in corrupted_acts:
                        clean = clean_acts.get(src_i, torch.zeros_like(corrupted_acts[src_i]))
                        diffs.append(corrupted_acts[src_i] - clean)
                    else:
                        diffs.append(torch.zeros(1, activations.shape[1], d_model,
                                                 device=device, dtype=activations.dtype))
                diff_stack = torch.stack(diffs, dim=2)  # [batch, pos, prev, d_model]
                if weights.dim() == 1:
                    update = einsum(diff_stack, weights,
                                    'batch pos src hidden, src -> batch pos hidden')
                else:
                    update = einsum(diff_stack, weights,
                                    'batch pos src hidden, src heads -> batch pos heads hidden')
                return activations + update
            return hook

        clean_fwd_hooks = [(hname, make_clean_hook(nname, fidx, is_a))
                           for hname, nname, fidx, is_a in source_hooks]
        dest_hooks = []
        for l in range(n_layers):
            for i, letter in enumerate('qkv'):
                node = graph.nodes[f'a{l}.h0']
                dest_hooks.append((node.qkv_inputs[i], make_dest_hook(node, letter=letter)))
            node = graph.nodes[f'm{l}']
            dest_hooks.append((node.in_hook, make_dest_hook(node)))
        node = graph.nodes['logits']
        dest_hooks.append((node.in_hook, make_dest_hook(node)))

        logits = model.run_with_hooks(clean_tokens, fwd_hooks=clean_fwd_hooks + dest_hooks,
                                      attention_mask=attention_mask)
        logit_diff = logits[0, -1, correct_idx] - logits[0, -1, incorrect_idx]
        return logit_diff.float() if corrupt_topk else -logit_diff.float()

    result = learn_scores(
        total, loss_fn, steps=args.steps, variant=args.masking,
        k_schedule=args.k_schedule, T=args.T, n_iters=args.n_iters, lr=args.lr,
        optimizer=getattr(args, "optimizer", "adam"), l0_lambda=args.l0_lambda,
        device=device, logger=logger, log_every=50,
    )
    scores = result.scores
    loss_log = result.loss_log
    train_time = result.train_time_s
    logger.info("Training complete in %.1fs", train_time)

    # === MIB Evaluation ===
    logger.info("Running MIB evaluation (edge level)...")

    graph.scores[:] = float('-inf')
    real_edges = graph.real_edge_mask.bool()
    graph.scores[real_edges] = scores.data.cpu()

    eval_dataset = HFEAPDataset(hf_task, model.tokenizer, split=args.split,
                                task=args.task, model_name=args.model)
    if args.eval_examples:
        eval_dataset.head(args.eval_examples)
    dataloader = eval_dataset.to_dataloader(batch_size=args.batch_size)
    metric = get_metric("logit_diff", args.task, model.tokenizer, model)
    attribution_metric = partial(metric, mean=False, loss=False)

    weighted_edge_counts, area_under, area_from_1, average, faithfulnesses, accuracies, acc_auc = \
        evaluate_area_under_curve(model, graph, dataloader, attribution_metric,
                                  level="edge", absolute=False)

    logger.info("MIB Results (edge level, live activations):")
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
        "accuracies": accuracies,
        "acc_auc": acc_auc,
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
