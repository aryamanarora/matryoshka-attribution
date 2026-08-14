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
from torch.utils.checkpoint import checkpoint

from learning_to_attribute import wandb_util

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
    wandb_util.add_args(parser)   # --no-wandb / --wandb-project / --wandb-entity; ON by default
    args = parser.parse_args()
    # Same project as the node-granularity MIB runs (same dataset); job_type separates them.
    wb = wandb_util.init("mib", f"edge_{args.task}_{args.model}_{args.masking}",
                         vars(args), project=args.wandb_project, entity=args.wandb_entity,
                         enabled=args.wandb, group=f"{args.task}/{args.model}", job_type="edge")

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

        # Memoize each source's `corrupted - clean` difference for the whole batch. torch.stack
        # COPIES, but autograd still retains the tensors it was handed, so before this every
        # destination rebuilt its own subtractions and each one stayed live alongside the stack
        # holding a copy of it -- ~sum(prev_index) ~= 34k intermediates on llama3, matching the
        # stacks byte for byte. Per source there are only ~1k, one per forward node.
        #
        # This change is bit-identical to not having it (verified on gpt2/ioi); it is purely a
        # memory measure, as is the checkpointing below. Together they reproduce the ORIGINAL
        # per-hook-stack scores exactly -- max |diff| 0.000e+00 on a 30-step gpt2/ioi run -- so no
        # stored edge result moves when it is re-run. (An intermediate version that shared retained
        # stacks across a prev_index did drift by 3.2e-04 on a 0.20 scale, because gradient into a
        # shared clean_acts entry summed in one place instead of arriving as three contributions.
        # Checkpointing removes that: each destination rebuilds its own stack in backward, which is
        # the original accumulation order. The drift was never a bug, but not having it is better.)
        #
        # Only cache the well-defined case. The zeros fallback below covers a source that is in
        # corrupted_acts but whose clean hook has not fired yet; per prev_index that is frozen
        # safely (verified), but a per-source entry outlives its prev_index, and the same source
        # CAN be populated by the time a later prev_index asks for it. Caching a zeros-fallback
        # would leak it forward, so those are rebuilt each time and never stored.
        diff_cache = {}
        zeros_cache = {}

        def diff_for(src_i, n_pos, dtype):
            cached = diff_cache.get(src_i)
            if cached is not None:
                return cached
            if src_i not in corrupted_acts:
                z = zeros_cache.get((n_pos, dtype))
                if z is None:
                    z = torch.zeros(1, n_pos, d_model, device=device, dtype=dtype)
                    zeros_cache[(n_pos, dtype)] = z
                return z          # constant, shared: torch.stack copies it anyway
            clean = clean_acts.get(src_i)
            if clean is None:
                return corrupted_acts[src_i] - torch.zeros_like(corrupted_acts[src_i])
            d = corrupted_acts[src_i] - clean
            diff_cache[src_i] = d
            return d

        # The stacks are RECOMPUTED in backward rather than retained. Sharing them per prev_index
        # and memoizing the per-source diffs were both real savings, but neither could fix this,
        # because the binding constraint is not a constant factor -- it is the length TAIL. Stack
        # memory is linear in sequence length (~286 MB per token position on llama3), and loss_fn
        # draws a random example per step: ARC medians are 52/61 tokens but the maxima are 178/186,
        # a 3.4x/3.0x tail. A median example needs ~17 GB of stacks, a tail example ~53 GB, which
        # is why step 1 passed and a later step died. mcqa (1.19x) and ioi (1.53x) have no tail
        # worth speaking of, which is exactly why those cells never showed this.
        #
        # Checkpointing makes peak memory independent of how many destinations there are: one stack
        # is live at a time instead of ~65, so the term drops from ~53 GB to ~1.6 GB at the worst
        # observed length. The einsum saves its inputs for backward, so no dict eviction can free
        # them -- discarding and recomputing is the only thing that does. Gradients are unchanged:
        # the recomputation is deterministic and sees identical inputs (the diffs are cached, and
        # clean_acts stay alive as graph inputs), which the gpt2 check confirms bit-for-bit.
        #
        # Checked at the worst case, not the average: a probe pinned to the LONGEST example in each
        # split (arc_challenge 186 tokens, arc_easy 178) trains at batch 2. An earlier probe drew
        # examples at random, passed step 1, and proved nothing -- against a 3x length tail a random
        # draw is a lottery over the exact thing being tested.
        def _stack_and_weight(weights, *diffs):
            stack = torch.stack(diffs, dim=2)  # [batch, pos, prev, d_model]
            if weights.dim() == 1:
                return einsum(stack, weights,
                              'batch pos src hidden, src -> batch pos hidden')
            return einsum(stack, weights,
                          'batch pos src hidden, src heads -> batch pos heads hidden')

        def make_dest_hook(dest_node, letter=None):
            prev_idx = graph.prev_index(dest_node)
            bwd_idx = graph.backward_index(dest_node, qkv=letter, attn_slice=True)
            weights = corruption_mask[:prev_idx, bwd_idx]  # [prev, ...] or [prev, n_heads]

            def hook(activations, hook):
                diffs = [diff_for(src_i, activations.shape[1], activations.dtype)
                         for src_i in range(prev_idx)]
                update = checkpoint(_stack_and_weight, weights, *diffs, use_reentrant=False)
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

    on_step = None if wb is None else (lambda step, k, lv, sc: wb.log(
        {"train/loss": lv, "train/k": k, "train/k_frac": k / total}, step=step))
    result = learn_scores(
        total, loss_fn, steps=args.steps, variant=args.masking,
        k_schedule=args.k_schedule, T=args.T, n_iters=args.n_iters, lr=args.lr,
        optimizer=getattr(args, "optimizer", "adam"), l0_lambda=args.l0_lambda,
        device=device, logger=logger, log_every=50, on_step=on_step,
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

    if wb is not None:
        # summary, not log: this is what the run table sorts on
        wb.summary.update({"mib/area_under": area_under, "mib/average": average,
                           "mib/acc_auc": acc_auc, "mib/area_from_1": area_from_1,
                           "total_edges": total, "output": str(output_dir)})
        wb.finish()


if __name__ == "__main__":
    main()
