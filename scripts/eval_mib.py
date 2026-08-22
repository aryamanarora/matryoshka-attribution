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

from learning_to_attribute import sigmoid_topk, learn_scores, normalize_mode, MODE_CHOICES
from learning_to_attribute import wandb_util
from learning_to_attribute.losses import attribution_loss
from learning_to_attribute.sigmoid_topk import sigmoid_topk_detached_tau
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
    parser.add_argument("--mib-path", type=str, default="MIB-circuit-track",
                        help="Path to cloned MIB-circuit-track repo")
    parser.add_argument("--model", type=str, default=None,
                        choices=list(MODEL_FULLNAMES.keys()))
    parser.add_argument("--task", type=str, default=None,
                        choices=list(TASKS_TO_HF.keys()))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--T", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"],
                        help="Mask-score optimizer (sgd accumulates raw g*delta; adam normalizes).")
    parser.add_argument("--n_iters", type=int, default=30)
    parser.add_argument("--split", type=str, default="validation",
                        help="Split to EVALUATE the circuit on (held out)")
    parser.add_argument("--train-split", type=str, default="train",
                        help="Split to TRAIN scores on (must differ from --split to avoid leakage)")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-schedule", default="log",
                        choices=["uniform", "log", "logit"],
                        help="How to sample k: uniform, log-uniform, or logit-uniform "
                             "(logit cancels the sigmoid_topk gate slope, making zero-init "
                             "SGD's expected score exactly activation-path IG -- schedules.py)")
    parser.add_argument("--mode", default="iso", choices=MODE_CHOICES,
                        help="iso (=sufficient, denoising): top-k stay clean, complement "
                             "corrupted; maximize retained clean behavior (this is "
                             "what MIB CPR measures, and what all our runs use). "
                             "cause (=necessary, noising): top-k get CF; find what breaks "
                             "behavior. (sufficient/necessary still accepted.)")
    parser.add_argument("--masking", default="topk",
                        choices=["topk", "topk_detached", "hard_topk", "hard_topk_identity", "hard_topk_identity_gumbel", "hard_topk_gumbel", "hard_topk_reinforce", "hard_concrete", "bernoulli_reinforce"],
                        help="topk: sigmoid top-k with random k (ours). "
                             "topk_detached: soft forward, detached tau (no coupling gradient). "
                             "hard_topk: random k + hard 0/1 mask with straight-through. "
                             "hard_topk_reinforce: fully binary forward+backward (REINFORCE). "
                             "bernoulli_reinforce: Bernoulli(sigmoid) fwd, REINFORCE bwd. "
                             "hard_concrete: Bernoulli(sigmoid) + L0 penalty (UGS-style).")
    parser.add_argument("--l0-lambda", type=float, default=1e-3,
                        help="L0 regularization weight for hard_concrete masking")
    parser.add_argument("--include-input", action="store_true",
                        help="Learn a score for the input embedding node")
    parser.add_argument("--eval-examples", type=int, default=None,
                        help="Max examples for MIB eval (default: all)")
    parser.add_argument("--train-batch-size", type=int, default=1,
                        help="Gradient accumulation batch size for training")
    parser.add_argument("--k-avg", type=int, default=1,
                        help="Average the gradient over this many independent k-draws per step "
                             "(reduces k-schedule variance; batch size only reduces example noise)")
    parser.add_argument("--natural-k-frac", type=float, default=0.0,
                        help="Fraction of steps using natural k (all scores >= 0)")
    parser.add_argument("--output", type=str, default="results/mib")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip the MIB eval; just train and save the train log (for convergence diagnostics)")
    wandb_util.add_args(parser)   # --no-wandb / --wandb-project / --wandb-entity; ON by default
    parser.add_argument("--wandb-name", default=None)

    # Config YAML
    temp_args, _ = parser.parse_known_args()
    if temp_args.config:
        config_path = Path(temp_args.config)
        if not config_path.exists() and not config_path.is_absolute():
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
    args.mode = normalize_mode(args.mode)   # iso/cause -> sufficient/necessary (both accepted)
    if args.model is None or args.task is None:
        parser.error("--model and --task are required (via CLI or config)")

    # W&B init. Project defaults to l2a-mib (one project per dataset, wandb_util.PROJECTS);
    # it used to default to "circuits", which pooled these with unrelated runs.
    wandb = wandb_util.init(
        "mib", args.wandb_name or f"{args.task}_{args.model}_{args.masking}_s{args.seed}",
        vars(args), project=args.wandb_project, entity=args.wandb_entity,
        enabled=args.wandb, group=f"{args.task}/{args.model}", job_type="node")

    # Add MIB to path
    mib_path = Path(args.mib_path).resolve()
    sys.path.insert(0, str(mib_path))
    sys.path.insert(0, str(mib_path / "EAP-IG" / "src"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # === Phase 1: Train node scores with HF model ===
    hf_model_name = MODEL_FULLNAMES[args.model]
    logger.info("Loading HF model %s for training...", hf_model_name)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    tokenizer.padding_side = "right"  # last_pos = attn_mask.sum()-1 assumes right padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # transformers renamed from_pretrained's `torch_dtype` to `dtype` in v5, and THIS FILE RUNS
    # UNDER BOTH. Per CLAUDE.md every gemma2 cell must be trained/scored in
    # MIB-circuit-track/.venv (transformers 4.46.3, TL 2.15.4, whose Gemma-2 forward is the
    # correct one), while gpt2/qwen2.5/llama3 run in .venv (transformers 5.9.0). Hardcoding
    # `dtype=` killed all 12 gemma2 jobs of the 2026-08-20 softlog_sgd sweep 28s in, with
    # `Gemma2ForCausalLM.__init__() got an unexpected keyword argument 'dtype'`. Gate on the
    # major version rather than trusting v5's deprecated `torch_dtype` alias to stay.
    import transformers as _tf
    _dtype_kw = "dtype" if int(_tf.__version__.split(".")[0]) >= 5 else "torch_dtype"
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        device_map="auto" if args.model in ("gemma2", "llama3") else None,
        **{_dtype_kw: torch.bfloat16
           if args.model in ("gemma2", "llama3", "qwen2.5") else torch.float32},
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
    dataset = HFEAPDataset(hf_task_name, tokenizer, split=args.train_split,
                           task=args.task, model_name=args.model)
    logger.info("Loaded %d TRAIN examples from %s (%s)", len(dataset), args.task, args.train_split)

    # Set up node-level hooks
    # For node mask, seq_len doesn't matter (position-agnostic), but we need a dummy value
    HooksCls = get_hooks_class(hf_model)
    # "necessary" (noising) corrupts the top-k; the hooker's `sufficient` flag patches
    # CF into the selected/top-k components, i.e. that IS the noising intervention.
    corrupt_topk = args.mode == "necessary"
    hooker = HooksCls(hf_model, "node", seq_len=1,
                       sufficient=corrupt_topk,
                       include_input=args.include_input)
    total = hooker.total
    logger.info("Node scores: %s", hooker.describe())

    hooker.register_hooks()

    # Training loop: the optimization (k-sampling, mask variants, REINFORCE/L0/bias-step,
    # optimizer) lives in learn_scores; this closure is the MIB *environment* — it samples a
    # same-length batch, applies the mask via the hooker, and returns the logit-diff loss.
    logger.info("Training for %d steps on %s/%s...", args.steps, args.task, args.model)
    n_examples = len(dataset)
    B = args.train_batch_size

    def loss_fn(mask):
        # Sample B same-length pairs (Python RNG; does not touch the torch RNG stream, so the
        # k/mask draws stay bit-identical to the pre-refactor loop).
        cleans, corrupteds, correct_ids, incorrect_ids = [], [], [], []
        attempts = 0
        while len(cleans) < B and attempts < B * 3:
            attempts += 1
            idx = random.randint(0, n_examples - 1)
            clean, corrupted, labels = dataset[idx]
            c_ids = tokenizer(clean, return_tensors="pt").input_ids
            s_ids = tokenizer(corrupted, return_tensors="pt").input_ids
            if c_ids.shape[1] != s_ids.shape[1]:
                continue
            cleans.append(clean)
            corrupteds.append(corrupted)
            correct_ids.append(labels[0])
            incorrect_ids.append(labels[1])
        if not cleans:
            return None                       # skip step (no same-length pairs sampled)
        actual_B = len(cleans)

        base_tok = tokenizer(cleans, return_tensors="pt", padding=True).to(device)
        src_tok = tokenizer(corrupteds, return_tensors="pt", padding=True).to(device)
        base_ids = base_tok.input_ids
        base_attn = base_tok.attention_mask
        last_pos = base_attn.sum(dim=1) - 1
        hooker.cache_cf_activations(src_tok.input_ids)

        hooker.mask = mask
        logits = hf_model(base_ids, attention_mask=base_attn).logits.float()
        last_logits = logits[torch.arange(actual_B, device=device), last_pos]
        correct_t = torch.tensor(correct_ids, device=device)
        incorrect_t = torch.tensor(incorrect_ids, device=device)
        # shared loss core: logit_diff = correct - incorrect; necessary/noising maximizes the
        # break (returns diff.mean()), sufficient/denoising minimizes -diff. Bit-identical to the
        # previous inline form.
        return attribution_loss("logit_diff", last_logits, correct_t, incorrect_t,
                                corrupt_topk=corrupt_topk)

    on_step = None
    if wandb:
        on_step = lambda step, k, lv, sc: wandb.log(
            {"loss": lv, "k": k, "k_frac": k / total}, step=step)

    result = learn_scores(
        total, loss_fn, steps=args.steps, variant=args.masking,
        k_schedule=args.k_schedule, k_avg=args.k_avg, T=args.T, n_iters=args.n_iters, lr=args.lr,
        optimizer=args.optimizer, l0_lambda=args.l0_lambda,
        natural_k_frac=args.natural_k_frac, use_bias=True, device=device,
        on_step=on_step, logger=logger, log_every=50,
    )
    scores = result.scores
    bias = result.bias
    loss_log = result.loss_log
    train_log = result.train_log
    train_time = result.train_time_s
    logger.info("Training complete in %.1fs", train_time)

    # Save per-step train log (step, k, k_frac, loss, bias_step) for convergence plots
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{args.task}_{args.model}_trainlog.csv"
    with open(log_path, "w") as f:
        f.write("step,k,k_frac,loss,bias_step\n")
        for row in train_log:
            f.write("%d,%.6g,%.6g,%.6g,%d\n" % row)
    logger.info("Saved train log to %s", log_path)

    if args.skip_eval:
        logger.info("Skipping MIB eval (--skip-eval)")
        hooker.remove_hooks()
        return

    if args.natural_k_frac > 0:
        n_pos = int((scores.detach() + bias.detach() >= 0).sum().item())
        logger.info("Learned bias=%.4f -> %d/%d nodes have (score+bias)>=0",
                    bias.item(), n_pos, total)
        # Rank-preserving calibration: shift scores so that >=0 means "in circuit".
        # Does NOT change CPR (global shift preserves the ranking / top-x%).
        with torch.no_grad():
            scores.data.add_(bias.data)
    hooker.remove_hooks()

    # Delete HF model to free memory. loss_fn closes over hf_model, so drop it too or the
    # model stays referenced and is not collected.
    del loss_fn
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

    # eval ranks by score; sufficient/denoising (our runs) => positive = important
    # to keep; necessary/noising => negative = important for breaking behavior.
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
    weighted_edge_counts, area_under, area_from_1, average, faithfulnesses, accuracies, acc_auc = \
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
        "accuracies": accuracies,
        "acc_auc": acc_auc,
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
        if name == "logits":
            nodes_dict[name] = {"in_graph": True}
        elif name == "input":
            nodes_dict[name] = {"in_graph": True}
            if input_score_val is not None:
                nodes_dict[name]["score"] = input_score_val
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
    node_names = list(graph.nodes.keys())
    for edge_name in graph.edges:
        src, rest = edge_name.split("->")
        dst = rest.split("<")[0]
        src_s = node_scores_tensor[node_names.index(src)].item() if src in graph.nodes and node_names.index(src) < len(node_scores_tensor) else 0
        dst_s = node_scores_tensor[node_names.index(dst)].item() if dst in graph.nodes and node_names.index(dst) < len(node_scores_tensor) else 0
        s = min(src_s, dst_s) if not (math.isnan(src_s) or math.isnan(dst_s)) else max(src_s, dst_s)
        if math.isnan(s):
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
