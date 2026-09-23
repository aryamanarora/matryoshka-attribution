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

from matryoshka_attribution import (sigmoid_topk, learn_scores, normalize_mode, MODE_CHOICES,
                                   CFActivationCache)
from matryoshka_attribution import wandb_util
from matryoshka_attribution.losses import attribution_loss, resolve_direction
from matryoshka_attribution.sigmoid_topk import sigmoid_topk_detached_tau
from matryoshka_attribution.models import LlamaAttributionHooks, GPT2AttributionHooks
from matryoshka_attribution.deps import find_mib_path

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
    parser.add_argument("--mib-path", type=str, default=None,
                        help="Path to cloned MIB-circuit-track repo")
    parser.add_argument("--model", type=str, default=None,
                        choices=list(MODEL_FULLNAMES.keys()))
    parser.add_argument("--task", type=str, default=None,
                        choices=list(TASKS_TO_HF.keys()))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--T", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--adam-eps", type=float, default=1e-8,
                        help="Adam epsilon (ignored for --optimizer sgd). The default is the "
                             "sign(g) regime at large substrates; 1e-2 restores magnitude "
                             "weighting (see scripts/sva/launch/submit_adam_eps_followup.sh). Encode a "
                             "non-default value in --output -- it is not in any filename.")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd", "none"],
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
    parser.add_argument("--mode", default="iso", choices=MODE_CHOICES + ["joint"],
                        help="iso (=sufficient, denoising): top-k stay clean, complement "
                             "corrupted; maximize retained clean behavior (this is "
                             "what MIB CPR measures, and what all our runs use). "
                             "cause (=necessary, noising): top-k get CF; find what breaks "
                             "behavior. joint: a coin flip between the two every step "
                             "(losses.resolve_direction, as in eval_sva.py). "
                             "(sufficient/necessary still accepted.)")
    parser.add_argument("--masking", default="topk",
                        choices=["topk", "topk_detached", "topk_identity", "hard_topk", "hard_topk_identity", "hard_topk_identity_gumbel", "hard_concrete", "bernoulli_reinforce"],
                        help="topk: sigmoid top-k with random k (ours). "
                             "topk_detached: soft forward, detached tau (no coupling gradient). "
                             "hard_topk: random k + hard 0/1 mask with straight-through. "
                             "bernoulli_reinforce: Bernoulli(sigmoid) fwd, REINFORCE bwd. "
                             "hard_concrete: Bernoulli(sigmoid) + L0 penalty (UGS-style).")
    parser.add_argument("--l0-lambda", type=float, default=1e-3,
                        help="L0 regularization weight for hard_concrete masking")
    # ZERO ABLATION (2026-09-18): non-circuit nodes are set to 0 instead of their counterfactual
    # activation, at TRAINING (LlamaAttributionHooks zero_ablation, the SVA+ zero runs' hook) and
    # at EVAL (MIB's evaluate_area_under_curve intervention="zero", which drops the corrupted
    # forward and subtracts the clean activations without adding anything back). The CF cache
    # still runs the source forward; its activations are simply never blended in.
    parser.add_argument("--ablation", default="patching", choices=["patching", "zero"])
    parser.add_argument("--include-input", action="store_true",
                        help="Learn a score for the input embedding node")
    parser.add_argument("--eval-examples", type=int, default=None,
                        help="Max examples for MIB eval (default: all)")
    parser.add_argument("--train-batch-size", type=int, default=1,
                        help="Gradient accumulation batch size for training")
    parser.add_argument("--cf-cache-gb", type=float, default=4.0,
                        help="Device-memory budget for the per-example CF-activation cache "
                             "(skips re-running the CF forward for resampled examples); "
                             "0 disables it")
    parser.add_argument("--k-avg", type=int, default=1,
                        help="Average the gradient over this many independent k-draws per step "
                             "(reduces k-schedule variance; batch size only reduces example noise)")
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
            else:
                # A key that matches no flag would otherwise be SILENTLY ignored -- and a
                # stale config (e.g. the retired natural-k-frac ones) would quietly run a
                # different experiment into its output dir. Fail loudly instead.
                parser.error(f"config {config_path}: unknown key {key!r}")

    args = parser.parse_args()
    # iso/cause -> sufficient/necessary (both accepted); "joint" stays as is and is resolved to
    # a direction per training step (losses.resolve_direction) below.
    args.mode = args.mode if args.mode == "joint" else normalize_mode(args.mode)
    if args.model is None or args.task is None:
        parser.error("--model and --task are required (via CLI or config)")

    # W&B init. Project defaults to l2a-mib (one project per dataset, wandb_util.PROJECTS);
    # it used to default to "circuits", which pooled these with unrelated runs.
    wandb = wandb_util.init(
        "mib", args.wandb_name or f"{args.task}_{args.model}_{args.masking}_s{args.seed}",
        vars(args), project=args.wandb_project, entity=args.wandb_entity,
        enabled=args.wandb, group=f"{args.task}/{args.model}", job_type="node")

    # Add MIB to path
    mib_path = find_mib_path(args.mib_path)
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
    # the tl2 env (transformers 4.46.3, TL 2.15.4, whose Gemma-2 forward is the
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
    # NOTE an unconditional hf_model.gradient_checkpointing_enable() sat here until
    # 2026-08-26. It was a silent NO-OP: transformers gates checkpointing on
    # `self.training`, and this model is in eval(). Every stored result therefore already
    # trained without checkpointing (which also proves the activation memory fits). If it
    # is ever really needed, enabling requires model.train() plus forcing dropout off.
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
                       include_input=args.include_input,
                       zero_ablation=args.ablation == "zero")
    total = hooker.total
    logger.info("Node scores: %s", hooker.describe())

    hooker.register_hooks()

    # Training loop: the optimization (k-sampling, mask variants, REINFORCE/L0/bias-step,
    # optimizer) lives in learn_scores; this closure is the MIB *environment* — it samples a
    # same-length batch, applies the mask via the hooker, and returns the logit-diff loss.
    logger.info("Training for %d steps on %s/%s...", args.steps, args.task, args.model)
    n_examples = len(dataset)
    B = args.train_batch_size

    # Pre-tokenize the train set ONCE. The old loss_fn made 3 tokenizer calls per accepted
    # example per step (2 for the length check, 1 for the padded batch) over a dataset that
    # never changes -- on small models that was most of the step's wall clock. The sampling
    # retry loop below keeps the exact random.randint sequence (same rejection predicate on
    # precomputed lengths), so batch composition is unchanged for a given seed.
    t0 = time.time()
    pretok = []   # (clean_ids [L], src_ids [L'], correct_id, incorrect_id, corrupted_str)
    for i in range(n_examples):
        clean, corrupted, labels = dataset[i]
        pretok.append((tokenizer(clean, return_tensors="pt").input_ids[0],
                       tokenizer(corrupted, return_tensors="pt").input_ids[0],
                       labels[0], labels[1], corrupted))
    logger.info("Pre-tokenized %d examples in %.1fs", n_examples, time.time() - t0)
    pad_id = tokenizer.pad_token_id

    def collate(seqs):
        """Right-pad 1-D id tensors into [B, P] ids + attention mask (tokenizer parity)."""
        lens = torch.tensor([s.shape[0] for s in seqs])
        P = int(lens.max())
        ids = torch.full((len(seqs), P), pad_id, dtype=torch.long)
        for b, s in enumerate(seqs):
            ids[b, : s.shape[0]] = s
        attn = (torch.arange(P)[None] < lens[:, None]).long()
        return ids.to(device), attn.to(device), lens

    # CF activations are deterministic per example; cache them instead of re-running the CF
    # forward for every resample (the cache runs the forward only on a batch's uncached
    # examples). Keyed by the corrupted prompt string. --cf-cache-gb 0 disables.
    cf_cache = CFActivationCache(hooker, max_gb=args.cf_cache_gb, logger=logger)

    def loss_fn(mask):
        # Sample B same-length pairs (Python RNG; does not touch the torch RNG stream, so the
        # k/mask draws stay bit-identical to the pre-refactor loop).
        picks = []
        attempts = 0
        while len(picks) < B and attempts < B * 3:
            attempts += 1
            idx = random.randint(0, n_examples - 1)
            if pretok[idx][0].shape[0] != pretok[idx][1].shape[0]:
                continue
            picks.append(pretok[idx])
        if not picks:
            return None                       # skip step (no same-length pairs sampled)
        actual_B = len(picks)

        base_ids, base_attn, lens = collate([p[0] for p in picks])
        src_ids, _, src_lens = collate([p[1] for p in picks])
        last_pos = (lens - 1).to(device)
        cf_cache.prepare([p[4] for p in picks], src_ids, src_lens.tolist())

        hooker.mask = mask
        # Per-step intervention direction: the fixed one for iso/cause, a coin flip for joint
        # (losses.resolve_direction, the same call eval_sva.py makes). The hooker reads its
        # `sufficient` flag at hook time, so flipping it here flips which side gets the CF.
        # For iso/cause this is a no-op (the flag was set at construction to the same value),
        # so every existing run is bit-identical; joint additionally consumes one torch.rand
        # per step, which is why it is a NEW mode and not a default.
        step_cause = resolve_direction(args.mode, corrupt_topk)
        hooker.sufficient = step_cause
        # NOTE logits stay in model dtype until after the last-token gather: .float() on the
        # full [B, P, vocab] tensor materialized ~50-160 MB fp32 in the autograd graph for a
        # gradient that is zero everywhere but last_pos. Cast-after-slice is grad-identical.
        logits = hf_model(base_ids, attention_mask=base_attn).logits
        last_logits = logits[torch.arange(actual_B, device=device), last_pos].float()
        correct_t = torch.tensor([p[2] for p in picks], device=device)
        incorrect_t = torch.tensor([p[3] for p in picks], device=device)
        # shared loss core: logit_diff = correct - incorrect; necessary/noising maximizes the
        # break (returns diff.mean()), sufficient/denoising minimizes -diff. Bit-identical to the
        # previous inline form.
        return attribution_loss("logit_diff", last_logits, correct_t, incorrect_t,
                                corrupt_topk=step_cause)

    on_step = None
    if wandb:
        on_step = lambda step, k, lv, sc: wandb.log(
            {"loss": lv, "k": k, "k_frac": k / total}, step=step)

    result = learn_scores(
        total, loss_fn, steps=args.steps, variant=args.masking,
        k_schedule=args.k_schedule, k_avg=args.k_avg, T=args.T, n_iters=args.n_iters, lr=args.lr,
        optimizer=args.optimizer, l0_lambda=args.l0_lambda, adam_eps=args.adam_eps,
        device=device, on_step=on_step, logger=logger, log_every=50,
    )
    scores = result.scores
    loss_log = result.loss_log
    train_log = result.train_log
    train_time = result.train_time_s
    logger.info("Training complete in %.1fs (%s)", train_time, cf_cache.stats())

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
            level="node", absolute=False,
            intervention="zero" if args.ablation == "zero" else "patching")

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
