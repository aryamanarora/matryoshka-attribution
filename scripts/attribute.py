"""Learn importance scores for a language model via adaptive sigmoid top-k masking."""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from learning_to_attribute import (
    sigmoid_topk, sigmoid_topk_hard, make_rotate_layer, CausalGymDataset, learn_scores,
    normalize_mode, wandb_util,
)
from learning_to_attribute.models import (
    LlamaAttributionHooks, LlamaSpanAttributionHooks,
    GPTNeoXAttributionHooks, GPTNeoXSpanAttributionHooks,
    GPT2AttributionHooks, GPT2SpanAttributionHooks,
)

# Map model_type from config to hook classes
HOOKS_REGISTRY = {
    "llama": (LlamaAttributionHooks, LlamaSpanAttributionHooks),
    "gpt_neox": (GPTNeoXAttributionHooks, GPTNeoXSpanAttributionHooks),
    "gpt2": (GPT2AttributionHooks, GPT2SpanAttributionHooks),
    "qwen2": (LlamaAttributionHooks, LlamaSpanAttributionHooks),
    "gemma2": (LlamaAttributionHooks, LlamaSpanAttributionHooks),
}


def get_hooks_classes(model):
    """Auto-detect model architecture and return (HooksCls, SpanHooksCls)."""
    model_type = model.config.model_type
    if model_type in HOOKS_REGISTRY:
        return HOOKS_REGISTRY[model_type]
    raise ValueError(f"Unsupported model_type: {model_type!r}. "
                     f"Supported: {list(HOOKS_REGISTRY.keys())}")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def sample_k(total: int, schedule: str = "uniform") -> float:
    """Sample k for sigmoid top-k masking.

    Args:
        total: total number of score parameters
        schedule: "uniform" samples k ~ Uniform(1, total),
                  "log" samples k ~ exp(Uniform(log(1), log(total)))
                  so that 1-10 is as likely as 10-100 as 100-1000.
    """
    if schedule == "log":
        import math
        log_k = math.log(1) + (math.log(total) - math.log(1)) * torch.rand(1).item()
        return math.exp(log_k)
    else:
        return 1.0 + (total - 1.0) * torch.rand(1).item()


def run_single(args, model, tokenizer, device, wandb):
    """Single text/cf_text pair mode (original behavior)."""

    def tokenize_text(text):
        if args.chat:
            messages = [{"role": "user", "content": text}]
            if args.seed_response:
                messages.append({"role": "assistant", "content": args.seed_response})
            rendered = tokenizer.apply_chat_template(
                messages, add_generation_prompt=args.seed_response is None,
                tokenize=False)
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            if args.seed_response:
                while ids and ids[-1] == tokenizer.eos_token_id:
                    ids.pop()
            return torch.tensor([ids], dtype=torch.long, device=device)
        else:
            return tokenizer(text, return_tensors="pt").input_ids.to(device)

    input_ids = tokenize_text(args.text)
    seq_len = input_ids.shape[1]
    tokens = [tokenizer.decode(t) for t in input_ids[0]]
    logger.info("Input: %r -> %d tokens: %s", args.text, seq_len, tokens)

    cf_logits = None
    if args.cf_text:
        cf_input_ids = tokenize_text(args.cf_text)
        assert cf_input_ids.shape[1] == seq_len, (
            f"CF must have same token length ({cf_input_ids.shape[1]} vs {seq_len})")
        logger.info("CF: %r -> %d tokens", args.cf_text, cf_input_ids.shape[1])

    HooksCls, _ = get_hooks_classes(model)
    hooker = HooksCls(model, args.mask, seq_len, sufficient=args.sufficient)
    total = hooker.total
    logger.info("Scores: %s", hooker.describe())

    if args.cf_text:
        cf_logits = hooker.cache_cf_activations(cf_input_ids)
        logger.info("CF top-5: %s", [tokenizer.decode(t) for t in cf_logits.topk(5).indices.tolist()])

    with torch.no_grad():
        clean_logits = model(input_ids).logits[0, -1].float()
        clean_probs = F.softmax(clean_logits, dim=-1)
    logger.info("Clean top-5: %s", [tokenizer.decode(t) for t in clean_logits.topk(5).indices.tolist()])

    if args.sufficient and args.cf_text:
        ref_probs = F.softmax(cf_logits, dim=-1)
        top5_indices = cf_logits.topk(5).indices
    else:
        ref_probs = clean_probs
        top5_indices = clean_logits.topk(5).indices

    hooker.register_hooks()
    logger.info("Training for %d steps...", args.steps)

    def loss_fn(mask):
        hooker.mask = mask
        logits = model(input_ids).logits[0, -1].float()
        if args.loss == "kl":
            return F.kl_div(F.log_softmax(logits, dim=-1), ref_probs, reduction="batchmean")
        elif args.loss == "top5":
            return -logits[top5_indices].sum()

    on_step = (lambda step, k, lv, sc: wandb.log(
        {"loss": lv, "k": k, "k_frac": k / total}, step=step)) if wandb else None
    res = learn_scores(total, loss_fn, steps=args.steps, variant="topk",
                       k_schedule=args.k_schedule, T=args.T, n_iters=args.n_iters,
                       lr=args.lr, device=device, on_step=on_step, logger=logger, log_every=50)
    scores = res.scores.to(device)        # back on device for the sparsity eval
    loss_log = res.loss_log
    train_time = res.train_time_s

    # Sparsity eval
    sparsities = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
    eval_results = _eval_sparsity(
        model, hooker, scores, input_ids, total, sparsities, device,
        ref_probs, args.loss, top5_indices,
        other_probs=(clean_probs if args.sufficient else F.softmax(cf_logits, dim=-1)) if args.cf_text else None,
        has_cf=bool(args.cf_text), sufficient=args.sufficient, wandb=wandb)

    hooker.remove_hooks()

    return {
        "scores": scores.data.cpu(), "tokens": tokens, "text": args.text,
        "cf_text": args.cf_text, "mask_type": args.mask,
        "loss_log": loss_log, "train_time_s": train_time,
        "hooker": hooker, "x_labels": tokens,
        "sparsity_eval": eval_results, "sparsities": sparsities,
    }


def run_dataset(args, model, tokenizer, device, wandb):
    """CausalGym dataset mode: sample pairs each step, span-indexed scores."""

    dataset = CausalGymDataset(args.dataset, seed=args.seed)
    logger.info("Dataset: %s (%d spans: %s)", args.dataset, dataset.num_spans,
                dataset.span_names)

    _, SpanHooksCls = get_hooks_classes(model)
    hooker = SpanHooksCls(
        model, args.mask, dataset.num_spans,
        pos_strategy=args.pos_strategy, sufficient=args.sufficient)
    # DAS: set up low-rank rotation layers
    das_rotations = {}
    if args.mask == "das":
        das_dim = args.das_dim or hooker.hidden_size
        hooker.set_das_dim(das_dim)
        for li in range(hooker.num_layers):
            das_rotations[li] = make_rotate_layer(hooker.hidden_size, das_dim).to(device)
            hooker.R[li] = das_rotations[li]

    total = hooker.total
    logger.info("Scores: %s", hooker.describe())

    # DAS trains the low-rank rotation matrices alongside the scores (second param group).
    extra_params, lr_extra = None, None
    if args.mask == "das":
        lr_extra = args.lr_rotation if args.lr_rotation is not None else args.lr
        extra_params = [p for rl in das_rotations.values() for p in rl.parameters()]

    hooker.register_hooks()
    logger.info("Training for %d steps (dataset=%s, loss=%s, pos=%s, flip=%s)...",
                args.steps, args.dataset, args.loss, args.pos_strategy, args.sufficient)

    def loss_fn(mask):
        pair = dataset.sample_pair()
        tok = dataset.tokenize_pair(pair, tokenizer, device=str(device))
        hooker.set_alignment(
            tok.base_alignment, tok.src_alignment,
            tok.base_input_ids.shape[1], tok.src_input_ids.shape[1])
        src_logits = hooker.cache_cf_activations(tok.src_input_ids)
        hooker.mask = mask
        logits = model(tok.base_input_ids).logits[0, -1].float()
        if args.loss == "ce":
            target_id = tok.src_label_id if args.sufficient else tok.base_label_id
            return F.cross_entropy(logits.unsqueeze(0), torch.tensor([target_id], device=device))
        elif args.loss == "kl":
            ref_probs = F.softmax(src_logits if args.sufficient else logits.detach(), dim=-1)
            return F.kl_div(F.log_softmax(logits, dim=-1), ref_probs, reduction="batchmean")

    on_step = (lambda step, k, lv, sc: wandb.log(
        {"loss": lv, "k": k, "k_frac": k / total}, step=step)) if wandb else None
    res = learn_scores(
        total, loss_fn, steps=args.steps,
        variant="hard_topk" if args.hard_fwd else "topk",
        k_schedule=args.k_schedule, T=args.T, n_iters=args.n_iters, lr=args.lr,
        natural_k_frac=args.natural_k_frac, use_bias=False,
        extra_params=extra_params, lr_extra=lr_extra, device=device,
        on_step=on_step, logger=logger, log_every=50)
    scores = res.scores.to(device)        # back on device for the sparsity eval
    loss_log = res.loss_log
    train_time = res.train_time_s
    logger.info("Training complete in %.1fs (%.2f step/s)",
                train_time, args.steps / train_time)

    # Sparsity eval averaged over multiple pairs
    n_eval = getattr(args, "n_eval", 100) or 100
    logger.info("Evaluating sparsity averaged over %d pairs...", n_eval)
    sparsities = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]

    # Accumulate results across eval examples
    all_eval = []
    for ei in range(n_eval):
        eval_pair = dataset.sample_pair()
        eval_tok = dataset.tokenize_pair(eval_pair, tokenizer, device=str(device))
        hooker.set_alignment(
            eval_tok.base_alignment, eval_tok.src_alignment,
            eval_tok.base_input_ids.shape[1], eval_tok.src_input_ids.shape[1])

        src_logits_eval = hooker.cache_cf_activations(eval_tok.src_input_ids)
        with torch.no_grad():
            clean_logits_eval = model(eval_tok.base_input_ids).logits[0, -1].float()
        ref_probs = F.softmax(src_logits_eval if args.sufficient else clean_logits_eval, dim=-1)
        other_probs = F.softmax(clean_logits_eval if args.sufficient else src_logits_eval, dim=-1)

        ce_target = eval_tok.src_label_id if args.sufficient else eval_tok.base_label_id
        ev = _eval_sparsity(
            model, hooker, scores, eval_tok.base_input_ids, total, sparsities, device,
            ref_probs, "kl", None, other_probs=other_probs, has_cf=True,
            sufficient=args.sufficient, wandb=None, ce_target_id=ce_target)
        all_eval.append(ev)
        if (ei + 1) % 20 == 0:
            logger.info("  eval %d/%d done", ei + 1, n_eval)

    # Average across examples and save per-example data for CI
    import numpy as np
    eval_results = {"sparsities": sparsities}
    for key in ["eval_learned", "eval_random", "eval_learned_ce", "eval_random_ce",
                "eval_learned_other", "eval_random_other"]:
        vals = [e[key] for e in all_eval if key in e]
        if vals:
            arr = np.array(vals)
            eval_results[key] = arr.mean(axis=0).tolist()
            eval_results[key + "_std"] = arr.std(axis=0).tolist()
            eval_results[key + "_all"] = arr.tolist()
    logger.info("Averaged eval over %d examples:", n_eval)
    for i, frac in enumerate(sparsities):
        k = max(1, int(frac * total))
        logger.info("  keep=%5.1f%% (%d/%d)  KL_L=%.4f  KL_R=%.4f  CE_L=%.4f",
                     frac * 100, k, total,
                     eval_results["eval_learned"][i], eval_results["eval_random"][i],
                     eval_results.get("eval_learned_ce", [0]*len(sparsities))[i])

    hooker.remove_hooks()

    result = {
        "scores": scores.data.cpu(), "tokens": None,
        "text": args.dataset, "cf_text": None,
        "mask_type": args.mask, "dataset": args.dataset,
        "span_names": dataset.span_names,
        "loss_log": loss_log, "train_time_s": train_time,
        "hooker": hooker, "x_labels": dataset.span_names,
        "sparsity_eval": eval_results, "sparsities": sparsities,
    }
    if args.mask == "das":
        result["das_rotations"] = {li: rl.weight.data.cpu() for li, rl in das_rotations.items()}
    return result


def _eval_sparsity(model, hooker, scores, input_ids, total, sparsities, device,
                   ref_probs, loss_type, top5_indices, other_probs=None,
                   has_cf=False, sufficient=False, wandb=None, ce_target_id=None):
    """Run sparsity evaluation sweep."""
    flat_scores = scores.data.clone()
    sorted_idx = flat_scores.argsort(descending=True)
    torch.manual_seed(0)
    random_idx = torch.randperm(total, device=device)

    eval_learned, eval_random = [], []
    eval_learned_other, eval_random_other = [], []
    eval_learned_ce, eval_random_ce = [], []

    for frac in sparsities:
        k = max(1, int(frac * total))
        for ordering, el, elo, el_ce in [
            (sorted_idx, eval_learned, eval_learned_other, eval_learned_ce),
            (random_idx, eval_random, eval_random_other, eval_random_ce),
        ]:
            hard_mask = torch.zeros(total, device=device)
            hard_mask[ordering[:k]] = 1.0
            hooker.mask = hard_mask

            with torch.no_grad():
                logits = model(input_ids).logits[0, -1].float()
                lp = F.log_softmax(logits, dim=-1)
                val = F.kl_div(lp, ref_probs, reduction="batchmean").item()
                tid = ce_target_id if ce_target_id is not None else ref_probs.argmax().item()
                ce_val = F.cross_entropy(logits.unsqueeze(0),
                                         torch.tensor([tid], device=device)).item()
            el.append(val)
            el_ce.append(ce_val)
            if has_cf and other_probs is not None:
                elo.append(F.kl_div(lp, other_probs, reduction="batchmean").item())

        other_name = "KL_clean" if sufficient else "KL_cf"
        line = f"keep={frac:6.1%} ({k:>7d}/{total})  KL_learned={eval_learned[-1]:.4f}  KL_random={eval_random[-1]:.4f}  CE_learned={eval_learned_ce[-1]:.4f}"
        if has_cf and other_probs is not None:
            line += f"  {other_name}_L={eval_learned_other[-1]:.6f}  {other_name}_R={eval_random_other[-1]:.6f}"
        logger.info(line)

    if wandb:
        for i, frac in enumerate(sparsities):
            d = {"eval/sparsity": frac, "eval/metric_learned": eval_learned[i],
                 "eval/metric_random": eval_random[i]}
            if has_cf and eval_learned_other:
                d["eval/other_learned"] = eval_learned_other[i]
                d["eval/other_random"] = eval_random_other[i]
            wandb.log(d)

    result = {"sparsities": sparsities, "eval_learned": eval_learned,
              "eval_random": eval_random,
              "eval_learned_ce": eval_learned_ce, "eval_random_ce": eval_random_ce}
    if eval_learned_other:
        result["eval_learned_other"] = eval_learned_other
        result["eval_random_other"] = eval_random_other
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--text", default="The capital of France is")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--T", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--n_iters", type=int, default=30)
    parser.add_argument("--output", default="results/attribution")
    parser.add_argument("--loss", choices=["kl", "top5", "ce"], default="kl")
    parser.add_argument("--mask", default="mlp",
                        choices=list(LlamaAttributionHooks.MASK_TYPES))
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--seed_response", default=None)
    parser.add_argument("--cf_text", default=None)
    parser.add_argument("--sufficient", action=argparse.BooleanOptionalAction, default=True,
                        help="Sufficient/denoising (default): top-k stay CLEAN, complement "
                             "corrupted; recover clean behavior. Use --no-sufficient for "
                             "necessary/noising (top-k get CF, find what flips). "
                             "Matches eval_mib.py's convention.")
    parser.add_argument("--mode", default=None,
                        choices=["iso", "cause", "sufficient", "necessary"],
                        help="Preferred alias for --sufficient: iso (=sufficient/denoising) "
                             "or cause (=necessary/noising). Overrides --sufficient if set.")
    parser.add_argument("--dataset", default=None,
                        help="CausalGym task, e.g. syntaxgym/agr_gender")
    parser.add_argument("--pos_strategy", default="last",
                        choices=["first", "last", "all"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k_schedule", default="uniform",
                        choices=["uniform", "log"],
                        help="How to sample k: uniform or log-uniform")
    parser.add_argument("--lr_rotation", type=float, default=None,
                        help="Learning rate for DAS rotation matrices (defaults to --lr)")
    parser.add_argument("--n_eval", type=int, default=100,
                        help="Number of examples for sparsity evaluation")
    parser.add_argument("--hard_fwd", action="store_true",
                        help="Hard binary mask in forward, straight-through gradient backward")
    parser.add_argument("--natural_k_frac", type=float, default=0.0,
                        help="Fraction of steps using natural k (all scores >= 0)")
    parser.add_argument("--das_dim", type=int, default=None,
                        help="DAS rotation subspace dimension (default: full d_model)")
    # underscore spelling to match every other option in this script
    wandb_util.add_args(parser, dash=False)   # ON by default; project defaults to l2a-causalgym
    parser.add_argument("--wandb_name", default=None)

    # Config YAML
    temp_args, _ = parser.parse_known_args()
    if temp_args.config:
        config_path = Path(temp_args.config)
        if not config_path.is_absolute():
            config_path = Path(__file__).parent / config_path
        logger.info("Loading config from %s", config_path)
        with open(config_path) as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            dest = key.replace("-", "_")
            for action in parser._actions:
                if action.dest == dest:
                    if isinstance(value, bool) and action.const is not None:
                        action.default = value
                    else:
                        action.default = value
                    break

    args = parser.parse_args()
    # --mode (iso/cause, the preferred names) overrides --sufficient when given, in the
    # user-facing sense (iso=sufficient/denoising -> True) BEFORE the legacy inversion below.
    if args.mode is not None:
        args.sufficient = (normalize_mode(args.mode) == "sufficient")
    # `--sufficient`/`sufficient:` (CLI + config) uses the consistent convention shared
    # with eval_mib.py: True = top-k stay CLEAN, complement corrupted (denoising /
    # sufficiency). The code below + sigmoid_das.intervene use the legacy convention
    # (True = top-k get CF / noising), so translate once here.
    args.sufficient = not args.sufficient
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # `wandb` below is the RUN object, not the module -- .log()/.finish() are the only two
    # members used, and both exist on it. Project defaults to l2a-causalgym (one per dataset).
    wandb = wandb_util.init(
        "causalgym", args.wandb_name or Path(args.output).name, vars(args),
        project=args.wandb_project, entity=args.wandb_entity, enabled=args.wandb)

    logger.info("Config: %s", json.dumps(vars(args), indent=2, default=str))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load model
    logger.info("Loading model %s...", args.model)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.gradient_checkpointing_enable()
    logger.info("Model loaded in %.1fs", time.time() - t0)

    # Dispatch
    if args.dataset:
        result = run_dataset(args, model, tokenizer, device, wandb)
    else:
        result = run_single(args, model, tokenizer, device, wandb)

    # Save
    save_dict = {
        "scores": result["scores"], "text": result["text"],
        "mask_type": result["mask_type"], "args": vars(args),
        "loss_log": result["loss_log"], "train_time_s": result["train_time_s"],
        "sparsity_eval": result["sparsity_eval"],
    }
    if result.get("tokens"):
        save_dict["tokens"] = result["tokens"]
    if result.get("span_names"):
        save_dict["span_names"] = result["span_names"]
    if result.get("cf_text"):
        save_dict["cf_text"] = result["cf_text"]
    if result.get("dataset"):
        save_dict["dataset"] = result["dataset"]
    if result.get("das_rotations"):
        save_dict["das_rotations"] = result["das_rotations"]
    torch.save(save_dict, f"{args.output}_scores.pt")
    logger.info("Saved scores to %s_scores.pt", args.output)

    # Top-50
    hooker = result["hooker"]
    flat_scores_cpu = result["scores"]
    top_vals, top_idxs = flat_scores_cpu.topk(min(50, flat_scores_cpu.numel()))
    x_labels = result["x_labels"]
    span_names = result.get("span_names")
    logger.info("Top-50:")
    for rank, (val, idx) in enumerate(zip(top_vals, top_idxs)):
        info = hooker.decode_index(idx.item(), span_names) if span_names else hooker.decode_index(idx.item())
        pos_key = "span" if "span" in info else "pos"
        pos_val = info[pos_key]
        pos_label = x_labels[pos_val] if pos_val < len(x_labels) else "?"
        if info["component"] == "mlp":
            label = f"mlp n={info['neuron']:>5d}"
        elif info["component"] == "resid":
            label = "resid"
        elif "head" in info:
            label = f"attn h={info['head']:>2d}"
        else:
            label = "attn (full)"
        logger.info("  %3d. L%2d %s=%2d (%10r) %s  score=%.4f",
                     rank + 1, info["layer"], pos_key, pos_val, pos_label, label, val)

    # Visualization
    fig, axes = plt.subplots(1, 4, figsize=(22, 5),
                             gridspec_kw={"width_ratios": [2, 1, 1, 1]})

    heatmap = hooker.scores_to_heatmap(flat_scores_cpu).numpy()
    n_x = heatmap.shape[1]
    im = axes[0].imshow(heatmap, aspect="auto", cmap="viridis")
    axes[0].set_xlabel("Span" if span_names else "Token position")
    axes[0].set_ylabel("Layer")
    axes[0].set_title(f"Max importance [{args.mask}]")
    axes[0].set_xticks(range(n_x))
    axes[0].set_xticklabels(x_labels[:n_x], rotation=45, ha="right", fontsize=7)
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    loss_label = {"kl": "KL", "top5": "-top5_sum", "ce": "CE"}[args.loss]
    axes[1].plot(result["loss_log"], linewidth=0.8)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel(loss_label)
    axes[1].set_title("Training Loss")

    ev = result["sparsity_eval"]
    pct = [s * 100 for s in result["sparsities"]]
    axes[2].plot(pct, ev["eval_learned"], "o-", label="Learned", markersize=4, linewidth=1.2)
    axes[2].plot(pct, ev["eval_random"], "o--", label="Random", markersize=4, linewidth=1.2)
    if "eval_learned_other" in ev:
        axes[2].plot(pct, ev["eval_learned_other"], "s-", label="Other (L)", markersize=4, linewidth=1.2, color="C2")
        axes[2].plot(pct, ev["eval_random_other"], "s--", label="Other (R)", markersize=4, linewidth=1.2, color="C3", alpha=0.6)
    axes[2].set_xlabel("% kept/patched")
    axes[2].set_ylabel("KL Divergence")
    axes[2].set_title("KL vs Sparsity")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].legend(fontsize=7)

    # CE vs sparsity
    if "eval_learned_ce" in ev:
        axes[3].plot(pct, ev["eval_learned_ce"], "o-", label="Learned", markersize=4, linewidth=1.2)
        axes[3].plot(pct, ev["eval_random_ce"], "o--", label="Random", markersize=4, linewidth=1.2)
        axes[3].set_xlabel("% kept/patched")
        axes[3].set_ylabel("Cross-Entropy")
        axes[3].set_title("CE vs Sparsity")
        axes[3].set_xscale("log")
        axes[3].legend(fontsize=7)

    title = f"[{args.mask}] {args.dataset or args.text!r}"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{args.output}.png", dpi=150)
    logger.info("Saved plot to %s.png", args.output)

    if wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
