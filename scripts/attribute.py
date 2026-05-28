"""Learn importance scores for a language model via adaptive sigmoid top-k masking."""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from learning_to_attribute import sigmoid_topk, LlamaAttributionHooks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config (relative to scripts/ or absolute)")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--text", default="The capital of France is")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--T", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--n_iters", type=int, default=30)
    parser.add_argument("--output", default="results/attribution")
    parser.add_argument("--loss", choices=["kl", "top5"], default="kl")
    parser.add_argument("--mask", default="mlp",
                        choices=list(LlamaAttributionHooks.MASK_TYPES))
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--seed_response", default=None)
    parser.add_argument("--cf_text", default=None,
                        help="Counterfactual text (must tokenize to same length as --text)")
    parser.add_argument("--flip", action="store_true",
                        help="Top-k get CF activation; target is CF distribution")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", default="learning-to-attribute")
    parser.add_argument("--wandb_name", default=None,
                        help="W&B run name (defaults to output basename)")

    # Load config YAML as defaults (CLI overrides)
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

    # Output setup
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # W&B init
    if args.wandb:
        import wandb
        run_name = args.wandb_name or Path(args.output).name
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
    else:
        wandb = None

    # Log full config
    logger.info("Config: %s", json.dumps(vars(args), indent=2, default=str))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load model
    logger.info("Loading model %s...", args.model)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.gradient_checkpointing_enable()
    logger.info("Model loaded in %.1fs", time.time() - t0)

    # Tokenize helper
    def tokenize_text(text):
        if args.chat:
            messages = [{"role": "user", "content": text}]
            if args.seed_response:
                messages.append({"role": "assistant", "content": args.seed_response})
            rendered = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=args.seed_response is None,
                tokenize=False,
            )
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

    # Counterfactual setup
    cf_logits = None
    if args.cf_text:
        cf_input_ids = tokenize_text(args.cf_text)
        cf_tokens = [tokenizer.decode(t) for t in cf_input_ids[0]]
        assert cf_input_ids.shape[1] == seq_len, (
            f"Counterfactual must have same token length as input "
            f"({cf_input_ids.shape[1]} vs {seq_len})")
        logger.info("CF: %r -> %d tokens: %s", args.cf_text, cf_input_ids.shape[1], cf_tokens)

    # Set up hooks
    hooker = LlamaAttributionHooks(model, args.mask, seq_len, flip=args.flip)
    total = hooker.total
    logger.info("Scores: %s", hooker.describe())

    # Cache counterfactual activations
    if args.cf_text:
        cf_logits = hooker.cache_cf_activations(cf_input_ids)
        cf_top5 = cf_logits.topk(5)
        logger.info("CF top-5: %s", [tokenizer.decode(t) for t in cf_top5.indices.tolist()])

    # Cache clean logits
    with torch.no_grad():
        clean_logits = model(input_ids).logits[0, -1].float()
        clean_probs = F.softmax(clean_logits, dim=-1)
    top5_clean = clean_logits.topk(5)
    logger.info("Clean top-5: %s", [tokenizer.decode(t) for t in top5_clean.indices.tolist()])

    # Training target
    if args.flip and args.cf_text:
        ref_probs = F.softmax(cf_logits, dim=-1)
        top5_indices = cf_logits.topk(5).indices
        logger.info("Target: CF distribution (--flip)")
    else:
        ref_probs = clean_probs
        top5_indices = top5_clean.indices

    # Score tensor
    scores = nn.Parameter(torch.zeros(total, device=device))
    optimizer = torch.optim.Adam([scores], lr=args.lr)

    # Register masking hooks
    hooker.register_hooks()

    # Training loop
    loss_log = []
    logger.info("Training for %d steps (T=%.2f, lr=%.4f, mask=%s)...",
                args.steps, args.T, args.lr, args.mask)
    t0 = time.time()
    for step in range(args.steps):
        k = 1.0 + (total - 1.0) * torch.rand(1).item()
        hooker.mask = sigmoid_topk(scores, k=k, T=args.T, n_iters=args.n_iters)

        logits = model(input_ids).logits[0, -1].float()

        if args.loss == "kl":
            log_probs = F.log_softmax(logits, dim=-1)
            loss = F.kl_div(log_probs, ref_probs, reduction="batchmean")
        else:  # top5
            loss = -logits[top5_indices].sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        loss_log.append(loss_val)
        if wandb:
            wandb.log({"loss": loss_val, "k": k, "k_frac": k / total}, step=step)
        if (step + 1) % 50 == 0 or step == 0:
            elapsed = time.time() - t0
            rate = (step + 1) / elapsed
            logger.info("Step %4d/%d  loss=%.6f  k=%.0f/%d  (%.1f step/s)",
                        step + 1, args.steps, loss_val, k, total, rate)

    train_time = time.time() - t0
    logger.info("Training complete in %.1fs (%.2f step/s)",
                train_time, args.steps / train_time)
    if wandb:
        wandb.log({"train_time_s": train_time,
                    "steps_per_s": args.steps / train_time})

    # === Sparsity evaluation ===
    logger.info("Evaluating metric vs sparsity...")
    sparsities = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
    flat_scores = scores.data.clone()
    sorted_idx = flat_scores.argsort(descending=True)

    torch.manual_seed(0)
    random_idx = torch.randperm(total, device=device)

    eval_learned = []
    eval_random = []
    eval_learned_other = []
    eval_random_other = []

    cf_probs = F.softmax(cf_logits, dim=-1) if args.cf_text else None

    def eval_metric(logits):
        if args.loss == "kl":
            log_probs = F.log_softmax(logits, dim=-1)
            return F.kl_div(log_probs, ref_probs, reduction="batchmean").item()
        else:
            return logits[top5_indices].sum().item()

    def eval_kl_other(logits):
        log_probs = F.log_softmax(logits, dim=-1)
        other_probs = clean_probs if args.flip else cf_probs
        return F.kl_div(log_probs, other_probs, reduction="batchmean").item()

    for frac in sparsities:
        k = max(1, int(frac * total))

        for ordering, eval_list, eval_list_other in [
            (sorted_idx, eval_learned, eval_learned_other),
            (random_idx, eval_random, eval_random_other),
        ]:
            hard_mask = torch.zeros(total, device=device)
            hard_mask[ordering[:k]] = 1.0
            hooker.mask = hard_mask

            with torch.no_grad():
                logits = model(input_ids).logits[0, -1].float()
                val = eval_metric(logits)
            eval_list.append(val)
            if args.cf_text:
                eval_list_other.append(eval_kl_other(logits))

        metric_name = "KL" if args.loss == "kl" else "top5_sum"
        other_name = "KL_clean" if args.flip else "KL_cf"
        logger.info("keep=%6.1f%% (%7d/%d)  %s_learned=%.4f  %s_random=%.4f%s",
                     frac * 100, k, total,
                     metric_name, eval_learned[-1],
                     metric_name, eval_random[-1],
                     f"  {other_name}_L={eval_learned_other[-1]:.6f}  "
                     f"{other_name}_R={eval_random_other[-1]:.6f}"
                     if args.cf_text else "")

    hooker.remove_hooks()

    # Log sparsity eval to wandb
    if wandb:
        for i, frac in enumerate(sparsities):
            log_dict = {
                "eval/sparsity": frac,
                "eval/metric_learned": eval_learned[i],
                "eval/metric_random": eval_random[i],
            }
            if args.cf_text:
                log_dict["eval/other_learned"] = eval_learned_other[i]
                log_dict["eval/other_random"] = eval_random_other[i]
            wandb.log(log_dict)

    # Save scores
    save_dict = {"scores": scores.data.cpu(), "tokens": tokens,
                 "text": args.text, "cf_text": args.cf_text,
                 "mask_type": args.mask,
                 "args": vars(args),
                 "loss_log": loss_log,
                 "train_time_s": train_time,
                 "sparsity_eval": {
                     "sparsities": sparsities,
                     "eval_learned": eval_learned,
                     "eval_random": eval_random,
                 }}
    if args.cf_text:
        save_dict["sparsity_eval"]["eval_learned_other"] = eval_learned_other
        save_dict["sparsity_eval"]["eval_random_other"] = eval_random_other
    torch.save(save_dict, f"{args.output}_scores.pt")
    logger.info("Saved scores to %s_scores.pt", args.output)

    # Top-50
    flat_scores_cpu = scores.data.cpu()
    top_vals, top_idxs = flat_scores_cpu.topk(min(50, total))
    logger.info("Top-50 (component, layer, pos, neuron/head, score):")
    for rank, (val, idx) in enumerate(zip(top_vals, top_idxs)):
        info = hooker.decode_index(idx.item())
        tok = tokens[info["pos"]]
        if info["component"] == "mlp":
            label = f"mlp n={info['neuron']:>5d}"
        elif info["component"] == "resid":
            label = "resid"
        elif "head" in info:
            label = f"attn h={info['head']:>2d}"
        else:
            label = "attn (full)"
        logger.info("  %3d. L%2d pos=%2d (%10r) %s  score=%.4f",
                     rank + 1, info["layer"], info["pos"], tok, label, val)

    # Visualization: 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(18, 5),
                             gridspec_kw={"width_ratios": [2, 1, 1]})

    # Heatmap
    heatmap = hooker.scores_to_heatmap(flat_scores_cpu).numpy()
    im = axes[0].imshow(heatmap, aspect="auto", cmap="viridis")
    axes[0].set_xlabel("Token position")
    axes[0].set_ylabel("Layer")
    axes[0].set_title(f"Max importance per (layer, pos) [{args.mask}]")
    axes[0].set_xticks(range(seq_len))
    axes[0].set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    # Loss curve
    loss_label = "KL Divergence" if args.loss == "kl" else "-top5_logit_sum"
    axes[1].plot(loss_log, linewidth=0.8)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel(loss_label)
    axes[1].set_title("Training Loss")
    if args.loss == "kl":
        axes[1].set_yscale("log")

    # Metric vs sparsity
    pct = [s * 100 for s in sparsities]
    if args.cf_text:
        target_lbl = "→CF" if args.flip else "→Clean"
        other_lbl = "→Clean" if args.flip else "→CF"
        axes[2].plot(pct, eval_learned, "o-", label=f"Learned ({target_lbl})", markersize=4, linewidth=1.2)
        axes[2].plot(pct, eval_random, "o--", label=f"Random ({target_lbl})", markersize=4, linewidth=1.2, alpha=0.6)
        axes[2].plot(pct, eval_learned_other, "s-", label=f"Learned ({other_lbl})", markersize=4, linewidth=1.2, color="C2")
        axes[2].plot(pct, eval_random_other, "s--", label=f"Random ({other_lbl})", markersize=4, linewidth=1.2, color="C3", alpha=0.6)
        xlabel = "% patched to CF" if args.flip else "% kept (clean)"
        axes[2].set_xlabel(xlabel)
        axes[2].set_ylabel("KL Divergence")
        axes[2].set_title(f"KL vs Sparsity ({target_lbl}, {other_lbl})")
        axes[2].set_xscale("log")
        axes[2].set_yscale("log")
        axes[2].legend(fontsize=7)
    else:
        metric_label = "KL Divergence" if args.loss == "kl" else "Top-5 Logit Sum"
        axes[2].plot(pct, eval_learned, "o-", label="Learned", markersize=4, linewidth=1.2)
        axes[2].plot(pct, eval_random, "o--", label="Random", markersize=4, linewidth=1.2)
        axes[2].set_xlabel("% kept")
        axes[2].set_ylabel(metric_label)
        axes[2].set_title(f"{metric_label} vs Sparsity")
        axes[2].set_xscale("log")
        if args.loss == "kl":
            axes[2].set_yscale("log")
        axes[2].legend()

    fig.suptitle(f"Sigmoid Top-K Attribution [{args.mask}]: {args.text!r}", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{args.output}.png", dpi=150)
    logger.info("Saved plot to %s.png", args.output)

    if wandb:
        wandb.log({"plot": wandb.Image(f"{args.output}.png")})
        wandb.finish()


if __name__ == "__main__":
    main()
