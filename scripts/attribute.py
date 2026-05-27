"""Learn neuron importance scores for a language model via adaptive sigmoid top-k masking."""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

from learning_to_attribute import sigmoid_topk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--text", default="The capital of France is")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--T", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--n_iters", type=int, default=30)
    parser.add_argument("--output", default="plots/attribution")
    parser.add_argument("--loss", choices=["kl", "top5"], default="kl",
                        help="kl: minimize KL to ref distribution. "
                             "top5: maximize sum of top-5 ref logits.")
    parser.add_argument("--chat", action="store_true",
                        help="Use instruct chat template (Llama 3.1 format)")
    parser.add_argument("--seed_response", default=None,
                        help="Seed the assistant response (e.g. 'Answer:')")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.gradient_checkpointing_enable()

    # Tokenize
    if args.chat:
        messages = [{"role": "user", "content": args.text}]
        if args.seed_response:
            messages.append({"role": "assistant", "content": args.seed_response})
        rendered = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=args.seed_response is None,
            tokenize=False,
        )
        input_ids_list = tokenizer.encode(rendered, add_special_tokens=False)
        if args.seed_response:
            while input_ids_list and input_ids_list[-1] == tokenizer.eos_token_id:
                input_ids_list.pop()
        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
    else:
        input_ids = tokenizer(args.text, return_tensors="pt").input_ids.to(device)
    seq_len = input_ids.shape[1]
    tokens = [tokenizer.decode(t) for t in input_ids[0]]
    print(f"Input: {args.text!r} -> {seq_len} tokens: {tokens}")

    # Model dimensions
    config = model.config
    num_layers = config.num_hidden_layers
    intermediate_size = config.intermediate_size
    total = num_layers * seq_len * intermediate_size
    print(f"Neurons: {num_layers} layers x {seq_len} pos x {intermediate_size} dim = {total:,}")

    # Cache reference logits
    with torch.no_grad():
        ref_logits = model(input_ids).logits[0, -1].float()
        ref_probs = F.softmax(ref_logits, dim=-1)
    top5_ref = ref_logits.topk(5)
    top5_indices = top5_ref.indices
    print(f"Reference top-5: {[tokenizer.decode(t) for t in top5_indices.tolist()]}")

    # Score tensor (global flat, float32)
    scores = nn.Parameter(torch.zeros(total, device=device))
    optimizer = torch.optim.Adam([scores], lr=args.lr)

    # Mutable container for current mask, read by hooks
    state = {"mask": None}

    # Register hooks on each layer's MLP down_proj
    hooks = []
    for layer_idx in range(num_layers):
        def make_hook(li):
            def hook(module, hook_args):
                x = hook_args[0]  # [1, seq_len, intermediate_size]
                start = li * seq_len * intermediate_size
                end = start + seq_len * intermediate_size
                m = state["mask"][start:end].view(1, seq_len, intermediate_size)
                return (x * m.to(x.dtype),)
            return hook
        layer = model.model.layers[layer_idx]
        h = layer.mlp.down_proj.register_forward_pre_hook(make_hook(layer_idx))
        hooks.append(h)

    # Training loop
    loss_log = []
    print(f"\nTraining for {args.steps} steps...")
    for step in range(args.steps):
        k = 1.0 + (total - 1.0) * torch.rand(1).item()
        state["mask"] = sigmoid_topk(scores, k=k, T=args.T, n_iters=args.n_iters)

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
        if (step + 1) % 50 == 0 or step == 0:
            print(f"  Step {step+1:>4d}/{args.steps}  loss={loss_val:.6f}  k={k:.0f}/{total}")

    # === Sparsity evaluation ===
    print("\nEvaluating metric vs sparsity...")
    sparsities = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
    flat_scores = scores.data.clone()
    sorted_idx = flat_scores.argsort(descending=True)

    torch.manual_seed(0)
    random_idx = torch.randperm(total, device=device)

    eval_learned = []
    eval_random = []

    def eval_metric(logits):
        if args.loss == "kl":
            log_probs = F.log_softmax(logits, dim=-1)
            return F.kl_div(log_probs, ref_probs, reduction="batchmean").item()
        else:
            return logits[top5_indices].sum().item()

    for frac in sparsities:
        k = max(1, int(frac * total))

        for ordering, eval_list in [
            (sorted_idx, eval_learned),
            (random_idx, eval_random),
        ]:
            hard_mask = torch.zeros(total, device=device)
            hard_mask[ordering[:k]] = 1.0
            state["mask"] = hard_mask

            with torch.no_grad():
                logits = model(input_ids).logits[0, -1].float()
                val = eval_metric(logits)
            eval_list.append(val)

        metric_name = "KL" if args.loss == "kl" else "top5_sum"
        print(f"  keep={frac:6.1%} ({k:>7d}/{total})  "
              f"{metric_name}_learned={eval_learned[-1]:.4f}  "
              f"{metric_name}_random={eval_random[-1]:.4f}")

    # Remove hooks
    for h in hooks:
        h.remove()

    # Save scores
    scores_3d = scores.data.view(num_layers, seq_len, intermediate_size).cpu()
    torch.save({"scores": scores_3d, "tokens": tokens, "text": args.text,
                "args": vars(args), "sparsity_eval": {
                    "sparsities": sparsities,
                    "eval_learned": eval_learned,
                    "eval_random": eval_random,
                }}, f"{args.output}_scores.pt")
    print(f"\nSaved scores to {args.output}_scores.pt")

    # Top-50 neurons
    flat_scores_cpu = scores_3d.flatten()
    top_vals, top_idxs = flat_scores_cpu.topk(50)
    print(f"\nTop-50 neurons (layer, pos, neuron_idx, score):")
    for rank, (val, idx) in enumerate(zip(top_vals, top_idxs)):
        idx = idx.item()
        layer = idx // (seq_len * intermediate_size)
        rem = idx % (seq_len * intermediate_size)
        pos = rem // intermediate_size
        neuron = rem % intermediate_size
        tok = tokens[pos]
        print(f"  {rank+1:>3d}. L{layer:>2d} pos={pos:>2d} ({tok!r:>10s}) neuron={neuron:>5d}  score={val:.4f}")

    # Visualization: 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(18, 5),
                             gridspec_kw={"width_ratios": [2, 1, 1]})

    # Heatmap: max score per (layer, position)
    heatmap = scores_3d.max(dim=-1).values.numpy()
    im = axes[0].imshow(heatmap, aspect="auto", cmap="viridis")
    axes[0].set_xlabel("Token position")
    axes[0].set_ylabel("Layer")
    axes[0].set_title("Max neuron importance per (layer, position)")
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
    metric_label = "KL Divergence" if args.loss == "kl" else "Top-5 Logit Sum"
    axes[2].plot(pct, eval_learned, "o-", label="Learned", markersize=4, linewidth=1.2)
    axes[2].plot(pct, eval_random, "o--", label="Random", markersize=4, linewidth=1.2)
    axes[2].set_xlabel("% neurons kept")
    axes[2].set_ylabel(metric_label)
    axes[2].set_title(f"{metric_label} vs Sparsity")
    axes[2].set_xscale("log")
    if args.loss == "kl":
        axes[2].set_yscale("log")
    axes[2].legend()

    fig.suptitle(f"Sigmoid Top-K Attribution: {args.text!r}", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{args.output}.png", dpi=150)
    print(f"Saved {args.output}.png")


if __name__ == "__main__":
    main()
