"""Multitask MAttr on arithmetic-wild (Llama-3.1-8B base), sufficient/denoising.

mask types:
  das  - ONE shared per-layer rotation across tasks, per-(task,span,dim) scores.
  mlp  - per-(task,layer,span,neuron) scores over MLP down_proj-input neurons (no rotation).
  sae  - shared per-layer Llama Scope SAE, per-(task,span,feature) scores.

Sufficient (top-k keep base, complement patched to cf), target = base answer token.
k-budget sampled UNIFORM during training.
"""
import argparse, json, pickle, time, logging, sys, os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from learning_to_attribute.data.arithmetic_wild import ArithmeticWildDataset
from learning_to_attribute.sigmoid_das import make_rotate_layer
from attribute_multitask import sample_k, train_step, eval_task, get_hooks_classes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("arith")
DATA_DIR = "/home/guests/aryaman/arithmetic-wild/datasets/Llama-3.1-8B"


def build_task(model, task, args, legacy_sufficient, shared, device):
    _, SpanHooksCls = get_hooks_classes(model)
    ds = ArithmeticWildDataset(task, DATA_DIR, seed=args.seed)
    hooker = SpanHooksCls(model, args.mask, ds.num_spans,
                          pos_strategy=args.pos_strategy, sufficient=legacy_sufficient)
    if args.mask == "das":
        hooker.set_das_dim(args.das_dim)
        for li in range(hooker.num_layers):
            hooker.R[li] = shared["rot"][li]
    elif args.mask == "sae":
        hooker.set_saes(shared["sae"])           # added to hooker
    scores = nn.Parameter(torch.zeros(hooker.total, device=device))
    return ds, hooker, scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--mask", default="mlp", choices=["das", "mlp", "sae"])
    p.add_argument("--tasks", default="months,weekdays,hours")
    p.add_argument("--das_dim", type=int, default=16)
    p.add_argument("--sae_repo", default="fnlp/Llama3_1-8B-Base-LXR-8x")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--T", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--lr_rotation", type=float, default=0.001)
    p.add_argument("--n_iters", type=int, default=30)
    p.add_argument("--k_schedule", default="uniform", choices=["uniform", "log"])
    p.add_argument("--hard_fwd", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sufficient", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pos_strategy", default="last", choices=["first", "last", "all"])
    p.add_argument("--loss", default="ce", choices=["ce", "kl"])
    p.add_argument("--n_eval", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="results/arith_mlp")
    args = p.parse_args()

    legacy_sufficient = not args.sufficient
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    for pr in model.parameters():
        pr.requires_grad_(False)

    tasks = args.tasks.split(",")
    hidden = model.config.hidden_size
    n_layers = model.config.num_hidden_layers

    shared = {}
    if args.mask == "das":
        shared["rot"] = {li: make_rotate_layer(hidden, args.das_dim).to(device) for li in range(n_layers)}
    elif args.mask == "sae":
        from learning_to_attribute.sae_loader import load_llama_scope_saes
        shared["sae"] = load_llama_scope_saes(args.sae_repo, n_layers, device)

    datasets, hookers, scores_by_task = {}, {}, {}
    for t in tasks:
        ds, hooker, sc = build_task(model, t, args, legacy_sufficient, shared, device)
        datasets[t], hookers[t], scores_by_task[t] = ds, hooker, sc
        logger.info("  %-12s spans=%d total=%d", t, ds.num_spans, hooker.total)

    param_groups = [{"params": list(scores_by_task.values()), "lr": args.lr}]
    if args.mask == "das":
        rot_params = [pp for rl in shared["rot"].values() for pp in rl.parameters()]
        param_groups.append({"params": rot_params, "lr": args.lr_rotation})
    optimizer = torch.optim.Adam(param_groups)

    logger.info("mask=%s tasks=%s steps=%d sufficient=%s", args.mask, tasks, args.steps, args.sufficient)
    running = {t: [] for t in tasks}
    loss_log = []
    t0 = time.time()
    for step in range(args.steps):
        optimizer.zero_grad()
        for t in tasks:
            loss, hooker = train_step(model, datasets[t], hookers[t], scores_by_task[t],
                                      tokenizer, device, args, legacy_sufficient)
            if step < 3 and not torch.isfinite(loss):
                logger.warning("NONFINITE loss task=%s step=%d loss=%s", t, step, loss.item())
            loss.backward()
            hooker.remove_hooks()
            running[t].append(loss.item())
        optimizer.step()
        if (step + 1) % max(1, args.steps // 100) == 0:
            avg = float(np.mean([np.mean(running[t][-20:]) for t in tasks]))
            loss_log.append((step + 1, avg))
            logger.info("step %5d/%d  mean_loss=%.4f  (%.1f task-fwd/s)",
                        step + 1, args.steps, avg, (step + 1) * len(tasks) / (time.time() - t0))

    sparsities = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    per_task_eval = {}
    for t in tasks:
        per_task_eval[t] = eval_task(model, datasets[t], hookers[t], scores_by_task[t],
                                     tokenizer, device, args, legacy_sufficient, sparsities)
        logger.info("  %s: CE_learned@0.5%%=%.4f", t, per_task_eval[t]["eval_learned_ce"][3])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "tasks": tasks, "mask": args.mask,
        "scores": {t: sc.detach().cpu() for t, sc in scores_by_task.items()},
        "num_spans": {t: datasets[t].num_spans for t in tasks},
        "span_names": {t: list(datasets[t].span_names) for t in tasks},
        "n_layers": n_layers, "intermediate_size": model.config.intermediate_size,
        "das_dim": args.das_dim if args.mask == "das" else None,
        "per_task_eval": per_task_eval, "sparsities": sparsities,
        "loss_log": loss_log, "args": vars(args),
    }
    if args.mask == "das":
        result["das_rotations"] = {li: rl.weight.data.cpu() for li, rl in shared["rot"].items()}
    with open(out.with_suffix(".pkl"), "wb") as f:
        pickle.dump(result, f)
    logger.info("saved %s.pkl", out)


if __name__ == "__main__":
    main()
