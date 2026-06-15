"""Multi-task DAS: ONE shared per-layer rotation across all CausalGym tasks, with a
separate set of latent (span x dim) scores per task.

Hypothesis: a single learned causal feature basis per layer can serve every task,
each task selecting a sparse subset of dimensions. The shared rotation acts as a
strong regularizer against per-task DAS overfitting ("interpretability illusion").

Run on sc via nlprun. Default model pythia-1b, das_dim 64, sufficient (denoising) mode.
"""

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from learning_to_attribute import (  # noqa: E402
    sigmoid_topk, sigmoid_topk_hard, make_rotate_layer, CausalGymDataset,
)
from attribute import get_hooks_classes, sample_k, _eval_sparsity  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("multitask_das")


def all_syntaxgym_tasks():
    return [t for t in CausalGymDataset.list_tasks() if t.startswith("syntaxgym/")]


def build_task(model, task, das_dim, pos_strategy, legacy_sufficient, das_rotations, device):
    """Make a (dataset, hooker, scores) bundle for one task, sharing das_rotations."""
    _, SpanHooksCls = get_hooks_classes(model)
    ds = CausalGymDataset(task, seed=42)
    hooker = SpanHooksCls(model, "das", ds.num_spans,
                          pos_strategy=pos_strategy, sufficient=legacy_sufficient)
    hooker.set_das_dim(das_dim)
    for li in range(hooker.num_layers):
        hooker.R[li] = das_rotations[li]  # share by reference
    scores = nn.Parameter(torch.zeros(hooker.total, device=device))
    return ds, hooker, scores


def train_step(model, ds, hooker, scores, tokenizer, device, args, legacy_sufficient):
    pair = ds.sample_pair()
    tok = ds.tokenize_pair(pair, tokenizer, device=str(device))
    hooker.set_alignment(tok.base_alignment, tok.src_alignment,
                         tok.base_input_ids.shape[1], tok.src_input_ids.shape[1])
    src_logits = hooker.cache_cf_activations(tok.src_input_ids)  # sets mask=None internally
    hooker.register_hooks()
    try:
        k = sample_k(hooker.total, args.k_schedule)
        mask_fn = sigmoid_topk_hard if args.hard_fwd else sigmoid_topk
        hooker.mask = mask_fn(scores, k=k, T=args.T, n_iters=args.n_iters)
        # cache the orthogonalized rotations once per forward (else re-materialized
        # on every per-position .weight access -> dominant cost)
        with parametrize.cached():
            logits = model(tok.base_input_ids).logits[0, -1].float()
        if args.loss == "ce":
            target_id = tok.src_label_id if legacy_sufficient else tok.base_label_id
            loss = F.cross_entropy(logits.unsqueeze(0),
                                   torch.tensor([target_id], device=device))
        else:  # kl
            ref = F.softmax(src_logits if legacy_sufficient else logits.detach(), dim=-1)
            loss = F.kl_div(F.log_softmax(logits, dim=-1), ref, reduction="batchmean")
    finally:
        pass
    return loss, hooker


def eval_task(model, ds, hooker, scores, tokenizer, device, args, legacy_sufficient, sparsities):
    """Average sparsity sweep over n_eval pairs for one task."""
    all_eval = []
    for _ in range(args.n_eval):
        pair = ds.sample_pair()
        tok = ds.tokenize_pair(pair, tokenizer, device=str(device))
        hooker.set_alignment(tok.base_alignment, tok.src_alignment,
                             tok.base_input_ids.shape[1], tok.src_input_ids.shape[1])
        src_logits = hooker.cache_cf_activations(tok.src_input_ids)
        hooker.register_hooks()
        # rotation is fixed during eval -> cache it across all sparsity forwards
        with torch.no_grad(), parametrize.cached():
            clean_logits = model(tok.base_input_ids).logits[0, -1].float()
            ref = F.softmax(src_logits if legacy_sufficient else clean_logits, dim=-1)
            other = F.softmax(clean_logits if legacy_sufficient else src_logits, dim=-1)
            ce_target = tok.src_label_id if legacy_sufficient else tok.base_label_id
            ev = _eval_sparsity(model, hooker, scores, tok.base_input_ids, hooker.total,
                                sparsities, device, ref, "kl", None, other_probs=other,
                                has_cf=True, sufficient=legacy_sufficient, wandb=None,
                                ce_target_id=ce_target)
        all_eval.append(ev)
        hooker.remove_hooks()
    out = {"sparsities": sparsities}
    for key in ["eval_learned", "eval_random", "eval_learned_ce", "eval_random_ce",
                "eval_learned_other", "eval_random_other"]:
        vals = [e[key] for e in all_eval if key in e]
        if vals:
            arr = np.array(vals)
            out[key] = arr.mean(axis=0).tolist()
            out[key + "_std"] = arr.std(axis=0).tolist()
    return out


def feature_vectors(scores_by_task, hookers, das_dim):
    """Aggregate per-task scores to per-(layer,dim) importance (max over spans)."""
    feats = {}
    for task, sc in scores_by_task.items():
        S = hookers[task].num_spans
        L = hookers[task].num_layers
        v = sc.detach().cpu().view(L, S, das_dim).max(dim=1).values  # [L, das_dim]
        feats[task] = v.flatten().numpy()  # [L * das_dim]
    return feats


def overlap_matrix(feats, top_frac=0.05):
    tasks = list(feats)
    n = len(tasks)
    sel = {}
    for t in tasks:
        v = feats[t]
        k = max(1, int(top_frac * len(v)))
        sel[t] = set(np.argsort(v)[::-1][:k].tolist())
    M = np.zeros((n, n))
    for i, a in enumerate(tasks):
        for j, b in enumerate(tasks):
            inter = len(sel[a] & sel[b])
            union = len(sel[a] | sel[b])
            M[i, j] = inter / union if union else 0.0
    return tasks, M


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="EleutherAI/pythia-1b")
    p.add_argument("--tasks", default="all",
                   help="'all' (29 syntaxgym) or comma-separated task names")
    p.add_argument("--das_dim", type=int, default=64)
    p.add_argument("--steps", type=int, default=60000, help="total steps across all tasks")
    p.add_argument("--T", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--lr_rotation", type=float, default=0.001)
    p.add_argument("--n_iters", type=int, default=30)
    p.add_argument("--k_schedule", default="log", choices=["uniform", "log"])
    p.add_argument("--hard_fwd", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--sufficient", action=argparse.BooleanOptionalAction, default=True,
                   help="Sufficient/denoising (default): top-k stay clean. Matches eval_mib.py.")
    p.add_argument("--pos_strategy", default="last", choices=["first", "last", "all"])
    p.add_argument("--loss", default="ce", choices=["ce", "kl"])
    p.add_argument("--n_eval", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="results/pythia1b_multitask_das")
    args = p.parse_args()

    # unify convention: external sufficient=True (denoising) -> legacy internal flag
    legacy_sufficient = not args.sufficient
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 device_map="auto")
    model.eval()
    for pr in model.parameters():
        pr.requires_grad_(False)

    tasks = all_syntaxgym_tasks() if args.tasks == "all" else args.tasks.split(",")
    logger.info("Training over %d tasks: %s", len(tasks), tasks)

    # Shared per-layer rotation
    hidden = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    das_rotations = {li: make_rotate_layer(hidden, args.das_dim).to(device)
                     for li in range(n_layers)}

    datasets, hookers, scores_by_task = {}, {}, {}
    for t in tasks:
        ds, hooker, sc = build_task(model, t, args.das_dim, args.pos_strategy,
                                    legacy_sufficient, das_rotations, device)
        datasets[t], hookers[t], scores_by_task[t] = ds, hooker, sc
        logger.info("  %-40s spans=%d total=%d", t, ds.num_spans, hooker.total)

    rot_params = [pp for rl in das_rotations.values() for pp in rl.parameters()]
    optimizer = torch.optim.Adam([
        {"params": list(scores_by_task.values()), "lr": args.lr},
        {"params": rot_params, "lr": args.lr_rotation},
    ])

    # Mixed multi-task training: each step accumulates gradients over ALL tasks
    # (one pair each), then a SINGLE optimizer.step() -> the shared rotation gets the
    # joint multi-task gradient, while each task's scores get only their own task's grad.
    logger.info("Training %d mixed steps (all %d tasks/step = %d task-forwards), "
                "sufficient(denoising)=%s, das_dim=%d",
                args.steps, len(tasks), args.steps * len(tasks), args.sufficient, args.das_dim)
    running = {t: [] for t in tasks}
    loss_log = []          # [(step, mean_loss)]
    per_task_log = []      # [(step, {task: mean_recent_loss})]
    log_every = max(1, args.steps // 200)
    t0 = time.time()
    for step in range(args.steps):
        optimizer.zero_grad()
        for t in tasks:
            loss, hooker = train_step(model, datasets[t], hookers[t], scores_by_task[t],
                                      tokenizer, device, args, legacy_sufficient)
            loss.backward()              # accumulates into shared rotation + scores[t]
            hooker.remove_hooks()
            running[t].append(loss.item())
        optimizer.step()                 # one joint update per step
        if (step + 1) % log_every == 0:
            rate = (step + 1) * len(tasks) / (time.time() - t0)
            per_task = {t: float(np.mean(running[t][-20:])) for t in tasks if running[t]}
            avg = float(np.mean(list(per_task.values())))
            loss_log.append((step + 1, avg))
            per_task_log.append((step + 1, per_task))
            logger.info("step %5d/%d  mean_loss=%.4f  (%.1f task-fwd/s)",
                        step + 1, args.steps, avg, rate)
    train_time = time.time() - t0
    logger.info("Training done in %.0fs", train_time)

    # Eval per task
    sparsities = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
    per_task_eval = {}
    for t in tasks:
        logger.info("Eval %s ...", t)
        per_task_eval[t] = eval_task(model, datasets[t], hookers[t], scores_by_task[t],
                                     tokenizer, device, args, legacy_sufficient, sparsities)
        ev = per_task_eval[t]
        logger.info("  %s: CE_L@5%%=%.4f KL_L@5%%=%.4f",
                    t, ev["eval_learned_ce"][4], ev["eval_learned"][4])

    # Cross-task feature overlap
    feats = feature_vectors(scores_by_task, hookers, args.das_dim)
    ov_tasks, ov_M = overlap_matrix(feats, top_frac=0.05)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "tasks": tasks,
        "scores": {t: sc.detach().cpu() for t, sc in scores_by_task.items()},
        "span_names": {t: datasets[t].span_names for t in tasks},
        "num_spans": {t: datasets[t].num_spans for t in tasks},
        "das_rotations": {li: rl.weight.data.cpu() for li, rl in das_rotations.items()},
        "per_task_eval": per_task_eval,
        "sparsities": sparsities,
        "feature_vectors": feats,
        "overlap_tasks": ov_tasks,
        "overlap_matrix": ov_M,
        "das_dim": args.das_dim,
        "train_time_s": train_time,
        "loss_log": loss_log,
        "per_task_loss_log": per_task_log,
        "args": vars(args),
    }
    pkl = out.with_suffix(".pkl") if out.suffix != ".pkl" else out
    with open(pkl, "wb") as f:
        pickle.dump(result, f)
    logger.info("Saved %s", pkl)


if __name__ == "__main__":
    main()
