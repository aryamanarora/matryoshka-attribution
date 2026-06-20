"""Generic multi-concept interchange MAttr.

For each causal concept exposed by the dataset (pair.targets: {concept: token_id}) learn a
SEPARATE per-(layer, span, unit) score set. Interchange (noising) masking: the top-k scored
units are patched to the counterfactual, the rest stay base; CE is to that concept's target
token. arithmetic-wild -> 3 concepts {input, offset, output}; CausalGym -> 1 concept {src}.

Same machinery (per-span SpanHooks) for any dataset/mask; nothing task-specific here.
"""
import argparse, json, pickle, time, logging, sys, os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from learning_to_attribute.data.arithmetic_wild import ArithmeticWildDataset
from learning_to_attribute.data.causalgym import CausalGymDataset
from learning_to_attribute.sigmoid_topk import sigmoid_topk_hard
from attribute_multitask import sample_k, get_hooks_classes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("concepts")
ARITH_DIR = "/home/guests/aryaman/arithmetic-wild/datasets/Llama-3.1-8B"


def make_dataset(kind, task, seed, mode="necessary"):
    if kind == "arith":
        return ArithmeticWildDataset(task, ARITH_DIR, seed=seed, target_mode=mode)
    return CausalGymDataset(task, seed=seed)


def forward_ce(model, hooker, scores, k, T, n_iters, base_ids, target_id, device):
    hooker.mask = sigmoid_topk_hard(scores, k=k, T=T, n_iters=n_iters)
    logits = model(base_ids).logits[0, -1].float()
    return F.cross_entropy(logits.unsqueeze(0), torch.tensor([target_id], device=device))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--dataset", default="arith", choices=["arith", "causalgym"])
    p.add_argument("--mask", default="sae", choices=["das", "mlp", "sae", "resid", "resid_dim"])
    p.add_argument("--per_token", action="store_true",
                   help="arith: each token gets its own span (operands grouped); covers all "
                        "positions. Required for a layer x position scalar (resid) map.")
    p.add_argument("--mode", default="necessary", choices=["necessary", "sufficient"],
                   help="necessary: top-k->cf, target=V-swapped; sufficient: top-k base, "
                        "complement->cf, target=everything-except-V swapped")
    p.add_argument("--tasks", default="months,weekdays,hours,addition")
    p.add_argument("--sae_repo", default="fnlp/Llama3_1-8B-Base-LXR-8x")
    p.add_argument("--das_dim", type=int, default=32)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--T", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--n_iters", type=int, default=30)
    p.add_argument("--k_schedule", default="uniform", choices=["uniform", "log"])
    p.add_argument("--pos_strategy", default="last", choices=["first", "last", "all"])
    p.add_argument("--n_eval", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="results/arith_sae_concepts")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto").eval()
    for pr in model.parameters():
        pr.requires_grad_(False)
    _, SpanHooks = get_hooks_classes(model)
    n_layers, hidden = model.config.num_hidden_layers, model.config.hidden_size

    shared = {}
    if args.mask == "sae":
        from learning_to_attribute.sae_loader import load_llama_scope_saes
        shared["sae"] = load_llama_scope_saes(args.sae_repo, n_layers, device)
    elif args.mask == "das":
        from learning_to_attribute.sigmoid_das import make_rotate_layer
        shared["rot"] = {li: make_rotate_layer(hidden, args.das_dim).to(device) for li in range(n_layers)}

    tasks = args.tasks.split(",")
    datasets, hookers, scores = {}, {}, {}     # scores[task][concept] = Parameter
    # necessary -> noising (top-k -> cf, hooker.sufficient=True); sufficient -> denoising
    hooker_suf = (args.mode == "necessary")
    pos_strategy = "all" if args.per_token else args.pos_strategy   # cover all operand tokens
    for t in tasks:
        ds = make_dataset(args.dataset, t, args.seed, mode=args.mode)
        if args.per_token and args.dataset == "arith":
            ds.build_schema(tokenizer)
            logger.info("  %-10s per-token: %d spans, %d/%d valid examples",
                        t, ds.num_spans, len(ds._valid), len(ds.bases))
        hooker = SpanHooks(model, args.mask, ds.num_spans, pos_strategy=pos_strategy, sufficient=hooker_suf)
        if args.mask == "sae":
            hooker.set_saes(shared["sae"])
        elif args.mask == "das":
            hooker.set_das_dim(args.das_dim)
            for li in range(hooker.num_layers):
                hooker.R[li] = shared["rot"][li]
        # discover concepts from a sample pair's targets
        tok0 = ds.tokenize_pair(ds.sample_pair(), tokenizer, device=str(device))
        concepts = list(tok0.targets.keys())
        datasets[t], hookers[t] = ds, hooker
        scores[t] = {c: nn.Parameter(torch.zeros(hooker.total, device=device)) for c in concepts}
        logger.info("  %-10s spans=%d total=%d concepts=%s", t, ds.num_spans, hooker.total, concepts)

    params = [pp for sc in scores.values() for pp in sc.values()]
    opt = torch.optim.Adam(params, lr=args.lr)
    logger.info("dataset=%s mask=%s tasks=%s steps=%d", args.dataset, args.mask, tasks, args.steps)

    running = {t: {c: [] for c in scores[t]} for t in tasks}
    t0 = time.time()
    for step in range(args.steps):
        opt.zero_grad()
        for t in tasks:
            ds, hooker = datasets[t], hookers[t]
            tok = ds.tokenize_pair(ds.sample_pair(), tokenizer, device=str(device))
            hooker.set_alignment(tok.base_alignment, tok.src_alignment,
                                 tok.base_input_ids.shape[1], tok.src_input_ids.shape[1])
            hooker.cache_cf_activations(tok.src_input_ids)
            hooker.register_hooks()
            for c, sc in scores[t].items():
                k = sample_k(hooker.total, args.k_schedule)
                loss = forward_ce(model, hooker, sc, k, args.T, args.n_iters,
                                  tok.base_input_ids, tok.targets[c], device)
                loss.backward()
                running[t][c].append(loss.item())
            hooker.remove_hooks()
        opt.step()
        if (step + 1) % max(1, args.steps // 100) == 0:
            msg = "  ".join(f"{t}:{'/'.join(f'{c}={np.mean(running[t][c][-20:]):.2f}' for c in scores[t])}"
                            for t in tasks)
            logger.info("step %5d/%d  %.1f it/s | %s", step + 1, args.steps,
                        (step + 1) / (time.time() - t0), msg)

    # eval: per (task, concept) sufficiency curve (CE + accuracy to that concept's target)
    sparsities = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    per_eval = {}
    for t in tasks:
        ds, hooker = datasets[t], hookers[t]
        per_eval[t] = {}
        for c, sc in scores[t].items():
            order = torch.argsort(sc.detach(), descending=True)
            tot = sc.numel()
            ce = np.zeros(len(sparsities)); acc = np.zeros(len(sparsities)); n = 0
            for _ in range(args.n_eval):
                tok = ds.tokenize_pair(ds.sample_pair(), tokenizer, device=str(device))
                hooker.set_alignment(tok.base_alignment, tok.src_alignment,
                                     tok.base_input_ids.shape[1], tok.src_input_ids.shape[1])
                hooker.cache_cf_activations(tok.src_input_ids)
                hooker.register_hooks()
                tid = tok.targets[c]
                with torch.no_grad():
                    for j, s in enumerate(sparsities):
                        k = max(1, int(round(s * tot)))
                        m = torch.zeros(tot, device=device); m[order[:k]] = 1.0
                        hooker.mask = m
                        lg = model(tok.base_input_ids).logits[0, -1].float()
                        ce[j] += F.cross_entropy(lg.unsqueeze(0), torch.tensor([tid], device=device)).item()
                        acc[j] += int(lg.argmax() == tid)
                hooker.remove_hooks()
                n += 1
            per_eval[t][c] = {"sparsities": sparsities, "ce": (ce / n).tolist(),
                              "acc": (acc / n).tolist()}
            logger.info("  %s/%s CE@5%%=%.3f acc@5%%=%.2f", t, c,
                        per_eval[t][c]["ce"][7], per_eval[t][c]["acc"][7])

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    result = {"dataset": args.dataset, "mask": args.mask, "tasks": tasks,
              "n_layers": n_layers, "intermediate_size": model.config.intermediate_size,
              "das_dim": args.das_dim if args.mask == "das" else None,
              "num_spans": {t: datasets[t].num_spans for t in tasks},
              "scores": {t: {c: sc.detach().cpu() for c, sc in scores[t].items()} for t in tasks},
              "per_eval": per_eval, "args": vars(args)}
    with open(out.with_suffix(".pkl"), "wb") as f:
        pickle.dump(result, f)
    logger.info("saved %s.pkl", out)


if __name__ == "__main__":
    main()
