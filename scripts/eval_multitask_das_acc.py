"""Eval-only re-run of the multitask DAS checkpoint: sweep ABSOLUTE k (powers of two,
1,2,4,8,... up to `total`, plus the all-features point `total`) and report, per task:

  accuracy   = fraction of pairs where p(correct continuation) > p(foil)
  p_correct  = mean prob of the correct (base/clean) continuation
  p_foil     = mean prob of the counterfactual continuation
  ce         = cross-entropy of the correct token

for both the learned (top-k by score) and a random dim ordering. No retraining: loads
the saved per-layer rotations + per-task scores from the training pkl.

Run on sc via nlprun (GPU). Pythia-1b -> 1 GPU, defaults fine.
    nlprun -g 1 -q jag -n mtdas-acc "uv run python scripts/eval_multitask_das_acc.py"
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from learning_to_attribute import CausalGymDataset  # noqa: E402
from learning_to_attribute.sigmoid_das import RotateLayer  # noqa: E402
from attribute import get_hooks_classes  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("mtdas_acc")


def powers_of_two_ks(total):
    """[1, 2, 4, 8, ...] up to total, plus total itself (all features)."""
    ks, k = [], 1
    while k < total:
        ks.append(k)
        k *= 2
    ks.append(total)
    return ks


@torch.no_grad()
def eval_task_acc(model, ds, hooker, scores, tokenizer, device, n_eval):
    total = hooker.total
    ks = powers_of_two_ks(total)
    sorted_idx = scores.detach().argsort(descending=True)
    torch.manual_seed(0)
    random_idx = torch.randperm(total, device=device)

    metrics = ["acc", "pc", "pf", "ce"]
    accum = {o: {m: [[] for _ in ks] for m in metrics} for o in ("learned", "random")}

    for _ in range(n_eval):
        pair = ds.sample_pair()
        tok = ds.tokenize_pair(pair, tokenizer, device=str(device))
        hooker.set_alignment(tok.base_alignment, tok.src_alignment,
                             tok.base_input_ids.shape[1], tok.src_input_ids.shape[1])
        hooker.cache_cf_activations(tok.src_input_ids)
        hooker.register_hooks()
        correct_id, foil_id = tok.base_label_id, tok.src_label_id

        for oname, ordering in (("learned", sorted_idx), ("random", random_idx)):
            for j, k in enumerate(ks):
                mask = torch.zeros(total, device=device)
                mask[ordering[:k]] = 1.0
                hooker.mask = mask
                logits = model(tok.base_input_ids).logits[0, -1].float()
                p = F.softmax(logits, dim=-1)
                pc, pf = p[correct_id].item(), p[foil_id].item()
                ce = F.cross_entropy(logits.unsqueeze(0),
                                     torch.tensor([correct_id], device=device)).item()
                accum[oname]["acc"][j].append(1.0 if pc > pf else 0.0)
                accum[oname]["pc"][j].append(pc)
                accum[oname]["pf"][j].append(pf)
                accum[oname]["ce"][j].append(ce)
        hooker.remove_hooks()

    out = {"ks": ks, "total": total, "n_eval": n_eval}
    for oname in ("learned", "random"):
        for m in metrics:
            arr = np.array(accum[oname][m])  # [len(ks), n_eval]
            out[f"{oname}_{m}"] = arr.mean(axis=1).tolist()
            out[f"{oname}_{m}_std"] = arr.std(axis=1).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/pythia1b_multitask_das.pkl")
    ap.add_argument("--output", default="results/pythia1b_multitask_das_acc.pkl")
    ap.add_argument("--n_eval", type=int, default=100)
    ap.add_argument("--tasks", default="all", help="'all' (from ckpt) or comma list")
    args = ap.parse_args()

    d = pickle.load(open(args.ckpt, "rb"))
    cfg = d["args"]
    das_dim = cfg["das_dim"]
    legacy_sufficient = not cfg["sufficient"]  # hooker's internal flag
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.bfloat16,
                                                 device_map="auto")
    model.eval()
    for pr in model.parameters():
        pr.requires_grad_(False)

    hidden = model.config.hidden_size
    n_layers = model.config.num_hidden_layers

    # Reconstruct fixed rotations from saved (already-orthogonal) weights. Plain
    # RotateLayer (no orthogonal parametrization needed -- rotations are frozen here).
    das_rotations = {}
    for li in range(n_layers):
        rl = RotateLayer(hidden, das_dim)
        with torch.no_grad():
            rl.weight.copy_(d["das_rotations"][li].to(rl.weight.dtype))
        das_rotations[li] = rl.to(device)

    _, SpanHooksCls = get_hooks_classes(model)
    tasks = d["tasks"] if args.tasks == "all" else args.tasks.split(",")

    results = {}
    for t in tasks:
        ds = CausalGymDataset(t, seed=42)
        hooker = SpanHooksCls(model, "das", ds.num_spans,
                              pos_strategy=cfg["pos_strategy"], sufficient=legacy_sufficient)
        hooker.set_das_dim(das_dim)
        for li in range(hooker.num_layers):
            hooker.R[li] = das_rotations[li]
        scores = d["scores"][t].to(device)
        ev = eval_task_acc(model, ds, hooker, scores, tokenizer, device, args.n_eval)
        results[t] = ev
        logger.info("%-40s total=%d  acc@max=%.3f  pc@max=%.3f  (ks=%s)",
                    t, ev["total"], ev["learned_acc"][-1], ev["learned_pc"][-1],
                    ev["ks"][:1] + ["..."] + ev["ks"][-2:])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pkl = out.with_suffix(".pkl") if out.suffix != ".pkl" else out
    with open(pkl, "wb") as f:
        pickle.dump({"per_task": results, "tasks": tasks,
                     "ks_note": "absolute k: powers of 2 up to total, plus total (all features)",
                     "n_eval": args.n_eval, "ckpt": args.ckpt, "args": cfg}, f)
    logger.info("Saved %s", pkl)


if __name__ == "__main__":
    main()
