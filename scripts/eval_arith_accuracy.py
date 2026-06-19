"""Measure base-answer ACCURACY vs #neurons kept/corrupted for the MLP suf/nec runs.

For each saved score set (sufficient: arith_mlp.pkl, necessary: arith_mlp_nec.pkl),
sweep k = #top-scored (layer,span,neuron) units selected and apply the learned mask:
  - sufficient (denoising): selected kept at BASE, rest -> CF  (base-acc rises with k)
  - necessary  (noising):   selected corrupted to CF, rest BASE (base-acc falls with k)
Accuracy = fraction of eval examples where argmax(logits[-1]) == base answer token.
Writes a tidy CSV: mode,task,frac,n_neurons,accuracy.
"""
import argparse, pickle, csv, os, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from learning_to_attribute.data.arithmetic_wild import ArithmeticWildDataset
from attribute_multitask import get_hooks_classes

DATA = "/home/guests/aryaman/arithmetic-wild/datasets/Llama-3.1-8B"
FRACS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1.0]
RUNS = {"sufficient": "results/arith_mlp.pkl", "necessary": "results/arith_mlp_nec.pkl"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    p.add_argument("--n_eval", type=int, default=100)
    p.add_argument("--out", default="results/arith_accuracy.csv")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto").eval()
    for pr in model.parameters():
        pr.requires_grad_(False)
    dev = torch.device("cuda")
    _, SpanHooks = get_hooks_classes(model)

    rows = []
    for mode, path in RUNS.items():
        r = pickle.load(open(path, "rb"))
        total = None
        # legacy_sufficient flips the hooker direction: sufficient run -> False (denoising)
        legacy_suf = not r["args"]["sufficient"]
        for task in r["tasks"]:
            ds = ArithmeticWildDataset(task, DATA, seed=123)  # held-out seed
            hooker = SpanHooks(model, "mlp", ds.num_spans, pos_strategy="last", sufficient=legacy_suf)
            scores = r["scores"][task].to(dev)
            total = scores.numel()
            order = torch.argsort(scores, descending=True)
            ks = [max(1, int(round(f * total))) for f in FRACS]
            correct = {k: 0 for k in ks}
            n = 0
            for _ in range(args.n_eval):
                base, cf = ds.sample_pair()
                tp = ds.tokenize_pair((base, cf), tok, device=str(dev))
                hooker.set_alignment(tp.base_alignment, tp.src_alignment,
                                     tp.base_input_ids.shape[1], tp.src_input_ids.shape[1])
                hooker.cache_cf_activations(tp.src_input_ids)
                hooker.register_hooks()
                with torch.no_grad():
                    for k in ks:
                        m = torch.zeros(total, device=dev)
                        m[order[:k]] = 1.0
                        hooker.mask = m
                        logit = model(tp.base_input_ids).logits[0, -1]
                        if int(logit.argmax()) == tp.base_label_id:
                            correct[k] += 1
                hooker.remove_hooks()
                n += 1
            for f, k in zip(FRACS, ks):
                rows.append({"mode": mode, "task": task, "frac": f,
                             "n_neurons": k, "accuracy": correct[k] / n})
            print(f"{mode:10s} {task:9s} done (n={n}); "
                  + " ".join(f"{f*100:g}%={correct[k]/n:.2f}" for f, k in zip(FRACS, ks)))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["mode", "task", "frac", "n_neurons", "accuracy"])
        w.writeheader(); w.writerows(rows)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
