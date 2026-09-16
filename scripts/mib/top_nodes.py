"""Name the top-k nodes of a MIB node-level run from its scores .pt (no model load).

    uv run python scripts/mib/top_nodes.py --task mcqa --model qwen2.5 --k 5 \
        --dirs mib_node_topk_uniform_lr05 mib_node_cause_topk_uniform_lr05

The score vector is laid out as eval_mib.py builds it: [input] (only if --include-input was
set, read from the saved args) + n_layers * n_heads attention heads (a{L}.h{H}, layer-major)
+ n_layers MLPs (m{L}). n_layers / n_heads come from the vector length and the model name.
"""
import argparse
from pathlib import Path

import torch

HEADS = {"gpt2": 12, "qwen2.5": 14, "gemma2": 8, "llama3": 32}


def names(n, include_input, model):
    h = HEADS[model]
    off = 1 if include_input else 0
    n_layers = (n - off) // (h + 1)
    assert off + n_layers * (h + 1) == n, f"{n} scores do not factor as {off}+L*({h}+1)"
    out = ["input"] if include_input else []
    out += [f"a{L}.h{H}" for L in range(n_layers) for H in range(h)]
    out += [f"m{L}" for L in range(n_layers)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--split", default="validation")
    a = ap.parse_args()
    for d in a.dirs:
        sd = torch.load(Path("results") / d / f"{a.task}_{a.model}_scores.pt", map_location="cpu")
        s = sd["scores"].float().view(-1)
        nm = names(s.numel(), bool(sd.get("args", {}).get("include_input", False)), a.model)
        order = torch.argsort(s, descending=True)
        top = [(nm[i], round(s[i].item(), 3)) for i in order[:a.k].tolist()]
        print(f"{d}: " + ", ".join(f"{n} ({v})" for n, v in top))


if __name__ == "__main__":
    main()
