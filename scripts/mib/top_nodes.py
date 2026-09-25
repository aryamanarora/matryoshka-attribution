"""Name the top-k nodes of a MIB node-level run from its scores .pt (no model load).

    uv run python scripts/mib/top_nodes.py --task mcqa --model qwen2.5 --k 5 \
        --dirs mib_node_topk_uniform_lr05 mib_node_cause_topk_uniform_lr05
    uv run python scripts/mib/top_nodes.py --all-cells --rank-of m0 --dirs <dirs...>
        # one table: the 1-based rank of that node (of N scored) per cell, one column per dir

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
    ap.add_argument("--task")
    ap.add_argument("--model")
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--rank-of", default=None, help="print this node's rank instead of the top-k")
    ap.add_argument("--all-cells", action="store_true", help="every cell of make_mib_table.COLUMNS")
    a = ap.parse_args()
    if a.all_cells:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import make_mib_table as M
        cells = [(t, m) for t, m, _ in M.COLUMNS]
    else:
        cells = [(a.task, a.model)]

    def load(d, task, model):
        p = Path("results") / d / f"{task}_{model}_scores.pt"
        if not p.exists():
            return None, None
        sd = torch.load(p, map_location="cpu")
        s = sd["scores"].float().view(-1)
        return s, names(s.numel(), bool(sd.get("args", {}).get("include_input", False)), model)

    if a.rank_of:
        print(f"{'cell':<32}" + "".join(f"{d[:28]:>30}" for d in a.dirs))
        for task, model in cells:
            row = []
            for d in a.dirs:
                s, nm = load(d, task, model)
                if s is None or a.rank_of not in nm:
                    row.append("---")
                    continue
                order = torch.argsort(s, descending=True).tolist()
                rank = order.index(nm.index(a.rank_of)) + 1
                row.append(f"{rank}/{s.numel()}")
            print(f"{task + '/' + model:<32}" + "".join(f"{r:>30}" for r in row))
        return
    for task, model in cells:
        for d in a.dirs:
            s, nm = load(d, task, model)
            if s is None:
                print(f"{task}/{model} {d}: no scores"); continue
            order = torch.argsort(s, descending=True)
            top = [(nm[i], round(s[i].item(), 3)) for i in order[:a.k].tolist()]
            print(f"{task}/{model} {d}: " + ", ".join(f"{n} ({v})" for n, v in top))


if __name__ == "__main__":
    main()
