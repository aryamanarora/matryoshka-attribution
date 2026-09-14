"""The note's two worked examples, recomputed on our model.

    uv run python scripts/vw/vw_examples.py --run results/vw/base --word ACETYLCHOLINE
    uv run python scripts/vw/vw_examples.py --run results/vw/base --source IN --top 15

1. THE MOTIVATING EXAMPLE. Turner, Wu & Batson (2026) complete ACETYLCHOLINE and split the
   logits into the direct, attention and MLP paths (exact, because the model has no
   normalization):

     "No path individually ranks ' E ' as the top output. Each places its largest
      contribution on a different, incorrect token: the direct path favors ' _Mrs ', the
      attention path favors ' ely ', and the MLP path favors ' ATION '. Instead, ' E ' is
      the one continuation every path scores somewhat highly."

   We print each path's top targets and where the true continuation sits in each, so the
   qualitative shape (no path is individually right; the answer is the consensus) can be
   checked rather than assumed. The specific competitor tokens are properties of THEIR
   trained model, not of the setup, so they are not expected to match token-for-token.

2. THE INTERFERENCE WEIGHT. From the same example:

     "the largest [Tokens->Logits] virtual weight [from ' IN '] votes to complete the word
      with ' utions '. This token never follows ' IN ' in the training set, so every time
      this virtual weight affects the output, it only makes the model's loss worse."
     "Re-sorting ' IN ''s weights by Fisher effectiveness not only drops ' utions ', it
      also lifts the ' E ' from ACETYLCHOLINE to second place."

   `--source IN` prints that source's top targets under every ranking side by side, with
   the corpus co-occurrence count for each (source, target) pair -- so "never follows in
   the training set" is a measured 0, not an assertion.
"""

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vw_model import VOCAB  # noqa: E402
from vw_prune import rankings  # noqa: E402
from vw_scores import D_VP, load_run  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SHOW = ["weight", "weight_abs", "fisher", "helpfulness", "era"]


def tokname(tk, i):
    return repr(tk.decode([int(i)])) if i < VOCAB else f"<pos {int(i)-VOCAB}>"


def show_paths(model, tk, word, device):
    ids = tk.encode(word).ids
    print(f"\n=== path decomposition: {word!r} -> {[tk.decode([i]) for i in ids]} ===")
    b = torch.tensor([ids], device=device)
    direct, attn, mlp = model.paths(b)
    q = len(ids) - 2                       # the position that predicts the final token
    tgt = ids[-1]
    tot = (direct + attn + mlp)[0, q]
    for name, v in (("direct", direct[0, q]), ("attention", attn[0, q]),
                    ("MLP", mlp[0, q]), ("TOTAL", tot)):
        top = v.topk(5)
        rank = int((v > v[tgt]).sum())
        print(f"  {name:<10} top: " + ", ".join(
            f"{tokname(tk,i)}({x:+.2f})" for x, i in zip(top.values.tolist(), top.indices.tolist()))
            + f"   | {tokname(tk,tgt)} rank {rank} ({v[tgt]:+.2f})")


def show_source(run, model, tk, sc, ranks, source, top, device):
    """One source row of the Tokens->Logits family, ranked every way we have."""
    ids = tk.encode(source, add_special_tokens=False).ids
    assert len(ids) == 1, f"{source!r} is {len(ids)} tokens: {[tk.decode([i]) for i in ids]}"
    i = ids[0]
    W, F, H = (sc[k][i].float() for k in ("W", "fisher", "helpfulness"))
    pair, src_count = sc["pair"][i].float(), float(sc["src_count"][i])
    print(f"\n=== source {tokname(tk,i)} (row {i}); active at {src_count:,.0f} positions ===")
    names = [n for n in SHOW if n in ranks] + [n for n in ranks if n not in SHOW
                                               and not n.startswith("random")]
    for name in names:
        s = ranks[name].view(D_VP, VOCAB)[i]
        order = s.argsort(descending=True)[:top]
        print(f"  by {name}:")
        for j in order.tolist():
            print(f"     {tokname(tk,j):<14} W {W[j]:+7.3f}  fisher {F[j]:.3e}  "
                  f"help {H[j]:+.3e}  followed {int(pair[j]):>7,}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--word", default="ACETYLCHOLINE")
    ap.add_argument("--source", default="IN")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tk = Tokenizer.from_file(str(ROOT / "data" / "vw" / "tok4096.json"))
    model = load_run(args.run, device)
    show_paths(model, tk, args.word, device)
    sc = torch.load(args.run / "scores" / "tl.pt", map_location=device, weights_only=False)
    show_source(args.run, model, tk, sc, rankings(args.run, device),
                args.source, args.top, device)


if __name__ == "__main__":
    main()
