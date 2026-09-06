"""Which SAE units does an attribution method actually rank first, per task?

*** THE ERROR NODE IS A UNIT AND IT WINS. *** Each (layer, position) block of a `*_sae_span` score
vector is d_sae=32,768 dictionary latents PLUS one ERROR node at index 32,768 -- the part of the
activation the dictionary fails to reconstruct (llama.py:157, `sae_width = d_sae + 1`). It is 1 in
32,769 units, i.e. 0.003% of the substrate, and for MAttr it fills essentially the whole top of the
ranking. A "top-10 features" list that does not say so reads as ten interpretable latents when it
is mostly the residual the dictionary could not explain.

So this prints THREE things and none of them alone is the answer:
  1. the raw top-k units, error nodes included and marked ERR;
  2. the top-k with error nodes excluded -- the interpretable latents, and the list you want if
     you are about to look anything up on Neuronpedia -- together with the rank the first real
     latent holds in the FULL ordering, which is the honest measure of how buried they are;
  3. the error-node share of the top 10 / 100 / 1k / 10k, for every method, since the share is
     what makes (1) and (2) different documents.

LATENT INDICES ARE PER-LAYER. Llama-Scope trains a separate SAE per layer, so latent 19823 at L31
and latent 19823 at L18 are unrelated features. Never aggregate a latent index across layers; the
cross-task table below keys on (layer, latent) for that reason.

POSITIONS ARE TOKEN POSITIONS, not content spans. On sva/arith there is no content-span schema, so
eval_sva.py falls back to one span per token position and every prompt in a task maps to the same
0..seq_len-1 index. Position p is therefore the p-th token of the prompt, and is comparable across
tasks only in so far as their prompts line up -- which they do not.

Run:  uv run python scripts/sva/top_sae_features.py [--site mlp|resid] [--method ...] [--topk 10]
"""
import argparse
import os
import sys

import torch

RES = "results/sva_sweep"
TASKS = ["nounpp", "rc", "simple", "within_rc", "addition", "months", "weekdays", "hours"]
METHODS = {"mattr-adam": ("sufficient_topk_adam_eps1e-2_bs1", "MAttr(Adam)"),
           "mattr-sgd": ("sufficient_topk_sgd_bs1", "MAttr(SGD)"),
           "ig": ("ig", "IG"), "ixg": ("ixg", "I×G")}
N_LAYERS = 32
D_SAE = 32768
W = D_SAE + 1            # +1 error node per (layer, position) block


def load(task, sub, tag):
    p = f"{RES}/{task}_llama3_{sub}_{tag}.scores.pt"
    if not os.path.exists(p):
        return None, None
    s = torch.load(p, map_location="cpu").float()
    n = s.numel()
    assert n % (N_LAYERS * W) == 0, f"{task}/{sub}: {n} not 32*{W}*seq_len"
    return s, n // (N_LAYERS * W)


def decode(i, seq):
    """flat index -> (layer, position, latent), latent == D_SAE meaning the error node."""
    layer, rem = divmod(int(i), seq * W)
    return (layer, *divmod(rem, W))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="mlp", choices=["mlp", "resid"])
    ap.add_argument("--method", default="mattr-adam", choices=list(METHODS))
    ap.add_argument("--topk", type=int, default=10)
    a = ap.parse_args()
    sub = f"{a.site}_sae_span" if a.site == "mlp" else "resid_sae_span"
    tag, mname = METHODS[a.method]

    print(f"=== TOP-{a.topk} UNITS, {mname}, {sub} (ERR = SAE error node) ===")
    seen, missing = {}, []
    for t in TASKS:
        s, seq = load(t, sub, tag)
        if s is None:
            missing.append(t)
            continue
        is_err = (torch.arange(s.numel()) % W) == D_SAE
        first_real = int((~is_err[torch.argsort(s, descending=True)]).nonzero()[0]) + 1
        raw = torch.topk(s, a.topk)
        s2 = s.clone()
        s2[is_err] = -float("inf")
        real = torch.topk(s2, a.topk)
        print(f"\n{t}  (seq_len={seq}, {s.numel():,} units)   "
              f"best real latent is rank {first_real} overall")
        print(f"  {'#':<4}{'layer':>6}{'pos':>5}{'unit':>9}{'score':>10}     "
              f"{'layer':>6}{'pos':>5}{'latent':>8}{'score':>10}")
        for r in range(a.topk):
            l0, p0, u0 = decode(raw.indices[r], seq)
            l1, p1, u1 = decode(real.indices[r], seq)
            print(f"  {r + 1:<4}{l0:>6}{p0:>5}{'ERR' if u0 == D_SAE else u0:>9}"
                  f"{raw.values[r]:>10.4g}     {l1:>6}{p1:>5}{u1:>8}{real.values[r]:>10.4g}")
            seen.setdefault((l1, u1), []).append(t)
    if missing:
        print(f"\nmissing: {', '.join(missing)}")

    print(f"\n\n=== error-node share of the top-k (error nodes are 1 in {W:,} units) ===")
    print(f"{'task':<11}{'method':<13}{'top10':>7}{'top100':>8}{'top1k':>7}{'top10k':>8}")
    for t in TASKS:
        for key, (tg, nm) in METHODS.items():
            s, seq = load(t, sub, tg)
            if s is None:
                continue
            e = ((torch.arange(s.numel()) % W) == D_SAE)[torch.argsort(s, descending=True)]
            print(f"{t:<11}{nm:<13}" + "".join(
                f"{e[:k].float().mean().item():>{w}.0%}"
                for k, w in ((10, 7), (100, 8), (1000, 7), (10000, 8))))

    print(f"\n\n=== (layer, latent) pairs in >1 task's top-{a.topk} real latents ===")
    for (l, lat), ts in sorted(seen.items(), key=lambda kv: -len(kv[1])):
        if len(ts) > 1:
            print(f"  L{l:<3} latent {lat:<7} {len(ts)} tasks: {', '.join(ts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
