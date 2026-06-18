"""Full analysis for the arithmetic-wild multitask MAttr experiments.

Produces, for each available method (mlp / das / sae):
  (1) per-layer importance profile at the last_token span (does MAttr localise to L18?)
  (2) for MLP: overlap of top-N layer-18 neurons with the paper's claimed neurons
      (arXiv 2605.01148 / neurons_per_task.json), with chance + enrichment.
A combined layer-profile figure is written to results/arith_layer_profiles.png.
"""
import json, pickle
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path("/home/guests/aryaman/learning-to-attribute/results")
PAPER = json.load(open("/home/guests/aryaman/arithmetic-wild/src/neurons_per_task.json"))
PAPER_LAYER, SPAN_LAST = 18, 2
METHODS = {"mlp": RES / "arith_mlp.pkl", "das": RES / "arith_das.pkl", "sae": RES / "arith_sae.pkl"}


def layer_profile(sc, L, S, comp, span=SPAN_LAST):
    """max|score| per layer at a span -> [L]."""
    return sc.view(L, S, comp)[:, span, :].abs().max(dim=1).values.numpy()


def load(path):
    if not path.exists():
        return None
    return pickle.load(open(path, "rb"))


def comp_size(r):
    if r["mask"] == "mlp":
        return r["intermediate_size"]
    if r["mask"] == "das":
        return r["das_dim"]
    return r["scores"][r["tasks"][0]].numel() // (r["n_layers"] * max(r["num_spans"].values()))


results = {m: load(p) for m, p in METHODS.items()}
avail = [m for m in results if results[m] is not None]
print("available methods:", avail)

# layer-profile figure
fig, axes = plt.subplots(1, len(avail), figsize=(6 * len(avail), 4), squeeze=False)
for ax, m in zip(axes[0], avail):
    r = results[m]; L = r["n_layers"]; comp = comp_size(r)
    for t in r["tasks"]:
        S = r["num_spans"][t]
        prof = layer_profile(r["scores"][t], L, S, comp)
        ax.plot(range(L), prof, marker=".", label=t)
    ax.axvline(PAPER_LAYER, color="k", ls="--", lw=1, label="paper L18")
    ax.set_title(f"{m.upper()} last-token layer importance")
    ax.set_xlabel("layer"); ax.set_ylabel("max|score|"); ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(RES / "arith_layer_profiles.png", dpi=110)
print("wrote", RES / "arith_layer_profiles.png")

# numeric summary
for m in avail:
    r = results[m]; L = r["n_layers"]; comp = comp_size(r)
    print(f"\n########## {m.upper()}  (steps={r['args']['steps']}) ##########")
    for t in r["tasks"]:
        S = r["num_spans"][t]
        prof = layer_profile(r["scores"][t], L, S, comp)
        order = np.argsort(-prof)
        r18 = int(np.where(order == PAPER_LAYER)[0][0]) + 1
        # final sufficiency CE at a few sparsities
        ev = r["per_task_eval"][t]["eval_learned_ce"]
        sp = r["sparsities"]
        ce = {f"{int(sp[i]*100) if sp[i]>=0.01 else sp[i]}%": round(ev[i], 2)
              for i in [3, 6, 8] if i < len(ev)}
        print(f"  {t:9s}: L18 importance rank {r18:2d}/{L} (top layers "
              f"{[int(x) for x in order[:4]]}); CE@sparsity {ce}")
        if m == "mlp":
            v18 = r["scores"][t].view(L, S, comp)[PAPER_LAYER, SPAN_LAST, :].numpy()
            pn = PAPER.get(t, []); N = len(pn)
            top = set(np.argsort(-v18)[:N].tolist())
            hit = len(top & set(pn))
            v_pn, v_all = float(v18[pn].mean()), float(v18.mean())
            ranks = [int(np.where(np.argsort(-v18) == n)[0][0]) for n in pn]
            print(f"             paper-neuron overlap top-{N}: {hit}/{N} "
                  f"(chance {N*N/comp:.2f}); paper-neuron score {v_pn:.3f} vs all {v_all:.3f}; "
                  f"median paper rank {int(np.median(ranks))}/{comp}")
