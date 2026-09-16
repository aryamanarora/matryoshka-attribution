"""Are a sparsity ladder's masks nested? Containment of each sparser rung's mask in each denser one.

    uv run python scripts/mib/ladder_nestedness.py [--ladders dbm np] [--split test]

Per (ladder, cell): every rung's EMITTED mask is its top-L0 nodes by graph score, with L0 the
same own-L0 rule eval_dbm_multisparsity.py uses (DBM: k_log[-1]; Node Pruning: the deterministic
hard-concrete gate count) plus the always-on `input`. For every ordered pair of rungs with
|S| < |D| the containment is |S & D| / |S| -- 1.0 when the sparser mask sits inside the denser
one. Reported per cell as the mean over ALL such pairs and over CONSECUTIVE rungs (adjacent in
mask size), plus the ladder-wide means. A perfectly nested ladder is 1.0 everywhere; an L1 or
target-s knob that re-solves the mask from scratch at each rung need not be.

Rungs whose mask is empty or full are skipped (they are the degenerate runs the ladder eval
also skips). No GPU; graph JSONs and scores .pt only.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_mib_table as M                       # COLUMNS  # noqa: E402
from eval_dbm_multisparsity import LADDERS, own_l0   # noqa: E402


def mask_of(ladder, knob, task, model):
    d = Path(LADDERS[ladder]["dir"].format(k=knob))
    g, s = d / f"graph_{task}_{model}.json", d / f"{task}_{model}_scores.pt"
    if not (g.exists() and s.exists()):
        return None
    sd = torch.load(s, map_location="cpu")
    nodes = json.load(open(g))["nodes"]
    scores = {n: v["score"] for n, v in nodes.items() if isinstance(v, dict) and v.get("score") is not None
              and n != "logits"}
    n_nodes = len(scores)
    unmasked = n_nodes - len(sd["scores"])
    k = int(round(own_l0(ladder, sd))) + unmasked
    if k < 1 or k >= n_nodes:
        return None
    top = sorted(scores, key=lambda n: -scores[n])[:k]
    return frozenset(top)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladders", nargs="+", default=["dbm", "np"])
    a = ap.parse_args()
    cols = [(t, m) for t, m, _ in M.COLUMNS]
    for ladder in a.ladders:
        print(f"\n=== {ladder} ladder ({', '.join(LADDERS[ladder]['knobs'])}) ===")
        print(f"{'cell':<32}{'rungs':>6}{'all pairs':>11}{'consecutive':>13}{'sizes (k)':>12}")
        all_pairs, all_cons = [], []
        for task, model in cols:
            masks = []
            for knob in LADDERS[ladder]["knobs"]:
                mk = mask_of(ladder, knob, task, model)
                if mk is not None:
                    masks.append((knob, mk))
            # order by size, sparsest first; drop exact-size duplicates (collided rungs)
            masks.sort(key=lambda x: len(x[1]))
            uniq = []
            for knob, mk in masks:
                if not uniq or len(mk) != len(uniq[-1][1]):
                    uniq.append((knob, mk))
            pairs = [len(S & D) / len(S) for i, (_, S) in enumerate(uniq) for _, D in uniq[i + 1:]]
            cons = [len(uniq[i][1] & uniq[i + 1][1]) / len(uniq[i][1]) for i in range(len(uniq) - 1)]
            if not pairs:
                print(f"{task + '/' + model:<32}{len(uniq):>6}{'---':>11}{'---':>13}")
                continue
            all_pairs.append(sum(pairs) / len(pairs)); all_cons.append(sum(cons) / len(cons))
            sizes = " ".join(str(len(mk)) for _, mk in uniq)
            print(f"{task + '/' + model:<32}{len(uniq):>6}{all_pairs[-1]:>11.3f}{all_cons[-1]:>13.3f}  {sizes}")
        if all_pairs:
            print(f"{'mean over cells':<32}{'':>6}{sum(all_pairs) / len(all_pairs):>11.3f}"
                  f"{sum(all_cons) / len(all_cons):>13.3f}")


if __name__ == "__main__":
    main()
